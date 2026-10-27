import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import anyio
from fastapi import BackgroundTasks, Depends, FastAPI, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.cache import ResponseCache, is_cacheable
from app.config import Settings, get_settings
from app.keys import ApiKey, lookup_key
from app.pricing import estimate_cost
from app.providers.base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
)
from app.providers.fake import FakeProvider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.ratelimit import RateLimiter
from app.router import Router, StreamResult
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, Usage
from app.usage import PostgresUsageStore, UsageRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("gateway")


def build_providers(settings: Settings) -> dict[str, Provider]:
    if settings.provider_mode == "fake":
        return {"groq": FakeProvider("groq", settings.fake_latency_ms), "gemini": FakeProvider("gemini", settings.fake_latency_ms)}
    # only providers that have a key: an unconfigured one would just fail every call
    providers: dict[str, Provider] = {}
    if settings.groq_api_key:
        providers["groq"] = GroqProvider()
    if settings.gemini_api_key:
        providers["gemini"] = GeminiProvider()
    if not providers:
        raise RuntimeError("no provider configured: set GROQ_API_KEY and/or GEMINI_API_KEY, or PROVIDER_MODE=fake")
    return providers


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # one of each for the whole app: connection pools are reused across requests
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    usage = await PostgresUsageStore.connect(settings.database_url)
    providers = build_providers(settings)
    log.info("providers: %s (mode=%s)", list(providers), settings.provider_mode)

    app.state.redis = redis
    app.state.usage = usage
    app.state.limiter = RateLimiter(redis, settings.rate_limit_capacity, settings.rate_limit_refill_per_s)
    app.state.cache = ResponseCache(redis, settings.cache_ttl_s)
    app.state.router = Router(
        providers,
        fallback_models={"groq": settings.groq_fallback_model, "gemini": settings.gemini_fallback_model},
        max_attempts=settings.max_attempts_per_provider,
        backoff_base_s=settings.backoff_base_s,
        backoff_max_s=settings.backoff_max_s,
    )
    yield
    # runs on shutdown: close pooled connections cleanly
    for p in providers.values():
        await p.aclose()
    await usage.aclose()
    await redis.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)

class GatewayError(Exception):
    """Errors the gateway itself raises (bad api key, rate limited), as opposed to provider errors."""

    def __init__(self, status: int, message: str, err_type: str, headers: dict[str, str] | None = None):
        self.status, self.message, self.err_type, self.headers = status, message, err_type, headers or {}


def _openai_error(status: int, message: str, err_type: str, headers: dict | None = None, **extra) -> JSONResponse:
    # openai's error shape, so openai sdks raise the matching exception (AuthenticationError, RateLimitError, ...)
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": err_type, **extra}}, headers=headers)


def provider_error_status(exc: ProviderError) -> tuple[int, str]:
    """Status code + error type the client sees for a provider error."""
    if isinstance(exc, ProviderRateLimitError):
        return 429, "rate_limit_error"
    if isinstance(exc, ProviderRequestError):
        # the client's request was bad (e.g. unknown model): pass the provider's 4xx through
        return exc.status_code or 400, "invalid_request_error"
    if isinstance(exc, ProviderAuthError):
        # our key for the provider is wrong: that's the gateway's fault, not the client's -> 502
        return 502, "upstream_auth_error"
    # provider down, 5xx or timeout
    return 502, "upstream_error"


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    status, err_type = provider_error_status(exc)
    headers = {}
    if isinstance(exc, ProviderRateLimitError) and exc.retry_after is not None:
        headers["Retry-After"] = str(int(exc.retry_after))
    return _openai_error(status, str(exc), err_type, headers, provider=exc.provider, code=exc.status_code)


@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    return _openai_error(exc.status, exc.message, exc.err_type, exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # fastapi's default is a 422 in its own format; openai clients expect a 400 invalid_request_error
    return _openai_error(400, str(exc.errors()), "invalid_request_error")


# dependencies 

async def require_api_key(request: Request) -> ApiKey:
    header = request.headers.get("authorization", "")
    # openai sdks send "Authorization: Bearer <key>"
    raw = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    if not raw:
        raise GatewayError(401, "missing api key: send 'Authorization: Bearer <key>'", "invalid_request_error")
    try:
        key = await lookup_key(request.app.state.redis, raw)
    except RedisError:
        # fail closed: unlike the rate limiter, letting unknown callers in when redis is down is not ok
        log.exception("auth lookup failed")
        raise GatewayError(503, "auth backend unavailable", "service_unavailable")
    if key is None:
        raise GatewayError(401, "invalid api key", "invalid_request_error")
    return key


async def rate_limit(request: Request, response: Response, key: ApiKey = Depends(require_api_key)) -> ApiKey:
    decision = await request.app.state.limiter.acquire(key.key_id)
    if not decision.allowed:
        # round up: telling a client "retry after 0" would make it retry instantly and fail again
        retry_after = max(1, int(decision.retry_after + 0.999))
        raise GatewayError(429, "rate limit exceeded for this api key", "rate_limit_error",
                           {"Retry-After": str(retry_after), "X-RateLimit-Remaining": "0"})
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
    return key


# routes 

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def _record(key: ApiKey, req: ChatCompletionRequest, started: float, provider: str, model: str, status: int,
            usage: Usage | None = None, cached: bool = False, error: str | None = None) -> UsageRecord:
    prompt = usage.prompt_tokens if usage else 0
    completion = usage.completion_tokens if usage else 0
    return UsageRecord(
        key_id=key.key_id, provider=provider, model=model, stream=req.stream, cached=cached, status=status,
        latency_ms=int((time.perf_counter() - started) * 1000),
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=usage.total_tokens if usage else 0,
        # a cache hit costs nothing: no provider was called
        cost_usd=0.0 if cached else estimate_cost(model, prompt, completion),
        error=error,
    )


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest,
    request: Request,
    response: Response,
    background: BackgroundTasks,
    key: ApiKey = Depends(rate_limit),
) -> ChatCompletionResponse | StreamingResponse:
    started = time.perf_counter()
    state = request.app.state
    router: Router = state.router

    cacheable = is_cacheable(req)
    if cacheable and (hit := await state.cache.get(req)):
        response.headers["X-Cache"] = "HIT"
        # background tasks run after the response is sent, so logging doesn't add latency
        background.add_task(state.usage.record, _record(key, req, started, "cache", hit.model, 200, hit.usage, cached=True))
        return hit

    if not req.stream:
        try:
            result = await router.complete(req)
        except ProviderError as exc:
            # log the failure too, then let provider_error_handler build the response
            await state.usage.record(_record(key, req, started, exc.provider, req.model, provider_error_status(exc)[0], error=str(exc)[:500]))
            raise
        response.headers["X-Provider"] = result.provider
        if cacheable:
            response.headers["X-Cache"] = "MISS"
            background.add_task(state.cache.set, req, result.response)
        background.add_task(state.usage.record, _record(key, req, started, result.provider, result.response.model, 200, result.response.usage))
        return result.response

    try:
        # retries and fallback happen in here, up to the first chunk
        stream = await router.stream(req)
    except ProviderError as exc:
        await state.usage.record(_record(key, req, started, exc.provider, req.model, provider_error_status(exc)[0], error=str(exc)[:500]))
        raise

    return StreamingResponse(
        _sse(stream, key, req, started, state),
        media_type="text/event-stream",
        # stop proxies (e.g. nginx) from buffering the stream and sending it all at once
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Provider": stream.provider,
                 "X-RateLimit-Remaining": response.headers.get("X-RateLimit-Remaining", "")},
    )


async def _sse(stream: StreamResult, key: ApiKey, req: ChatCompletionRequest, started: float, state) -> AsyncIterator[str]:
    """Format chunks as server-sent events: 'data: <json>' + blank line, ending with 'data: [DONE]'."""
    usage: Usage | None = None
    status, error = 200, None
    try:
        if stream.first is not None:
            usage = stream.first.usage or usage
            yield f"data: {stream.first.model_dump_json(exclude_none=True)}\n\n"
        async for chunk in stream.chunks:
            # providers put token usage on the last chunk; remember it for the usage log
            usage = chunk.usage or usage
            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
        yield "data: [DONE]\n\n"
    except ProviderError as exc:
        # mid-stream failure: status 200 is already sent, so all we can do is send an error event
        # and stop. no [DONE], so the client can tell the answer is incomplete.
        status, error = provider_error_status(exc)[0], str(exc)[:500]
        payload = {"error": {"message": str(exc), "type": "upstream_error", "provider": exc.provider}}
        yield f"data: {json.dumps(payload)}\n\n"
    finally:
        # also runs if the client disconnects. starlette then cancels this task, and a cancelled task
        # can't await anything, so shield the cleanup: close the upstream connection and log the row
        with anyio.CancelScope(shield=True):
            await stream.chunks.aclose()
            await state.usage.record(_record(key, req, started, stream.provider, stream.model, status, usage, error=error))


@app.get("/usage")
async def get_usage(request: Request, days: int = Query(30, ge=1, le=365), key: ApiKey = Depends(require_api_key)) -> dict:
    """Totals for the calling api key over the last `days` days."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    summary = await request.app.state.usage.summary(key.key_id, since)
    return {"key_id": key.key_id, "name": key.name, "since": since.isoformat(), **summary}
