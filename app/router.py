"""Picks a provider for each request, retries with backoff, and falls back to the other provider.

Policy (DECISIONS.md 010):
  ProviderRequestError      -> give up now (the client's input is bad; every provider would reject it)
  ProviderAuthError         -> skip to the next provider (our key for this one is bad)
  ProviderRateLimitError    -> retry once if Retry-After is short, else next provider
  ProviderUnavailableError  -> retry with backoff, then next provider
"""

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from app.providers.base import (
    Provider,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderRequestError,
)
from app.schemas import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse

log = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass
class CompletionResult:
    response: ChatCompletionResponse
    provider: str


@dataclass
class StreamResult:
    provider: str
    model: str
    # already pulled from the provider so failures before it could still fall back (DECISIONS.md 006)
    first: ChatCompletionChunk | None
    chunks: AsyncIterator[ChatCompletionChunk]


class Router:
    def __init__(
        self,
        providers: dict[str, Provider],
        fallback_models: dict[str, str],
        max_attempts: int = 2,
        backoff_base_s: float = 0.5,
        backoff_max_s: float = 4.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if not providers:
            raise ValueError("router needs at least one provider")
        self._providers = providers
        self._fallback_models = fallback_models
        self._max_attempts = max_attempts
        self._base = backoff_base_s
        self._cap = backoff_max_s
        # injectable so tests don't actually wait
        self._sleep = sleep

    @property
    def providers(self) -> list[Provider]:
        return list(self._providers.values())

    def plan(self, model: str) -> list[tuple[Provider, str]]:
        """Ordered (provider, model) pairs to try: the model's own provider first, then fallbacks."""
        primary = "gemini" if model.startswith("gemini") else "groq"
        plan = []
        if primary in self._providers:
            plan.append((self._providers[primary], model))
        for name, provider in self._providers.items():
            if name != primary:
                # the other provider can't serve this model name, so use its default model
                plan.append((provider, self._fallback_models[name]))
        return plan

    def _backoff(self, attempt: int) -> float:
        # "full jitter": random wait between 0 and the exponential ceiling, so many clients
        # retrying at once don't all hit the provider again at the same instant
        return random.uniform(0, min(self._cap, self._base * 2**attempt))

    async def _run(self, req: ChatCompletionRequest, call: Callable[[Provider, ChatCompletionRequest], Awaitable[T]]) -> tuple[Provider, str, T]:
        last_error: ProviderError | None = None

        for provider, model in self.plan(req.model):
            # model_copy keeps the client's other params, only the model name changes
            attempt_req = req if model == req.model else req.model_copy(update={"model": model})

            for attempt in range(self._max_attempts):
                try:
                    return provider, model, await call(provider, attempt_req)
                except ProviderRequestError:
                    # bad input from the client: retrying or switching won't help
                    raise
                except ProviderAuthError as e:
                    last_error = e
                    log.warning("provider %s auth failed, falling back: %s", provider.name, e)
                    break
                except ProviderRateLimitError as e:
                    last_error = e
                    is_last_attempt = attempt == self._max_attempts - 1
                    # only wait if the provider told us a short wait; a long one means "go elsewhere"
                    if e.retry_after is not None and e.retry_after <= self._cap and not is_last_attempt:
                        log.warning("provider %s rate-limited, retrying in %.1fs", provider.name, e.retry_after)
                        await self._sleep(e.retry_after)
                        continue
                    log.warning("provider %s rate-limited, falling back", provider.name)
                    break
                except ProviderError as e:
                    # unavailable (5xx / timeout / network) or anything else from the provider
                    last_error = e
                    if attempt < self._max_attempts - 1:
                        delay = self._backoff(attempt)
                        log.warning("provider %s failed (%s), retry %d in %.2fs", provider.name, e, attempt + 1, delay)
                        await self._sleep(delay)
                        continue
                    log.warning("provider %s failed after %d attempts, falling back", provider.name, self._max_attempts)
                    break

        # every provider failed; the client sees the last error
        assert last_error is not None
        raise last_error

    async def complete(self, req: ChatCompletionRequest) -> CompletionResult:
        async def call(provider: Provider, r: ChatCompletionRequest) -> ChatCompletionResponse:
            return await provider.complete(r)

        provider, _, response = await self._run(req, call)
        return CompletionResult(response=response, provider=provider.name)

    async def stream(self, req: ChatCompletionRequest) -> StreamResult:
        async def call(provider: Provider, r: ChatCompletionRequest):
            chunks = provider.stream(r)
            # pull the first chunk inside the retry loop: errors up to here can still retry or fall back.
            # after this returns, the stream is committed to this provider
            try:
                first = await anext(chunks)
            except StopAsyncIteration:
                first = None
            return chunks, first

        provider, model, (chunks, first) = await self._run(req, call)
        return StreamResult(provider=provider.name, model=model, first=first, chunks=chunks)
