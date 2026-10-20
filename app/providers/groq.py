import json
from collections.abc import AsyncIterator

import httpx
from pydantic import ValidationError

from .base import (
    Provider,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailableError,
)
from ._http import parse_retry_after
from app.schemas import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse
from app.config import get_settings


class GroqProvider(Provider):
    name = "groq"

    def __init__(self):
        super().__init__()
        config = get_settings()
        self._client = httpx.AsyncClient(
            base_url=config.groq_base_url,
            headers={"Authorization": f"Bearer {config.groq_api_key}"},
            timeout=config.upstream_timeout_s,
        )

    async def complete(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            response = await self._client.post("/chat/completions", json=req.model_dump(exclude_none=True))
        except httpx.TimeoutException as e:
            raise ProviderUnavailableError(self.name, "timeout") from e
        except httpx.TransportError as e:          # DNS, connection refused, connection reset...
            raise ProviderUnavailableError(self.name, f"network error: {e}") from e

        self._raise_for_status(response)
        try:
            return ChatCompletionResponse.model_validate_json(response.content)
        except ValidationError as e:
            # 200 but not the shape we expect: provider's fault, so treat like a 5xx (DECISIONS.md 011)
            raise ProviderUnavailableError(self.name, f"unexpected response body: {e}") from e

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        # force stream on, whatever the caller set
        body = req.model_dump(exclude_none=True) | {"stream": True}

        try:
            # client.stream() returns headers first and lets us read the body piece by piece,
            # instead of waiting for the whole response like post() does
            async with self._client.stream("POST", "/chat/completions", json=body) as response:
                if response.status_code >= 400:
                    # error bodies are small; read it fully so the error message has the text
                    await response.aread()
                    self._raise_for_status(response)

                # sse format: each event is a "data: <json>" line, events separated by blank lines
                async for line in response.aiter_lines():
                    # skip blank separators and ":" comment lines (used as keep-alives)
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue

                    payload = line[len("data:"):].strip()
                    # groq ends the stream with a literal [DONE], not json
                    if payload == "[DONE]":
                        return

                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError as e:
                        raise ProviderUnavailableError(self.name, f"bad stream event: {payload[:200]}") from e

                    # provider can fail after sending 200; it reports that as an error event
                    if "error" in data:
                        raise ProviderUnavailableError(self.name, f"mid-stream error: {data['error']}")

                    # groq puts token usage of a stream in x_groq.usage on the last chunk,
                    # not in the standard "usage" field; move it so callers only look in one place
                    groq_usage = (data.get("x_groq") or {}).get("usage")
                    if groq_usage and not data.get("usage"):
                        data["usage"] = groq_usage

                    try:
                        chunk = ChatCompletionChunk.model_validate(data)
                    except ValidationError as e:
                        raise ProviderUnavailableError(self.name, f"unexpected stream chunk: {e}") from e
                    yield chunk
        except httpx.TimeoutException as e:
            # with streaming, the read timeout is per chunk: no new bytes for N seconds
            raise ProviderUnavailableError(self.name, "timeout") from e
        except httpx.TransportError as e:
            raise ProviderUnavailableError(self.name, f"network error: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate a non-2xx groq response into our gateway errors. Shared by complete() and stream()."""
        if response.status_code == 429:
            raise ProviderRateLimitError(
                self.name, response.text, retry_after=parse_retry_after(response.headers.get("retry-after"))
            )
        elif response.status_code >= 500:
            raise ProviderUnavailableError(self.name, response.text, response.status_code)
        elif response.status_code in (401, 403):
            raise ProviderAuthError(self.name, response.text, response.status_code)
        elif response.status_code >= 400:
            raise ProviderRequestError(self.name, response.text, response.status_code)
