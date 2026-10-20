"""Gemini adapter: translates openai-format requests/responses to gemini's native generateContent api.

openai                         gemini
------                         ------
model (in body)                model (in url path)
messages[role=system]          systemInstruction.parts[].text
messages[role=user|assistant]  contents[role=user|model].parts[].text
temperature / max_tokens       generationConfig.temperature / maxOutputTokens
choices[0].message.content     candidates[0].content.parts[].text
usage.prompt_tokens            usageMetadata.promptTokenCount
"""

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import (
    Provider,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailableError,
)
from ._http import parse_retry_after
from app.config import get_settings
from app.schemas import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChunkChoice,
    Delta,
    Usage,
)

# gemini finish reasons -> openai finish_reason
_FINISH_REASONS = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
}


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self):
        super().__init__()
        config = get_settings()
        self._client = httpx.AsyncClient(
            base_url=config.gemini_base_url,
            # gemini takes the key in its own header, not "Authorization: Bearer"
            headers={"x-goog-api-key": config.gemini_api_key},
            timeout=config.upstream_timeout_s,
        )

    async def complete(self, req: ChatCompletionRequest) -> ChatCompletionResponse:
        try:
            response = await self._client.post(f"/models/{req.model}:generateContent", json=self._to_gemini(req))
        except httpx.TimeoutException as e:
            raise ProviderUnavailableError(self.name, "timeout") from e
        except httpx.TransportError as e:
            raise ProviderUnavailableError(self.name, f"network error: {e}") from e

        self._raise_for_status(response)
        try:
            return self._to_openai(response.json(), req.model)
        except (ValueError, KeyError, TypeError) as e:
            # 200 but not the shape we expect: provider's fault (DECISIONS.md 011)
            raise ProviderUnavailableError(self.name, f"unexpected response body: {e}") from e

    async def stream(self, req: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        # one id/created for the whole stream, like openai does
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        first = True

        try:
            # alt=sse makes gemini stream as server-sent events instead of one big json array
            async with self._client.stream(
                "POST", f"/models/{req.model}:streamGenerateContent", params={"alt": "sse"}, json=self._to_gemini(req)
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    self._raise_for_status(response)

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    # unlike openai, every gemini event is a full response object, and there's no [DONE]:
                    # the stream simply ends when the body ends
                    try:
                        data = json.loads(line[len("data:"):])
                        chunk = self._to_chunk(data, chunk_id, created, req.model, first)
                    except (ValueError, KeyError, TypeError) as e:
                        raise ProviderUnavailableError(self.name, f"bad stream event: {line[:200]}") from e
                    first = False
                    yield chunk
        except httpx.TimeoutException as e:
            raise ProviderUnavailableError(self.name, "timeout") from e
        except httpx.TransportError as e:
            raise ProviderUnavailableError(self.name, f"network error: {e}") from e

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- request translation -------------------------------------------------

    def _to_gemini(self, req: ChatCompletionRequest) -> dict[str, Any]:
        system_parts: list[dict] = []
        contents: list[dict] = []
        for msg in req.messages:
            if msg.content is None:
                # e.g. an assistant message that only had tool calls; nothing to send as text
                continue
            if msg.role == "system":
                # gemini has no system role inside contents; it's a separate top-level field
                system_parts.append({"text": msg.content})
            elif msg.role in ("user", "assistant"):
                # gemini calls the assistant "model"
                contents.append({"role": "user" if msg.role == "user" else "model", "parts": [{"text": msg.content}]})
            else:
                raise ProviderRequestError(self.name, f"role '{msg.role}' is not supported by the gemini adapter", 400)

        body: dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        # optional openai params that have a gemini equivalent; others are silently dropped
        extra = req.model_extra or {}
        config: dict[str, Any] = {}
        if req.temperature is not None:
            config["temperature"] = req.temperature
        if req.max_tokens is not None:
            config["maxOutputTokens"] = req.max_tokens
        if extra.get("top_p") is not None:
            config["topP"] = extra["top_p"]
        if extra.get("stop"):
            # openai allows a string or a list; gemini wants a list
            stop = extra["stop"]
            config["stopSequences"] = [stop] if isinstance(stop, str) else stop
        if config:
            body["generationConfig"] = config
        return body

    # --- response translation ------------------------------------------------

    def _text_and_finish(self, data: dict) -> tuple[str, str | None]:
        candidates = data.get("candidates") or []
        if not candidates:
            # no candidates means the prompt itself was blocked (promptFeedback.blockReason)
            return "", "content_filter" if data.get("promptFeedback") else None
        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        # skip "thought" parts: thinking models return their reasoning separately from the answer
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        reason = cand.get("finishReason")
        return text, (_FINISH_REASONS.get(reason, "stop") if reason else None)

    def _usage(self, data: dict) -> Usage | None:
        meta = data.get("usageMetadata")
        if not meta:
            return None
        prompt = meta.get("promptTokenCount", 0)
        # thinking tokens are billed as output, so count them as completion tokens
        completion = meta.get("candidatesTokenCount", 0) + meta.get("thoughtsTokenCount", 0)
        return Usage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=meta.get("totalTokenCount", prompt + completion))

    def _to_openai(self, data: dict, model: str) -> ChatCompletionResponse:
        text, finish = self._text_and_finish(data)
        return ChatCompletionResponse(
            id=f"chatcmpl-{data.get('responseId') or uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[Choice(index=0, message=ChatMessage(role="assistant", content=text), finish_reason=finish or "stop")],
            usage=self._usage(data),
        )

    def _to_chunk(self, data: dict, chunk_id: str, created: int, model: str, first: bool) -> ChatCompletionChunk:
        text, finish = self._text_and_finish(data)
        return ChatCompletionChunk(
            id=chunk_id,
            created=created,
            model=model,
            # openai sends role only on the first chunk
            choices=[ChunkChoice(index=0, delta=Delta(role="assistant" if first else None, content=text or None), finish_reason=finish)],
            # gemini repeats running usage on many events; only report it once, on the final chunk
            usage=self._usage(data) if finish else None,
        )

    # --- errors --------------------------------------------------------------

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Translate a non-2xx gemini response into our gateway errors."""
        status = response.status_code
        if status < 400:
            return
        # gemini errors look like {"error": {"code", "message", "status", "details": [...]}}
        try:
            error = response.json().get("error", {})
        except ValueError:
            error = {}
        details = error.get("details") or []
        message = error.get("message") or response.text

        if status == 429:
            raise ProviderRateLimitError(self.name, message, retry_after=self._retry_after(response, details))
        if status >= 500:
            raise ProviderUnavailableError(self.name, message, status)
        # a wrong key comes back as 400 with reason API_KEY_INVALID, not 401 (DECISIONS.md 005)
        bad_key = any(d.get("reason") == "API_KEY_INVALID" for d in details)
        if status in (401, 403) or bad_key:
            raise ProviderAuthError(self.name, message, status)
        raise ProviderRequestError(self.name, message, status)

    def _retry_after(self, response: httpx.Response, details: list[dict]) -> float | None:
        # prefer the standard header, then gemini's RetryInfo detail, e.g. {"retryDelay": "7s"}
        header = parse_retry_after(response.headers.get("retry-after"))
        if header is not None:
            return header
        for d in details:
            delay = d.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    return None
        return None
