import json

import httpx
import pytest

from app.providers.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailableError,
)
from app.providers.groq import GroqProvider
from app.schemas import ChatCompletionRequest

REQ = ChatCompletionRequest(model="openai/gpt-oss-20b", messages=[{"role": "user", "content": "hi"}])
OK = {"id": "x", "created": 1, "model": "m", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
      "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]}


def groq_with(handler) -> GroqProvider:
    p = GroqProvider()
    p._client = httpx.AsyncClient(base_url="https://groq.test/openai/v1", transport=httpx.MockTransport(handler))
    return p


def sse(*events) -> str:
    return "".join(f"data: {e if isinstance(e, str) else json.dumps(e)}\n\n" for e in events)


def chunk(content=None, finish=None, **extra):
    return {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish}], **extra}


async def test_complete_ok_sends_openai_body():
    seen = {}

    def handler(r):
        seen["path"], seen["body"] = r.url.path, json.loads(r.content)
        return httpx.Response(200, json=OK)

    out = await groq_with(handler).complete(REQ)
    assert out.usage.total_tokens == 2
    assert seen["path"] == "/openai/v1/chat/completions"
    # exclude_none: unset params like temperature are not sent as null
    assert "temperature" not in seen["body"]


@pytest.mark.parametrize("status,exc", [
    (400, ProviderRequestError), (404, ProviderRequestError),
    (401, ProviderAuthError), (403, ProviderAuthError),
    (500, ProviderUnavailableError), (503, ProviderUnavailableError),
])
async def test_status_mapping(status, exc):
    with pytest.raises(exc):
        await groq_with(lambda r: httpx.Response(status, text="err")).complete(REQ)


async def test_429_parses_retry_after():
    p = groq_with(lambda r: httpx.Response(429, headers={"retry-after": "7"}, text="slow"))
    with pytest.raises(ProviderRateLimitError) as e:
        await p.complete(REQ)
    assert e.value.retry_after == 7.0


async def test_timeout_becomes_unavailable():
    def handler(r):
        raise httpx.ReadTimeout("slow", request=r)

    with pytest.raises(ProviderUnavailableError):
        await groq_with(handler).complete(REQ)


async def test_bad_200_body_becomes_unavailable():
    with pytest.raises(ProviderUnavailableError):
        await groq_with(lambda r: httpx.Response(200, json={"nope": 1})).complete(REQ)


async def test_stream_parses_chunks_and_moves_x_groq_usage():
    body = sse(chunk("Hel"), chunk("lo"), chunk(finish="stop", x_groq={"usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}), "[DONE]")
    body = ": keep-alive\n\n" + body
    p = groq_with(lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    chunks = [c async for c in p.stream(REQ)]
    assert [c.choices[0].delta.content for c in chunks] == ["Hel", "lo", None]
    assert chunks[-1].usage.total_tokens == 5


async def test_stream_error_status_raises_before_any_chunk():
    p = groq_with(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(ProviderAuthError):
        [c async for c in p.stream(REQ)]


async def test_stream_mid_stream_error_event():
    body = sse(chunk("Hel"), {"error": {"message": "overloaded"}})
    p = groq_with(lambda r: httpx.Response(200, text=body))
    got = []
    with pytest.raises(ProviderUnavailableError):
        async for c in p.stream(REQ):
            got.append(c)
    assert len(got) == 1
