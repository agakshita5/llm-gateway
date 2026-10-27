import json

import httpx
import pytest

from app.providers.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailableError,
)
from app.providers.gemini import GeminiProvider
from app.schemas import ChatCompletionRequest

REQ = ChatCompletionRequest(
    model="gemini-2.5-flash",
    messages=[{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Hi"},
              {"role": "assistant", "content": "Hello"}, {"role": "user", "content": "Bye"}],
    temperature=0.5, max_tokens=50, stop="END",
)
OK = {"candidates": [{"content": {"role": "model", "parts": [{"text": "thinking...", "thought": True}, {"text": "Bye!"}]},
                      "finishReason": "STOP"}],
      "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "thoughtsTokenCount": 5, "totalTokenCount": 17},
      "responseId": "abc"}


def gemini_with(handler) -> GeminiProvider:
    p = GeminiProvider()
    p._client = httpx.AsyncClient(base_url="https://gemini.test/v1beta", transport=httpx.MockTransport(handler))
    return p


async def test_request_translation():
    seen = {}

    def handler(r):
        seen["url"], seen["body"] = str(r.url), json.loads(r.content)
        return httpx.Response(200, json=OK)

    await gemini_with(handler).complete(REQ)
    body = seen["body"]
    # model goes in the url, not the body
    assert seen["url"].endswith("/models/gemini-2.5-flash:generateContent")
    assert body["systemInstruction"] == {"parts": [{"text": "Be brief"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert body["generationConfig"] == {"temperature": 0.5, "maxOutputTokens": 50, "stopSequences": ["END"]}


async def test_response_translation_skips_thoughts_and_counts_them_as_output():
    out = await gemini_with(lambda r: httpx.Response(200, json=OK)).complete(REQ)
    assert out.choices[0].message.content == "Bye!"
    assert out.choices[0].finish_reason == "stop"
    assert (out.usage.prompt_tokens, out.usage.completion_tokens, out.usage.total_tokens) == (10, 7, 17)
    assert out.model == "gemini-2.5-flash"


async def test_blocked_prompt_is_content_filter():
    body = {"promptFeedback": {"blockReason": "SAFETY"}}
    out = await gemini_with(lambda r: httpx.Response(200, json=body)).complete(REQ)
    assert out.choices[0].finish_reason == "content_filter"


async def test_tool_role_rejected():
    req = ChatCompletionRequest(model="gemini-x", messages=[{"role": "tool", "content": "x"}])
    with pytest.raises(ProviderRequestError):
        await gemini_with(lambda r: httpx.Response(200, json=OK)).complete(req)


def gemini_error(status, reason=None, retry_delay=None):
    details = []
    if reason:
        details.append({"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": reason})
    if retry_delay:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    return httpx.Response(status, json={"error": {"code": status, "message": "boom", "status": "X", "details": details}})


async def test_bad_key_400_becomes_auth_error():
    with pytest.raises(ProviderAuthError):
        await gemini_with(lambda r: gemini_error(400, reason="API_KEY_INVALID")).complete(REQ)


async def test_plain_400_is_request_error():
    with pytest.raises(ProviderRequestError):
        await gemini_with(lambda r: gemini_error(400)).complete(REQ)


async def test_429_reads_retry_info():
    with pytest.raises(ProviderRateLimitError) as e:
        await gemini_with(lambda r: gemini_error(429, retry_delay="3.5s")).complete(REQ)
    assert e.value.retry_after == 3.5


async def test_503_is_unavailable():
    with pytest.raises(ProviderUnavailableError):
        await gemini_with(lambda r: gemini_error(503)).complete(REQ)


async def test_stream_translation():
    events = [
        {"candidates": [{"content": {"parts": [{"text": "By"}]}}], "usageMetadata": {"promptTokenCount": 10}},
        {"candidates": [{"content": {"parts": [{"text": "e!"}]}, "finishReason": "STOP"}],
         "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12}},
    ]
    body = "".join(f"data: {json.dumps(e)}\r\n\r\n" for e in events)
    seen = {}

    def handler(r):
        seen["url"] = str(r.url)
        return httpx.Response(200, text=body)

    chunks = [c async for c in gemini_with(handler).stream(REQ)]
    assert "streamGenerateContent?alt=sse" in seen["url"]
    assert [c.choices[0].delta.content for c in chunks] == ["By", "e!"]
    assert chunks[0].choices[0].delta.role == "assistant" and chunks[1].choices[0].delta.role is None
    # usage only on the final chunk, even though gemini sent partial usage earlier
    assert chunks[0].usage is None and chunks[1].usage.total_tokens == 12
    assert chunks[1].choices[0].finish_reason == "stop"
    assert len({c.id for c in chunks}) == 1
