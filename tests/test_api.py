import json

from app.providers.base import ProviderAuthError, ProviderUnavailableError
from tests.helpers import make_chunks, make_response

BODY = {"model": "llama", "messages": [{"role": "user", "content": "hi"}]}


def test_health(gateway):
    assert gateway.client.get("/health").json() == {"status": "ok"}


def test_missing_and_invalid_api_key(gateway):
    r = gateway.client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 401 and r.json()["error"]["type"] == "invalid_request_error"
    r = gateway.client.post("/v1/chat/completions", json=BODY, headers={"Authorization": "Bearer gw-nope"})
    assert r.status_code == 401


async def test_completion_logs_usage(gateway):
    await gateway.new_key()
    gateway.use(groq=[make_response("openai/gpt-oss-20b", tokens=(1000, 500))], gemini=[])
    r = gateway.client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "hi"
    assert r.headers["x-provider"] == "groq" and r.headers["x-ratelimit-remaining"] == "4"
    row = gateway.usage.rows[0]
    assert (row.provider, row.status, row.total_tokens, row.cached) == ("groq", 200, 1500, False)
    # 1000 in * $0.075/1M + 500 out * $0.30/1M
    assert abs(row.cost_usd - 0.000225) < 1e-9


async def test_fallback_is_visible_in_header(gateway):
    await gateway.new_key()
    gateway.use(groq=[ProviderAuthError("groq", "bad key", 401)], gemini=[make_response("gemini-default")])
    r = gateway.client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 200 and r.headers["x-provider"] == "gemini" and r.json()["model"] == "gemini-default"


async def test_all_providers_down_is_502_and_logged(gateway):
    await gateway.new_key()
    down = ProviderUnavailableError("x", "down", 503)
    gateway.use(groq=[down, down], gemini=[down, down])
    r = gateway.client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_error"
    assert gateway.usage.rows[0].status == 502


async def test_rate_limit_returns_429_with_retry_after(gateway):
    await gateway.new_key()
    gateway.use(groq=[make_response()] * 5, gemini=[])
    codes = [gateway.client.post("/v1/chat/completions", json=BODY).status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]
    r = gateway.client.post("/v1/chat/completions", json=BODY)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["type"] == "rate_limit_error"


async def test_cache_hit_skips_provider(gateway):
    await gateway.new_key()
    gateway.use(groq=[make_response(text="first")], gemini=[])
    body = BODY | {"temperature": 0}
    r1 = gateway.client.post("/v1/chat/completions", json=body)
    r2 = gateway.client.post("/v1/chat/completions", json=body)
    assert r1.headers["x-cache"] == "MISS" and r2.headers["x-cache"] == "HIT"
    assert r2.json()["choices"][0]["message"]["content"] == "first"
    assert len(gateway.providers["groq"].calls) == 1
    assert gateway.usage.rows[-1].cached and gateway.usage.rows[-1].cost_usd == 0


async def test_stream_ok_ends_with_done_and_logs_usage(gateway):
    await gateway.new_key()
    gateway.use(groq=[make_chunks()], gemini=[])
    r = gateway.client.post("/v1/chat/completions", json=BODY | {"stream": True})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = [line.removeprefix("data: ") for line in r.text.split("\n\n") if line]
    assert events[-1] == "[DONE]"
    assert [json.loads(e)["choices"][0]["delta"].get("content") for e in events[:-1]] == ["Hel", "lo", None]
    assert gateway.usage.rows[0].total_tokens == 5 and gateway.usage.rows[0].stream


async def test_stream_error_before_first_chunk_is_real_status(gateway):
    await gateway.new_key()
    bad = ProviderAuthError("x", "bad key", 401)
    gateway.use(groq=[bad], gemini=[bad])
    r = gateway.client.post("/v1/chat/completions", json=BODY | {"stream": True})
    assert r.status_code == 502 and r.headers["content-type"] == "application/json"


async def test_mid_stream_error_sends_error_event_without_done(gateway):
    await gateway.new_key()
    gateway.use(groq=[[make_chunks()[0], ProviderUnavailableError("groq", "died", None)]], gemini=[])
    r = gateway.client.post("/v1/chat/completions", json=BODY | {"stream": True})
    assert r.status_code == 200
    assert '"error"' in r.text and "[DONE]" not in r.text
    assert gateway.usage.rows[0].status == 502


async def test_usage_endpoint_only_shows_own_key(gateway):
    await gateway.new_key("alice")
    gateway.use(groq=[make_response(tokens=(3, 2))] * 2, gemini=[])
    gateway.client.post("/v1/chat/completions", json=BODY)
    gateway.client.post("/v1/chat/completions", json=BODY)
    alice = gateway.client.get("/usage").json()
    assert alice["requests"] == 2 and alice["total_tokens"] == 10 and alice["name"] == "alice"
    await gateway.new_key("bob")
    assert gateway.client.get("/usage").json()["requests"] == 0


async def test_validation_error_is_openai_400(gateway):
    await gateway.new_key()
    r = gateway.client.post("/v1/chat/completions", json={"model": "m"})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
