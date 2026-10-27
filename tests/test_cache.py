from app.cache import ResponseCache, cache_key, is_cacheable
from app.schemas import ChatCompletionRequest
from tests.helpers import make_response


def req(**kw):
    return ChatCompletionRequest(**({"model": "m", "messages": [{"role": "user", "content": "hi"}]} | kw))


def test_only_deterministic_non_stream_requests_are_cacheable():
    assert is_cacheable(req(temperature=0))
    assert not is_cacheable(req())  # unset temperature = provider default, i.e. random
    assert not is_cacheable(req(temperature=0.7))
    assert not is_cacheable(req(temperature=0, stream=True))


def test_key_ignores_field_order_and_stream_flag():
    a = ChatCompletionRequest.model_validate({"model": "m", "temperature": 0, "messages": [{"role": "user", "content": "hi"}]})
    b = ChatCompletionRequest.model_validate({"messages": [{"content": "hi", "role": "user"}], "temperature": 0, "model": "m"})
    assert cache_key(a) == cache_key(b)
    assert cache_key(req(temperature=0)) == cache_key(req(temperature=0, stream=False))


def test_key_changes_with_any_param():
    base = cache_key(req(temperature=0))
    assert cache_key(req(temperature=0, max_tokens=5)) != base
    assert cache_key(req(temperature=0, model="other")) != base
    assert cache_key(req(temperature=0, top_p=0.5)) != base  # extra openai params count too


async def test_roundtrip_and_ttl(redis):
    cache = ResponseCache(redis, ttl_s=60)
    r = req(temperature=0)
    assert await cache.get(r) is None
    await cache.set(r, make_response(text="cached!"))
    assert (await cache.get(r)).choices[0].message.content == "cached!"
    assert 0 < await redis.ttl(cache_key(r)) <= 60
