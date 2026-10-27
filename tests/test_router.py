import pytest

from app.providers.base import (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderUnavailableError,
)
from app.router import Router
from app.schemas import ChatCompletionRequest
from tests.helpers import ScriptedProvider, make_chunks, make_response

FALLBACKS = {"groq": "groq-default", "gemini": "gemini-default"}


def req(model="llama", stream=False):
    return ChatCompletionRequest(model=model, messages=[{"role": "user", "content": "hi"}], stream=stream)


def router(groq, gemini, sleeps=None):
    async def fake_sleep(s):
        if sleeps is not None:
            sleeps.append(s)

    providers = {"groq": ScriptedProvider("groq", groq), "gemini": ScriptedProvider("gemini", gemini)}
    return Router(providers, FALLBACKS, max_attempts=2, backoff_base_s=0.5, backoff_max_s=4.0, sleep=fake_sleep), providers


def down():
    return ProviderUnavailableError("x", "down", 503)


async def test_success_first_try():
    r, p = router([make_response()], [])
    result = await r.complete(req())
    assert result.provider == "groq" and len(p["groq"].calls) == 1


async def test_routes_gemini_models_to_gemini_first():
    r, p = router([], [make_response()])
    assert (await r.complete(req("gemini-2.5-flash"))).provider == "gemini"


async def test_retries_unavailable_with_backoff_then_succeeds():
    sleeps = []
    r, p = router([down(), make_response()], [], sleeps)
    result = await r.complete(req())
    assert result.provider == "groq" and len(p["groq"].calls) == 2
    # full jitter: a random wait between 0 and base * 2^0
    assert len(sleeps) == 1 and 0 <= sleeps[0] <= 0.5


async def test_falls_back_after_max_attempts_with_fallback_model():
    r, p = router([down(), down()], [make_response("gemini-default")])
    result = await r.complete(req("llama"))
    assert result.provider == "gemini"
    assert p["gemini"].calls[0].model == "gemini-default"
    # other params are kept when the model is swapped
    assert p["gemini"].calls[0].messages == req().messages


async def test_auth_error_falls_back_without_retry():
    r, p = router([ProviderAuthError("groq", "bad key", 401)], [make_response()])
    assert (await r.complete(req())).provider == "gemini"
    assert len(p["groq"].calls) == 1


async def test_request_error_neither_retries_nor_falls_back():
    r, p = router([ProviderRequestError("groq", "bad model", 404)], [make_response()])
    with pytest.raises(ProviderRequestError):
        await r.complete(req())
    assert len(p["gemini"].calls) == 0


async def test_short_rate_limit_waits_retry_after_then_retries():
    sleeps = []
    r, p = router([ProviderRateLimitError("groq", "slow", retry_after=2), make_response()], [], sleeps)
    assert (await r.complete(req())).provider == "groq"
    assert sleeps == [2]


@pytest.mark.parametrize("retry_after", [30, None])
async def test_long_or_unknown_rate_limit_falls_back_immediately(retry_after):
    sleeps = []
    r, p = router([ProviderRateLimitError("groq", "slow", retry_after=retry_after)], [make_response()], sleeps)
    assert (await r.complete(req())).provider == "gemini"
    assert sleeps == []


async def test_all_fail_raises_last_error():
    r, _ = router([down(), down()], [ProviderAuthError("gemini", "bad key", 403)])
    with pytest.raises(ProviderAuthError):
        await r.complete(req())


async def test_stream_falls_back_when_failing_before_first_chunk():
    r, p = router([ProviderAuthError("groq", "bad", 401)], [make_chunks()])
    result = await r.stream(req(stream=True))
    assert result.provider == "gemini" and result.first.choices[0].delta.content == "Hel"
    rest = [c async for c in result.chunks]
    assert len(rest) == 2


async def test_stream_does_not_fall_back_after_first_chunk():
    chunks = make_chunks()
    r, p = router([[chunks[0], down()]], [make_chunks()])
    result = await r.stream(req(stream=True))
    assert result.provider == "groq"
    with pytest.raises(ProviderUnavailableError):
        [c async for c in result.chunks]
    assert len(p["gemini"].calls) == 0


async def test_unconfigured_primary_goes_straight_to_other_provider():
    providers = {"groq": ScriptedProvider("groq", [make_response("groq-default")])}
    r = Router(providers, FALLBACKS)
    result = await r.complete(req("gemini-2.5-flash"))
    assert result.provider == "groq" and providers["groq"].calls[0].model == "groq-default"
