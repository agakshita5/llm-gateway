import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.cache import ResponseCache
from app.keys import create_key
from app.main import app
from app.ratelimit import RateLimiter
from app.router import Router
from app.usage import InMemoryUsageStore


async def _no_sleep(_seconds: float) -> None:
    pass


@pytest.fixture
def redis():
    # in-memory redis that also runs lua scripts (via the lupa package)
    return fakeredis.FakeAsyncRedis(decode_responses=True)


@pytest.fixture
def gateway(redis):
    """The real app wired to fakes: fakeredis, in-memory usage store, scripted providers.

    call gateway.use(groq=[...], gemini=[...]) to script provider outcomes, then gateway.client to make requests.
    """

    class Gateway:
        def __init__(self):
            self.usage = InMemoryUsageStore()
            self.client = TestClient(app)  # no "with": the real lifespan (real redis/postgres) doesn't run
            app.state.redis = redis
            app.state.usage = self.usage
            app.state.limiter = RateLimiter(redis, capacity=5, refill_per_s=0.01)
            app.state.cache = ResponseCache(redis, ttl_s=60)
            self.use(groq=[], gemini=[])

        def use(self, **outcomes):
            from tests.helpers import ScriptedProvider

            self.providers = {name: ScriptedProvider(name, script) for name, script in outcomes.items()}
            app.state.router = Router(self.providers, {"groq": "groq-default", "gemini": "gemini-default"}, sleep=_no_sleep)

        async def new_key(self, name="test") -> str:
            raw, _ = await create_key(redis, name)
            self.client.headers["Authorization"] = f"Bearer {raw}"
            return raw

    return Gateway()
