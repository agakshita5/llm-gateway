import asyncio

import pytest

from app.ratelimit import RateLimiter


async def test_allows_burst_up_to_capacity_then_blocks(redis):
    limiter = RateLimiter(redis, capacity=3, refill_per_s=1)
    results = [await limiter.acquire("k") for _ in range(4)]
    assert [r.allowed for r in results] == [True, True, True, False]
    assert [r.remaining for r in results[:3]] == [2, 1, 0]
    # one token short at 1 token/s -> about 1s to wait
    assert 0 < results[3].retry_after <= 1.0


async def test_refills_over_time(redis):
    limiter = RateLimiter(redis, capacity=2, refill_per_s=20)
    assert (await limiter.acquire("k")).allowed and (await limiter.acquire("k")).allowed
    assert not (await limiter.acquire("k")).allowed
    await asyncio.sleep(0.1)  # 20/s * 0.1s = 2 tokens back
    assert (await limiter.acquire("k")).allowed


async def test_keys_have_separate_buckets(redis):
    limiter = RateLimiter(redis, capacity=1, refill_per_s=0.01)
    assert (await limiter.acquire("a")).allowed
    assert not (await limiter.acquire("a")).allowed
    assert (await limiter.acquire("b")).allowed


async def test_concurrent_requests_never_overspend(redis):
    # 50 requests at the same moment against a bucket of 20: exactly 20 may pass.
    # a read-then-write in python (instead of lua) would let some race past the check.
    limiter = RateLimiter(redis, capacity=20, refill_per_s=0.001)
    results = await asyncio.gather(*(limiter.acquire("k") for _ in range(50)))
    assert sum(r.allowed for r in results) == 20


async def test_fails_open_when_redis_is_down():
    from redis.asyncio import Redis

    dead = Redis.from_url("redis://localhost:1/0", socket_connect_timeout=0.2)
    limiter = RateLimiter(dead, capacity=1, refill_per_s=1)
    assert (await limiter.acquire("k")).allowed
