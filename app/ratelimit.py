"""Per-key token bucket in redis, made atomic with a lua script (DECISIONS.md 013).

each key has a bucket of `capacity` tokens that refills at `refill_per_s`. a request takes 1 token;
an empty bucket means 429. the bucket is stored as just two numbers: tokens left and last update time.
"""

import logging
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

# runs inside redis as one atomic step: no other command can run between the read and the write,
# so two concurrent requests can never both take the last token
TOKEN_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local refill_per_s = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])

-- use redis's clock, not the caller's: every gateway instance then agrees on "now"
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000

local bucket = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
-- a key we've never seen starts with a full bucket
local tokens = tonumber(bucket[1]) or capacity
local ts = tonumber(bucket[2]) or now

-- refill for the time that passed since the last request, never above capacity
tokens = math.min(capacity, tokens + math.max(0, now - ts) * refill_per_s)

local allowed = 0
local retry_after = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
else
  -- how long until enough tokens have refilled
  retry_after = (cost - tokens) / refill_per_s
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
-- an idle bucket refills completely after capacity/rate seconds, so it can be deleted then
redis.call('EXPIRE', KEYS[1], math.ceil(capacity / refill_per_s) + 1)

-- lua numbers come back to redis as integers (0.4 -> 0), so send fractions as strings
return {allowed, tostring(retry_after), tostring(tokens)}
"""


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after: float
    remaining: int


class RateLimiter:
    def __init__(self, redis: Redis, capacity: int, refill_per_s: float):
        self._capacity = capacity
        self._refill = refill_per_s
        # register_script sends the script once and then calls it by its sha (EVALSHA)
        self._script = redis.register_script(TOKEN_BUCKET_LUA)

    async def acquire(self, key_id: str, cost: int = 1) -> RateLimitDecision:
        try:
            allowed, retry_after, tokens = await self._script(keys=[f"ratelimit:{key_id}"], args=[self._capacity, self._refill, cost])
        except RedisError:
            # fail open: a redis outage shouldn't take the whole gateway down
            log.exception("rate limiter unavailable, allowing request")
            return RateLimitDecision(allowed=True, retry_after=0.0, remaining=self._capacity)
        return RateLimitDecision(allowed=bool(int(allowed)), retry_after=float(retry_after), remaining=int(float(tokens)))
