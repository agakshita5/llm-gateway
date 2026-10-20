"""Exact-match response cache in redis (DECISIONS.md 014)."""

import hashlib
import json
import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.schemas import ChatCompletionRequest, ChatCompletionResponse

log = logging.getLogger(__name__)

# bump to invalidate every cached entry at once (e.g. after changing the response format)
CACHE_VERSION = "v1"


def is_cacheable(req: ChatCompletionRequest) -> bool:
    # only deterministic requests: unset temperature means the provider default (~1), i.e. random answers
    return not req.stream and req.temperature == 0


def cache_key(req: ChatCompletionRequest) -> str:
    body = req.model_dump(exclude_none=True, exclude={"stream"})
    # sort_keys + fixed separators: the same request always gives the same bytes, whatever order the client sent
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return f"cache:{CACHE_VERSION}:{hashlib.sha256(canonical.encode()).hexdigest()}"


class ResponseCache:
    def __init__(self, redis: Redis, ttl_s: int):
        self._redis = redis
        self._ttl_s = ttl_s

    async def get(self, req: ChatCompletionRequest) -> ChatCompletionResponse | None:
        try:
            raw = await self._redis.get(cache_key(req))
        except RedisError:
            # fail open: a broken cache is just a cache miss
            log.exception("cache read failed")
            return None
        return ChatCompletionResponse.model_validate_json(raw) if raw else None

    async def set(self, req: ChatCompletionRequest, response: ChatCompletionResponse) -> None:
        try:
            # SET with ex=ttl: redis deletes the entry itself when it expires
            await self._redis.set(cache_key(req), response.model_dump_json(), ex=self._ttl_s)
        except RedisError:
            log.exception("cache write failed")
