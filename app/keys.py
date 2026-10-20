"""Gateway API keys, stored hashed in redis (DECISIONS.md 012).

create a key:  python -m app.keys create --name alice
list keys:     python -m app.keys list
revoke a key:  python -m app.keys revoke <key_id>
"""

import argparse
import asyncio
import hashlib
import secrets
import time
from dataclasses import dataclass

from redis.asyncio import Redis

from app.config import get_settings

KEY_PREFIX = "gw-"


@dataclass
class ApiKey:
    # short, non-secret id derived from the hash: used in logs, rate-limit buckets and usage rows
    key_id: str
    name: str


def hash_key(raw: str) -> str:
    # sha256 is fine here (unlike passwords): keys are long random strings, so brute force is hopeless
    return hashlib.sha256(raw.encode()).hexdigest()


def _redis_key(digest: str) -> str:
    return f"apikey:{digest}"


async def create_key(redis: Redis, name: str) -> tuple[str, ApiKey]:
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    digest = hash_key(raw)
    await redis.hset(_redis_key(digest), mapping={"name": name, "created_at": int(time.time())})
    # the raw key is returned once and never stored
    return raw, ApiKey(key_id=digest[:16], name=name)


async def lookup_key(redis: Redis, raw: str) -> ApiKey | None:
    digest = hash_key(raw)
    name = await redis.hget(_redis_key(digest), "name")
    if name is None:
        return None
    return ApiKey(key_id=digest[:16], name=name)


async def _cli() -> None:
    parser = argparse.ArgumentParser(prog="python -m app.keys")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create").add_argument("--name", required=True)
    sub.add_parser("list")
    sub.add_parser("revoke").add_argument("key_id")
    args = parser.parse_args()

    redis = Redis.from_url(get_settings().redis_url, decode_responses=True)
    try:
        if args.cmd == "create":
            raw, key = await create_key(redis, args.name)
            print(f"key_id: {key.key_id}\napi key (shown once, save it now): {raw}")
        elif args.cmd == "list":
            # scan, not KEYS: KEYS blocks redis while it walks every key
            async for k in redis.scan_iter("apikey:*"):
                info = await redis.hgetall(k)
                print(k.removeprefix("apikey:")[:16], info.get("name"), info.get("created_at"))
        elif args.cmd == "revoke":
            async for k in redis.scan_iter(f"apikey:{args.key_id}*"):
                await redis.delete(k)
                print("revoked", args.key_id)
                break
            else:
                print("no key with id", args.key_id)
    finally:
        await redis.aclose()


if __name__ == "__main__":
    asyncio.run(_cli())
