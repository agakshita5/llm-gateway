"""Usage log in postgres: one row per completed request."""

import logging
from decimal import Decimal
from dataclasses import dataclass, field
from datetime import datetime, timezone

import asyncpg

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_log (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    key_id            TEXT        NOT NULL,
    provider          TEXT        NOT NULL,
    model             TEXT        NOT NULL,
    stream            BOOLEAN     NOT NULL,
    cached            BOOLEAN     NOT NULL,
    status            INT         NOT NULL,
    prompt_tokens     INT         NOT NULL DEFAULT 0,
    completion_tokens INT         NOT NULL DEFAULT 0,
    total_tokens      INT         NOT NULL DEFAULT 0,
    latency_ms        INT         NOT NULL,
    cost_usd          NUMERIC(14, 8) NOT NULL DEFAULT 0,
    error             TEXT
);
-- GET /usage filters by key and time range, so index exactly that
CREATE INDEX IF NOT EXISTS usage_log_key_time ON usage_log (key_id, created_at);
"""


@dataclass
class UsageRecord:
    key_id: str
    provider: str
    model: str
    stream: bool
    cached: bool
    status: int
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class UsageStore:
    """Interface so tests can use an in-memory store instead of postgres."""

    async def record(self, rec: UsageRecord) -> None: ...

    async def summary(self, key_id: str, since: datetime) -> dict: ...

    async def aclose(self) -> None: ...


class PostgresUsageStore(UsageStore):
    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @classmethod
    async def connect(cls, dsn: str) -> "PostgresUsageStore":
        # a pool keeps a few connections open; opening one per query costs a tcp + auth handshake
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=10)
        async with pool.acquire() as conn:
            await conn.execute(SCHEMA)
        return cls(pool)

    async def record(self, rec: UsageRecord) -> None:
        try:
            # $1..$n placeholders: asyncpg sends values separately from the sql, so no sql injection
            await self._pool.execute(
                """INSERT INTO usage_log (created_at, key_id, provider, model, stream, cached, status, prompt_tokens,
                                          completion_tokens, total_tokens, latency_ms, cost_usd, error)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)""",
                rec.created_at, rec.key_id, rec.provider, rec.model, rec.stream, rec.cached, rec.status,
                rec.prompt_tokens, rec.completion_tokens, rec.total_tokens, rec.latency_ms, rec.cost_usd, rec.error,
            )
        except (asyncpg.PostgresError, OSError):
            # never fail a request because logging failed
            log.exception("failed to write usage row")

    async def summary(self, key_id: str, since: datetime) -> dict:
        totals = await self._pool.fetchrow(
            """SELECT count(*) AS requests,
                      count(*) FILTER (WHERE cached) AS cache_hits,
                      count(*) FILTER (WHERE status >= 400) AS errors,
                      coalesce(sum(prompt_tokens), 0) AS prompt_tokens,
                      coalesce(sum(completion_tokens), 0) AS completion_tokens,
                      coalesce(sum(total_tokens), 0) AS total_tokens,
                      coalesce(sum(cost_usd), 0) AS cost_usd,
                      percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
                      percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
               FROM usage_log WHERE key_id = $1 AND created_at >= $2""",
            key_id, since,
        )
        by_model = await self._pool.fetch(
            """SELECT provider, model, count(*) AS requests, coalesce(sum(total_tokens), 0) AS total_tokens,
                      coalesce(sum(cost_usd), 0) AS cost_usd
               FROM usage_log WHERE key_id = $1 AND created_at >= $2
               GROUP BY provider, model ORDER BY requests DESC""",
            key_id, since,
        )
        return _summary_dict(dict(totals), [dict(r) for r in by_model])

    async def aclose(self) -> None:
        await self._pool.close()


class InMemoryUsageStore(UsageStore):
    """For tests: same interface, rows kept in a list."""

    def __init__(self):
        self.rows: list[UsageRecord] = []

    async def record(self, rec: UsageRecord) -> None:
        self.rows.append(rec)

    async def summary(self, key_id: str, since: datetime) -> dict:
        rows = [r for r in self.rows if r.key_id == key_id and r.created_at >= since]
        latencies = sorted(r.latency_ms for r in rows)
        totals = {
            "requests": len(rows),
            "cache_hits": sum(r.cached for r in rows),
            "errors": sum(r.status >= 400 for r in rows),
            "prompt_tokens": sum(r.prompt_tokens for r in rows),
            "completion_tokens": sum(r.completion_tokens for r in rows),
            "total_tokens": sum(r.total_tokens for r in rows),
            "cost_usd": sum(r.cost_usd for r in rows),
            "p50_latency_ms": latencies[len(latencies) // 2] if latencies else None,
            "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
        }
        groups: dict[tuple[str, str], dict] = {}
        for r in rows:
            g = groups.setdefault((r.provider, r.model), {"provider": r.provider, "model": r.model, "requests": 0, "total_tokens": 0, "cost_usd": 0.0})
            g["requests"] += 1
            g["total_tokens"] += r.total_tokens
            g["cost_usd"] += r.cost_usd
        return _summary_dict(totals, sorted(groups.values(), key=lambda g: -g["requests"]))


def _summary_dict(totals: dict, by_model: list[dict]) -> dict:
    # postgres returns Decimal for sums; make everything plain json numbers
    def num(v):
        return float(v) if isinstance(v, Decimal) else v

    return {**{k: num(v) for k, v in totals.items()}, "by_model": [{k: num(v) for k, v in g.items()} for g in by_model]}
