# LLM Gateway — learning project

## Who I am
Final-year B.Tech CSE student (no internship yet), targeting SDE + AI roles. I've built full-stack AI apps
(Next.js, FastAPI, Supabase, Groq, LiveKit) with heavy AI assistance. Goal of this project: learn backend and
systems fundamentals deeply enough to defend every line in an interview — not just ship it.

## What we're building (one week)
A single OpenAI-compatible API in front of multiple LLM providers. Python + FastAPI.

In scope:
1. OpenAI-compatible `POST /v1/chat/completions` (any OpenAI SDK works by changing base URL)
2. Two providers (Groq + Gemini) behind a common provider interface (adapter pattern);
   retry with exponential backoff, fallback to the other provider on 429 / 5xx / timeout
3. Streaming (SSE) passthrough
4. API keys + per-key rate limiting — token bucket in Redis, atomic via Lua script
5. Exact-match response cache in Redis with TTL
6. Usage logging in Postgres (tokens, latency, provider, est. cost) + `GET /usage`
7. Docker Compose (api + redis + postgres), pytest with mocked providers, Locust/k6 load test with real numbers

Out of scope (do NOT add): semantic cache, batch queues, dashboard UI, Go, OpenTelemetry.

## Day plan
| Day | Work |
|---|---|
| 1 | Skeleton, provider interface, Groq provider, OpenAI-compatible endpoint, streaming |
| 2 | Second provider, retries + fallback, tests with mocked providers |
| 3 | Redis rate limiter + cache, Postgres usage log, Docker Compose (cut cache first if behind) |
| 4 | Load test; README with architecture diagram + numbers; read through every file |
| 5 | I delete the rate limiter + fallback logic and rewrite them from memory |
| 6 | I add one feature solo: circuit breaker OR per-key monthly token budget |
| 7 | Mock interview on this project + short write-up post |

## How to work with me (important)
- Before each component, explain the design options and trade-offs briefly and let ME choose. Record the
  decision and the reason in `DECISIONS.md`.
- I write the core logic (rate limiter, retry/fallback, cache keying). You may write boilerplate
  (project setup, configs, Dockerfile, CI, Pydantic models, test scaffolding).
- When I write code, review it like a senior engineer — point out bugs and gaps, but don't silently rewrite it.
- Explain concepts in plain terms; assume I know Python/FastAPI basics but not Redis, Lua, SSE internals,
  backoff strategies, or load testing.
- No code gets committed that I can't explain. After each component, quiz me with 2–3 interview-style questions.
- Days 5–7: do not give me code. Hints only, unless I explicitly ask.
- Keep scope tight. If I try to add out-of-scope things, remind me.

## Progress log
At the end of every session, update this section: what's done, what's next, open questions.

- 2026-10-05 → 10-09: skeleton, provider interface + exceptions (user-written, reviewed), Groq adapter.
- 2026-10-20: plan switched — Claude writes all remaining code, user approves decisions and reviews.
  Done: Groq stream, endpoint, Gemini adapter (native API), router (retry/backoff/fallback), API keys,
  Lua token-bucket rate limiter, exact-match cache, Postgres usage log + GET /usage, Dockerfile + compose,
  56 tests, Locust load test (fake provider; ~2 ms p50 / 11 ms p99 gateway overhead), README.
  Verified end-to-end with the real OpenAI SDK against real Groq + Gemini, including real Groq→Gemini fallback.
  Decisions 001–016 in DECISIONS.md (gitignored by choice).
- Next: user installs Docker and runs `docker compose up` (not verifiable here — Docker isn't installed);
  user reviews the code; nothing is committed yet. Then the learning wrap-up: answer quesToAddress,
  rewrite rate limiter + fallback from memory, solo feature (circuit breaker or monthly budget), mock interview.
- Open: should CLAUDE.md's "how to work with me" be updated to reflect the switch?
