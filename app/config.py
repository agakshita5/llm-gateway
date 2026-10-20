from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Loaded from environment variables, falling back to .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # providers
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_fallback_model: str = "openai/gpt-oss-20b"
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_fallback_model: str = "gemini-2.5-flash"
    upstream_timeout_s: float = 30.0
    # "real" calls groq/gemini; "fake" uses an in-process stub (load tests, local dev without keys)
    provider_mode: str = "real"
    fake_latency_ms: int = 200

    # retry / fallback (see DECISIONS.md 010)
    max_attempts_per_provider: int = 2
    backoff_base_s: float = 0.5
    backoff_max_s: float = 4.0

    # storage
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql://localhost:5432/llm_gateway"

    # rate limit: token bucket per api key (see DECISIONS.md 013)
    rate_limit_capacity: int = 20
    rate_limit_refill_per_s: float = 1.0

    # cache (see DECISIONS.md 014)
    cache_ttl_s: int = 3600


@lru_cache
def get_settings() -> Settings:
    return Settings()
