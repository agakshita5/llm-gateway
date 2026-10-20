"""Estimated cost per model, in usd per 1M tokens (input, output).

these are list prices copied by hand and will drift; check the provider pricing pages before trusting
the cost numbers. unknown models cost 0 and are flagged in the usage log by cost_usd = 0.
"""

PRICES_PER_1M: dict[str, tuple[float, float]] = {
    # groq
    "openai/gpt-oss-20b": (0.10, 0.50),
    "openai/gpt-oss-120b": (0.15, 0.75),
    # gemini
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-pro": (1.25, 10.00),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price_in, price_out = PRICES_PER_1M.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
