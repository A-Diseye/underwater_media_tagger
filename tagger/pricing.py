"""Claude API prices, to report what each agent call costs. Shared by the curation agent and
the library search agent."""

from __future__ import annotations

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"

# Dollars per million tokens: (input, output, cache write, cache read).
PRICES = {
    HAIKU: (0.10, 0.50, 0.125, 0.01),
    SONNET: (2.00, 10.00, 2.50, 0.10),
}


def cost_usd(model: str, usage) -> float:
    input_price, output_price, write_price, read_price = PRICES[model]
    tokens = (
        usage.input_tokens * input_price
        + usage.output_tokens * output_price
        + (getattr(usage, "cache_creation_input_tokens", 0) or 0) * write_price
        + (getattr(usage, "cache_read_input_tokens", 0) or 0) * read_price
    )
    return tokens / 1_000_000
