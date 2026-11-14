"""Dollar cost of a run from its logged usage, and the pre-flight estimate for a matrix.

Prices are USD per 1M tokens: input, cached input, output. Table = DEFAULT_PRICES overlaid by the
config's `prices:` section (RunConfig.prices). A price that is not confirmed is null: cost
reporting then says so (`complete: false`) and the matrix pre-flight refuses unless the user
passes an explicit override (scripts/run_matrix.py --price-per-1m).

Cost of a set of calls (OpenAI usage semantics, llm/chat_params.py):
    cached prompt tokens are a SUBSET of prompt_tokens; reasoning tokens a SUBSET of completion_tokens
    cost = (prompt - cached) * input + cached * cached_input + completion * output     (/ 1e6)
Reasoning tokens are billed as output and are already inside completion_tokens: never added twice.
If cached_input is null but cached tokens were reported, they are charged at the input price
(an upper bound) and the block says so.

Pre-flight (conservative upper bound, no API calls):
    per_run_tokens = total_token_budget + overshoot allowance
    overshoot allowance = max_completion_tokens + prompt allowance (2 * context_cap_tokens, default
                          16000; the Budget's overshoot is at most one call)
    rate = max(input, cached_input, output) per token  (every token billed at the dearest rate)
    estimate = runs * per_run_tokens * rate
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

PRICES_CHECKED = "2026-11-14"
_OPENAI_PRICING = "https://developers.openai.com/api/docs/pricing (Standard tier, text tokens)"


class ModelPrice(BaseModel):
    """USD per 1M tokens. null = not confirmed (cost reported as unknown; pre-flight refuses)."""

    model_config = {"extra": "forbid"}

    input: float | None = Field(None, ge=0)
    cached_input: float | None = Field(None, ge=0)
    output: float | None = Field(None, ge=0)
    source: str | None = None
    checked: str | None = Field(None, description="date the price was checked (YYYY-MM-DD)")
    note: str | None = None

    @property
    def known(self) -> bool:
        return self.input is not None and self.output is not None


# Checked on PRICES_CHECKED against the official pricing page and the model pages
# (developers.openai.com/api/docs/models/gpt-5-mini, .../gpt-5-nano). Both models are marked
# Deprecated there (notice 2026-06-11, shutdown 2026-12-11: developers.openai.com/api/docs/deprecations).
DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gpt-5-nano": ModelPrice(input=0.05, cached_input=0.005, output=0.40, source=_OPENAI_PRICING,
                             checked=PRICES_CHECKED, note="deprecated; shutdown 2026-12-11"),
    "gpt-5-mini": ModelPrice(input=0.25, cached_input=0.025, output=2.00, source=_OPENAI_PRICING,
                             checked=PRICES_CHECKED, note="deprecated; shutdown 2026-12-11"),
    "gpt-5": ModelPrice(input=1.25, cached_input=0.125, output=10.00, source=_OPENAI_PRICING,
                        checked=PRICES_CHECKED, note="deprecated; shutdown 2026-12-11"),
}

_SNAPSHOT = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def price_table(overrides: dict | None = None) -> dict[str, ModelPrice]:
    table = dict(DEFAULT_PRICES)
    for k, v in (overrides or {}).items():
        table[k] = v if isinstance(v, ModelPrice) else ModelPrice(**(v or {}))
    return table


def lookup_price(model: str | None, table: dict[str, ModelPrice]) -> tuple[str | None, ModelPrice | None]:
    """(table key, price) for `model`: exact, then without a provider prefix ("openai/"), then
    without a snapshot date suffix (gpt-5-mini-2025-08-07 -> gpt-5-mini)."""
    if not model:
        return None, None
    cands = [model]
    if "/" in model:
        cands.append(model.split("/", 1)[1])
    cands += [_SNAPSHOT.sub("", c) for c in list(cands)]
    for c in cands:
        if c in table:
            return c, table[c]
    return None, None


def usage_cost(prompt_tokens: int | None, completion_tokens: int | None, cached_tokens: int | None,
               price: ModelPrice | None) -> float | None:
    """USD for the given usage totals (see the module docstring); None if the price is unknown."""
    if price is None or not price.known:
        return None
    p, c = int(prompt_tokens or 0), int(completion_tokens or 0)
    cached = min(int(cached_tokens or 0), p)
    cached_rate = price.cached_input if price.cached_input is not None else price.input
    return ((p - cached) * price.input + cached * cached_rate + c * price.output) / 1e6


def cost_block(usage: dict, model: str | None, table: dict[str, ModelPrice]) -> dict:
    """summary.json["cost"] from Budget.usage_breakdown() (or the same shape rebuilt offline)."""
    key, price = lookup_price(model, table)
    notes = []
    if price is None:
        notes.append(f"no price for model {model!r} in the price table")
    elif not price.known:
        notes.append(f"price for {key!r} is not confirmed (null)")
    by_role = {}
    for role in ("agent", "sleep"):
        u = usage.get(role) or {}
        by_role[role] = usage_cost(u.get("prompt_tokens"), u.get("completion_tokens"), u.get("cached_tokens"), price)
    tot = usage.get("total") or {}
    total = usage_cost(tot.get("prompt_tokens"), tot.get("completion_tokens"), tot.get("cached_tokens"), price)
    if price is not None and price.known and price.cached_input is None and (tot.get("cached_tokens") or 0) > 0:
        notes.append("cached_input price unknown: cached tokens charged at the input price")
    return {
        "currency": "USD",
        "model": model,
        "price_key": key,
        "price_per_1m": price.model_dump() if price is not None else None,
        "price_source": price.source if price is not None else None,
        "by_role": {r: (None if v is None else round(v, 6)) for r, v in by_role.items()},
        "total": None if total is None else round(total, 6),
        "complete": total is not None,
        "formula": "((prompt - cached) * input + cached * cached_input + completion * output) / 1e6",
        "notes": notes,
    }


def conservative_rate_per_1m(price: ModelPrice | None) -> float | None:
    if price is None or not price.known:
        return None
    return max(x for x in (price.input, price.cached_input, price.output) if x is not None)


def overshoot_allowance(cfg: Any) -> dict:
    prompt = 2 * int(cfg.context_cap_tokens) if getattr(cfg, "context_cap_tokens", None) else 16000
    completion = int(getattr(cfg, "max_completion_tokens", 0) or 0)
    return {"max_completion_tokens": completion, "prompt_allowance": prompt, "total": completion + prompt}


def preflight_estimate(cfg: Any, n_runs: int, table: dict[str, ModelPrice], *,
                       model: str | None = None, rate_override_per_1m: float | None = None) -> dict:
    """Conservative dollar estimate for `n_runs` runs of `cfg` (module docstring). No API calls.

    `cost` is None when no price is known and no override is given."""
    model = model or getattr(cfg, "model", None)
    key, price = lookup_price(model, table)
    allow = overshoot_allowance(cfg)
    per_run = int(cfg.total_token_budget) + allow["total"]
    if rate_override_per_1m is not None:
        rate, rate_source = float(rate_override_per_1m), "override (--price-per-1m)"
    else:
        rate = conservative_rate_per_1m(price)
        rate_source = (f"max(input, cached_input, output) of {key!r}" if rate is not None else None)
    cost = None if rate is None else n_runs * per_run * rate / 1e6
    return {
        "model": model, "price_key": key, "price_per_1m": price.model_dump() if price is not None else None,
        "runs": int(n_runs), "total_token_budget": int(cfg.total_token_budget), "overshoot_allowance": allow,
        "per_run_tokens": per_run, "total_tokens": n_runs * per_run,
        "rate_per_1m": rate, "rate_source": rate_source, "cost": None if cost is None else round(cost, 4),
    }


__all__ = ["DEFAULT_PRICES", "ModelPrice", "conservative_rate_per_1m", "cost_block", "lookup_price",
           "overshoot_allowance", "preflight_estimate", "price_table", "usage_cost"]
