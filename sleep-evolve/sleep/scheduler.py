"""NREM vs REM sleep mode.

NREM (default): stabilize recent evidence. Full replay budget, verification-oriented.
REM: the search has plateaued, so look for cross-cutting abstractions and new directions:
     reduced replay budget (rem_replay_budget), the prompt asks for generalizations, and the
     optional schema store is updated (REM only).

Rule (deterministic):
  1. budget fraction used >= late_budget_fraction  -> NREM (late in the run, consolidate what works)
  2. steps since the best improved >= plateau_window -> REM
  3. otherwise                                       -> NREM
With the scheduler disabled the mode is always NREM.
"""

from __future__ import annotations

from sleep.base import step_events
from sleep.pressure import steps_since_improvement


def choose_mode(history: list[dict], step: int, budget=None, *, enabled: bool = True,
                plateau_window: int = 15, late_budget_fraction: float = 0.8) -> tuple[str, str]:
    if not enabled:
        return "nrem", "scheduler_off"
    if budget is not None and getattr(budget, "total_tokens", 0):
        frac = budget.used / budget.total_tokens
        if frac >= late_budget_fraction:
            return "nrem", f"late_budget({frac:.2f})"
    stall = steps_since_improvement(step_events(history), step)
    if stall >= plateau_window:
        return "rem", f"plateau({stall})"
    return "nrem", "progress"
