"""SleepLite: the drowsy response only. Forgetting in code, never a sleep LLM call.

Reads the loop's fatigue level (harness/fatigue.py) exactly like FullSleep with gating=fatigue,
but its depth is capped at `forget`: whenever the agent is drowsy or worse it forgets (drops
recent raw steps memory covers, shrinks recent steps to one-line records, strengthens matched dead
ends, refreshes the goal item), at most every `forget_min_gap` steps. It never consolidates, never
replays and never calls the LLM, so all of its budget goes to agent steps. Without a fatigue
reading it never sleeps.
"""

from __future__ import annotations

from typing import Any

from harness.fatigue import SleepDepth
from sleep.full import FullSleep

LITE_FIXED = {"gating": "fatigue", "max_depth": "forget"}
LITE_DEFAULTS = {"replay": False, "schema": False, "pressure": False}


class SleepLite(FullSleep):
    name = "sleep_lite"

    def __init__(self, **params: Any):
        bad = {k: params[k] for k in LITE_FIXED if k in params and params[k] != LITE_FIXED[k]}
        if bad:
            raise ValueError(f"sleep_lite fixes {LITE_FIXED}; got {bad}")
        super().__init__(**{**LITE_DEFAULTS, **params, **LITE_FIXED})

    def should_sleep(self, step: int, history: list[dict]) -> bool:
        self._planned = None
        if step <= 0 or self.fatigue is None:
            return False
        return self._fatigue_gate(step)

    def sleep(self, task: str, block: str, history: list[dict], llm, budget) -> str:
        step = max((h.get("step", 0) for h in history), default=0)
        self._planned = None
        self.forget(history, step=step)  # code only: `llm` is never touched
        return self.memory_section(self.max_block_tokens)

    def consolidate(self, *args, **kwargs) -> dict:  # pragma: no cover - guard
        raise RuntimeError("sleep_lite never consolidates (no sleep LLM calls)")


assert SleepDepth.parse(LITE_FIXED["max_depth"]) == SleepDepth.FORGET
