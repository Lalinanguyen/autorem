"""Evaluator-call accounting: a shared EvalBudget and the CountingEvaluator wrapper.

Replays are evaluator calls too: sleep arms get extra sandbox feedback the other arms do
not, unless the calls are counted and capped. Every evaluate() in a run (agent steps,
replays, GEPA sandbox evals) goes through one CountingEvaluator bound to one EvalBudget,
so `max_evaluations` is a hard, shared cap — the eval-matched counterpart of the token
Budget.

Semantics (deliberately stricter than the token Budget, which tolerates one-call overshoot):
  1. Before a call: check() raises EvalExhausted when the limit is already reached, so
     `used` never exceeds `limit`.
  2. After a call: charge() records it under a role ("agent" | "sleep") for reporting.
EvalExhausted propagates like BudgetExhausted: out of evaluate(), through the sleep
pipeline, and the loop stops the run with stop_reason "eval_limit".
"""

from __future__ import annotations

import threading
from typing import Any


class EvalExhausted(Exception):
    pass


class EvalBudget:
    def __init__(self, limit: int | None = None):
        if limit is not None and limit <= 0:
            raise ValueError("eval limit must be positive or None")
        self.limit = None if limit is None else int(limit)
        self.agent_evals = 0
        self.sleep_evals = 0
        self._lock = threading.Lock()

    @property
    def used(self) -> int:
        return self.agent_evals + self.sleep_evals

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)

    @property
    def exhausted(self) -> bool:
        return self.limit is not None and self.used >= self.limit

    def check(self) -> None:
        if self.exhausted:
            raise EvalExhausted(f"eval budget exhausted: used {self.used}/{self.limit}")

    def charge(self, role: str = "agent") -> None:
        with self._lock:
            if role == "sleep":
                self.sleep_evals += 1
            else:
                self.agent_evals += 1

    def snapshot(self) -> dict:
        return {"used": self.used, "limit": self.limit, "remaining": self.remaining,
                "agent_evals": self.agent_evals, "sleep_evals": self.sleep_evals}


class CountingEvaluator:
    """Wraps an Evaluator; every evaluate() is checked against and charged to an EvalBudget.

    All other attribute access (problem_id, name, best_known, timeout_s, description,
    seed_program, ...) is delegated, so the wrapper satisfies the Evaluator protocol.
    tagged(role) returns a sibling wrapper on the same budget for the sleep path.
    """

    def __init__(self, evaluator: Any, budget: EvalBudget, role: str = "agent"):
        object.__setattr__(self, "_evaluator", evaluator)
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(self, "_role", role)

    def tagged(self, role: str) -> "CountingEvaluator":
        return CountingEvaluator(self._evaluator, self._budget, role)

    @property
    def budget(self) -> EvalBudget:
        return self._budget

    @property
    def inner(self):
        """The wrapped evaluator (before counting)."""
        return self._evaluator

    def evaluate(self, program: str):
        self._budget.check()
        res = self._evaluator.evaluate(program)
        self._budget.charge(self._role)
        return res

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_evaluator"), name)

    def __setattr__(self, name: str, value):
        setattr(self._evaluator, name, value)
