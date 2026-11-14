"""The matched-budget control: one token total shared by agent and sleep calls.

Rules (LLMClient enforces them on every call, and it is the only code path to the API):
  1. Before a call: `check()` raises BudgetExhausted if nothing remains, and
     `clamp_completion()` caps max_tokens at the remaining budget.
  2. After a call: `charge()` records the API-reported usage under its role. If the
     total now exceeds the budget, the call is still recorded (the tokens were spent),
     `overshoot` grows, and BudgetExhausted is raised so the caller never sees the
     response. Overshoot is therefore at most one call's prompt tokens, and no
     response produced past the limit can influence a run.

Usage breakdown (logging and cost only; never part of the charge): `charge()` also takes the
API-reported `cached_tokens` (a subset of prompt_tokens) and `reasoning_tokens` (a subset of
completion_tokens). Every charged call, including a discarded overshooting one, is appended to
`calls` (role, prompt, completion, cached, reasoning); `usage_breakdown()` aggregates them per
role, and `usage_since(i)` sums calls[i:] for the per-line JSONL fields (harness/logging.py
USAGE_FIELDS).
"""

from __future__ import annotations

import threading
from typing import Literal

Role = Literal["agent", "sleep"]
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens")


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, total_tokens: int):
        if total_tokens <= 0:
            raise ValueError("total_tokens must be positive")
        self.total_tokens = int(total_tokens)
        self.agent_tokens = 0
        self.sleep_tokens = 0
        self.agent_calls = 0
        self.sleep_calls = 0
        self.discarded_calls = 0  # calls charged but whose response was withheld
        self.calls: list[dict] = []  # per-call usage ledger (see the module docstring)
        self._lock = threading.Lock()  # background sleep (task 7) will charge from another thread

    @property
    def used(self) -> int:
        return self.agent_tokens + self.sleep_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.total_tokens - self.used)

    @property
    def overshoot(self) -> int:
        return max(0, self.used - self.total_tokens)

    @property
    def exhausted(self) -> bool:
        return self.used >= self.total_tokens

    def check(self) -> None:
        if self.exhausted:
            raise BudgetExhausted(f"budget exhausted: used {self.used}/{self.total_tokens}")

    def clamp_completion(self, max_tokens: int) -> int:
        self.check()
        return max(1, min(int(max_tokens), self.remaining))

    def charge(self, role: Role, prompt_tokens: int, completion_tokens: int, *,
               cached_tokens: int | None = None, reasoning_tokens: int | None = None) -> None:
        if role not in ("agent", "sleep"):
            raise ValueError(f"unknown role {role!r}")
        if int(prompt_tokens) < 0 or int(completion_tokens) < 0:
            raise ValueError("token counts must be non-negative")
        n = int(prompt_tokens) + int(completion_tokens)
        with self._lock:
            self.calls.append({
                "role": role, "prompt_tokens": int(prompt_tokens), "completion_tokens": int(completion_tokens),
                "cached_tokens": None if cached_tokens is None else int(cached_tokens),
                "reasoning_tokens": None if reasoning_tokens is None else int(reasoning_tokens),
            })
            if role == "agent":
                self.agent_tokens += n
                self.agent_calls += 1
            else:
                self.sleep_tokens += n
                self.sleep_calls += 1
            if self.used > self.total_tokens:
                self.discarded_calls += 1
                raise BudgetExhausted(
                    f"{role} call pushed usage to {self.used}/{self.total_tokens}; response discarded"
                )

    def usage_since(self, start: int = 0) -> dict:
        """Sums over calls[start:]: {"calls", prompt/completion/cached/reasoning tokens}.

        cached / reasoning are None when no call in the range reported them; all token fields are
        None when the range is empty."""
        with self._lock:
            calls = self.calls[start:]
        return _aggregate(calls)

    def usage_breakdown(self) -> dict:
        """{"agent": agg, "sleep": agg, "total": agg} over every charged call (discarded ones included)."""
        with self._lock:
            calls = list(self.calls)
        return {"agent": _aggregate([c for c in calls if c["role"] == "agent"]),
                "sleep": _aggregate([c for c in calls if c["role"] == "sleep"]),
                "total": _aggregate(calls)}

    def snapshot(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "used": self.used,
            "agent_tokens": self.agent_tokens,
            "sleep_tokens": self.sleep_tokens,
            "agent_calls": self.agent_calls,
            "sleep_calls": self.sleep_calls,
            "overshoot": self.overshoot,
            "discarded_calls": self.discarded_calls,
        }


def _aggregate(calls: list[dict]) -> dict:
    out: dict = {"calls": len(calls)}
    for k in USAGE_KEYS:
        vals = [c[k] for c in calls if c.get(k) is not None]
        out[k] = sum(vals) if vals else (0 if calls and k in ("prompt_tokens", "completion_tokens") else None)
    return out
