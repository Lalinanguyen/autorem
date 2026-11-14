"""The matched-budget control: one token total shared by agent and sleep calls.

Rules (LLMClient enforces them on every call, and it is the only code path to the API):
  1. Before a call: `check()` raises BudgetExhausted if nothing remains, and
     `clamp_completion()` caps max_tokens at the remaining budget.
  2. After a call: `charge()` records the API-reported usage under its role. If the
     total now exceeds the budget, the call is still recorded (the tokens were spent),
     `overshoot` grows, and BudgetExhausted is raised so the caller never sees the
     response. Overshoot is therefore at most one call's prompt tokens, and no
     response produced past the limit can influence a run.
"""

from __future__ import annotations

import threading
from typing import Literal

Role = Literal["agent", "sleep"]


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

    def charge(self, role: Role, prompt_tokens: int, completion_tokens: int) -> None:
        if role not in ("agent", "sleep"):
            raise ValueError(f"unknown role {role!r}")
        n = int(prompt_tokens) + int(completion_tokens)
        if n < 0:
            raise ValueError("token counts must be non-negative")
        with self._lock:
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
