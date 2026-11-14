"""SleepPolicy contract plus helpers for rendering history.

`history` is the in-memory list of log records (the same dicts written to the JSONL
file, see harness/logging.py), including earlier "sleep" events. A policy returns
the NEW consolidated block, which replaces the old one in the agent's context. Every
LLM call a policy makes must use `llm.complete(..., role="sleep")` so it is charged
to the shared Budget.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from harness.budget import Budget
    from llm.client import LLMClient


@runtime_checkable
class SleepPolicy(Protocol):
    name: str

    def should_sleep(self, step: int, history: list[dict]) -> bool: ...

    def sleep(self, task: str, block: str, history: list[dict], llm: "LLMClient", budget: "Budget") -> str:
        """Return the new consolidated block."""
        ...


def step_events(history: list[dict]) -> list[dict]:
    return [h for h in history if h.get("event", "step") == "step"]


def steps_since_last_sleep(history: list[dict]) -> list[dict]:
    last = max((i for i, h in enumerate(history) if h.get("event") == "sleep"), default=-1)
    return step_events(history[last + 1 :])


def render_step(h: dict, max_error_chars: int = 300) -> str:
    if h["valid"]:
        delta = h["delta_vs_best"]
        d = f", delta_vs_best={delta:+.6g}" if delta is not None else ""
        outcome = f"score={h['score']:.6g}{d}"
    else:
        outcome = f"INVALID: {(h['error'] or '')[:max_error_chars]}"
    return f"step {h['step']} | {h['program_id']} (parent {h['parent_id']}) | {outcome}"


def render_steps(history: list[dict]) -> str:
    return "\n".join(render_step(h) for h in step_events(history)) or "(none)"
