"""Baseline: never consolidate. The agent sees task + last K steps (+ parent program)."""

from __future__ import annotations


class NoSleep:
    name = "no_sleep"

    def should_sleep(self, step: int, history: list[dict]) -> bool:
        return False

    def sleep(self, task, block, history, llm, budget) -> str:
        return block
