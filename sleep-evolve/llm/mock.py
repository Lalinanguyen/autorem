"""Deterministic mock transport for tests and dry runs. Plug it into LLMClient like a real one.

Every call reports fixed usage (prompt_tokens + completion_tokens), so budget arithmetic
in tests is exact. Responses come from `script` (a list consumed in order, then
cycled) or, by default, from a generator that writes a random program for the mock
evaluator (agent role) or a canned summary (sleep role).
"""

from __future__ import annotations

import random

from llm.client import LLMResponse, Message


class MockTransport:
    def __init__(
        self,
        prompt_tokens: int = 60,
        completion_tokens: int = 40,
        seed: int = 0,
        script: list[str] | None = None,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.script = list(script or [])
        self.calls: list[list[Message]] = []
        self._rng = random.Random(seed)

    def __call__(self, messages: list[Message], max_tokens: int) -> LLMResponse:
        self.calls.append(messages)
        is_sleep = "consolidat" in messages[0].get("content", "").lower()
        if self.script:
            text = self.script[(len(self.calls) - 1) % len(self.script)]
        elif is_sleep:
            text = f"Consolidated notes after {len(self.calls)} calls: values near 0.7 score best."
        else:
            xs = [round(self._rng.uniform(0.0, 1.0), 4) for _ in range(5)]
            text = f"Trying new values.\n```python\ndef construct():\n    return {xs}\n```"
        return LLMResponse(text, self.prompt_tokens, self.completion_tokens)
