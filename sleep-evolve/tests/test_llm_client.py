import re
from pathlib import Path

import pytest

from harness.budget import Budget
from llm.client import LLMClient, LLMResponse, TransientLLMError

ROOT = Path(__file__).resolve().parents[1]


class FlakyTransport:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def __call__(self, messages, max_tokens):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientLLMError("HTTP 429")
        return LLMResponse("ok", 10, 5)


def test_retries_transient_errors_with_backoff():
    delays = []
    budget = Budget(1000)
    llm = LLMClient(budget, FlakyTransport(3), sleep_fn=delays.append, backoff_base_s=1.0)
    resp = llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    assert resp.text == "ok" and len(delays) == 3
    assert delays[0] < delays[1] < delays[2]  # exponential (with jitter in [0.5, 1.0])
    assert budget.used == 15  # failed attempts are not charged: the API reported no usage


def test_gives_up_after_max_retries():
    llm = LLMClient(Budget(1000), FlakyTransport(99), max_retries=2, sleep_fn=lambda s: None)
    with pytest.raises(TransientLLMError):
        llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)


def test_requires_budget():
    with pytest.raises(TypeError):
        LLMClient(None, FlakyTransport(0))  # type: ignore[arg-type]


def test_no_module_bypasses_the_client():
    """Only llm/client.py may talk to the API, so every call is charged to the Budget."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or path == ROOT / "llm" / "client.py" or path.parent.name == "tests":
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import openai|from openai\b)", text, re.M) or "chat.completions" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
