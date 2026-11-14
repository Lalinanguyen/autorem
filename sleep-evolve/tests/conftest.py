import pytest

from harness.budget import Budget
from harness.config import RunConfig
from llm.client import LLMClient
from llm.mock import MockTransport


@pytest.fixture
def make_llm():
    def _make(total: int, **mock_kwargs):
        budget = Budget(total)
        transport = MockTransport(**mock_kwargs)
        return LLMClient(budget, transport, sleep_fn=lambda s: None), budget, transport

    return _make


@pytest.fixture
def mock_cfg():
    def _make(**kw):
        base = dict(problem_id=0, total_token_budget=1000, max_steps=1000, k_recent=3)
        base.update(kw)
        return RunConfig(**base)

    return _make
