import pytest

from harness.budget import Budget, BudgetExhausted


def test_agent_and_sleep_share_one_total():
    b = Budget(1000)
    b.charge("agent", 300, 100)
    b.charge("sleep", 200, 100)
    assert (b.agent_tokens, b.sleep_tokens, b.used, b.remaining) == (400, 300, 700, 300)
    b.charge("sleep", 200, 100)  # exactly at the limit: allowed
    assert b.used == 1000 and b.exhausted
    with pytest.raises(BudgetExhausted):
        b.check()


def test_sleep_tokens_alone_can_exhaust_the_agent():
    b = Budget(500)
    b.charge("sleep", 400, 100)
    with pytest.raises(BudgetExhausted):
        b.clamp_completion(10)  # no agent call possible


def test_overshooting_call_is_recorded_and_raises():
    b = Budget(1000)
    b.charge("agent", 900, 0)
    with pytest.raises(BudgetExhausted):
        b.charge("sleep", 150, 50)
    assert b.used == 1100 and b.overshoot == 100 and b.discarded_calls == 1


def test_completion_is_clamped_to_remaining():
    b = Budget(1000)
    b.charge("agent", 950, 0)
    assert b.clamp_completion(4096) == 50


def test_client_charges_budget_by_role(make_llm):
    llm, budget, _ = make_llm(1000, prompt_tokens=60, completion_tokens=40)
    llm.complete([{"role": "user", "content": "hi"}], role="agent", max_tokens=10)
    llm.complete([{"role": "system", "content": "consolidate"}], role="sleep", max_tokens=10)
    assert (budget.agent_tokens, budget.sleep_tokens) == (100, 100)
    assert (budget.agent_calls, budget.sleep_calls) == (1, 1)


def test_client_refuses_call_when_exhausted_and_withholds_overshooting_response(make_llm):
    llm, budget, transport = make_llm(150, prompt_tokens=60, completion_tokens=40)
    llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    with pytest.raises(BudgetExhausted):  # 200 > 150: charged, response withheld
        llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    n = len(transport.calls)
    with pytest.raises(BudgetExhausted):  # nothing left: transport never called
        llm.complete([{"role": "user", "content": "x"}], role="sleep", max_tokens=10)
    assert len(transport.calls) == n
