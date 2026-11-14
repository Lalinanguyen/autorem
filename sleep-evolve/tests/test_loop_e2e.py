"""Mock evaluator + mock LLM, end to end through the real loop, subprocess sandbox, and logger."""

import json
import time

from evaluators.mock import MockEvaluator
from harness.logging import FIELDS, read_history
from harness.loop import run
from sleep.no_sleep import NoSleep
from sleep.summarize import Summarize

GOOD = "```python\ndef construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.6]\n```"
HANG = "```python\ndef construct():\n    while True:\n        pass\n```"


def test_no_sleep_stops_exactly_at_budget(tmp_path, make_llm, mock_cfg):
    llm, budget, transport = make_llm(1000, prompt_tokens=60, completion_tokens=40)
    s = run(mock_cfg(), MockEvaluator(), llm, NoSleep(), out_dir=tmp_path)

    assert s["stop_reason"] == "budget_exhausted"
    assert budget.used == 1000 and budget.overshoot == 0
    assert s["steps_completed"] == 10 and len(transport.calls) == 10

    hist = read_history(tmp_path / s["run_id"] / "history.jsonl")
    assert [h["step"] for h in hist] == list(range(11))  # seed + 10 agent steps
    assert all(set(h) == set(FIELDS) for h in hist)
    assert hist[-1]["agent_tokens"] == 1000 and hist[-1]["sleep_tokens"] == 0
    assert hist[0]["parent_id"] is None and hist[1]["parent_id"] == "p0"
    assert (tmp_path / s["run_id"] / "best_program.py").exists()
    assert json.loads((tmp_path / s["run_id"] / "summary.json").read_text())["best_score"] == s["best_score"]


def test_summarize_charges_sleep_to_the_same_budget(tmp_path, make_llm, mock_cfg):
    llm, budget, _ = make_llm(1000, prompt_tokens=60, completion_tokens=40)
    s = run(mock_cfg(), MockEvaluator(), llm, Summarize(every_n=3), out_dir=tmp_path)

    assert budget.used == 1000 and budget.overshoot == 0
    # 10 calls total; sleeps after steps 3 and 6 -> 8 agent steps + 2 sleeps.
    assert (budget.agent_calls, budget.sleep_calls) == (8, 2)
    assert s["budget"]["agent_tokens"] + s["budget"]["sleep_tokens"] == 1000

    hist = read_history(tmp_path / s["run_id"] / "history.jsonl")
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert [h["step"] for h in sleeps] == [3, 6]
    assert sleeps[0]["details"]["block"].startswith("Consolidated notes")
    assert hist[-1]["sleep_tokens"] == 200


def test_block_reaches_agent_but_raw_history_does_not(tmp_path, make_llm, mock_cfg):
    llm, _, transport = make_llm(1000)
    run(mock_cfg(k_recent=2), MockEvaluator(), llm, Summarize(every_n=3), out_dir=tmp_path)
    agent_msgs = [m for m in transport.calls if "consolidat" not in m[0]["content"].lower()]
    after_sleep = agent_msgs[3][1]["content"]
    assert "Consolidated notes" in after_sleep
    assert after_sleep.count("step ") <= 2  # only the last K=2 steps are rendered


def test_max_steps_stops_before_budget(tmp_path, make_llm, mock_cfg):
    llm, budget, _ = make_llm(10_000)
    s = run(mock_cfg(max_steps=4), MockEvaluator(), llm, NoSleep(), out_dir=tmp_path)
    assert s["stop_reason"] == "max_steps" and s["steps_completed"] == 4 and budget.used == 400


def test_hanging_program_is_killed_and_logged_invalid(tmp_path, make_llm, mock_cfg):
    llm, _, _ = make_llm(10_000, script=[HANG, GOOD])
    t0 = time.monotonic()
    s = run(mock_cfg(max_steps=2), MockEvaluator(timeout_s=2), llm, NoSleep(), out_dir=tmp_path)
    assert time.monotonic() - t0 < 30

    hist = read_history(tmp_path / s["run_id"] / "history.jsonl")
    hung, good = hist[1], hist[2]
    assert hung["valid"] is False and "timeout" in hung["error"] and hung["score"] is None
    assert good["valid"] is True  # the loop carried on
    assert s["steps_completed"] == 2


def test_crashing_and_codeless_responses_are_invalid(tmp_path, make_llm, mock_cfg):
    crash = "```python\ndef construct():\n    raise RuntimeError('boom')\n```"
    llm, _, _ = make_llm(10_000, script=[crash, "no code here", GOOD])
    s = run(mock_cfg(max_steps=3), MockEvaluator(), llm, NoSleep(), out_dir=tmp_path)
    hist = read_history(tmp_path / s["run_id"] / "history.jsonl")
    assert "boom" in hist[1]["error"] and "code block" in hist[2]["error"]
    assert hist[3]["valid"] and s["best_program_id"] == "p3"
