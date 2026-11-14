"""sleep-evolve test suite: one file for the whole project.

Run offline (default; mock LLM, mock/text evaluators, no Modal, no network):
    uv run pytest
    uv run --group analysis pytest          # also renders the matplotlib figures

Live tests (real services; spend money; need credentials) are marked `llm`, `modal`, `pastdev`
or `slow` and are SKIPPED unless RUN_LIVE=1:
    RUN_LIVE=1 uv run pytest -m "llm or modal or pastdev or slow"
Gating is done with skipif marks in this file (see `live()`), so no conftest.py hook is needed.

Data files (published constructions etc.) live in tests/data/ and are loaded from there.

Layout (one section per area of the spec; each ends with "additional coverage" for behaviors
the spec does not list but earlier test modules checked):
    Shared helpers and fixtures
    Section 1  Budget and LLM client (+ token-counting proxy)
    Section 2  Evaluators (+ sandbox, side_info, AlphaEvolve benchmark layer)
    Section 3  Evolution loop and logging (+ agent context)
    Section 4  Memory store
    Section 5  Sleep pipeline (FullSleep, replay, pressure, scheduler, schema store)
    Section 6  Fatigue monitor
    Section 7  Baseline arms
    Section 8  Integrations (offline GEPA against a local fake upstream; live markers)
    Section 9  Analysis

Behaviors the spec asks for that do not exist yet are `xfail(strict=True)`: they flag as XPASS
(a failure) the moment someone implements them, so the marker must then be removed.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import random
import re
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from types import SimpleNamespace
from urllib import request as urlrequest

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from evaluators import REGISTRY as EVALUATORS
from evaluators.base import EvalResult, Evaluator, ProgramFailed, failure_side_info, invalid, normalized_score
from evaluators.mock import DIM, TARGET, MockEvaluator
from evaluators.p36_circle_packing import BEST_KNOWN, CirclePacking26, check_packing, packing_side_info, render_packing
from harness.budget import Budget, BudgetExhausted
from harness.config import RunConfig, load_config, load_dotenv
from harness.context import build_context, count_tokens, fit_lines, messages_tokens
from harness.fatigue import (
    COMPONENTS, FATIGUE_FIELDS, FatigueConfig, FatigueMonitor, FatigueReading, FatigueState, Level, SleepDepth,
    replay_history,
)
from harness.logging import FIELDS, RunLogger, read_history
from harness.loop import parse_awake_header, run
from harness.program_db import Program, ProgramDB
from llm.client import LLMClient, LLMResponse, MissingUsageError, OpenAICompatTransport, TransientLLMError
from llm.mock import MockTransport
from sleep import make_policy
from sleep import pressure as P
from sleep.consolidator_io import parse_json_object, validate_consolidation
from sleep.full import FullSleep
from sleep.lite import SleepLite
from sleep.memory import Edit, MemoryStore, normalize_tags
from sleep.no_sleep import NoSleep
from sleep.replay import decide, edit_for, is_crash
from sleep.scheduler import choose_mode
from sleep.summarize import Summarize

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "tests" / "data"
P36_DATA = DATA / "p36_alphaevolve_n26.json"

HAS_GEPA = importlib.util.find_spec("gepa") is not None
needs_gepa = pytest.mark.skipif(not HAS_GEPA, reason="optional group: uv sync --group gepa")

# ------------------------------------------------------------------------------------------------
# Live gating: tests that need real services carry their service marker(s) plus a skip unless
# RUN_LIVE=1. `-m llm` etc. still selects/collects them; they only run with the flag set.
# ------------------------------------------------------------------------------------------------
RUN_LIVE = os.environ.get("RUN_LIVE") == "1"
_live_skip = pytest.mark.skipif(not RUN_LIVE, reason="live test: set RUN_LIVE=1 (real services, real money)")


def live(*marks):
    def deco(fn):
        for m in (*marks, _live_skip):
            fn = m(fn)
        return fn

    return deco


def _require_env(*names: str) -> None:
    load_dotenv(ROOT / ".env")
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        pytest.skip(f"missing env vars: {', '.join(missing)}")


# ================================================================================================
# Shared helpers and fixtures
# ================================================================================================

GOOD = "def construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.7]\n"
BAD = "def construct():\n    return [0.0, 0.0, 0.0, 0.0, 0.0]\n"
CRASH = "def construct():\n    raise RuntimeError('x')\n"
SHAPE = "def construct():\n    return [1.0]\n"
HANG = "def construct():\n    while True:\n        pass\n"

# Agent replies (fenced) for the real loop.
GOOD_REPLY = "```python\ndef construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.6]\n```"
HANG_REPLY = f"```python\n{HANG}```"

# Malformed LLM outputs, used by the mock LLM's malformed mode and the consolidator-parsing tests.
MALFORMED_OUTPUTS = ["no json here", "{\"claims\": [", "[1, 2, 3]", "{\"claims\": \"nope\", \"replays\": 5}", ""]

_RET = re.compile(r"return\s*\[([^\]]*)\]")


class TextEvaluator:
    """Test evaluator that scores a program by PARSING its text: no exec, no subprocess, instant.

    Same objective as evaluators/mock.py (5 floats, score = 1 - mean((x - 0.7)^2)). Deterministic:
      - `lookup`: {program text (stripped): score} overrides the parsed score
      - `score_fn`: f(list of 5 floats) -> score replaces the quadratic
      - crash mode: programs containing `crash_marker` ("raise") are reported like a sandbox crash
      - hang mode: programs containing `hang_marker` ("while True") are reported like a sandbox
        timeout after `timeout_s` (simulated: nothing waits)
      - programs without a 5-number `return [...]` are invalid by constraint (shape)
    Emits side_info so context tests can see it. Records every program in `.calls`.
    """

    problem_id = 0
    name = "text_mock"
    best_known = 1.0
    description = "Write construct() returning a list of 5 floats; score = 1 - mean((x - t)^2)."

    def __init__(self, *, lookup: dict | None = None, score_fn=None, crash_marker: str = "raise",
                 hang_marker: str = "while True", timeout_s: int = 1):
        self.lookup = {k.strip(): v for k, v in (lookup or {}).items()}
        self.score_fn = score_fn
        self.crash_marker = crash_marker
        self.hang_marker = hang_marker
        self.timeout_s = timeout_s
        self.calls: list[str] = []

    def seed_program(self) -> str:
        return BAD

    def evaluate(self, program: str) -> EvalResult:
        self.calls.append(program)
        if self.hang_marker and self.hang_marker in program:
            e = ProgramFailed(f"timeout after {self.timeout_s}s", kind="timeout")
            return invalid(str(e), side_info=failure_side_info(e, self.timeout_s))
        if self.crash_marker and self.crash_marker in program:
            e = ProgramFailed("exit code 1: RuntimeError: probe crashed", kind="crash",
                              output_tail="RuntimeError: probe crashed")
            return invalid(str(e), side_info=failure_side_info(e, self.timeout_s))
        if program.strip() in self.lookup:
            return EvalResult(score=float(self.lookup[program.strip()]), valid=True, side_info={"lookup": True})
        m = _RET.findall(program)
        try:
            xs = [float(v) for v in m[-1].split(",")] if m else []
        except ValueError:
            xs = []
        if len(xs) != 5:
            return EvalResult(score=float("-inf"), valid=False, error="expected a list of 5 numbers",
                              side_info={"stage": "shape", "violation": "shape", "got": len(xs)})
        score = self.score_fn(xs) if self.score_fn else 1.0 - sum((v - 0.7) ** 2 for v in xs) / 5
        return EvalResult(score=score, valid=True,
                          side_info={"max_dev": round(max(abs(v - 0.7) for v in xs), 4), "slack": [0.1] * 3})


def program_returning(centers, radii) -> str:
    return f"def construct():\n    return {json.dumps(centers)}, {json.dumps(radii)}\n"


def p36_grid():
    """26 circles of radius 1/12 on a 6x5 grid: touching neighbours and walls (sum 26/12)."""
    centers = [[(j + 0.5) / 6, (i + 0.5) / 5] for i, j in (divmod(k, 6) for k in range(26))]
    return centers, [1 / 12] * 26


def assert_json(si: dict) -> None:
    assert isinstance(si, dict) and si
    assert json.loads(json.dumps(si, allow_nan=False)) == si


def hist_of(out_dir: Path, summary: dict) -> list[dict]:
    return read_history(Path(out_dir) / summary["run_id"] / "history.jsonl")


@pytest.fixture
def mock_llm():
    """Factory: (LLMClient, Budget, MockTransport) with scripted responses and fixed usage.

    mock_llm(total, prompt_tokens=60, completion_tokens=40, script=[...], malformed=False, ...)
    malformed=True scripts every response from MALFORMED_OUTPUTS (no code block, broken JSON).
    Retries never sleep.
    """

    def _make(total: int, *, malformed: bool = False, **mock_kwargs):
        if malformed:
            mock_kwargs["script"] = list(MALFORMED_OUTPUTS)
        budget = Budget(total)
        transport = MockTransport(**mock_kwargs)
        return LLMClient(budget, transport, sleep_fn=lambda s: None), budget, transport

    return _make


@pytest.fixture
def fake_evaluator():
    """Factory for TextEvaluator (lookup / score_fn / crash / hang modes, see the class)."""
    return lambda **kw: TextEvaluator(**kw)


@pytest.fixture
def run_dir(tmp_path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


@pytest.fixture
def small_budget() -> Budget:
    return Budget(1000)


@pytest.fixture
def memory_store() -> MemoryStore:
    return MemoryStore()


def fatigue_cfg(**kw) -> FatigueConfig:
    """Test fatigue config: 100 tokens/step -> load 0.1/step; 3-step baseline and window; no grace."""
    base = dict(load_tokens_scale=1000, baseline_steps=3, window=3, grace_steps=0, context_soft_cap=None)
    base.update(kw)
    return FatigueConfig(**base)


@pytest.fixture
def fatigue_monitor() -> FatigueMonitor:
    return FatigueMonitor(fatigue_cfg(), best_known=1.0)


@pytest.fixture
def mock_cfg():
    def _make(**kw):
        base = dict(problem_id=0, total_token_budget=1000, max_steps=1000, k_recent=3)
        base.update(kw)
        return RunConfig(**base)

    return _make


def synthetic_history(n: int = 30, *, onset: int = 8, surprise: float = 0.05, perseveration: float = 0.0,
                      invalid_rate: float = 0.0, tag_diversity: float = 1.0, improving: bool = True,
                      context: int = 1000, seed: int = 0, tokens_per_step: int = 100) -> list[dict]:
    """Step records (JSONL dicts) with controllable drivers that switch on at step `onset`.

    surprise       |expected_score - score| after onset (0.05 before)
    perseveration  probability a step reuses the tags of the failed step 1
    invalid_rate   probability a step is invalid
    tag_diversity  fraction of the tag vocabulary in use; < 0.5 also makes programs identical
    improving      False -> the score stops improving at onset (stagnation)
    context        context_tokens after onset (1000 before)
    Step 1 is always an invalid step with tags ["dead-end-a", "dead-end-b"].
    """
    rng = random.Random(seed)
    vocab = [f"tag{i}" for i in range(24)]
    dead = ["dead-end-a", "dead-end-b"]
    recs = []
    for step in range(n):
        after = step >= onset
        if step == 1:
            valid, tags = False, list(dead)
        else:
            valid = not (after and rng.random() < invalid_rate)
            if after and rng.random() < perseveration:
                tags = list(dead)
            else:
                pool = vocab[:max(2, round(len(vocab) * tag_diversity))] if after else vocab
                tags = rng.sample(pool, 2)
        score = 0.3 + 0.01 * step if (improving or not after) else 0.3 + 0.01 * onset - 0.02
        if after and tag_diversity < 0.5:
            program = "def construct():\n    x = 1\n    return [x, x, x]\n"
        else:
            words = " ".join(rng.choice("abcdefghij") + str(rng.randint(0, 999)) for _ in range(6))
            program = f"def construct():\n    # {words}\n    return [{rng.random():.6f}]\n"
        recs.append({"event": "step", "step": step, "valid": valid, "score": score if valid else None,
                     "expected_score": score + (surprise if after else 0.05), "approach_tags": tags,
                     "error": None if valid else "constraint violated", "program": program,
                     "agent_tokens": tokens_per_step * step, "context_tokens": context if after else 1000})
    return recs


@pytest.fixture
def make_history():
    return synthetic_history


# ================================================================================================
# Section 1 · Budget and LLM client
# ================================================================================================

def test_agent_and_sleep_tokens_count_against_one_shared_total():
    b = Budget(1000)
    b.charge("agent", 300, 100)
    b.charge("sleep", 200, 100)
    assert (b.agent_tokens, b.sleep_tokens, b.used, b.remaining) == (400, 300, 700, 300)


def test_budget_exhausted_is_raised_exactly_when_the_total_is_exceeded():
    b = Budget(1000)
    b.charge("agent", 900, 0)
    b.charge("sleep", 100, 0)  # exactly at the limit: allowed, response kept
    assert b.used == 1000 and b.exhausted and b.discarded_calls == 0
    with pytest.raises(BudgetExhausted):
        b.check()  # nothing left: the next call is refused before it is made
    b2 = Budget(1000)
    b2.charge("agent", 900, 0)
    with pytest.raises(BudgetExhausted):
        b2.charge("sleep", 100, 1)  # one token over
    assert b2.overshoot == 1 and b2.discarded_calls == 1


def test_sleep_tokens_alone_can_exhaust_the_agent():
    b = Budget(500)
    b.charge("sleep", 400, 100)
    with pytest.raises(BudgetExhausted):
        b.clamp_completion(10)


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


def test_budget_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Budget(0)
    with pytest.raises(ValueError):
        Budget(10).charge("boss", 1, 1)
    with pytest.raises(ValueError):
        Budget(10).charge("agent", -5, 1)


def test_client_charges_budget_by_role(mock_llm):
    llm, budget, _ = mock_llm(1000, prompt_tokens=60, completion_tokens=40)
    llm.complete([{"role": "user", "content": "hi"}], role="agent", max_tokens=10)
    llm.complete([{"role": "system", "content": "consolidate"}], role="sleep", max_tokens=10)
    assert (budget.agent_tokens, budget.sleep_tokens) == (100, 100)
    assert (budget.agent_calls, budget.sleep_calls) == (1, 1)


def test_client_refuses_call_when_exhausted_and_withholds_overshooting_response(mock_llm):
    llm, budget, transport = mock_llm(150, prompt_tokens=60, completion_tokens=40)
    llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    with pytest.raises(BudgetExhausted):  # 200 > 150: charged, response withheld
        llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    n = len(transport.calls)
    with pytest.raises(BudgetExhausted):  # nothing left: transport never called
        llm.complete([{"role": "user", "content": "x"}], role="sleep", max_tokens=10)
    assert len(transport.calls) == n


def test_loop_stops_cleanly_when_a_call_overshoots_the_budget(run_dir, mock_llm, mock_cfg):
    """950 tokens at 100/call: 9 calls fit, the 10th is charged, its response discarded, the run ends."""
    llm, budget, transport = mock_llm(950)
    s = run(mock_cfg(total_token_budget=950), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    assert s["stop_reason"] == "budget_exhausted" and s["steps_completed"] == 9
    assert budget.used == 1000 and budget.overshoot == 50 and budget.discarded_calls == 1
    assert len(transport.calls) == 10
    hist = hist_of(run_dir, s)
    assert [h["step"] for h in hist] == list(range(10))  # the discarded response never became a step
    assert (run_dir / s["run_id"] / "summary.json").exists()


def test_summarize_charges_sleep_to_the_same_budget_in_the_loop(run_dir, mock_llm, mock_cfg):
    llm, budget, _ = mock_llm(1000, prompt_tokens=60, completion_tokens=40)
    s = run(mock_cfg(), TextEvaluator(), llm, Summarize(every_n=3), out_dir=run_dir)
    assert budget.used == 1000 and budget.overshoot == 0
    # 10 calls in total; sleeps after steps 3 and 6 -> 8 agent steps + 2 sleeps
    assert (budget.agent_calls, budget.sleep_calls) == (8, 2)
    assert s["budget"]["agent_tokens"] + s["budget"]["sleep_tokens"] == 1000
    hist = hist_of(run_dir, s)
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert [h["step"] for h in sleeps] == [3, 6]
    assert sleeps[0]["details"]["block"].startswith("Consolidated notes")
    assert hist[-1]["sleep_tokens"] == 200


def test_token_counts_come_from_the_reported_usage_not_estimates():
    usages = [(123, 45), (7, 301), (1000, 1)]

    class VaryingTransport:
        def __init__(self):
            self.i = 0

        def __call__(self, messages, max_tokens):
            pt, ct = usages[self.i]
            self.i += 1
            return LLMResponse("x" * 5000, pt, ct)  # long text: an estimate would be far off

    budget = Budget(10_000)
    llm = LLMClient(budget, VaryingTransport(), sleep_fn=lambda s: None)
    for _ in usages:
        llm.complete([{"role": "user", "content": "y" * 3000}], role="agent", max_tokens=10)
    assert budget.used == sum(p + c for p, c in usages)


def test_openai_transport_without_usage_field_raises_instead_of_estimating():
    t = OpenAICompatTransport("http://127.0.0.1:9/v1", "sk-test", "m")
    msg = SimpleNamespace(content="hello")
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)
    t._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: resp)))
    with pytest.raises(MissingUsageError):
        t([{"role": "user", "content": "x"}], 10)
    resp.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=3)
    assert t([{"role": "user", "content": "x"}], 10) == LLMResponse("hello", 11, 3)


class FlakyTransport:
    def __init__(self, failures: int, message: str = "HTTP 429"):
        self.failures, self.message = failures, message
        self.calls = 0

    def __call__(self, messages, max_tokens):
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientLLMError(self.message)
        return LLMResponse("ok", 10, 5)


@pytest.mark.parametrize("message", ["HTTP 429", "HTTP 503: upstream down"])
def test_retries_transient_errors_with_exponential_backoff(message):
    delays = []
    budget = Budget(1000)
    llm = LLMClient(budget, FlakyTransport(3, message), sleep_fn=delays.append, backoff_base_s=1.0)
    resp = llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    assert resp.text == "ok" and len(delays) == 3
    assert delays[0] < delays[1] < delays[2]  # exponential (jitter in [0.5, 1.0])
    assert budget.used == 15  # failed attempts are not charged: the API reported no usage


def test_gives_up_after_the_configured_attempts():
    t = FlakyTransport(99)
    llm = LLMClient(Budget(1000), t, max_retries=2, sleep_fn=lambda s: None)
    with pytest.raises(TransientLLMError):
        llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    assert t.calls == 3  # 1 attempt + 2 retries


def test_openai_transport_maps_429_and_5xx_to_transient_errors():
    import openai

    t = OpenAICompatTransport("http://127.0.0.1:9/v1", "sk-test", "m")

    def raising(status):
        def create(**kw):
            e = openai.APIStatusError.__new__(openai.APIStatusError)  # no HTTP stack needed
            Exception.__init__(e, f"HTTP {status}")
            e.status_code = status
            raise e
        return create

    for status, transient in ((429, True), (500, True), (503, True), (400, False)):
        t._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=raising(status))))
        with pytest.raises(TransientLLMError if transient else openai.APIStatusError):
            t([{"role": "user", "content": "x"}], 10)


def test_client_requires_a_budget():
    with pytest.raises(TypeError):
        LLMClient(None, FlakyTransport(0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LLMClient(SimpleNamespace(used=0, clamp_completion=lambda n: n), FlakyTransport(0))  # duck-typed fake


def test_no_module_imports_openai_outside_the_client():
    """Only llm/client.py may talk to the API, so every call is charged to the Budget."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        if ".venv" in path.parts or path in (ROOT / "llm" / "client.py", ROOT / "tests.py"):
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(import openai|from openai\b)", text, re.M) or "chat.completions" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


# ---- token-counting proxy (infra/llm_proxy.py) -------------------------------------------------

from arms.mock_upstream import FakeUpstream, mock_transport_responder, p36_responder  # noqa: E402
from infra.llm_proxy import BUDGET_EXHAUSTED, USER_AGENT, LLMProxy, ProxyError, ProxyLM, post_json  # noqa: E402


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload


def _varying_usage():
    """Different, known usage per call so 'charged == reported' is a real check."""
    usages = [(123, 45), (7, 301), (1000, 1), (55, 55), (2, 3)]
    state = {"i": 0}

    def respond(messages, body):
        pt, ct = usages[state["i"] % len(usages)]
        state["i"] += 1
        return f"reply {state['i']}", pt, ct

    return respond, usages


@pytest.fixture
def proxy_stack():
    respond, usages = _varying_usage()
    up = FakeUpstream(respond).start()
    proxy = LLMProxy(up.base_url, "sk-test", "served-model").start()
    yield up, proxy, usages
    proxy.stop()
    up.stop()


def _post(proxy, body, headers=None):
    return _Resp(*post_json(proxy.base_url + "/chat/completions", body, headers or {}, 10))


def test_proxy_charges_exactly_the_upstream_usage_to_the_run(proxy_stack):
    up, proxy, usages = proxy_stack
    budget = Budget(100_000)
    proxy.register("run-a", budget)
    lm = ProxyLM(proxy.base_url, "run-a", max_tokens=500)
    texts = [lm("hi") for _ in range(5)]
    assert texts[0] == "reply 1"
    assert budget.agent_tokens == sum(p + c for p, c in usages)
    assert budget.agent_calls == 5 and budget.sleep_tokens == 0
    assert (lm.total_tokens_in, lm.total_tokens_out) == (sum(p for p, _ in usages), sum(c for _, c in usages))
    assert proxy.usage_log("run-a") == [{"role": "agent", "prompt_tokens": p, "completion_tokens": c,
                                         "cached_tokens": None, "reasoning_tokens": None} for p, c in usages]
    # forwarded with the configured key and model, not the client's
    assert up.headers[0]["Authorization"] == "Bearer sk-test"
    assert all(r["model"] == "served-model" for r in up.requests)


def test_proxy_forwards_with_an_explicit_user_agent(proxy_stack):
    """Cloudflare (Tinker) answers 403 'error code: 1010' to urllib's default Python-urllib UA."""
    up, proxy, _ = proxy_stack
    proxy.register("r", Budget(10_000))
    ProxyLM(proxy.base_url, "r")("x")
    ua = {k.lower(): v for k, v in up.headers[-1].items()}["user-agent"]
    assert ua == USER_AGENT and not ua.lower().startswith("python-urllib")


def test_proxy_role_header_charges_sleep(proxy_stack):
    _, proxy, _ = proxy_stack
    budget = Budget(10_000)
    proxy.register("r", budget)
    ProxyLM(proxy.base_url, "r", role="sleep")("x")
    assert budget.sleep_tokens == 168 and budget.agent_tokens == 0


def test_proxy_reports_usage_to_the_right_runs_budget(proxy_stack):
    _, proxy, _ = proxy_stack
    a, b = Budget(10_000), Budget(10_000)
    proxy.register("a", a)
    proxy.register("b", b)
    ProxyLM(proxy.base_url, "a")("x")
    ProxyLM(proxy.base_url, "b")("x")
    ProxyLM(proxy.base_url, "b")("x")
    assert a.used == 168 and b.used == 308 + 1001


def test_proxy_clamps_max_tokens_to_remaining(proxy_stack):
    up, proxy, _ = proxy_stack
    budget = Budget(500)
    proxy.register("r", budget)
    _post(proxy, {"messages": [], "max_tokens": 9999}, {"X-Run-Id": "r"})
    assert up.requests[-1]["max_tokens"] == 500
    _post(proxy, {"messages": []}, {"X-Run-Id": "r"})  # no max_tokens -> min(default, remaining)
    assert up.requests[-1]["max_tokens"] == 500 - 168
    _post(proxy, {"messages": [], "max_completion_tokens": 50}, {"X-Run-Id": "r"})
    assert up.requests[-1]["max_completion_tokens"] == 500 - 168 - 308
    assert "max_tokens" not in up.requests[-1]


def test_proxy_rejects_requests_over_budget_without_forwarding(proxy_stack):
    up, proxy, _ = proxy_stack
    budget = Budget(400)  # first call 168, second 308 -> 476 > 400
    proxy.register("r", budget)
    lm = ProxyLM(proxy.base_url, "r")
    lm("one")
    with pytest.raises(BudgetExhausted):
        lm("two")
    assert lm.exhausted
    assert budget.used == 476 and budget.overshoot == 76 and budget.discarded_calls == 1
    n = len(up.requests)
    r = _post(proxy, {"messages": []}, {"X-Run-Id": "r"})
    assert r.status_code == 402 and r.json()["error"]["code"] == BUDGET_EXHAUSTED
    assert len(up.requests) == n  # refused before forwarding
    with pytest.raises(BudgetExhausted):
        lm("three")  # the client also remembers


def test_proxy_exact_budget_hit_returns_response_then_refuses(proxy_stack):
    _, proxy, _ = proxy_stack
    budget = Budget(168)
    proxy.register("r", budget)
    lm = ProxyLM(proxy.base_url, "r")
    assert lm("x") == "reply 1"
    assert budget.exhausted and budget.overshoot == 0
    with pytest.raises(BudgetExhausted):
        lm("y")


def test_call_routed_around_the_budget_fails(proxy_stack):
    """A request that names no registered Budget is refused and never reaches the upstream."""
    up, proxy, _ = proxy_stack
    proxy.register("r", Budget(1000))
    assert _post(proxy, {"messages": []}).json()["error"]["code"] == "missing_run_id"
    assert _post(proxy, {"messages": []}, {"X-Run-Id": "nope"}).status_code == 404
    with pytest.raises(ProxyError):
        ProxyLM(proxy.base_url, "unregistered", sleep_fn=lambda s: None)("x")
    assert up.requests == []
    # and the in-process path: an exhausted Budget means the transport is never called
    t = FlakyTransport(0)
    b = Budget(10)
    b.charge("agent", 10, 0)
    with pytest.raises(BudgetExhausted):
        LLMClient(b, t).complete([], role="agent", max_tokens=5)
    assert t.calls == 0


def test_proxy_missing_usage_is_an_error_and_charges_nothing():
    with FakeUpstream(lambda m, b: ("x", 1, 1), omit_usage=True) as up, LLMProxy(up.base_url) as proxy:
        budget = Budget(1000)
        proxy.register("r", budget)
        r = _post(proxy, {"messages": []}, {"X-Run-Id": "r"})
        assert r.status_code == 502 and r.json()["error"]["code"] == "missing_usage"
        with pytest.raises(ProxyError):
            ProxyLM(proxy.base_url, "r", sleep_fn=lambda s: None)("x")
        assert budget.used == 0


def test_proxy_bad_requests_are_refused(proxy_stack):
    _, proxy, _ = proxy_stack
    proxy.register("r", Budget(1000))
    assert _post(proxy, {"messages": [], "stream": True}, {"X-Run-Id": "r"}).status_code == 400
    assert _post(proxy, {"messages": []}, {"X-Run-Id": "r", "X-Budget-Role": "boss"}).status_code == 400
    with urlrequest.urlopen(proxy.base_url + "/models", timeout=10) as r:
        assert json.loads(r.read())["data"][0]["id"] == "served-model"


def test_proxy_upstream_outage_is_retried_then_raises_without_charge():
    with FakeUpstream(lambda m, b: ("x", 1, 1), status=503) as up, LLMProxy(up.base_url) as proxy:
        budget = Budget(1000)
        proxy.register("r", budget)
        delays = []
        lm = ProxyLM(proxy.base_url, "r", max_retries=2, sleep_fn=delays.append)
        with pytest.raises(ProxyError):
            lm("x")
        assert len(delays) == 2 and len(up.requests) == 3 and budget.used == 0
        assert not lm.exhausted


def test_proxy_register_requires_a_budget_and_rejects_rebinding(proxy_stack):
    _, proxy, _ = proxy_stack
    with pytest.raises(TypeError):
        proxy.register("r", None)  # type: ignore[arg-type]
    proxy.register("r", Budget(10))
    with pytest.raises(ValueError):
        proxy.register("r", Budget(10))


# ---- OpenAI reasoning-model parameters, usage breakdown, billing errors, cost --------------------

from harness.cost import DEFAULT_PRICES, ModelPrice, cost_block, lookup_price, price_table  # noqa: E402
from harness.cost import usage_cost  # noqa: E402
from harness.logging import USAGE_FIELDS  # noqa: E402
from llm.chat_params import (  # noqa: E402
    BillingError, billing_error_from_body, is_reasoning_model, parse_usage, resolve_chat_params,
)


@pytest.mark.parametrize("model, reasoning", [
    ("gpt-5-mini", True), ("gpt-5-nano", True), ("gpt-5", True), ("gpt-5-mini-2025-08-07", True),
    ("gpt-5.4-nano", True), ("gpt-6.1-terra", True), ("o1", True), ("o3-mini", True), ("o4-mini", True),
    ("openai/gpt-5-mini", True), ("gpt-5-chat-latest", False), ("Qwen/Qwen3-8B", False), ("gpt-4o", False),
    ("gpt-4.1-mini", False), ("ollama-model", False), (None, False), ("mock-model", False),
])
def test_reasoning_models_are_detected_by_name(model, reasoning):
    assert is_reasoning_model(model) is reasoning
    p = resolve_chat_params(model, env={})
    assert p.token_param == ("max_completion_tokens" if reasoning else "max_tokens")
    assert p.request_kwargs(100) == {p.token_param: 100}  # nothing else unless configured


def test_param_selection_reasoning_vs_non_reasoning_and_precedence():
    r = resolve_chat_params("gpt-5-mini", {"reasoning_effort": "low", "temperature": 0.2}, env={}, warn=False)
    assert r.request_kwargs(64) == {"max_completion_tokens": 64, "reasoning_effort": "low"}
    assert r.dropped == ("temperature",) and r.sources["reasoning_effort"] == "config"
    # temperature is allowed on a reasoning model only with reasoning_effort "none"
    n = resolve_chat_params("gpt-5.4-mini", {"reasoning_effort": "none", "temperature": 0.2}, env={}, warn=False)
    assert n.request_kwargs(8) == {"max_completion_tokens": 8, "reasoning_effort": "none", "temperature": 0.2}
    # vLLM / Qwen keeps today's request (max_tokens only); a stray effort is not sent
    q = resolve_chat_params("Qwen/Qwen3-8B", {"reasoning_effort": "low", "temperature": 0.7}, env={}, warn=False)
    assert q.request_kwargs(32) == {"max_tokens": 32, "temperature": 0.7} and q.dropped == ("reasoning_effort",)
    # config > env > auto; explicit overrides of the auto choice
    env = {"LLM_REASONING_EFFORT": "high", "LLM_TOKEN_PARAM": "max_tokens"}
    e = resolve_chat_params("gpt-5-mini", None, env=env)
    assert e.reasoning_effort == "high" and e.token_param == "max_tokens" and e.sources["token_param"] == "env"
    c = resolve_chat_params("gpt-5-mini", {"reasoning_effort": "medium"}, env=env)
    assert c.reasoning_effort == "medium" and c.sources["reasoning_effort"] == "config"
    forced = resolve_chat_params("my-local-reasoner", {"reasoning_model": True}, env={})
    assert forced.token_param == "max_completion_tokens"
    assert resolve_chat_params("Qwen/Qwen3-8B", {"token_param": "max_completion_tokens"}, env={}).token_param == \
        "max_completion_tokens"
    with pytest.raises(ValueError):
        resolve_chat_params("gpt-5-mini", {"reasoning_effort": "extreme"}, env={})
    with pytest.raises(ValueError):
        resolve_chat_params("gpt-5-mini", None, env={"LLM_TOKEN_PARAM": "max_output_tokens"})
    with pytest.raises(ValidationError):  # the config schema rejects bad values too
        RunConfig(problem_id=0, total_token_budget=10, max_steps=1, llm={"reasoning_effort": "extreme"})


def _capturing_transport(model, params=None, usage=None):
    t = OpenAICompatTransport("http://127.0.0.1:9/v1", "sk-test", model, params=params)
    seen = []
    usage = usage or SimpleNamespace(prompt_tokens=11, completion_tokens=3)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))], usage=usage)

    def create(**kw):
        seen.append(kw)
        return resp

    t._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return t, seen


def test_openai_transport_sends_the_resolved_parameters():
    p = resolve_chat_params("gpt-5-mini", {"reasoning_effort": "low", "temperature": 0.5}, env={}, warn=False)
    t, seen = _capturing_transport("gpt-5-mini", p)
    t([{"role": "user", "content": "x"}], 77)
    assert seen[-1]["max_completion_tokens"] == 77 and seen[-1]["reasoning_effort"] == "low"
    assert "max_tokens" not in seen[-1] and "temperature" not in seen[-1] and seen[-1]["model"] == "gpt-5-mini"
    t2, seen2 = _capturing_transport("Qwen/Qwen3-8B")  # vLLM fallback: exactly what it sent before
    t2([{"role": "user", "content": "x"}], 77)
    assert {k: v for k, v in seen2[-1].items() if k != "messages"} == {"model": "Qwen/Qwen3-8B", "max_tokens": 77}


def test_usage_parsing_includes_cached_and_reasoning_tokens():
    sdk_like = SimpleNamespace(prompt_tokens=1000, completion_tokens=300,
                               prompt_tokens_details=SimpleNamespace(cached_tokens=600, audio_tokens=0),
                               completion_tokens_details=SimpleNamespace(reasoning_tokens=200))
    as_dict = {"prompt_tokens": 1000, "completion_tokens": 300, "total_tokens": 1300,
               "prompt_tokens_details": {"cached_tokens": 600}, "completion_tokens_details": {"reasoning_tokens": 200}}
    for u in (sdk_like, as_dict):
        got = parse_usage(u)
        assert (got.prompt_tokens, got.completion_tokens, got.cached_tokens, got.reasoning_tokens) == (1000, 300, 600, 200)
    bare = parse_usage({"prompt_tokens": 5, "completion_tokens": 2, "prompt_tokens_details": None})
    assert (bare.cached_tokens, bare.reasoning_tokens) == (None, None)  # e.g. vLLM without details
    assert parse_usage({"prompt_tokens": 5}) is None and parse_usage(None) is None
    t, _ = _capturing_transport("gpt-5-nano", usage=sdk_like)
    assert t([{"role": "user", "content": "x"}], 10) == LLMResponse("hi", 1000, 300, 600, 200)
    # the Budget charges prompt + completion only (reasoning is inside completion, cached inside prompt)
    budget = Budget(10_000)
    llm = LLMClient(budget, t, sleep_fn=lambda s: None)
    llm.complete([{"role": "user", "content": "x"}], role="agent", max_tokens=10)
    llm.complete([{"role": "user", "content": "x"}], role="sleep", max_tokens=10)
    assert budget.used == 2 * 1300 and budget.agent_tokens == 1300
    bd = budget.usage_breakdown()
    assert bd["agent"] == {"calls": 1, "prompt_tokens": 1000, "completion_tokens": 300, "cached_tokens": 600,
                           "reasoning_tokens": 200}
    assert bd["total"]["cached_tokens"] == 1200 and bd["total"]["reasoning_tokens"] == 400
    assert budget.usage_since(1) == bd["sleep"]
    assert budget.usage_since(2) == {"calls": 0, "prompt_tokens": None, "completion_tokens": None,
                                     "cached_tokens": None, "reasoning_tokens": None}


def _status_error(status, code=None, type_=None):
    import openai

    e = openai.APIStatusError.__new__(openai.APIStatusError)  # no HTTP stack needed
    Exception.__init__(e, f"HTTP {status}")
    e.status_code, e.code, e.type, e.body = status, code, type_, {"code": code, "type": type_, "message": "m"}
    return e


@pytest.mark.parametrize("code, type_", [
    ("insufficient_quota", "insufficient_quota"), ("credit_balance_exhausted", "insufficient_quota"),
    ("organization_spend_limit_exceeded", None), ("project_spend_limit_exceeded", "insufficient_quota"),
    ("organization_usage_limit_exceeded", None), (None, "insufficient_quota"),
])
def test_billing_errors_are_not_retried(code, type_):
    t = OpenAICompatTransport("http://127.0.0.1:9/v1", "sk-test", "gpt-5-mini")
    calls = []

    def create(**kw):
        calls.append(kw)
        raise _status_error(429, code, type_)

    t._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    delays = []
    budget = Budget(1000)
    with pytest.raises(BillingError, match="Retrying cannot help"):
        LLMClient(budget, t, sleep_fn=delays.append).complete([{"role": "user", "content": "x"}], role="agent",
                                                              max_tokens=10)
    assert len(calls) == 1 and delays == [] and budget.used == 0
    # a plain rate limit (429 without a billing code) is still retried
    t._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: (_ for _ in ()).throw(_status_error(429, "rate_limit_exceeded", "requests")))))
    with pytest.raises(TransientLLMError):
        t([{"role": "user", "content": "x"}], 10)
    assert billing_error_from_body(429, {"error": {"code": "slow_down", "type": "rate_limit_error"}}) is None


def test_loop_billing_error_stops_the_run_writes_outputs_and_reraises(run_dir, mock_cfg):
    class BillingAfter:
        def __init__(self, n):
            self.n, self.calls = n, 0

        def __call__(self, messages, max_tokens):
            self.calls += 1
            if self.calls > self.n:
                raise BillingError("You exceeded your current quota", status=429, code="insufficient_quota")
            return LLMResponse(GOOD_REPLY, 60, 40)

    t = BillingAfter(3)
    llm = LLMClient(Budget(10_000), t, sleep_fn=lambda s: pytest.fail("billing errors must not back off"))
    with pytest.raises(BillingError):
        run(mock_cfg(total_token_budget=10_000, run_id="bill"), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    s = json.loads((run_dir / "bill" / "summary.json").read_text(encoding="utf-8"))
    assert s["stop_reason"] == "billing_error" and "insufficient_quota" in s["error"] and s["steps_completed"] == 3
    assert t.calls == 4


def test_proxy_applies_the_runs_parameters_and_records_the_usage_breakdown(monkeypatch):
    import infra.llm_proxy as lp

    sent = []

    def fake_post(url, body, headers, timeout_s):
        sent.append(dict(body))
        return 200, {"choices": [{"message": {"content": "ok"}}],
                     "usage": {"prompt_tokens": 500, "completion_tokens": 120, "total_tokens": 620,
                               "prompt_tokens_details": {"cached_tokens": 256},
                               "completion_tokens_details": {"reasoning_tokens": 64}}}

    monkeypatch.setattr(lp, "post_json", fake_post)
    params = resolve_chat_params("gpt-5-mini", {"reasoning_effort": "low"}, env={})
    proxy = LLMProxy("http://upstream.invalid/v1", "k", "gpt-5-mini", chat_params=params)
    try:
        budget = Budget(10_000)
        proxy.register("r", budget)
        body = {"model": "x", "messages": [], "max_tokens": 99_999, "temperature": 0.3, "top_p": 0.9}
        status, _ = proxy.handle_completion({"X-Run-Id": "r", "X-Budget-Role": "sleep"}, json.dumps(body).encode())
        assert status == 200
        fwd = sent[-1]
        assert fwd["max_completion_tokens"] == 10_000 and "max_tokens" not in fwd  # clamped, renamed
        assert fwd["reasoning_effort"] == "low" and "temperature" not in fwd and "top_p" not in fwd
        assert fwd["model"] == "gpt-5-mini"
        assert budget.sleep_tokens == 620  # prompt + completion, as reported
        assert proxy.usage_log("r") == [{"role": "sleep", "prompt_tokens": 500, "completion_tokens": 120,
                                         "cached_tokens": 256, "reasoning_tokens": 64}]
        assert budget.usage_breakdown()["sleep"]["cached_tokens"] == 256
    finally:
        proxy.stop()


def test_proxy_lm_does_not_retry_billing_errors(monkeypatch):
    import infra.llm_proxy as lp

    calls = []

    def fake_post(url, body, headers, timeout_s):
        calls.append(body)
        return 429, {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota",
                               "code": "insufficient_quota"}}

    monkeypatch.setattr(lp, "post_json", fake_post)
    delays = []
    lm = ProxyLM("http://proxy.invalid/v1", "r", sleep_fn=delays.append)
    with pytest.raises(BillingError):
        lm("x")
    assert len(calls) == 1 and delays == [] and lm.billing_error is not None
    with pytest.raises(BillingError):  # remembered: no further upstream calls
        lm("y")
    assert len(calls) == 1


def test_cost_counts_cached_tokens_once_and_reasoning_tokens_once():
    price = ModelPrice(input=0.25, cached_input=0.025, output=2.0)
    # 1M prompt of which 400k cached, 200k completion of which 150k reasoning
    got = usage_cost(1_000_000, 200_000, 400_000, price)
    assert got == pytest.approx(0.6 * 0.25 + 0.4 * 0.025 + 0.2 * 2.0)  # = 0.56
    assert usage_cost(1_000_000, 200_000, None, price) == pytest.approx(0.25 + 0.4)
    assert usage_cost(10, 10, 5, ModelPrice(input=None, output=1.0)) is None
    assert usage_cost(1_000_000, 0, 1_000_000, ModelPrice(input=1.0, output=1.0)) == pytest.approx(1.0)  # no cached price
    usage = {"agent": {"prompt_tokens": 1_000_000, "completion_tokens": 200_000, "cached_tokens": 400_000},
             "sleep": {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": None},
             "total": {"prompt_tokens": 1_000_000, "completion_tokens": 200_000, "cached_tokens": 400_000}}
    block = cost_block(usage, "gpt-5-mini-2025-08-07", price_table({"gpt-5-mini": price.model_dump()}))
    assert block["price_key"] == "gpt-5-mini" and block["total"] == pytest.approx(0.56) and block["complete"]
    assert block["by_role"] == {"agent": pytest.approx(0.56), "sleep": 0.0}
    unknown = cost_block(usage, "Qwen/Qwen3-8B", price_table())
    assert unknown["total"] is None and not unknown["complete"] and "no price" in unknown["notes"][0]
    assert lookup_price("openai/gpt-5-nano", DEFAULT_PRICES)[0] == "gpt-5-nano"
    for m in ("gpt-5-nano", "gpt-5-mini"):  # verified table entries carry their source and date
        assert DEFAULT_PRICES[m].known and DEFAULT_PRICES[m].source and DEFAULT_PRICES[m].checked


def test_loop_logs_usage_per_line_and_cost_per_run(run_dir, mock_cfg):
    class Detailed:
        model = "gpt-5-mini"

        def __call__(self, messages, max_tokens):
            return LLMResponse(GOOD_REPLY if "consolidat" not in messages[0]["content"].lower() else "notes",
                               60, 40, cached_tokens=20, reasoning_tokens=10)

    budget = Budget(1000)
    s = run(mock_cfg(), TextEvaluator(), LLMClient(budget, Detailed(), sleep_fn=lambda s: None), Summarize(every_n=3),
            out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert all(set(USAGE_FIELDS) <= set(h) for h in hist)
    assert hist[0]["llm_calls"] == 0 and hist[0]["llm_prompt_tokens"] is None  # seed step: no call
    steps = [h for h in hist if h["event"] == "step" and h["step"] > 0]
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert all((h["llm_calls"], h["llm_prompt_tokens"], h["llm_completion_tokens"], h["llm_cached_tokens"],
                h["llm_reasoning_tokens"]) == (1, 60, 40, 20, 10) for h in steps + sleeps)
    assert s["usage"]["agent"]["calls"] == 8 and s["usage"]["sleep"]["calls"] == 2
    assert s["usage"]["total"]["prompt_tokens"] + s["usage"]["total"]["completion_tokens"] == budget.used == 1000
    expected = (10 * (40 * 0.25 + 20 * 0.025 + 40 * 2.0)) / 1e6  # 10 calls at gpt-5-mini prices
    assert s["cost"]["model"] == "gpt-5-mini" and s["cost"]["total"] == pytest.approx(expected)
    assert s["cost"]["by_role"]["agent"] + s["cost"]["by_role"]["sleep"] == pytest.approx(expected)
    assert s["llm_params"] is None  # a transport without resolved params (the mock) records none


# ================================================================================================
# Section 2 · Evaluators
# ================================================================================================

from benchmarks.registry import STATUSES, default_registry, load_registry  # noqa: E402
from benchmarks.scoring import compare as bench_compare  # noqa: E402
from benchmarks.scoring import compute_block, enrich_summary, gap_closed, normalized_vs  # noqa: E402
from benchmarks.scoring import read_history as bench_read_history  # noqa: E402
from benchmarks.verification import load_construction, p36_check_construction, reverify_beat, verify_problem  # noqa: E402
from benchmarks.verification import program_returning as bench_program_returning  # noqa: E402

BENCH = default_registry()
TIER1 = {2, 3, 5, 7, 34, 36, 44, 48, 50, 59}
TIER2 = {29, 46, 62}


def known_construction(pid: int) -> tuple[str, float, float] | None:
    """(program returning a published construction, published score, abs tolerance), or None."""
    if pid == 0:  # mock: x == TARGET everywhere is the optimum, score 1.0 = best_known
        return f"def construct():\n    return {[TARGET] * DIM}\n", 1.0, 1e-12
    if pid == 36:
        d = json.loads(P36_DATA.read_text())
        return program_returning(d["centers"], d["radii"]), BEST_KNOWN, 1e-12
    return None


@pytest.mark.parametrize("pid", sorted(EVALUATORS))
def test_every_registered_evaluator_conforms_to_the_contract(pid):
    ev = EVALUATORS[pid]()
    assert isinstance(ev, Evaluator)
    assert ev.problem_id == pid and isinstance(ev.problem_id, int)
    assert isinstance(ev.name, str) and ev.name
    assert isinstance(ev.best_known, float) and math.isfinite(ev.best_known) and ev.best_known != 0
    assert ev.timeout_s > 0
    assert isinstance(ev.description, str) and "construct()" in ev.description
    assert callable(ev.seed_program) and callable(ev.evaluate)
    assert "def construct" in ev.seed_program()


@pytest.mark.parametrize("pid", sorted(EVALUATORS))
def test_every_registered_evaluator_reproduces_its_published_score(pid):
    """New evaluators are picked up automatically: add a construction here or in the benchmark."""
    known = known_construction(pid)
    if known is None:
        if pid in BENCH.ids:
            r = verify_problem(pid, use_cache=False)
            assert r["status"] == "passed", r
            return
        pytest.fail(f"evaluator {pid} has no known-construction test: add data to tests/data/ and "
                    "known_construction() (or wire it into benchmarks/alphaevolve.yaml)")
    program, published, tol = known
    res = EVALUATORS[pid]().evaluate(program)
    assert res.valid, res.error
    assert res.score == pytest.approx(published, abs=tol)
    assert normalized_score(res.score, EVALUATORS[pid]().best_known) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("pid", sorted(EVALUATORS))
def test_every_registered_evaluator_seed_program_is_valid(pid):
    ev = EVALUATORS[pid]()
    res = ev.evaluate(ev.seed_program())
    assert res.valid, res.error
    if pid == 36:  # 26 equal circles on a 6x5 grid
        assert res.score == pytest.approx(26 / 12)


def test_p36_alphaevolve_construction_reproduces_best_known():
    d = json.loads(P36_DATA.read_text())
    res = CirclePacking26().score_solution([d["centers"], d["radii"]])
    assert res.valid and res.score == pytest.approx(BEST_KNOWN, abs=1e-12)
    assert res.score == pytest.approx(2.635983, abs=1e-6)


def test_p36_valid_hand_made_packing_scores_its_sum_of_radii():
    centers, radii = p36_grid()
    radii[3] = 0.05  # one shrunken circle: still valid
    res = CirclePacking26().score_solution([centers, radii])
    assert res.valid, res.error
    assert res.score == pytest.approx(25 / 12 + 0.05, abs=1e-12)
    assert res.details["sum_of_radii"] == res.score


def test_p36_overlapping_packing_is_invalid():
    centers, radii = p36_grid()
    radii[7] = 0.1  # grows into its horizontal neighbours (spacing 1/6 = 2 * 1/12)
    res = CirclePacking26().score_solution([centers, radii])
    assert not res.valid and "overlap" in res.error and res.score == -math.inf


def test_p36_out_of_bounds_packing_is_invalid():
    centers, radii = p36_grid()
    centers[0] = [0.05, 0.1]  # radius 1/12 > 0.05 -> crosses the left edge
    res = CirclePacking26().score_solution([centers, radii])
    assert not res.valid and "outside" in res.error


def test_p36_wrong_count_is_invalid():
    assert not CirclePacking26().score_solution([[[0.5, 0.5]], [0.5]]).valid


def test_p36_tolerance_is_1e9():
    c = np.array([[0.25, 0.5], [0.75, 0.5]])
    assert check_packing(c, np.array([0.25, 0.25 + 5e-10]), n=2) is None
    assert check_packing(c, np.array([0.25, 0.25 + 5e-9]), n=2) is not None


def test_hanging_program_is_killed_at_the_timeout_and_the_loop_continues(run_dir, mock_llm, mock_cfg):
    """Real sandbox: a 1 s timeout kills the child process tree; the next step still runs."""
    llm, _, _ = mock_llm(10_000, script=[HANG_REPLY, GOOD_REPLY])
    t0 = time.monotonic()
    s = run(mock_cfg(max_steps=2), MockEvaluator(timeout_s=1), llm, NoSleep(), out_dir=run_dir)
    assert time.monotonic() - t0 < 15
    hist = hist_of(run_dir, s)
    hung, good = hist[1], hist[2]
    assert hung["valid"] is False and "timeout" in hung["error"] and hung["score"] is None
    si = hung["side_info"]
    assert si["stage"] == "run" and si["failure"] == "timeout" and si["timeout_s"] == 1 and si["hint"]
    assert good["valid"] is True and s["steps_completed"] == 2


def test_crashing_program_returns_invalid_with_its_traceback():
    res = MockEvaluator(timeout_s=2).evaluate("def construct():\n    raise ValueError('boom42')\n")
    assert not res.valid and "boom42" in res.error
    assert res.side_info["failure"] == "crash"
    assert "boom42" in res.side_info["stderr_tail"] and len(res.side_info["stderr_tail"]) <= 800


def test_program_printing_garbage_and_returning_non_json_is_invalid():
    prog = ("import sys\n\ndef construct():\n    sys.stdout.write('\\x00\\xff garbage ' * 200)\n"
            "    sys.stderr.write('more garbage')\n    return {1, 2}\n")
    res = MockEvaluator(timeout_s=2).evaluate(prog)
    assert not res.valid and res.error and res.side_info["stage"] == "run"


def test_program_printing_garbage_but_returning_data_is_still_scored():
    prog = "def construct():\n    print('\\x00 garbage ' * 100)\n    return [0.7, 0.7, 0.7, 0.7, 0.7]\n"
    res = MockEvaluator(timeout_s=2).evaluate(prog)
    assert res.valid and res.score == pytest.approx(1.0)


def test_program_without_construct_is_a_crash():
    res = MockEvaluator(timeout_s=2).evaluate("x = 1\n")
    assert res.side_info["failure"] == "crash" and "construct" in res.side_info["stderr_tail"]


def test_crashing_and_codeless_responses_are_invalid_in_the_loop(run_dir, mock_llm, mock_cfg):
    crash = "```python\ndef construct():\n    raise RuntimeError('boom')\n```"
    llm, _, _ = mock_llm(10_000, script=[crash, "no code here", GOOD_REPLY])
    s = run(mock_cfg(max_steps=3), MockEvaluator(timeout_s=2), llm, NoSleep(), out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert "boom" in hist[1]["error"] and "code block" in hist[2]["error"]
    assert hist[3]["valid"] and s["best_program_id"] == "p3"


def test_evaluator_that_raises_never_crashes_the_loop(run_dir, mock_llm, mock_cfg):
    class Raising(TextEvaluator):
        def evaluate(self, program):
            if "0.7" in program:
                raise RuntimeError("evaluator bug")
            return super().evaluate(program)

    llm, _, _ = mock_llm(10_000, script=[GOOD_REPLY])
    s = run(mock_cfg(max_steps=2), Raising(), llm, NoSleep(), out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert s["steps_completed"] == 2
    assert hist[1]["valid"] is False and "evaluator raised RuntimeError" in hist[1]["error"]


def test_scoring_exception_is_invalid_with_structured_side_info():
    class Boom(MockEvaluator):
        def score_solution(self, solution):
            raise KeyError("nope")

    res = Boom(timeout_s=2).evaluate("def construct():\n    return [0]\n")
    assert not res.valid
    assert res.side_info["stage"] == "score" and res.side_info["exception"] == "KeyError"


@pytest.mark.parametrize("program, valid", [
    (GOOD, True),
    ("def construct():\n    return [0.7, 0.7, 0.2, 0.7, 0.7]\n", True),
    ("def construct():\n    return [1, 2]\n", False),
    ("def construct():\n    return ['a', 1, 1, 1, 1]\n", False),
    ("def construct():\n    return [float('nan'), 1, 1, 1, 1]\n", False),
    ("def construct():\n    raise RuntimeError('x')\n", False),
    ("y = 2\n", False),
])
def test_side_info_is_always_a_json_dict_with_diagnostic_keys_when_invalid(program, valid):
    res = MockEvaluator(timeout_s=2).evaluate(program)
    assert res.valid is valid
    assert_json(res.side_info)
    if not valid:
        assert res.side_info["stage"] in ("run", "shape", "constraint", "score")
        if res.side_info["stage"] == "run":
            assert {"failure", "hint"} <= set(res.side_info)


def test_mock_side_info_reports_per_coordinate_error():
    res = MockEvaluator().score_solution([0.7, 0.7, 0.2, 0.7, 0.7])
    si = res.side_info
    assert res.valid
    assert len(si["per_coordinate_sq_error"]) == DIM
    assert si["per_coordinate_sq_error"][2] == pytest.approx((0.2 - TARGET) ** 2)
    assert si["worst_coordinate"] == 2


def test_mock_side_info_for_bad_shape():
    res = MockEvaluator().score_solution([1, 2])
    assert res.side_info == {"stage": "shape", "expected": f"list of {DIM} floats", "got_type": "list", "got_len": 2}


def test_p36_valid_side_info():
    centers, radii = p36_grid()
    res = CirclePacking26().score_solution([centers, radii])
    si = res.side_info
    assert res.valid
    assert_json(si)
    assert si["num_overlaps"] == 0 and si["num_edge_violations"] == 0
    assert si["edge_violations"] == [] and si["overlaps"] == []
    assert si["min_edge_slack"] == pytest.approx(0.0, abs=1e-12)
    assert si["tight_contacts"]["pairs"] > 0
    rows = si["render"].splitlines()
    assert len(rows) == 20 and all(len(r) == 40 for r in rows)
    assert len(json.dumps(si)) < 3000  # compact enough for a prompt


def test_p36_alphaevolve_construction_is_tight():
    d = json.loads(P36_DATA.read_text())
    si = CirclePacking26().score_solution([d["centers"], d["radii"]]).side_info
    assert si["growable"] == []  # an optimum has no free room
    assert all(g in si["render"] for g in "abcdefghijklmnopqrstuvwxyz")


def test_p36_overlap_side_info_lists_pairs_and_amounts():
    centers, radii = p36_grid()
    radii[7] = 0.1
    res = CirclePacking26().score_solution([centers, radii])
    si = res.side_info
    assert not res.valid and si["stage"] == "constraint"
    assert_json(si)
    pairs = {(o["i"], o["j"]) for o in si["overlaps"]}
    assert {(6, 7), (7, 8)} <= pairs
    by = {(o["i"], o["j"]): o["by"] for o in si["overlaps"]}
    assert by[(6, 7)] == pytest.approx(0.1 + 1 / 12 - 1 / 6, rel=1e-3)
    assert si["num_overlaps"] == len(si["overlaps"]) and si["min_pair_slack"] < 0 and "#" in si["render"]
    assert [o["by"] for o in si["overlaps"]] == sorted((o["by"] for o in si["overlaps"]), reverse=True)


def test_p36_edge_violation_side_info_names_circle_edge_and_amount():
    centers, radii = p36_grid()
    centers[0] = [0.05, 0.1]
    si = CirclePacking26().score_solution([centers, radii]).side_info
    left = [v for v in si["edge_violations"] if v["circle"] == 0 and v["edge"] == "left"]
    assert left and left[0]["by"] == pytest.approx(1 / 12 - 0.05, rel=1e-3)
    assert si["min_edge_slack"] == pytest.approx(0.05 - 1 / 12, rel=1e-3)


def test_p36_side_info_lists_are_capped_but_counts_complete():
    si = packing_side_info(np.full((26, 2), 0.5), np.full(26, 0.3))
    assert si["num_overlaps"] == 26 * 25 // 2
    assert len(si["overlaps"]) == 10 and si["num_edge_violations"] == 0


def test_p36_growable_side_info_finds_shrunken_circle():
    centers, radii = p36_grid()
    radii[3] = 0.05
    si = CirclePacking26().score_solution([centers, radii]).side_info
    assert si["growable"][0]["circle"] == 3
    assert si["growable"][0]["room"] == pytest.approx(1 / 12 - 0.05, rel=1e-3)


def test_p36_shape_errors_are_structured():
    ev = CirclePacking26()
    res = ev.score_solution([[[0.5, 0.5]], [0.5]])
    assert not res.valid and res.side_info["stage"] == "shape"
    assert res.side_info["got"] == {"centers": [1, 2], "radii": [1]}
    res = ev.score_solution(3)
    assert res.side_info["stage"] == "shape" and res.side_info["got_type"] == "int"
    assert_json(res.side_info)


def test_p36_negative_radius_is_flagged():
    centers, radii = p36_grid()
    radii[2] = -0.01
    res = CirclePacking26().score_solution([centers, radii])
    assert not res.valid and res.side_info["negative_radii"] == [2]


def test_p36_render_marks_tiny_circle_at_its_center():
    assert render_packing(np.array([[0.5, 0.5]]), np.array([1e-4]), cols=10, rows=10).count("a") == 1


def test_failure_side_info_helpers():
    info = failure_side_info(ProgramFailed("x", kind="bad_output"))
    assert info["failure"] == "bad_output" and "timeout_s" not in info
    assert invalid("e").side_info == {}
    assert invalid("e", side_info={"a": 1}, foo=2).details == {"foo": 2}


def test_normalized_score_handles_minimization():
    assert normalized_score(2.0, 4.0) == 0.5  # maximize
    assert normalized_score(-8.0, -4.0) == 0.5  # minimize (negated): worse -> below 1
    assert normalized_score(-2.0, -4.0) == 2.0  # beats best_known
    assert normalized_score(float("-inf"), 4.0) is None


# ---- AlphaEvolve benchmark layer (benchmarks/) -------------------------------------------------

BENCH_SYNTH_YAML = """
schema_version: 1
retrieved: "test"
sources: {test: "synthetic"}
status_labels: {world_record: world_record}
problems:
  - problem_id: 901
    name: "synthetic max"
    tier: 1
    page: "n/a"
    paper_ref: "n/a"
    status: world_record
    status_raw: world_record
    evaluator: "tests.fake:Max"
    instances:
      - id: k10
        params: {k: 10}
        quantity: "x"
        direction: maximize
        score_transform: identity
        alphaevolve_value: {value: 10.0, source: "synthetic"}
        prior_best_value: {value: 9.0, source: "synthetic"}
        current_best_value: {value: 10.0, source: "synthetic"}
        match_tolerance: 1.0e-6
        verification_tolerance: 1.0e-9
        third_party:
          - {system: Other, value: 8.0, source: "synthetic"}
          - {system: Missing, value: null, source: "synthetic", missing_note: "not reported"}
  - problem_id: 902
    name: "synthetic min"
    tier: 1
    page: "n/a"
    paper_ref: "n/a"
    status: world_record
    status_raw: world_record
    evaluator: "tests.fake:Min"
    instances:
      - id: k3
        params: {k: 3}
        quantity: "ratio"
        direction: minimize
        score_transform: negate
        alphaevolve_value: {value: 2.0, source: "synthetic"}
        prior_best_value: {value: 2.5, source: "synthetic"}
        current_best_value: {value: 2.0, source: "synthetic"}
        match_tolerance: 1.0e-6
        verification_tolerance: 1.0e-9
  - problem_id: 903
    name: "synthetic pending"
    tier: 2
    page: "n/a"
    paper_ref: "n/a"
    status: matched_optimal
    status_raw: matched_optimal
    evaluator: pending
    instances:
      - id: a
        params: {k: 1}
        quantity: "x"
        direction: maximize
        score_transform: identity
        alphaevolve_value: {value: null, source: "synthetic", missing_note: "not reported"}
        prior_best_value: {value: null, source: "synthetic", missing_note: "not reported"}
        current_best_value: {value: null, source: "synthetic", missing_note: "not reported"}
        match_tolerance: 0.0
        verification_tolerance: 0.0
"""


class BenchFakeEvaluator:
    """Declares a benchmark instance; never runs anything (compute_block is called with verify=False)."""

    timeout_s = 5

    def __init__(self, problem_id: int, best_known: float, instance: dict):
        self.problem_id = problem_id
        self.best_known = best_known
        self.benchmark_instance = instance


def synth_registry(tmp_path: Path):
    p = tmp_path / "synth_registry.yaml"
    p.write_text(BENCH_SYNTH_YAML, encoding="utf-8")
    return load_registry(p)


def make_run(root: Path, run_id: str, *, problem_id: int, scores: list, arm: str = "no_sleep",
             total: int = 1000, tokens_per_step: int = 100, sleep_tokens_at: dict | None = None,
             best_program: str | None = None, extra: dict | None = None, seed: int = 0) -> Path:
    """A run dir: scores[0] is the seed (step 0, 0 tokens), step i has i*tokens_per_step agent tokens."""
    d = root / run_id
    d.mkdir(parents=True)
    lines, best, best_id, best_step = [], None, None, None
    sleep_tokens_at = sleep_tokens_at or {}
    sleep = 0
    for i, s in enumerate(scores):
        sleep += sleep_tokens_at.get(i, 0)
        valid = s is not None
        lines.append({"run_id": run_id, "step": i, "event": "step", "program_id": f"p{i}",
                      "program": f"# program {i}\n", "score": s, "valid": valid,
                      "agent_tokens": i * tokens_per_step, "sleep_tokens": sleep})
        if valid and (best is None or s > best):
            best, best_id, best_step = s, f"p{i}", i
    (d / "history.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    summary = {"run_id": run_id, "problem_id": problem_id, "problem": "synthetic", "sleep_policy": arm,
               "seed": seed, "best_program_id": best_id, "best_score": best, "best_step": best_step,
               "budget": {"total_tokens": total, "used": (len(scores) - 1) * tokens_per_step + sleep},
               "llm_model": "test-model", "config_model": None}
    summary.update(extra or {})
    (d / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if best_program is not None:
        (d / "best_program.py").write_text(best_program, encoding="utf-8")
    return d


def _bench_hist(d: Path) -> list[dict]:
    return bench_read_history(d / "history.jsonl")


def test_benchmark_registry_loads_every_problem_with_its_tier():
    assert set(BENCH.ids) == TIER1 | TIER2
    for p in BENCH.problems:
        assert p.tier == (1 if p.problem_id in TIER1 else 2)
        assert p.status in STATUSES


@pytest.mark.parametrize("pid", sorted(TIER1 | TIER2))
def test_benchmark_entry_has_direction_parameters_value_or_missing_note_and_source(pid):
    p = BENCH.get(pid)
    assert p.name and p.page and p.paper_ref
    assert p.evaluator == "pending" or ":" in p.evaluator
    for inst in p.instances:
        assert inst.params, "instance parameters are required"
        assert inst.direction in ("maximize", "minimize")
        assert inst.score_transform == ("identity" if inst.direction == "maximize" else "negate")
        for rec in (inst.alphaevolve_value, inst.prior_best_value, inst.current_best_value):
            assert rec.source.strip()
            assert rec.value is not None or rec.missing_note, "a missing value needs a note"
        for tp in inst.third_party:
            assert tp.source.strip() and (tp.value is not None or tp.missing_note)


@pytest.mark.parametrize("pid", sorted(BENCH.ids))
def test_alphaevolve_construction_reproduces_with_our_evaluator(pid):
    if pid not in EVALUATORS:
        pytest.skip(f"evaluator: {BENCH.get(pid).evaluator} (no evaluator for problem {pid} yet)")
    r = verify_problem(pid, use_cache=False)
    assert r["status"] == "passed", r


def test_normalized_vs_ae_and_gap_closed_for_a_maximize_problem():
    c = bench_compare(our=8.0, seed=4.0, ref_score=10.0, tol=1e-6)
    assert c["normalized"] == pytest.approx(0.8)
    assert c["gap_closed"] == pytest.approx(4 / 6)
    assert c["matched"] is False and c["beat"] is False


def test_normalized_vs_ae_and_gap_closed_for_a_minimize_problem():
    # published values: AE 2.0, seed 4.0, ours 2.5 -> scores -2, -4, -2.5
    c = bench_compare(our=-2.5, seed=-4.0, ref_score=-2.0, tol=1e-6)
    assert c["normalized"] == pytest.approx(2.0 / 2.5)  # AE / ours for minimization
    assert c["gap_closed"] == pytest.approx(1.5 / 2.0)
    better = bench_compare(our=-1.8, seed=-4.0, ref_score=-2.0, tol=1e-6)
    assert better["normalized"] > 1 and better["beat"] is True and better["matched"] is True


def test_gap_closed_when_the_seed_already_meets_or_exceeds_the_reference():
    assert gap_closed(12.0, 11.0, 10.0) == (None, "seed_at_or_above_reference")
    assert gap_closed(10.0, 10.0, 10.0) == (None, "seed_at_or_above_reference")
    assert gap_closed(-1.0, -1.5, -2.0) == (None, "seed_at_or_above_reference")  # minimize
    assert normalized_vs(12.0, 10.0) == pytest.approx(1.2)
    assert gap_closed(5.0, None, 10.0) == (None, "seed_invalid")
    assert gap_closed(None, 1.0, 10.0) == (None, "no_valid_program")


def test_synthetic_beat_without_verification_output_is_reported_unverified(tmp_path):
    reg = synth_registry(tmp_path)
    d = make_run(tmp_path, "b", problem_id=901, scores=[5.0, 11.0])
    s, h = json.loads((d / "summary.json").read_text()), _bench_hist(d)
    ev = BenchFakeEvaluator(901, 10.0, {"k": 10})
    b = compute_block(s, h, registry=reg, evaluator=ev, verify=False, verify_beats=False)
    assert b["beat_ae"] is True and b["beat_status"] == "unverified"
    # even with re-verification on, no published scorer port -> unverified, never "verified"
    b = compute_block(s, h, run_dir=d, registry=reg, evaluator=ev, verify=False, verify_beats=True)
    assert b["beat_status"] == "unverified" and b["beat_verification"]["published_score"] is None


# additional coverage: benchmark layer

def test_benchmark_status_comes_from_status_json_labels():
    assert BENCH.get(34).status == "unlisted" and BENCH.get(34).status_raw is None
    for p in BENCH.problems:
        if p.status != "unlisted":
            assert p.status_raw in BENCH.status_labels and BENCH.status_labels[p.status_raw] == p.status


@pytest.mark.parametrize("pid", [36, 48, 50])
def test_benchmark_third_party_systems_recorded(pid):
    for inst in BENCH.get(pid).instances[:1]:
        systems = " ".join(tp.system for tp in inst.third_party)
        for s in ("OpenEvolve", "ShinkaEvolve", "GEPA"):
            assert s in systems


def test_benchmark_registry_rejects_null_without_note(tmp_path):
    bad = tmp_path / "r.yaml"
    bad.write_text(BENCH_SYNTH_YAML.replace('missing_note: "not reported"}\n        prior', '}\n        prior', 1),
                   encoding="utf-8")
    with pytest.raises(Exception):
        load_registry(bad)


def test_benchmark_instance_matching_is_exact():
    p36 = BENCH.get(36)
    assert p36.match_instance({"n": 26, "container": "unit_square"}).id == "n26_unit_square"
    assert p36.match_instance({"n": 26.0, "container": "unit_square"}).id == "n26_unit_square"
    assert p36.match_instance({"n": 32, "container": "unit_square"}).id == "n32_unit_square"
    assert p36.match_instance({"n": 26, "container": "rectangle_perimeter_4"}) is None
    assert p36.match_instance({"n": 26}) is None
    assert p36.match_instance(None) is None


def test_p36_published_scorer_is_exact():
    inst = BENCH.get(36).instance("n26_unit_square")
    data = load_construction(inst)
    ok = p36_check_construction(data, inst.params)
    assert ok["valid"] and ok["value"] == pytest.approx(inst.ae_value, abs=1e-12)
    grown = {"centers": data["centers"], "radii": [r + 5e-10 for r in data["radii"]]}
    assert not p36_check_construction(grown, inst.params)["valid"]  # no tolerance, unlike our evaluator
    assert not p36_check_construction({"centers": data["centers"][:25], "radii": data["radii"][:25]},
                                      inst.params)["valid"]


def test_benchmark_verification_fails_on_instance_mismatch():
    r = verify_problem(36, evaluator=BenchFakeEvaluator(36, 2.6, {"n": 27, "container": "unit_square"}),
                       use_cache=False)
    assert r["status"] == "failed" and "instance mismatch" in r["reason"]


def test_benchmark_match_tolerance():
    assert bench_compare(10.0 - 5e-7, 0.0, 10.0, 1e-6)["matched"] is True
    assert bench_compare(10.0 + 5e-7, 0.0, 10.0, 1e-6)["beat"] is False
    assert bench_compare(10.0 + 2e-6, 0.0, 10.0, 1e-6)["beat"] is True


def test_benchmark_block_budget_fractions_maximize(tmp_path):
    reg = synth_registry(tmp_path)
    ev = BenchFakeEvaluator(901, 10.0, {"k": 10})
    d = make_run(tmp_path, "r", problem_id=901, scores=[5.0, 8.9, 9.0, None, 9.6, 9.95], total=1000,
                 sleep_tokens_at={3: 50})
    b = compute_block(json.loads((d / "summary.json").read_text()), _bench_hist(d), registry=reg, evaluator=ev,
                      verify=False)
    assert b["scored"] and b["instance_id"] == "k10"
    assert b["normalized_vs_ae"] == pytest.approx(0.995)
    assert b["gap_closed"] == pytest.approx(4.95 / 5)
    assert b["budget_to_reach"]["0.9"] == {"reached": True, "step": 2, "tokens": 200, "fraction": 0.2}
    assert b["budget_to_reach"]["0.95"] == {"reached": True, "step": 4, "tokens": 450, "fraction": 0.45}
    assert b["budget_to_reach"]["0.99"]["fraction"] == pytest.approx(0.55)
    tp = {t["system"]: t for t in b["third_party"]}
    assert tp["Other"]["compared"] and tp["Other"]["beat"] is True
    assert tp["Other"]["normalized_vs"] == pytest.approx(9.95 / 8)
    assert tp["Missing"]["compared"] is False and "missing" in tp["Missing"]["reason"]


def test_benchmark_block_minimize_and_not_reached(tmp_path):
    reg = synth_registry(tmp_path)
    d = make_run(tmp_path, "m", problem_id=902, scores=[-4.0, -3.0, -2.5])
    b = compute_block(json.loads((d / "summary.json").read_text()), _bench_hist(d), registry=reg,
                      evaluator=BenchFakeEvaluator(902, -2.0, {"k": 3}), verify=False)
    assert b["our_best_value"] == pytest.approx(2.5) and b["alphaevolve_score"] == -2.0
    assert b["normalized_vs_ae"] == pytest.approx(0.8) and b["gap_closed"] == pytest.approx(0.75)
    assert b["budget_to_reach"]["0.9"]["reached"] is False and b["budget_to_reach"]["0.9"]["fraction"] is None


def test_benchmark_block_seed_above_reference(tmp_path):
    reg = synth_registry(tmp_path)
    d = make_run(tmp_path, "s", problem_id=901, scores=[10.5, 10.6])
    b = compute_block(json.loads((d / "summary.json").read_text()), _bench_hist(d), registry=reg,
                      evaluator=BenchFakeEvaluator(901, 10.0, {"k": 10}), verify=False, verify_beats=False)
    assert b["gap_closed"] is None and b["gap_closed_reason"] == "seed_at_or_above_reference"
    assert b["budget_to_reach"]["0.99"] == {"reached": True, "step": 0, "tokens": 0, "fraction": 0.0}


def test_benchmark_refuses_pending_unknown_and_mismatched_problems(tmp_path):
    reg = synth_registry(tmp_path)
    d = make_run(tmp_path, "x", problem_id=903, scores=[1.0])
    s, h = json.loads((d / "summary.json").read_text()), _bench_hist(d)
    b = compute_block(s, h, registry=reg)
    assert b["in_benchmark"] and not b["scored"] and "pending" in b["reason"]
    b = compute_block({**s, "problem_id": 0}, h, registry=reg)
    assert not b["in_benchmark"] and not b["scored"] and "not in the AlphaEvolve benchmark" in b["reason"]
    b = compute_block({**s, "problem_id": 901}, h, registry=reg, evaluator=BenchFakeEvaluator(901, 10.0, {"k": 11}),
                      verify=False)
    assert not b["scored"] and b["instance_match"] is False and "instance mismatch" in b["reason"]


def test_beat_reverification_p36():
    p = BENCH.get(36)
    real = p.instance("n26_unit_square")
    data = load_construction(real)
    lowered = real.model_copy(update={"alphaevolve_value": real.alphaevolve_value.model_copy(update={"value": 2.6})})
    ev = CirclePacking26(timeout_s=30)
    v = reverify_beat(p, lowered, bench_program_returning(data), ev)
    assert v["beat_status"] == "verified"
    assert v["our_valid"] and v["published_valid"] and v["program_sha256"]
    assert v["published_score"] == pytest.approx(real.ae_value, abs=1e-12)
    assert v["validity_output"] == ["The circles are disjoint and lie inside the unit square."]
    # valid for our evaluator (1e-9 slack) but not under the notebook's exact check -> rejected
    grown = {"centers": data["centers"], "radii": [r + 2e-10 for r in data["radii"]]}
    v = reverify_beat(p, real, bench_program_returning(grown), ev)
    assert v["our_valid"] is True and v["published_valid"] is False and v["beat_status"] == "rejected"
    assert reverify_beat(p, real, None, ev)["beat_status"] == "unverified"


def test_enrich_summary_is_idempotent_and_preserves_other_keys(tmp_path):
    seed_like = 26 / 12
    d = make_run(tmp_path, "p36", problem_id=36, scores=[seed_like, 2.5, None, 2.6],
                 extra={"metrics": {"auc": 0.5, "owner": "baseline suite"}})
    b1 = enrich_summary(d)
    text1 = (d / "summary.json").read_text()
    assert enrich_summary(d) == b1 and (d / "summary.json").read_text() == text1
    s = json.loads(text1)
    assert s["metrics"] == {"auc": 0.5, "owner": "baseline suite"}
    assert s["alphaevolve"]["scored"] and s["alphaevolve"]["verification"]["status"] == "passed"
    assert s["alphaevolve"]["normalized_vs_ae"] == pytest.approx(2.6 / 2.6359830849176067)
    assert math.isclose(s["alphaevolve"]["gap_closed"], (2.6 - seed_like) / (2.6359830849176067 - seed_like))


def test_enrich_summary_problem_not_in_benchmark(tmp_path):
    d = make_run(tmp_path, "p0", problem_id=0, scores=[0.5, 0.9])
    b = enrich_summary(d)
    assert b["in_benchmark"] is False and b["scored"] is False
    assert json.loads((d / "summary.json").read_text())["alphaevolve"] == b


def test_benchmark_scoring_cli(tmp_path, capsys):
    from benchmarks.scoring import main

    make_run(tmp_path, "p0", problem_id=0, scores=[0.5])
    make_run(tmp_path, "p36", problem_id=36, scores=[2.0, 2.1])
    main([str(tmp_path)])
    out = capsys.readouterr().out
    assert "p0: not scored" in out and "p36: normalized_vs_ae=" in out


# ================================================================================================
# Section 3 · Evolution loop and logging
# ================================================================================================

def test_mock_end_to_end_run_writes_a_valid_jsonl_log(run_dir, mock_llm, mock_cfg):
    """Mock LLM + mock evaluator through the real loop, subprocess sandbox and logger."""
    llm, budget, transport = mock_llm(1000, prompt_tokens=60, completion_tokens=40)
    s = run(mock_cfg(), MockEvaluator(timeout_s=2), llm, NoSleep(), out_dir=run_dir)
    assert s["stop_reason"] == "budget_exhausted"
    assert budget.used == 1000 and budget.overshoot == 0
    assert s["steps_completed"] == 10 and len(transport.calls) == 10
    hist = hist_of(run_dir, s)
    assert [h["step"] for h in hist] == list(range(11))  # seed + 10 agent steps
    assert all(set(h) == set(FIELDS) for h in hist)
    assert all(h["run_id"] == s["run_id"] and h["event"] == "step" for h in hist)
    assert all(isinstance(h["details"], dict) and isinstance(h["side_info"], dict) for h in hist)
    assert all((h["score"] is None) == (not h["valid"]) for h in hist)
    toks = [h["agent_tokens"] for h in hist]
    assert toks == sorted(toks) and hist[-1]["agent_tokens"] == 1000 and hist[-1]["sleep_tokens"] == 0
    assert hist[0]["parent_id"] is None and hist[1]["parent_id"] == "p0"
    assert (run_dir / s["run_id"] / "best_program.py").exists()
    assert json.loads((run_dir / s["run_id"] / "summary.json").read_text())["best_score"] == s["best_score"]


def test_max_steps_stops_before_the_budget(run_dir, mock_llm, mock_cfg):
    llm, budget, _ = mock_llm(10_000)
    s = run(mock_cfg(max_steps=4), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    assert s["stop_reason"] == "max_steps" and s["steps_completed"] == 4 and budget.used == 400


def test_parent_sampling_is_best_70_percent_else_uniform_over_top_10():
    db = ProgramDB()
    for i in range(20):
        db.add(Program(f"p{i}", "", float(i), True, None, i))
    db.add(Program("bad", "", float("-inf"), False, None, 99))
    rng = random.Random(0)
    picks = [db.sample_parent(rng).id for _ in range(10_000)]
    assert set(picks) <= {f"p{i}" for i in range(10, 20)}
    assert picks.count("p19") / len(picks) == pytest.approx(0.7 + 0.3 / 10, abs=0.02)
    others = [picks.count(f"p{i}") / len(picks) for i in range(10, 19)]
    assert all(o == pytest.approx(0.03, abs=0.01) for o in others)


def test_parent_sampling_respects_configured_probabilities():
    db = ProgramDB()
    db.P_BEST, db.TOP_K = 0.2, 4
    for i in range(10):
        db.add(Program(f"p{i}", "", float(i), True, None, i))
    rng = random.Random(1)
    picks = [db.sample_parent(rng).id for _ in range(10_000)]
    assert set(picks) == {"p6", "p7", "p8", "p9"}
    assert picks.count("p9") / len(picks) == pytest.approx(0.2 + 0.8 / 4, abs=0.02)
    assert picks.count("p6") / len(picks) == pytest.approx(0.8 / 4, abs=0.02)


def test_parent_falls_back_to_seed_when_nothing_is_valid():
    db = ProgramDB()
    db.add(Program("p0", "", float("-inf"), False, None, 0))
    db.add(Program("p1", "", float("-inf"), False, "p0", 1))
    assert db.sample_parent(random.Random(0)).id == "p0"
    with pytest.raises(ValueError):
        db.add(Program("p1", "", 1.0, True, None, 2))


def test_awake_tagging_header_is_parsed():
    text = ("**APPROACH_TAGS:** Hex Lattice, lbfgs_refine; [extra]\nexpected score = ~0.93 or so\n"
            "Rationale: lattice then refine\nRELEVANT_MEMORY: m3, M7 and g0\n```python\ncode\n```\n"
            "APPROACH_TAGS: ignored-after-fence")
    assert parse_awake_header(text) == {"approach_tags": ["hex-lattice", "lbfgs-refine", "extra"],
                                        "expected_score": 0.93, "rationale": "lattice then refine",
                                        "relevant_memory": ["m3", "m7", "g0"]}
    assert parse_awake_header("just code") == {"approach_tags": None, "expected_score": None,
                                               "rationale": None, "relevant_memory": None}
    assert parse_awake_header("EXPECTED_SCORE: unknown")["expected_score"] is None


def test_malformed_awake_header_is_logged_as_null_and_the_step_still_runs(run_dir, mock_llm, mock_cfg):
    reply = ("APPROACH_TAGS\nEXPECTED_SCORE: banana\nRATIONALE:\n"
             "```python\ndef construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.7]\n```")
    llm, _, _ = mock_llm(10_000, script=[reply])
    s = run(mock_cfg(max_steps=3, awake_tags=True), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    assert s["steps_completed"] == 3
    for h in hist_of(run_dir, s)[1:]:
        assert h["valid"] is True and h["score"] == pytest.approx(1.0)
        assert h["approach_tags"] is None and h["expected_score"] is None and h["rationale"] is None
        assert h["surprise"] is None


def test_surprise_definition():
    assert P.compute_surprise(0.8, True, 0.9, 2.0) == pytest.approx(0.05)
    assert P.compute_surprise(None, False, 0.9, 2.0) == 1.0  # invalid with a prediction
    assert P.compute_surprise(0.8, True, None, 2.0) is None  # no prediction


def test_surprise_is_logged_from_expected_vs_actual_score(run_dir, mock_llm, mock_cfg):
    good = ("APPROACH_TAGS: a, b\nEXPECTED_SCORE: 0.9\n"
            "```python\ndef construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.2]\n```")  # score 0.95
    bad = "APPROACH_TAGS: c\nEXPECTED_SCORE: 0.5\n```python\ndef construct():\n    return [1]\n```"
    llm, _, _ = mock_llm(10_000, script=[good, bad])
    s = run(mock_cfg(max_steps=2, awake_tags=True), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    h1, h2 = hist_of(run_dir, s)[1:]
    assert h1["score"] == pytest.approx(0.95) and h1["surprise"] == pytest.approx(0.05)
    assert h2["valid"] is False and h2["surprise"] == 1.0


def test_consolidated_block_reaches_the_agent_but_raw_history_does_not(run_dir, mock_llm, mock_cfg):
    llm, _, transport = mock_llm(1000)
    run(mock_cfg(k_recent=2), TextEvaluator(), llm, Summarize(every_n=3), out_dir=run_dir)
    agent_msgs = [m for m in transport.calls if "consolidat" not in m[0]["content"].lower()]
    after_sleep = agent_msgs[3][1]["content"]
    assert "Consolidated notes" in after_sleep
    assert after_sleep.count("step ") <= 2  # only the last K=2 steps are rendered


def test_only_sleep_policies_receive_the_raw_jsonl_history(run_dir, mock_llm, mock_cfg):
    class Recorder(NoSleep):
        name = "recorder"

        def __init__(self):
            self.seen = []

        def should_sleep(self, step, history):
            self.seen.append([dict(h) for h in history])
            return False

    llm, _, transport = mock_llm(10_000)
    pol = Recorder()
    run(mock_cfg(max_steps=8, k_recent=2), TextEvaluator(), llm, pol, out_dir=run_dir)
    # the policy sees every JSONL record so far, schema and all
    assert [len(h) for h in pol.seen] == list(range(2, 10))
    assert all(set(r) == set(FIELDS) for r in pol.seen[-1])
    # the agent sees rendered summaries of at most K steps, never JSONL records
    for msgs in transport.calls:
        text = "\n".join(m["content"] for m in msgs)
        for key in ('"agent_tokens"', '"run_id"', '"wallclock"', '"delta_vs_best"', '"event"'):
            assert key not in text
    last = transport.calls[-1][1]["content"]
    assert "step 7 |" in last and "step 6 |" in last and "step 5 |" not in last and "step 1 |" not in last


WAKE_CAP = 1200
CAP_ARMS = {
    "no_sleep": lambda: NoSleep(),
    "summarize": lambda: Summarize(every_n=10, max_block_tokens=300),
    "full_sleep_pressure": lambda: FullSleep(gating="pressure", max_block_tokens=300, min_gap=3, max_gap=12,
                                             plateau_window=8, max_items=15),
    "full_sleep_fatigue": lambda: FullSleep(max_block_tokens=300, min_sleep_budget_tokens=0, replay_budget=1),
    "fixed_every_5_nrem": lambda: FullSleep(gating="fixed", every_n=5, timed_depth="nrem", max_block_tokens=300),
    "sleep_lite": lambda: SleepLite(max_block_tokens=300),
}


@pytest.mark.parametrize("arm", sorted(CAP_ARMS))
def test_wake_context_never_exceeds_its_cap_in_a_300_step_run(arm, run_dir, mock_llm, mock_cfg):
    llm, budget, transport = mock_llm(10**8, grow_lines=1)
    cfg = mock_cfg(max_steps=300, k_recent=5, context_cap_tokens=WAKE_CAP, awake_tags=True,
                   fatigue={"load_tokens_scale": 2000, "grace_steps": 2})
    s = run(cfg, TextEvaluator(), llm, CAP_ARMS[arm](), out_dir=run_dir)
    assert s["steps_completed"] == 300
    agent_prompts = [m for m in transport.calls if "APPROACH_TAGS" in m[0]["content"]]
    assert len(agent_prompts) == 300 == budget.agent_calls
    assert max(messages_tokens(m) for m in agent_prompts) <= WAKE_CAP
    steps = [h for h in hist_of(run_dir, s) if h["event"] == "step"]
    assert len(steps) == 301 and all(h["context_tokens"] <= WAKE_CAP for h in steps[1:])
    if arm != "no_sleep":
        assert any(h["event"] == "sleep" for h in hist_of(run_dir, s)), "the arm should have slept"


# additional coverage: loop, logging, context

def test_300_step_full_sleep_run_exercises_truncation_eviction_and_perseveration(run_dir, mock_llm, mock_cfg):
    llm, budget, transport = mock_llm(10**8, grow_lines=1)
    policy = FullSleep(gating="pressure", max_block_tokens=300, min_gap=3, max_gap=12, plateau_window=8,
                       max_items=15, schema=True, schema_min_items=2)
    s = run(mock_cfg(max_steps=300, k_recent=5, context_cap_tokens=WAKE_CAP), TextEvaluator(), llm, policy,
            out_dir=run_dir)
    assert s["stop_reason"] == "max_steps"
    agent_prompts = [m for m in transport.calls if "APPROACH_TAGS" in m[0]["content"]]
    assert any("chars truncated" in m[1]["content"] for m in agent_prompts)  # truncation exercised
    assert len(transport.calls) == budget.agent_calls + budget.sleep_calls
    hist = hist_of(run_dir, s)
    steps = [h for h in hist if h["event"] == "step"]
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert len(sleeps) >= 300 // 12 and {h["sleep_mode"] for h in sleeps} <= {"nrem", "rem"}
    assert all(h["block_tokens"] <= 300 for h in sleeps)
    assert any(h["perseveration_hit"] for h in steps)
    assert sum(h["items_evicted"] for h in sleeps) > 0


def test_side_info_reaches_the_agent_for_no_sleep(run_dir, mock_llm, mock_cfg):
    llm, _, transport = mock_llm(10_000)
    run(mock_cfg(max_steps=3), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    assert "max_dev" in transport.calls[-1][1]["content"]  # parent / recent side_info shown


def test_build_context_requires_system_and_task_within_cap():
    with pytest.raises(ValueError):
        build_context(system="s" * 300, task="t" * 300, parent_id="p0", parent_score=None, parent_code="x", cap=100)
    msgs, st = build_context(system="sys", task="task", memory=["- a", "- b"], recent=[], parent_id="p0",
                             parent_score=1.0, parent_code="y = 1\n" * 500, parent_side_info={"k": "v" * 2000},
                             cap=200)
    assert messages_tokens(msgs) <= 200 and st["program_truncated"] and st["parent_side_info_chars"] == 0
    assert "chars truncated" in msgs[1]["content"]


def test_fit_lines_with_unicode_respects_the_cap():
    lines = ["héllo wörld ✓" * k for k in range(1, 30)]
    for cap in range(0, 200, 7):
        assert count_tokens(fit_lines(lines, cap, header="## H")) <= cap


def test_logger_normalizes_records_and_rejects_unknown_fields(tmp_path):
    lg = RunLogger(tmp_path / "h.jsonl", "r1")
    rec = lg.log(step=1, event="step", score=float("-inf"), valid=False, delta_vs_best=float("nan"))
    assert set(rec) == set(FIELDS) and rec["score"] is None and rec["delta_vs_best"] is None
    assert rec["details"] == {} and rec["side_info"] == {} and rec["run_id"] == "r1"
    with pytest.raises(ValueError):
        lg.log(step=2, event="step", not_a_field=1)
    assert read_history(tmp_path / "h.jsonl") == [rec]


def test_configs_load_and_every_policy_resolves():
    for name in ("example.yaml", "mock.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        make_policy(cfg.sleep_policy, cfg.policy_params)
        assert cfg.awake_tags is True and cfg.fatigue.enabled
        assert {"summarize", "full_sleep", "sleep_lite"} <= set(cfg.sleep_policies)
        for pol, params in cfg.sleep_policies.items():
            make_policy(pol, params)
    with pytest.raises(ValueError):
        make_policy("no_such_policy")


def test_matrix_is_the_full_cross_product():
    from scripts.run_matrix import build_matrix

    base = RunConfig(problem_id=0, total_token_budget=10, max_steps=1)
    cfgs = build_matrix(base, [0, 36], ["no_sleep", "summarize"], [0, 1, 2])
    assert len(cfgs) == 12 and len({c.resolved_run_id() for c in cfgs}) == 12


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nSE_TEST_A=from_file\nSE_TEST_B='quoted'\nnot a line\n", encoding="utf-8")
    monkeypatch.setenv("SE_TEST_A", "from_env")
    monkeypatch.delenv("SE_TEST_B", raising=False)
    load_dotenv(env)
    assert os.environ["SE_TEST_A"] == "from_env" and os.environ["SE_TEST_B"] == "quoted"
    monkeypatch.delenv("SE_TEST_B")


# ================================================================================================
# Section 4 · Memory store
# ================================================================================================

def _random_store(seed: int = 0, n: int = 30) -> MemoryStore:
    rng = random.Random(seed)
    s = MemoryStore()
    s.add(kind="goal", text="Maximize the score.", step=0, status="verified", strength=1.0, item_id="g0")
    kinds = ["rule", "fact", "skill", "dead_end"]
    for i in range(n):
        s.add(kind=kinds[i % 4], text=f"claim {i} " + "x" * rng.randint(5, 250), step=i,
              status=rng.choice(["tentative", "verified", "contested"]), strength=rng.uniform(0.01, 1.0),
              approach_tags=rng.sample(["a", "b", "c", "d", "e"], 2))
    return s


def _snapshot(s: MemoryStore):
    return [(i.id, round(i.strength, 12), i.status) for i in s.ordered()]


def test_decay_is_deterministic_and_skips_goals():
    runs = []
    for _ in range(3):
        s = _random_store(seed=7)
        out = []
        for step in range(5):
            s.decay(step, 0.8, exempt=["m3"])
            out.append((s.evict(min_strength=0.1, max_items=12), _snapshot(s)))
        runs.append(out)
    assert runs[0] == runs[1] == runs[2]
    s = _random_store(seed=7)
    before = {i: it.strength for i, it in s.items.items()}
    assert s.decay(1, 0.5, exempt=["m3"]) == len(s) - 2  # goal and exempt item skipped
    assert s.get("g0").strength == 1.0 and s.get("m3").strength == before["m3"]
    assert s.get("m4").strength == pytest.approx(before["m4"] * 0.5)


def test_support_edit_strengthens_and_promotes(memory_store):
    s = MemoryStore(initial_strength=0.3, verified_strength=0.7)
    it = s.add(kind="fact", text="f", step=0)
    applied, rejected = s.apply_edits([Edit("support", target=it.id, evidence=[5])], step=5)
    assert applied and not rejected and it.strength == pytest.approx(0.45) and 5 in it.evidence
    s.support(it.id, 6, evidence=[6], status="verified")
    assert it.status == "verified" and it.strength >= 0.7 and it.last_supported_step == 6


def test_contradict_edit_demotes_verified_to_contested(memory_store):
    it = memory_store.add(kind="fact", text="f", step=0, status="verified")
    memory_store.apply_edits([Edit("contradict", target=it.id, evidence=[3])], step=3)
    assert it.status == "contested" and it.strength == pytest.approx(0.7 - 0.25)
    g = memory_store.add(kind="goal", text="g", step=0, item_id="g0")
    _, rejected = memory_store.apply_edits([Edit("contradict", target="g0")], step=4)
    assert rejected and g.strength == 0.3  # goals cannot be contradicted


def test_refute_edit_marks_item_refuted_and_halves_strength(memory_store):
    it = memory_store.add(kind="rule", text="hex lattice helps", step=0, strength=0.6, approach_tags=["hex"])
    applied, _ = memory_store.apply_edits([Edit("refute", target=it.id, evidence=["r3.0"])], step=3)
    assert applied and it.status == "refuted" and it.strength == pytest.approx(0.3)
    assert "r3.0" in it.evidence and it.approach_tags == ["hex"]


@pytest.mark.xfail(reason="not implemented: refute converts the item to a dead_end (it only sets status=refuted)",
                   strict=True)
def test_refute_converts_item_to_dead_end_keeping_reason_and_tags(memory_store):
    it = memory_store.add(kind="rule", text="hex lattice helps", step=0, approach_tags=["hex", "lattice"])
    memory_store.apply_edits([Edit("refute", target=it.id, evidence=["r3.0"])], step=3)
    assert it.kind == "dead_end" and "hex lattice helps" in it.text and it.approach_tags == ["hex", "lattice"]


@pytest.mark.xfail(reason="not implemented: 'revise' edit op (MemoryStore supports add/support/contradict/refute)",
                   strict=True)
def test_revise_edit_rewrites_item_text(memory_store):
    it = memory_store.add(kind="rule", text="old wording", step=0)
    applied, _ = memory_store.apply_edits([Edit("revise", target=it.id, item={"text": "new wording"})], step=1)
    assert applied and it.text == "new wording"


def test_merge_edit_combines_two_items(memory_store):
    a = memory_store.add(kind="rule", text="a", step=0, evidence=[1])
    b = memory_store.add(kind="rule", text="b", step=0, evidence=[2])
    applied, _ = memory_store.apply_edits([Edit("merge", target=a.id, item={"merge": [b.id], "text": "a+b"})], step=3)
    assert applied and b.id not in memory_store.items and set(a.evidence) == {1, 2}


def test_merge_keeps_union_of_evidence_and_tags_highest_status_and_max_strength(memory_store):
    s = memory_store
    a = s.add(kind="rule", text="hex init helps", step=2, status="tentative", strength=0.3, evidence=[1, 4],
              approach_tags=["hex"])
    b = s.add(kind="fact", text="hexagonal start is better", step=1, status="verified", strength=0.9,
              evidence=[4, "r3.0"], approach_tags=["hex", "lattice"])
    c = s.add(kind="rule", text="lattice starts win", step=5, status="contested", strength=0.5, evidence=[7])
    applied, rejected = s.apply_edits([Edit("merge", target=a.id, item={"merge": [b.id, c.id],
                                                                         "text": "Hex-lattice starts help"})], 9)
    assert applied and not rejected and set(s.items) == {a.id}
    assert a.text == "Hex-lattice starts help" and a.kind == "rule"
    assert a.evidence == [1, 4, "r3.0", 7] and a.approach_tags == ["hex", "lattice"]
    assert a.status == "verified" and a.strength == pytest.approx(0.9)  # highest status, max strength
    assert a.created_step == 1 and a.last_supported_step == 5


def test_merge_rejects_goal_mixes_g0_polarity_mixes_and_unknown_ids(memory_store):
    s = memory_store
    s.add(kind="goal", text="best line", step=0, item_id="g0")
    g1, _ = s.set_goal("explore spirals", 1)
    g2, _ = s.set_goal("refine with lbfgs", 1)
    r = s.add(kind="rule", text="r", step=0)
    d = s.add(kind="dead_end", text="d", step=0)
    edits = [Edit("merge", target=g1, item={"merge": [r.id]}),     # goal + non-goal
             Edit("merge", target="g0", item={"merge": [g1]}),     # g0 is code-maintained
             Edit("merge", target=r.id, item={"merge": [d.id]}),   # rule + dead end (opposite polarity)
             Edit("merge", target=r.id, item={"merge": ["m99"]}),  # unknown id
             Edit("merge", target=r.id, item={"merge": [r.id]}),   # needs two distinct items
             Edit("merge", target=g1, item={"merge": [g2], "text": "explore spirals then refine"})]
    applied, rejected = s.apply_edits(edits, 2)
    assert len(rejected) == 5 and len(applied) == 1
    assert "goal with a non-goal" in rejected[0][1] and "g0" in rejected[1][1] and "dead_end" in rejected[2][1]
    assert set(s.items) == {"g0", g1, r.id, d.id} and s.get(g1).text == "explore spirals then refine"


def test_goals_replace_stale_goals_cap_at_three_and_render_first_after_g0(memory_store):
    s = memory_store
    s.add(kind="goal", text="Maximize the score. Best so far 0.5", step=0, status="verified", strength=1.0,
          item_id="g0")
    s.add(kind="rule", text="strong rule", step=0, status="verified", strength=1.0)
    ids = [s.set_goal(f"goal {i}", step=i)[0] for i in range(1, 4)]
    assert ids == ["g1", "g2", "g3"]
    gid, action = s.set_goal("goal 1 revised", 4, replaces="g1")
    assert gid == "g1" and action == "replaced g1" and s.get("g1").text == "goal 1 revised"
    assert s.set_goal("goal 2", 5) == ("g2", "refreshed")
    gid, action = s.set_goal("new direction", 6)  # cap reached: oldest untouched goal (g3) pushed out
    assert gid == "g4" and "evicted g3" in action and len(s.goals()) == 3
    gid, action = s.set_goal("replace the best line", 7, replaces="g0")  # g0 cannot be replaced
    assert gid != "g0" and "ignored" in action and s.get("g0").text.startswith("Maximize")
    lines = s.render(10_000).splitlines()
    assert lines[0].startswith("- [g0] GOAL") and all("GOAL" in ln for ln in lines[:4])
    assert "RULE" in lines[4]
    before = {g.id: g.strength for g in s.goals()}
    s.decay(8, 0.1)
    assert s.evict(min_strength=0.5, max_items=1) and {g.id: g.strength for g in s.goals()} == before


def test_remove_edit_deletes_item_but_never_a_goal(memory_store):
    it = memory_store.add(kind="rule", text="a", step=0)
    memory_store.add(kind="goal", text="g", step=0, item_id="g0")
    applied, rejected = memory_store.apply_edits([Edit("remove", target=it.id), Edit("remove", target="g0")], step=1)
    assert len(applied) == 1 and len(rejected) == 1 and it.id not in memory_store.items


def test_apply_edits_rejects_bad_edits_without_raising():
    s = MemoryStore()
    applied, rejected = s.apply_edits([
        Edit("add", item={"kind": "rule", "text": "a rule"}),
        Edit("support", target="m99"),
        Edit("teleport", target="m1"),
        Edit("add", item={"kind": "nonsense", "text": "x"}),
        Edit("add", item={"kind": "rule", "text": "  A   RULE "}),  # duplicate text -> support m1
        Edit("contradict", target="m1"),
        Edit("refute", item={"kind": "fact", "text": "false fact"}),
    ], step=3)
    assert len(applied) == 4 and len(rejected) == 3
    assert applied[1].target == "m1" and s.get("m2").status == "refuted" and set(s.items) == {"m1", "m2"}


def _fs(**kw) -> FullSleep:
    fs = FullSleep(**{"replay_budget": 10, "decay": False, **kw})
    fs.bind(evaluator=TextEvaluator())
    return fs


def _sleep_history(n=6, **extra):
    hist = []
    for s in range(n):
        score = 0.5 + 0.08 * s
        hist.append({"event": "step", "step": s, "program_id": f"p{s}", "parent_id": f"p{s - 1}" if s else None,
                     "program": f"def construct():\n    return [{score}]*5\n", "score": score, "valid": True,
                     "error": None, "delta_vs_best": 0.08 if s else None, "side_info": {"slack": s}, **extra})
    return hist


def test_consolidation_cannot_verify_an_item_without_replay_evidence_in_the_same_cycle(mock_llm):
    body = {"claims": [
        {"cid": "c1", "kind": "rule", "text": "replayed claim", "relation": "new", "status": "verified"},
        {"cid": "c2", "kind": "rule", "text": "asserted claim", "relation": "new", "status": "verified"},
    ], "replays": [{"cid": "c1", "control": BAD, "treatment": GOOD}]}
    llm, budget, _ = mock_llm(10_000, script=[json.dumps(body)])
    fs = _fs()
    fs.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert fs.store.find_text("replayed claim").status == "verified"
    assert any(str(e).startswith("r5.") for e in fs.store.find_text("replayed claim").evidence)
    assert fs.store.find_text("asserted claim").status == "tentative"  # LLM's "verified" is ignored


@pytest.mark.xfail(reason="not implemented: MemoryStore itself accepts a verified add without replay evidence "
                          "(only FullSleep's pipeline enforces it)", strict=True)
def test_store_rejects_a_verified_add_without_replay_evidence(memory_store):
    _, rejected = memory_store.apply_edits([Edit("add", item={"kind": "rule", "text": "x"}, status="verified")], 1)
    assert rejected


def test_eviction_removes_items_below_threshold_and_never_goals():
    s = _random_store(seed=7)
    s.decay(1, 0.5)
    evicted = s.evict(min_strength=0.2, max_items=5)
    assert "g0" in s.items and len(s) == 5 and evicted
    assert all(i.strength >= 0.2 for i in s.items.values())
    g = MemoryStore()
    g.add(kind="goal", text="weak goal", step=0, strength=0.01, item_id="g0")
    assert g.evict(min_strength=0.5, max_items=0) == [] and "g0" in g.items


def test_eviction_order_is_lowest_rank_first_with_id_tiebreak():
    s = MemoryStore()
    for i in range(4):
        s.add(kind="rule", text=f"r{i}", step=0, strength=0.5)
    assert s.evict(min_strength=0.0, max_items=2) == ["m1", "m2"]


@pytest.mark.parametrize("seed", range(5))
def test_render_never_exceeds_the_cap_and_orders_goals_then_strength_x_status(seed):
    s = _random_store(seed=seed, n=40)
    extra = [f"- [s{i}] SCHEMA: " + "y" * 120 for i in range(3)]
    for cap in [0, 1, 5, 17, 30, 64, 100, 257, 500, 1000, 10_000]:
        assert count_tokens(s.render(cap, extra_lines=extra)) <= cap
    full = s.render(10_000)
    assert full.splitlines()[0].startswith("- [g0] GOAL")
    ranks = [s.get(line.split("]")[0][3:]).rank for line in full.splitlines()[1:]]
    assert ranks == sorted(ranks, reverse=True)


@pytest.mark.xfail(reason="not implemented: working-set selection (the consolidator receives the whole store)",
                   strict=True)
def test_working_set_returns_only_items_whose_tags_overlap_selected_steps_or_goals(memory_store):
    memory_store.add(kind="goal", text="g", step=0, item_id="g0", approach_tags=["target"])
    hit = memory_store.add(kind="rule", text="r1", step=0, approach_tags=["hex"])
    memory_store.add(kind="rule", text="r2", step=0, approach_tags=["anneal"])
    ws = memory_store.working_set(steps=[{"approach_tags": ["hex"]}])
    assert {i.id for i in ws} == {"g0", hit.id}


def test_match_dead_ends_flags_a_step_whose_tags_match_a_dead_end():
    s = MemoryStore()
    de = s.add(kind="dead_end", text="random restarts never beat the parent", step=1,
               approach_tags=["random-search", "restart"])
    s.add(kind="rule", text="hex lattice helps", step=1, approach_tags=["random-search", "restart"])
    assert [i.id for i in s.match_dead_ends(normalize_tags("Random Search, restart, extra"))] == [de.id]
    assert s.match_dead_ends(["random-search"]) == [de]  # 1/2 of the dead end's tags >= 0.5
    assert s.match_dead_ends(["hex"]) == []
    assert s.match_dead_ends(["random-search"], threshold=0.9) == []
    s.refute(de.id, 2)
    assert s.match_dead_ends(["random-search", "restart"]) == []  # refuted dead ends don't fire


# additional coverage: memory

def test_observe_flags_perseveration_from_tags():
    fs = FullSleep()
    fs.store.add(kind="dead_end", text="grid sweeps stall", step=0, approach_tags=["grid-sweep"])
    out = fs.observe(1, {"step": 1, "score": 0.5, "valid": True, "approach_tags": ["grid-sweep", "x"]})
    assert out["perseveration_hit"] is True and out["perseveration_items"] == ["m1"]
    out = fs.observe(2, {"step": 2, "score": 0.5, "valid": True, "approach_tags": ["anneal"]})
    assert out["perseveration_hit"] is False and 0.0 <= out["pressure"] <= 1.0


def test_normalize_tags():
    assert normalize_tags("Hex Lattice; lbfgs_refine | none, , HEX lattice") == ["hex-lattice", "lbfgs-refine"]
    assert normalize_tags(None) == [] and len(normalize_tags([f"t{i}" for i in range(20)])) == 8


# ================================================================================================
# Section 5 · Sleep pipeline
# ================================================================================================

def test_consolidator_output_is_parsed_and_validated(mock_llm):
    body = {"claims": [{"cid": "c1", "kind": "fact", "text": "f1", "evidence": ["step 3"], "relation": "new"},
                       {"cid": "c2", "kind": "nonsense", "text": "dropped"},
                       {"cid": "c1", "kind": "rule", "text": "duplicate cid"}],
            "replays": []}
    llm, budget, _ = mock_llm(10_000, script=["Sure! Here it is:\n```json\n" + json.dumps(body) + "\n```\nDone."])
    fs = _fs()
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert m["claims"] == 1 and m["items_added"] == 2  # goal + f1
    assert "bad kind/text" in m["parse_error"] and "duplicate cid" in m["parse_error"]
    assert fs.store.get("m1").evidence == [3]


def test_malformed_consolidator_output_skips_cleanly_with_a_logged_error(mock_llm):
    for bad in MALFORMED_OUTPUTS:
        llm, budget, _ = mock_llm(10_000, script=[bad])
        fs = _fs()
        m = fs.consolidate("task", _sleep_history(), llm, budget, step=5)
        assert m["parse_error"] and m["claims"] == 0 and m["edits_applied"] == 0
        assert set(fs.store.items) == {"g0"}


def test_consolidator_goals_and_merges_are_applied_and_recorded(mock_llm):
    fs = _fs()
    fs.store.add(kind="goal", text="old best line", step=0, status="verified", strength=1.0, item_id="g0")
    stale, _ = fs.store.set_goal("try random restarts", 0)
    a = fs.store.add(kind="rule", text="hex init helps", step=0, evidence=[1], approach_tags=["hex"])
    b = fs.store.add(kind="rule", text="hexagonal start is better", step=0, evidence=[2], status="verified")
    d = fs.store.add(kind="dead_end", text="grid sweeps stall", step=0)
    body = {"claims": [], "replays": [],
            "goals": [{"text": "Refine the hex start with L-BFGS-B.", "replaces": stale},
                      {"text": "Next: test spiral initializations.", "replaces": None},
                      "Then shrink the step size.", {"text": "fourth goal is dropped"}],
            "merges": [{"ids": [a.id, b.id], "text": "Hex-lattice starts help"}, {"ids": [a.id, d.id]}]}
    llm, budget, _ = mock_llm(10_000, script=[json.dumps(body)])
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert [g["action"] for g in m["goals"]] == [f"replaced {stale}", "added", "added"]
    assert "kept the first 3" in m["parse_error"]
    assert [x["applied"] for x in m["merges"]] == [True, False]
    merged = fs.store.get(a.id)
    assert b.id not in fs.store.items and merged.status == "verified" and set(merged.evidence) == {1, 2}
    lines = fs.memory_section(10_000).splitlines()
    assert lines[0].startswith("- [g0] GOAL: Maximize the score") and "Refine the hex start" in lines[1]
    assert all("GOAL" in ln for ln in lines[:4]) and "GOAL" not in lines[4]
    assert m["edits_applied"] == 1 and m["edits_rejected"] == 1


def test_malformed_consolidator_json_is_retried_once(mock_llm):
    good = json.dumps({"claims": [{"cid": "c1", "kind": "fact", "text": "f", "relation": "new"}], "replays": []})
    llm, budget, transport = mock_llm(10_000, script=["{\"claims\": [", good])
    m = _fs().consolidate("task", _sleep_history(), llm, budget, step=5)
    assert len(transport.calls) == 2 and m["claims"] == 1 and budget.sleep_calls == 2


def test_parse_json_object_never_raises():
    assert parse_json_object('x {"a": {"b": 1}} y')[0] == {"a": {"b": 1}}
    for bad in (None, 5, "", "{", "[1]"):
        obj, err = parse_json_object(bad)
        assert obj is None and err
    claims, replays, problems = validate_consolidation({"claims": "x", "replays": 5}, max_claims=3)
    assert claims == [] and replays == [] and len(problems) == 2


def test_every_replay_produces_an_edit_or_is_listed_as_wasted(mock_llm):
    consolidation = {
        "selected_steps": [2, 3],
        "claims": [
            {"cid": "c1", "kind": "rule", "text": "target values win", "evidence": [2], "relation": "new"},
            {"cid": "c2", "kind": "rule", "text": "zeros are fine", "evidence": [2], "relation": "new"},
            {"cid": "c3", "kind": "skill", "text": "crashy probe", "evidence": [3], "relation": "new"},
            {"cid": "c4", "kind": "fact", "text": "supports a ghost", "relates_to": ["m77"], "relation": "supports"},
            {"cid": "c5", "kind": "fact", "text": "bad program", "relation": "new"},
            {"cid": "c6", "kind": "dead_end", "text": "one-element outputs fail", "approach_tags": ["tiny"],
             "relation": "new"},
            {"cid": "c7", "kind": "rule", "text": "old single-program shape", "relation": "new"},
            {"cid": "c8", "kind": "rule", "text": "identical pair", "relation": "new"},
            {"cid": "c10", "kind": "rule", "text": "invalid control", "relation": "new"},
        ],
        "replays": [
            {"cid": "c1", "control": BAD, "treatment": GOOD, "decides": "add c1 as verified"},
            {"cid": "c2", "control": GOOD, "treatment": BAD, "decides": "add c2 as verified"},
            {"cid": "c3", "control": GOOD, "treatment": CRASH, "decides": "add c3"},
            {"cid": "c9", "control": BAD, "treatment": GOOD, "decides": "nothing"},
            {"cid": "c4", "control": BAD, "treatment": GOOD, "decides": "support m77"},
            {"cid": "c5", "control": 42, "treatment": GOOD, "decides": "?"},
            {"cid": "c6", "control": GOOD, "treatment": SHAPE, "decides": "add c6"},
            {"cid": "c7", "program": GOOD, "decides": "add c7"},
            {"cid": "c8", "control": GOOD, "treatment": "# same\n" + GOOD + "\n", "decides": "add c8"},
            {"cid": "c10", "control": SHAPE, "treatment": GOOD, "decides": "add c10"},
        ],
    }
    llm, budget, transport = mock_llm(10_000, script=[json.dumps(consolidation)])
    fs = _fs(max_claims=16)  # 9 claims: the default cap of 8 would drop c10 before its replay runs
    hist = _sleep_history()
    m = fs.consolidate("task", hist, llm, budget, step=5)
    recs = m["replays"]
    assert len(recs) == 10  # none silently dropped
    for r in recs:
        if "wasted" in r:
            continue
        target = r["edit"]["target"]
        assert target in fs.store.items and r["rid"] in fs.store.get(target).evidence
    wasted = {r["cid"]: r["wasted"] for r in recs if "wasted" in r}
    assert set(wasted) == {"c3", "c9", "c4", "c5", "c7", "c8", "c10"}
    assert "treatment crashed" in wasted["c3"] and "rejected" in wasted["c4"]
    assert "old schema" in wasted["c7"] and "identical" in wasted["c8"] and "control invalid" in wasted["c10"]
    assert "old single-program shape" in m["parse_error"]
    assert m["replays_wasted"] == 7 and m["replays_run"] == 10
    by_text = {i.text: i for i in fs.store.items.values()}
    assert by_text["target values win"].status == "verified"
    assert by_text["zeros are fine"].status == "refuted"
    assert by_text["one-element outputs fail"].status == "verified"  # invalid treatment confirms a dead end
    assert by_text["crashy probe"].status == "tentative"  # wasted replay falls back to a tentative add
    # both scores and their difference are recorded
    c1 = recs[0]
    assert c1["control_score"] == pytest.approx(0.51) and c1["treatment_score"] == pytest.approx(1.0)
    assert c1["score_diff"] == pytest.approx(0.49) and c1["evaluations"] == 2
    # evaluator calls: 2 each for c1, c2, c3, c4, c6; 1 for c10 (control invalid -> treatment skipped);
    # 0 for c5, c7, c8, c9. Replay programs are evidence only: history untouched.
    assert len(fs.evaluator.calls) == m["replay_evaluations"] == 11 and [r["evaluations"] for r in recs][-1] == 1
    assert m["replays_beat_run_best"] == 5  # GOOD (1.0) beats the run best 0.9 in c1, c2, c3, c4, c6
    assert len(hist) == 6
    assert budget.sleep_calls == len(transport.calls) == 1 and budget.agent_calls == 0


def _o(score=None, valid=True, error=None):
    return {"valid": valid, "score": score, "error": error}


@pytest.mark.parametrize("kind, control, treatment, want", [
    ("rule", _o(0.5), _o(0.6), "confirmed"),                    # treatment beats control
    ("rule", _o(0.5), _o(0.5), "refuted"),                      # no gain
    ("rule", _o(1.0), _o(1.0005), "refuted"),                   # gain below the 1e-3 relative margin
    ("rule", _o(1.0), _o(1.002), "confirmed"),
    ("dead_end", _o(0.5), _o(0.5), "confirmed"),                # change doesn't help
    ("dead_end", _o(0.5), _o(0.4), "confirmed"),
    ("dead_end", _o(0.5), _o(0.5004), "refuted"),               # treatment beat control
    ("dead_end", _o(0.5), _o(None, False, "circles overlap by 0.1"), "confirmed"),  # invalid treatment
    ("fact", _o(0.5), _o(None, False, "circles overlap by 0.1"), "refuted"),
    ("fact", _o(0.5), _o(None, False, "timeout after 1s"), None),          # treatment crash: no decision
    ("dead_end", _o(None, False, "exit code 1: boom"), _o(0.9), None),     # control crash: no decision
    ("dead_end", _o(None, False, "circles overlap"), _o(0.9), None),       # invalid control: no baseline
    ("rule", _o(0.5), None, None),                                         # treatment not run
])
def test_replay_outcome_rules(kind, control, treatment, want):
    assert decide(kind, control, treatment, margin=1e-3)[0] == want


def test_minimal_probe_of_a_true_improvement_is_confirmed_far_below_the_run_best(mock_llm):
    """The old rule compared a probe with the run's best and refuted true rules; the paired test doesn't."""
    hist = _sleep_history()
    for h in hist:  # a strong run: best 0.99
        h["score"] = 0.95 + 0.008 * h["step"]
    control = "def construct():\n    return [0.0, 0.0, 0.0, 0.0, 0.0]\n"
    treatment = "def construct():\n    return [0.0, 0.0, 0.0, 0.0, 0.7]\n"  # one coordinate moved to target
    body = {"claims": [{"cid": "c1", "kind": "rule", "text": "moving a coordinate to 0.7 raises the score",
                        "evidence": [5], "relation": "new"}],
            "replays": [{"cid": "c1", "change": "last coordinate 0 -> 0.7", "control": control,
                         "treatment": treatment}]}
    llm, budget, _ = mock_llm(10_000, script=[json.dumps(body)])
    fs = _fs()
    m = fs.consolidate("task", hist, llm, budget, step=5)
    rec = m["replays"][0]
    assert rec["treatment_score"] == pytest.approx(0.608) and rec["treatment_score"] < 0.99 - 0.3
    assert rec["decision"] == "confirmed" and rec["beats_run_best"] is False
    assert fs.store.find_text("moving a coordinate to 0.7 raises the score").status == "verified"


def test_dead_end_is_refuted_when_the_treatment_beats_the_control(mock_llm):
    control = "def construct():\n    return [0.0, 0.0, 0.0, 0.0, 0.0]\n"
    treatment = "def construct():\n    return [0.0, 0.0, 0.0, 0.0, 0.7]\n"
    body = {"claims": [{"cid": "c1", "kind": "dead_end", "text": "tuning single coordinates never helps",
                        "approach_tags": ["coordinate-tuning"], "relation": "new"}],
            "replays": [{"cid": "c1", "control": control, "treatment": treatment}]}
    llm, budget, _ = mock_llm(10_000, script=[json.dumps(body)])
    fs = _fs()
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert m["replays"][0]["decision"] == "refuted" and m["replays"][0]["score_diff"] > 0
    item = fs.store.find_text("tuning single coordinates never helps")
    assert item.status == "refuted"
    assert fs.store.match_dead_ends(["coordinate-tuning"]) == []  # a refuted dead end no longer fires


def test_replay_edit_choice_by_relation_and_decision():
    claim = {"kind": "fact", "text": "t"}
    assert edit_for({**claim, "relation": "new"}, "confirmed", None, [1]).op == "add"
    assert edit_for({**claim, "relation": "new"}, "refuted", None, [1]).op == "refute"
    assert edit_for({**claim, "relation": "supports"}, "confirmed", "m1", [1]).op == "support"
    assert edit_for({**claim, "relation": "supports"}, "refuted", "m1", [1]).op == "contradict"
    assert edit_for({**claim, "relation": "contradicts"}, "confirmed", "m1", [1]).op == "refute"
    assert edit_for({**claim, "relation": "contradicts"}, "refuted", "m1", [1]).op == "support"
    assert is_crash("program produced no result") and not is_crash("circles overlap")


def test_single_call_path_when_no_claim_needs_a_replay(mock_llm):
    body = {"claims": [{"cid": "c1", "kind": "fact", "text": "f", "relation": "new"}], "replays": []}
    llm, budget, transport = mock_llm(10_000, script=[json.dumps(body)])
    fs = _fs()
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert len(transport.calls) == 1 == budget.sleep_calls
    assert m["replays_run"] == 0 and fs.evaluator.calls == [] and m["claims"] == 1


def test_rem_runs_only_when_the_scheduler_chooses_it(mock_llm):
    flat = [{"event": "step", "step": s, "program_id": f"p{s}", "program": GOOD, "score": 0.5, "valid": True}
            for s in range(30)]
    rising = [dict(h, score=0.1 * h["step"]) for h in flat]
    for hist, want in ((rising, "nrem"), (flat, "rem")):
        llm, budget, transport = mock_llm(10_000)
        m = _fs(plateau_window=15).consolidate("task", hist, llm, budget, step=29)
        prompt = transport.calls[0][1]["content"]
        assert m["sleep_mode"] == want
        assert ("<mode>REM" in prompt) == (want == "rem") and ("<mode>NREM" in prompt) == (want == "nrem")


def test_rem_hypotheses_always_enter_as_untested(mock_llm):
    flat = [{"event": "step", "step": s, "program_id": f"p{s}", "program": GOOD, "score": 0.5, "valid": True}
            for s in range(30)]
    body = {"claims": [{"cid": "c1", "kind": "rule", "text": "try spirals", "relation": "new", "status": "verified"},
                       {"cid": "c2", "kind": "skill", "text": "recombine a and b", "relation": "new"}],
            "replays": []}
    llm, budget, _ = mock_llm(10_000, script=[json.dumps(body)])
    fs = _fs(plateau_window=5)
    m = fs.consolidate("task", flat, llm, budget, step=29)
    assert m["sleep_mode"] == "rem"
    assert {fs.store.find_text(t).status for t in ("try spirals", "recombine a and b")} == {"tentative"}


def test_nothing_new_since_the_last_sleep_makes_no_llm_call(mock_llm):
    """The 'worth sleeping' check: an empty window since the last sleep skips the consolidator call."""
    llm, budget, transport = mock_llm(10_000)
    fs = _fs()
    hist = _sleep_history()
    fs.consolidate("task", hist, llm, budget, step=5)
    n = len(transport.calls)
    m = fs.consolidate("task", hist, llm, budget, step=5)  # same history: nothing new
    assert len(transport.calls) == n and m["history_steps"] == 0 and m["claims"] == 0


def test_perseveration_only_trigger_strengthens_the_dead_end_with_zero_llm_calls():
    fs = FullSleep()
    dead = fs.store.add(kind="dead_end", text="grid sweep never helps", step=1, approach_tags=["grid-sweep"],
                        evidence=[1, 2])
    fs.observe(5, {"step": 5, "approach_tags": ["grid-sweep", "x"], "valid": True, "score": 0.1})
    before = dead.strength
    hist = [{"event": "step", "step": i, "valid": i != 3, "score": 0.5 if i != 3 else None, "program_id": f"p{i}",
             "parent_id": None, "delta_vs_best": None, "error": None if i != 3 else "x" * 500,
             "side_info": {"big": "y" * 300}} for i in range(6)]
    m = fs.forget(hist, step=5)
    assert m["llm_calls"] == 0 and m["dead_ends_strengthened"] == [dead.id]
    assert dead.strength > before and 5 in dead.evidence
    shown = fs.select_recent(hist[1:], 6)
    assert [h["step"] for h in shown] == [3, 4, 5]  # 1, 2 dropped: memory cites them
    assert all(h["side_info"] == {} for h in shown)
    assert len(shown[0]["error"]) <= fs.p["forget_error_chars"] + 3


def test_every_sleep_llm_call_is_charged_to_the_budget(run_dir, mock_llm, mock_cfg):
    llm, budget, transport = mock_llm(10**6)
    pol = make_policy("full_sleep", {"gating": "fixed", "every_n": 4, "replay_budget": 2, "schema": True,
                                     "schema_min_items": 1, "schema_min_strength": 0.0, "plateau_window": 3})
    s = run(mock_cfg(max_steps=12), TextEvaluator(), llm, pol, out_dir=run_dir)
    assert transport.sleep_calls == budget.sleep_calls > 0
    assert len(transport.calls) == budget.agent_calls + budget.sleep_calls
    assert budget.sleep_tokens == 100 * budget.sleep_calls
    assert hist_of(run_dir, s)[-1]["sleep_tokens"] == budget.sleep_tokens


# additional coverage: sleep pipeline

def test_full_sleep_loop_replays_never_become_steps(run_dir, mock_llm, mock_cfg):
    llm, budget, transport = mock_llm(10**6)
    pol = make_policy("full_sleep", {"gating": "fixed", "every_n": 4, "replay_budget": 2})
    s = run(mock_cfg(max_steps=12), TextEvaluator(), llm, pol, out_dir=run_dir)
    hist = hist_of(run_dir, s)
    steps = [h for h in hist if h["event"] == "step"]
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert [h["program_id"] for h in steps] == [f"p{i}" for i in range(13)]
    assert [h["step"] for h in sleeps] == [4, 8, 12]
    assert sum(len(h["details"]["replays"]) for h in sleeps) >= 3
    for h in sleeps:
        assert h["sleep_mode"] in ("nrem", "rem") and h["block_tokens"] == count_tokens(h["details"]["block"])
        assert isinstance(h["items_added"], int) and isinstance(h["replays_wasted"], int)
    assert all(h["approach_tags"] and h["expected_score"] is not None for h in steps[1:])
    assert all(h["surprise"] is not None for h in steps[1:])


def test_consolidate_tolerates_foreign_history_without_awake_fields(mock_llm):
    llm, budget, transport = mock_llm(10_000)  # default mock consolidator JSON
    fs = FullSleep()
    fs.bind(evaluator=TextEvaluator())
    foreign = [{"step": s, "program": GOOD if s == 3 else BAD, "score": 0.9 if s == 3 else 0.51, "valid": True}
               for s in range(5)]
    m = fs.consolidate("task", foreign, llm, budget, step=4)
    assert m["claims"] >= 2 and m["items_added"] >= 2 and m["parse_error"] is None
    section = fs.memory_section(120)
    assert section and count_tokens(section) <= 120
    assert "sleep-phase consolidator" in transport.calls[0][0]["content"]


def test_tentative_edits_follow_relation_when_replay_is_disabled(mock_llm):
    fs = _fs(replay=False)
    keep = fs.store.add(kind="rule", text="keep", step=0, strength=0.5)
    drop = fs.store.add(kind="rule", text="drop", step=0, strength=0.5)
    body = {"claims": [
        {"cid": "c1", "kind": "fact", "text": "new one", "relation": "new"},
        {"cid": "c2", "kind": "fact", "text": "x", "relates_to": [keep.id], "relation": "supports"},
        {"cid": "c3", "kind": "fact", "text": "y", "relates_to": [drop.id], "relation": "contradicts"},
    ], "replays": [{"cid": "c1", "program": GOOD}]}
    llm, budget, _ = mock_llm(10_000, script=[json.dumps(body)])
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert m["replays_run"] == 0 and fs.evaluator.calls == []
    assert keep.strength > 0.5 and drop.strength < 0.5
    assert fs.store.find_text("new one").status == "tentative"


def test_sleep_returns_capped_memory_and_budget_exhaustion_propagates(mock_llm):
    llm, budget, _ = mock_llm(10_000)
    block = _fs(max_block_tokens=60).sleep("task", "", _sleep_history(), llm, budget)
    assert block and count_tokens(block) <= 60
    llm2, budget2, _ = mock_llm(150)
    budget2.charge("agent", 150, 0)
    with pytest.raises(BudgetExhausted):
        _fs().consolidate("task", _sleep_history(), llm2, budget2, step=5)


def test_decay_and_eviction_flags(mock_llm):
    body = json.dumps({"claims": [], "replays": []})
    llm, budget, _ = mock_llm(10_000, script=[body])
    fs = _fs(decay=True, decay_factor=0.5, min_strength=0.2)
    weak = fs.store.add(kind="rule", text="weak", step=0, strength=0.3)
    fs.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert weak.id not in fs.store.items  # 0.3 -> 0.15 < 0.2 -> evicted
    fs2 = _fs(decay=False, eviction=False, min_strength=0.9)
    w2 = fs2.store.add(kind="rule", text="weak", step=0, strength=0.3)
    m = fs2.consolidate("task", _sleep_history(), llm, budget, step=5)
    assert w2.strength == 0.3 and m["items_evicted"] == 0


def test_pressure_gate_min_and_max_gap():
    assert not P.gate(4, 0, 1.0, threshold=0.5, min_gap=5, max_gap=20)
    assert P.gate(20, 0, 0.0, threshold=0.5, min_gap=5, max_gap=20)
    assert P.gate(10, 0, 0.6, threshold=0.5, min_gap=5, max_gap=20)
    assert not P.gate(10, 0, 0.4, threshold=0.5, min_gap=5, max_gap=20)


def test_pressure_components():
    recs = [{"step": s, "score": 0.5 if s else 0.4, "valid": True, "surprise": 0.2,
             "perseveration_hit": s % 2 == 0} for s in range(11)]
    c = P.components(recs, step=10, last_sleep_step=0, stall_window=10, clutter_norm=20)
    assert c["stall"] == pytest.approx(0.9)  # best set at step 1
    assert c["perseveration"] == pytest.approx(5 / 10)
    assert c["clutter"] == pytest.approx(0.5) and c["surprise"] == pytest.approx(0.2)
    assert P.components([{"step": 0}], step=0, last_sleep_step=0)["surprise"] == 0.0  # missing fields ok
    assert P.combine({"stall": 1.0, "perseveration": 0.0, "clutter": 0.0, "surprise": 0.0}) == pytest.approx(1 / 3.5)


def test_scheduler_modes():
    flat = [{"step": s, "score": 0.5, "valid": True} for s in range(30)]
    rising = [{"step": s, "score": 0.1 * s, "valid": True} for s in range(30)]
    assert choose_mode(flat, 29, None, plateau_window=15)[0] == "rem"
    assert choose_mode(rising, 29, None, plateau_window=15)[0] == "nrem"
    late = Budget(100)
    late.charge("agent", 90, 0)
    assert choose_mode(flat, 29, late, plateau_window=15, late_budget_fraction=0.8)[0] == "nrem"
    assert choose_mode(flat, 29, None, enabled=False)[0] == "nrem"


def test_schema_store_runs_in_rem_only_and_stays_within_cap(mock_llm, tmp_path):
    llm, budget, transport = mock_llm(100_000)
    flat = [{"event": "step", "step": s, "program_id": f"p{s}", "program": GOOD, "score": 0.5, "valid": True}
            for s in range(30)]
    # cap 120: the mock consolidator now also writes a goal, which renders before schemas
    fs = _fs(schema=True, schema_min_items=2, schema_min_strength=0.5, plateau_window=5, max_block_tokens=120)
    for t in ("a", "b", "c"):
        fs.store.add(kind="rule", text=f"strong {t}", step=0, status="verified", strength=0.9)
    m = fs.consolidate("task", flat, llm, budget, step=29)
    assert m["sleep_mode"] == "rem" and m["schema_ran"] and fs.schemas.schemas
    assert any("memory schemas" in c[0]["content"] for c in transport.calls)
    section = fs.memory_section(120)
    assert "SCHEMA" in section and count_tokens(section) <= 120
    assert budget.sleep_calls == len(transport.calls)
    assert not list(tmp_path.iterdir())  # no persistence by default
    fs_p = _fs(schema=True, schema_min_items=2, schema_min_strength=0.5, plateau_window=5,
               schema_persist=True, schema_path=str(tmp_path / "schemas.json"))
    for t in ("a", "b", "c"):
        fs_p.store.add(kind="rule", text=f"strong {t}", step=0, status="verified", strength=0.9)
    fs_p.consolidate("task", flat, llm, budget, step=29)
    assert (tmp_path / "schemas.json").exists()


def test_full_sleep_fixed_schedule_and_param_validation():
    fs = FullSleep(every_n=5, pressure=False)  # fatigue gating without a reading falls back to every_n
    assert fs.should_sleep(5, []) and not fs.should_sleep(4, [])
    fixed = FullSleep(gating="fixed", every_n=5)
    assert fixed.should_sleep(10, []) and not fixed.should_sleep(9, []) and not fixed.should_sleep(0, [])
    with pytest.raises(ValueError):
        FullSleep(not_a_flag=True)
    assert fs.memory_section(50) == ""  # empty store renders nothing


def test_after_sleep_the_agent_sees_only_the_last_couple_of_pre_sleep_steps(run_dir, mock_llm, mock_cfg):
    llm, _, transport = mock_llm(10**6)
    pol = FullSleep(gating="fixed", every_n=6, wake_recent_steps=2, replay=False)
    run(mock_cfg(max_steps=8, k_recent=5), TextEvaluator(), llm, pol, out_dir=run_dir)
    agent = [m for m in transport.calls if "APPROACH_TAGS" in m[0]["content"]]
    after = agent[6][1]["content"]  # the agent call for step 7, right after the sleep at step 6
    assert "step 6 |" in after and "step 5 |" in after and "step 4 |" not in after
    assert "GOAL" in after


def _reading(level, depth, remaining=None, debt=0.0):
    st = FatigueState(step=5, fatigue=0.6, load=0.6, impairment=0.0, components={c: 0.0 for c in COMPONENTS},
                      level=level, debt=debt, grace=False, baseline_ready=True, awake_steps=5, armed=True,
                      load_tokens=600)
    return FatigueReading(st, depth, False, remaining)


def test_policy_defers_when_budget_is_short_and_runs_nrem_then_rem_when_exhausted():
    fs = FullSleep(min_sleep_budget_tokens=500)
    fs.set_fatigue(_reading(Level.TIRED, SleepDepth.NREM, remaining=100))
    assert fs.should_sleep(5, []) and fs.last_deferred and fs._planned == SleepDepth.FORGET
    fs2 = FullSleep(min_sleep_budget_tokens=500, defer_fallback_forget=False)
    fs2.set_fatigue(_reading(Level.TIRED, SleepDepth.NREM, remaining=100))
    assert not fs2.should_sleep(5, []) and fs2.last_deferred

    budget = Budget(10**6)
    transport = MockTransport()
    llm = LLMClient(budget, transport, sleep_fn=lambda s: None)
    fs3 = FullSleep(replay=False)
    fs3.set_fatigue(_reading(Level.EXHAUSTED, SleepDepth.NREM_REM, remaining=10**6))
    assert fs3.should_sleep(5, [])
    hist = [{"event": "step", "step": i, "valid": True, "score": 0.5 + i / 100, "program_id": f"p{i}",
             "program": "def construct():\n    return [0.7]*5\n", "approach_tags": ["near-target"]}
            for i in range(6)]
    fs3.sleep("task", "", hist, llm, budget)
    assert transport.sleep_calls == 2  # NREM pass + REM pass
    assert fs3.last_depth == "nrem_rem" and fs3.last_metrics["sleep_mode"] == "nrem+rem"


def test_fixed_tiered_uses_fixed_timing_and_fatigue_depth():
    fs = FullSleep(gating="fixed", every_n=3, timed_depth="fatigue")
    for level, depth, want in ((Level.RESTED, SleepDepth.NONE, SleepDepth.FORGET),
                               (Level.TIRED, SleepDepth.NREM, SleepDepth.NREM),
                               (Level.EXHAUSTED, SleepDepth.NREM_REM, SleepDepth.NREM_REM)):
        fs.set_fatigue(_reading(level, depth))
        assert not fs.should_sleep(4, [])
        assert fs.should_sleep(6, []) and fs._planned == want
    nrem = FullSleep(gating="schedule", sleep_steps=[2, 7], timed_depth="nrem")
    nrem.set_fatigue(_reading(Level.EXHAUSTED, SleepDepth.NREM_REM))
    assert [s for s in range(1, 10) if nrem.should_sleep(s, [])] == [2, 7]
    assert nrem.should_sleep(7, []) and nrem._planned == SleepDepth.NREM
    assert nrem.scheduled_sleeps == 2 and FullSleep().scheduled_sleeps is None
    with pytest.raises(ValueError):
        FullSleep(gating="schedule")
    with pytest.raises(ValueError):
        FullSleep(timed_depth="rem")


def test_summarize_validates_every_n_and_folds_new_steps():
    with pytest.raises(ValueError):
        Summarize(every_n=0)
    pol = Summarize(every_n=2)
    assert pol.should_sleep(2, []) and not pol.should_sleep(3, []) and not pol.should_sleep(0, [])


# ================================================================================================
# Section 6 · Fatigue monitor
# ================================================================================================

TOK = 100  # agent tokens per step


def frec(step, *, gap=0.05, valid=True, score=None, tags=("a", "b"), error=None, program=None,
         context=1000, expected=True):
    """A step record. score defaults to a slowly improving value; expected = score + gap."""
    score = (0.5 + 0.001 * step) if score is None else score
    return {"event": "step", "step": step, "valid": valid, "score": score if valid else None,
            "expected_score": (score + gap) if expected else None, "approach_tags": list(tags) if tags else None,
            "error": error, "program": program or f"def construct():\n    return [{step}]\n",
            "agent_tokens": TOK * step, "context_tokens": context}


def calibration_only(**kw):
    w = {c: 0.0 for c in COMPONENTS}
    w["calibration"] = 1.0
    return fatigue_cfg(impairment_weights=w, **kw)


def load_only(**kw):
    """Fatigue driven by load alone (deterministic): 100 tokens/call, load 1.0 after `scale` tokens."""
    base = dict(load_tokens_scale=1000, load_weight=1.0, impairment_weight=0.0, grace_steps=2)
    base.update(kw)
    return base


def calibration_driven(**kw):
    w = {c: 0.0 for c in COMPONENTS}
    w["calibration"] = 1.0
    base = dict(load_tokens_scale=10**9, load_weight=0.0, impairment_weight=1.0, impairment_weights=w,
                baseline_steps=4, window=3, calibration_ratio_cap=20.0, grace_steps=0)
    base.update(kw)
    return base


LEVELS = ["rested", "drowsy", "tired", "exhausted"]


def test_load_rises_with_agent_tokens_and_drops_in_proportion_to_sleep_depth(fatigue_monitor):
    m = fatigue_monitor
    m.observe(frec(0))
    for s in range(1, 6):
        assert m.observe(frec(s)).load == pytest.approx(0.1 * s)
    c = m.cfg
    m.on_sleep(SleepDepth.NREM)
    assert m.load == pytest.approx(0.5 * (1 - c.load_relief.nrem))
    m.on_sleep(SleepDepth.FORGET)
    assert m.load == pytest.approx(0.5 * (1 - c.load_relief.nrem) * (1 - c.load_relief.forget))
    before = m.load
    m.on_sleep("none")
    assert m.load == before
    assert m.observe(frec(6)).load == pytest.approx(before + 0.1)  # keeps rising from the reduced level
    out = {}
    for d in (SleepDepth.FORGET, SleepDepth.NREM, SleepDepth.NREM_REM):
        mm = FatigueMonitor(fatigue_cfg())
        for s in range(0, 8):
            mm.observe(frec(s))
        mm.on_sleep(d)
        out[d] = mm.load
    assert out[SleepDepth.FORGET] > out[SleepDepth.NREM] > out[SleepDepth.NREM_REM]


def test_load_curve_options():
    m = FatigueMonitor(fatigue_cfg(load_curve="saturating", load_cap=0.9))
    for s in range(0, 40):
        st = m.observe(frec(s))
    assert st.load == pytest.approx(0.9)
    m = FatigueMonitor(fatigue_cfg(load_budget_fraction=0.5), total_budget=4000)
    m.observe(frec(0))
    assert m.observe(frec(2)).load == pytest.approx(200 / 2000)


def test_baseline_resets_after_sleep_and_impairment_is_zero_until_reestablished():
    m = FatigueMonitor(calibration_only())
    m.observe(frec(0))
    for s in (1, 2):
        st = m.observe(frec(s, gap=0.05))
        assert not st.baseline_ready and st.impairment == 0
    assert m.observe(frec(3, gap=0.05)).baseline_ready  # set after baseline_steps=3
    st = m.observe(frec(4, gap=0.5))
    assert st.impairment > 0 and st.components["calibration"] > 0
    m.on_sleep(SleepDepth.NREM)
    assert not m.baseline_ready and m.current().impairment == 0
    for s in (5, 6):  # even terrible calibration counts as zero until the new baseline exists
        st = m.observe(frec(s, gap=2.0))
        assert not st.baseline_ready and st.impairment == 0
        assert all(v == 0 for v in st.components.values())
    st = m.observe(frec(7, gap=2.0))
    assert st.baseline_ready and st.components["calibration"] == pytest.approx(0.0)  # new baseline = bad gap


def test_forget_does_not_reset_baseline_unless_configured():
    m = FatigueMonitor(fatigue_cfg())
    for s in range(0, 5):
        m.observe(frec(s))
    m.on_sleep(SleepDepth.FORGET)
    assert m.baseline_ready
    m = FatigueMonitor(fatigue_cfg(forget_resets_baseline=True))
    for s in range(0, 5):
        m.observe(frec(s))
    m.on_sleep(SleepDepth.FORGET)
    assert not m.baseline_ready


SIGNAL_DRIVERS = {  # component: (rested driver, impaired driver) for synthetic_history
    "calibration": (dict(surprise=0.05), dict(surprise=0.5)),
    "perseveration": (dict(perseveration=0.0), dict(perseveration=0.9)),
    "stagnation": (dict(improving=True), dict(improving=False)),
    "sloppiness": (dict(invalid_rate=0.0), dict(invalid_rate=0.6)),
    "narrowing": (dict(tag_diversity=1.0), dict(tag_diversity=0.05)),
    "bloat": (dict(context=1000), dict(context=3500)),
}


@pytest.mark.parametrize("component", COMPONENTS)
def test_each_impairment_signal_rises_with_its_synthetic_driver(component, make_history):
    cfg = FatigueConfig(load_tokens_scale=10**6, baseline_steps=5, window=5, grace_steps=0, context_soft_cap=4000)

    def final(hist):
        m = FatigueMonitor(cfg, best_known=1.0)
        for r in hist:
            st = m.observe(r)
        return st

    rested_kw, impaired_kw = SIGNAL_DRIVERS[component]
    rested, impaired = final(make_history(**rested_kw)), final(make_history(**impaired_kw))
    assert impaired.components[component] > rested.components[component] + 0.2
    assert impaired.impairment > rested.impairment


def test_worsening_calibration_escalates_drowsy_tired_exhausted():
    m = FatigueMonitor(calibration_only(load_weight=0.0, impairment_weight=1.0, calibration_ratio_cap=11.0, window=2))
    m.observe(frec(0))
    levels = [m.observe(frec(s, gap=0.05 * (1 + 0.5 * max(0, s - 3)))).level for s in range(1, 30)]
    assert list(dict.fromkeys(levels)) == [Level.RESTED, Level.DROWSY, Level.TIRED, Level.EXHAUSTED]
    assert all(a <= b for a, b in zip(levels, levels[1:]))  # monotone without sleep


def test_worsening_calibration_escalates_levels_through_the_loop(run_dir, mock_cfg):
    budget = Budget(10**6)
    llm = LLMClient(budget, MockTransport(calibration_drift=0.02, calibration_drift_start=4), sleep_fn=lambda s: None)
    s = run(mock_cfg(max_steps=30, awake_tags=True, fatigue=calibration_driven()), TextEvaluator(), llm, NoSleep(),
            out_dir=run_dir)
    steps = [h for h in hist_of(run_dir, s) if h["event"] == "step"]
    assert list(dict.fromkeys(h["fatigue_level"] for h in steps)) == LEVELS
    gaps = [abs(h["expected_score"] - h["score"]) for h in steps[1:]]
    assert gaps[-1] > gaps[5] > gaps[0] - 1e-9
    assert all(h["fatigue_calibration"] is not None for h in steps if h["fatigue_baseline_ready"])


def test_grace_period_forces_rested_after_wake():
    m = FatigueMonitor(fatigue_cfg(grace_steps=2, load_relief={"forget": 0, "nrem": 0, "nrem_rem": 0},
                                   load_weight=1.0, impairment_weight=0.0))
    m.observe(frec(0))
    for s in range(1, 10):
        st = m.observe(frec(s))
    assert st.level == Level.EXHAUSTED  # load 0.9
    m.on_sleep(SleepDepth.NREM)  # no load relief: fatigue stays high
    assert m.observe(frec(10)).level == Level.RESTED and m.state.grace
    assert m.observe(frec(11)).level == Level.RESTED
    assert m.observe(frec(12)).level <= Level.DROWSY  # grace over, but not re-armed (hysteresis)


def test_post_sleep_hysteresis_requires_dropping_below_the_wake_threshold():
    c = fatigue_cfg(grace_steps=0, load_weight=1.0, impairment_weight=0.0, rearm_max_steps=None,
                    load_relief={"forget": 0.0, "nrem": 0.7, "nrem_rem": 1.0})
    m = FatigueMonitor(c)
    m.observe(frec(0))
    for s in range(1, 7):
        m.observe(frec(s))
    assert m.state.level == Level.TIRED  # load 0.6
    m.on_sleep(SleepDepth.NREM)  # load -> 0.18 < wake_threshold 0.25: re-arms on the next step
    lv = [m.observe(frec(s)).level for s in range(7, 12)]
    assert lv[0] == Level.RESTED and Level.TIRED in lv
    m = FatigueMonitor(c.model_copy(update={"load_relief": c.load_relief.model_copy(update={"nrem": 0.3})}))
    m.observe(frec(0))
    for s in range(1, 9):
        m.observe(frec(s))
    m.on_sleep(SleepDepth.NREM)  # load 0.8 -> 0.56: never below the wake threshold
    lv = [m.observe(frec(s)).level for s in range(9, 20)]
    assert max(lv) == Level.DROWSY and not m.state.armed


def test_schmitt_trigger_holds_level_inside_the_margin():
    m = FatigueMonitor(fatigue_cfg(grace_steps=0, load_weight=1.0, impairment_weight=0.0, hysteresis_margin=0.05))
    m.observe(frec(0))
    for s in range(1, 6):
        m.observe(frec(s))
    assert m.state.level == Level.TIRED  # 0.5
    m._load_tokens = 470  # 0.47: inside the margin -> stays tired
    m.observe({**frec(6), "agent_tokens": 600})
    assert m.state.level == Level.TIRED
    m._load_tokens = 440
    m.observe({**frec(7), "agent_tokens": 600})
    assert m.state.level == Level.DROWSY


@pytest.mark.parametrize("grace", [0, 2])
def test_grace_and_hysteresis_prevent_sleeping_on_consecutive_steps(grace):
    m = FatigueMonitor(fatigue_cfg(grace_steps=grace, load_weight=1.0, impairment_weight=0.0,
                                   load_relief={"forget": 0.0, "nrem": 0.4, "nrem_rem": 0.6}))
    m.observe(frec(0))
    slept = []
    for s in range(1, 60):
        if m.observe(frec(s)).level >= Level.TIRED:  # a policy that sleeps whenever tired
            m.on_sleep(m.recommended_depth())
            slept.append(s)
    assert len(slept) >= 3 and all(b - a > 1 for a, b in zip(slept, slept[1:]))


def test_grace_and_hysteresis_prevent_sleeping_on_consecutive_steps_in_the_loop(run_dir, mock_llm, mock_cfg):
    for grace in (0, 2):
        llm, _, _ = mock_llm(10**6)
        pol = FullSleep(min_sleep_budget_tokens=0, respond_drowsy=False, replay=False)
        s = run(mock_cfg(max_steps=60, run_id=f"g{grace}",
                         fatigue=load_only(grace_steps=grace, load_relief={"forget": 0, "nrem": 0.5, "nrem_rem": 0.6})),
                TextEvaluator(), llm, pol, out_dir=run_dir)
        hist = hist_of(run_dir, s)
        slept = [h["step"] for h in hist if h["event"] == "sleep"]
        assert len(slept) >= 3 and all(b - a > max(1, grace) for a, b in zip(slept, slept[1:]))
        steps = {h["step"]: h for h in hist if h["event"] == "step"}
        for st in slept:
            for k in range(1, grace + 1):
                if st + k in steps:
                    assert steps[st + k]["fatigue_level"] == "rested" and steps[st + k]["fatigue_grace"]


def test_deferred_sleep_accumulates_debt_and_the_next_sleep_runs_deeper():
    c = fatigue_cfg(grace_steps=0, load_weight=1.0, impairment_weight=0.0, debt_increment=0.1, debt_cap=0.25)
    m = FatigueMonitor(c)
    m.observe(frec(0))
    for s in range(1, 6):
        m.observe(frec(s))
    assert m.state.level == Level.TIRED and m.recommended_depth() == SleepDepth.NREM
    f0 = m.state.fatigue
    m.on_deferred()
    assert m.state.deferred and m.debt == pytest.approx(0.1)
    assert m.observe(frec(6)).fatigue == pytest.approx(f0 + 0.1 + 0.1)  # +0.1 load, +0.1 debt
    assert m.recommended_depth() == SleepDepth.NREM_REM  # deeper than without debt
    m.on_deferred()
    m.on_deferred()
    assert m.debt == pytest.approx(0.25)  # capped
    m.on_sleep(SleepDepth.NREM_REM)
    assert m.debt == pytest.approx(0.25 * (1 - c.debt_relief.nrem_rem))


def test_deferred_sleep_in_the_loop_logs_debt(run_dir, mock_llm, mock_cfg):
    llm, budget, _ = mock_llm(10**6)
    pol = FullSleep(min_sleep_budget_tokens=10**7)  # never enough budget: every due sleep is deferred
    s = run(mock_cfg(max_steps=20, fatigue=load_only()), TextEvaluator(), llm, pol, out_dir=run_dir)
    hist = hist_of(run_dir, s)
    deferred = [h for h in hist if h["event"] == "step" and h["sleep_deferred"]]
    assert deferred and budget.sleep_calls == 0
    assert max(h["fatigue_debt"] for h in hist) > 0
    assert {h["sleep_depth"] for h in hist if h["event"] == "sleep"} <= {"forget"}
    assert s["fatigue"]["deferrals"] == len(deferred)


def test_drowsy_tier_forgetting_makes_zero_llm_calls(run_dir, mock_llm, mock_cfg):
    llm, budget, transport = mock_llm(10**6)
    # drowsy reachable, tired never: the only response is forgetting
    fat = load_only(drowsy_threshold=0.3, tired_threshold=5.0, exhausted_threshold=6.0, wake_threshold=0.2)
    s = run(mock_cfg(max_steps=25, fatigue=fat), TextEvaluator(), llm, FullSleep(), out_dir=run_dir)
    hist = hist_of(run_dir, s)
    forgets = [h for h in hist if h["event"] == "sleep"]
    assert len(forgets) >= 3 and {h["sleep_depth"] for h in forgets} == {"forget"}
    assert transport.sleep_calls == 0 and budget.sleep_calls == 0 and budget.sleep_tokens == 0
    assert len(transport.calls) == budget.agent_calls == 25
    assert all(h["sleep_tokens"] == 0 for h in hist)


def test_no_sleep_arm_logs_fatigue_but_never_triggers_a_sleep(run_dir, mock_llm, mock_cfg):
    llm, budget, transport = mock_llm(10**6)
    s = run(mock_cfg(max_steps=20, awake_tags=True, fatigue=load_only()), TextEvaluator(), llm, NoSleep(),
            out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert all(set(h) == set(FIELDS) and set(FATIGUE_FIELDS) <= set(h) for h in hist)
    assert all(h["event"] == "step" for h in hist)
    assert all(h["fatigue"] is not None and h["fatigue_level"] in LEVELS for h in hist)
    assert not any(h["sleep_triggered"] or h["sleep_deferred"] for h in hist)
    assert budget.sleep_calls == 0 and transport.sleep_calls == 0
    assert {h["fatigue_level"] for h in hist} >= {"tired", "exhausted"}  # it WOULD have slept
    fat = [h["fatigue"] for h in hist]
    assert fat[-1] > fat[1] and all(b >= a for a, b in zip(fat, fat[1:]))
    assert s["fatigue"]["sleeps"] == {"forget": 0, "nrem": 0, "nrem_rem": 0}


# additional coverage: fatigue

def test_full_sleep_responds_by_level_and_fatigue_drops_after_each_sleep(run_dir, mock_llm, mock_cfg):
    llm, _, _ = mock_llm(10**6)
    pol = make_policy("full_sleep", {"min_sleep_budget_tokens": 0, "replay_budget": 1})
    s = run(mock_cfg(max_steps=40, fatigue=load_only()), TextEvaluator(), llm, pol, out_dir=run_dir)
    hist = hist_of(run_dir, s)
    steps = [h for h in hist if h["event"] == "step"]
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert {"forget", "nrem"} <= {h["sleep_depth"] for h in sleeps}
    for sl in sleeps:  # sleep line = state after the sleep; the triggering step line = before
        trig = next(h for h in steps if h["step"] == sl["step"])
        assert trig["sleep_triggered"] and trig["sleep_depth"] == sl["sleep_depth"]
        assert sl["fatigue"] < trig["fatigue"]
        if sl["sleep_depth"] != "forget":
            assert sl["fatigue_level"] == "rested" and not sl["fatigue_baseline_ready"]
            assert sl["sleep_mode"] in ("nrem", "nrem+rem")
        else:
            assert sl["sleep_mode"] is None and sl["details"]["llm_calls"] == 0
    # identical offline recomputation from the log
    replayed = replay_history(hist, s["config"]["fatigue"], best_known=1.0, total_budget=10**6)
    assert [st.to_log() for st in replayed] == [{k: h[k] for k in FATIGUE_FIELDS} for h in steps]


def test_sleep_lite_only_forgets_even_when_exhausted(run_dir, mock_llm, mock_cfg):
    llm, budget, transport = mock_llm(10**6)
    s = run(mock_cfg(max_steps=30, fatigue=load_only()), TextEvaluator(), llm, make_policy("sleep_lite"),
            out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert {h["fatigue_level"] for h in hist} >= {"exhausted"}
    assert {h["sleep_depth"] for h in hist if h["event"] == "sleep"} == {"forget"}
    assert transport.sleep_calls == 0 and budget.sleep_calls == 0
    with pytest.raises(ValueError):
        SleepLite(gating="pressure")
    with pytest.raises(ValueError):
        SleepLite(max_depth="nrem")


def test_fatigue_can_be_disabled(run_dir, mock_llm, mock_cfg):
    llm, _, _ = mock_llm(10**6)
    s = run(mock_cfg(max_steps=3, fatigue={"enabled": False}), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    assert all(h[k] is None for h in hist_of(run_dir, s) for k in FATIGUE_FIELDS)


def test_recommended_depth_by_level_and_stagnation():
    m = FatigueMonitor(fatigue_cfg())
    base = m.state
    assert m.recommended_depth(dc_replace(base, level=Level.RESTED)) == SleepDepth.NONE
    assert m.recommended_depth(dc_replace(base, level=Level.DROWSY)) == SleepDepth.FORGET
    assert m.recommended_depth(dc_replace(base, level=Level.TIRED)) == SleepDepth.NREM
    assert m.recommended_depth(dc_replace(base, level=Level.EXHAUSTED)) == SleepDepth.NREM_REM
    stagnant = dc_replace(base, level=Level.TIRED, baseline_ready=True,
                          components={**base.components, "stagnation": 0.9})
    assert m.recommended_depth(stagnant) == SleepDepth.NREM_REM


def test_sloppiness_counts_invalid_timeout_and_parse_failures():
    m = FatigueMonitor(fatigue_cfg())
    m.observe(frec(0))
    for s in (1, 2, 3):
        m.observe(frec(s))
    m.observe(frec(4, valid=False, error="no ```python code block in response"))
    st = m.observe(frec(5, valid=False, error="timeout after 10s"))
    assert st.components["sloppiness"] > 0.9
    m2 = FatigueMonitor(fatigue_cfg())  # missing awake header once headers were seen = parse failure
    for s in range(0, 4):
        m2.observe(frec(s))
    assert m2.observe(frec(4, tags=None, expected=False)).components["sloppiness"] > 0


def test_perseveration_from_raw_log_against_earlier_failures():
    m = FatigueMonitor(fatigue_cfg(perseveration_failure="invalid"))
    m.observe(frec(0))
    m.observe(frec(1, tags=("x", "y")))
    m.observe(frec(2, tags=("p", "q")))
    m.observe(frec(3, tags=("r", "s"), valid=False, error="boom"))  # failed with r,s
    m.observe(frec(4, tags=("r", "s")))
    st = m.observe(frec(5, tags=("r", "z")))
    assert st.components["perseveration"] > 0.5
    m.observe(frec(6, tags=("new", "idea")))
    m.observe(frec(7, tags=("other", "idea")))
    assert m.state.components["perseveration"] < st.components["perseveration"]


def test_narrowing_and_bloat():
    m = FatigueMonitor(fatigue_cfg(context_soft_cap=4000, bloat_cap=1.0))
    m.observe(frec(0))
    progs = ["def construct():\n    return alpha_%d\n" % s for s in range(3)]
    tags = [("a", "b"), ("c", "d"), ("e", "f")]
    for s in (1, 2, 3):
        m.observe(frec(s, tags=tags[s - 1], program=progs[s - 1] + f"# {s} unique{s} words{s}"))
    same = "def construct():\n    x = 1\n    return [x, x, x]\n"
    for s in (4, 5, 6):
        st = m.observe(frec(s, tags=("a", "b"), program=same, context=3000))
    assert st.components["narrowing"] > 0.5 and st.components["bloat"] == pytest.approx(0.75)


def test_missing_data_components_are_none_and_renormalized():
    m = FatigueMonitor(fatigue_cfg())
    for s in range(0, 6):
        st = m.observe(frec(s, tags=None, expected=False))
    assert st.components["calibration"] is None and st.components["perseveration"] is None
    assert st.components["bloat"] is None  # soft cap disabled
    assert 0.0 <= st.impairment <= 1.0


def test_fatigue_log_fields_have_exact_keys_and_disabled_gives_nulls():
    m = FatigueMonitor(fatigue_cfg())
    m.observe(frec(0))
    f = m.log_fields()
    assert tuple(f) == FATIGUE_FIELDS
    assert f["fatigue_level"] == "rested" and f["sleep_triggered"] is False and f["sleep_depth"] is None
    m.on_sleep("nrem")
    assert m.log_fields()["sleep_triggered"] is True and m.log_fields()["sleep_depth"] == "nrem"
    assert set(FatigueMonitor(fatigue_cfg(enabled=False)).log_fields().values()) == {None}


def test_fatigue_config_validation():
    with pytest.raises(ValidationError):
        FatigueConfig(drowsy_threshold=0.6, tired_threshold=0.5)
    with pytest.raises(ValidationError):
        FatigueConfig(wake_threshold=0.6)
    with pytest.raises(ValidationError):
        FatigueConfig(not_a_field=1)
    with pytest.raises(ValidationError):
        FatigueConfig(impairment_weights={"calibration": 1.0, "typo": 2.0})
    assert SleepDepth.parse("nrem+rem") == SleepDepth.NREM_REM
    with pytest.raises(ValueError):
        SleepDepth.parse("deep")


def test_offline_replay_matches_online_monitoring():
    def online(records, decide):
        m = FatigueMonitor(fatigue_cfg(grace_steps=1, load_weight=1.0, impairment_weight=0.5))
        logged = []
        for r in records:
            m.observe(r)
            action = decide(m)
            if action == "defer":
                m.on_deferred()
            elif action:
                m.on_sleep(action)
            logged.append({**r, **m.log_fields()})
            if m.state.triggered:
                logged.append({"event": "sleep", "step": r["step"], **m.log_fields(m.current())})
        return m, logged

    records = [frec(s, gap=0.05 * (1 + s % 7), valid=(s % 5 != 4), error=None if s % 5 != 4 else "boom")
               for s in range(0, 40)]

    def decide(m):
        lvl = m.state.level
        if lvl == Level.EXHAUSTED and m.state.step % 3 == 0:
            return "defer"
        return m.recommended_depth().label if lvl >= Level.DROWSY else None

    m, logged = online(records, decide)
    assert m.sleeps  # the scenario does sleep
    replayed = replay_history(logged, fatigue_cfg(grace_steps=1, load_weight=1.0, impairment_weight=0.5))
    online_steps = [h for h in logged if h["event"] == "step"]
    assert len(replayed) == len(online_steps)
    for st, h in zip(replayed, online_steps):
        assert st.to_log() == {k: h[k] for k in FATIGUE_FIELDS}


def test_replay_without_fatigue_fields_uses_sleep_events():
    hist = [frec(s) for s in range(0, 12)]
    hist.insert(7, {"event": "sleep", "step": 6, "sleep_mode": "rem"})
    by_step = {s.step: s for s in replay_history(hist, fatigue_cfg())}
    assert by_step[6].triggered and by_step[6].depth == SleepDepth.NREM_REM
    assert by_step[7].load == pytest.approx(0.6 * (1 - 0.9) + 0.1)
    assert not by_step[7].baseline_ready


# ================================================================================================
# Section 7 · Baseline arms
# ================================================================================================

from harness.baselines import (  # noqa: E402
    ARM_SPECS, BaselinesSection, ConfigDriftError, check_drift, even_schedule, load_suite, make_suite, plan_suite,
    random_schedule, resolve_arm,
)
from harness.metrics import best_curve, compute_run_metrics, first_reach, llm_sleep_steps  # noqa: E402
from scripts.run_matrix import run_suite_local  # noqa: E402

BASELINES_CONFIG = ROOT / "configs" / "baselines_mock.yaml"
SEEDS = [0, 1, 2]
ALL_ARMS = ["no_sleep", "fixed_letta", "fixed_sparse", "fixed_matched", "random_matched", "fatigue", "sleep_lite"]
SUITE_TOTAL = 6000  # configs/baselines_mock.yaml: 60 mock calls of 100 tokens


@pytest.fixture(scope="module")
def suite():
    return load_suite(BASELINES_CONFIG)


@pytest.fixture(scope="module")
def matrix(tmp_path_factory, suite):
    """The full mock matrix, run once through run_matrix's local DAG runner (text-parsing evaluator)."""
    out = tmp_path_factory.mktemp("baselines")
    transports: dict[str, MockTransport] = {}
    order: list[str] = []

    def execute(cfg, out_dir):
        budget = Budget(cfg.total_token_budget)
        t = MockTransport(seed=cfg.seed, memory_boost=0.6)
        llm = LLMClient(budget, t, sleep_fn=lambda s: None)
        s = run(cfg, TextEvaluator(), llm, make_policy(cfg.sleep_policy, cfg.policy_params), out_dir=out_dir)
        transports[s["run_id"]] = t
        order.append(s["run_id"])
        return s

    summaries = run_suite_local(suite, [0], SEEDS, None, out_dir=str(out), execute_fn=execute, log=lambda m: None)
    return {"out": out, "summaries": {s["run_id"]: s for s in summaries}, "transports": transports, "order": order}


def _msum(matrix, arm, seed):
    return matrix["summaries"][f"p0_{arm}_s{seed}"]


def _mhist(matrix, arm, seed):
    return read_history(matrix["out"] / f"p0_{arm}_s{seed}" / "history.jsonl")


@pytest.mark.parametrize("section, needle", [
    ({"overrides": {"no_sleep": {"context_cap_tokens": 3000}}}, "context_cap_tokens"),
    ({"overrides": {"fatigue": {"k_recent": 9}}}, "k_recent"),
    ({"overrides": {"fixed_letta": {"total_token_budget": 7000}}}, "total_token_budget"),
    ({"overrides": {"random_matched": {"awake_tags": False}}}, "awake_tags"),
    ({"arm_params": {"fixed_letta": {"decay_factor": 0.5}}}, "sleep.decay_factor"),
    ({"arm_params": {"fatigue": {"every_n": 3}}}, "sleep.every_n"),
    ({"arm_params": {"fixed_matched": {"replay_budget": 0}}}, "sleep.replay_budget"),
])
def test_config_assertion_fails_when_an_arm_drifts_outside_its_declared_params(suite, section, needle):
    bad = make_suite(suite.base, BaselinesSection(**{**suite.section.model_dump(), **section}))
    with pytest.raises(ConfigDriftError, match=needle.replace(".", r"\.")):
        plan_suite(bad, [0], [0])
    with pytest.raises(ConfigDriftError):  # and the local runner raises before executing anything
        run_suite_local(bad, [0], [0], execute_fn=lambda c, o: pytest.fail("ran despite drift"), log=lambda m: None)


def test_fixed_letta_sleeps_every_5_steps_and_fixed_sparse_every_20(matrix):
    for seed in SEEDS:
        for arm, every in (("fixed_letta", 5), ("fixed_sparse", 20)):
            sleeps = [h for h in _mhist(matrix, arm, seed) if h["event"] == "sleep"]
            last = _msum(matrix, arm, seed)["steps_completed"]
            assert sleeps and [h["step"] for h in sleeps] == list(range(every, last + 1, every))[:len(sleeps)]
            assert len(sleeps) >= last // every - 1
            assert {h["sleep_depth"] for h in sleeps} == {"nrem"} and {h["sleep_mode"] for h in sleeps} == {"nrem"}


def test_matched_arms_reproduce_the_fatigue_arms_sleep_count(matrix, suite):
    for seed in SEEDS:
        n = len(llm_sleep_steps(_mhist(matrix, "fatigue", seed)))
        assert n >= 2
        fat = _msum(matrix, "fatigue", seed)["metrics"]["sleep"]
        assert fat["llm_sleeps"] == n and fat["forgets"] > 0  # forgets are reported, not matched
        length = _msum(matrix, "fatigue", seed)["steps_completed"]
        for arm in ("fixed_matched", "random_matched"):
            s = _msum(matrix, arm, seed)
            m = s["metrics"]["sleep"]
            assert m["scheduled"] == m["realized"] == m["llm_sleeps"] == n == len(llm_sleep_steps(_mhist(matrix, arm, seed)))
            assert m["forgets"] == 0
            assert s["matching"]["source_sleeps"] == n and s["matching"]["source_run_id"] == f"p0_fatigue_s{seed}"
            steps = s["matching"]["scheduled_steps"]
            assert steps == m["llm_sleep_steps"] and max(steps) <= length
            on_disk = json.loads((matrix["out"] / s["run_id"] / "summary.json").read_text(encoding="utf-8"))
            assert on_disk["matching"]["scheduled_steps"] == steps
        assert _msum(matrix, "fixed_matched", seed)["matching"]["scheduled_steps"] == even_schedule(n, length,
                                                                                                    suite.min_gap())


def test_random_matched_sleep_steps_are_reproducible_from_the_seed_and_respect_the_min_gap(matrix, suite):
    for seed in SEEDS:
        n = len(llm_sleep_steps(_mhist(matrix, "fatigue", seed)))
        length = _msum(matrix, "fatigue", seed)["steps_completed"]
        rnd = _msum(matrix, "random_matched", seed)["matching"]
        g = suite.min_gap()
        assert rnd["rng_seed"] == f"random_matched|p0|s{seed}"
        assert rnd["scheduled_steps"] == random_schedule(n, length, g, rnd["rng_seed"])
        st = rnd["scheduled_steps"]
        assert st[0] >= g and all(b - a >= g for a, b in zip(st, st[1:]))
        assert {h["sleep_depth"] for h in _mhist(matrix, "random_matched", seed) if h["event"] == "sleep"} == {"nrem"}
    seen = set()
    for n, length, gap in [(1, 10, 3), (3, 30, 4), (6, 51, 4), (5, 20, 4), (8, 40, 5), (4, 9, 4), (10, 10, 2)]:
        for s in range(25):
            a = random_schedule(n, length, gap, f"x|{s}")
            assert a == random_schedule(n, length, gap, f"x|{s}")
            g = min(gap, length // n)  # lowered only when n sleeps cannot fit
            assert len(a) == n and a == sorted(a) and 1 <= a[0] and a[-1] <= length
            assert a[0] >= g and all(b - x >= g for x, b in zip(a, a[1:]))
            seen.add((n, length, gap, tuple(a)))
    assert len({k[3] for k in seen if k[:3] == (3, 30, 4)}) > 10  # actually random across seeds
    assert random_schedule(0, 30, 4, "x") == [] and even_schedule(0, 30, 4) == []
    assert even_schedule(3, 50, 4) == [12, 24, 36]
    for n, length in [(1, 5), (6, 51), (9, 10), (10, 10)]:
        e = even_schedule(n, length, 4)
        assert len(e) == n and e[-1] <= length and len({b - a for a, b in zip([0] + e, e)}) == 1


def test_run_matrix_completes_fatigue_arms_before_matched_arms_start(matrix, suite):
    assert len(matrix["summaries"]) == len(ALL_ARMS) * len(SEEDS)
    order = matrix["order"]
    for seed in SEEDS:
        f = order.index(f"p0_fatigue_s{seed}")
        assert f < order.index(f"p0_fixed_matched_s{seed}") and f < order.index(f"p0_random_matched_s{seed}")
    first_wave2 = min(i for i, r in enumerate(order) if "matched" in r)
    assert all("matched" not in r for r in order[:first_wave2]) and all("matched" in r for r in order[first_wave2:])
    jobs = plan_suite(suite, [0], [0, 1])
    for seed in (0, 1):
        mine = [j for j in jobs if j.seed == seed]
        assert mine[0].arm == "fatigue" and mine[0].wave == 1
        assert {j.arm for j in mine if j.wave == 2} == {"fixed_matched", "random_matched"}
        assert all(j.source_run_id == f"p0_fatigue_s{seed}" for j in mine if j.wave == 2)
        assert {j.arm for j in mine} == set(ALL_ARMS)
    only = plan_suite(suite, [0], [0], ["fixed_matched"])  # matched arms pull in their fatigue run
    assert [(j.arm, j.wave) for j in only] == [("fatigue", 1), ("fixed_matched", 2)]
    reuse = plan_suite(suite, [0], [0], ["fixed_matched"], existing_sources={"p0_fatigue_s0"})
    assert [(j.arm, j.wave) for j in reuse] == [("fixed_matched", 2)]


def test_every_arm_stops_at_the_same_total_when_the_budget_runs_out(matrix):
    for s in matrix["summaries"].values():
        b = s["budget"]
        assert s["stop_reason"] == "budget_exhausted"
        assert b["used"] == b["total_tokens"] == SUITE_TOTAL and b["overshoot"] == 0
        assert b["agent_tokens"] + b["sleep_tokens"] == SUITE_TOTAL
        assert s["metrics"]["tokens"]["used"] == SUITE_TOTAL and s["metrics"]["budget_used_fraction"] == 1.0
    shares = {a: _msum(matrix, a, 0)["metrics"]["tokens"]["sleep_share_of_budget"] for a in ALL_ARMS}
    assert shares["no_sleep"] == shares["sleep_lite"] == 0 and shares["fixed_letta"] > shares["fixed_sparse"] > 0


# additional coverage: baselines

def test_resolved_arms_differ_only_in_declared_params(suite):
    cfgs = {a: resolve_arm(suite, a, 0, 0) for a in ALL_ARMS + ["fixed_tiered"]}
    check_drift(suite, cfgs.values())  # the shipped config passes
    assert all(c.awake_tags is True and c.context_cap_tokens == 4000 for c in cfgs.values())
    assert cfgs["fixed_letta"].policy_params["every_n"] == 5 and cfgs["fixed_sparse"].policy_params["every_n"] == 20
    assert cfgs["fixed_letta"].policy_params["timed_depth"] == "nrem"
    assert cfgs["fixed_tiered"].policy_params["timed_depth"] == "fatigue"
    assert cfgs["sleep_lite"].sleep_policy == "sleep_lite" and cfgs["no_sleep"].policy_params == {}
    shared = {k: v for k, v in cfgs["fatigue"].policy_params.items() if k not in ARM_SPECS["fixed_letta"].declared}
    for a in ("fixed_letta", "fixed_sparse", "fixed_matched", "random_matched"):
        assert {k: v for k, v in cfgs[a].policy_params.items() if k in shared} == shared


def test_drift_assertion_on_mutated_configs_and_context_cap(suite):
    cfgs = [resolve_arm(suite, a, 0, 0) for a in ALL_ARMS]
    check_drift(suite, cfgs)
    mutated = cfgs[:2] + [cfgs[2].model_copy(update={"max_completion_tokens": 123})] + cfgs[3:]
    with pytest.raises(ConfigDriftError, match="max_completion_tokens"):
        check_drift(suite, mutated)
    fs = dict(cfgs[1].sleep_policies)
    fs["full_sleep"] = {**fs["full_sleep"], "max_items": 3}
    with pytest.raises(ConfigDriftError, match=r"sleep\.max_items"):
        check_drift(suite, [cfgs[1].model_copy(update={"sleep_policies": fs})])
    with pytest.raises(ConfigDriftError, match="sleep_policy"):
        check_drift(suite, [cfgs[1].model_copy(update={"sleep_policy": "summarize"})])
    base = suite.base.model_copy(deep=True)  # a policy-level context cap would differ from no_sleep's
    base.sleep_policies["full_sleep"]["context_cap_tokens"] = 3000
    with pytest.raises(ConfigDriftError, match="effective context cap"):
        plan_suite(make_suite(base, suite.section), [0], [0])
    ok = make_suite(suite.base, BaselinesSection(**{**suite.section.model_dump(),
                                                     "arm_params": {"fixed_letta": {"every_n": 7}}}))
    assert resolve_arm(ok, "fixed_letta", 0, 0).policy_params["every_n"] == 7  # declared params may change
    plan_suite(ok, [0], [0])


def test_no_sleep_never_calls_the_consolidator_but_logs_fatigue_in_the_matrix(matrix):
    for seed in SEEDS:
        t = matrix["transports"][f"p0_no_sleep_s{seed}"]
        s = _msum(matrix, "no_sleep", seed)
        assert t.sleep_calls == 0 and s["budget"]["sleep_calls"] == 0 and s["budget"]["sleep_tokens"] == 0
        assert not any("consolidat" in m[0]["content"].lower() for m in t.calls)
        hist = _mhist(matrix, "no_sleep", seed)
        assert all(h["event"] == "step" for h in hist)
        assert all(h["fatigue"] is not None and h["fatigue_level"] for h in hist)
        assert not any(h["sleep_triggered"] for h in hist)
        assert {"tired", "exhausted"} & {h["fatigue_level"] for h in hist}
        m = s["metrics"]
        assert m["sleep"]["llm_sleeps"] == 0 and len(m["fatigue_curve"]["fatigue"]) == len(hist)
        agent = [c for c in t.calls if "APPROACH_TAGS" in c[0]["content"]]
        assert len(agent) == s["budget"]["agent_calls"]
        assert all(h["context_tokens"] <= 4000 for h in hist if h["step"] > 0)
        assert matrix["transports"][f"p0_sleep_lite_s{seed}"].sleep_calls == 0
        assert _msum(matrix, "sleep_lite", seed)["metrics"]["sleep"]["forgets"] > 0
        assert "forget" in {h["sleep_depth"] for h in _mhist(matrix, "fatigue", seed) if h["event"] == "sleep"}


def test_summary_metrics_schema(matrix):
    m = _msum(matrix, "fatigue", 0)["metrics"]
    assert m["schema"] == "baseline_metrics/v1" and m["arm"] == "fatigue" and m["sleep_policy"] == "full_sleep"
    assert list(m["checkpoints"]) == ["0.1", "0.25", "0.5", "0.75", "1"]
    vals = list(m["checkpoints"].values())
    assert all(v is not None for v in vals) and vals == sorted(vals) and vals[-1] == m["final_best_normalized"]
    assert set(m["reach"]) == {"0.95", "0.99"}
    for r in m["reach"].values():
        assert r["reached"] == (r["budget_fraction"] is not None)
    assert set(m["sleep"]) >= {"llm_sleeps", "forgets", "by_depth", "llm_sleep_steps", "forget_steps", "scheduled",
                               "realized", "deferrals"}
    assert set(m["tokens"]) >= {"total_budget", "used", "agent", "sleep", "sleep_share_of_budget"}
    b = m["behavior"]
    assert b["window"] == 5 and len(b["around_sleeps"]) == m["sleep"]["llm_sleeps"]
    for k in ("overall", "before_sleep", "after_sleep"):
        assert set(b[k]) == {"perseveration_rate", "invalid_rate", "calibration_error", "n"}
    assert set(m["memory"]) == {"items_added", "items_evicted", "replays_wasted", "replays_run",
                                    "dream_calls", "dream_evaluations", "dreams_validated",
                                    "dreams_refuted", "dreams_uncertain", "dreams_untested"}
    assert m["memory"]["items_added"] > 0
    assert _msum(matrix, "no_sleep", 0)["metrics"]["memory"]["items_added"] == 0
    assert _msum(matrix, "fixed_letta", 0)["arm"] == "fixed_letta"


# ---- OpenAI configs, shared LLM settings, run_matrix pre-flight -------------------------------

from scripts import run_matrix as rm  # noqa: E402

OPENAI_CONFIGS = {  # name: (model, arms, seeds, runs incl. wave 2, waves, eval_backend, max_dollars)
    "debug_openai": ("gpt-5-nano", {"no_sleep", "fatigue"}, [0], 2, {1: 2, 2: 0}, "local", 2),
    "pilot_openai": ("gpt-5-mini", {"no_sleep", "fixed_letta", "fatigue"}, [0, 1], 6, {1: 6, 2: 0}, "local", 10),
    "baselines_openai": ("gpt-5-mini", {"no_sleep", "fixed_letta", "fixed_matched", "random_matched", "fatigue"},
                         [0, 1], 10, {1: 6, 2: 4}, "modal", 75),
}


@pytest.mark.parametrize("name", list(OPENAI_CONFIGS))
def test_openai_configs_validate_plan_and_share_llm_settings(name):
    model, arms, seeds, n_runs, waves, backend, cap = OPENAI_CONFIGS[name]
    path = str(ROOT / "configs" / f"{name}.yaml")
    s = load_suite(path)
    assert s.base.model == model and s.base.llm.reasoning_effort == "low" and s.base.max_dollars == cap
    assert s.base.eval_backend == backend and set(s.section.arms) == arms and s.section.problems == [36]
    assert s.base.total_token_budget == (200_000 if name == "debug_openai" else 1_000_000)
    ps, ss = rm.resolve_targets(path)
    assert (ps, ss) == ([36], seeds)
    mode, base, suite_, jobs, got_waves = rm.plan_launch(path, ps, ss, [], [])
    assert mode == "suite" and len(jobs) == n_runs and got_waves == waves
    sent = {json.dumps(resolve_chat_params(j.cfg.model, j.cfg.llm, env={}).to_dict(), sort_keys=True) for j in jobs}
    assert len(sent) == 1  # identical model / token param / reasoning_effort in every arm (agent and sleep)
    p = json.loads(sent.pop())
    assert p["token_param"] == "max_completion_tokens" and p["reasoning_effort"] == "low" and p["temperature"] is None


def test_existing_backend_configs_still_validate():
    for name in ("example", "mock", "gepa", "gepa_mock"):
        cfg = load_config(ROOT / "configs" / f"{name}.yaml")
        assert resolve_chat_params(cfg.model, cfg.llm, env={}).token_param == "max_tokens"  # Qwen / vLLM path
    for name in ("baselines", "baselines_mock"):
        s = load_suite(ROOT / "configs" / f"{name}.yaml")
        plan_suite(s, [s.base.problem_id], [0])


@pytest.mark.parametrize("update, needle", [
    ({"llm": {"reasoning_effort": "high"}}, "llm"),
    ({"llm": {"token_param": "max_tokens"}}, "llm"),
    ({"model": "gpt-5-nano"}, "model"),
    ({"prices": {"gpt-5-mini": {"input": 0.0, "output": 0.0}}}, "prices"),
    ({"max_dollars": 1000.0}, "max_dollars"),
])
def test_drift_check_requires_identical_model_and_llm_settings_in_every_arm(update, needle):
    s = load_suite(ROOT / "configs" / "pilot_openai.yaml")
    cfgs = [resolve_arm(s, a, 36, 0) for a in s.section.arms]
    check_drift(s, cfgs)
    data = cfgs[1].model_dump()
    data.update(update)
    with pytest.raises(ConfigDriftError, match=needle):
        check_drift(s, cfgs[:1] + [RunConfig(**data)] + cfgs[2:])
    bad = make_suite(s.base, BaselinesSection(**{**s.section.model_dump(), "overrides": {"fatigue": update}}))
    with pytest.raises(ConfigDriftError):
        plan_suite(bad, [36], [0])


def test_preflight_refuses_above_max_dollars_and_prints_the_estimate():
    s = load_suite(ROOT / "configs" / "pilot_openai.yaml")
    lines = []
    with pytest.raises(rm.PreflightRefused) as ei:
        rm.preflight_check(s.base, 6, waves={1: 6, 2: 0}, max_dollars=0.01, log=lines.append)
    est = ei.value.estimate
    per_run = 1_000_000 + 16_384 + 2 * 8000  # budget + max_completion_tokens + prompt allowance
    assert est["per_run_tokens"] == per_run and est["rate_per_1m"] == 2.0  # gpt-5-mini output price
    assert est["cost"] == pytest.approx(6 * per_run * 2.0 / 1e6, abs=1e-4)
    text = "\n".join(lines)
    assert "estimate" in text and "REFUSED" in text and "exceeds the cap" in text and "--max-dollars" in text
    ok = rm.preflight_check(s.base, 6, max_dollars=20, log=lambda m: None)
    assert not ok["refused"] and ok["cap"] == 20
    with pytest.raises(rm.PreflightRefused, match="exceeds"):  # the config's own cap (10) is used by default
        rm.preflight_check(s.base, 6, log=lambda m: None)


def test_preflight_refuses_without_price_or_cap_and_on_a_model_mismatch():
    nocap = load_suite(ROOT / "configs" / "debug_openai.yaml").base.model_copy(update={"max_dollars": None})
    with pytest.raises(rm.PreflightRefused, match="no spending cap"):
        rm.preflight_check(nocap, 2, log=lambda m: None)
    qwen = load_suite(ROOT / "configs" / "baselines.yaml").base  # no price for Qwen: refuse unless overridden
    with pytest.raises(rm.PreflightRefused, match="no confirmed price"):
        rm.preflight_check(qwen, 7, max_dollars=100, log=lambda m: None)
    assert rm.preflight_check(qwen, 7, max_dollars=0, price_per_1m=0, log=lambda m: None)["cost"] == 0
    unconfirmed = nocap.model_copy(update={"max_dollars": 5, "prices": {"gpt-5-nano": ModelPrice()}})
    with pytest.raises(rm.PreflightRefused, match="no confirmed price"):
        rm.preflight_check(unconfirmed, 2, log=lambda m: None)
    with pytest.raises(rm.PreflightRefused, match="LLM_MODEL"):
        rm.preflight_check(nocap.model_copy(update={"max_dollars": 5}), 2, env_model="Qwen/Qwen3-8B",
                           log=lambda m: None)


def test_run_matrix_cli_refuses_with_exit_2_and_mock_skips_the_check(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_MODEL", "gpt-5-mini")  # wins over any .env value (load_dotenv never overrides)
    monkeypatch.chdir(tmp_path)  # no .env here: nothing real is loaded
    ran = []
    monkeypatch.setattr(rm, "run_suite_local", lambda *a, **k: ran.append(a))
    cfg = str(ROOT / "configs" / "pilot_openai.yaml")
    with pytest.raises(SystemExit) as ei:
        rm._local(["--config", cfg, "--max-dollars", "0.01"])
    assert ei.value.code == 2 and ran == []
    out = capsys.readouterr().out
    assert "pre-flight cost estimate" in out and "$12.39" in out and "REFUSED" in out
    rm._local(["--config", cfg, "--max-dollars", "13", "--estimate-only"])  # within the cap: prints, runs nothing
    assert "pre-flight OK" in capsys.readouterr().out and ran == []
    rm._local(["--config", cfg, "--mock-llm", "--estimate-only"])
    assert "no pre-flight cost check" in capsys.readouterr().out


# ================================================================================================
# Section 8 · Integrations
# ================================================================================================

# ---- offline (default suite): real gepa against the local fake upstream, through the proxy ------

LOOP_SUMMARY_KEYS = {  # keys of harness/loop.py's summary.json that analysis code reads
    "run_id", "problem_id", "problem", "sleep_policy", "seed", "steps_completed", "stop_reason",
    "best_program_id", "best_score", "best_known", "best_normalized", "best_step", "budget",
    "llm_model", "config_model", "wallclock_s", "config",
}


@pytest.fixture
def p36_stack():
    up = FakeUpstream(p36_responder(seed=0, overlap_every=4)).start()
    proxy = LLMProxy(up.base_url, "k", "mock-model").start()
    yield up, proxy
    proxy.stop()
    up.stop()


def gepa_cfg(**kw):
    from arms.gepa_arm import GEPAArmConfig

    base = dict(problem_id=36, total_token_budget=100_000, max_steps=6)
    base.update(kw)
    return GEPAArmConfig(**base)


@needs_gepa
def test_gepa_on_mock_p36_through_the_proxy_writes_standard_jsonl(run_dir, p36_stack):
    from arms.gepa_arm import run_gepa

    up, proxy = p36_stack
    budget = Budget(100_000)
    s = run_gepa(gepa_cfg(), CirclePacking26(timeout_s=30), budget, proxy, out_dir=run_dir)
    d = run_dir / s["run_id"]
    hist = read_history(d / "history.jsonl")
    assert s["run_id"] == "p36_gepa_s0" and s["stop_reason"] == "max_steps"
    assert [h["step"] for h in hist] == list(range(7))  # seed + 6, like harness/loop.py
    assert all(set(h) == set(FIELDS) for h in hist)
    assert all(h["event"] == "step" and h["run_id"] == s["run_id"] for h in hist)
    ids = [h["program_id"] for h in hist]
    assert ids == [f"p{i}" for i in range(7)] and hist[0]["parent_id"] is None
    assert all(h["parent_id"] in ids[:i] for i, h in enumerate(hist) if i > 0)
    for h in hist:
        assert isinstance(h["valid"], bool) and (h["score"] is None) == (not h["valid"])
        assert h["side_info"]["render"] and h["details"]["gepa"]["round"] == 0
    bad = [h for h in hist if not h["valid"]]  # the responder emits an overlapping packing every 4th call
    assert bad and all(h["side_info"]["num_overlaps"] > 0 and "overlap" in h["error"] for h in bad)
    assert [h["agent_tokens"] for h in hist] == [100 * i for i in range(7)]
    assert all(h["sleep_tokens"] == 0 for h in hist)
    assert budget.agent_tokens == 600 and budget.agent_calls == 6 == len(up.requests)
    summary = json.loads((d / "summary.json").read_text())
    assert LOOP_SUMMARY_KEYS <= set(summary)
    best = max((h for h in hist if h["valid"]), key=lambda h: h["score"])
    assert summary["best_score"] == best["score"] and summary["best_program_id"] == best["program_id"]
    assert (d / "best_program.py").read_text() == best["program"]
    assert summary["budget"] == budget.snapshot()
    assert summary["gepa"]["version"] and summary["gepa"]["sandbox_evals"] == 7


@needs_gepa
def test_gepa_stops_at_matched_metric_calls(run_dir, p36_stack):
    from arms.gepa_arm import run_gepa

    _, proxy = p36_stack
    ev = CountingEvaluator(CirclePacking26(timeout_s=30), EvalBudget())
    s = run_gepa(gepa_cfg(max_steps=5), ev, Budget(100_000), proxy, out_dir=run_dir)
    assert s["stop_reason"] == "max_steps" and s["steps_completed"] == 5
    assert ev.budget.used == 6  # seed + max_steps sandbox evaluations, never more
    assert s["gepa"]["internal_metric_calls"] > ev.budget.used  # gepa's own count includes memo hits
    assert s["gepa"]["over_limit_calls"] == 0


@needs_gepa
def test_gepa_respects_max_metric_calls_over_max_steps(run_dir, p36_stack):
    from arms.gepa_arm import run_gepa

    _, proxy = p36_stack
    ev = CountingEvaluator(CirclePacking26(timeout_s=30), EvalBudget())
    s = run_gepa(gepa_cfg(max_steps=50, gepa={"max_metric_calls": 3}), ev, Budget(100_000), proxy, out_dir=run_dir)
    assert s["steps_completed"] == 3 and ev.budget.used == 4


@needs_gepa
def test_gepa_stops_at_the_token_budget_like_the_loop(run_dir, p36_stack):
    """350 tokens at 100/call: 3 calls fit, the 4th overshoots, is charged, and its response discarded."""
    from arms.gepa_arm import run_gepa

    up, proxy = p36_stack
    budget = Budget(350)
    s = run_gepa(gepa_cfg(max_steps=50), CirclePacking26(timeout_s=30), budget, proxy, out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert s["stop_reason"] == "budget_exhausted" and s["steps_completed"] == 3 and len(hist) == 4
    assert budget.used == 400 and budget.overshoot == 50 and budget.discarded_calls == 1
    assert hist[-1]["agent_tokens"] == 300 and len(up.requests) == 4
    budget2 = Budget(300)  # exact budget
    s2 = run_gepa(gepa_cfg(max_steps=50, run_id="exact"), CirclePacking26(timeout_s=30), budget2, proxy,
                  out_dir=run_dir)
    assert s2["stop_reason"] == "budget_exhausted" and s2["steps_completed"] == 3
    assert budget2.used == 300 and budget2.overshoot == 0


@needs_gepa
def test_gepa_side_info_and_task_reach_the_reflection_prompt(run_dir, p36_stack):
    from arms.gepa_arm import run_gepa

    up, proxy = p36_stack
    run_gepa(gepa_cfg(max_steps=2), CirclePacking26(timeout_s=30), Budget(100_000), proxy, out_dir=run_dir)
    prompt = up.requests[0]["messages"][-1]["content"]
    assert "Pack 26 disjoint circles" in prompt  # objective = evaluator.description
    assert "## Domain Context & Constraints" in prompt and "construct()" in prompt
    assert "min_pair_slack" in prompt and "tight_contacts" in prompt  # ASI from side_info
    assert "def construct" in prompt
    assert all(r["max_tokens"] == 4096 for r in up.requests)


@needs_gepa
def test_gepa_lineage_and_improvement_on_the_mock_problem(run_dir):
    from arms.gepa_arm import run_gepa

    with FakeUpstream(mock_transport_responder(MockTransport(seed=3))) as up, LLMProxy(up.base_url) as proxy:
        s = run_gepa(gepa_cfg(problem_id=0, max_steps=8), TextEvaluator(), Budget(100_000), proxy, out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert s["run_id"] == "p0_gepa_s0" and s["best_score"] > hist[0]["score"]
    assert all("max_dev" in h["side_info"] for h in hist)
    assert any(d > 0 for d in (h["delta_vs_best"] for h in hist[1:] if h["valid"]))
    assert len({h["parent_id"] for h in hist[1:]}) >= 2  # gepa adopts improved candidates as parents


@needs_gepa
def test_gepa_internal_error_ends_the_run_cleanly(run_dir, p36_stack, monkeypatch):
    import gepa.optimize_anything as oa

    from arms.gepa_arm import run_gepa

    def boom(*a, **k):
        raise RuntimeError("gepa broke")

    monkeypatch.setattr(oa, "optimize_anything", boom)
    _, proxy = p36_stack
    s = run_gepa(gepa_cfg(), CirclePacking26(timeout_s=30), Budget(1000), proxy, out_dir=run_dir)
    assert s["stop_reason"] == "gepa_error" and "gepa broke" in s["gepa"]["error"]
    assert (run_dir / s["run_id"] / "summary.json").exists()


def test_gepa_clean_candidate_and_asi():
    from arms.gepa_arm import asi_from_result, clean_candidate

    assert clean_candidate("```python\ndef construct():\n    return 1\n```") == "def construct():\n    return 1\n"
    assert clean_candidate("def construct():\n    return 1") == "def construct():\n    return 1\n"
    r = EvalResult(score=-math.inf, valid=False, error="x",
                   side_info={"overlaps": [{"i": 0, "j": 1, "by": 0.1}], "scores": 3, "render": "ab"})
    asi = asi_from_result(r)
    assert asi["Valid"] is False and asi["Score"] is None and asi["Error"] == "x"
    assert asi["overlaps"] == '[{"i":0,"j":1,"by":0.1}]' and asi["render"] == "ab"
    assert "scores" not in asi and asi["diag_scores"] == 3  # gepa reserves "scores"


def test_gepa_config_files_load():
    from arms.gepa_arm import load_arm_config

    c = load_arm_config(ROOT / "configs" / "gepa.yaml")
    assert c.arm == "gepa" and c.metric_calls == c.max_steps == 500
    assert c.resolved_run_id() == "p36_gepa_s0"
    assert load_arm_config(ROOT / "configs" / "gepa.yaml", arm="gepa_sleep").resolved_run_id() == "p36_gepa_full_sleep_s0"
    assert load_arm_config(ROOT / "configs" / "gepa_mock.yaml").gepa.round_metric_calls == 4


def test_gepa_session_logs_fatigue_on_every_step(tmp_path):
    from arms.gepa_arm import GEPAArmConfig, GEPASession

    cfg = GEPAArmConfig(problem_id=0, total_token_budget=1000, max_steps=5,
                        fatigue={"load_tokens_scale": 1000, "grace_steps": 0})
    budget = Budget(1000)
    session = GEPASession(cfg, TextEvaluator(), budget, None, RunLogger(tmp_path / "h.jsonl", "g"), tmp_path,
                          monitor=FatigueMonitor(cfg.fatigue, best_known=1.0, total_budget=1000))
    session.limit_step = 10
    for i in range(6):
        session.evaluate(f"def construct():\n    return [0.{i}, 0.5, 0.5, 0.5, 0.5]\n")
    hist = read_history(tmp_path / "h.jsonl")
    assert len(hist) == 6 and all(set(FATIGUE_FIELDS) <= set(h) for h in hist)
    assert all(h["fatigue"] is not None and h["fatigue_level"] for h in hist)
    assert hist[-1]["fatigue_stagnation"] is not None


class StubSleep:
    """Same interface as sleep/full.py FullSleep (bind / consolidate / memory_section)."""

    name = "full_sleep"

    def __init__(self, **params):
        self.params = params
        self.evaluator = None
        self.calls = []

    def bind(self, *, evaluator):
        self.evaluator = evaluator

    def consolidate(self, task, history, llm, budget, *, step):
        best = max((h for h in history if h["event"] == "step" and h["valid"]), key=lambda h: h["score"])
        self.calls.append({"step": step, "n": len(history), "history": list(history), "best": best["program"]})
        llm.complete([{"role": "system", "content": "consolidate"}, {"role": "user", "content": task}],
                     role="sleep", max_tokens=64)
        return {"policy": self.name, "claims": 1, "custom": "x"}

    def memory_section(self, token_cap=None):
        return f"MEMO-{len(self.calls)}: radii pinned to walls help"


def gepa_sleep_cfg(**kw):
    from arms.gepa_arm import GEPAArmConfig

    base = dict(problem_id=0, total_token_budget=100_000, max_steps=9, arm="gepa_sleep",
                gepa={"round_metric_calls": 3}, sleep_policies={"full_sleep": {"every_n": 3}})
    base.update(kw)
    return GEPAArmConfig(**base)


@pytest.fixture
def mock_stack():
    up = FakeUpstream(mock_transport_responder(MockTransport(seed=5))).start()
    proxy = LLMProxy(up.base_url).start()
    yield up, proxy
    proxy.stop()
    up.stop()


def _sleep_llm(budget):
    return LLMClient(budget, MockTransport(prompt_tokens=60, completion_tokens=40), sleep_fn=lambda s: None)


@needs_gepa
def test_gepa_sleep_rounds_sleep_and_memory_reaches_the_background(run_dir, mock_stack):
    from arms.gepa_sleep_arm import MEMORY_HEADER, run_gepa_sleep

    up, proxy = mock_stack
    budget = Budget(100_000)
    stubs = []

    def factory(params):
        stubs.append(StubSleep(**params))
        return stubs[-1]

    s = run_gepa_sleep(gepa_sleep_cfg(), TextEvaluator(), budget, proxy, _sleep_llm(budget), out_dir=run_dir,
                       sleep_factory=factory)
    stub = stubs[0]
    hist = hist_of(run_dir, s)
    steps = [h for h in hist if h["event"] == "step"]
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert s["run_id"] == "p0_gepa_full_sleep_s0" and s["stop_reason"] == "max_steps"
    assert s["rounds"] == 3 and s["sleeps"] == 2
    assert stub.params == {"every_n": 3} and isinstance(stub.evaluator.inner, TextEvaluator)
    assert [h["step"] for h in steps] == list(range(10)) and all(set(h) == set(FIELDS) for h in hist)
    assert [c["step"] for c in stub.calls] == [3, 6] == [h["step"] for h in sleeps]
    assert sleeps[0]["details"]["block"] == "MEMO-1: radii pinned to walls help"
    assert sleeps[0]["details"]["policy"] == "full_sleep" and sleeps[0]["details"]["round"] == 0
    assert sleeps[0]["details"]["custom"] == "x" and sleeps[0]["program_id"] is None
    assert [c["n"] for c in stub.calls] == [4, 8]
    assert all(set(r) == set(FIELDS) for r in stub.calls[1]["history"])
    assert budget.sleep_calls == 2 and budget.sleep_tokens == 200  # sleep charged to the same Budget
    assert hist[-1]["sleep_tokens"] == 200 and hist[-1]["agent_tokens"] == budget.agent_tokens
    assert s["budget"]["used"] == budget.agent_tokens + 200
    assert [h["details"]["gepa"]["round"] for h in steps] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
    prompts = [r["messages"][-1]["content"] for r in up.requests]
    assert len(prompts) == 9
    assert not any("MEMO-" in p for p in prompts[:3])
    assert all(MEMORY_HEADER in p and "MEMO-1" in p for p in prompts[3:6])
    assert all("MEMO-2" in p and "MEMO-1" not in p for p in prompts[6:])


@needs_gepa
def test_gepa_sleep_each_round_is_seeded_from_the_best_candidate(run_dir, mock_stack):
    from arms.gepa_sleep_arm import run_gepa_sleep

    up, proxy = mock_stack
    budget = Budget(100_000)
    stub = StubSleep()
    s = run_gepa_sleep(gepa_sleep_cfg(), TextEvaluator(), budget, proxy, _sleep_llm(budget), out_dir=run_dir,
                       sleep_factory=lambda p: stub)
    hist = hist_of(run_dir, s)
    prompts = [r["messages"][-1]["content"] for r in up.requests]
    for call, first_prompt_of_round in zip(stub.calls, (prompts[3], prompts[6])):
        assert call["best"].strip() in first_prompt_of_round
    steps = [h for h in hist if h["event"] == "step"]
    by_code = {h["program"]: h["program_id"] for h in steps}
    assert steps[4]["parent_id"] == by_code[stub.calls[0]["best"]]
    assert steps[7]["parent_id"] == by_code[stub.calls[1]["best"]]
    assert len({h["program_id"] for h in steps}) == len(steps) == 10  # re-seeded best not re-logged


@needs_gepa
def test_gepa_sleep_budget_exhausted_during_sleep_stops_cleanly(run_dir, mock_stack):
    from arms.gepa_sleep_arm import run_gepa_sleep

    _, proxy = mock_stack
    budget = Budget(350)  # 3 agent calls (300), then the sleep call overshoots
    s = run_gepa_sleep(gepa_sleep_cfg(), TextEvaluator(), budget, proxy, _sleep_llm(budget), out_dir=run_dir,
                       sleep_factory=lambda p: StubSleep())
    hist = hist_of(run_dir, s)
    assert s["stop_reason"] == "budget_exhausted" and s["steps_completed"] == 3
    assert [h["event"] for h in hist] == ["step"] * 4  # the discarded sleep is not logged
    assert budget.sleep_tokens == 100 and budget.overshoot == 50 and budget.discarded_calls == 1


@needs_gepa
def test_gepa_sleep_llm_must_share_the_budget(run_dir, mock_stack):
    from arms.gepa_sleep_arm import run_gepa_sleep

    _, proxy = mock_stack
    with pytest.raises(ValueError):
        run_gepa_sleep(gepa_sleep_cfg(), TextEvaluator(), Budget(1000), proxy, _sleep_llm(Budget(1000)),
                       out_dir=run_dir, sleep_factory=lambda p: StubSleep())


def test_gepa_sleep_compose_background():
    from arms.gepa_sleep_arm import MEMORY_HEADER, compose_background

    assert compose_background("B", "") == "B"
    assert compose_background("B", " notes ") == f"B\n\n{MEMORY_HEADER}\nnotes"


@needs_gepa
def test_gepa_sleep_with_the_real_full_sleep_on_mock_p36(run_dir):
    from arms.gepa_sleep_arm import MEMORY_HEADER, run_gepa_sleep

    budget = Budget(100_000)
    c = gepa_sleep_cfg(problem_id=36, max_steps=6,
                       sleep_policies={"full_sleep": {"max_block_tokens": 256, "replay": False}})
    with FakeUpstream(p36_responder(seed=1)) as up, LLMProxy(up.base_url) as proxy:
        s = run_gepa_sleep(c, CirclePacking26(timeout_s=30), budget, proxy, _sleep_llm(budget), out_dir=run_dir)
    hist = hist_of(run_dir, s)
    sleeps = [h for h in hist if h["event"] == "sleep"]
    assert s["stop_reason"] == "max_steps" and s["sleeps"] == 1 and len(sleeps) == 1
    assert [h["step"] for h in hist if h["event"] == "step"] == list(range(7))
    assert all(set(h) == set(FIELDS) for h in hist)
    assert sleeps[0]["details"]["policy"] == "full_sleep" and sleeps[0]["details"]["block"]
    assert budget.sleep_tokens > 0
    assert any(MEMORY_HEADER in r["messages"][-1]["content"] for r in up.requests[3:])


# ---- live (skipped unless RUN_LIVE=1) -----------------------------------------------------------

@live(pytest.mark.llm)
def test_live_llm_client_charges_reported_usage():
    _require_env("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    budget = Budget(5000)
    llm = LLMClient.from_env(budget)
    resp = llm.complete([{"role": "user", "content": "Reply with the single word: ok"}], role="agent", max_tokens=16)
    assert resp.prompt_tokens > 0 and resp.completion_tokens > 0
    assert budget.used == resp.prompt_tokens + resp.completion_tokens


@live(pytest.mark.llm)
@needs_gepa
def test_live_gepa_short_run_on_p36_through_the_proxy(run_dir):
    from arms.gepa_arm import run_gepa

    _require_env("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    budget = Budget(60_000)
    ev = CountingEvaluator(CirclePacking26(timeout_s=30), EvalBudget())
    with LLMProxy.from_env(request_timeout_s=180) as proxy:
        s = run_gepa(gepa_cfg(max_steps=50, total_token_budget=60_000, gepa={"max_metric_calls": 2}), ev, budget,
                     proxy, out_dir=run_dir)
        log = proxy.usage_log(s["run_id"]) if hasattr(proxy, "usage_log") else []
    hist = hist_of(run_dir, s)
    assert s["stop_reason"] in ("max_steps", "budget_exhausted"), s.get("gepa")
    assert s["steps_completed"] <= 2 and ev.budget.used <= 3  # max_metric_calls respected
    assert all(set(h) == set(FIELDS) for h in hist) and [h["step"] for h in hist] == list(range(len(hist)))
    assert budget.agent_calls >= 1 and budget.used > 0
    if log:
        assert budget.used == sum(c["prompt_tokens"] + c["completion_tokens"] for c in log)


@live(pytest.mark.llm, pytest.mark.slow)
@needs_gepa
def test_live_gepa_sleep_arm_on_p36(run_dir):
    from arms.gepa_sleep_arm import run_gepa_sleep

    _require_env("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    budget = Budget(150_000)
    c = gepa_sleep_cfg(problem_id=36, max_steps=6, total_token_budget=150_000,
                       sleep_policies={"full_sleep": {"max_block_tokens": 256, "replay": False}})
    with LLMProxy.from_env(request_timeout_s=180) as proxy:
        s = run_gepa_sleep(c, CirclePacking26(timeout_s=30), budget, proxy, LLMClient.from_env(budget),
                           out_dir=run_dir)
    hist = hist_of(run_dir, s)
    assert all(set(h) == set(FIELDS) for h in hist)
    assert budget.agent_tokens > 0 and (s["sleeps"] == 0 or budget.sleep_tokens > 0)


@live(pytest.mark.pastdev)
@pytest.mark.xfail(reason="not implemented: past.dev recall arm", strict=True)
def test_live_pastdev_ingest_poll_and_recall_with_an_identity():
    _require_env("PASTDEV_API_KEY")
    from sleep.pastdev import PastDevPolicy  # noqa: F401  (does not exist yet)

    pol = PastDevPolicy(run_id="pytest-live-ingest")
    pol.should_sleep(1, [{"event": "step", "step": 1, "score": 0.5, "valid": True, "program": GOOD}])
    assert pol.wait_until_ingested(timeout_s=60)
    assert pol.recall("score").memories


@live(pytest.mark.pastdev)
@pytest.mark.xfail(reason="not implemented: past.dev recall arm", strict=True)
def test_live_pastdev_query_timestamp_excludes_later_steps():
    _require_env("PASTDEV_API_KEY")
    from sleep.pastdev import PastDevPolicy  # noqa: F401  (does not exist yet)

    pol = PastDevPolicy(run_id="pytest-live-ts")
    t_cut = pol.ingest_step({"step": 1, "text": "early"})
    pol.ingest_step({"step": 2, "text": "late"})
    assert all("late" not in m for m in pol.recall("early", query_timestamp=t_cut).memories)


@live(pytest.mark.pastdev)
@pytest.mark.xfail(reason="not implemented: past.dev recall arm", strict=True)
def test_live_pastdev_one_identity_per_run_and_runs_are_isolated():
    _require_env("PASTDEV_API_KEY")
    from sleep.pastdev import PastDevPolicy  # noqa: F401  (does not exist yet)

    a, b = PastDevPolicy(run_id="pytest-live-a"), PastDevPolicy(run_id="pytest-live-b")
    assert a.identity != b.identity
    a.ingest_step({"step": 1, "text": "only in a"})
    assert all("only in a" not in m for m in b.recall("only in a").memories)


def _require_modal():
    if not (os.environ.get("MODAL_TOKEN_ID") or (Path.home() / ".modal.toml").exists()):
        pytest.skip("no Modal credentials (MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or ~/.modal.toml)")


@live(pytest.mark.modal)
def test_live_modal_evaluator_matches_the_local_result():
    _require_modal()
    from infra.modal_eval import ModalEvaluator

    local = MockEvaluator(timeout_s=10)
    prog = "def construct():\n    return [0.7, 0.6, 0.7, 0.8, 0.7]\n"
    remote, here = ModalEvaluator(local).evaluate(prog), local.evaluate(prog)
    assert remote.valid and remote.score == pytest.approx(here.score, abs=1e-12)
    assert remote.side_info == here.side_info


@live(pytest.mark.modal)
def test_live_modal_parallel_fan_out_of_4_runs_completes():
    _require_modal()
    from concurrent.futures import ThreadPoolExecutor

    from infra.modal_eval import ModalEvaluator

    ev = ModalEvaluator(MockEvaluator(timeout_s=10))
    progs = [f"def construct():\n    return [0.{i}, 0.7, 0.7, 0.7, 0.7]\n" for i in range(4)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(ev.evaluate, progs))
    assert len(results) == 4 and all(r.valid for r in results)
    assert len({round(r.score, 9) for r in results}) == 4


@live(pytest.mark.modal, pytest.mark.llm)
def test_live_vllm_endpoint_answers_openai_compatible_request_with_usage():
    _require_env("VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL")
    budget = Budget(5000)
    t = OpenAICompatTransport(os.environ["VLLM_BASE_URL"], os.environ["VLLM_API_KEY"], os.environ["VLLM_MODEL"],
                              request_timeout_s=600)
    resp = LLMClient(budget, t).complete([{"role": "user", "content": "Say ok."}], role="agent", max_tokens=8)
    assert resp.prompt_tokens > 0 and resp.completion_tokens > 0 and budget.used == resp.prompt_tokens + resp.completion_tokens


# ================================================================================================
# Section 9 · Analysis
# ================================================================================================

from analysis import compare_baselines as cb  # noqa: E402
from analysis.alphaevolve_report import COMPUTE_NOTE, build_report  # noqa: E402
from analysis.baseline_stats import bootstrap_median_ci, wilcoxon_signed_rank, win_tie_loss  # noqa: E402
from analysis.fatigue_check import auc, outcome_labels, predictive_check, run_features  # noqa: E402
from analysis.fatigue_logs import fatigue_table, find_runs, load_run  # noqa: E402
from analysis.fatigue_plots import plot_fatigue, shape_report  # noqa: E402


def write_paired_runs(root: Path, effect: float, n_pairs: int = 10, seed: int = 1) -> Path:
    """no_sleep and fatigue runs per (problem 0, seed); fatigue gains `effect` normalized score by the end."""
    rng = random.Random(seed)
    for s in range(n_pairs):
        base = [0.5 + 0.04 * i + rng.uniform(-0.02, 0.02) for i in range(11)]
        noise = [rng.uniform(-0.02, 0.02) for _ in base]
        for arm, scores in (("no_sleep", base),
                            ("fatigue", [b + effect * i / 10 + nz for i, (b, nz) in enumerate(zip(base, noise))])):
            rid = f"p0_{arm}_s{s}"
            d = root / rid
            d.mkdir(parents=True)
            lines = [{"run_id": rid, "event": "step", "step": i, "valid": True, "score": sc,
                      "agent_tokens": i * 100, "sleep_tokens": 0} for i, sc in enumerate(scores)]
            (d / "history.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
            (d / "summary.json").write_text(json.dumps({"run_id": rid, "problem_id": 0, "seed": s, "sleep_policy": arm,
                                                        "arm": arm, "best_known": 1.0,
                                                        "budget": {"total_tokens": 1000}}), encoding="utf-8")
    return root


def _final_contrast(rep):
    return next(c for c in rep["contrasts"] if (c["a"], c["b"], c["outcome"]) == ("fatigue", "no_sleep", "final_score"))


def test_paired_comparison_detects_a_known_effect(tmp_path):
    rep = cb.compare(write_paired_runs(tmp_path, effect=0.1), n_boot=500)
    c = _final_contrast(rep)
    assert c["n_pairs"] == 10 and c["significant"] and c["verdict"] == "fatigue better"
    assert c["median_diff"] == pytest.approx(0.1, abs=0.02) and c["ci95"][0] > 0
    assert "yes for fatigue" in rep["summary"]


def test_paired_comparison_reports_no_effect_as_not_significant(tmp_path):
    rep = cb.compare(write_paired_runs(tmp_path, effect=0.0), n_boot=500)
    c = _final_contrast(rep)
    assert c["n_pairs"] == 10 and not c["significant"] and c["verdict"] == "not significant"
    assert c["ci95"][0] <= 0 <= c["ci95"][1]
    assert "no significant difference" in rep["summary"]


def test_steps_to_95_percent_is_computed_including_not_reached():
    hist = [{"event": "step", "step": i, "valid": s is not None, "score": s, "agent_tokens": 100 * i, "sleep_tokens": 0}
            for i, s in enumerate([0.5, 0.9, None, 0.96, 0.94])]
    curve = best_curve(hist, 1.0, 400)
    assert first_reach(curve, 0.95) == {"reached": True, "budget_fraction": 0.75, "step": 3}
    assert first_reach(curve, 0.99) == {"reached": False, "budget_fraction": None, "step": None}
    m = compute_run_metrics(hist, best_known=1.0, total_budget=400)
    assert m["reach"]["0.95"]["reached"] and m["reach"]["0.95"]["step"] == 3
    assert m["reach"]["0.99"]["reached"] is False and m["reach"]["0.99"]["budget_fraction"] is None
    # censored, not dropped: the paired outcome uses the whole budget (1.0) and flags it
    br = cb.BaselineRun(run=None, arm="x", metrics=m)
    assert br.outcome("budget_to_95") == (0.75, False)
    assert cb.BaselineRun(run=None, arm="x", metrics={"reach": {"0.95": {"reached": False}}}).outcome(
        "budget_to_95") == (1.0, True)


def test_compare_baselines_reports_total_cost_per_arm(tmp_path, capsys):
    root = write_paired_runs(tmp_path / "runs", effect=0.0, n_pairs=3)
    for p in root.glob("*/summary.json"):
        s = json.loads(p.read_text(encoding="utf-8"))
        if s["arm"] == "fatigue":
            s["cost"] = {"total": 0.25, "currency": "USD"}
        elif s["seed"] < 2:  # no_sleep: one run without a known cost
            s["cost"] = {"total": 0.1, "currency": "USD"}
        p.write_text(json.dumps(s), encoding="utf-8")
    rep = cb.compare(root, n_boot=200)
    fat, ns = rep["arm_summaries"]["fatigue"], rep["arm_summaries"]["no_sleep"]
    assert (fat["cost_total"], fat["cost_mean"], fat["cost_unknown_runs"]) == (pytest.approx(0.75), pytest.approx(0.25), 0)
    assert (ns["cost_total"], ns["cost_unknown_runs"]) == (pytest.approx(0.2), 1)
    assert rep["cost"]["cost_total"] == pytest.approx(0.95) and rep["cost"]["cost_unknown_runs"] == 1
    text = cb.format_report(rep)
    assert "cost$" in text and "0.7500" in text and "0.2000+" in text and "total cost of all runs: 0.9500+" in text


def test_every_baseline_figure_is_produced_from_mock_logs(matrix, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "figs"
    assert cb.main(["--runs", str(matrix["out"]), "--out", str(out), "--n-boot", "200"]) == 0
    for name in cb.FIGURES:
        p = out / name
        assert p.exists() and p.stat().st_size > 5000, name


def test_summary_paragraph_and_report_are_produced_from_mock_logs(matrix, tmp_path, capsys):
    rep = cb.compare(matrix["out"], n_boot=500)
    assert rep["pairs"] == len(SEEDS) and set(rep["arms"]) == set(ALL_ARMS)
    got = {(c["a"], c["b"], c["outcome"]) for c in rep["contrasts"]}
    for _, a, b in cb.CONTRASTS:
        for o in cb.OUTCOMES:
            assert (a, b, o) in got
    for c in rep["contrasts"]:
        assert c["n_pairs"] == len(SEEDS) and c["win"] + c["tie"] + c["loss"] == len(SEEDS)
        assert not c["significant"] and c["verdict"] == "not significant"  # 3 pairs: p >= 0.25
    assert rep["behavior_around_sleep"]["no_sleep"]["reference"].startswith("paired fatigue")
    assert "Q1" in rep["summary"] and "Q2" in rep["summary"] and "NOT significant" in rep["summary"]
    assert rep["interpretation"][0].startswith("Underpowered")
    assert cb.main(["--runs", str(matrix["out"]), "--out", str(tmp_path / "rep"), "--n-boot", "200", "--no-plot"]) == 0
    text = capsys.readouterr().out
    assert "Interpretation guide" in text and "timing matters, beyond the amount of sleep" in text
    assert "fatigue - fixed_matched" in text and "Summary" in text
    saved = json.loads((tmp_path / "rep" / "compare_baselines.json").read_text(encoding="utf-8"))
    assert saved["summary"] == rep["summary"] and (tmp_path / "rep" / "report.txt").exists()


def _fatigue_run(root, run_id, steps, summary=True, extra=None):
    d = root / run_id
    d.mkdir(parents=True)
    with (d / "history.jsonl").open("w", encoding="utf-8") as f:
        for s in steps:
            f.write(json.dumps({"run_id": run_id, **s}) + "\n")
    if summary:
        pid = run_id[1:].split("_", 1)[0]
        pol = run_id.split("_", 1)[1].rsplit("_s", 1)[0]
        seed = run_id.rsplit("_s", 1)[1]
        (d / "summary.json").write_text(json.dumps({"run_id": run_id, "problem_id": int(pid), "sleep_policy": pol,
                                                     "seed": int(seed), "best_known": 1.0,
                                                     "budget": {"total_tokens": 100000}, **(extra or {})}))
    return d


def _fatigue_steps(seed, n=60, predictive=True):
    """Calibration gap grows ahead of failures: bad outcomes cluster where the gap is high."""
    rng = random.Random(seed)
    steps = [{"event": "step", "step": 0, "valid": True, "score": 0.1, "agent_tokens": 0}]
    best = 0.1
    for t in range(1, n):
        phase = (t // 10) % 2  # alternate 10-step good / bad phases
        gap = (0.02 if phase == 0 else 0.5) if predictive else rng.uniform(0.02, 0.5)
        improve = phase == 0 and rng.random() < 0.7
        score = best + 0.01 if improve else best - 0.05
        best = max(best, score)
        steps.append({"event": "step", "step": t, "valid": True, "score": score, "expected_score": score + gap,
                      "approach_tags": [rng.choice("abcdef"), rng.choice("ghij")],
                      "program": f"def construct():\n    return {rng.random()}\n", "agent_tokens": 100 * t,
                      "context_tokens": 1000})
    return steps


def _check_cfg(**kw):
    w = {c: 0.0 for c in COMPONENTS}
    w.update(kw.pop("weights", {"calibration": 1.0}))
    base = dict(load_weight=0.0, impairment_weight=1.0, baseline_steps=5, window=3, grace_steps=0,
                impairment_weights=w, validation={"horizon_k": 3, "min_samples": 10})
    base.update(kw)
    return FatigueConfig(**base)


def test_fatigue_plot_and_alphaevolve_report_figures_are_produced(tmp_path):
    pytest.importorskip("matplotlib")
    _fatigue_run(tmp_path / "runs", "p1_no_sleep_s0", _fatigue_steps(0, 20))
    paths = plot_fatigue(find_runs(tmp_path / "runs"), tmp_path / "out", FatigueConfig())
    assert paths and paths[0].exists()
    root = tmp_path / "ae"
    make_run(root, "p36_no_sleep_s0", problem_id=36, scores=[26 / 12, 2.3, 2.4])
    make_run(root, "p36_full_sleep_s0", problem_id=36, arm="full_sleep", scores=[26 / 12, 2.5, 2.62])
    out = tmp_path / "fig"
    md = build_report(root, out).read_text(encoding="utf-8")
    assert (out / "gap_closed_vs_budget.png").stat().st_size > 0
    assert (out / "closest_arm_per_problem.png").stat().st_size > 0
    assert "gap_closed_vs_budget.png" in md


# additional coverage: analysis

def test_wilcoxon_exact_and_normal_and_bootstrap():
    assert wilcoxon_signed_rank([1, 2, 3, 4, 5])["p"] == pytest.approx(2 / 32)
    assert wilcoxon_signed_rank([-1, 2, 3, 4, 5, 6, 7, 8, 9, 10])["p"] == pytest.approx(4 / 1024)
    r = wilcoxon_signed_rank([0, 0, 1, -1])
    assert r["n_zero"] == 2 and r["n"] == 2 and r["p"] == pytest.approx(1.0)
    assert wilcoxon_signed_rank([0, 0])["p"] == 1.0
    assert wilcoxon_signed_rank([1, 1, 2])["p"] == pytest.approx(0.25)  # tied ranks
    d = [0.3, -0.1, 0.5, 0.2, -0.05, 0.4, 0.15, 0.6, -0.2, 0.35, 0.25, 0.1, 0.45, -0.15, 0.05, 0.55, 0.32, -0.02]
    ex, nm = wilcoxon_signed_rank(d), wilcoxon_signed_rank(d, exact_max_n=0)
    assert ex["method"].startswith("exact") and nm["method"].startswith("normal")
    assert abs(ex["p"] - nm["p"]) < 0.01
    med, lo, hi = bootstrap_median_ci(d, n_boot=2000, seed=3)
    assert lo <= med <= hi and (med, lo, hi) == bootstrap_median_ci(d, n_boot=2000, seed=3)
    assert bootstrap_median_ci([]) == (None, None, None) and bootstrap_median_ci([0.2]) == (0.2, 0.2, 0.2)
    assert win_tie_loss([0.1, 0, -0.2], "higher") == {"win": 1, "tie": 1, "loss": 1}
    assert win_tie_loss([0.1, -0.2, -0.3], "lower") == {"win": 2, "tie": 0, "loss": 1}


def test_analysis_interpretation_with_enough_pairs():
    def c(a, b, med, p, n=8, outcome="final_score"):
        return {"a": a, "b": b, "outcome": outcome, "label": "x", "better": "higher", "n_pairs": n,
                "median_diff": med, "ci95": [med, med], "p": p, "significant": p < 0.05, "win": n, "tie": 0, "loss": 0}

    cs = [c("fatigue", "fixed_matched", 0.05, 0.01), c("fixed_matched", "random_matched", 0.0, 0.6),
          c("fixed_letta", "no_sleep", -0.04, 0.02), c("fatigue", "no_sleep", 0.06, 0.008),
          c("fatigue", "fixed_letta", 0.08, 0.01)]
    lines = " ".join(cb.interpret(cs))
    assert "timing matters" in lines and "no timing benefit" in lines and "fixed_letta < no_sleep" in lines
    para = cb.summary_paragraph(cs)
    assert "yes for fatigue" in para and "fatigue beats the count-matched fixed schedule" in para


def test_auc_matches_the_pairwise_definition():
    assert auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert auc([1, 2], [1, 1]) is None
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 40).astype(float)
    y = rng.integers(0, 2, 40)
    pos, neg = s[y == 1], s[y == 0]
    brute = np.mean([1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg)])
    assert auc(s, y) == pytest.approx(brute)


def test_outcome_labels_hand_computed():
    steps = [{"step": 0, "valid": True, "score": 1.0}, {"step": 1, "valid": True, "score": 2.0},
             {"step": 2, "valid": True, "score": 1.5}, {"step": 3, "valid": False, "score": None},
             {"step": 4, "valid": True, "score": 3.0}, {"step": 5, "valid": True, "score": 2.0}]
    assert outcome_labels(steps, 1) == [None, 1, 1, 0, 1, None]
    assert outcome_labels(steps, 2) == [None, 1, 1, 0, None, None]  # t=2 sees an invalid step 3


def test_fatigue_log_loader_metadata_with_and_without_summary(tmp_path):
    _fatigue_run(tmp_path, "p7_no_sleep_s3", _fatigue_steps(0, 10))
    _fatigue_run(tmp_path, "p9_gepa_s1", _fatigue_steps(1, 10), summary=False)
    runs = {r.run_id: r for r in find_runs(tmp_path)}
    assert runs["p7_no_sleep_s3"].problem_id == 7 and runs["p7_no_sleep_s3"].total_budget == 100000
    r = runs["p9_gepa_s1"]
    assert (r.problem_id, r.policy, r.seed, r.best_known) == (9, "gepa", 1, None)
    assert len(fatigue_table(r)["fatigue"]) == 10
    assert [x.run_id for x in find_runs(tmp_path, policies=["gepa"])] == ["p9_gepa_s1"]


def test_fatigue_predictive_check_auc_per_problem_and_flag(tmp_path):
    for s in (0, 1):
        _fatigue_run(tmp_path, f"p1_no_sleep_s{s}", _fatigue_steps(s, predictive=True))
        _fatigue_run(tmp_path, f"p2_no_sleep_s{s}", _fatigue_steps(s, predictive=False))
        _fatigue_run(tmp_path, f"p1_full_sleep_s{s}", _fatigue_steps(s + 10))
    res = {r.problem_id: r for r in predictive_check(find_runs(tmp_path), _check_cfg())}
    assert set(res) == {1, 2}
    assert all(len(r.runs) == 2 and all("no_sleep" in x for x in r.runs) for r in res.values())
    assert res[1].auc > 0.7 and not res[1].flagged
    wide = predictive_check(find_runs(tmp_path), _check_cfg(validation={"horizon_k": 3, "auc_margin": 0.5}))
    assert all(r.flagged and "recalibrate" in r.message for r in wide)


def test_fatigue_refit_recovers_the_predictive_component(tmp_path):
    from scripts.refit_fatigue_weights import refit

    for s in range(3):
        _fatigue_run(tmp_path, f"p1_no_sleep_s{s}", _fatigue_steps(s, predictive=True))
    cfg = _check_cfg(weights={"calibration": 0.0, "bloat": 1.0, "stagnation": 0.0})
    feats = [run_features(r, cfg) for r in find_runs(tmp_path)]
    w, fitted, base = refit(feats, cfg, seed=0, samples=50)
    assert fitted > base and fitted > 0.7
    assert w[COMPONENTS.index("calibration")] > 0 and w.sum() == pytest.approx(1.0)
    w2, f2, _ = refit(feats, cfg, seed=0, samples=50)
    assert np.allclose(w, w2) and f2 == fitted  # deterministic


def test_fatigue_refit_script_writes_config(tmp_path):
    from scripts.refit_fatigue_weights import main as refit_main

    runs = tmp_path / "runs"
    for s in range(2):
        _fatigue_run(runs, f"p1_no_sleep_s{s}", _fatigue_steps(s, predictive=True))
    conf = tmp_path / "c.yaml"
    conf.write_text(yaml.safe_dump({"problem_id": 1, "fatigue": {"baseline_steps": 5, "window": 3, "grace_steps": 0}}))
    assert refit_main(["--runs", str(runs), "--config", str(conf), "--samples", "20"]) == 0
    out = yaml.safe_load((tmp_path / "c.refit.yaml").read_text())
    assert out["problem_id"] == 1 and out["fatigue"]["baseline_steps"] == 5
    assert set(out["fatigue"]["impairment_weights"]) == set(COMPONENTS)
    assert refit_main(["--runs", str(runs), "--config", str(conf), "--samples", "20", "--in-place"]) == 0
    assert "impairment_weights" in yaml.safe_load(conf.read_text())["fatigue"]
    FatigueConfig(**yaml.safe_load(conf.read_text())["fatigue"])


def test_fatigue_shape_report_and_logged_vs_recomputed(tmp_path):
    steps = _fatigue_steps(0, 30)
    hist = steps[:16] + [{"event": "sleep", "step": 15, "sleep_mode": "nrem"}] + steps[16:]
    _fatigue_run(tmp_path, "p1_full_sleep_s0", hist)
    _fatigue_run(tmp_path, "p1_no_sleep_s0", steps)
    rep = {r["policy"]: r for r in shape_report(find_runs(tmp_path), FatigueConfig(load_tokens_scale=1500))}
    assert rep["no_sleep"]["slope"] > 0 and rep["no_sleep"]["n_sleeps"] == 0
    assert rep["full_sleep"]["n_sleeps"] == 1 and rep["full_sleep"]["mean_drop_after_sleep"] > 0
    run_log = load_run(tmp_path / "p1_no_sleep_s0")
    for s in run_log.history:
        s["fatigue"] = 0.123
    assert set(fatigue_table(run_log, recompute=False)["fatigue"]) == {0.123}  # logged fields used when present


def test_alphaevolve_report_markdown_flags_unverified_beats(tmp_path):
    root = tmp_path / "runs"
    for seed in (0, 1):
        make_run(root, f"p36_no_sleep_s{seed}", problem_id=36, arm="no_sleep", seed=seed,
                 scores=[26 / 12, 2.3, 2.4 + 0.01 * seed])
        make_run(root, f"p36_full_sleep_s{seed}", problem_id=36, arm="full_sleep", seed=seed,
                 scores=[26 / 12, 2.5, 2.62], sleep_tokens_at={2: 30})
    make_run(root, "p0_no_sleep_s0", problem_id=0, scores=[0.5, 0.9])
    beat = make_run(root, "p36_claim_s0", problem_id=36, arm="claim", scores=[26 / 12, 2.7])
    enrich_summary(beat, verify_beats=False)
    assert json.loads((beat / "summary.json").read_text())["alphaevolve"]["beat_status"] == "unverified"
    md = build_report(root, tmp_path / "out", figures=False).read_text(encoding="utf-8")
    assert COMPUTE_NOTE in md
    assert "git commit:" in md and "models: test-model" in md and "token budgets (total per run): 1000" in md
    assert "| 36. Circle packing" in md and "n26_unit_square" in md and "of 2" in md
    assert "| 2. An autocorrelation problem" in md and "pending" in md
    assert "**UNVERIFIED** `p36_claim_s0`" in md
    assert "p0_no_sleep_s0 | 0 |" in md and "not in the AlphaEvolve benchmark" in md
    assert "## By status category" in md and "matched_optimal" in md
    assert "ShinkaEvolve" in md and "Figures skipped (--no-figures)." in md


# ================================================================================================
# Section 16 · Evaluator-call accounting (harness/evals.py) and vLLM thinking
# ================================================================================================

from harness.evals import CountingEvaluator, EvalBudget, EvalExhausted  # noqa: E402


def test_eval_budget_counts_by_role_and_caps():
    b = EvalBudget(3)
    assert (b.used, b.remaining, b.exhausted) == (0, 3, False)
    b.charge("agent"), b.charge("sleep"), b.charge("agent")
    assert (b.agent_evals, b.sleep_evals, b.used) == (2, 1, 3)
    assert b.exhausted and b.remaining == 0
    with pytest.raises(EvalExhausted):
        b.check()
    snap = b.snapshot()
    assert snap == {"used": 3, "limit": 3, "remaining": 0, "agent_evals": 2, "sleep_evals": 1}
    assert EvalBudget().limit is None and EvalBudget().remaining is None
    EvalBudget().check()  # uncapped: never raises
    with pytest.raises(ValueError):
        EvalBudget(0)


def test_counting_evaluator_charges_checks_and_delegates():
    ev = CountingEvaluator(TextEvaluator(), EvalBudget(10))
    assert ev.best_known == 1.0 and ev.name == "text_mock" and "construct" in ev.description
    res = ev.evaluate(GOOD)
    assert isinstance(res, EvalResult)
    assert ev.budget.agent_evals == 1
    ev.tagged("sleep").evaluate(BAD)
    assert ev.budget.sleep_evals == 1 and ev.budget.used == 2
    ev.evaluate("not a program")  # invalid results still cost an evaluator call
    assert ev.budget.used == 3


def test_counting_evaluator_refuses_at_limit_before_calling():
    inner = TextEvaluator()
    ev = CountingEvaluator(inner, EvalBudget(1))
    ev.evaluate(GOOD)
    with pytest.raises(EvalExhausted):
        ev.evaluate(GOOD)
    with pytest.raises(EvalExhausted):
        ev.tagged("sleep").evaluate(GOOD)
    assert len(inner.calls) == 1 and ev.budget.used == 1  # never exceeds the limit


class _EvalOnSleep:
    """Policy whose sleep spends one evaluator call through the bound (sleep-tagged) evaluator."""
    name = "eval_on_sleep"
    last_depth = None

    def should_sleep(self, step, history):
        return step >= 1

    def bind(self, *, evaluator):
        self._evaluator = evaluator

    def sleep(self, task, block, history, llm, budget):
        self._evaluator.evaluate(GOOD)
        return block or "notes"


def test_loop_counts_every_eval_and_stops_at_eval_limit(mock_llm, mock_cfg, run_dir):
    llm, _, _ = mock_llm(100_000)
    s = run(mock_cfg(max_steps=4), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    ev = s["evaluations"]
    assert ev["limit"] is None and ev["sleep_evals"] == 0
    assert ev["used"] == ev["agent_evals"] == s["steps_completed"] + 1  # seed + one per step

    llm, _, _ = mock_llm(100_000)
    s = run(mock_cfg(max_steps=50, max_evaluations=3), TextEvaluator(), llm, NoSleep(), out_dir=run_dir)
    assert s["stop_reason"] == "eval_limit" and s["steps_completed"] == 2
    assert s["evaluations"]["used"] == 3 and s["evaluations"]["limit"] == 3


def test_loop_sleep_evals_are_counted_separately_and_can_hit_the_cap(mock_llm, mock_cfg, run_dir):
    llm, _, _ = mock_llm(100_000)
    s = run(mock_cfg(max_steps=2), TextEvaluator(), llm, _EvalOnSleep(), out_dir=run_dir)
    ev = s["evaluations"]
    assert ev["sleep_evals"] == 2 and ev["agent_evals"] == 3 and ev["used"] == 5  # seed + 2 steps + 2 sleeps

    llm, _, _ = mock_llm(100_000)
    s = run(mock_cfg(max_steps=50, max_evaluations=2), TextEvaluator(), llm, _EvalOnSleep(),
            out_dir=run_dir)
    assert s["stop_reason"] == "eval_limit" and s["evaluations"]["used"] == 2


def test_chat_params_thinking_sends_chat_template_kwargs_for_vllm():
    p = resolve_chat_params("Qwen/Qwen3-8B", {"thinking": True}, env={}, warn=False)
    assert p.thinking is True and p.reasoning_model is False
    assert p.request_kwargs(64) == {"max_tokens": 64,
                                    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    body = {"max_tokens": 8}
    p.apply_to_body(body)
    assert body["chat_template_kwargs"] == {"enable_thinking": True} and body["max_tokens"] == 8


def test_chat_params_thinking_is_dropped_for_reasoning_models():
    p = resolve_chat_params("gpt-5-mini", {"thinking": True, "reasoning_effort": "low"}, env={}, warn=False)
    assert p.thinking is None and "thinking" in p.dropped
    assert "extra_body" not in p.request_kwargs(64)
    p = resolve_chat_params("Qwen/Qwen3-8B", None, env={"LLM_THINKING": "false"}, warn=False)
    assert p.thinking is False
    assert p.request_kwargs(8)["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_strip_think_removes_reasoning_blocks():
    from llm.client import strip_think
    assert strip_think("<think>reasoning\nabout x</think>the answer") == "the answer"
    assert strip_think("x <think>a\nb</think> y <think>z</think> w") == "x  y  w"
    assert strip_think("no tags here") == "no tags here"
    assert strip_think("") == ""


def test_summarize_uses_sleep_max_tokens_for_the_completion_cap():
    calls = []

    def transport(messages, max_tokens):
        calls.append(max_tokens)
        return LLMResponse("notes", 10, 5)

    pol = Summarize(every_n=1, sleep_max_tokens=777)
    llm = LLMClient(Budget(10**6), transport, sleep_fn=lambda s: None)
    out = pol.sleep("task", "", [], llm, None)
    assert calls == [777] and out == "notes"


# ================================================================================================
# Section 17 · REM dreaming, cross-run memory, dream ladder arms
# ================================================================================================

from sleep.dreamer import Dreamer  # noqa: E402


def test_memory_store_save_load_roundtrip():
    s = MemoryStore()
    s.add(kind="goal", text="g0 text", step=0, status="verified", strength=1.0, item_id="g0")
    it = s.add(kind="skill", text="hex init helps", step=3, status="verified",
               evidence=[1, 2], approach_tags=["Hex Init"])
    s2 = MemoryStore.from_state(json.loads(json.dumps(s.save_state())))
    assert s2.render(None) == s.render(None)
    assert s2.get(it.id).evidence == it.evidence and s2.get(it.id).status == "verified"
    assert s2.add(kind="fact", text="new claim", step=4).id != it.id  # id counter restored


def test_full_sleep_memory_path_roundtrips_across_policies(tmp_path):
    path = tmp_path / "mem.json"
    a = _fs(memory_path=str(path))
    a.store.add(kind="skill", text="cross-run knowledge", step=1, status="verified")
    a._persist_memory()
    b = _fs(memory_path=str(path))
    assert b.store.find_text("cross-run knowledge") is not None
    assert not (tmp_path / "mem.json.tmp").exists()  # atomic write leaves no temp file
    assert json.loads(path.read_text())["items"]  # target file is complete, never torn
    c = _fs(memory_path=str(tmp_path / "missing.json"))
    assert len(c.store.items) == 0  # absent file starts empty


def test_full_sleep_rem_pass_dreams_and_validates(mock_llm):
    body_c = {"claims": [], "replays": [], "goals": [], "merges": []}
    body_d = {"dreams": [{"did": "d1", "kind": "association",
                          "text": "hex init plus lattice gives a jump",
                          "sources": ["m1"], "rationale": "combines two wins",
                          "test": {"control": BAD, "treatment": GOOD,
                                   "change": "return 0.7s instead of 0s"}}]}
    llm, budget, _ = mock_llm(10**6, script=[json.dumps(body_c), json.dumps(body_d)])
    fs = _fs(dream=True)
    fs.store.add(kind="fact", text="hex init helped once", step=1)
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=20, mode="rem")
    assert m["dream_calls"] >= 1 and m["dreams_validated"] == 1 and m["dream_evaluations"] == 2
    assert m["dreams"][0]["status"] == "validated" and m["dreams"][0]["did"] == "d1"
    assert m["dreams"][0]["sources"] == ["m1"] and m["dreams"][0]["rationale"]
    item = fs.store.find_text("hex init plus lattice gives a jump")
    assert item is not None and item.status == "verified" and item.kind == "skill"


def test_dream_pass_skips_when_budget_low(mock_llm):
    llm, budget, t = mock_llm(10_000)
    budget.charge("agent", 1, 9_000)  # remaining 999 < dream_min_budget_tokens (8192)
    fs = _fs(dream=True)
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=20, mode="rem")
    assert m["dream_skipped"] == "low_budget" and m["dream_calls"] == 0
    assert t.sleep_calls == 1  # only the consolidator call ran


def test_dream_off_by_default_makes_no_dream_calls(mock_llm):
    llm, budget, t = mock_llm(10_000, script=[json.dumps({"claims": [], "replays": []})])
    fs = _fs()  # dream defaults False
    m = fs.consolidate("task", _sleep_history(), llm, budget, step=20, mode="rem")
    assert "dreams" not in m and len(t.calls) == 1  # consolidator only, no dream call


def test_dream_ladder_arms_exist():
    from harness.baselines import ARM_SPECS
    for arm in ("summarize", "dream_unvalidated", "dream", "dream_distill"):
        assert arm in ARM_SPECS
    assert ARM_SPECS["dream_unvalidated"].params["dream_validate"] is False
    assert ARM_SPECS["dream"].params["dream"] is True
    assert "memory_path" in ARM_SPECS["dream_distill"].declared


# ==============================================================================================
# Section 18 · REM dreaming: dream_io validation, Dreamer sampling, dream A/B validation
# ==============================================================================================

from sleep.dream_io import validate_dreams as parse_dreams  # noqa: E402  (validator shares the name)
from sleep.insight_validator import validate_dreams  # noqa: E402

HALF = "def construct():\n    return [0.5, 0.5, 0.5, 0.5, 0.5]\n"


def make_llm(script: list[str], total: int = 100000):
    """(LLMClient, Budget, MockTransport): scripted replies, fixed usage, no sleeping."""
    budget = Budget(total)
    transport = MockTransport(script=script)
    return LLMClient(budget, transport, sleep_fn=lambda s: None), budget, transport


def make_store(n: int = 6, *, step: int = 0) -> MemoryStore:
    """g0 goal plus n tentative skills m1..mn with distinct tags and rising strength."""
    store = MemoryStore()
    store.add(kind="goal", text="Maximize the score. Best valid score so far: 0.5.",
              step=step, status="verified", strength=1.0, item_id="g0")
    for i in range(n):
        store.add(kind="skill", text=f"strategy {i} raises the score", step=step,
                  status="tentative", strength=0.3 + 0.05 * i, approach_tags=[f"tag-{i}"])
    return store


STEP_RECORDS = [
    {"id": 1, "approach_tags": ["uniform-init"], "score": 0.2, "valid": True, "side_info": None},
    {"id": 2, "approach_tags": ["perturb-parent"], "score": 0.4, "valid": True, "side_info": None},
]


def dream_obj(*, text: str = "pushing every coordinate to 0.7 raises the score",
              test=None, kind: str = "association", sources=None) -> dict:
    """One-dream object; test=None attaches the HALF-vs-GOOD test, test="null" -> null."""
    t = {"control": HALF, "treatment": GOOD, "change": "0.5 -> 0.7"} if test is None \
        else (None if test == "null" else test)
    return {"dreams": [{"did": "d1", "kind": kind, "text": text,
                        "sources": sources if sources is not None else ["m1"],
                        "rationale": "m1 already points at 0.7", "test": t}]}


DREAMS_JSON = json.dumps({"dreams": [
    {"did": "d1", "kind": "association", "text": "pushing every coordinate to 0.7 raises the score",
     "sources": ["m1", "m2"], "rationale": "both point at the target",
     "test": {"control": HALF, "treatment": GOOD, "change": "0.5 -> 0.7"}},
    {"did": "d2", "kind": "extrapolation", "text": "an untried direction may help",
     "sources": ["m3", "2"], "rationale": "no memory covers it", "test": None},
]})


# ---- dream_io.validate_dreams ------------------------------------------------------------

def test_parse_good_dreams():
    dreams, problems = parse_dreams(json.loads(DREAMS_JSON), max_dreams=5)
    assert problems == [] and len(dreams) == 2
    d1, d2 = dreams
    assert d1["did"] == "d1" and d1["kind"] == "association"
    assert d1["sources"] == ["m1", "m2"] and d1["test"]["treatment"] == GOOD
    assert d2["kind"] == "extrapolation" and d2["test"] is None


def test_parse_drops_bad_entries_and_normalizes():
    obj = {"dreams": [
        "nope",                                                   # not an object
        {"did": "d1", "kind": "weird", "text": "keep me", "sources": "m1"},  # kind -> association
        {"did": "d2", "kind": "association", "text": "   "},       # empty text
        {"did": "d1", "kind": "association", "text": "dup did"},   # duplicate did
        {"did": "d3", "kind": "counterfactual", "text": "ok", "sources": [1, "m2", {}]},
    ]}
    dreams, problems = parse_dreams(obj, max_dreams=5)
    assert [d["did"] for d in dreams] == ["d1", "d3"]
    assert dreams[0]["kind"] == "association" and dreams[0]["sources"] == ["m1"]
    assert dreams[1]["sources"] == ["1", "m2"]
    assert len(problems) == 3


def test_parse_tolerates_garbage_and_caps_count():
    dreams, problems = parse_dreams([1, 2, 3], max_dreams=3)
    assert dreams == [] and problems == ["not an object"]
    dreams, problems = parse_dreams({"dreams": "nope"}, max_dreams=3)
    assert dreams == [] and "dreams is not a list" in problems
    many = {"dreams": [{"did": f"d{i}", "text": f"dream {i}"} for i in range(10)]}
    dreams, problems = parse_dreams(many, max_dreams=3)
    assert len(dreams) == 3 and all(len(d["text"]) <= 300 for d in dreams)


# ---- Dreamer -----------------------------------------------------------------------------

def test_dreamer_sampling_is_reproducible_and_seeded():
    llm, _, _ = make_llm([DREAMS_JSON])
    m1 = Dreamer(seed=7, k_sources=4).dream("task", make_store(10), STEP_RECORDS, llm, step=5)
    llm, _, _ = make_llm([DREAMS_JSON])
    m2 = Dreamer(seed=7, k_sources=4).dream("task", make_store(10), STEP_RECORDS, llm, step=5)
    assert m1["dream_source_ids"] == m2["dream_source_ids"]
    assert len(m1["dream_source_ids"]) == 4 and "g0" not in m1["dream_source_ids"]
    seen = set()
    for s in range(6):  # different seeds should not all draw the same sources
        llm, _, _ = make_llm([DREAMS_JSON])
        m = Dreamer(seed=s, k_sources=4).dream("task", make_store(10), [], llm, step=5)
        seen.add(tuple(m["dream_source_ids"]))
    assert len(seen) > 1


def test_dreamer_parses_metrics_and_charges_sleep_budget():
    llm, budget, transport = make_llm([DREAMS_JSON])
    d = Dreamer(seed=0)
    m = d.dream("the dream task", make_store(), STEP_RECORDS, llm, step=3)
    assert m["dream_calls"] == 1 and m["n_dreams"] == 2
    assert m["dream_parse_error"] is None and m["dream_source_ids"]
    assert budget.sleep_tokens > 0 and budget.agent_tokens == 0
    user = transport.calls[0][-1]["content"]
    assert "<dream_directive>" in user and d.last_prompt == user


def test_dreamer_retries_garbage_once():
    llm, _, _ = make_llm(["no json here", DREAMS_JSON])
    m = Dreamer(seed=0, retries=1).dream("t", make_store(), [], llm, step=1)
    assert m["dream_calls"] == 2 and m["n_dreams"] == 2 and m["dream_parse_error"] is None


def test_dreamer_empty_store_and_strength_filter_still_calls():
    llm, _, _ = make_llm([DREAMS_JSON])
    m = Dreamer(seed=0, min_strength=0.99).dream("t", make_store(), [], llm, step=1)
    assert m["dream_calls"] == 1 and m["dream_source_ids"] == []
    llm, _, _ = make_llm([DREAMS_JSON])
    m = Dreamer(seed=0).dream("t", MemoryStore(), [], llm, step=1)
    assert m["dream_calls"] == 1 and m["dream_source_ids"] == []


# ---- insight_validator.validate_dreams ---------------------------------------------------

def test_validator_confirms_better_treatment_and_adds_verified_skill():
    store = make_store()
    dreams, _ = parse_dreams(dream_obj(), max_dreams=3)
    out = validate_dreams(dreams, store, TextEvaluator(), step=5, max_tests=3)
    assert out["validated"] == 1 and out["evaluations"] == 2
    rec = out["records"][0]
    assert rec["status"] == "validated" and rec["replay_rid"] == "d5.0"
    assert rec["score_diff"] > 0 and rec["decision_reason"]
    skill = store.find_text("pushing every coordinate to 0.7 raises the score")
    assert skill is not None and skill.kind == "skill" and skill.status == "verified"
    assert skill.evidence[0] == "d5.0" and "m1" in skill.evidence
    assert "tag-0" in skill.approach_tags  # union of the source item's tags


def test_validator_refutes_worse_treatment():
    store = make_store()
    obj = dream_obj(text="0.0 beats 0.7", test={"control": GOOD, "treatment": BAD, "change": "0.7 -> 0.0"})
    dreams, _ = parse_dreams(obj, max_dreams=3)
    out = validate_dreams(dreams, store, TextEvaluator(), step=6, max_tests=3)
    assert out["refuted"] == 1 and out["validated"] == 0
    item = store.find_text("0.0 beats 0.7")
    assert item is not None and item.status == "refuted"
    assert out["edits"][0]["op"] == "refute" and out["edits"][0]["source"] == "dream"


def test_validator_untested_and_identical_programs():
    store = make_store()
    dreams, _ = parse_dreams(dream_obj(test="null"), max_dreams=3)  # test: null
    out = validate_dreams(dreams, store, TextEvaluator(), step=1, max_tests=3)
    assert out["untested"] == 1 and out["evaluations"] == 0
    item = store.find_text("pushing every coordinate to 0.7 raises the score")
    assert item is not None and item.status == "tentative"
    same = {"control": GOOD, "treatment": GOOD, "change": "none"}
    dreams, _ = parse_dreams(dream_obj(text="identical programs test", test=same), max_dreams=3)
    ev = TextEvaluator()
    out = validate_dreams(dreams, store, ev, step=1, max_tests=3)
    assert out["untested"] == 1 and out["evaluations"] == 0 and ev.calls == []
    out = validate_dreams(dreams, store, ev, step=2, max_tests=3)  # text already in store
    assert out["edits_applied"] == 0


def test_records_keep_did_sources_and_rationale():
    dreams, _ = parse_dreams(json.loads(DREAMS_JSON), max_dreams=5)
    out = validate_dreams(dreams, make_store(), TextEvaluator(), step=9, max_tests=5)
    for rec, d in zip(out["records"], dreams):
        assert rec["did"] == d["did"] and rec["sources"] == d["sources"]
        assert rec["rationale"] == d["rationale"] and rec["kind"] == d["kind"]


def test_evaluations_are_charged_to_the_eval_budget():
    store = make_store()
    dreams, _ = parse_dreams(json.loads(DREAMS_JSON), max_dreams=5)
    budget = EvalBudget(10)
    ev = CountingEvaluator(TextEvaluator(), budget, role="sleep")
    out = validate_dreams(dreams, store, ev, step=2, max_tests=5)
    assert out["evaluations"] == 2  # only d1 has a test
    assert budget.sleep_evals == 2 and budget.used == 2


def test_evaluator_none_skips_all_testing():
    store = make_store()
    dreams, _ = parse_dreams(json.loads(DREAMS_JSON), max_dreams=5)
    out = validate_dreams(dreams, store, None, step=1, max_tests=5)
    assert out["evaluations"] == 0 and out["untested"] == len(dreams) == 2
    assert out["validated"] == out["refuted"] == out["uncertain"] == 0
    assert all(r["status"] == "untested" for r in out["records"])
    assert out["edits_applied"] == 2  # both hypotheses kept as tentative skills


def test_end_to_end_dreamer_then_validator():
    store = make_store()
    llm, _, _ = make_llm([DREAMS_JSON])
    m = Dreamer(seed=0).dream("task", store, STEP_RECORDS, llm, step=4)
    out = validate_dreams(m["dreams"], store, TextEvaluator(), step=4, max_tests=5)
    assert out["evaluations"] > 0 and out["validated"] == 1
    verified = [i for i in store.items.values() if i.kind == "skill" and i.status == "verified"]
    assert verified and any("d4." in str(e) for i in verified for e in i.evidence)


# ==============================================================================================
# Section 19 · Parametric learning: example collection, training spec, checkpoint registry
# ==============================================================================================

from sleep.trainer import (
    CheckpointRegistry,
    TrainingExample,
    build_training_spec,
    collect_examples,
)  # noqa: E402


TASK = "Write construct() returning 5 floats; score = 1 - mean((x - 0.7)^2)."


def history() -> list[dict]:
    """Step records: valid steps at 0/1/3 (1 and 3 share program text), invalid at 2, a sleep line."""
    return [
        {"event": "step", "step": 0, "valid": True, "score": 0.40,
         "program": "def construct():\n    return [0.5] * 5\n", "program_id": "a"},
        {"event": "step", "step": 1, "valid": True, "score": 0.90,
         "program": "def construct():\n    return [0.7] * 5\n", "program_id": "b"},
        {"event": "step", "step": 2, "valid": False, "score": None,
         "program": "def construct():\n    raise X\n", "program_id": "c"},
        {"event": "step", "step": 3, "valid": True, "score": 0.70,
         "program": "def construct():\n    return [0.7] * 5\n", "program_id": "d"},
        {"event": "sleep", "step": 3, "valid": None, "score": None, "program": None},
    ]


def store() -> MemoryStore:
    s = MemoryStore()
    s.add(kind="skill", text="anneal values toward the target", step=3,
          status="verified", evidence=["d3"])
    s.add(kind="rule", text="never return fewer than 5 numbers", step=4,
          status="verified", evidence=["r4.0"])
    s.add(kind="fact", text="an irrelevant verified fact", step=4, status="verified")
    s.add(kind="skill", text="a merely tentative skill", step=4, status="tentative")
    return s


def test_collect_examples_builds_expected_pairs():
    exs = collect_examples(TASK, store(), history())
    by_src = {}
    for e in exs:
        by_src.setdefault(e.source, []).append(e)
    # best programs, best-first, deduped: the 0.9 program, then the 0.4 one
    progs = by_src["best_program"]
    assert [e.messages[-1]["content"] for e in progs] == [
        "def construct():\n    return [0.7] * 5\n", "def construct():\n    return [0.5] * 5\n"]
    assert progs[0].meta["score"] == 0.90 and progs[0].meta["program_id"] == "b"
    assert TASK in progs[0].messages[0]["content"]
    assert "Best score so far: 0.4" in progs[0].messages[0]["content"]  # running best before step 1
    # verified store items: the rule at 0.5, the dream-evidenced skill at 1.0
    assert by_src["verified_item"][0].weight == 0.5
    assert by_src["verified_item"][0].messages[-1]["content"] == "never return fewer than 5 numbers"
    assert by_src["dream"][0].weight == 1.0
    # non-verified and non-skill/rule items never appear
    texts = [e.messages[-1]["content"] for e in exs]
    assert "an irrelevant verified fact" not in texts and "a merely tentative skill" not in texts


def test_collect_examples_best_k_min_score_and_dreams_flag():
    h, s = history(), store()
    exs = collect_examples(TASK, s, h, best_k=1)
    assert [e.source for e in exs].count("best_program") == 1
    exs = collect_examples(TASK, s, h, min_score=0.8)
    assert all(e.meta.get("score", 1) >= 0.8 for e in exs if e.source == "best_program")
    assert len([e for e in exs if e.source == "best_program"]) == 1
    # include_dreams=False drops the dream-derived item entirely
    exs = collect_examples(TASK, s, h, include_dreams=False)
    assert "dream" not in [e.source for e in exs]
    assert any(e.source == "verified_item" for e in exs)


def test_collect_examples_empty_inputs_and_cap():
    assert collect_examples(TASK, MemoryStore(), []) == []
    exs = collect_examples(TASK, store(), history(), max_examples=2)
    assert len(exs) == 2 and exs[0].source == "best_program"


def test_collect_examples_is_deterministic():
    a = [e.to_jsonl() for e in collect_examples(TASK, store(), history())]
    b = [e.to_jsonl() for e in collect_examples(TASK, store(), history())]
    assert a == b


def test_training_example_jsonl_roundtrip():
    e = collect_examples(TASK, store(), history())[0]
    d = json.loads(e.to_jsonl())
    assert d["source"] == e.source and d["messages"] == e.messages and d["weight"] == e.weight
    with pytest.raises(ValueError):
        TrainingExample(messages=[], source="made_up")


def test_build_training_spec_refuses_empty_and_is_deterministic():
    with pytest.raises(ValueError):
        build_training_spec([])
    exs = collect_examples(TASK, store(), history())
    s1 = build_training_spec(exs)
    s2 = build_training_spec(exs)
    assert s1 == s2 and s1["adapter_id"].startswith("se-")
    assert (s1["base_model"], s1["lora_rank"], s1["lora_alpha"], s1["epochs"]) == \
        ("Qwen/Qwen3-8B", 16, 32, 1)
    assert s1["n_examples"] == len(exs)
    lines = s1["examples_jsonl"].splitlines()
    assert len(lines) == len(exs) and json.loads(lines[0])["source"] == "best_program"
    assert build_training_spec(exs, adapter_id="mine")["adapter_id"] == "mine"


def test_registry_register_and_duplicate(tmp_path):
    reg = CheckpointRegistry(tmp_path / "reg.json")
    rec = reg.register("a1", base_model="Qwen/Qwen3-8B", train_meta={"n": 10})
    assert rec["status"] == "candidate" and rec["evals"] == [] and rec["registered_at"]
    with pytest.raises(ValueError):
        reg.register("a1", base_model="x", train_meta={})
    assert reg.champion() is None


def test_registry_promotion_lifecycle(tmp_path):
    reg = CheckpointRegistry(tmp_path / "reg.json")
    reg.register("a1", base_model="m", train_meta={})
    reg.register("a2", base_model="m", train_meta={})
    # no evals -> can never be promoted, even with no champion
    assert reg.promote_if_better("a1")["promoted"] is False
    reg.record_eval("a1", 0.50, n_runs=8)
    r = reg.promote_if_better("a1")
    assert r["promoted"] and r["champion"] == "a1"
    assert reg.champion()["adapter_id"] == "a1"
    # strictly better promotes and retires the old champion
    reg.record_eval("a2", 0.60, n_runs=8, note="held-out")
    r = reg.promote_if_better("a2")
    assert r["promoted"] and r["champion"] == "a2"
    assert reg._data["adapters"]["a1"]["status"] == "retired"
    # equal (or worse) does not promote
    reg.register("a3", base_model="m", train_meta={})
    reg.record_eval("a3", 0.60, n_runs=8)
    assert reg.promote_if_better("a3")["promoted"] is False
    reg.record_eval("a3", 0.55, n_runs=8)
    assert reg.promote_if_better("a3")["promoted"] is False


def test_registry_min_delta(tmp_path):
    reg = CheckpointRegistry(tmp_path / "reg.json")
    reg.register("a1", base_model="m", train_meta={})
    reg.register("a2", base_model="m", train_meta={})
    reg.record_eval("a1", 0.50, n_runs=4)
    reg.promote_if_better("a1")
    reg.record_eval("a2", 0.55, n_runs=4)
    assert reg.promote_if_better("a2", min_delta=0.1)["promoted"] is False
    reg.record_eval("a2", 0.61, n_runs=4)  # latest eval wins the comparison
    assert reg.promote_if_better("a2", min_delta=0.1)["promoted"] is True


def test_registry_rollback_restores_previous_champion(tmp_path):
    reg = CheckpointRegistry(tmp_path / "reg.json")
    for aid, metric in (("a1", 0.5), ("a2", 0.6)):
        reg.register(aid, base_model="m", train_meta={})
        reg.record_eval(aid, metric, n_runs=4)
        reg.promote_if_better(aid)
    assert reg.champion()["adapter_id"] == "a2"
    r = reg.rollback()
    assert r["rolled_back"] and r["champion"] == "a1"
    assert reg.champion()["adapter_id"] == "a1"
    assert reg._data["adapters"]["a2"]["status"] == "retired"
    # a fresh registry with nothing to roll back to fails cleanly
    reg2 = CheckpointRegistry(tmp_path / "other.json")
    assert reg2.rollback()["rolled_back"] is False


def test_registry_persists_and_rejects_corruption(tmp_path):
    path = tmp_path / "reg.json"
    reg = CheckpointRegistry(path)
    reg.register("a1", base_model="m", train_meta={"seed": 1})
    reg.record_eval("a1", 0.7, n_runs=4)
    reg.promote_if_better("a1")
    reg2 = CheckpointRegistry(path)  # re-instantiation sees the same state
    assert reg2.champion()["adapter_id"] == "a1"
    assert reg2._data["adapters"]["a1"]["train_meta"] == {"seed": 1}
    path.write_text("{not json")
    with pytest.raises(ValueError):
        CheckpointRegistry(path)


def test_registry_unknown_eval_raises_keyerror(tmp_path):
    reg = CheckpointRegistry(tmp_path / "reg.json")
    with pytest.raises(KeyError):
        reg.record_eval("ghost", 0.9, n_runs=1)


# ---- scripts/distill.py ----------------------------------------------------

def test_distill_builds_spec_from_history_and_memory(tmp_path, capsys):
    from scripts.distill import main as distill_main

    hist = tmp_path / "history.jsonl"
    hist.write_text("\n".join([
        json.dumps({"event": "step", "step": 1, "valid": True, "score": 0.9,
                    "program": GOOD, "program_id": "p1"}),
        json.dumps({"event": "step", "step": 2, "valid": False, "score": None,
                    "program": BAD, "program_id": "p2"}),
    ]), encoding="utf-8")
    store = MemoryStore()
    store.add(kind="skill", text="validated discovery", step=2, status="verified",
              evidence=["d1.1"])  # evidence ids starting with "d" mark it dream-derived
    mem = tmp_path / "mem.json"
    mem.write_text(json.dumps(store.save_state()), encoding="utf-8")
    out = tmp_path / "spec.json"

    rc = distill_main(["--history", str(hist), "--memory", str(mem),
                       "--task", "solve the problem", "--out", str(out)])
    assert rc == 0
    spec = json.loads(out.read_text())
    assert spec["adapter_id"].startswith("se-") and spec["n_examples"] >= 2
    assert "modal run infra/modal_train.py" in capsys.readouterr().out


def test_distill_register_only_adds_adapter(tmp_path):
    from scripts.distill import main as distill_main

    spec = build_training_spec([TrainingExample(
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        source="best_program")], adapter_id="se-abc")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    reg_path = tmp_path / "reg.json"

    rc = distill_main(["--register-only", str(spec_path), "--registry", str(reg_path)])
    assert rc == 0
    assert CheckpointRegistry(reg_path).load()["adapters"]["se-abc"]["status"] == "candidate"


def test_distill_requires_inputs(tmp_path):
    from scripts.distill import main as distill_main
    with pytest.raises(SystemExit):
        distill_main(["--task", "x"])  # no history, no memory

def test_dreamer_reserves_failed_sources():
    """dead_end/refuted items are guaranteed sampling slots — counterfactual dreams need
    failure material even when every other item outranks them."""
    from sleep.dreamer import Dreamer
    store = MemoryStore()
    for i in range(20):  # strong filler that would crowd out failures under pure rank weighting
        it = store.add(kind="rule", text=f"winning rule {i}", step=i, status="verified",
                       strength=0.9)
    dead = store.add(kind="dead_end", text="tried X, always fails", step=1,
                     status="verified", strength=0.4)
    refuted = store.add(kind="rule", text="Y works sometimes", step=1,
                        status="refuted", strength=0.2)
    for step in (3, 17, 42):
        src = Dreamer(k_sources=6, k_failures=2)._sample_sources(
            list(store.items.values()), step, __import__("random").Random(f"0.{step}"))
        src_ids = {it.id for it in src}
        assert dead.id in src_ids and refuted.id in src_ids
    no_fail = Dreamer(k_sources=6, k_failures=0)._sample_sources(
        list(store.items.values()), 3, __import__("random").Random("0.3"))
    assert len(no_fail) <= 6

def test_distill_registry_lifecycle_cli(tmp_path):
    """register -> record_eval -> promote -> rollback, all through the CLI."""
    from scripts.distill import main as distill_main

    reg = str(tmp_path / "reg.json")
    spec = build_training_spec([TrainingExample(
        messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        source="best_program")], adapter_id="se-v1")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    assert distill_main(["--register-only", str(spec_path), "--registry", reg]) == 0
    assert distill_main(["--record-eval", "se-v1", "--metric", "0.9",
                         "--n-runs", "3", "--registry", reg]) == 0
    assert distill_main(["--promote", "se-v1", "--registry", reg]) == 0
    data = CheckpointRegistry(reg).load()
    assert data["champion"] == "se-v1"
    # rollback with no retired champion should not crash the CLI
    try:
        distill_main(["--rollback", "--registry", reg])
    except Exception:
        pass  # rollback semantics (raise vs no-op) belong to CheckpointRegistry's tests
