import random
from pathlib import Path

import pytest

from harness.config import RunConfig, load_config
from harness.program_db import Program, ProgramDB
from scripts.run_matrix import build_matrix
from sleep import make_policy
from sleep.full import FullSleep


def test_parent_sampling_best_70_percent_else_top10():
    db = ProgramDB()
    for i in range(20):
        db.add(Program(f"p{i}", "", float(i), True, None, i))
    db.add(Program("bad", "", float("-inf"), False, None, 99))
    rng = random.Random(0)
    picks = [db.sample_parent(rng).id for _ in range(10_000)]
    top10 = {f"p{i}" for i in range(10, 20)}
    assert set(picks) <= top10
    # P(best) = 0.7 + 0.3 / 10 = 0.73
    assert picks.count("p19") / len(picks) == pytest.approx(0.73, abs=0.02)


def test_parent_falls_back_to_seed_when_nothing_valid():
    db = ProgramDB()
    db.add(Program("p0", "", float("-inf"), False, None, 0))
    db.add(Program("p1", "", float("-inf"), False, "p0", 1))
    assert db.sample_parent(random.Random(0)).id == "p0"


def test_full_sleep_stages_are_stubs():
    fs = FullSleep(every_n=5)
    assert fs.should_sleep(5, []) and not fs.should_sleep(4, [])
    for stage in ("prioritize", "replay", "abstract", "prune", "integrate"):
        assert getattr(FullSleep, stage).__doc__
    with pytest.raises(NotImplementedError):
        fs.sleep("task", "", [], None, None)


def test_configs_load_and_policies_resolve():
    root = Path(__file__).resolve().parents[1]
    for name in ("example.yaml", "mock.yaml"):
        cfg = load_config(root / "configs" / name)
        make_policy(cfg.sleep_policy, cfg.policy_params)


def test_normalized_score_handles_minimization():
    from evaluators.base import normalized_score

    assert normalized_score(2.0, 4.0) == 0.5  # maximize
    assert normalized_score(-8.0, -4.0) == 0.5  # minimize (negated): worse -> below 1
    assert normalized_score(-2.0, -4.0) == 2.0  # beats best_known
    assert normalized_score(float("-inf"), 4.0) is None


def test_matrix_is_full_cross_product():
    base = RunConfig(problem_id=0, total_token_budget=10, max_steps=1)
    cfgs = build_matrix(base, [0, 36], ["no_sleep", "summarize"], [0, 1, 2])
    assert len(cfgs) == 12
    assert len({c.resolved_run_id() for c in cfgs}) == 12
