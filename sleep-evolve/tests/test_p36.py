"""Problem 36 evaluator. Reference construction: AlphaEvolve's n=26 packing from the repo notebook."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from evaluators.p36_circle_packing import BEST_KNOWN, CirclePacking26, check_packing

DATA = Path(__file__).with_name("data") / "p36_alphaevolve_n26.json"


def program_returning(centers, radii) -> str:
    return f"def construct():\n    return {json.dumps(centers)}, {json.dumps(radii)}\n"


@pytest.fixture(scope="module")
def ev():
    return CirclePacking26(timeout_s=30)


def test_alphaevolve_construction_reproduces_best_known(ev):
    d = json.loads(DATA.read_text())
    res = ev.evaluate(program_returning(d["centers"], d["radii"]))
    assert res.valid, res.error
    assert res.score == pytest.approx(BEST_KNOWN, abs=1e-12)
    assert res.score == pytest.approx(2.635983, abs=1e-6)


def test_hand_made_grid_scores_its_sum(ev):
    # 26 circles of radius 1/12 on a 6x5 grid: touching neighbours, touching the walls.
    centers = [[(j + 0.5) / 6, (i + 0.5) / 5] for i, j in (divmod(k, 6) for k in range(26))]
    radii = [1 / 12] * 26
    res = ev.evaluate(program_returning(centers, radii))
    assert res.valid, res.error
    assert res.score == pytest.approx(26 / 12, abs=1e-12)


def test_seed_program_is_valid(ev):
    res = ev.evaluate(ev.seed_program())
    assert res.valid, res.error
    assert res.score == pytest.approx(26 / 12)


def test_overlapping_packing_is_invalid(ev):
    centers = [[(j + 0.5) / 6, (i + 0.5) / 5] for i, j in (divmod(k, 6) for k in range(26))]
    radii = [1 / 12] * 26
    radii[7] = 0.1  # grows into its horizontal neighbours (spacing 1/6 = 2 * 1/12)
    res = ev.evaluate(program_returning(centers, radii))
    assert not res.valid and "overlap" in res.error
    assert res.score == -math.inf


def test_circle_outside_square_is_invalid(ev):
    centers = [[(j + 0.5) / 6, (i + 0.5) / 5] for i, j in (divmod(k, 6) for k in range(26))]
    centers[0] = [0.05, 0.1]  # radius 1/12 > 0.05 -> crosses the left edge
    res = ev.evaluate(program_returning(centers, [1 / 12] * 26))
    assert not res.valid and "outside" in res.error


def test_wrong_count_is_invalid(ev):
    res = ev.evaluate(program_returning([[0.5, 0.5]], [0.5]))
    assert not res.valid


def test_tolerance_is_1e9():
    c = np.array([[0.25, 0.5], [0.75, 0.5]])
    assert check_packing(c, np.array([0.25, 0.25 + 5e-10]), n=2) is None  # within 1e-9
    assert check_packing(c, np.array([0.25, 0.25 + 5e-9]), n=2) is not None
