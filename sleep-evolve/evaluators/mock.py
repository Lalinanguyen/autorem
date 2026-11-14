"""A cheap, deterministic evaluator for harness tests and local dry runs.

construct() returns a list of DIM floats. score = 1 - mean((x - TARGET)^2), so the
maximum is 1.0 at x == TARGET. It still runs in a subprocess, so the timeout and
crash paths are the real ones.
"""

from __future__ import annotations

import math
from typing import Any

from evaluators.base import EvalResult, SubprocessEvaluator, invalid

DIM = 5
TARGET = 0.7


class MockEvaluator(SubprocessEvaluator):
    problem_id = 0
    name = "mock_quadratic"
    best_known = 1.0
    description = (
        f"Write construct() returning a list of {DIM} floats. "
        f"Score is 1 - mean((x - t)^2) for a hidden target t; higher is better."
    )

    def __init__(self, timeout_s: int = 10):
        self.timeout_s = timeout_s

    def seed_program(self) -> str:
        return f"def construct():\n    return [0.0] * {DIM}\n"

    def score_solution(self, solution: Any) -> EvalResult:
        if not isinstance(solution, list) or len(solution) != DIM:
            return invalid(f"expected a list of {DIM} numbers")
        xs = [float(v) for v in solution]
        if not all(math.isfinite(v) for v in xs):
            return invalid("non-finite value")
        score = 1.0 - sum((v - TARGET) ** 2 for v in xs) / DIM
        return EvalResult(score=score, valid=True)
