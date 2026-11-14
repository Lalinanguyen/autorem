"""Evaluator registry. Add new problems to REGISTRY (problem_id -> factory)."""

from __future__ import annotations

from typing import Callable

from evaluators.base import EvalResult, Evaluator
from evaluators.mock import MockEvaluator
from evaluators.p36_circle_packing import CirclePacking26

REGISTRY: dict[int, Callable[..., Evaluator]] = {
    0: MockEvaluator,
    36: CirclePacking26,
}


def make_evaluator(problem_id: int, **kwargs) -> Evaluator:
    try:
        factory = REGISTRY[problem_id]
    except KeyError:
        raise ValueError(f"no evaluator for problem {problem_id}; known: {sorted(REGISTRY)}") from None
    return factory(**kwargs)


__all__ = ["EvalResult", "Evaluator", "REGISTRY", "make_evaluator"]
