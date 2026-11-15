"""Baseline: never consolidate. The agent sees task + last K steps (+ parent program).

Optional eval-matched control (`extra_eval_schedule`, the baseline arm no_sleep_extra_evals):
[[step, k], ...] asks the loop to spend k extra evaluator calls right after agent step `step`, each on
a small perturbation of that step's parent (harness/perturb.py). The schedule is copied from the
paired sleeping run's log (harness/baselines.py: k = the sleep evaluations that run spent at its sleep
after `step`), so both arms use the same number of evaluator calls at the same points. No LLM call,
no memory: the loop does the evaluating (harness/loop.py), because only it owns the ProgramDB.
"""

from __future__ import annotations


class NoSleep:
    name = "no_sleep"
    extra_eval_schedule: dict = {}  # class defaults (read-only) for subclasses that skip __init__
    extra_eval_sigma: float = 0.05

    def __init__(self, extra_eval_schedule=None, extra_eval_sigma: float = 0.05):
        sched: dict[int, int] = {}
        for item in extra_eval_schedule or []:
            step, k = (item if isinstance(item, (list, tuple)) else (item, 1))
            step, k = int(step), int(k)
            if step <= 0 or k < 0:
                raise ValueError(f"extra_eval_schedule entries must be [step > 0, k >= 0], got {item!r}")
            sched[step] = sched.get(step, 0) + k
        self.extra_eval_schedule = sched
        self.extra_eval_sigma = float(extra_eval_sigma)

    def should_sleep(self, step: int, history: list[dict]) -> bool:
        return False

    def sleep(self, task, block, history, llm, budget) -> str:
        return block

    def extra_evals_after(self, step: int) -> int:
        """Extra perturbation evaluations the loop should run after agent step `step` (0 = none)."""
        return self.extra_eval_schedule.get(step, 0)
