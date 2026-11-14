"""FullSleep: the multi-stage consolidation pipeline. STUBS ONLY; see DEVIN_TASKS.md task 4.

Pipeline, run every `every_n` steps:
    prioritize -> replay -> abstract -> prune -> integrate
Every LLM call in any stage must use llm.complete(..., role="sleep").
"""

from __future__ import annotations

from typing import Any


class FullSleep:
    name = "full_sleep"

    def __init__(self, every_n: int = 10, max_block_tokens: int = 2048, **params: Any):
        self.every_n = every_n
        self.max_block_tokens = max_block_tokens
        self.params = params

    def should_sleep(self, step: int, history: list[dict]) -> bool:
        return step > 0 and step % self.every_n == 0

    def sleep(self, task, block, history, llm, budget) -> str:
        items = self.prioritize(task, block, history, llm, budget)
        evidence = self.replay(task, items, llm, budget)
        abstractions = self.abstract(task, block, items, evidence, llm, budget)
        kept = self.prune(task, block, abstractions, llm, budget)
        return self.integrate(task, block, kept, llm, budget)

    def prioritize(self, task, block, history, llm, budget) -> list[dict]:
        """Pick the history items worth consolidating.

        Score each step since the last sleep by surprise and usefulness: large
        positive/negative delta_vs_best, new error types, lineages that improved, and
        repeated failures. Return the top items (records from `history`), bounded so
        the later stages fit the budget. Uses no LLM, or a cheap one.
        """
        raise NotImplementedError

    def replay(self, task, items, llm, budget) -> list[dict]:
        """Re-test hypotheses by running code against an EVALUATOR SNAPSHOT.

        For selected items, the LLM proposes small probes (ablations, parameter sweeps,
        minimal variants). A replay tool runs them on a frozen copy of the evaluator, in a
        subprocess or Modal, never in-process. Probe results are evidence, NOT new search
        steps: they are not added to the ProgramDB or the agent's last-K window. LLM tokens
        go to role="sleep". Evaluator compute is tracked separately in the run summary.
        """
        raise NotImplementedError

    def abstract(self, task, block, items, evidence, llm, budget) -> list[str]:
        """Turn prioritized items plus replay evidence into general, reusable insights.

        Each insight is a short claim with its support (step ids, probe results), such as
        "radii pinned to the boundary score higher; steps 12, 19; probe r3". Uses
        role="sleep" LLM calls.
        """
        raise NotImplementedError

    def prune(self, task, block, abstractions, llm, budget) -> list[str]:
        """Drop stale, contradicted, or redundant insights from the old block plus the new ones.

        Must keep the block under max_block_tokens. Contradictions are resolved toward
        the newer, better-supported claim.
        """
        raise NotImplementedError

    def integrate(self, task, block, kept, llm, budget) -> str:
        """Write the final consolidated block the agent will read in place of raw history.

        A stable structure (e.g. Best so far / Insights / Dead ends / Next ideas) keeps
        blocks comparable across sleeps. Returns the new block string.
        """
        raise NotImplementedError
