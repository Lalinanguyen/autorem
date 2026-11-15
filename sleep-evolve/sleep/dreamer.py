"""The REM dreamer: one role="sleep" LLM call that recombines sampled memory into hypotheses.

Dreaming is generation only: no evaluator calls, no store mutation — sleep/insight_validator.py
decides what survives. The biological point is controlled randomness, so source selection mixes:
  - weighted sampling WITHOUT replacement over non-goal items with strength >= min_strength,
    weight = rank * (1 + age_bonus) where age_bonus = 0.5 for items not supported since
    step - 20 (old material gets a slightly better chance to be recombined), plus
  - a small uniform-random exploration set (~1/3 of k_sources) so the obvious top-ranked items
    do not win every dream.
Sampling is reproducible: rng defaults to a Random seeded from (seed, step), so the same store,
step and seed always yield the same dream_source_ids.

Goals are never sampled as sources; their texts go into <goals> instead. `memory_extra` items
(dicts or MemoryItems) are shown in the memory block without joining the sampling pool.

Bounded prompt: at most `max_steps` step records are shown (the last ones), and a raw JSONL record
(one carrying `program`) is reduced to a compact consolidator_io.step_view (no program text,
side_info/error capped). Before this, FullSleep._dream passed every raw step record of the run,
programs, details and side_info included, so the dream prompt grew with run length (the cause of
the ~140k-token dream-arm sleep call). Programs a dream may diff against come separately as
`bases` (sleep/program_ref.base_programs: count- and char-capped, exact text).
"""

from __future__ import annotations

import json
import random
from typing import Any

from harness.context import count_tokens
from sleep.consolidator_io import RETRY_NOTE, step_view
from sleep.dream_io import DREAM_SYSTEM, DREAM_USER, parse_json_object, validate_dreams
from sleep.memory import MemoryItem, MemoryStore
from sleep.program_ref import render_bases


class Dreamer:
    def __init__(self, k_sources: int = 6, max_dreams: int = 3, dream_max_tokens: int = 4096,
                 retries: int = 1, seed: int = 0, min_strength: float = 0.0,
                 k_failures: int = 2, temperature_note: str = "vary widely",
                 program_chars: int = 2000, max_steps: int = 12):
        self.max_steps = max_steps
        self.k_sources = k_sources
        self.k_failures = k_failures
        self.max_dreams = max_dreams
        self.dream_max_tokens = dream_max_tokens
        self.retries = retries
        self.seed = seed
        self.min_strength = min_strength
        self.temperature_note = temperature_note
        self.program_chars = program_chars
        self.last_prompt: str | None = None

    def dream(self, task: str, store: MemoryStore, step_records: list[dict], llm, *,
              step: int, rng: random.Random | None = None,
              memory_extra: list | None = None, bases: list[dict] | None = None) -> dict:
        """One dream call. Returns metrics; BudgetExhausted propagates out of llm.complete.

        `bases`: programs shown in full (program_ref.base_programs) for diff-first tests."""
        rng = rng if rng is not None else random.Random(f"{self.seed}.{step}")  # reproducible
        pool = [it for it in store.items.values()
                if it.kind != "goal" and (it.strength >= self.min_strength
                                          or it.status == "refuted" or it.kind == "dead_end")]
        sources = self._sample_sources(pool, step, rng)
        items = [self._item_view(it) for it in sources] + [self._item_view(e) for e in (memory_extra or [])]
        goals = [it for it in store.ordered() if it.kind == "goal"]
        directive = (
            f"Produce up to {self.max_dreams} dreams recombined from the sources above. "
            f"Cover the kinds: aim for at least one association, one counterfactual and one "
            f"extrapolation when the material allows it; fewer dreams is fine, never pad. "
            f"Sampling note: {self.temperature_note} — recombine sources that look unrelated; "
            f"novelty is the point, but every dream still cites real ids from memory or steps."
        )
        user = DREAM_USER.substitute(
            task=task, n_items=len(items), memory_items_json=json.dumps(items, indent=0),
            programs=render_bases(list(bases or [])),
            steps_jsonl="\n".join(json.dumps(self._step_view(s), default=str)
                                  for s in (list(step_records)[-self.max_steps:] if self.max_steps else [])),
            goals="\n".join(f"- [{g.id}] {g.text}" for g in goals) or "(none)",
            dream_directive=directive,
        )
        self.last_prompt = user
        messages = [{"role": "system", "content": DREAM_SYSTEM}, {"role": "user", "content": user}]
        obj, err, calls = None, None, 0
        for _ in range(1 + max(0, int(self.retries))):
            resp = llm.complete(messages, role="sleep", max_tokens=self.dream_max_tokens)
            calls += 1
            obj, err = parse_json_object(resp.text)
            if obj is not None:
                break
            messages = messages[:2] + [{"role": "assistant", "content": resp.text[:2000]},
                                       {"role": "user", "content": RETRY_NOTE}]
        dreams, problems = validate_dreams(obj, max_dreams=self.max_dreams)
        for d in dreams:  # bound what can reach the sandbox (full programs only; refs/diffs stay exact)
            if d["test"]:
                for k in ("control", "treatment"):
                    v = d["test"].get(k)
                    if isinstance(v, str):
                        d["test"][k] = v[: self.program_chars]
                    elif isinstance(v, dict) and isinstance(v.get("program"), str):
                        d["test"][k] = {**v, "program": v["program"][: self.program_chars]}
        return {
            "dream_calls": calls, "n_dreams": len(dreams), "dreams": dreams,
            "dream_parse_error": err or ("; ".join(problems) if problems else None),
            "dream_source_ids": [it.id for it in sources],
            "dream_prompt_tokens": count_tokens(DREAM_SYSTEM) + count_tokens(user),
        }

    @staticmethod
    def _step_view(s) -> dict:
        """A raw JSONL step record (it has `program`) -> compact step_view; other dicts as given."""
        if isinstance(s, dict) and "program" in s:
            return step_view(s, {})
        return s if isinstance(s, dict) else {"step": s}

    # ---- source sampling -----------------------------------------------------------
    def _sample_sources(self, pool: list[MemoryItem], step: int, rng: random.Random) -> list[MemoryItem]:
        """Failed material first (substrate for counterfactuals), then ~1/3 of the remaining
        slots uniform at random (exploration), the rest rank-weighted."""
        if not pool or self.k_sources <= 0:
            return []
        failures = [it for it in pool if it.status == "refuted" or it.kind == "dead_end"]
        n_fail = min(len(failures), max(0, self.k_failures), self.k_sources)
        chosen_failures = rng.sample(failures, n_fail)
        chosen = {it.id for it in chosen_failures}
        pool = [it for it in pool if it.id not in chosen]
        k_rest = self.k_sources - n_fail
        n_explore = min(len(pool), max(1, round(k_rest / 3)), k_rest)
        explore = rng.sample(pool, n_explore) if n_explore else []
        chosen = {it.id for it in explore}
        rest = [it for it in pool if it.id not in chosen]
        weights = [it.rank * (1.0 + (0.5 if it.last_supported_step < step - 20 else 0.0))
                   for it in rest]
        weighted = self._weighted_sample(rest, weights, k_rest - n_explore, rng)
        return chosen_failures + explore + weighted

    @staticmethod
    def _weighted_sample(items: list[MemoryItem], weights: list[float], k: int,
                         rng: random.Random) -> list[MemoryItem]:
        """Weighted sample without replacement; all-zero weights fall back to uniform."""
        items, weights, out = list(items), list(weights), []
        while items and len(out) < k:
            total = sum(weights)
            if total <= 0:
                idx = rng.randrange(len(items))
            else:
                r, acc, idx = rng.random() * total, 0.0, len(items) - 1
                for j, w in enumerate(weights):
                    acc += w
                    if r < acc:
                        idx = j
                        break
            out.append(items.pop(idx))
            weights.pop(idx)
        return out

    @staticmethod
    def _item_view(it: Any) -> dict:
        """id/kind/text/strength/status/approach_tags — the fields a dream may cite."""
        if isinstance(it, MemoryItem):
            return {"id": it.id, "kind": it.kind, "text": it.text, "strength": round(it.strength, 3),
                    "status": it.status, "approach_tags": it.approach_tags}
        it = it if isinstance(it, dict) else {}
        return {"id": it.get("id"), "kind": it.get("kind"), "text": it.get("text"),
                "strength": it.get("strength"), "status": it.get("status"),
                "approach_tags": it.get("approach_tags") or []}
