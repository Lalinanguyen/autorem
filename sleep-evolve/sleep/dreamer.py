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
"""

from __future__ import annotations

import json
import random
from typing import Any

from sleep.consolidator_io import RETRY_NOTE
from sleep.dream_io import DREAM_SYSTEM, DREAM_USER, parse_json_object, validate_dreams
from sleep.memory import MemoryItem, MemoryStore


class Dreamer:
    def __init__(self, k_sources: int = 6, max_dreams: int = 3, dream_max_tokens: int = 4096,
                 retries: int = 1, seed: int = 0, min_strength: float = 0.0,
                 temperature_note: str = "vary widely", program_chars: int = 2000):
        self.k_sources = k_sources
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
              memory_extra: list | None = None) -> dict:
        """One dream call. Returns metrics; BudgetExhausted propagates out of llm.complete."""
        rng = rng if rng is not None else random.Random(f"{self.seed}.{step}")  # reproducible
        pool = [it for it in store.items.values()
                if it.kind != "goal" and it.strength >= self.min_strength]
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
            steps_jsonl="\n".join(json.dumps(s, default=str) for s in step_records),
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
        for d in dreams:  # bound what can reach the sandbox
            if d["test"]:
                for k in ("control", "treatment"):
                    if isinstance(d["test"].get(k), str):
                        d["test"][k] = d["test"][k][: self.program_chars]
        return {
            "dream_calls": calls, "n_dreams": len(dreams), "dreams": dreams,
            "dream_parse_error": err or ("; ".join(problems) if problems else None),
            "dream_source_ids": [it.id for it in sources],
        }

    # ---- source sampling -----------------------------------------------------------
    def _sample_sources(self, pool: list[MemoryItem], step: int, rng: random.Random) -> list[MemoryItem]:
        """~1/3 of k_sources uniform at random (exploration), the rest rank-weighted."""
        if not pool or self.k_sources <= 0:
            return []
        n_explore = min(len(pool), max(1, round(self.k_sources / 3)), self.k_sources)
        explore = rng.sample(pool, n_explore)
        chosen = {it.id for it in explore}
        rest = [it for it in pool if it.id not in chosen]
        weights = [it.rank * (1.0 + (0.5 if it.last_supported_step < step - 20 else 0.0))
                   for it in rest]
        weighted = self._weighted_sample(rest, weights, self.k_sources - n_explore, rng)
        return explore + weighted

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
