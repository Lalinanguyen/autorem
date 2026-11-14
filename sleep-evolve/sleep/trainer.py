"""Level-2 parametric learning: turn validated sleep-time discoveries into training data.

Inference APIs do not change a model's weights. Prompting, memory blocks, replay and
consolidation only move what fits inside the context window; the base model stays bit-for-bit
identical. Real weight updates happen ONLY through this pipeline:

    collect_examples -> build_training_spec -> Modal GPU train (infra/modal_train.py)
        -> held-out evals -> CheckpointRegistry.promote_if_better -> serve the adapter via vLLM

Promotion is earned, not assumed: an adapter becomes champion only after record_eval has logged
a measured improvement over the current champion on held-out runs. Training loss alone never
promotes anything.

Everything in this module is pure Python and fully testable offline: no Modal import, no LLM
calls, no evaluations. The heavyweight training code lives inside infra/modal_train.py, which
pytest never imports.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SOURCES = ("dream", "best_program", "verified_item")
MEMORY_TOKEN_CAP = 400
STRATEGY_QUESTION = "What strategy applies here?"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrainingExample:
    """One SFT example: a chat transcript plus provenance.

    messages: [{"role": "user"|"assistant"|"system", "content": str}, ...]
    source:   "dream" | "best_program" | "verified_item"
    weight:   relative importance hint for the trainer (dream-derived > best_program > verified_item)
    meta:     provenance (step/score/program_id or item_id); always JSON-serializable
    """

    messages: list[dict]
    source: str
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown example source {self.source!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict())


def _user_prompt(task: str, best_score: float | None, memory_block: str) -> str:
    """Compact 'task + best-so-far + memory context' string for the user message."""
    parts = [f"Task:\n{task}"]
    parts.append(f"Best score so far: {best_score:.6g}" if best_score is not None
                 else "Best score so far: none yet")
    if memory_block:
        parts.append(f"Memory:\n{memory_block}")
    parts.append("Write the next program.")
    return "\n\n".join(parts)


def collect_examples(task: str, store, history: list[dict], *, max_examples: int = 64,
                     min_score: float | None = None, best_k: int = 5,
                     include_dreams: bool = True) -> list[TrainingExample]:
    """Build SFT examples from run history (step-record dicts) and a MemoryStore. Pure function.

    History: dicts like the history.jsonl lines — keys step/score/valid/program/program_id/event.
    The best `best_k` distinct VALID programs by score (>= min_score if given) become examples:
    user = task + best-score-so-far + rendered memory, assistant = the program text.

    Store: items with status "verified" and kind in ("skill", "rule") become short Q/A pairs at
    weight 0.5. Verified items whose evidence contains an id starting with "d" are dream-derived:
    when include_dreams they are emitted as source "dream" at weight 1.0, else dropped entirely.

    Ordering: program examples by score desc then step asc; store items by weight desc then item id.
    Programs dedupe by stripped text. Result capped at max_examples. No LLM calls, no evals.
    """
    # Running best valid score BEFORE each step (the context the agent actually saw).
    best_before: dict[int, float | None] = {}
    best: float | None = None
    for r in sorted(history, key=lambda r: r.get("step") or 0):
        if r.get("event", "step") != "step":
            continue
        step = r.get("step") or 0
        best_before.setdefault(step, best)
        if r.get("valid") and r.get("score") is not None:
            best = r["score"] if best is None else max(best, r["score"])

    steps = [r for r in history
             if r.get("event", "step") == "step" and r.get("valid")
             and r.get("program") and r.get("score") is not None]
    if min_score is not None:
        steps = [r for r in steps if r["score"] >= min_score]
    steps.sort(key=lambda r: (-r["score"], r.get("step") or 0))

    memory_block = store.render(MEMORY_TOKEN_CAP) if store is not None else ""

    picked: list[dict] = []
    seen: set[str] = set()
    for r in steps:
        if len(picked) >= best_k:
            break
        key = " ".join(str(r["program"]).split())
        if key in seen:
            continue
        seen.add(key)
        picked.append(r)

    examples = [
        TrainingExample(
            messages=[
                {"role": "user",
                 "content": _user_prompt(task, best_before.get(r.get("step") or 0), memory_block)},
                {"role": "assistant", "content": r["program"]},
            ],
            source="best_program",
            weight=1.0,
            meta={"step": r.get("step"), "score": r["score"], "program_id": r.get("program_id")},
        )
        for r in picked
    ]

    store_exs: list[TrainingExample] = []
    for it in store.items.values():
        if it.status != "verified" or it.kind not in ("skill", "rule"):
            continue
        dream = any(str(e).startswith("d") for e in it.evidence)
        if dream:
            if not include_dreams:
                continue
            source, weight = "dream", 1.0
        else:
            source, weight = "verified_item", 0.5
        store_exs.append(TrainingExample(
            messages=[{"role": "user", "content": STRATEGY_QUESTION},
                      {"role": "assistant", "content": it.text}],
            source=source, weight=weight,
            meta={"item_id": it.id, "kind": it.kind}))
    store_exs.sort(key=lambda e: (-e.weight, str(e.meta["item_id"])))

    return (examples + store_exs)[:max_examples]


def build_training_spec(examples: list[TrainingExample], *, base_model: str = "Qwen/Qwen3-8B",
                        lora_rank: int = 16, epochs: int = 1, lr: float = 1e-4,
                        max_len: int = 4096, adapter_id: str | None = None) -> dict:
    """Job spec dict consumed by infra/modal_train.py's train().

    Refuses (ValueError) on an empty example list — a paid GPU job is never launched on nothing.
    adapter_id defaults to a stable id derived from the first example's content and the count,
    so the same data always produces the same id.
    """
    if not examples:
        raise ValueError("refusing to build a training spec with zero examples")
    if adapter_id is None:
        digest = hashlib.sha256(
            f"{examples[0].to_jsonl()}|{len(examples)}".encode("utf-8")).hexdigest()[:12]
        adapter_id = f"se-{digest}"
    return {
        "adapter_id": adapter_id,
        "base_model": base_model,
        "lora_rank": int(lora_rank),
        "lora_alpha": int(2 * lora_rank),
        "epochs": int(epochs),
        "lr": float(lr),
        "max_len": int(max_len),
        "n_examples": len(examples),
        "examples_jsonl": "\n".join(e.to_jsonl() for e in examples),
    }


class CheckpointRegistry:
    """JSON-file registry of trained LoRA adapters and the single champion.

    An adapter starts as a "candidate", accumulates evals via record_eval, and is promoted to
    "champion" only when its latest eval metric beats the current champion's by > min_delta.
    A demoted champion becomes "retired" and can be restored by rollback().

    The file may not exist initially (empty registry). A corrupt file raises ValueError —
    a half-destroyed registry is never silently trusted.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = self.load()

    def load(self) -> dict:
        """(Re)read the registry file. Missing file -> empty registry; corrupt JSON -> ValueError."""
        if not self.path.exists():
            self._data = {"champion": None, "adapters": {}}
            return self._data
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"corrupt checkpoint registry {self.path}: {e}") from e
        if not isinstance(data, dict) or not isinstance(data.get("adapters"), dict):
            raise ValueError(f"corrupt checkpoint registry {self.path}: not a registry dict")
        data.setdefault("champion", None)
        self._data = data
        return self._data

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    def register(self, adapter_id: str, *, base_model: str, train_meta: dict) -> dict:
        """Append a new candidate record. Duplicate adapter ids are refused (ValueError)."""
        if adapter_id in self._data["adapters"]:
            raise ValueError(f"adapter {adapter_id!r} is already registered")
        rec = {"adapter_id": adapter_id, "base_model": base_model,
               "registered_at": _utcnow(), "train_meta": dict(train_meta),
               "status": "candidate", "evals": []}
        self._data["adapters"][adapter_id] = rec
        self.save()
        return rec

    def record_eval(self, adapter_id: str, metric: float, *, n_runs: int, note: str = "") -> dict:
        """Append {metric, n_runs, note, at} to the adapter's evals. Unknown id -> KeyError."""
        rec = self._data["adapters"].get(adapter_id)
        if rec is None:
            raise KeyError(f"unknown adapter {adapter_id!r}")
        ev = {"metric": float(metric), "n_runs": int(n_runs), "note": note, "at": _utcnow()}
        rec["evals"].append(ev)
        self.save()
        return ev

    def promote_if_better(self, adapter_id: str, *, min_delta: float = 0.0) -> dict:
        """Promote `adapter_id` iff its LATEST eval beats the champion's latest by > min_delta.

        A candidate with no evals can never be promoted. With no champion, the first adapter
        that has an eval becomes champion. Returns {"promoted", "champion", "reason"}.
        """
        rec = self._data["adapters"].get(adapter_id)
        if rec is None:
            raise KeyError(f"unknown adapter {adapter_id!r}")
        champ_id = self._data["champion"]
        if not rec["evals"]:
            return {"promoted": False, "champion": champ_id,
                    "reason": f"{adapter_id} has no evals"}
        metric = rec["evals"][-1]["metric"]
        if champ_id is None:
            self._promote(rec)
            return {"promoted": True, "champion": adapter_id,
                    "reason": "first adapter with an eval"}
        champ = self._data["adapters"][champ_id]
        champ_metric = champ["evals"][-1]["metric"] if champ["evals"] else float("-inf")
        if metric > champ_metric + min_delta:
            champ["status"] = "retired"
            self._promote(rec)
            return {"promoted": True, "champion": adapter_id,
                    "reason": f"{metric:.6g} > champion {champ_metric:.6g} + {min_delta}"}
        return {"promoted": False, "champion": champ_id,
                "reason": f"{metric:.6g} <= champion {champ_metric:.6g} + {min_delta}"}

    def _promote(self, rec: dict) -> None:
        rec["status"] = "champion"
        self._data["champion"] = rec["adapter_id"]
        self.save()

    def champion(self) -> dict | None:
        """The current champion record, or None if nothing has ever been promoted."""
        cid = self._data["champion"]
        return self._data["adapters"].get(cid) if cid else None

    def rollback(self) -> dict:
        """Retire the current champion and re-promote the most recently registered retired
        adapter that has evals. Returns {"rolled_back", "champion", "reason"}."""
        champ_id = self._data["champion"]
        if champ_id is None:
            return {"rolled_back": False, "champion": None, "reason": "no champion"}
        # Candidates BEFORE the current champion is retired, or it would immediately re-win.
        retired = [r for r in self._data["adapters"].values()
                   if r["status"] == "retired" and r["evals"]]
        if not retired:
            return {"rolled_back": False, "champion": champ_id,
                    "reason": "no retired adapter with evals"}
        prev = retired[-1]  # dicts keep insertion order: last-registered retired = most recent
        self._data["adapters"][champ_id]["status"] = "retired"
        self._promote(prev)
        return {"rolled_back": True, "champion": prev["adapter_id"],
                "reason": f"rolled back from {champ_id} to {prev['adapter_id']}"}
