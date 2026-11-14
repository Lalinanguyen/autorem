"""Structured long-term memory for FullSleep: MemoryItem, Edit, MemoryStore.

Everything here is deterministic (no randomness, no wall clock): the same sequence of
edits/decays/evictions always yields the same store, ids and rendering.

Kinds:   goal | rule | fact | skill | dead_end
Status:  tentative | verified | contested | refuted
Goals are never decayed or evicted and always render first.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from harness.context import fit_lines

KINDS = ("goal", "rule", "fact", "skill", "dead_end")
STATUSES = ("tentative", "verified", "contested", "refuted")
STATUS_WEIGHT = {"verified": 1.0, "tentative": 0.6, "contested": 0.35, "refuted": 0.2}
EDIT_OPS = ("add", "support", "contradict", "refute")
MAX_EVIDENCE = 8
MAX_TAGS = 8


def normalize_tags(tags: Iterable | str | None) -> list[str]:
    """Lowercase, spaces/underscores -> '-', strip punctuation, dedupe (order kept), at most MAX_TAGS."""
    if tags is None:
        return []
    if isinstance(tags, str):
        tags = re.split(r"[,;|]", tags)
    out: list[str] = []
    for t in tags:
        if not isinstance(t, (str, int, float)):
            continue
        s = re.sub(r"[\s_]+", "-", str(t).strip().lower())
        s = re.sub(r"[^a-z0-9.+=-]", "", s).strip("-")[:40]
        if s and s not in ("none", "n-a", "na") and s not in out:
            out.append(s)
    return out[:MAX_TAGS]


@dataclass
class MemoryItem:
    id: str
    kind: str
    text: str
    strength: float
    status: str = "tentative"
    evidence: list = field(default_factory=list)  # step ids (or replay ids like "r12.0")
    approach_tags: list[str] = field(default_factory=list)
    scope: str = "run"
    created_step: int = 0
    last_supported_step: int = 0

    @property
    def rank(self) -> float:
        return self.strength * STATUS_WEIGHT.get(self.status, 0.5)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Edit:
    """One memory edit. op: add | support | contradict | refute.

    add:        `item` holds the new item's fields (kind, text, status, strength, ...).
    support:    strengthen `target`; status may be promoted (e.g. to verified).
    contradict: weaken `target`; a verified item becomes contested.
    refute:     mark `target` refuted. With no target and an `item`, the claim is added
                with status refuted (a tested-and-false claim the agent should not retry).
    """

    op: str
    target: str | None = None
    item: dict | None = None
    status: str | None = None
    evidence: list = field(default_factory=list)
    amount: float | None = None
    source: str = "claim"  # claim | replay | goal

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


class MemoryStore:
    def __init__(
        self,
        *,
        initial_strength: float = 0.3,
        verified_strength: float = 0.7,
        support_gain: float = 0.15,
        contradict_loss: float = 0.25,
        max_item_chars: int = 300,
    ):
        self.items: dict[str, MemoryItem] = {}
        self.initial_strength = initial_strength
        self.verified_strength = verified_strength
        self.support_gain = support_gain
        self.contradict_loss = contradict_loss
        self.max_item_chars = max_item_chars
        self._next = 1

    # ---- basic access -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def get(self, item_id: str) -> MemoryItem | None:
        return self.items.get(item_id)

    def find_text(self, text: str) -> MemoryItem | None:
        key = _norm_text(text)
        return next((it for it in self.items.values() if _norm_text(it.text) == key), None)

    def add(self, *, kind: str, text: str, step: int, status: str = "tentative",
            strength: float | None = None, evidence: Iterable = (), approach_tags: Iterable = (),
            scope: str = "run", item_id: str | None = None) -> MemoryItem:
        if kind not in KINDS:
            raise ValueError(f"unknown memory kind {kind!r}")
        if status not in STATUSES:
            raise ValueError(f"unknown status {status!r}")
        text = " ".join(str(text).split())[: self.max_item_chars]
        if not text:
            raise ValueError("empty memory text")
        if item_id is None:
            item_id = f"m{self._next}"
            self._next += 1
        elif item_id in self.items:
            raise ValueError(f"memory id {item_id!r} already exists")
        if strength is None:
            strength = self.verified_strength if status == "verified" else self.initial_strength
        it = MemoryItem(
            id=item_id, kind=kind, text=text, strength=_clip(strength), status=status,
            evidence=_dedupe(evidence)[-MAX_EVIDENCE:], approach_tags=normalize_tags(list(approach_tags)),
            scope=scope, created_step=step, last_supported_step=step,
        )
        self.items[item_id] = it
        return it

    # ---- the four mechanisms ------------------------------------------------------
    def support(self, item_id: str, step: int, evidence: Iterable = (), amount: float | None = None,
                status: str | None = None) -> bool:
        it = self.items.get(item_id)
        if it is None:
            return False
        it.strength = _clip(it.strength + (self.support_gain if amount is None else amount))
        if status == "verified":
            it.status = "verified"
            it.strength = max(it.strength, self.verified_strength)
        elif it.status in ("contested", "refuted"):
            it.status = "tentative"
        it.evidence = _dedupe(list(it.evidence) + list(evidence))[-MAX_EVIDENCE:]
        it.last_supported_step = max(it.last_supported_step, step)
        return True

    def contradict(self, item_id: str, step: int, evidence: Iterable = (), amount: float | None = None) -> bool:
        it = self.items.get(item_id)
        if it is None or it.kind == "goal":
            return False
        it.strength = _clip(it.strength - (self.contradict_loss if amount is None else amount))
        if it.status == "verified":
            it.status = "contested"
        it.evidence = _dedupe(list(it.evidence) + list(evidence))[-MAX_EVIDENCE:]
        return True

    def refute(self, item_id: str, step: int, evidence: Iterable = ()) -> bool:
        it = self.items.get(item_id)
        if it is None or it.kind == "goal":
            return False
        it.status = "refuted"
        it.strength = _clip(it.strength * 0.5)
        it.evidence = _dedupe(list(it.evidence) + list(evidence))[-MAX_EVIDENCE:]
        return True

    def decay(self, step: int, factor: float, *, exempt: Iterable[str] = ()) -> int:
        """Multiply every non-goal item's strength by `factor` (items in `exempt` skipped). Returns count."""
        skip = set(exempt)
        n = 0
        for it in self.items.values():
            if it.kind == "goal" or it.id in skip:
                continue
            it.strength = _clip(round(it.strength * factor, 12))
            n += 1
        return n

    def evict(self, *, min_strength: float, max_items: int | None = None) -> list[str]:
        """Remove non-goal items below min_strength, then the lowest-ranked until <= max_items.

        Order is fully determined by (rank, last_supported_step, created_step, id).
        Returns the evicted ids in eviction order.
        """
        evicted = [it.id for it in sorted(self.items.values(), key=lambda i: i.id)
                   if it.kind != "goal" and it.strength < min_strength]
        for i in evicted:
            del self.items[i]
        if max_items is not None and len(self.items) > max_items:
            candidates = sorted(
                (it for it in self.items.values() if it.kind != "goal"),
                key=lambda i: (i.rank, i.last_supported_step, i.created_step, _id_key(i.id)),
            )
            for it in candidates[: len(self.items) - max_items]:
                del self.items[it.id]
                evicted.append(it.id)
        return evicted

    def apply_edits(self, edits: Iterable[Edit], step: int) -> tuple[list[Edit], list[tuple[Edit, str]]]:
        """Apply edits in order. Returns (applied, [(rejected_edit, reason)]). Never raises on bad edits."""
        applied: list[Edit] = []
        rejected: list[tuple[Edit, str]] = []
        for e in edits:
            try:
                reason = self._apply_one(e, step)
            except (ValueError, TypeError) as ex:
                reason = f"{type(ex).__name__}: {ex}"
            if reason is None:
                applied.append(e)
            else:
                rejected.append((e, reason))
        return applied, rejected

    def _apply_one(self, e: Edit, step: int) -> str | None:
        if e.op not in EDIT_OPS:
            return f"unknown op {e.op!r}"
        if e.op == "add" or (e.op == "refute" and e.target is None):
            if not e.item or not str(e.item.get("text", "")).strip():
                return "add without item text"
            status = "refuted" if e.op == "refute" else (e.status or e.item.get("status") or "tentative")
            dup = self.find_text(e.item["text"])
            if dup is not None:  # same claim again: treat as support / refute of the existing item
                if status == "refuted":
                    return None if self.refute(dup.id, step, e.evidence) else "cannot refute a goal"
                self.support(dup.id, step, e.evidence, status=status)
                e.target = dup.id
                return None
            it = self.add(kind=e.item.get("kind", "fact"), text=e.item["text"], step=step, status=status,
                          strength=e.item.get("strength"), evidence=e.evidence or e.item.get("evidence", ()),
                          approach_tags=e.item.get("approach_tags", ()), scope=e.item.get("scope", "run"),
                          item_id=e.item.get("id"))
            if status == "refuted":
                it.strength = _clip(self.initial_strength * 0.5)
            e.target = it.id
            return None
        if e.target not in self.items:
            return f"unknown target {e.target!r}"
        ok = {
            "support": lambda: self.support(e.target, step, e.evidence, e.amount, e.status),
            "contradict": lambda: self.contradict(e.target, step, e.evidence, e.amount),
            "refute": lambda: self.refute(e.target, step, e.evidence),
        }[e.op]()
        return None if ok else f"{e.op} not allowed on {e.target}"

    # ---- queries ------------------------------------------------------------------
    def match_dead_ends(self, tags: Iterable[str], threshold: float = 0.5) -> list[MemoryItem]:
        """Dead-end items (not refuted) whose tags are covered by `tags`.

        Match iff |dead_end_tags & tags| / |dead_end_tags| >= threshold. Sorted by rank desc, then id.
        """
        query = set(normalize_tags(list(tags)))
        if not query:
            return []
        hits = []
        for it in self.items.values():
            if it.kind != "dead_end" or it.status == "refuted" or not it.approach_tags:
                continue
            overlap = len(query & set(it.approach_tags)) / len(it.approach_tags)
            if overlap >= threshold:
                hits.append(it)
        return sorted(hits, key=lambda i: (-i.rank, _id_key(i.id)))

    def ordered(self) -> list[MemoryItem]:
        """Goals first (by id), then everything else by strength x status weight desc (ties: id)."""
        goals = sorted((i for i in self.items.values() if i.kind == "goal"), key=lambda i: _id_key(i.id))
        rest = sorted((i for i in self.items.values() if i.kind != "goal"), key=lambda i: (-i.rank, _id_key(i.id)))
        return goals + rest

    @staticmethod
    def render_item(it: MemoryItem) -> str:
        label = "DEAD END" if it.kind == "dead_end" else it.kind.upper()
        tags = f" [tags: {', '.join(it.approach_tags)}]" if it.approach_tags else ""
        if it.kind == "goal":
            return f"- [{it.id}] GOAL: {it.text}"
        return f"- [{it.id}] {label} ({it.status}, {it.strength:.2f}): {it.text}{tags}"

    def render(self, token_cap: int | None, extra_lines: Iterable[str] = (), *,
               min_strength: float = 0.0) -> str:
        """Goals, then `extra_lines` (e.g. schemas), then items with strength >= min_strength
        by rank. Never exceeds token_cap."""
        ordered = self.ordered()
        goals = [self.render_item(i) for i in ordered if i.kind == "goal"]
        rest = [self.render_item(i) for i in ordered
                if i.kind != "goal" and i.strength >= min_strength]
        return fit_lines([*goals, *extra_lines, *rest], token_cap)

    def to_json_list(self, max_evidence: int = 5) -> list[dict]:
        return [
            {"id": i.id, "kind": i.kind, "text": i.text, "strength": round(i.strength, 3), "status": i.status,
             "evidence": list(i.evidence)[-max_evidence:], "approach_tags": i.approach_tags}
            for i in self.ordered()
        ]


def _clip(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _dedupe(xs: Iterable) -> list:
    out = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def _id_key(item_id: str) -> tuple:
    m = re.match(r"([a-z]+)(\d+)$", item_id)
    return (m.group(1), int(m.group(2))) if m else (item_id, 0)
