"""Structured long-term memory for FullSleep: MemoryItem, Edit, MemoryStore.

Everything here is deterministic (no randomness, no wall clock): the same sequence of
edits/decays/evictions always yields the same store, ids and rendering.

Kinds:   goal | rule | fact | skill | dead_end
Status:  tentative | verified | contested | refuted
Goals are never decayed or evicted and always render first. g0 is the code-maintained best-score
line; consolidator goals get ids g1, g2, ... (set_goal: at most `max_goals` of them; when full, the
oldest goal not touched this sleep is replaced). remove never removes a goal; merge never mixes a
goal with a non-goal and never touches g0.

Status ranking (merge keeps the highest): verified > tentative > contested > refuted (STATUS_WEIGHT).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from harness.context import fit_lines

KINDS = ("goal", "rule", "fact", "skill", "dead_end")
STATUSES = ("tentative", "verified", "contested", "refuted")
STATUS_WEIGHT = {"verified": 1.0, "tentative": 0.6, "contested": 0.35, "refuted": 0.2}
EDIT_OPS = ("add", "support", "contradict", "refute", "merge", "remove")
PROTECTED_GOALS = ("g0",)  # maintained by code; never replaced or merged by the consolidator
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
    """One memory edit. op: add | support | contradict | refute | merge | remove.

    add:        `item` holds the new item's fields (kind, text, status, strength, ...).
    support:    strengthen `target`; status may be promoted (e.g. to verified).
    contradict: weaken `target`; a verified item becomes contested.
    refute:     mark `target` refuted. With no target and an `item`, the claim is added
                with status refuted (a tested-and-false claim the agent should not retry).
    merge:      combine `target` with the ids in item["merge"] into `target` (optional new
                item["text"]); see MemoryStore.merge. The other items are removed.
    remove:     delete `target`. Never removes a goal.
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
        self._next_goal = 1

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

    def remove(self, item_id: str) -> str | None:
        """Delete a non-goal item. Returns None on success, else the rejection reason."""
        it = self.items.get(item_id)
        if it is None:
            return f"unknown target {item_id!r}"
        if it.kind == "goal":
            return f"cannot remove goal {item_id}"
        del self.items[item_id]
        return None

    def merge(self, target: str, others: Iterable[str] | str, step: int, text: str | None = None) -> str | None:
        """Merge `others` into `target`. Returns None on success, else the rejection reason.

        The kept item (id and kind of `target`) gets: `text` if given (else target's text); the
        union of evidence (order kept, capped at MAX_EVIDENCE, newest kept) and of tags (capped
        at MAX_TAGS); the highest status by STATUS_WEIGHT; the max strength; the min created_step
        and the max last_supported_step. The other items are deleted.
        Rejected: fewer than two distinct ids, unknown ids, a goal mixed with a non-goal, g0, a
        dead_end mixed with a non-dead_end (opposite polarity), or merged text equal to the text
        of an item outside the merge.
        """
        if isinstance(others, str):
            others = [others]
        ids = _dedupe([str(target), *[str(o) for o in (others or [])]])
        if len(ids) < 2:
            return "merge needs at least two distinct items"
        missing = [i for i in ids if i not in self.items]
        if missing:
            return f"unknown merge ids {missing}"
        its = [self.items[i] for i in ids]
        is_goal = [it.kind == "goal" for it in its]
        if any(is_goal) and not all(is_goal):
            return "cannot merge a goal with a non-goal"
        if any(i in PROTECTED_GOALS for i in ids):
            return "cannot merge the best-score goal g0"
        is_dead = [it.kind == "dead_end" for it in its]
        if any(is_dead) and not all(is_dead):
            return "cannot merge a dead_end with a non-dead_end"
        keep = its[0]
        new_text = " ".join(str(text).split())[: self.max_item_chars] if text and str(text).strip() else keep.text
        dup = self.find_text(new_text)
        if dup is not None and dup.id not in ids:
            return f"merged text duplicates {dup.id}"
        keep.text = new_text
        keep.status = max((it.status for it in its), key=lambda st: STATUS_WEIGHT.get(st, 0.0))
        keep.strength = _clip(max(it.strength for it in its))
        keep.evidence = _dedupe(e for it in its for e in it.evidence)[-MAX_EVIDENCE:]
        keep.approach_tags = normalize_tags([t for it in its for t in it.approach_tags])
        keep.created_step = min(it.created_step for it in its)
        keep.last_supported_step = max(it.last_supported_step for it in its)
        for it in its[1:]:
            del self.items[it.id]
        return None

    # ---- goals ----------------------------------------------------------------------
    def goals(self, *, include_protected: bool = False) -> list[MemoryItem]:
        return sorted((i for i in self.items.values() if i.kind == "goal"
                       and (include_protected or i.id not in PROTECTED_GOALS)), key=lambda i: _id_key(i.id))

    def set_goal(self, text: str, step: int, *, replaces: str | None = None, max_goals: int = 3,
                 keep: Iterable[str] = ()) -> tuple[str | None, str]:
        """Add or replace one consolidator goal. Returns (goal id or None, action).

        action: "refreshed" (same text as an existing goal), "replaced gN" (in place, id kept),
        "added", "added, evicted gN" (cap reached: the oldest goal not in `keep` goes), or
        "rejected: ...". A `replaces` that is unknown, not a goal, or g0 is ignored (noted in the
        action) and the goal is added instead.
        """
        text = " ".join(str(text or "").split())[: self.max_item_chars]
        if not text:
            return None, "rejected: empty text"
        key = _norm_text(text)
        same = next((g for g in self.goals(include_protected=True) if _norm_text(g.text) == key), None)
        if same is not None:
            same.last_supported_step = max(same.last_supported_step, step)
            return same.id, "refreshed"
        note = ""
        if replaces:
            old = self.items.get(str(replaces))
            if old is not None and old.kind == "goal" and old.id not in PROTECTED_GOALS:
                old.text, old.last_supported_step = text, step
                return old.id, f"replaced {old.id}"
            note = f" (ignored replaces={replaces!r})"
        if max_goals <= 0:
            return None, "rejected: max_goals is 0" + note
        keep = set(keep)
        current = self.goals()
        evicted = None
        if len(current) >= max_goals:
            pool = sorted((g for g in current if g.id not in keep),
                          key=lambda g: (g.last_supported_step, g.created_step, _id_key(g.id)))
            if not pool:
                return None, "rejected: goal cap reached" + note
            evicted = pool[0].id
            del self.items[evicted]
        while f"g{self._next_goal}" in self.items:
            self._next_goal += 1
        gid = f"g{self._next_goal}"
        self._next_goal += 1
        self.add(kind="goal", text=text, step=step, status="tentative", strength=1.0, item_id=gid)
        return gid, "added" + (f", evicted {evicted}" if evicted else "") + note

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
        if e.op == "merge":
            others = (e.item or {}).get("merge") or (e.item or {}).get("ids") or []
            return self.merge(str(e.target), others, step, text=(e.item or {}).get("text"))
        if e.op == "remove":
            return self.remove(str(e.target))
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

    # ---- cross-run persistence ------------------------------------------------------
    def save_state(self) -> dict:
        """Full store state as JSON-able data: every item plus the id counters, so a later run
        (or a different problem) resumes with the same ids it would have issued next."""
        return {
            "items": [it.to_dict() for it in self.ordered()],
            "_next": self._next, "_next_goal": self._next_goal,
            "initial_strength": self.initial_strength, "verified_strength": self.verified_strength,
            "support_gain": self.support_gain, "contradict_loss": self.contradict_loss,
            "max_item_chars": self.max_item_chars,
        }

    @classmethod
    def from_state(cls, state: dict) -> "MemoryStore":
        """Rebuild a store from save_state(). Missing files should be handled by the caller (start
        an empty store); a corrupt dict raises."""
        store = cls(
            initial_strength=state.get("initial_strength", 0.3),
            verified_strength=state.get("verified_strength", 0.7),
            support_gain=state.get("support_gain", 0.15),
            contradict_loss=state.get("contradict_loss", 0.25),
            max_item_chars=state.get("max_item_chars", 300),
        )
        for d in state.get("items") or []:
            it = MemoryItem(**{k: d[k] for k in MemoryItem.__dataclass_fields__ if k in d})
            store.items[it.id] = it
        if "_next" in state:
            store._next = int(state["_next"])
        else:  # defensive: derive from the highest m-id + 1
            store._next = 1 + max((_id_key(i)[1] for i in store.items if i.startswith("m")), default=0)
        if "_next_goal" in state:
            store._next_goal = int(state["_next_goal"])
        return store


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
