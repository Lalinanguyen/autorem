"""Coupled replay: every replay program decides exactly one memory Edit, or is counted as wasted.

Replay programs run ONLY through `evaluator.evaluate` (subprocess sandbox / Modal). Their
results are evidence: never added to ProgramDB, never logged as "step" events, never shown
in the agent's recent-steps window. They are recorded in the sleep event's details.

Outcome rule (deterministic; the LLM's free-text confirms_if/refutes_if are recorded, not trusted):
  ref  = best valid score among the claim's evidence steps; if none, the run's best score
  tol  = rel_tol * max(|ref|, 1e-12)
  crash (timeout, non-zero exit, no/non-JSON result, evaluator raised)  -> no decision (wasted)
  invalid but scored (constraint violation, bad shape)  -> dead_end claim confirmed, other claims refuted
  valid, dead_end claim     -> refuted iff score > run_best + tol, else confirmed
  valid, rule/fact/skill    -> confirmed iff score >= ref - tol, else refuted

Edit chosen from (relation, decision); target = the first `relates_to` id (or an id named in
`decides`):
  new          confirmed -> add claim as verified        refuted -> add claim as refuted
  supports     confirmed -> support target (verified)    refuted -> contradict target
  contradicts  confirmed -> refute target                refuted -> support target
supports/contradicts with no target id behave like `new`.

Wasted (counted in replays_wasted): no matching claim, missing/unparseable program, crash,
or an edit the store rejects (e.g. unknown target).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from sleep.memory import Edit, MemoryStore

CRASH_MARKERS = ("timeout", "exit code", "program produced no result", "result is not valid json",
                 "evaluator raised", "does not define construct")


def is_crash(error: str | None) -> bool:
    e = (error or "").lower()
    return any(e.startswith(m) or m in e[:60] for m in CRASH_MARKERS)


def decide(claim: dict, valid: bool, score: float | None, error: str | None,
           ref: float | None, run_best: float | None, rel_tol: float = 1e-6) -> str | None:
    if not valid:
        if is_crash(error):
            return None
        return "confirmed" if claim["kind"] == "dead_end" else "refuted"
    if score is None or not math.isfinite(score):
        return None
    if claim["kind"] == "dead_end":
        if run_best is None:
            return "confirmed"
        return "refuted" if score > run_best + rel_tol * max(abs(run_best), 1e-12) else "confirmed"
    if ref is None:
        return "confirmed"
    return "confirmed" if score >= ref - rel_tol * max(abs(ref), 1e-12) else "refuted"


def resolve_target(claim: dict, decides: str, store: MemoryStore) -> str | None:
    ids = list(claim.get("relates_to") or [])
    for i in ids:
        if i in store.items:
            return i
    if ids:
        return ids[0]  # unknown id -> the edit will be rejected (wasted)
    m = re.search(r"\b([mg]\d+)\b", decides or "")
    return m.group(1) if m else None


def edit_for(claim: dict, decision: str, target: str | None, evidence: list) -> Edit:
    relation = claim.get("relation", "new")
    item = {"kind": claim["kind"], "text": claim["text"], "approach_tags": claim.get("approach_tags", [])}
    if relation == "new" or target is None:
        if decision == "confirmed":
            return Edit("add", item=item, status="verified", evidence=evidence, source="replay")
        return Edit("refute", item=item, evidence=evidence, source="replay")
    if relation == "supports":
        if decision == "confirmed":
            return Edit("support", target=target, status="verified", evidence=evidence, source="replay")
        return Edit("contradict", target=target, evidence=evidence, source="replay")
    # contradicts
    if decision == "confirmed":
        return Edit("refute", target=target, evidence=evidence, source="replay")
    return Edit("support", target=target, evidence=evidence, source="replay")


@dataclass
class ReplayReport:
    edits_applied: list[Edit] = field(default_factory=list)
    decided_cids: set = field(default_factory=set)
    wasted: int = 0
    records: list[dict] = field(default_factory=list)


def run_replays(replays: list[dict], claims: dict[str, dict], store: MemoryStore, evaluator, *,
                step: int, max_replays: int, step_scores: dict, run_best: float | None,
                rel_tol: float = 1e-6, program_chars: int = 400) -> ReplayReport:
    """Run up to `max_replays` replays; each applies exactly one Edit to `store` or counts as wasted."""
    rep = ReplayReport()
    for i, r in enumerate(replays[: max(0, max_replays)]):
        rid = f"r{step}.{i}"
        rec: dict = {"rid": rid, "cid": r.get("cid"), "decides": str(r.get("decides", ""))[:200],
                     "confirms_if": str(r.get("confirms_if", ""))[:200],
                     "refutes_if": str(r.get("refutes_if", ""))[:200]}
        rep.records.append(rec)
        claim = claims.get(r.get("cid"))
        program = r.get("program")
        if claim is None:
            rec["wasted"] = "no matching claim"
        elif not isinstance(program, str) or "construct" not in program:
            rec["wasted"] = "unparseable program"
        elif evaluator is None:
            rec["wasted"] = "no evaluator bound"
        if "wasted" in rec:
            rep.wasted += 1
            continue
        rec["program"] = program[:program_chars]
        try:
            res = evaluator.evaluate(program)
            valid, score, error = bool(res.valid), res.score, res.error
        except Exception as e:  # evaluators should not raise; a replay must never crash a sleep
            valid, score, error = False, None, f"evaluator raised {type(e).__name__}: {e}"
        score = score if (score is not None and math.isfinite(score)) else None
        rec.update(valid=valid, score=score, error=(error or None) and str(error)[:300])
        ev_scores = [step_scores[s] for s in claim.get("evidence", []) if step_scores.get(s) is not None]
        ref = max(ev_scores) if ev_scores else run_best
        rec["ref_score"] = ref
        decision = decide(claim, valid, score, error, ref, run_best, rel_tol)
        rec["decision"] = decision
        if decision is None:
            rec["wasted"] = "crashed without decision"
            rep.wasted += 1
            continue
        target = resolve_target(claim, rec["decides"], store)
        edit = edit_for(claim, decision, target, [rid, *claim.get("evidence", [])])
        applied, rejected = store.apply_edits([edit], step)
        rec["edit"] = edit.to_dict()
        if rejected:
            rec["wasted"] = f"edit rejected: {rejected[0][1]}"
            rep.wasted += 1
            continue
        rep.edits_applied.extend(applied)
        rep.decided_cids.add(claim["cid"])
    return rep
