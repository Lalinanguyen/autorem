"""Coupled paired replay: every replay is an A/B test that decides exactly one memory Edit, or is wasted.

A replay is TWO programs written by the consolidator: a `control` and a `treatment` that differs
from it only by the claimed change. Both run ONLY through `evaluator.evaluate` (subprocess
sandbox / Modal). Their results are evidence: never added to ProgramDB, never logged as "step"
events, never shown in the agent's recent-steps window. They are recorded in the sleep event's
details (control/treatment scores, their difference, evaluator calls used).

Why paired: a minimal probe program usually scores far below the run's best, so comparing it to
the best (the old rule) refuted true rules and confirmed almost every dead end. Only the
treatment - control difference is judged.

Outcome rule (deterministic; the LLM's free-text fields are recorded, not trusted):
  scale = max(|control score|, 1e-12);  diff = treatment score - control score
  control crashed (timeout, non-zero exit, no/non-JSON result, evaluator raised) -> no decision
  control invalid but scored (constraint violation)  -> no decision: there is no baseline
  treatment crashed                                   -> no decision
  treatment invalid but scored, valid control         -> dead_end confirmed, rule/fact/skill refuted
  both valid, rule/fact/skill  -> confirmed iff diff >  max(margin, rel_tol) * scale, else refuted
  both valid, dead_end         -> confirmed iff diff <= rel_tol * scale (the change doesn't help),
                                  else refuted (the treatment beat the control)
`margin` (FullSleep `replay_margin`, default DEFAULT_MARGIN) is RELATIVE to |control score| so it
means the same on every problem's scale; `rel_tol` only absorbs float noise.
"No decision" replays are wasted (counted in replays_wasted).

Evaluation order: control first; if the control is unusable (crash / invalid / no score) the
treatment is NOT run (no baseline, so it could not decide anything). A replay therefore costs
0, 1 or 2 evaluator calls; `evaluations` in each record and ReplayReport.evaluations count them.

Diff-first shapes (sleep/program_ref.py): the preferred replay is a REFERENCE control
({"base": "<program id>"}, resolved exactly from the history) plus a SEARCH/REPLACE treatment
({"diff": ...}, harness/edits.py) applied to it, which enforces "differs only by the claimed change"
structurally. Full programs ({"program": ..., "reason": ...} or the original bare strings) are the
fallback and keep working. An unresolvable base or a diff that does not apply wastes the replay with
the reason (no evaluator call); the per-replay record carries the edit kinds, base id, diff stats,
apply error, full-rewrite reasons and the sha of each evaluated program (program_ref.pair_record).
The paired decision rule below is unchanged.

Code-level checks of "differs only by the claimed change" (the prompt defines the rest): both
programs must be strings defining `def construct(`; control and treatment with the same AST
(only comments/whitespace differ) are wasted without running. The line diff is recorded
(`program_diff`) so reviewers can audit how small the change was.

Edit chosen from (relation, decision); target = the first `relates_to` id (or an id named in
`decides`):
  new          confirmed -> add claim as verified        refuted -> add claim as refuted
  supports     confirmed -> support target (verified)    refuted -> contradict target
  contradicts  confirmed -> refute target                refuted -> support target
supports/contradicts with no target id behave like `new`.

Wasted (counted in replays_wasted): no matching claim, the old single-program shape (no control),
missing/unparseable/identical programs, no decision (see above), or an edit the store rejects.

`beats_run_best` flags a replay whose control or treatment scored above the run's best agent
score: a minimal probe beating the search is surprising and worth checking for confounds.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

from sleep.consolidator_io import diff_summary
from harness.evals import EvalExhausted
from sleep.memory import Edit, MemoryStore
from sleep.program_ref import ProgramIndex, pair_record, resolve_pair

DEFAULT_MARGIN = 1e-3  # relative to |control score|; a guess, see module docstring
CRASH_MARKERS = ("timeout", "exit code", "program produced no result", "result is not valid json",
                 "evaluator raised", "does not define construct")
_CONSTRUCT = re.compile(r"\bdef\s+construct\s*\(")


def is_crash(error: str | None) -> bool:
    e = (error or "").lower()
    return any(e.startswith(m) or m in e[:60] for m in CRASH_MARKERS)


def _usable(o: dict | None) -> bool:
    return bool(o) and bool(o.get("valid")) and o.get("score") is not None and math.isfinite(o["score"])


def decide(kind: str, control: dict | None, treatment: dict | None, *, margin: float = DEFAULT_MARGIN,
           rel_tol: float = 1e-6) -> tuple[str | None, str]:
    """Paired decision. control/treatment: {"valid", "score", "error"} (None = not run).

    Returns (decision, reason) with decision "confirmed" | "refuted" | None (no decision).
    """
    if control is None:
        return None, "control not run"
    if not _usable(control):
        if not control.get("valid") and is_crash(control.get("error")):
            return None, "control crashed"
        if not control.get("valid"):
            return None, "control invalid (no baseline)"
        return None, "control has no finite score"
    if treatment is None:
        return None, "treatment not run"
    if not treatment.get("valid"):
        if is_crash(treatment.get("error")):
            return None, "treatment crashed"
        return ("confirmed" if kind == "dead_end" else "refuted"), "treatment invalid, control valid"
    if not _usable(treatment):
        return None, "treatment has no finite score"
    c, t = float(control["score"]), float(treatment["score"])
    diff, scale = t - c, max(abs(c), 1e-12)
    if kind == "dead_end":
        if diff > rel_tol * scale:
            return "refuted", "treatment beat control"
        return "confirmed", "treatment did not beat control"
    if diff > max(margin, rel_tol) * scale:
        return "confirmed", "treatment beat control by more than the margin"
    return "refuted", "treatment did not beat control by the margin"


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


def is_program(p) -> bool:
    return isinstance(p, str) and bool(_CONSTRUCT.search(p))


def same_program(a: str, b: str) -> bool:
    """True if a and b have the same AST (only comments/whitespace differ); text compare if unparseable."""
    try:
        return ast.dump(ast.parse(a)) == ast.dump(ast.parse(b))
    except (SyntaxError, ValueError):
        return a.strip() == b.strip()


@dataclass
class ReplayReport:
    edits_applied: list[Edit] = field(default_factory=list)
    decided_cids: set = field(default_factory=set)
    wasted: int = 0
    evaluations: int = 0  # evaluator calls made by replays (control + treatment)
    beats_run_best: int = 0
    records: list[dict] = field(default_factory=list)


def _evaluate(evaluator, program: str) -> dict:
    try:
        res = evaluator.evaluate(program)
        valid, score, error = bool(res.valid), res.score, res.error
    except EvalExhausted:
        raise  # the shared eval cap stops the run; it is not a replay failure
    except Exception as e:  # evaluators should not raise; a replay must never crash a sleep
        valid, score, error = False, None, f"evaluator raised {type(e).__name__}: {e}"
    score = score if (score is not None and math.isfinite(score)) else None
    return {"valid": valid, "score": score, "error": (error or None) and str(error)[:300]}


def run_replays(replays: list[dict], claims: dict[str, dict], store: MemoryStore, evaluator, *,
                step: int, max_replays: int, margin: float = DEFAULT_MARGIN, rel_tol: float = 1e-6,
                run_best: float | None = None, program_chars: int = 4000,
                index: ProgramIndex | None = None) -> ReplayReport:
    """Run up to `max_replays` paired replays; each applies exactly one Edit to `store` or counts as wasted.

    `index` (sleep/program_ref.py ProgramIndex over the run history) resolves ref controls
    ({"base": id}); diff treatments ({"diff": ...}) are applied to the resolved control. A ref / diff
    that does not resolve wastes the replay with the reason (no evaluator call)."""
    rep = ReplayReport()
    for i, r in enumerate(replays[: max(0, max_replays)]):
        rid = f"r{step}.{i}"
        rec: dict = {"rid": rid, "cid": r.get("cid"), "decides": str(r.get("decides", ""))[:200],
                     "change": str(r.get("change", ""))[:200], "evaluations": 0}
        rep.records.append(rec)
        claim = claims.get(r.get("cid"))
        control, treatment = r.get("control"), r.get("treatment")
        res = None
        if claim is None:
            rec["wasted"] = "no matching claim"
        elif control is None and treatment is None and "program" in r:
            rec["wasted"] = "single-program replay (old schema): no control to compare against"
        else:
            res = resolve_pair(control, treatment, index)
            rec.update(pair_record(res, full_chars=program_chars))
            control, treatment = res.control, res.treatment
            if res.error:
                rec["wasted"] = res.error
            elif not is_program(control):
                rec["wasted"] = "unparseable control program"
            elif not is_program(treatment):
                rec["wasted"] = "unparseable treatment program"
            elif same_program(control, treatment):
                rec["wasted"] = "control and treatment are identical"
            elif evaluator is None:
                rec["wasted"] = "no evaluator bound"
        if "wasted" in rec:
            rep.wasted += 1
            continue
        rec["program_diff"] = diff_summary(control, treatment)
        c_out = _evaluate(evaluator, control)
        t_out = _evaluate(evaluator, treatment) if _usable(c_out) else None
        rec["evaluations"] = 1 + (t_out is not None)
        rep.evaluations += rec["evaluations"]
        c_score = c_out["score"] if c_out["valid"] else None
        t_score = t_out["score"] if (t_out and t_out["valid"]) else None
        rec.update(control=c_out, treatment=t_out, control_score=c_score, treatment_score=t_score,
                   score_diff=(t_score - c_score) if (c_score is not None and t_score is not None) else None,
                   margin=margin)
        beats = run_best is not None and any(s is not None and s > run_best for s in (c_score, t_score))
        rec["beats_run_best"] = beats
        rep.beats_run_best += beats
        decision, why = decide(claim["kind"], c_out, t_out, margin=margin, rel_tol=rel_tol)
        rec["decision"], rec["decision_reason"] = decision, why
        if decision is None:
            rec["wasted"] = f"no decision: {why}"
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
