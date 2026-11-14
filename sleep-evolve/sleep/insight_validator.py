"""Dream validation: run each dream's paired A/B test through the replay machinery.

Dreams are hypotheses until the sandbox says otherwise. Each dream may carry a paired
control/treatment test (same contract as sleep/replay.py): the control runs first and the
treatment only when the control is usable, so a dream costs 0, 1 or 2 evaluator calls.
The decision reuses replay.decide; dream kinds are never "dead_end", so the ordinary branch
applies: validated iff treatment beats control by more than margin * |control score|.

This module reuses replay's private `_evaluate`/`_usable` helpers on purpose: it lives in the
same package and needs byte-identical evaluator-call semantics (EvalExhausted propagation,
crash classification, score sanitation) so a dream test is exactly a replay.

Outcome -> memory (one Edit per dream, source="dream"):
  validated  -> add the dream text as a "skill" with status verified, tagged by the union of
                its source items' approach_tags, evidence = [rid, *sources]
  refuted    -> refute that same item (adds a tested-and-false claim the agent won't retry)
  uncertain  -> add as tentative (no decision: crash / unusable control / no score)
  untested   -> add as tentative, only when the store does not already hold that text
Every record keeps did/sources/rationale plus the scores and decision reason — the full
provenance trail. Store mutation happens only through MemoryStore.apply_edits.
"""

from __future__ import annotations

from sleep.memory import Edit, MemoryStore, normalize_tags  # MemoryStore used for typing only
from sleep.replay import _evaluate, _usable, decide, is_program, same_program


def validate_dreams(dreams: list[dict], store: MemoryStore, evaluator, *,
                    step: int, max_tests: int, margin: float = 1e-3,
                    rel_tol: float = 1e-6) -> dict:
    """Test up to `max_tests` dreams' A/B tests and apply the resulting edits to `store`.

    Returns {"records", "validated", "refuted", "uncertain", "untested", "evaluations",
    "edits_applied", "edits"}. evaluator=None skips all testing (every dream is untested,
    zero evaluations).
    """
    records: list[dict] = []
    edits: list[Edit] = []
    counts = {"validated": 0, "refuted": 0, "uncertain": 0, "untested": 0}
    evaluations = 0
    n_tests = 0
    for i, raw in enumerate(dreams):
        d = raw if isinstance(raw, dict) else {}
        rid = f"d{step}.{i}"
        rec: dict = {
            "did": d.get("did"), "kind": d.get("kind"), "text": d.get("text"),
            "sources": list(d.get("sources") or []), "rationale": d.get("rationale") or "",
            "status": "untested", "replay_rid": None, "evaluations": 0,
            "control_score": None, "treatment_score": None, "score_diff": None,
            "decision_reason": None,
        }
        records.append(rec)
        test = d.get("test")
        control = test.get("control") if isinstance(test, dict) else None
        treatment = test.get("treatment") if isinstance(test, dict) else None
        if evaluator is None:
            rec["decision_reason"] = "no evaluator bound"
        elif test is None:
            rec["decision_reason"] = "no test attached"
        elif not is_program(control):
            rec["decision_reason"] = "unparseable control program"
        elif not is_program(treatment):
            rec["decision_reason"] = "unparseable treatment program"
        elif same_program(control, treatment):
            rec["decision_reason"] = "control and treatment are identical"
        elif n_tests >= max_tests:
            rec["decision_reason"] = "dream test cap reached"
        else:
            n_tests += 1
            rec["replay_rid"] = rid
            c_out = _evaluate(evaluator, control)
            t_out = _evaluate(evaluator, treatment) if _usable(c_out) else None
            rec["evaluations"] = 1 + (t_out is not None)
            evaluations += rec["evaluations"]
            c_score = c_out["score"] if c_out["valid"] else None
            t_score = t_out["score"] if (t_out and t_out["valid"]) else None
            rec.update(control_score=c_score, treatment_score=t_score,
                       score_diff=(t_score - c_score)
                       if (c_score is not None and t_score is not None) else None)
            decision, why = decide(d.get("kind") or "association", c_out, t_out,
                                   margin=margin, rel_tol=rel_tol)
            rec["decision_reason"] = why
            rec["status"] = {"confirmed": "validated", "refuted": "refuted"}.get(decision,
                                                                               "uncertain")
        counts[rec["status"]] += 1
        edit = _dream_edit(rec, store, rid)
        if edit is not None:
            rec["edit"] = edit.to_dict()
            applied, rejected = store.apply_edits([edit], step)
            if rejected:
                rec["edit_rejected"] = rejected[0][1]
            edits.extend(applied)
    return {
        "records": records,
        "validated": counts["validated"], "refuted": counts["refuted"],
        "uncertain": counts["uncertain"], "untested": counts["untested"],
        "evaluations": evaluations, "edits_applied": len(edits),
        "edits": [e.to_dict() for e in edits],
    }


def _dream_edit(rec: dict, store: MemoryStore, rid: str) -> Edit | None:
    """One Edit from the dream's outcome; None when nothing should be written."""
    tags: list[str] = []
    for s in rec["sources"]:
        it = store.get(s)
        if it is not None:
            tags.extend(it.approach_tags)
    item = {"kind": "skill", "text": rec["text"], "approach_tags": normalize_tags(tags)}
    evidence = [rid, *rec["sources"]]
    status = rec["status"]
    if status == "validated":
        return Edit("add", item=item, status="verified", evidence=evidence, source="dream")
    if status == "refuted":
        return Edit("refute", item=item, evidence=evidence, source="dream")
    if rec["text"] and store.find_text(rec["text"]) is None:
        return Edit("add", item=item, status="tentative", evidence=evidence, source="dream")
    return None
