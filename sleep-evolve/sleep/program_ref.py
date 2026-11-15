"""Diff-first programs written during sleep: replay pairs and dream tests (harness/edits.py diffs).

Every program the consolidator or the dreamer asks the sandbox to run is a PAIR (control, treatment)
that must differ only by the claimed change. Writing both in full invites accidental differences, so
the preferred shape references an existing program and edits it:

    "control":   {"base": "<program id>"}     a REFERENCE to a logged program (e.g. "p12", a step id
                                              like 12 / "12" / "step 12", or "best" = the run best);
                                              resolved EXACTLY from the history records' `program`
    "treatment": {"diff": "<<<<<<< SEARCH ... >>>>>>> REPLACE"}
                                              SEARCH/REPLACE blocks applied to the resolved control
                                              (harness/edits.py rules: exact match first, then
                                              trailing-whitespace only; each block must match once)

Fallbacks, only when the preferred shape cannot express the test:
    "control":   {"program": "<full python>", "reason": "..."}  or a bare string (older outputs)
    "treatment": {"program": "<full python>", "reason": "..."}  or a bare string (older outputs)
A full treatment is still accepted with a ref control (and vice versa). Bare strings are the original
shape and keep parsing unchanged (kind "full", reason null).

Resolution failures are never raised: resolve_pair returns error text, and the caller counts the
replay / dream test as wasted / uncertain with that reason:
    "control base 'p99' not found in the run history"
    "treatment diff apply failed: SEARCH block 1 not found in parent"
    "unparseable control program: unknown shape" (likewise for the treatment)

Recording (pair_record): the evaluated programs are recorded reconstructably and compactly:
    control_kind / treatment_kind   "ref" | "full" | "diff"
    control_base                    the program id the ref resolved to (null for a full control)
    treatment_base                  the control's id for a diff ("control" if the control was full)
    control_sha / treatment_sha     sha1[:12] of the exact evaluated text
    treatment_diff                  the diff text (capped at diff_chars; a capped diff is flagged)
    control_program / treatment_program
                                    FULL text (capped at full_chars) only for kind "full"; for ref /
                                    diff the text is reconstructable from the history (`program` of
                                    control_base) plus treatment_diff, so it is not duplicated
    diff_blocks / diff_lines_changed / diff_fuzzy_blocks, apply_error, control_reason, full_reason
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from harness.edits import DiffError, apply_diff

BEST_ALIASES = ("best", "run_best", "best_program")


def sha(text: str | None) -> str | None:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12] if isinstance(text, str) else None


def program_index(records: list[dict]) -> tuple[dict[str, str], dict[int, str], str | None]:
    """(program_id -> program, step -> program_id, run-best program id) from history records.

    Only records with a non-empty `program` string count (steps, extra evals); the run best is the
    highest finite valid score."""
    by_id: dict[str, str] = {}
    by_step: dict[int, str] = {}
    best_id, best = None, None
    for h in records:
        pid, prog = h.get("program_id"), h.get("program")
        if not pid or not isinstance(prog, str) or not prog.strip():
            continue
        by_id[str(pid)] = prog
        if h.get("event", "step") == "step" and isinstance(h.get("step"), int):
            by_step[h["step"]] = str(pid)
        sc = h.get("score")
        if h.get("valid") and isinstance(sc, (int, float)) and sc == sc and abs(sc) != float("inf"):
            if best is None or sc > best:
                best, best_id = sc, str(pid)
    return by_id, by_step, best_id


@dataclass
class ProgramIndex:
    programs: dict[str, str] = field(default_factory=dict)
    steps: dict[int, str] = field(default_factory=dict)
    best_id: str | None = None

    @classmethod
    def from_records(cls, records: list[dict]) -> "ProgramIndex":
        return cls(*program_index(records))

    def resolve_id(self, ref) -> str | None:
        """Program id for a reference: a program id, a step id (int / "12" / "step 12"), or "best"."""
        if isinstance(ref, bool) or ref is None:
            return None
        if isinstance(ref, (int, float)) and float(ref).is_integer():
            return self.steps.get(int(ref))
        s = str(ref).strip()
        if s.lower() in BEST_ALIASES:
            return self.best_id
        if s in self.programs:
            return s
        m = re.fullmatch(r"(?:step\s*)?(\d+)", s, re.I)
        if m:
            return self.steps.get(int(m.group(1)))
        return None


@dataclass
class PairResolution:
    control: str | None = None
    treatment: str | None = None
    error: str | None = None           # first resolution failure (None = both programs resolved)
    info: dict = field(default_factory=dict)  # pair_record fields


def _full_text(x) -> str | None:
    if isinstance(x, str):
        return x
    if isinstance(x, dict) and isinstance(x.get("program"), str):
        return x["program"]
    if isinstance(x, dict) and isinstance(x.get("code"), str):
        return x["code"]
    return None


def _reason(x) -> str | None:
    r = x.get("reason") if isinstance(x, dict) else None
    return " ".join(r.split())[:300] if isinstance(r, str) and r.strip() else None


def resolve_pair(control, treatment, index: ProgramIndex | None, *, full_chars: int | None = None,
                 diff_chars: int = 8000) -> PairResolution:
    """Resolve a (control, treatment) spec to the two exact programs (shapes in the module docstring).

    `full_chars` caps FULL programs only (the old dreamer behavior); refs and diffs are never cut."""
    index = index or ProgramIndex()
    out = PairResolution()
    info = out.info
    info.update(control_kind=None, control_base=None, control_reason=None, treatment_kind=None,
                treatment_base=None, full_reason=None, diff_blocks=None, diff_lines_changed=None,
                diff_fuzzy_blocks=None, apply_error=None, treatment_diff=None, treatment_diff_capped=False)
    # ---- control
    if isinstance(control, dict) and ("base" in control or "ref" in control):
        ref = control.get("base", control.get("ref"))
        info["control_kind"] = "ref"
        pid = index.resolve_id(ref)
        if pid is None:
            info["control_base"] = None
            out.error = f"control base {str(ref)[:40]!r} not found in the run history"
            info["apply_error"] = out.error
            return out
        info["control_base"] = pid
        out.control = index.programs[pid]
    else:
        text = _full_text(control)
        if text is None:
            out.error = "unparseable control program: unknown shape" if control is not None else "unparseable control program: none given"
            return out
        info["control_kind"], info["control_reason"] = "full", _reason(control)
        out.control = text[:full_chars] if full_chars else text
    # ---- treatment
    if isinstance(treatment, dict) and isinstance(treatment.get("diff"), str):
        d = treatment["diff"]
        info["treatment_kind"] = "diff"
        info["treatment_base"] = info["control_base"] or "control"
        info["treatment_diff"] = d[:diff_chars]
        info["treatment_diff_capped"] = len(d) > diff_chars
        try:
            prog, st = apply_diff(out.control, d)
        except DiffError as e:
            out.error = f"treatment {e}"
            info["apply_error"] = out.error
            return out
        info.update(diff_blocks=st["blocks"], diff_lines_changed=st["lines_changed"],
                    diff_fuzzy_blocks=st["fuzzy_blocks"])
        out.treatment = prog
    else:
        text = _full_text(treatment)
        if text is None:
            out.error = "unparseable treatment program: unknown shape" if treatment is not None else "unparseable treatment program: none given"
            return out
        info["treatment_kind"], info["full_reason"] = "full", _reason(treatment)
        out.treatment = text[:full_chars] if full_chars else text
    return out


def pair_record(res: PairResolution, *, full_chars: int = 4000) -> dict:
    """Compact, reconstructable record of a resolved pair (see the module docstring)."""
    rec = dict(res.info)
    rec["control_sha"], rec["treatment_sha"] = sha(res.control), sha(res.treatment)
    if rec.get("control_kind") == "full" and res.control is not None:
        rec["control_program"] = res.control[:full_chars]
    if rec.get("treatment_kind") == "full" and res.treatment is not None:
        rec["treatment_program"] = res.treatment[:full_chars]
    if not rec.get("treatment_diff"):
        rec.pop("treatment_diff", None)
        rec.pop("treatment_diff_capped", None)
    return rec


def kind_counts(records: list[dict], prefix: str) -> dict[str, int]:
    """Per-sleep counts of sleep edit kinds over replay / dream records (keys = metrics SLEEP_EDIT_KEYS)."""
    def n(pred) -> int:
        return sum(1 for r in records if pred(r))
    return {
        f"{prefix}_control_ref": n(lambda r: r.get("control_kind") == "ref"),
        f"{prefix}_control_full": n(lambda r: r.get("control_kind") == "full"),
        f"{prefix}_treatment_diff": n(lambda r: r.get("treatment_kind") == "diff"),
        f"{prefix}_treatment_full": n(lambda r: r.get("treatment_kind") == "full"),
        f"{prefix}_diff_failures": n(lambda r: str(r.get("apply_error") or "").startswith("treatment diff apply")),
        f"{prefix}_base_missing": n(lambda r: str(r.get("apply_error") or "").startswith("control base")),
    }


def base_programs(records: list[dict], *, prefer: list[dict] | None = None, k: int = 4,
                  max_chars: int = 12000) -> list[dict]:
    """The programs shown IN FULL (exact text) to a sleep LLM call so it can diff against them.

    Order: the run best, then `prefer` (e.g. the window's notable steps), then the latest valid steps;
    valid programs only, deduplicated by text, at most `k`, and a program that would push the total
    over `max_chars` is skipped (never truncated: a truncated base cannot be diffed). Everything else
    is referenced by id only in the step views. Returns [{"program_id", "step", "score", "program"}]."""
    valid = [h for h in records if h.get("valid") and isinstance(h.get("program"), str) and h["program"].strip()
             and h.get("program_id") and isinstance(h.get("score"), (int, float))]
    best = max(valid, key=lambda h: h["score"], default=None)
    order = ([best] if best else []) + [h for h in (prefer or []) if h in valid] + valid[::-1]
    out: list[dict] = []
    seen: set = set()
    total = 0
    for h in order:
        if len(out) >= k:
            break
        prog = h["program"]
        if prog in seen or total + len(prog) > max_chars:
            continue
        seen.add(prog)
        total += len(prog)
        out.append({"program_id": h["program_id"], "step": h.get("step"),
                    "score": round(h["score"], 6) if isinstance(h["score"], float) else h["score"],
                    "program": prog})
    return out


def render_bases(bases: list[dict]) -> str:
    """The <programs> block body: each base with its id, exact text in a fenced block."""
    if not bases:
        return "(none shown: write full programs)"
    return "\n\n".join(f"[{b['program_id']}] step {b.get('step')} score {b.get('score')}\n```python\n"
                       f"{b['program']}```" for b in bases)


__all__ = ["BEST_ALIASES", "PairResolution", "ProgramIndex", "base_programs", "kind_counts", "pair_record",
           "program_index", "render_bases", "resolve_pair", "sha"]
