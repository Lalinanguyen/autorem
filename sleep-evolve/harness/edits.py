"""Agent edit formats: full rewrites and SEARCH/REPLACE diffs against the parent program.

Part of the AGENT, not of the sleep policy: every arm run by harness/loop.py uses the same
RunConfig.edit_mode (the baseline drift check enforces it). The GEPA arms (arms/gepa_arm.py) use
gepa's own proposer and never come through here, so edit_mode does not apply to them.

Diff format (the one aider and OpenEvolve use); any number of blocks per reply:

    <<<<<<< SEARCH
    (lines copied exactly from the parent program)
    =======
    (replacement lines; empty = delete)
    >>>>>>> REPLACE

Markers are whole lines: 5 to 9 '<' + "SEARCH", 5 to 9 '=', 5 to 9 '>' + "REPLACE" (7 is canonical),
with optional surrounding whitespace. Lines outside blocks (rationale, header, ``` fences around the
blocks, a stray ======= markdown rule) are ignored. Malformed markers (a REPLACE without an open SEARCH
and separator, a SEARCH inside an open block, a second separator in one block, an unclosed block, or
markers that form no complete block) fail the whole reply: no block of it is applied.

Matching rules (apply_blocks), per block, in reply order, each against the text produced by the
previous blocks:
  1. exact: the SEARCH lines must equal a run of consecutive whole lines of the current text.
     Exactly one occurrence -> replaced. More than one -> ambiguous (error; no fallback).
  2. only when there is no exact occurrence: the same comparison with TRAILING whitespace ignored on
     every line (rstrip; leading indentation and inner whitespace still have to match exactly; also
     absorbs CRLF). Exactly one -> replaced; more than one -> ambiguous; none -> not found.
  The REPLACE lines are inserted verbatim. An empty SEARCH section is an error (no blind inserts).
  A result with no non-whitespace content is an error. Line endings are normalized to "\n" and the
  result always ends with one trailing newline.

Edit kind (decide_edit), by mode:
  full  today's behavior exactly: the last fenced code block (extract_program), SEARCH/REPLACE ignored.
  diff  SEARCH/REPLACE only; a reply with only a fenced program is rejected.
  auto  SEARCH/REPLACE markers present -> diff (even if a fenced block is present too);
        else a fenced block -> full rewrite; else no code.
        When the diff fails and a fenced program is also present, the program is used instead
        (kind "full", diff_error records the failed diff) — a salvage, not an invalid step.
  In auto the PROMPT makes diffs the default (harness/loop.py AGENT_SYSTEM_AUTO): a full rewrite only
  when a diff is not feasible, with a `FULL_REWRITE_REASON: ...` line (parse_full_rewrite_reason,
  bucketed by reason_category). Full rewrites are never rejected in auto, only logged and counted.
  A step whose parent was truncated in the prompt runs with diffs forbidden (see harness/loop.py):
  it behaves like "full": a reply with SEARCH/REPLACE markers gets the diff part rejected
  (diff_error), but a fenced program in the same reply is still used — a full program is
  complete and safe regardless of what the prompt showed.

Error strings (EditOutcome.error, logged as `error` and, for edit-protocol failures, `diff_error`):
  "diff apply failed: SEARCH block 2 not found in parent"
  "diff apply failed: SEARCH block 1 matches 3 places in parent (ambiguous)"
  "diff apply failed: malformed SEARCH/REPLACE markers: <detail>"
  "diff apply failed: SEARCH block 1 is empty"
  "diff apply failed: the result is an empty program"
  "diff rejected: the parent was truncated in the prompt, so this step needs a full rewrite"
  (both also appear as diff_error on a salvaged step: kind "full", code set, diff failed/ignored)
  "full rewrite rejected: edit_mode is diff, reply with SEARCH/REPLACE blocks"
  "no SEARCH/REPLACE blocks in response"                     (diff mode, nothing usable)
  "no ```python code block or SEARCH/REPLACE blocks in response"   (auto mode, nothing usable)
  "no ```python code block in response"                      (full mode, unchanged)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Literal

EditMode = Literal["full", "diff", "auto"]
EDIT_MODES = ("full", "diff", "auto")

_SEARCH = re.compile(r"^\s*<{5,9}\s*SEARCH\s*$")
_SEP = re.compile(r"^\s*={5,9}\s*$")
_REPLACE = re.compile(r"^\s*>{5,9}\s*REPLACE\s*$")
# case-insensitive py-ish tags only: ```Python / ```py3 / ```python3 / ``` all give a program;
# a ```json or ```text block is not extracted (it is not a program and must not cost an evaluation)
_PY_BLOCK = re.compile(r"```(?:python3?|py3?)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK = re.compile(r"```(?:python3?|py3?)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)

DIFF_FAILED = "diff apply failed"
NO_CODE_FULL = "no ```python code block in response"
NO_CODE_AUTO = "no ```python code block or SEARCH/REPLACE blocks in response"
NO_CODE_DIFF = "no SEARCH/REPLACE blocks in response"
FULL_REJECTED = "full rewrite rejected: edit_mode is diff, reply with SEARCH/REPLACE blocks"
DIFF_REJECTED_TRUNCATED = "diff rejected: the parent was truncated in the prompt, so this step needs a full rewrite"


_REASON = re.compile(r"^[ \t>*_#-]*FULL[ _-]?REWRITE[ _-]?REASON[*_ \t]*[:=][ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)

# Full-rewrite reason categories (reason_category): first keyword group that matches wins.
REASON_CATEGORIES = (
    ("parent_truncated", ("truncat", "did not fit", "didn't fit", "cut off", "incomplete parent")),
    ("parent_not_shown", ("not shown", "no parent", "missing parent", "parent is missing", "without a parent")),
    ("previous_diff_failed", ("previous diff", "last diff", "diff failed", "failed to apply", "did not apply",
                              "didn't apply", "not found in parent", "apply fail")),
    ("restructure", ("restructur", "rewrite most", "most of the program", "rewrit", "redesign", "new approach",
                     "different algorithm", "replace the whole", "entire program", "from scratch", "overhaul")),
)


def parse_full_rewrite_reason(text: str) -> str | None:
    """Lenient parse of a `FULL_REWRITE_REASON: ...` line before the first code fence / SEARCH
    marker (any case, `_`/space/`-` separators, markdown decoration). None when absent or empty."""
    head = (text or "").split("```", 1)[0]
    cut = first_marker_offset(head)
    if cut is not None:
        head = head[:cut]
    m = _REASON.search(head)
    if not m:
        return None
    val = m.group(1).strip().strip("*_`").strip()
    return val[:300] or None


def reason_category(reason: str | None) -> str:
    """Bucket a full-rewrite reason: parent_truncated | parent_not_shown | previous_diff_failed |
    restructure | other | none (no reason given)."""
    if not reason:
        return "none"
    low = reason.lower()
    for cat, keys in REASON_CATEGORIES:
        if any(k in low for k in keys):
            return cat
    return "other"


class DiffError(ValueError):
    """A SEARCH/REPLACE reply that cannot be parsed or applied; str() is the logged error."""


@dataclass(frozen=True)
class EditBlock:
    search: list[str]
    replace: list[str]


@dataclass
class EditOutcome:
    code: str | None                 # the FULL resulting program, or None when there is none
    kind: Literal["full", "diff"] | None  # what produced the program; with no program, what the reply attempted
    error: str | None = None         # why there is no program (None when code is set)
    diff_error: str | None = None    # edit-protocol failure (diff apply / rejected format), else None
    stats: dict = field(default_factory=dict)  # diff only: blocks, lines_changed, fuzzy_blocks


def extract_program(text: str) -> str | None:
    """Return the last fenced code block, preferring ```python blocks (the "full" edit kind)."""
    blocks = _PY_BLOCK.findall(text) or _CODE_BLOCK.findall(text)
    return blocks[-1].strip() + "\n" if blocks else None


def _norm(text: str) -> str:
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def has_diff_markers(text: str) -> bool:
    """True if any line is a SEARCH or REPLACE marker (well-formed or not)."""
    return any(_SEARCH.match(ln) or _REPLACE.match(ln) for ln in _norm(text).split("\n"))


def first_marker_offset(text: str) -> int | None:
    """Character offset of the first SEARCH marker line, or None (used to bound the awake header)."""
    pos = 0
    for ln in (text or "").splitlines(keepends=True):
        if _SEARCH.match(ln.rstrip("\r\n")):
            return pos
        pos += len(ln)
    return None


def parse_blocks(text: str) -> list[EditBlock]:
    """Every SEARCH/REPLACE block of `text`, in order. Raises DiffError on malformed markers."""
    blocks: list[EditBlock] = []
    state = "out"  # out | search | replace
    search: list[str] = []
    replace: list[str] = []
    for i, ln in enumerate(_norm(text).split("\n"), 1):
        if _SEARCH.match(ln):
            if state != "out":
                raise DiffError(f"{DIFF_FAILED}: malformed SEARCH/REPLACE markers: SEARCH on line {i} "
                                f"inside block {len(blocks) + 1}")
            state, search, replace = "search", [], []
        elif _SEP.match(ln) and state != "out":
            if state == "replace":
                raise DiffError(f"{DIFF_FAILED}: malformed SEARCH/REPLACE markers: second ======= on line {i} "
                                f"in block {len(blocks) + 1}")
            state = "replace"
        elif _REPLACE.match(ln):
            if state != "replace":
                raise DiffError(f"{DIFF_FAILED}: malformed SEARCH/REPLACE markers: REPLACE on line {i} "
                                f"without SEARCH and =======")
            blocks.append(EditBlock(search, replace))
            state = "out"
        elif state == "search":
            search.append(ln)
        elif state == "replace":
            replace.append(ln)
    if state != "out":
        raise DiffError(f"{DIFF_FAILED}: malformed SEARCH/REPLACE markers: block {len(blocks) + 1} is not closed "
                        f"(missing {'=======' if state == 'search' else '>>>>>>> REPLACE'})")
    if not blocks:
        raise DiffError(f"{DIFF_FAILED}: malformed SEARCH/REPLACE markers: no complete block")
    return blocks


def _occurrences(lines: list[str], needle: list[str], key=lambda s: s) -> list[int]:
    n = len(needle)
    want = [key(x) for x in needle]
    return [i for i in range(len(lines) - n + 1) if [key(x) for x in lines[i:i + n]] == want]


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _reindent(replace: list[str], search: list[str], parent_lines: list[str]) -> list[str]:
    """Shift REPLACE lines to the parent's indentation level (leading-whitespace fallback).

    Uniform shift by (p0 - s0): s0 = the model's indent on its first non-blank SEARCH line,
    p0 = the parent's indent on the corresponding matched line. Every non-blank REPLACE line
    moves by the same amount, so relative structure inside the replacement is preserved;
    blank lines are kept blank."""
    j = next((k for k, s in enumerate(search) if s.strip()), 0)
    s0 = _indent(search[j])
    p0 = _indent(parent_lines[j]) if j < len(parent_lines) else ""
    delta = len(p0) - len(s0)
    out = []
    for r in replace:
        if not r.strip():
            out.append("")
        elif delta >= 0:
            out.append(" " * delta + r)
        else:
            out.append(r[min(-delta, len(r) - len(r.lstrip())):])
    return out


def apply_blocks(parent: str, blocks: list[EditBlock], *, allow_indent_fuzzy: bool = False) -> tuple[str, dict]:
    """Apply `blocks` in order to `parent` (rules in the module docstring) -> (program, stats).

    With allow_indent_fuzzy (sleep-side tests only; agent edits stay strict), a third tier after
    exact and rstrip: whole lines compared with leading AND trailing whitespace ignored; on a
    unique match the REPLACE lines are re-indented to the parent's level (_reindent)."""
    src = _norm(parent)
    trailing_nl = src.endswith("\n")
    lines = src[:-1].split("\n") if trailing_nl else src.split("\n")
    if src == "":
        lines = []
    fuzzy = indent_fuzzy = 0
    for b, blk in enumerate(blocks, 1):
        if not blk.search or all(not s.strip() for s in blk.search):
            raise DiffError(f"{DIFF_FAILED}: SEARCH block {b} is empty")
        hits = _occurrences(lines, blk.search)
        if not hits:
            hits = _occurrences(lines, blk.search, key=str.rstrip)
            if len(hits) == 1:
                fuzzy += 1
        replace = list(blk.replace)
        if not hits and allow_indent_fuzzy:
            hits = _occurrences(lines, blk.search, key=str.strip)
            if len(hits) == 1:
                indent_fuzzy += 1
                replace = _reindent(blk.replace, blk.search, lines[hits[0]:hits[0] + len(blk.search)])
        if not hits:
            raise DiffError(f"{DIFF_FAILED}: SEARCH block {b} not found in parent")
        if len(hits) > 1:
            raise DiffError(f"{DIFF_FAILED}: SEARCH block {b} matches {len(hits)} places in parent (ambiguous)")
        i = hits[0]
        lines = lines[:i] + replace + lines[i + len(blk.search):]
    out = "\n".join(lines) + ("\n" if trailing_nl or not lines else "")
    if not out.strip():
        raise DiffError(f"{DIFF_FAILED}: the result is an empty program")
    if not out.endswith("\n"):
        out += "\n"
    return out, {"blocks": len(blocks), "lines_changed": lines_changed(src, out), "fuzzy_blocks": fuzzy,
                 "indent_fuzzy_blocks": indent_fuzzy}


def lines_changed(a: str, b: str) -> int:
    """Lines removed + lines added between two programs (difflib opcodes; unchanged lines don't count)."""
    sa, sb = a.splitlines(), b.splitlines()
    n = 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, sa, sb, autojunk=False).get_opcodes():
        if tag != "equal":
            n += (i2 - i1) + (j2 - j1)
    return n


NEAR_COPY_THRESHOLD = 0.9


def search_copy_fidelity(parent: str, text: str, *, near: float = NEAR_COPY_THRESHOLD) -> dict | None:
    """How faithfully each SEARCH block was copied from the parent. A measure only: it never
    changes whether a diff applies (apply_blocks decides that).

    Per block, against every window of the parent with the same number of lines:
      exact       an identical window exists
      whitespace  identical once each line is stripped of leading/trailing whitespace
      near        the most similar window has difflib ratio >= `near` (a near-miss copy)
      miss        anything else (the model wrote lines that are not in the parent)
    `copy_score` is the mean over blocks of the best window similarity (1.0 = exact copy).
    Returns None when the reply has no parseable SEARCH/REPLACE blocks."""
    try:
        blocks = parse_blocks(text)
    except DiffError:
        return None
    src = _norm(parent)
    lines = src.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    out = {"blocks": len(blocks), "exact": 0, "whitespace": 0, "near": 0, "miss": 0}
    sims = []
    for blk in blocks:
        k = len(blk.search)
        if k == 0 or not lines:
            out["miss"] += 1
            sims.append(0.0)
            continue
        if _occurrences(lines, blk.search):
            out["exact"] += 1
            sims.append(1.0)
            continue
        needle = "\n".join(blk.search)
        best = 0.0
        sm = difflib.SequenceMatcher(None, autojunk=False)
        sm.set_seq2(needle)
        for i in range(max(1, len(lines) - k + 1)):
            sm.set_seq1("\n".join(lines[i:i + k]))
            if sm.real_quick_ratio() <= best or sm.quick_ratio() <= best:
                continue
            best = max(best, sm.ratio())
        if _occurrences(lines, blk.search, key=str.strip):
            out["whitespace"] += 1
        elif best >= near:
            out["near"] += 1
        else:
            out["miss"] += 1
        sims.append(best)
    out["copy_score"] = round(sum(sims) / len(sims), 6)
    out["min_similarity"] = round(min(sims), 6)
    return out


def apply_diff(parent: str, text: str, *, allow_indent_fuzzy: bool = False) -> tuple[str, dict]:
    """parse_blocks + apply_blocks. Raises DiffError."""
    return apply_blocks(parent, parse_blocks(text), allow_indent_fuzzy=allow_indent_fuzzy)


def decide_edit(text: str, parent: str, mode: EditMode, *, diffs_forbidden: bool = False) -> EditOutcome:
    """Turn an agent reply into the full program to evaluate (rules in the module docstring).

    Never raises on agent output: every failure becomes EditOutcome(code=None, error=...)."""
    text = text or ""
    if mode not in EDIT_MODES:
        raise ValueError(f"unknown edit_mode {mode!r}")
    if mode == "full" and not diffs_forbidden:
        code = extract_program(text)
        return EditOutcome(code, "full" if code else None, None if code else NO_CODE_FULL)
    markers = has_diff_markers(text)
    if diffs_forbidden:
        if markers:
            code = extract_program(text)  # the diff is rejected; a fenced program is still usable
            if code:
                return EditOutcome(code, "full", None, DIFF_REJECTED_TRUNCATED)
            return EditOutcome(None, "diff", DIFF_REJECTED_TRUNCATED, DIFF_REJECTED_TRUNCATED)
        code = extract_program(text)
        return EditOutcome(code, "full" if code else None, None if code else NO_CODE_FULL)
    if markers:
        copy = {"copy": search_copy_fidelity(parent, text)}  # measured whether or not the diff applies
        try:
            code, stats = apply_diff(parent, text)
        except DiffError as e:
            code = extract_program(text) if mode == "auto" else None  # salvage: the reply's
            if code:                                                  # fenced program stands alone
                return EditOutcome(code, "full", None, str(e), copy)
            return EditOutcome(None, "diff", str(e), str(e), copy)
        return EditOutcome(code, "diff", None, None, {**stats, **copy})
    code = extract_program(text)
    if mode == "auto":
        return EditOutcome(code, "full" if code else None, None if code else NO_CODE_AUTO)
    if code:  # diff mode, full rewrite
        return EditOutcome(None, "full", FULL_REJECTED, FULL_REJECTED)
    return EditOutcome(None, None, NO_CODE_DIFF, NO_CODE_DIFF)


__all__ = [
    "DIFF_FAILED", "DiffError", "EDIT_MODES", "EditBlock", "EditMode", "EditOutcome", "apply_blocks", "apply_diff",
    "decide_edit", "extract_program", "first_marker_offset", "has_diff_markers", "lines_changed", "parse_blocks",
    "parse_full_rewrite_reason", "reason_category", "search_copy_fidelity",
]
