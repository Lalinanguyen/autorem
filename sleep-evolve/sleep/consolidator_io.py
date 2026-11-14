"""Consolidator prompt (PRIORITIZE / EXTRACT CLAIMS / PLAN PAIRED REPLAYS / GOALS / MERGES),
robust JSON parsing, history view.

The history view tolerates records that lack awake fields (approach_tags, expected_score,
rationale, surprise, perseveration_hit): GEPA and other loops pass JSONL history they wrote
themselves. Every per-step field is length-capped, so the prompt is bounded.
"""

from __future__ import annotations

import difflib
import json
import re
from string import Template

from sleep.memory import normalize_tags

CLAIM_KINDS = ("rule", "fact", "skill", "dead_end")
RELATIONS = ("new", "supports", "contradicts")

CONSOLIDATOR_SYSTEM = (
    "You are the sleep-phase consolidator for an agent doing evolutionary program search on a\n"
    "math optimization problem. The agent is paused. You never work on the problem directly; you\n"
    "maintain its long-term memory. The agent will wake up seeing ONLY: the task, the memory you\n"
    "maintain, and its last few steps. Anything you don't keep is lost to it."
)

CONSOLIDATOR_USER = Template("""<task>$task</task>

<memory items="$n_items">
$memory_items_json
</memory>

<history since_step="$last_sleep_step" to_step="$step">
$steps_jsonl
<!-- per step: id, approach_tags, expected_score, rationale, actual score, valid,
     delta_vs_best, surprise, side_info (violations, overlaps, slack, errors),
     perseveration_hit, program diff summary -->
</history>

<sleep_pressure>stall=$stall perseveration=$persev surprise=$surprise</sleep_pressure>
$mode_note
STEP 1 · PRIORITIZE
Select at most $k_steps steps worth learning from, ranked by:
1. surprise (a large gap between expected_score and actual), the agent's own prediction error
2. large score changes, in either direction
3. new or repeated failure modes in side_info
4. perseveration hits: the agent retried something memory already marks as a dead end.
   Ask why the dead end didn't stop it. Was the item too vague? Too weak? Badly tagged?
Skip routine steps that only confirmed what memory already says.

STEP 2 · EXTRACT CLAIMS
From the selected steps, write candidate claims: causal, reusable statements, not narration.
  good: "Initializing centers on a hex lattice before L-BFGS-B gives a +0.02 score jump at n=26"
  bad:  "Step 41 tried hex init and it went up"
Also list existing memory items that the new evidence bears on (supports or contradicts).

STEP 3 · PLAN REPLAYS (paired A/B tests)
For up to $replay_budget claims that are both decision-relevant and unverified, write a PAIRED
test: two small programs that each define construct().
  control:   a simple, deterministic baseline program WITHOUT the claimed change
  treatment: the control with ONLY the claimed change applied
"Differs only by the claimed change" means: same structure, same parameters, same random seeds,
same iteration counts, same everything else; the line(s) implementing the change are the only
difference. Fix every random seed. Identical programs are discarded unscored.
Only the difference is judged, so the programs need not be competitive with the best score:
  rule / fact / skill: confirmed if treatment beats control by more than $margin_note
  dead_end (the treatment applies the dead-end idea): confirmed if treatment does not beat control
  an invalid treatment with a valid control confirms a dead end and refutes other claims
  a crash in either program, or an invalid control, decides nothing (wasted)
Every replay must name the memory edit it will decide. Don't propose replays whose outcome
wouldn't change memory.

STEP 4 · GOALS
Write 1-$max_goals goals, one sentence each: the current strategy and the next experiments. The
agent sees goals first, right after the best-score line (g0: maintained automatically, never
replace it). Goals never decay. Each goal either replaces a stale goal (its id in "replaces") or
is new ("replaces": null); at most $max_goals are kept, and a new goal pushes out the oldest.

STEP 5 · MERGE NEAR-DUPLICATES
If two or more memory items say the same thing in different words, merge them: list their ids
and write one combined statement. The merged item keeps the union of their evidence and tags and
the strongest status. Never merge a goal with a non-goal, or a dead end with a non-dead end.

Output JSON only:
{
  "selected_steps": [ids],
  "claims": [{"cid": "c1", "kind": "rule|fact|skill|dead_end", "text": "...",
              "evidence": [step ids], "approach_tags": [...],
              "relates_to": [memory ids], "relation": "new|supports|contradicts"}],
  "replays": [{"cid": "c1", "change": "one line: what the treatment changes",
               "control": "...python defining construct()...",
               "treatment": "...the control with only the change applied...",
               "decides": "add c1 as verified | refute m7 | ..."}],
  "goals": [{"text": "one sentence", "replaces": "g2" or null}],
  "merges": [{"ids": ["m3", "m7"], "text": "one combined statement"}]
}""")

REM_NOTE = ("<mode>REM: the search has plateaued. Prefer claims that generalize across many steps or "
            "point to untried directions; replays are scarce this sleep.</mode>\n")
NREM_NOTE = "<mode>NREM: stabilize and verify what the recent steps showed.</mode>\n"
RETRY_NOTE = ("Your reply contained no valid JSON object. Reply again with the JSON object only, "
              "exactly in the requested format, with no other text.")


def parse_json_object(text: str) -> tuple[dict | None, str | None]:
    """Find the first JSON object in `text` (fenced or bare). Never raises. Returns (obj, error)."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty response"
    candidates = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL) + [text.strip()]
    first, last = text.find("{"), text.rfind("}")
    if 0 <= first < last:
        candidates.append(text[first:last + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj, None
        except (json.JSONDecodeError, ValueError):
            pass
    dec = json.JSONDecoder()
    for m in list(re.finditer(r"\{", text))[:50]:
        try:
            obj, _ = dec.raw_decode(text, m.start())
            if isinstance(obj, dict):
                return obj, None
        except (json.JSONDecodeError, ValueError):
            continue
    return None, "no JSON object found"


def _as_step_ids(xs) -> list[int]:
    out = []
    for x in xs if isinstance(xs, list) else [xs] if xs is not None else []:
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            out.append(x)
        elif isinstance(x, float) and x.is_integer():
            out.append(int(x))
        elif isinstance(x, str) and re.fullmatch(r"\s*(?:step\s*)?(\d+)\s*", x, re.I):
            out.append(int(re.sub(r"\D", "", x)))
    return out


def validate_consolidation(obj: dict | None, *, max_claims: int, max_chars: int = 300) -> tuple[list[dict], list[dict], list[str]]:
    """Return (claims, replays, problems). Bad entries are dropped and reported, never raised."""
    problems: list[str] = []
    if not isinstance(obj, dict):
        return [], [], ["not an object"]
    claims: list[dict] = []
    raw_claims = obj.get("claims", [])
    if not isinstance(raw_claims, list):
        problems.append("claims is not a list")
        raw_claims = []
    seen = set()
    for i, c in enumerate(raw_claims):
        if not isinstance(c, dict):
            problems.append(f"claim {i} not an object")
            continue
        kind = str(c.get("kind", "")).strip().lower().replace("-", "_").replace(" ", "_")
        text = c.get("text")
        if kind not in CLAIM_KINDS or not isinstance(text, str) or not text.strip():
            problems.append(f"claim {i} has bad kind/text")
            continue
        cid = str(c.get("cid") or f"c{i + 1}")
        if cid in seen:
            problems.append(f"duplicate cid {cid}")
            continue
        seen.add(cid)
        relation = str(c.get("relation", "new")).strip().lower()
        rel = c.get("relates_to", [])
        claims.append({
            "cid": cid, "kind": kind, "text": " ".join(text.split())[:max_chars],
            "evidence": _as_step_ids(c.get("evidence", [])),
            "approach_tags": normalize_tags(c.get("approach_tags")),
            "relates_to": [str(r) for r in (rel if isinstance(rel, list) else [rel]) if isinstance(r, (str, int))],
            "relation": relation if relation in RELATIONS else "new",
        })
        if len(claims) >= max_claims:
            break
    replays = obj.get("replays", [])
    if not isinstance(replays, list):
        problems.append("replays is not a list")
        replays = []
    replays = [r for r in replays if isinstance(r, dict)]
    for i, r in enumerate(replays):
        if "program" in r and "control" not in r and "treatment" not in r:
            problems.append(f"replay {i} ({r.get('cid')}) uses the old single-program shape; "
                            "needs control + treatment")
    return claims, replays, problems


def validate_goals_and_merges(obj: dict | None, *, max_goals: int = 3,
                              max_chars: int = 300) -> tuple[list[dict], list[dict], list[str]]:
    """Return (goals, merges, problems) from the consolidator object. Never raises.

    goals:  [{"text": str, "replaces": str | None}] (at most max_goals; a bare string is a new goal)
    merges: [{"ids": [str, ...] (>= 2 distinct), "text": str | None}]
    Missing keys are fine (older outputs had neither).
    """
    problems: list[str] = []
    if not isinstance(obj, dict):
        return [], [], []
    goals: list[dict] = []
    raw = obj.get("goals", [])
    if not isinstance(raw, list):
        problems.append("goals is not a list")
        raw = []
    for i, g in enumerate(raw):
        if isinstance(g, str):
            g = {"text": g}
        if not isinstance(g, dict) or not isinstance(g.get("text"), str) or not g["text"].strip():
            problems.append(f"goal {i} has no text")
            continue
        rep = g.get("replaces")
        goals.append({"text": " ".join(g["text"].split())[:max_chars],
                      "replaces": str(rep) if isinstance(rep, (str, int)) and str(rep).strip() else None})
    if len(goals) > max_goals:
        problems.append(f"{len(goals)} goals given; kept the first {max_goals}")
        goals = goals[:max_goals]
    merges: list[dict] = []
    raw = obj.get("merges", [])
    if not isinstance(raw, list):
        problems.append("merges is not a list")
        raw = []
    for i, mg in enumerate(raw):
        ids = mg.get("ids") if isinstance(mg, dict) else None
        ids = list(dict.fromkeys(str(x) for x in ids if isinstance(x, (str, int)))) if isinstance(ids, list) else []
        if len(ids) < 2:
            problems.append(f"merge {i} needs at least two ids")
            continue
        text = mg.get("text")
        merges.append({"ids": ids, "text": " ".join(text.split())[:max_chars]
                       if isinstance(text, str) and text.strip() else None})
    return goals, merges, problems


def diff_summary(parent_code: str | None, code: str | None, max_lines: int = 10, max_chars: int = 300) -> str:
    if not code:
        return "no program"
    if parent_code is None:
        return f"new program, {len(code.splitlines())} lines"
    changed = [ln for ln in difflib.unified_diff(parent_code.splitlines(), code.splitlines(), lineterm="", n=0)
               if ln[:1] in "+-" and not ln.startswith(("+++", "---"))]
    plus = sum(1 for ln in changed if ln.startswith("+"))
    s = f"+{plus}/-{len(changed) - plus} lines: " + " | ".join(ln[:80] for ln in changed[:max_lines])
    return s[:max_chars]


def step_view(h: dict, programs: dict[str, str], *, side_info_chars: int = 240, text_chars: int = 200) -> dict:
    from harness.context import compact_json

    def rnd(x):
        return round(x, 6) if isinstance(x, float) else x

    return {
        "id": h.get("step"),
        "approach_tags": h.get("approach_tags") or [],
        "expected_score": rnd(h.get("expected_score")),
        "rationale": (h.get("rationale") or "")[:text_chars] or None,
        "score": rnd(h.get("score")),
        "valid": h.get("valid"),
        "delta_vs_best": rnd(h.get("delta_vs_best")),
        "surprise": rnd(h.get("surprise")),
        "side_info": compact_json(h.get("side_info"), side_info_chars) or None,
        "error": (h.get("error") or "")[:text_chars] or None,
        "perseveration_hit": h.get("perseveration_hit"),
        "diff": diff_summary(programs.get(h.get("parent_id")), h.get("program")),
    }
