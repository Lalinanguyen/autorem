"""Dream prompt (REM-phase hypothesis generation) and robust output validation.

The dreamer recombines memory items and step records into DREAMS: hypotheses, never facts.
Every dream cites the memory/step ids it was recombined from, and any program-testable dream
carries a paired A/B test (control vs treatment, the same contract as sleep/replay.py) that
sleep/insight_validator.py runs in the sandbox before memory is touched. Tests are diff-first
(sleep/program_ref.py): control {"base": "<program id>"} + treatment {"diff": SEARCH/REPLACE},
full programs ({"program", "reason"} or bare strings, the original shape) only as the fallback.

Parsing reuses consolidator_io.parse_json_object (fenced or bare JSON, never raises); the
validator below tolerates missing/garbage fields the way validate_consolidation does: bad
entries are dropped into a problems list, never raised on.
"""

from __future__ import annotations

from string import Template

from sleep.consolidator_io import parse_json_object  # re-exported: callers parse with this

DREAM_KINDS = ("association", "counterfactual", "extrapolation")

DREAM_SYSTEM = (
    "You are the REM-phase dreamer for an agent doing evolutionary program search on a\n"
    "math optimization problem. The agent is paused. You never work on the problem directly;\n"
    "you generate DREAMS: hypotheses produced by recombining the agent's memory and recent\n"
    "step record in ways the waking search has not tried. Dreams are hypotheses, never facts.\n"
    "Ground every dream in the provided memory items and step records and cite their ids.\n"
    "Do NOT merely summarize what memory already says: the point is novel recombination."
)

DREAM_USER = Template("""<task>$task</task>

<memory items="$n_items">
$memory_items_json
</memory>

<programs note="exact text; diff only against these">
$programs
</programs>

<steps>
$steps_jsonl
</steps>

<goals>
$goals
</goals>

<dream_directive>
$dream_directive
</dream_directive>

Kinds of dream:
  association:     combine two or more previously unrelated strategies/items into one hypothesis
  counterfactual:  revisit a failed or low-scoring step under a different assumption
  extrapolation:   a direction not present in memory at all (the riskiest kind; still cite the
                   items or steps whose absence or trend suggests it)

Each dream's "text" is ONE sentence: causal, concrete and testable, not narration. Every dream
names its "sources" (the memory ids and step ids it was recombined from) and gives a short
"rationale" for why those sources suggest it.

If the hypothesis is program-testable, attach a paired A/B "test" (each side defines construct()),
DIFF-FIRST:
  control:   {"base": "<program id>"}: an existing program from <programs> WITHOUT the change
  treatment: {"diff": "<<<<<<< SEARCH\\n(lines copied exactly from the control)\\n=======\\n(changed
             lines)\\n>>>>>>> REPLACE"}: the control with ONLY the claimed change; each SEARCH must
             match exactly one place in the control (use \\n for newlines in JSON)
Fall back to full programs ({"program": "...", "reason": "..."}) only when no existing program fits
(control) or a diff cannot express the change (treatment). A diff that does not apply leaves the
dream untested.
"Differs only by the claimed change" means: same structure, same parameters, same random seeds,
same iteration counts, same everything else; the line(s) implementing the change are the only
difference. Fix every random seed. Identical programs are discarded untested.
The programs need not be competitive with the best score; only the difference is judged.
"test" may be null only for hypotheses that genuinely cannot be sandbox-tested.

Output JSON only:
{
  "dreams": [{"did": "d1", "kind": "association|counterfactual|extrapolation",
              "text": "one-sentence testable hypothesis, causal and concrete",
              "sources": ["m3", "m7", "12"], "rationale": "why these sources suggest it",
              "test": {"control": {"base": "p12"},
                       "treatment": {"diff": "<<<<<<< SEARCH\\n...\\n=======\\n...\\n>>>>>>> REPLACE"},
                       "change": "what differs, one line"} | null}]
}""")


def validate_dreams(obj: dict | None, *, max_dreams: int,
                    max_chars: int = 300) -> tuple[list[dict], list[str]]:
    """Return (dreams, problems). Bad entries are dropped and reported, never raised.

    Each kept dream: {"did", "kind" (default "association"), "text", "sources" (list[str]),
    "rationale", "test" ({"control", "treatment", "change"} or None)}. Text and rationale are
    whitespace-collapsed and capped at max_chars; dids are deduped; at most max_dreams kept.
    """
    problems: list[str] = []
    if not isinstance(obj, dict):
        return [], ["not an object"]
    raw = obj.get("dreams", [])
    if not isinstance(raw, list):
        problems.append("dreams is not a list")
        raw = []
    dreams: list[dict] = []
    seen: set = set()
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            problems.append(f"dream {i} not an object")
            continue
        kind = str(d.get("kind", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if kind not in DREAM_KINDS:
            kind = "association"
        text = d.get("text")
        if not isinstance(text, str) or not text.strip():
            problems.append(f"dream {i} has no text")
            continue
        did = str(d.get("did") or f"d{i + 1}")
        if did in seen:
            problems.append(f"duplicate did {did}")
            continue
        seen.add(did)
        src = d.get("sources", [])
        if not isinstance(src, list):
            src = [src] if src is not None else []
        sources = [str(s) for s in src if isinstance(s, (str, int)) and str(s).strip()]
        rationale = d.get("rationale")
        dreams.append({
            "did": did, "kind": kind, "text": " ".join(text.split())[:max_chars],
            "sources": sources,
            "rationale": " ".join(rationale.split())[:max_chars]
            if isinstance(rationale, str) else "",
            "test": _as_test(d.get("test"), i, problems, max_chars),
        })
        if len(dreams) >= max_dreams:
            break
    return dreams, problems


def _as_test(t, i: int, problems: list[str], max_chars: int) -> dict | None:
    """Normalize a dream's optional A/B test; None when absent or garbage. Never raises."""
    if t is None:
        return None
    if not isinstance(t, dict):
        problems.append(f"dream {i} test is not an object")
        return None
    change = t.get("change")
    return {"control": t.get("control"), "treatment": t.get("treatment"),
            "change": " ".join(change.split())[:max_chars] if isinstance(change, str) else ""}
