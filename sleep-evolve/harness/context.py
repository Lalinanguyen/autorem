"""Bounded agent context: a deterministic local token counter and build_context().

`count_tokens` is ONLY for sizing prompts (caps on the agent context and on rendered
memory). Budget charging always uses the API-reported usage (llm/client.py). The counter
is ceil(utf8_bytes / 3), which over-counts typical BPE tokenizers (~4 bytes/token), so a
cap expressed in counter tokens is a conservative cap in real tokens.

build_context() assembles the agent messages and guarantees, by measuring the final
strings, that count_tokens(system) + count_tokens(user) <= cap. System prompt, task and
the fixed framing are required (ValueError if they alone exceed the cap). Everything
else shrinks in this order until the context fits:
  1. drop the oldest recent steps
  2. trim the parent's side_info (halve its char budget, down to nothing)
  3. drop memory lines from the end (memory is rendered strongest-first)
  4. truncate the parent program (head + tail with a marker)
"""

from __future__ import annotations

import json
from typing import Any, Iterable

MEMORY_TITLE = "## Consolidated notes from earlier in this run"
INSTRUCTION = "Write an improved version of the parent program."


def count_tokens(text: str) -> int:
    """Deterministic prompt-size estimate: ceil(utf8 bytes / 3)."""
    return -(-len(text.encode("utf-8")) // 3)


def messages_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m.get("content", "")) for m in messages)


def fit_lines(lines: Iterable[str], cap: int | None, header: str = "") -> str:
    """Greedily join `lines` (in order) under `header` so the result never exceeds `cap` tokens.

    A line that does not fit is skipped (later, shorter lines may still fit). Returns ""
    if nothing fits. With cap=None every line is kept.
    """
    out = header
    kept = 0
    for line in lines:
        cand = f"{out}\n{line}" if out else line
        if cap is None or count_tokens(cand) <= cap:
            out, kept = cand, kept + 1
    return out if kept else ""


def compact_json(obj: Any, max_chars: int) -> str:
    """JSON for a dict like side_info, cut to max_chars (with an ellipsis marker)."""
    if not obj or max_chars <= 0:
        return ""
    try:
        s = json.dumps(obj, separators=(",", ":"), default=str, allow_nan=True)
    except (TypeError, ValueError):
        s = str(obj)
    return s if len(s) <= max_chars else s[: max(0, max_chars - 3)] + "..."


def truncate_middle(code: str, keep: int) -> str:
    """Keep ~60% head and ~40% tail of `code` (keep chars total) with a marker in between."""
    if keep >= len(code):
        return code
    keep = max(0, keep)
    head_n = (keep * 3 + 4) // 5
    tail_n = keep - head_n
    head = code[:head_n]
    tail = code[len(code) - tail_n:] if tail_n else ""
    return f"{head}\n# ... [{len(code) - keep} chars truncated to fit the context cap] ...\n{tail}"


def render_recent_step(h: dict, side_info_chars: int) -> str:
    from sleep.base import render_step

    line = render_step(h)
    tags = h.get("approach_tags")
    if tags:
        line += f" | tags={','.join(tags)}"
    if h.get("expected_score") is not None:
        line += f" | expected={h['expected_score']:.6g}"
    if h.get("perseveration_hit"):
        line += " | repeated a known dead end"
    si = compact_json(h.get("side_info"), side_info_chars)
    if si:
        line += f" | side_info={si}"
    return line


def build_context(
    *,
    system: str,
    task: str,
    memory: str | list[str] = "",
    recent: list[dict] | None = None,
    parent_id: str,
    parent_score: float | None,
    parent_code: str,
    parent_side_info: dict | None = None,
    cap: int | None = None,
    step_side_info_chars: int = 160,
    parent_side_info_chars: int = 600,
    instruction: str = INSTRUCTION,
) -> tuple[list[dict], dict]:
    """Return (messages, stats). See the module docstring for the shrink order and guarantee."""
    recent = list(recent or [])
    mem_lines = [ln for ln in (memory.splitlines() if isinstance(memory, str) else memory) if ln.strip()]
    score_s = f"{parent_score:.6g}" if parent_score is not None else "invalid"

    def assemble(n_recent: int, psi_chars: int, n_mem: int, prog_keep: int) -> list[dict]:
        parts = [f"## Task\n{task}"]
        if n_mem:
            parts.append(MEMORY_TITLE + "\n" + "\n".join(mem_lines[:n_mem]))
        shown = recent[len(recent) - n_recent:] if n_recent else []
        if shown:
            parts.append("## Most recent steps\n" + "\n".join(
                render_recent_step(h, step_side_info_chars) for h in shown))
        code = truncate_middle(parent_code, prog_keep)
        parts.append(f"## Parent program ({parent_id}, score {score_s})\n```python\n{code}```")
        psi = compact_json(parent_side_info, psi_chars)
        if psi:
            parts.append(f"## Evaluator diagnostics for the parent\n{psi}")
        parts.append(instruction)
        return [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]

    full = len(parent_code)
    state = [len(recent), parent_side_info_chars, len(mem_lines), full]
    msgs = assemble(*state)
    if cap is not None and messages_tokens(msgs) > cap:
        minimal = assemble(0, 0, 0, 0)
        if messages_tokens(minimal) > cap:
            raise ValueError(
                f"system prompt + task need {messages_tokens(minimal)} tokens, over the context cap {cap}"
            )
        while messages_tokens(msgs) > cap:
            if state[0] > 0:
                state[0] -= 1
            elif state[1] > 0:
                state[1] = state[1] // 2 if state[1] >= 40 else 0
            elif state[2] > 0:
                state[2] -= 1
            else:  # binary search the largest program prefix/suffix that fits; keep=0 fits (checked)
                lo, hi = 0, full
                while lo < hi:
                    mid = (lo + hi + 1) // 2
                    if messages_tokens(assemble(0, 0, 0, mid)) <= cap:
                        lo = mid
                    else:
                        hi = mid - 1
                state[3] = lo
                msgs = assemble(*state)
                break
            msgs = assemble(*state)
    tokens = messages_tokens(msgs)
    assert cap is None or tokens <= cap
    stats = {
        "tokens": tokens,
        "cap": cap,
        "recent_shown": state[0],
        "recent_dropped": len(recent) - state[0],
        "memory_lines_shown": state[2],
        "memory_lines_dropped": len(mem_lines) - state[2],
        "parent_side_info_chars": state[1],
        "program_truncated": state[3] < full,
    }
    return msgs, stats
