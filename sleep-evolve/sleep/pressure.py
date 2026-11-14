"""Sleep pressure: when should the agent stop and consolidate?

Four components, each in [0, 1], computed from step records (the JSONL dicts):
  stall          steps since the best valid score last improved / stall_window
  perseveration  fraction of steps since the last sleep that hit a known dead end
                 (perseveration_hit, set by FullSleep.observe via tag matching)
  clutter        unconsolidated backlog: steps since the last sleep / clutter_norm
  surprise       mean per-step surprise since the last sleep (capped at 1)
pressure = weighted mean of the components.

Gating: never sleep before `min_gap` steps since the last sleep, always sleep at
`max_gap`, in between sleep iff pressure >= threshold.

Missing fields (history from another loop) count as 0 / are skipped.
"""

from __future__ import annotations

import math

from sleep.base import step_events as _step_records

DEFAULT_WEIGHTS = {"stall": 1.0, "perseveration": 1.0, "clutter": 0.5, "surprise": 1.0}


def compute_surprise(score: float | None, valid: bool | None, expected: float | None,
                     best_known: float | None) -> float | None:
    """|score - expected| / |best_known| for valid steps with a prediction; invalid + prediction -> 1.0."""
    if expected is None or not math.isfinite(expected):
        return None
    if not valid or score is None or not math.isfinite(score):
        return 1.0
    scale = abs(best_known) if best_known else 1.0
    return abs(score - expected) / scale


def steps_since_improvement(records: list[dict], step: int) -> int:
    best = -math.inf
    last = None
    for h in records:
        s = h.get("score")
        if h.get("valid") and s is not None and s > best:
            best, last = s, h.get("step", 0)
    first = records[0].get("step", 0) if records else step
    return step - (last if last is not None else first)


def components(records: list[dict], *, step: int, last_sleep_step: int,
               stall_window: int = 10, clutter_norm: int = 20) -> dict:
    records = _step_records(records)
    since = [h for h in records if h.get("step", 0) > last_sleep_step]
    n = len(since)
    surprises = [min(1.0, h["surprise"]) for h in since if h.get("surprise") is not None]
    return {
        "stall": min(1.0, steps_since_improvement(records, step) / max(1, stall_window)),
        "perseveration": (sum(1 for h in since if h.get("perseveration_hit")) / n) if n else 0.0,
        "clutter": min(1.0, n / max(1, clutter_norm)),
        "surprise": (sum(surprises) / len(surprises)) if surprises else 0.0,
    }


def combine(comp: dict, weights: dict | None = None) -> float:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(w[k] for k in comp if w.get(k, 0) > 0)
    if total <= 0:
        return 0.0
    return round(sum(w.get(k, 0) * v for k, v in comp.items()) / total, 6)


def gate(step: int, last_sleep_step: int, pressure: float, *, threshold: float, min_gap: int, max_gap: int) -> bool:
    gap = step - last_sleep_step
    if gap < min_gap:
        return False
    if gap >= max_gap:
        return True
    return pressure >= threshold
