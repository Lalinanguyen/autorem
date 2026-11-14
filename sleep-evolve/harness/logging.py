"""Run history as JSONL: one line per event. Only sleep policies and analysis read it; the agent never does.

Line schema (every key always present):
  run_id         str
  step           int    agent step (0 = seed program); a sleep line carries the step it ran AFTER
  parent_id      str | null
  program_id     str | null   (null for sleep events)
  program        str | null   full source (null for sleep events)
  score          float | null (null when invalid or for sleep events; JSON has no -inf)
  valid          bool | null
  error          str | null
  delta_vs_best  float | null score minus the best valid score before this step
  agent_tokens   int    CUMULATIVE agent tokens charged so far in the run
  sleep_tokens   int    CUMULATIVE sleep tokens charged so far in the run
  wallclock      float  seconds since run start
  event          "step" | "sleep"
  details        dict   evaluator details (step) or {"policy", "block"} (sleep)
  side_info      dict   EvalResult.side_info: actionable evaluator diagnostics ({} for sleep events)

Awake / sleep metrics (always present; null when the policy or event does not produce them):
  approach_tags     list[str] | null  step: tags from the agent's APPROACH_TAGS header line
  expected_score    float | null      step: the agent's EXPECTED_SCORE prediction
  rationale         str | null        step: RATIONALE header line (<= 500 chars)
  relevant_memory   list[str] | null  step: memory ids from RELEVANT_MEMORY
  surprise          float | null      step: |score - expected| / |best_known|; 1.0 if invalid with a
                                      prediction; null without a prediction
  perseveration_hit bool | null       step: tags matched a dead-end memory item (policy observe hook)
  context_tokens    int | null        step: local-counter size of the agent prompt (harness/context.py)
  pressure          float | null      step: sleep pressure after this step; sleep: pressure at sleep time
  sleep_mode        "nrem"|"rem"|null sleep
  block_tokens      int | null        sleep: local-counter size of the new block / memory section
  items_added       int | null        sleep: memory items created
  items_evicted     int | null        sleep: memory items evicted
  replays_wasted    int | null        sleep: replays that decided no memory edit

Fatigue (harness/fatigue.py; every arm, every line; null when fatigue is disabled or the writer,
e.g. a GEPA arm, does not run the monitor). Step lines carry the state observed at that step plus
the sleep decision taken after it; sleep lines carry the state right after the sleep.
  fatigue                 float  load_weight*load + impairment_weight*impairment + debt
  fatigue_load            float  homeostatic load in [0, load_cap]
  fatigue_impairment      float  weighted mean of the components below, in [0, 1]
  fatigue_calibration     float | null   each component in [0, 1]; 0 until the rested baseline
  fatigue_perseveration   float | null   exists, null when the log lacks the data
  fatigue_stagnation      float | null
  fatigue_sloppiness      float | null
  fatigue_narrowing       float | null
  fatigue_bloat           float | null
  fatigue_level           "rested" | "drowsy" | "tired" | "exhausted"
  fatigue_debt            float  sleep debt (from deferred sleeps)
  fatigue_grace           bool   inside the post-wake grace period
  fatigue_baseline_ready  bool   the rested baseline has been measured since the last wake
  sleep_triggered         bool   a sleep/forget of depth >= forget ran after this step
  sleep_deferred          bool   a due sleep was deferred after this step (debt added)
  sleep_depth             "forget" | "nrem" | "nrem_rem" | null

LLM usage since the previous line (harness/budget.py Budget.usage_since; every writer: loop, GEPA arms).
A step line covers the agent call that produced it (GEPA: the reflection calls since the previous
evaluation); a sleep line covers that sleep's LLM calls. A discarded (budget-overshooting) call has
no line; it is still in summary.json["usage"]. Values as reported by the API, never estimated.
  llm_calls              int | null   charged calls since the previous line (0 = none)
  llm_prompt_tokens      int | null   sum of usage.prompt_tokens (null when llm_calls is 0)
  llm_completion_tokens  int | null   sum of usage.completion_tokens (includes reasoning tokens)
  llm_cached_tokens      int | null   sum of usage.prompt_tokens_details.cached_tokens (subset of prompt;
                                      null when no call reported it)
  llm_reasoning_tokens   int | null   sum of usage.completion_tokens_details.reasoning_tokens (subset of
                                      completion; null when no call reported it)

`details` is an addition to the spec'd schema: it carries the consolidated block so
masked-prediction and analysis (tasks 5 and 8) can replay what the agent saw.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from harness.fatigue import FATIGUE_FIELDS

USAGE_FIELDS = ("llm_calls", "llm_prompt_tokens", "llm_completion_tokens", "llm_cached_tokens",
                "llm_reasoning_tokens")

FIELDS = (
    "run_id", "step", "parent_id", "program_id", "program", "score", "valid", "error",
    "delta_vs_best", "agent_tokens", "sleep_tokens", "wallclock", "event", "details", "side_info",
    "approach_tags", "expected_score", "rationale", "relevant_memory", "surprise", "perseveration_hit",
    "context_tokens", "pressure", "sleep_mode", "block_tokens", "items_added", "items_evicted",
    "replays_wasted",
    *FATIGUE_FIELDS,
    *USAGE_FIELDS,
)


class UsageCursor:
    """Per-line usage fields: the Budget's calls since the previous take()."""

    def __init__(self, budget: Any):
        self.budget = budget
        self.pos = len(getattr(budget, "calls", []))

    def take(self) -> dict:
        calls = getattr(self.budget, "calls", None)
        if calls is None:
            return {k: None for k in USAGE_FIELDS}
        agg = self.budget.usage_since(self.pos)
        self.pos += agg["calls"]
        return {"llm_calls": agg["calls"], "llm_prompt_tokens": agg["prompt_tokens"],
                "llm_completion_tokens": agg["completion_tokens"], "llm_cached_tokens": agg["cached_tokens"],
                "llm_reasoning_tokens": agg["reasoning_tokens"]}


def _finite_or_none(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


class RunLogger:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")  # one file per run, start fresh
        self.run_id = run_id
        self._t0 = time.monotonic()

    def log(self, **fields: Any) -> dict:
        return self.write(self.build(**fields))

    def build(self, **fields: Any) -> dict:
        """The normalized record for `fields` (validated, not yet written)."""
        record = {k: None for k in FIELDS}
        record.update(fields)
        record["run_id"] = self.run_id
        record["wallclock"] = round(time.monotonic() - self._t0, 4)
        record["score"] = _finite_or_none(record["score"])
        record["delta_vs_best"] = _finite_or_none(record["delta_vs_best"])
        record["expected_score"] = _finite_or_none(record["expected_score"])
        record["surprise"] = _finite_or_none(record["surprise"])
        record["details"] = record["details"] or {}
        record["side_info"] = record["side_info"] or {}
        unknown = set(record) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown log fields: {unknown}")
        return record

    def write(self, record: dict) -> dict:
        """Append a record from build() (possibly updated with known fields) to the file."""
        unknown = set(record) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown log fields: {unknown}")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record


def read_history(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
