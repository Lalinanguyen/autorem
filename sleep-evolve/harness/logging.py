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

`details` is an addition to the spec'd schema: it carries the consolidated block so
masked-prediction and analysis (tasks 5 and 8) can replay what the agent saw.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

FIELDS = (
    "run_id", "step", "parent_id", "program_id", "program", "score", "valid", "error",
    "delta_vs_best", "agent_tokens", "sleep_tokens", "wallclock", "event", "details",
)


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
        record = {k: None for k in FIELDS}
        record.update(fields)
        record["run_id"] = self.run_id
        record["wallclock"] = round(time.monotonic() - self._t0, 4)
        record["score"] = _finite_or_none(record["score"])
        record["delta_vs_best"] = _finite_or_none(record["delta_vs_best"])
        record["details"] = record["details"] or {}
        unknown = set(record) - set(FIELDS)
        if unknown:
            raise ValueError(f"unknown log fields: {unknown}")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record


def read_history(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
