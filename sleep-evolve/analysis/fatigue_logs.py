"""Load run logs and recompute fatigue offline from history.jsonl with any FatigueConfig.

A run directory holds history.jsonl and (usually) summary.json. Metadata (problem_id, policy,
seed, best_known, total budget) comes from summary.json when present, else from the run id
pattern `p<problem>_<policy>_s<seed>`. Works on every arm's logs, including GEPA logs written
through RunLogger: missing fields are simply treated as absent data.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from harness.fatigue import COMPONENTS, FatigueConfig, FatigueState, replay_history
from harness.logging import read_history

_RUN_ID = re.compile(r"^p(?P<problem>\d+)_(?P<policy>.+)_s(?P<seed>\d+)$")


@dataclass
class RunLog:
    run_id: str
    path: Path
    history: list[dict]
    problem_id: int | None = None
    policy: str | None = None
    seed: int | None = None
    best_known: float | None = None
    total_budget: int | None = None
    summary: dict = field(default_factory=dict)

    @property
    def steps(self) -> list[dict]:
        return [h for h in self.history if h.get("event", "step") == "step"]


def load_run(run_dir: str | Path) -> RunLog:
    d = Path(run_dir)
    history = read_history(d / "history.jsonl")
    summary: dict = {}
    if (d / "summary.json").exists():
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    run_id = summary.get("run_id") or (history[0].get("run_id") if history else None) or d.name
    m = _RUN_ID.match(str(run_id)) or _RUN_ID.match(d.name)
    cfg = summary.get("config") or {}
    problem = summary.get("problem_id", cfg.get("problem_id"))
    policy = summary.get("sleep_policy", cfg.get("sleep_policy"))
    seed = summary.get("seed", cfg.get("seed"))
    if m:
        problem = int(m["problem"]) if problem is None else problem
        policy = m["policy"] if policy is None else policy
        seed = int(m["seed"]) if seed is None else seed
    total = (summary.get("budget") or {}).get("total_tokens") or cfg.get("total_token_budget")
    return RunLog(run_id=str(run_id), path=d, history=history, problem_id=problem, policy=policy, seed=seed,
                  best_known=summary.get("best_known"), total_budget=total, summary=summary)


def find_runs(root: str | Path, *, policies: Iterable[str] | None = None,
              problems: Iterable[int] | None = None) -> list[RunLog]:
    """Every run directory under `root` (searched recursively) holding a history.jsonl."""
    pol = set(policies) if policies is not None else None
    prob = set(problems) if problems is not None else None
    runs = []
    for h in sorted(Path(root).rglob("history.jsonl")):
        r = load_run(h.parent)
        if (pol is None or r.policy in pol) and (prob is None or r.problem_id in prob):
            runs.append(r)
    return runs


def recompute_fatigue(run: RunLog | list[dict], cfg: FatigueConfig | dict | None = None, *,
                      best_known: float | None = None, total_budget: int | None = None) -> list[FatigueState]:
    """One FatigueState per step record, recomputed from the raw log with `cfg`."""
    if isinstance(run, RunLog):
        history = run.history
        best_known = run.best_known if best_known is None else best_known
        total_budget = run.total_budget if total_budget is None else total_budget
    else:
        history = run
    return replay_history(history, cfg, best_known=best_known, total_budget=total_budget)


def fatigue_table(run: RunLog, cfg: FatigueConfig | dict | None = None, *, recompute: bool = True) -> dict:
    """Column arrays for one run: step, fatigue, load, impairment, each component, level, sleep flags.

    recompute=False reads the logged fatigue fields (falls back to recomputing if absent).
    """
    steps = run.steps
    logged = steps and all(s.get("fatigue") is not None for s in steps)
    cols: dict[str, list] = {k: [] for k in ("step", "fatigue", "load", "impairment", "level", "debt",
                                            "triggered", "deferred", "depth", *COMPONENTS)}
    if not recompute and logged:
        for s in steps:
            cols["step"].append(s.get("step"))
            cols["fatigue"].append(s["fatigue"])
            cols["load"].append(s.get("fatigue_load"))
            cols["impairment"].append(s.get("fatigue_impairment"))
            cols["level"].append(s.get("fatigue_level"))
            cols["debt"].append(s.get("fatigue_debt"))
            cols["triggered"].append(bool(s.get("sleep_triggered")))
            cols["deferred"].append(bool(s.get("sleep_deferred")))
            cols["depth"].append(s.get("sleep_depth"))
            for c in COMPONENTS:
                cols[c].append(s.get(f"fatigue_{c}"))
        if not any(cols["triggered"]):  # logs without decision flags: mark sleep events
            sleep_steps = {h.get("step") for h in run.history if h.get("event") == "sleep"}
            cols["triggered"] = [st in sleep_steps for st in cols["step"]]
        return cols
    for st in recompute_fatigue(run, cfg):
        cols["step"].append(st.step)
        cols["fatigue"].append(st.fatigue)
        cols["load"].append(st.load)
        cols["impairment"].append(st.impairment)
        cols["level"].append(st.level.label)
        cols["debt"].append(st.debt)
        cols["triggered"].append(st.triggered)
        cols["deferred"].append(st.deferred)
        cols["depth"].append(st.depth.label if st.triggered else None)
        for c in COMPONENTS:
            cols[c].append(st.components.get(c))
    return cols


def load_fatigue_config(path: str | Path | None) -> FatigueConfig:
    """FatigueConfig from a run config YAML's `fatigue:` section (or defaults)."""
    if path is None:
        return FatigueConfig()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return FatigueConfig(**(data.get("fatigue") or {}))
