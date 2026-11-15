"""Baseline comparison suite: named arms over ONE shared base config, differing only in the trigger.

  Q1  does an agent do better with sleep than without?          fatigue / fixed_letta vs no_sleep
  Q2  does fatigue-triggered sleep beat fixed-schedule sleep?    fatigue vs fixed_letta / fixed_matched,
                                                                 fixed_matched vs random_matched

Every sleeping arm is FullSleep (or SleepLite) with the SAME shared settings: consolidator prompts,
memory store, replay, decay, render, wake behavior (the base config's `sleep_policies.full_sleep`).
All arms share problems, seeds, model, total token budget (sleep tokens included), evaluator, awake
header (`awake_tags: true`), side_info, context layout and context cap. Each arm declares the policy
parameters it may change (ARM_SPECS); nothing else may differ.

Arms (ARM_SPECS):
  no_sleep        NoSleep: task + last K steps + parent program under the shared context cap. No memory
                  store, no consolidation; the fatigue monitor still runs and logs.
  fixed_letta     FullSleep gating=fixed, every_n=5 (Letta's default), timed_depth=nrem
  fixed_sparse    FullSleep gating=fixed, every_n=20, timed_depth=nrem
  fixed_matched   FullSleep gating=schedule, timed_depth=nrem; sleep_steps = the fatigue arm's LLM-sleep
                  count n on the same (problem, seed), at a fixed interval L // (n + 1) (L = the fatigue
                  run's last step): separates WHEN from HOW OFTEN
  random_matched  FullSleep gating=schedule, timed_depth=nrem; n steps sampled uniformly (seeded per
                  (problem, seed)) among all schedules in [1, L] with gaps >= min_gap: does timing matter?
  fatigue         FullSleep gating=fatigue: drowsy -> code-only forget, tired -> NREM, exhausted -> NREM+REM
  sleep_lite      SleepLite (optional): drowsy-tier forgetting only, zero sleep LLM calls
  fixed_tiered    (variant, not in the default list) gating=fixed, every_n=5, timed_depth=fatigue
Arms 2-5 run NREM only (timed_depth=nrem: no REM, no forget tier, scheduler not used for depth).
  summarize       Summarize policy every_n=5: plain notes, no structured memory (dream-ladder arm 3)
  dream_unvalidated  FullSleep gating=fatigue + REM dreaming; dream hypotheses enter memory as
                  tentative items but are NEVER sandbox-tested (arm 4: does validation matter?)
  dream           FullSleep gating=fatigue + REM dreaming + A/B dream validation (arm 5)
  dream_distill   `dream` plus an optional memory_path: the store persists across runs/problems so
                  validated discoveries accumulate for the offline distillation pipeline (arm 6;
                  actual LoRA training happens outside runs via sleep/trainer.py + infra/modal_train.py)
  no_sleep_extra_evals  NoSleep + extra_eval_schedule (variant, not in the default list): the eval-matched
                  Q1 control. Wave 2: for every sleep line of the fatigue run on the same (problem, seed) it
                  spends as many extra evaluator calls as that sleep did (replays + dream tests: the
                  increase of the line's cumulative `sleep_evaluations`, else details replay_evaluations +
                  dream_evaluations), after the same step, each on a small float-literal perturbation of
                  that step's parent (harness/perturb.py). Improving perturbations enter the ProgramDB, so
                  the control can actually use its extra evaluations (a judgment call: the strongest
                  control). Reads the source log like fixed_matched does.

"Sleep count" for matching = LLM sleeps (depth nrem or nrem_rem) in the fatigue arm's log; its
code-only forgets are not counted (they are reported separately in the metrics).

Drift rule (check_drift, raised before any run starts): for every arm, the fully resolved RunConfig
is flattened (`config_view`: every RunConfig field except run_id / out_dir / baseline_arm / sleep_policy /
sleep_policies, plus `sleep.<param>` for the parameters the arm's policy receives; for no_sleep the
shared sleep settings it carries unused) and diffed against the shared base. Any difference outside
`sleep.<declared param>` (and problem_id / seed, which define the pair) fails, as does a wrong
policy, `awake_tags` not true, or an effective context cap (min of context_cap_tokens and the
policy's own cap) that differs between arms. The LLM identity is also asserted explicitly: the config
`model`, the `llm:` block, the resolved request parameters (llm/chat_params.py: token param,
reasoning_effort, temperature, under the shared env), `prices` and `max_dollars` must be identical
in every arm, so agent and sleep calls of every arm get the same model and settings. The agent's edit format
(`edit_mode`: full / diff / auto, harness/edits.py) is an ordinary RunConfig field, so it is compared
like any other and must be identical in every arm: it belongs to the agent, not the sleep policy. Per-arm `overrides` / undeclared `arm_params` in the
suite YAML therefore fail the check.

Run order (plan_suite): wave 1 = fatigue (auto-added when a matched arm needs it) and every
non-matched arm; wave 2 = fixed_matched / random_matched, whose schedules are computed from the wave-1
fatigue log (attach_schedule). A schedule always fits inside the fatigue run's length; a matched run
can still end (budget) before all scheduled sleeps happen, so metrics log scheduled vs realized.
"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, Field

from harness.config import RunConfig
from llm.chat_params import resolve_chat_params
from harness.metrics import llm_sleep_steps, step_records

SHARED_POLICY = "full_sleep"  # the sleep_policies entry holding the shared sleep settings
PAIR_KEYS = ("problem_id", "seed")
VIEW_SKIP = ("run_id", "baseline_arm", "sleep_policy", "sleep_policies", "out_dir")
MATCHED_KINDS = ("even", "random", "evals")


class ConfigDriftError(ValueError):
    """An arm's resolved config differs from the shared base outside its declared parameters."""


@dataclass(frozen=True)
class ArmSpec:
    name: str
    policy: str                                   # no_sleep | full_sleep | sleep_lite
    params: dict = field(default_factory=dict)    # default values of the declared params
    declared: frozenset = frozenset()             # policy params this arm may set
    matched: str | None = None                    # "even" | "random" | "evals": schedule from the fatigue arm
    role: str = ""


def _spec(name, policy, params=None, matched=None, role="", extra=()):
    params = dict(params or {})
    return ArmSpec(name, policy, params, frozenset(params) | frozenset(extra), matched, role)


ARM_SPECS: dict[str, ArmSpec] = {s.name: s for s in (
    _spec("no_sleep", "no_sleep", role="Q1 control: no memory, no consolidation"),
    _spec("no_sleep_extra_evals", "no_sleep", {"extra_eval_schedule": []}, matched="evals",
          extra=("extra_eval_sigma",),
          role="Q1 eval-matched control: no_sleep + the fatigue run's sleep evaluations as parent perturbations"),
    _spec("fixed_letta", "full_sleep", {"gating": "fixed", "every_n": 5, "timed_depth": "nrem"},
          role="Letta-style NREM every 5 steps"),
    _spec("fixed_sparse", "full_sleep", {"gating": "fixed", "every_n": 20, "timed_depth": "nrem"},
          role="NREM every 20 steps"),
    _spec("fixed_matched", "full_sleep", {"gating": "schedule", "sleep_steps": [], "timed_depth": "nrem"},
          matched="even", role="fixed interval, sleep count matched to fatigue (when vs how often)"),
    _spec("random_matched", "full_sleep", {"gating": "schedule", "sleep_steps": [], "timed_depth": "nrem"},
          matched="random", role="random steps, sleep count matched to fatigue (timing control)"),
    _spec("fatigue", "full_sleep", {"gating": "fatigue"}, role="fatigue-triggered, tiered depth"),
    _spec("sleep_lite", "sleep_lite", {"gating": "fatigue", "max_depth": "forget"},
          role="drowsy-tier forgetting only (zero sleep LLM calls)"),
    _spec("fixed_tiered", "full_sleep", {"gating": "fixed", "every_n": 5, "timed_depth": "fatigue"},
          role="variant: fixed timing, depth from fatigue level"),
    # Dreaming arms (eval ladder arm 3-5; arm 6 = dream + persisted memory, distillation is offline)
    _spec("summarize", "summarize", {"every_n": 5}, role="NREM-by-time summarization (plain notes arm)"),
    _spec("dream_unvalidated", "full_sleep",
          {"gating": "fatigue", "dream": True, "dream_validate": False},
          role="NREM + REM associative dreaming; hypotheses land tentative, never sandbox-tested"),
    _spec("dream", "full_sleep", {"gating": "fatigue", "dream": True, "dream_validate": True},
          role="NREM + REM dreaming + A/B dream validation"),
    _spec("dream_distill", "full_sleep",
          {"gating": "fatigue", "dream": True, "dream_validate": True},
          extra=("memory_path",), role="dream + validation; memory persists across "
          "runs via memory_path so validated discoveries accumulate for offline distillation"),
)}

DEFAULT_ARMS = ("no_sleep", "fixed_letta", "fixed_sparse", "fixed_matched", "random_matched", "fatigue",
                "sleep_lite")
SOURCE_ARM = "fatigue"


class MatchingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    min_gap: int | None = Field(None, gt=0, description="min steps between matched sleeps (and from step 0); "
                                                        "None = fatigue.grace_steps + 1")
    seed_salt: str = Field("random_matched", description="salt of the per-(problem, seed) random schedule")


class BaselinesSection(BaseModel):
    """The `baselines:` section of a suite YAML (configs/baselines.yaml)."""

    model_config = {"extra": "forbid"}

    arms: list[str] = Field(default_factory=lambda: list(DEFAULT_ARMS))
    arm_params: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="values for an arm's declared params (others fail the drift check)")
    overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="per-arm RunConfig overrides; anything here fails the drift check "
                                          "unless it only touches declared params (testing hook)")
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    problems: list[int] | None = Field(None, description="default problems for run_matrix (CLI --problems wins)")
    seeds: list[int] | None = Field(None, description="default seeds for run_matrix (CLI --seeds wins)")


@dataclass
class Suite:
    base: RunConfig
    section: BaselinesSection

    @property
    def shared_sleep(self) -> dict:
        return dict(self.base.sleep_policies.get(SHARED_POLICY, {}))

    def min_gap(self) -> int:
        g = self.section.matching.min_gap
        return int(g) if g is not None else self.base.fatigue.grace_steps + 1


def is_suite_file(path: str | Path) -> bool:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return isinstance(data, dict) and "baselines" in data


def load_suite(path: str | Path, **overrides: Any) -> Suite:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = BaselinesSection(**(data.pop("baselines", None) or {}))
    data.update({k: v for k, v in overrides.items() if v is not None})
    return make_suite(RunConfig(**data), section)


def make_suite(base: RunConfig, section: BaselinesSection | dict | None = None) -> Suite:
    section = section if isinstance(section, BaselinesSection) else BaselinesSection(**(section or {}))
    unknown = (set(section.arms) | set(section.arm_params) | set(section.overrides)) - set(ARM_SPECS)
    if unknown:
        raise ValueError(f"unknown baseline arms {sorted(unknown)}; known: {sorted(ARM_SPECS)}")
    if SHARED_POLICY not in base.sleep_policies:
        raise ValueError(f"suite base config needs sleep_policies.{SHARED_POLICY} (the shared sleep settings)")
    return Suite(base, section)


# ---------------------------------------------------------------------------------------------
# Resolution and the drift assertion
# ---------------------------------------------------------------------------------------------


def arm_params(suite: Suite, arm: str) -> dict:
    spec = ARM_SPECS[arm]
    return {**copy.deepcopy(spec.params), **copy.deepcopy(suite.section.arm_params.get(arm, {}))}


def resolve_arm(suite: Suite, arm: str, problem_id: int, seed: int, *, sleep_steps: list[int] | None = None,
                extra_eval_schedule: list | None = None, out_dir: str | None = None) -> RunConfig:
    """The fully resolved RunConfig of `arm` on (problem_id, seed)."""
    spec = ARM_SPECS[arm]
    shared = suite.shared_sleep
    data = suite.base.model_dump()
    params = arm_params(suite, arm)
    if sleep_steps is not None:
        params["sleep_steps"] = sorted(int(s) for s in sleep_steps)
    if extra_eval_schedule is not None:
        params["extra_eval_schedule"] = sorted([int(s), int(k)] for s, k in extra_eval_schedule)
    policies: dict[str, dict] = {SHARED_POLICY: copy.deepcopy(shared)}
    if spec.policy != "no_sleep":
        policies[spec.policy] = {**copy.deepcopy(shared), **params}
    elif params:  # no_sleep variants get only their declared params (NoSleep ignores the shared settings)
        policies["no_sleep"] = params
    data.update(problem_id=problem_id, seed=seed, sleep_policy=spec.policy, sleep_policies=policies,
                baseline_arm=arm, run_id=None)
    if out_dir is not None:
        data["out_dir"] = out_dir
    for k, v in (suite.section.overrides.get(arm) or {}).items():
        data[k] = copy.deepcopy(v)
    return RunConfig(**data)


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out or {prefix.rstrip("."): {}}
    return {prefix.rstrip("."): obj}


def effective_policy_params(cfg: RunConfig) -> dict:
    """What the selected policy receives; for no_sleep the shared settings it carries unused."""
    if cfg.sleep_policy == "no_sleep":
        return {**cfg.sleep_policies.get(SHARED_POLICY, {}), **cfg.sleep_policies.get("no_sleep", {})}
    return cfg.policy_params


def effective_context_cap(cfg: RunConfig) -> int | None:
    pol = None if cfg.sleep_policy == "no_sleep" else cfg.policy_params.get("context_cap_tokens")
    caps = [c for c in (cfg.context_cap_tokens, pol) if c]
    return min(caps) if caps else None


def config_view(cfg: RunConfig) -> dict[str, Any]:
    """Flat comparable view of a resolved config (see the module docstring)."""
    data = cfg.model_dump()
    for k in VIEW_SKIP:
        data.pop(k, None)
    view = _flatten(data)
    view.update(_flatten(effective_policy_params(cfg), "sleep."))
    return {k: v for k, v in view.items() if k}


def base_view(suite: Suite) -> dict[str, Any]:
    cfg = suite.base.model_copy(update={"sleep_policy": SHARED_POLICY})
    return config_view(cfg)


_MISSING = object()


def llm_identity(cfg: RunConfig) -> dict:
    """Model, LLM settings and resolved request parameters of a config (identical in every arm)."""
    try:
        params = resolve_chat_params(cfg.model, cfg.llm, warn=False).to_dict()
    except ValueError as e:
        params = {"error": str(e)}
    return {"model": cfg.model, "llm": cfg.llm.model_dump(), "resolved": params,
            "prices": {k: v.model_dump() for k, v in sorted(cfg.prices.items())}, "max_dollars": cfg.max_dollars}


def drift_problems(suite: Suite, cfgs: Iterable[RunConfig]) -> list[str]:
    """Every violation of the one-difference rule among `cfgs` (empty list = OK)."""
    ref = base_view(suite)
    ref_llm = llm_identity(suite.base)
    problems: list[str] = []
    caps: dict[Any, set] = {}
    for cfg in cfgs:
        ident = llm_identity(cfg)
        if ident != ref_llm:
            diff = sorted(k for k in ident if ident[k] != ref_llm.get(k))
            problems.append(f"{cfg.baseline_arm or cfg.resolved_run_id()}: LLM settings differ from the shared base "
                            f"({', '.join(diff)}): {ident} vs {ref_llm}")
        arm = cfg.baseline_arm
        if arm not in ARM_SPECS:
            problems.append(f"{cfg.resolved_run_id()}: not a baseline arm ({arm!r})")
            continue
        spec = ARM_SPECS[arm]
        if cfg.sleep_policy != spec.policy:
            problems.append(f"{arm}: sleep_policy {cfg.sleep_policy!r} != declared {spec.policy!r}")
        if cfg.awake_tags is not True:
            problems.append(f"{arm}: awake_tags must be true in every arm (fatigue calibration needs EXPECTED_SCORE)")
        allowed = {f"sleep.{k}" for k in spec.declared} | set(PAIR_KEYS)
        view = config_view(cfg)
        for k in sorted(set(view) | set(ref)):
            if k in allowed or any(k.startswith(a + ".") for a in allowed):
                continue
            a, b = view.get(k, _MISSING), ref.get(k, _MISSING)
            if a != b:
                show = (lambda x: "<missing>" if x is _MISSING else repr(x))
                problems.append(f"{arm}: {k} = {show(a)} but the shared base has {show(b)} "
                                f"(not a declared parameter of {arm}: {sorted(spec.declared) or 'none'})")
        caps.setdefault((cfg.problem_id, cfg.seed), set()).add(effective_context_cap(cfg))
        caps.setdefault("all", set()).add(effective_context_cap(cfg))
    for key, vals in caps.items():
        if len(vals) > 1:
            problems.append(f"effective context cap differs between arms ({key}): {sorted(map(str, vals))}")
    return problems


def check_drift(suite: Suite, cfgs: Iterable[RunConfig]) -> None:
    problems = drift_problems(suite, list(cfgs))
    if problems:
        raise ConfigDriftError("baseline arms differ outside their declared parameters:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------------------------
# Matched schedules
# ---------------------------------------------------------------------------------------------


def feasible_gap(n: int, length: int, min_gap: int) -> int:
    """min_gap, lowered if n sleeps with that gap (first one >= gap after step 0) cannot fit in length."""
    if n <= 0:
        return max(1, min_gap)
    return max(1, min(min_gap, length // n))


def even_schedule(n: int, length: int, min_gap: int = 1) -> list[int]:
    """n sleeps at a fixed interval max(gap, L // (n + 1)): steps interval, 2*interval, ..., n*interval <= L."""
    n = max(0, min(int(n), int(length)))
    if n == 0:
        return []
    interval = max(feasible_gap(n, length, min_gap), length // (n + 1), 1)
    return [interval * (i + 1) for i in range(n)]


def random_schedule(n: int, length: int, min_gap: int, rng_seed: Any) -> list[int]:
    """n steps in [1, length], uniform over all sets whose consecutive gaps (and the first step,
    counted from step 0) are >= min_gap. Reproducible from `rng_seed`.

    Bijection: t_i = s_i - i * (g - 1) maps valid schedules to n-subsets of [1, length - n (g - 1)]."""
    n = max(0, min(int(n), int(length)))
    if n == 0:
        return []
    g = feasible_gap(n, length, min_gap)
    m = length - n * (g - 1)
    rng = random.Random(rng_seed)
    t = sorted(rng.sample(range(1, m + 1), n))
    return [ti + (i + 1) * (g - 1) for i, ti in enumerate(t)]


def random_seed_for(suite: Suite, problem_id: int, seed: int) -> str:
    return f"{suite.section.matching.seed_salt}|p{problem_id}|s{seed}"


def sleep_eval_schedule(history: list[dict]) -> list[list[int]]:
    """[[step, k], ...]: evaluator calls each sleep line of a run spent (replays + dream tests).

    k = the line's cumulative `sleep_evaluations` minus the previous line's; for logs without the
    field, details replay_evaluations + dream_evaluations. Sleeps that spent nothing are omitted."""
    out: list[list[int]] = []
    prev = 0
    for h in history:
        cur = h.get("sleep_evaluations")
        if h.get("event") == "sleep":
            if isinstance(cur, int):
                k = cur - prev
            else:
                d = h.get("details") or {}
                k = int(d.get("replay_evaluations") or 0) + int(d.get("dream_evaluations") or 0)
            if k > 0:
                out.append([int(h.get("step") or 0), int(k)])
        if isinstance(cur, int):
            prev = cur
    return out


def matched_source(history: list[dict]) -> dict:
    """LLM-sleep count, length and per-sleep evaluator calls of a source (fatigue) run log."""
    steps = step_records(history)
    sleeps = llm_sleep_steps(history)
    return {"n_sleeps": len(sleeps), "sleep_steps": sleeps,
            "length": max((int(h.get("step") or 0) for h in steps), default=0),
            "eval_schedule": sleep_eval_schedule(history)}


def schedule_for(suite: Suite, arm: str, problem_id: int, seed: int, source: dict) -> tuple[list, dict]:
    """Matched schedule of `arm` from the source log: sleep steps ("even" / "random") or, for the
    "evals" kind, the [[step, k], ...] extra-evaluation schedule."""
    spec = ARM_SPECS[arm]
    n, length = source["n_sleeps"], source["length"]
    g = suite.min_gap()
    if spec.matched == "evals":
        sched = [list(x) for x in source.get("eval_schedule") or []]
        info = {"kind": "evals", "source_arm": SOURCE_ARM, "source_sleep_evaluations": sum(k for _, k in sched),
                "source_length": length, "scheduled_extra_evals": sched}
        return sched, info
    if spec.matched == "even":
        steps = even_schedule(n, length, g)
        rng_seed = None
    elif spec.matched == "random":
        rng_seed = random_seed_for(suite, problem_id, seed)
        steps = random_schedule(n, length, g, rng_seed)
    else:
        raise ValueError(f"{arm} is not a matched arm")
    info = {"kind": spec.matched, "source_arm": SOURCE_ARM, "source_sleeps": n, "source_sleep_steps":
            source.get("sleep_steps"), "source_length": length, "min_gap": g,
            "effective_min_gap": feasible_gap(n, length, g), "rng_seed": rng_seed, "scheduled_steps": steps}
    return steps, info


# ---------------------------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------------------------


@dataclass
class Job:
    arm: str
    problem_id: int
    seed: int
    wave: int
    cfg: RunConfig
    source_run_id: str | None = None   # matched arms: the fatigue run whose log sets the schedule

    @property
    def run_id(self) -> str:
        return self.cfg.resolved_run_id()

    def to_dict(self) -> dict:
        return {"arm": self.arm, "problem_id": self.problem_id, "seed": self.seed, "wave": self.wave,
                "cfg": self.cfg.model_dump(), "source_run_id": self.source_run_id}

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        return cls(d["arm"], d["problem_id"], d["seed"], d["wave"], RunConfig(**d["cfg"]), d.get("source_run_id"))


def plan_suite(suite: Suite, problems: Iterable[int], seeds: Iterable[int], arms: Iterable[str] | None = None, *,
               out_dir: str | None = None, existing_sources: set[str] | None = None) -> list[Job]:
    """Jobs in two waves; raises ConfigDriftError before anything runs.

    Matched arms need the fatigue run of their (problem, seed): it is added to wave 1 unless its run id
    is in `existing_sources` (a log already on disk / on the volume, reused)."""
    arms = list(arms or suite.section.arms)
    unknown = set(arms) - set(ARM_SPECS)
    if unknown:
        raise ValueError(f"unknown baseline arms {sorted(unknown)}; known: {sorted(ARM_SPECS)}")
    existing = existing_sources or set()
    jobs: list[Job] = []
    for p in problems:
        for s in seeds:
            src_cfg = resolve_arm(suite, SOURCE_ARM, p, s, out_dir=out_dir)
            src_id = src_cfg.resolved_run_id()
            need_src = any(ARM_SPECS[a].matched for a in arms)
            wave1 = [a for a in arms if not ARM_SPECS[a].matched]
            if need_src and SOURCE_ARM not in wave1 and src_id not in existing:
                wave1.insert(0, SOURCE_ARM)
            if SOURCE_ARM in wave1:  # fatigue first within the wave
                wave1.remove(SOURCE_ARM)
                wave1.insert(0, SOURCE_ARM)
            for a in wave1:
                jobs.append(Job(a, p, s, 1, resolve_arm(suite, a, p, s, out_dir=out_dir)))
            for a in arms:
                if ARM_SPECS[a].matched:
                    jobs.append(Job(a, p, s, 2, resolve_arm(suite, a, p, s, out_dir=out_dir), source_run_id=src_id))
    check_drift(suite, [j.cfg for j in jobs])
    return jobs


def attach_schedule(suite: Suite, job: Job, history: list[dict]) -> tuple[RunConfig, dict]:
    """The matched job's config with sleep_steps from the source log (re-checked for drift)."""
    steps, info = schedule_for(suite, job.arm, job.problem_id, job.seed, matched_source(history))
    if ARM_SPECS[job.arm].matched == "evals":
        cfg = resolve_arm(suite, job.arm, job.problem_id, job.seed, extra_eval_schedule=steps,
                          out_dir=job.cfg.out_dir)
    else:
        cfg = resolve_arm(suite, job.arm, job.problem_id, job.seed, sleep_steps=steps, out_dir=job.cfg.out_dir)
    check_drift(suite, [cfg])
    info["source_run_id"] = job.source_run_id
    return cfg, info


def read_history_file(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def add_matching_to_summary(run_dir: str | Path, info: dict) -> None:
    """Record the matched schedule (source, kind, scheduled steps) in the run's summary.json."""
    p = Path(run_dir) / "summary.json"
    if not p.exists():
        return
    summary = json.loads(p.read_text(encoding="utf-8"))
    summary["matching"] = info
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")


__all__ = [
    "ARM_SPECS", "DEFAULT_ARMS", "ArmSpec", "BaselinesSection", "ConfigDriftError", "Job", "MatchingConfig", "Suite",
    "add_matching_to_summary", "attach_schedule", "check_drift", "config_view", "drift_problems", "effective_context_cap",
    "even_schedule", "is_suite_file", "load_suite", "make_suite", "matched_source", "plan_suite", "random_schedule",
    "resolve_arm", "schedule_for", "sleep_eval_schedule",
]
