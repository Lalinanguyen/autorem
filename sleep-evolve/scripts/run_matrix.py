"""Fan out runs, one per Modal CPU container, or sequentially on this machine.

Two modes, picked by the config file:

1. Baseline suite (the YAML has a `baselines:` section, harness/baselines.py): problems x seeds x
   arms, in dependency order. Wave 1 = the fatigue arm (always first) and every non-matched arm; wave
   2 = fixed_matched / random_matched (sleep schedules) and no_sleep_extra_evals (extra-evaluation
   schedule), computed from the wave-1 fatigue log of the same (problem, seed). The drift assertion (all arms identical outside their declared
   params, including model and LLM settings) runs before anything is launched.

   On Modal (needs the secret "sleep-evolve-llm" with LLM_BASE_URL, LLM_API_KEY, LLM_MODEL):
       uv run modal run scripts/run_matrix.py --config configs/baselines_openai.yaml
       uv run modal run scripts/run_matrix.py --config configs/baselines.yaml --problems 36 --seeds 0,1 \
           --arms fixed_matched,random_matched --reuse      # wave 2 only, fatigue logs already on the volume
   Wave 2 starts after every wave-1 container has committed the runs volume; each wave-2 container
   reloads the volume and reads /runs/<fatigue run id>/history.jsonl.

   Locally, sequentially (with the mock):
       uv run python -m scripts.run_matrix --config configs/baselines_mock.yaml --problems 0 --seeds 0,1,2 \
           --mock-llm --out-dir runs/baselines_mock

2. Plain policy matrix (problems x policies x seeds, every other setting shared):
       uv run modal run scripts/run_matrix.py --config configs/example.yaml \
           --problems 36 --policies no_sleep,summarize --seeds 0,1,2
       uv run python -m scripts.run_matrix --config configs/mock.yaml \
           --problems 0 --policies no_sleep,summarize --seeds 0,1 --mock-llm

Results land in the Modal volume "sleep-evolve-runs":
    uv run modal volume get sleep-evolve-runs / ./runs_modal

Pre-flight cost check (every non-mock launch, local or Modal; harness/cost.py): before anything runs,
the planned job count (both waves) x per-run tokens (total_token_budget + one call of overshoot
allowance) x the model's dearest per-token price is printed, and the launch is REFUSED (exit 2) when
  - the estimate exceeds the cap: --max-dollars, else the config's `max_dollars`;
  - no cap is set at all;
  - the config model has no confirmed price (pass --price-per-1m <USD per 1M tokens> to override,
    e.g. --price-per-1m 0 for a self-hosted vLLM endpoint, whose cost is GPU time);
  - LLM_MODEL (when set locally) differs from the config's `model` (the estimate would be wrong).
--estimate-only (alias --dry-run) prints the estimate and exits without running anything. The mock
path (--mock-llm) skips the check. --problems / --seeds default to the suite's `baselines.problems` /
`baselines.seeds`, else the config's problem_id / seed.
    uv run python -m scripts.run_matrix --config configs/pilot_openai.yaml --estimate-only

A BillingError (quota / spend limit; never retried) stops the matrix at the first run that hits it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal  # noqa: E402

from harness.baselines import (  # noqa: E402
    Job, Suite, add_matching_to_summary, attach_schedule, is_suite_file, load_suite, make_suite, plan_suite,
    read_history_file,
)
from harness.config import RunConfig, load_config  # noqa: E402
from harness.cost import preflight_estimate, price_table  # noqa: E402

RUN_TIMEOUT_S = 6 * 60 * 60
VOLUME_PATH = "/runs"

app = modal.App("sleep-evolve-matrix")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "openai>=1.40", "pydantic>=2.6", "pyyaml>=6.0")
    .add_local_python_source("evaluators", "harness", "llm", "sleep", "scripts", "infra")
    # run_one.execute scores runs with benchmarks/ (registry: benchmarks/alphaevolve.yaml, a data file
    # that the default NON_PYTHON_FILES ignore would drop)
    .add_local_python_source("benchmarks", ignore=["**/*.pyc"])
)
runs_volume = modal.Volume.from_name("sleep-evolve-runs", create_if_missing=True)

ExecuteFn = Callable[[RunConfig, str], dict]


def build_matrix(base: RunConfig, problems: list[int], policies: list[str], seeds: list[int]) -> list[RunConfig]:
    """Every (problem, policy, seed) combination, sharing all other settings with `base`."""
    out = []
    for p in problems:
        for pol in policies:
            for s in seeds:
                cfg = base.model_copy(update={"problem_id": p, "sleep_policy": pol, "seed": s, "run_id": None})
                out.append(cfg)
    return out


# ---------------------------------------------------------------------------------------------
# Baseline suite: shared by the local and the Modal paths
# ---------------------------------------------------------------------------------------------


def suite_to_dict(suite: Suite) -> dict:
    return {"base": suite.base.model_dump(), "section": suite.section.model_dump()}


def suite_from_dict(d: dict) -> Suite:
    return make_suite(RunConfig(**d["base"]), d["section"])


def run_job(suite: Suite, job: Job, out_dir: str, execute_fn: ExecuteFn) -> dict:
    """Run one planned job. Matched jobs first read their source fatigue log from `out_dir`."""
    cfg, info = job.cfg, None
    if job.source_run_id:
        src = Path(out_dir) / job.source_run_id / "history.jsonl"
        if not src.exists():
            raise FileNotFoundError(f"{job.run_id} needs the fatigue log {src} (wave 1 failed or not run)")
        cfg, info = attach_schedule(suite, job, read_history_file(src))
    summary = execute_fn(cfg, out_dir)
    if info is not None:
        add_matching_to_summary(Path(out_dir) / cfg.resolved_run_id(), info)
        summary["matching"] = info
    return summary


def existing_runs(out_dir: str | Path) -> set[str]:
    d = Path(out_dir)
    return {p.parent.name for p in d.glob("*/history.jsonl")} if d.exists() else set()


def run_suite_local(suite: Suite, problems: list[int], seeds: list[int], arms: list[str] | None = None, *,
                    out_dir: str | None = None, mock_llm: bool = False, reuse: bool = False,
                    execute_fn: ExecuteFn | None = None, log: Callable[[str], None] = print) -> list[dict]:
    """Sequential suite run in dependency order (wave 1, then wave 2). Returns the summaries."""
    out = str(out_dir or suite.base.out_dir)
    if execute_fn is None:
        from scripts.run_one import execute

        def execute_fn(cfg: RunConfig, od: str) -> dict:
            return execute(cfg, mock_llm=mock_llm, out_dir=od)

    jobs = plan_suite(suite, problems, seeds, arms, out_dir=out,
                      existing_sources=existing_runs(out) if reuse else None)
    summaries = []
    for wave in (1, 2):
        for job in (j for j in jobs if j.wave == wave):
            s = run_job(suite, job, out, execute_fn)
            m = s.get("metrics") or {}
            sl = m.get("sleep") or {}
            cost = (s.get("cost") or {}).get("total")
            log(f"[wave {wave}] {s['run_id']}: best_norm={m.get('final_best_normalized')} "
                f"steps={s['steps_completed']} llm_sleeps={sl.get('llm_sleeps')} "
                f"(scheduled {sl.get('scheduled')}) forgets={sl.get('forgets')} "
                f"used={s['budget']['used']}/{s['budget']['total_tokens']} "
                f"cost={'-' if cost is None else f'${cost:.4f}'} stop={s['stop_reason']}")
            summaries.append(s)
    return summaries


# ---------------------------------------------------------------------------------------------
# Pre-flight cost check
# ---------------------------------------------------------------------------------------------


class PreflightRefused(Exception):
    """The pre-flight check refused to launch (the CLI exits with code 2)."""

    def __init__(self, reasons: list[str], estimate: dict):
        super().__init__("; ".join(reasons))
        self.reasons, self.estimate = reasons, estimate


def _csv(s: str, cast=str) -> list:
    return [cast(x) for x in (s or "").split(",") if x.strip()]


def resolve_targets(config: str, problems: str = "", seeds: str = "") -> tuple[list[int], list[int]]:
    """CLI lists, else the suite's baselines.problems / seeds, else the config's problem_id / seed."""
    import yaml

    data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    section = data.get("baselines") or {}
    ps = _csv(problems, int) or [int(x) for x in section.get("problems") or []] or [int(data.get("problem_id", 0))]
    ss = _csv(seeds, int) or [int(x) for x in section.get("seeds") or []] or [int(data.get("seed", 0))]
    return ps, ss


def _money(x: float | None) -> str:
    if x is None:
        return "unknown"
    return f"${x:,.2f}" if abs(x * 100 - round(x * 100)) < 1e-9 or x >= 1 else f"${x:,.4f}"


def preflight_evaluations(base: RunConfig, n_runs: int) -> dict:
    """Evaluator-call bounds for the pre-flight printout (no effect on the refusal rules)."""
    shared = base.sleep_policies.get("full_sleep", {})
    cap = base.max_evaluations
    return {"agent_per_run_max": base.max_steps + 1, "replay_budget": int(shared.get("replay_budget", 2)),
            "max_evaluations": cap, "total_cap": None if cap is None else cap * n_runs}


def preflight_check(base: RunConfig, n_runs: int, *, waves: dict | None = None, max_dollars: float | None = None,
                    price_per_1m: float | None = None, env_model: str | None = None,
                    log: Callable[[str], None] = print) -> dict:
    """Print the conservative estimate; raise PreflightRefused if the launch must not start."""
    est = preflight_estimate(base, n_runs, price_table(base.prices), rate_override_per_1m=price_per_1m)
    cap, cap_src = (max_dollars, "--max-dollars") if max_dollars is not None else (base.max_dollars,
                                                                                   "config max_dollars")
    pr, a = est["price_per_1m"], est["overshoot_allowance"]
    price_txt = ("no price in the table" if pr is None else
                 f"input {_money(pr['input'])}, cached input {_money(pr['cached_input'])}, output "
                 f"{_money(pr['output'])} per 1M; source {pr.get('source')}, checked {pr.get('checked')}")
    wave_txt = "" if not waves else " (" + ", ".join(f"wave {w}: {n}" for w, n in sorted(waves.items())) + ")"
    log("pre-flight cost estimate (conservative upper bound; no API calls)")
    log(f"  model           {est['model']}  [{price_txt}]")
    log(f"  runs            {est['runs']}{wave_txt}")
    log(f"  per-run tokens  {est['per_run_tokens']:,} = total_token_budget {est['total_token_budget']:,} + overshoot "
        f"allowance {a['total']:,} (max_completion_tokens {a['max_completion_tokens']:,} + prompt allowance "
        f"{a['prompt_allowance']:,})")
    log(f"  rate            {_money(est['rate_per_1m'])} per 1M tokens ({est['rate_source'] or 'no price'})")
    log(f"  estimate        {_money(est['cost'])} = {est['runs']} x {est['per_run_tokens']:,} tokens x "
        f"{_money(est['rate_per_1m'])} / 1M")
    log(f"  cap             {_money(cap)} ({cap_src})")
    est["evaluations"] = evals_info = preflight_evaluations(base, n_runs)
    log(f"  evaluations     information only (the dollar estimate is token-based): per run agent steps <= "
        f"{evals_info['agent_per_run_max']:,}, replays <= 2 x {evals_info['replay_budget']} per sleep (control + "
        f"treatment) + dream tests, max_evaluations "
        f"{'uncapped' if evals_info['max_evaluations'] is None else format(evals_info['max_evaluations'], ',')}"
        + ("" if evals_info["total_cap"] is None else f" (all runs <= {evals_info['total_cap']:,})"))
    reasons = []
    if est["cost"] is None:
        reasons.append(f"no confirmed price for model {est['model']!r}: add it to the config's `prices:` "
                       "or pass --price-per-1m")
    if cap is None:
        reasons.append("no spending cap: set `max_dollars` in the config or pass --max-dollars")
    elif est["cost"] is not None and est["cost"] > cap:
        reasons.append(f"estimate {_money(est['cost'])} exceeds the cap {_money(cap)} ({cap_src})")
    if env_model and base.model and env_model != base.model:
        reasons.append(f"LLM_MODEL={env_model!r} but the config model is {base.model!r}: calls would use "
                       "LLM_MODEL, so the estimate and the arms' shared model would be wrong")
    est.update(cap=cap, cap_source=cap_src, refused=bool(reasons), reasons=reasons)
    if reasons:
        for r in reasons:
            log(f"REFUSED: {r}")
        raise PreflightRefused(reasons, est)
    log("pre-flight OK")
    return est


def plan_launch(config: str, problems: list[int], seeds: list[int], policies: list[str], arms: list[str], *,
                out_dir: str | None = None, existing: set[str] | None = None):
    """(mode, base config, suite or None, jobs or cfgs, {wave: count}) for the launch the CLI describes."""
    if is_suite_file(config) and not policies:
        suite = load_suite(config)
        jobs = plan_suite(suite, problems, seeds, arms or None, out_dir=out_dir, existing_sources=existing)
        waves = {w: sum(1 for j in jobs if j.wave == w) for w in (1, 2)}
        return "suite", suite.base, suite, jobs, waves
    base = load_config(config)
    cfgs = build_matrix(base, problems, policies, seeds)
    return "matrix", base, None, cfgs, {1: len(cfgs)}


# ---------------------------------------------------------------------------------------------
# Modal
# ---------------------------------------------------------------------------------------------

_FN_KW = dict(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=RUN_TIMEOUT_S,
    volumes={VOLUME_PATH: runs_volume},
    secrets=[modal.Secret.from_name("sleep-evolve-llm")],
)


@app.function(**_FN_KW)
def run_remote(cfg_dict: dict) -> dict:
    from scripts.run_one import execute

    cfg = RunConfig(**cfg_dict)
    cfg = cfg.model_copy(update={"eval_backend": "local"})  # already in a Modal CPU container
    try:
        summary = execute(cfg, out_dir=VOLUME_PATH)
    finally:
        runs_volume.commit()
    return {k: v for k, v in summary.items() if k != "config"}


@app.function(**_FN_KW)
def run_suite_job(payload: dict) -> dict:
    """One baseline-suite job on Modal; wave-2 jobs read the committed wave-1 fatigue log."""
    from scripts.run_one import execute

    suite = suite_from_dict(payload["suite"])
    job = Job.from_dict(payload["job"])
    if job.source_run_id:
        runs_volume.reload()

    def execute_fn(cfg: RunConfig, od: str) -> dict:
        return execute(cfg.model_copy(update={"eval_backend": "local"}), out_dir=od)

    try:
        summary = run_job(suite, job, VOLUME_PATH, execute_fn)
    finally:
        runs_volume.commit()
    return {k: v for k, v in summary.items() if k != "config"}


def _volume_runs() -> set[str]:
    try:
        return {Path(e.path).name for e in runs_volume.listdir("/")}
    except Exception:  # pragma: no cover - remote listing is best effort
        return set()


@app.local_entrypoint()
def main(config: str, problems: str = "", seeds: str = "", policies: str = "", arms: str = "", reuse: bool = False,
         max_dollars: float = -1.0, price_per_1m: float = -1.0, estimate_only: bool = False):
    """Modal launch. max_dollars / price_per_1m < 0 = not given (config max_dollars / price table)."""
    ps, ss = resolve_targets(config, problems, seeds)
    suite_mode = is_suite_file(config) and not policies
    mode, base, suite, planned, waves = plan_launch(
        config, ps, ss, _csv(policies), _csv(arms), out_dir=VOLUME_PATH,
        existing=_volume_runs() if (reuse and suite_mode) else None)
    try:
        preflight_check(base, len(planned), waves=waves, max_dollars=max_dollars if max_dollars >= 0 else None,
                        price_per_1m=price_per_1m if price_per_1m >= 0 else None,
                        env_model=os.environ.get("LLM_MODEL"))
    except PreflightRefused:
        raise SystemExit(2)
    if estimate_only:
        return
    if mode == "suite":
        sd = suite_to_dict(suite)
        for wave in (1, 2):
            batch = [{"suite": sd, "job": j.to_dict()} for j in planned if j.wave == wave]
            if not batch:
                continue
            print(f"wave {wave}: launching {len(batch)} runs")
            for summary in run_suite_job.map(batch, return_exceptions=True):  # blocks until the wave is done
                print(json.dumps(summary, indent=2, default=str))
        return
    print(f"launching {len(planned)} runs")
    for summary in run_remote.map([c.model_dump() for c in planned], return_exceptions=True):
        print(json.dumps(summary, indent=2, default=str))


def _local(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--problems", default="", help="default: baselines.problems, else the config's problem_id")
    ap.add_argument("--seeds", default="", help="default: baselines.seeds, else the config's seed")
    ap.add_argument("--policies", default="", help="plain matrix mode (ignored for suite configs unless set)")
    ap.add_argument("--arms", default="", help="suite mode: comma-separated arms (default: the config's list)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reuse", action="store_true", help="suite mode: reuse fatigue logs already in --out-dir")
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--max-dollars", type=float, default=None,
                    help="refuse to start if the conservative estimate exceeds this (default: config max_dollars)")
    ap.add_argument("--price-per-1m", type=float, default=None,
                    help="override the per-token rate used by the estimate (USD per 1M tokens)")
    ap.add_argument("--estimate-only", "--dry-run", dest="estimate_only", action="store_true",
                    help="print the pre-flight estimate and exit without running anything")
    args = ap.parse_args(argv)

    from harness.config import load_dotenv
    from llm.chat_params import BillingError
    from scripts.run_one import execute

    load_dotenv()
    ps, ss = resolve_targets(args.config, args.problems, args.seeds)
    suite_mode = is_suite_file(args.config) and not args.policies
    out = args.out_dir
    existing = None
    if args.reuse and suite_mode:
        existing = existing_runs(out or load_suite(args.config).base.out_dir)
    mode, base, suite, planned, waves = plan_launch(args.config, ps, ss, _csv(args.policies), _csv(args.arms),
                                                    out_dir=out, existing=existing)
    if args.mock_llm:
        print(f"mock LLM: {len(planned)} runs, no pre-flight cost check")
    else:
        try:
            preflight_check(base, len(planned), waves=waves, max_dollars=args.max_dollars,
                            price_per_1m=args.price_per_1m, env_model=os.environ.get("LLM_MODEL"))
        except PreflightRefused:
            raise SystemExit(2)
    if args.estimate_only:
        return
    try:
        if mode == "suite":
            run_suite_local(suite, ps, ss, _csv(args.arms) or None, out_dir=out, mock_llm=args.mock_llm,
                            reuse=args.reuse)
            return
        for cfg in planned:
            s = execute(cfg, mock_llm=args.mock_llm, out_dir=out)
            print(f"{s['run_id']}: best={s['best_score']} steps={s['steps_completed']} stop={s['stop_reason']}")
    except BillingError as e:
        print(f"STOPPED: {e}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    _local()
