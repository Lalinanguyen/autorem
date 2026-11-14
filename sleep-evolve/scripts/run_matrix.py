"""Fan out runs, one per Modal CPU container, or sequentially on this machine.

Two modes, picked by the config file:

1. Baseline suite (the YAML has a `baselines:` section, harness/baselines.py): problems x seeds x
   arms, in dependency order. Wave 1 = the fatigue arm (always first) and every non-matched arm; wave
   2 = fixed_matched / random_matched, whose sleep schedules are computed from the wave-1 fatigue log
   of the same (problem, seed). The drift assertion (all arms identical outside their declared
   params) runs before anything is launched.

   On Modal (needs the secret "sleep-evolve-llm" with LLM_BASE_URL, LLM_API_KEY, LLM_MODEL):
       uv run modal run scripts/run_matrix.py --config configs/baselines.yaml --problems 36 --seeds 0,1,2,3,4
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
"""

from __future__ import annotations

import argparse
import json
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
            log(f"[wave {wave}] {s['run_id']}: best_norm={m.get('final_best_normalized')} "
                f"steps={s['steps_completed']} llm_sleeps={sl.get('llm_sleeps')} "
                f"(scheduled {sl.get('scheduled')}) forgets={sl.get('forgets')} "
                f"used={s['budget']['used']}/{s['budget']['total_tokens']} stop={s['stop_reason']}")
            summaries.append(s)
    return summaries


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


def _csv(s: str, cast=str) -> list:
    return [cast(x) for x in s.split(",") if x.strip()]


def _volume_runs() -> set[str]:
    try:
        return {Path(e.path).name for e in runs_volume.listdir("/")}
    except Exception:  # pragma: no cover - remote listing is best effort
        return set()


@app.local_entrypoint()
def main(config: str, problems: str, seeds: str, policies: str = "", arms: str = "", reuse: bool = False):
    if is_suite_file(config) and not policies:
        suite = load_suite(config)
        jobs = plan_suite(suite, _csv(problems, int), _csv(seeds, int), _csv(arms) or None,
                          out_dir=VOLUME_PATH, existing_sources=_volume_runs() if reuse else None)
        sd = suite_to_dict(suite)
        for wave in (1, 2):
            batch = [{"suite": sd, "job": j.to_dict()} for j in jobs if j.wave == wave]
            if not batch:
                continue
            print(f"wave {wave}: launching {len(batch)} runs")
            for summary in run_suite_job.map(batch, return_exceptions=True):  # blocks until the wave is done
                print(json.dumps(summary, indent=2, default=str))
        return
    base = load_config(config)
    cfgs = build_matrix(base, _csv(problems, int), _csv(policies), _csv(seeds, int))
    print(f"launching {len(cfgs)} runs")
    for summary in run_remote.map([c.model_dump() for c in cfgs], return_exceptions=True):
        print(json.dumps(summary, indent=2, default=str))


def _local(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--policies", default="", help="plain matrix mode (ignored for suite configs unless set)")
    ap.add_argument("--arms", default="", help="suite mode: comma-separated arms (default: the config's list)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--reuse", action="store_true", help="suite mode: reuse fatigue logs already in --out-dir")
    ap.add_argument("--mock-llm", action="store_true")
    args = ap.parse_args(argv)

    from harness.config import load_dotenv
    from scripts.run_one import execute

    load_dotenv()
    if is_suite_file(args.config) and not args.policies:
        suite = load_suite(args.config)
        run_suite_local(suite, _csv(args.problems, int), _csv(args.seeds, int), _csv(args.arms) or None,
                        out_dir=args.out_dir, mock_llm=args.mock_llm, reuse=args.reuse)
        return
    base = load_config(args.config)
    for cfg in build_matrix(base, _csv(args.problems, int), _csv(args.policies), _csv(args.seeds, int)):
        s = execute(cfg, mock_llm=args.mock_llm, out_dir=args.out_dir)
        print(f"{s['run_id']}: best={s['best_score']} steps={s['steps_completed']} stop={s['stop_reason']}")


if __name__ == "__main__":
    _local()
