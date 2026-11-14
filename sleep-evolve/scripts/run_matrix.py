"""Fan out problems x policies x seeds, one run per Modal CPU container.

On Modal (needs the secret "sleep-evolve-llm" with LLM_BASE_URL, LLM_API_KEY, LLM_MODEL):
    uv run modal run scripts/run_matrix.py --config configs/example.yaml \
        --problems 36 --policies no_sleep,summarize --seeds 0,1,2
Results land in the Modal volume "sleep-evolve-runs":
    uv run modal volume get sleep-evolve-runs / ./runs_modal

Locally, sequentially (handy with the mock):
    uv run python -m scripts.run_matrix --config configs/mock.yaml \
        --problems 0 --policies no_sleep,summarize --seeds 0,1 --mock-llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modal  # noqa: E402

from harness.config import RunConfig, load_config  # noqa: E402

RUN_TIMEOUT_S = 6 * 60 * 60
VOLUME_PATH = "/runs"

app = modal.App("sleep-evolve-matrix")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "openai>=1.40", "pydantic>=2.6", "pyyaml>=6.0")
    .add_local_python_source("evaluators", "harness", "llm", "sleep", "scripts", "infra")
)
runs_volume = modal.Volume.from_name("sleep-evolve-runs", create_if_missing=True)


def build_matrix(base: RunConfig, problems: list[int], policies: list[str], seeds: list[int]) -> list[RunConfig]:
    """Every (problem, policy, seed) combination, sharing all other settings with `base`."""
    out = []
    for p in problems:
        for pol in policies:
            for s in seeds:
                cfg = base.model_copy(update={"problem_id": p, "sleep_policy": pol, "seed": s, "run_id": None})
                out.append(cfg)
    return out


@app.function(
    image=image,
    cpu=2.0,
    memory=4096,
    timeout=RUN_TIMEOUT_S,
    volumes={VOLUME_PATH: runs_volume},
    secrets=[modal.Secret.from_name("sleep-evolve-llm")],
)
def run_remote(cfg_dict: dict) -> dict:
    from scripts.run_one import execute

    cfg = RunConfig(**cfg_dict)
    cfg = cfg.model_copy(update={"eval_backend": "local"})  # already in a Modal CPU container
    try:
        summary = execute(cfg, out_dir=VOLUME_PATH)
    finally:
        runs_volume.commit()
    return {k: v for k, v in summary.items() if k != "config"}


def _csv(s: str, cast=str) -> list:
    return [cast(x) for x in s.split(",") if x.strip()]


@app.local_entrypoint()
def main(config: str, problems: str, policies: str, seeds: str):
    base = load_config(config)
    cfgs = build_matrix(base, _csv(problems, int), _csv(policies), _csv(seeds, int))
    print(f"launching {len(cfgs)} runs")
    for summary in run_remote.map([c.model_dump() for c in cfgs], return_exceptions=True):
        print(json.dumps(summary, indent=2, default=str))


def _local() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--problems", required=True)
    ap.add_argument("--policies", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--mock-llm", action="store_true")
    args = ap.parse_args()

    from harness.config import load_dotenv
    from scripts.run_one import execute

    load_dotenv()
    base = load_config(args.config)
    for cfg in build_matrix(base, _csv(args.problems, int), _csv(args.policies), _csv(args.seeds, int)):
        s = execute(cfg, mock_llm=args.mock_llm)
        print(f"{s['run_id']}: best={s['best_score']} steps={s['steps_completed']} stop={s['stop_reason']}")


if __name__ == "__main__":
    _local()
