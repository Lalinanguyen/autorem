"""Run one (problem, policy, seed) configuration.

    uv run python -m scripts.run_one --config configs/mock.yaml --mock-llm
    uv run python -m scripts.run_one --config configs/example.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluators import make_evaluator  # noqa: E402
from harness.budget import Budget  # noqa: E402
from harness.config import RunConfig, load_config, load_dotenv  # noqa: E402
from harness.loop import run  # noqa: E402
from llm.client import BillingError, LLMClient  # noqa: E402
from sleep import make_policy  # noqa: E402


def execute(cfg: RunConfig, *, mock_llm: bool = False, out_dir: str | None = None) -> dict:
    budget = Budget(cfg.total_token_budget)
    if mock_llm:
        from llm.mock import MockTransport

        llm = LLMClient(budget, MockTransport(seed=cfg.seed))
    else:
        llm = LLMClient.from_env(budget, cfg.llm)

    eval_kwargs = {"timeout_s": cfg.eval_timeout_s} if cfg.eval_timeout_s else {}
    evaluator = make_evaluator(cfg.problem_id, **eval_kwargs)
    if cfg.eval_backend == "modal":
        from infra.modal_eval import ModalEvaluator

        evaluator = ModalEvaluator(evaluator)

    policy = make_policy(cfg.sleep_policy, cfg.policy_params)
    summary = run(cfg, evaluator, llm, policy, out_dir=out_dir)

    from benchmarks.scoring import enrich_after_run  # adds the `alphaevolve` block to summary.json

    return enrich_after_run(summary, out_dir or cfg.out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mock-llm", action="store_true", help="use the deterministic mock LLM (no network)")
    ap.add_argument("--problem-id", type=int)
    ap.add_argument("--sleep-policy")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_config(
        args.config, problem_id=args.problem_id, sleep_policy=args.sleep_policy,
        seed=args.seed, run_id=args.run_id,
    )
    try:
        summary = execute(cfg, mock_llm=args.mock_llm, out_dir=args.out_dir)
    except BillingError as e:  # outputs were written (stop_reason "billing_error"); never retried
        print(f"STOPPED: {e}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
