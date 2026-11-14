"""Run one GEPA-arm configuration (requires `uv sync --group gepa`).

    uv run python -m arms.run --config configs/gepa_mock.yaml --mock-llm          # offline, p36
    uv run python -m arms.run --config configs/gepa_mock.yaml --mock-llm --arm gepa_sleep
    uv run python -m arms.run --config configs/gepa.yaml                          # real endpoint
    uv run python -m arms.run --config configs/gepa.yaml --arm gepa_sleep --seed 1

Real runs read LLM_BASE_URL / LLM_API_KEY / LLM_MODEL (as scripts/run_one.py does). An in-process
proxy (infra/llm_proxy.py) forwards GEPA's calls to LLM_BASE_URL and charges the run's Budget.
--mock-llm replaces the upstream with arms/mock_upstream.py (no network, no keys).
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arms.gepa_arm import GEPAArmConfig, load_arm_config, run_gepa  # noqa: E402
from evaluators import make_evaluator  # noqa: E402
from harness.budget import Budget  # noqa: E402
from harness.config import load_dotenv  # noqa: E402
from infra.llm_proxy import LLMProxy  # noqa: E402
from llm.client import LLMClient  # noqa: E402


def execute(cfg: GEPAArmConfig, *, mock_llm: bool = False, out_dir: str | None = None) -> dict:
    budget = Budget(cfg.total_token_budget)
    eval_kwargs = {"timeout_s": cfg.eval_timeout_s} if cfg.eval_timeout_s else {}
    evaluator = make_evaluator(cfg.problem_id, **eval_kwargs)
    if cfg.eval_backend == "modal":
        from infra.modal_eval import ModalEvaluator

        evaluator = ModalEvaluator(evaluator)

    with ExitStack() as stack:
        if mock_llm:
            from arms.mock_upstream import FakeUpstream, mock_transport_responder, p36_responder
            from llm.mock import MockTransport

            responder = (p36_responder(seed=cfg.seed) if cfg.problem_id == 36
                         else mock_transport_responder(MockTransport(seed=cfg.seed)))
            upstream = stack.enter_context(FakeUpstream(responder))
            proxy = stack.enter_context(LLMProxy(upstream.base_url, "mock-key", "mock-model"))
            sleep_llm = LLMClient(budget, MockTransport(seed=cfg.seed))
        else:
            proxy = stack.enter_context(LLMProxy.from_env())
            sleep_llm = LLMClient.from_env(budget)

        if cfg.arm == "gepa":
            summary = run_gepa(cfg, evaluator, budget, proxy, out_dir=out_dir)
        else:
            from arms.gepa_sleep_arm import run_gepa_sleep

            summary = run_gepa_sleep(cfg, evaluator, budget, proxy, sleep_llm, out_dir=out_dir)

    from benchmarks.scoring import enrich_after_run  # adds the `alphaevolve` block to summary.json

    return enrich_after_run(summary, out_dir or cfg.out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mock-llm", action="store_true", help="fake local upstream (no network)")
    ap.add_argument("--arm", choices=["gepa", "gepa_sleep"])
    ap.add_argument("--problem-id", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-id")
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    load_dotenv()
    cfg = load_arm_config(args.config, arm=args.arm, problem_id=args.problem_id, seed=args.seed,
                          run_id=args.run_id)
    summary = execute(cfg, mock_llm=args.mock_llm, out_dir=args.out_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
