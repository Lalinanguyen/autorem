"""Build a LoRA training spec from run outputs — the offline half of Level-2 learning.

Reads a run's history.jsonl (best programs) and a persisted memory file (verified items,
including dream-derived discoveries from `memory_path` runs), produces the spec dict that
infra/modal_train.py's train() consumes, and writes it to disk. Nothing here calls an LLM,
evaluates anything, or spends money.

    uv run python -m scripts.distill \
        --history runs/dream/history.jsonl \
        --memory runs/dream_memory.json \
        --task-file problems/p36.txt \
        --out runs/spec_se.json

Then train on Modal GPU (paid — approval required):
    uv run modal run infra/modal_train.py --spec-path runs/spec_se.json

After the job lands on the sleep-evolve-adapters volume, serve it (SERVE_LORA=<id>),
run held-out evals vs the base model, then close the loop:
    uv run python -m scripts.distill --register-only runs/spec_se.json \
        --registry runs/checkpoints.json
    uv run python -m scripts.distill --record-eval <adapter_id> \
        --metric <held-out score> --n-runs <N> --registry runs/checkpoints.json
    uv run python -m scripts.distill --promote <adapter_id> --min-delta 0.0 \
        --registry runs/checkpoints.json
    uv run python -m scripts.distill --rollback --registry runs/checkpoints.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sleep.memory import MemoryStore  # noqa: E402
from sleep.trainer import CheckpointRegistry, build_training_spec, collect_examples  # noqa: E402


def load_history(paths: list[str]) -> list[dict]:
    records: list[dict] = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_store(path: str | None) -> MemoryStore | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"memory file not found: {p}")
    return MemoryStore.from_state(json.loads(p.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a LoRA training spec from run outputs")
    ap.add_argument("--history", nargs="*", default=[], help="history.jsonl files to draw programs from")
    ap.add_argument("--memory", default=None, help="memory_path JSON from dream_distill runs")
    ap.add_argument("--task", default=None, help="task text for the SFT user prompt")
    ap.add_argument("--task-file", default=None, help="file containing the task text")
    ap.add_argument("--out", default=None, help="write the spec JSON here (default: stdout)")
    ap.add_argument("--max-examples", type=int, default=64)
    ap.add_argument("--best-k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--no-dreams", action="store_true", help="exclude dream-derived memory items")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--adapter-id", default=None)
    ap.add_argument("--register-only", default=None, metavar="SPEC_JSON",
                    help="skip example collection; just register this spec's adapter in --registry")
    ap.add_argument("--record-eval", default=None, metavar="ADAPTER_ID",
                    help="record a held-out eval result for an adapter (requires --metric/--n-runs/--registry)")
    ap.add_argument("--metric", type=float, default=None)
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--promote", default=None, metavar="ADAPTER_ID",
                    help="promote an adapter to champion if its latest eval beats the champion's")
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument("--rollback", action="store_true", help="restore the last retired champion")
    ap.add_argument("--registry", default=None, help="CheckpointRegistry JSON file")
    args = ap.parse_args(argv)

    if args.register_only:
        if not args.registry:
            ap.error("--register-only requires --registry")
        spec = json.loads(Path(args.register_only).read_text(encoding="utf-8"))
        rec = CheckpointRegistry(args.registry).register(
            spec["adapter_id"], base_model=spec["base_model"], train_meta=spec)
        print(json.dumps(rec, indent=2))
        return 0

    if args.record_eval:
        if not args.registry or args.metric is None or args.n_runs is None:
            ap.error("--record-eval requires --registry, --metric and --n-runs")
        rec = CheckpointRegistry(args.registry).record_eval(
            args.record_eval, args.metric, n_runs=args.n_runs, note=args.note)
        print(json.dumps(rec, indent=2))
        return 0

    if args.promote:
        if not args.registry:
            ap.error("--promote requires --registry")
        print(json.dumps(
            CheckpointRegistry(args.registry).promote_if_better(
                args.promote, min_delta=args.min_delta),
            indent=2))
        return 0

    if args.rollback:
        if not args.registry:
            ap.error("--rollback requires --registry")
        print(json.dumps(CheckpointRegistry(args.registry).rollback(), indent=2))
        return 0

    task = args.task
    if args.task_file:
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    if not task:
        ap.error("--task or --task-file is required")
    if not args.history and not args.memory:
        ap.error("nothing to distill: pass --history and/or --memory")

    history = load_history(args.history)
    store = load_store(args.memory)
    examples = collect_examples(task, store, history, max_examples=args.max_examples,
                                min_score=args.min_score, best_k=args.best_k,
                                include_dreams=not args.no_dreams)
    spec = build_training_spec(examples, base_model=args.base_model,
                               lora_rank=args.lora_rank, epochs=args.epochs, lr=args.lr,
                               max_len=args.max_len, adapter_id=args.adapter_id)
    text = json.dumps(spec, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        n_dream = sum(1 for e in examples if e.source == "dream")
        print(f"wrote {args.out}: {spec['adapter_id']} — {len(examples)} examples "
              f"({n_dream} dream-derived)")
        print(f"train: uv run modal run infra/modal_train.py --spec-path {args.out}")
        print(f"serve: SERVE_LORA={spec['adapter_id']} uv run modal deploy infra/modal_vllm.py")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
