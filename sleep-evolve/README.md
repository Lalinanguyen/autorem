# sleep-evolve

Tests **sleep-time consolidation** for LLM-driven evolutionary program search (AlphaEvolve-style)
on problems from [google-deepmind/alphaevolve_repository_of_problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems).

An agent proposes programs, an evaluator scores them, and every step is logged. Every so often a
**sleep** phase rewrites the run history into a compact consolidated block, which the agent reads
in place of the raw history. Sleep policies are compared at a **matched total token budget**: sleep
tokens and agent tokens draw from the same `Budget`.

Real evaluators exist for problems 7 (difference bases), 36 (circle packing), 50 (max/min distance
ratio) and 59 (no-isosceles grid subsets); each reproduces its published AlphaEvolve construction
(see `verify_problem`). Everything else is mocked.

## Setup

```bash
uv sync                      # Python 3.11, installs deps + pytest
cp .env.example .env         # then fill in LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
```

The LLM is any OpenAI-compatible endpoint, configured **only** by `LLM_BASE_URL`, `LLM_API_KEY`
and `LLM_MODEL`. The endpoint must return the `usage` field: token counts are never estimated.

## Commands

```bash
uv run pytest                                                         # tests
uv run python -m scripts.run_one --config configs/mock.yaml --mock-llm  # local mock run
uv run python -m scripts.run_one --config configs/example.yaml          # real p36 run
uv run modal deploy infra/modal_vllm.py                                  # optional vLLM endpoint
uv run modal deploy infra/modal_eval.py                                  # optional remote evaluators
uv run modal run scripts/run_matrix.py --config configs/example.yaml \
    --problems 36 --policies no_sleep,summarize --seeds 0,1,2            # matrix on Modal
```

Each run writes `runs/<run_id>/history.jsonl`, `summary.json` and `best_program.py`.

## Layout and contracts

| Path | Role |
|---|---|
| `evaluators/base.py` | `EvalResult`, `Evaluator` protocol, subprocess sandbox (`run_program`), `normalized_score` |
| `evaluators/mock.py` | problem 0: cheap quadratic for tests |
| `evaluators/p36_circle_packing.py` | problem 36: 26 circles in the unit square, max sum of radii |
| `evaluators/p7_difference_bases.py` | problem 7: integer set B covering differences 1..n, min \|B\|²/n |
| `evaluators/p50_max_min_ratio.py` | problem 50: 16 planar points, min (max/min pairwise distance)² |
| `evaluators/p59_no_isosceles.py` | problem 59: largest isosceles-free subset of the 64x64 grid |
| `llm/client.py` | `LLMClient.complete(messages, *, role, max_tokens)`: the only path to the API |
| `llm/mock.py` | deterministic mock transport with fixed usage |
| `harness/budget.py` | `Budget`: one total for agent + sleep, raises `BudgetExhausted` |
| `harness/logging.py` | JSONL history schema (read by sleep policies and analysis, never by the agent) |
| `harness/program_db.py` | all programs; parent = best w.p. 0.7, else uniform over the top 10 |
| `harness/loop.py` | the evolution loop |
| `harness/config.py` | `RunConfig` (YAML) |
| `sleep/` | `SleepPolicy` protocol; `NoSleep`, `Summarize`, `SleepLite`, `FullSleep` |
| `sleep/memory.py` | `MemoryStore`: typed items (goal/rule/fact/skill/dead_end), statuses, decay/evict/merge, `save_state`/`from_state` cross-run persistence |
| `sleep/replay.py` | paired A/B replay: control vs treatment through the sandbox, difference decides the claim |
| `sleep/dreamer.py`, `sleep/dream_io.py`, `sleep/insight_validator.py` | REM dreaming: seeded recombination of memory into hypotheses; each is validated by an A/B sandbox test before it can become verified (see below) |
| `sleep/trainer.py`, `infra/modal_train.py` | parametric learning: collect training examples from validated discoveries, LoRA spec, Modal GPU job, checkpoint registry with promote-only-if-better |
| `harness/evals.py` | `EvalBudget` + `CountingEvaluator`: every evaluator call (steps, replays, dream tests) counted by role; optional shared `max_evaluations` |
| `infra/modal_eval.py` | evaluators on Modal CPU; `ModalEvaluator` |
| `infra/modal_vllm.py` | vLLM OpenAI-compatible server on Modal (model from `serve:` in config) |
| `scripts/run_one.py`, `scripts/run_matrix.py` | single run; problems × policies × seeds fan-out |

### Design decisions worth knowing

- **Sandboxing.** A candidate program defines `construct()`. It runs in a fresh `python -I`
  subprocess in a temp dir, with a hard timeout that kills the whole process tree. Only the JSON
  return value comes back, and scoring happens on data in the parent. Program code never runs in
  the harness process.
- **Budget semantics.** Before each call, the client refuses if nothing remains and caps
  `max_tokens` at what remains. After the call, it charges the API-reported usage. If that pushes
  the total over the limit, the tokens are still recorded (`overshoot`), the response is
  **discarded**, and the run stops. No output generated past the budget ever affects a run;
  overshoot is at most one call's prompt.
- **No bypass.** The Budget belongs to the `LLMClient`, the loop reads it from `llm.budget`, and
  a test fails if any module other than `llm/client.py` imports `openai`.
- **Tokens in the log are cumulative** (`agent_tokens`, `sleep_tokens` at the time of the line).
  Each line also carries a `details` dict (evaluator details, or the sleep policy and block).
- **Sleep timing.** `should_sleep(step, history)` is checked right after step `step` is logged. A
  sleep line carries that step number, and the new block applies from step `step + 1`.
- **Invalid scores** are `-inf` internally and `null` in JSONL.
- **Normalized score** is `score / best_known` for maximization problems, and
  `best_known / score` for minimization problems (scores negated), so 1.0 always means "matches
  best known". Use `evaluators.base.normalized_score`.
- **p36** uses the notebook's verification logic with a 1e-9 tolerance on containment and overlap.
  `best_known = 2.6359830849176067` is the sum of AlphaEvolve's n=26 construction
  (`tests/data/p36_alphaevolve_n26.json`, copied from the notebook).

- **Model name.** The harness calls `LLM_MODEL`. The config's `model` is recorded in the summary,
  with a warning if it differs. `serve.model` is what `infra/modal_vllm.py` deploys.

## Edit formats (`edit_mode`)

The agent replies with a full rewrite (a ```python block) or SEARCH/REPLACE diffs against the
parent (`harness/edits.py`, aider/OpenEvolve format):

```
<<<<<<< SEARCH
(lines copied exactly from the parent)
=======
(replacement lines)
>>>>>>> REPLACE
```

Every block must match the parent exactly once (exact whole-line match first, then a
trailing-whitespace-only fallback). `full` (default) is the original behavior; `diff` accepts
diffs only; `auto` (used by the `*_openai.yaml` configs) makes diffs the default and allows a
full rewrite only with a `FULL_REWRITE_REASON:` line, which is logged and counted. A failed diff
is an invalid step (`diff_error`, counted as fatigue sloppiness), never a crash. If the parent had
to be truncated to fit the context cap, that step is full-rewrite only. The logged `program` is
always the full resulting program. `edit_mode` is part of the agent, so the drift check forces it
to be identical in every arm (GEPA arms use gepa's own proposer and ignore it).

Sleep is diff-first too: replays and dream tests use control `{"base": "<program id>"}` plus
treatment `{"diff": ...}` (`sleep/program_ref.py`), so a treatment differs from its control by
exactly the claimed change; full programs are a fallback with a reason. Sleep prompts show a
capped set of exact base programs (`base_programs`, `base_program_chars`) and reference everything
else by id. Per-step fields: `edit_mode`, `edit_kind`, `diff_blocks`, `diff_lines_changed`,
`diff_error`, `parent_truncated`, `full_rewrite_reason`; per-run totals in `summary.metrics.edits`.

## REM dreaming and parametric learning

FullSleep's REM pass (plateau / exhausted fatigue tier) can dream: `sleep/dreamer.py` samples
memory items with a seeded explore/exploit mix and asks the sleep model to recombine them into
**hypotheses** — associations, counterfactuals, extrapolations — each citing its source ids.
A dream is never trusted: `sleep/insight_validator.py` turns each testable hypothesis into the
same paired A/B sandbox test the consolidator uses. Confirmed -> added as a `verified` skill
(source `dream`, evidence `d<step>.<i>`); refuted -> stored refuted; uncertain or untestable ->
`tentative` at best. Validation evals are charged to the run's shared `EvalBudget` under the
sleep role, capped per pass by `dream_evals`, and dreaming is skipped below
`dream_min_budget_tokens` remaining budget.

Ladder arms (`harness/baselines.py`): `summarize`, `dream_unvalidated` (dreams enter memory
untested), `dream` (validation on), `dream_distill` (dream + `memory_path`: the store persists
across runs/problems so validated discoveries accumulate). Example matrix:
`configs/dream_openai.yaml`.

Level-2 (weights): `sleep/trainer.py` turns verified items and best programs into SFT examples
and builds a training spec; `scripts/distill.py` is the offline entrypoint
(`--history` + `--memory` -> spec.json; `--register-only` files the adapter in a
`CheckpointRegistry`). `infra/modal_train.py` runs the actual LoRA job on Modal GPU
(`uv run modal run infra/modal_train.py --spec-path spec.json`, adapter ->
`sleep-evolve-adapters` volume); `CheckpointRegistry` promotes an adapter only on a measured
improvement and keeps rollback history. `infra/modal_vllm.py` serves a promoted adapter with
`SERVE_LORA=<adapter_id>` (vLLM `--enable-lora`); the base model stays reachable at its own
name for champion/challenger runs. Inference APIs never change weights — this pipeline is the
only weight-update path.
