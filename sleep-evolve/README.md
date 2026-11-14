# sleep-evolve

Tests **sleep-time consolidation** for LLM-driven evolutionary program search (AlphaEvolve-style)
on problems from [google-deepmind/alphaevolve_repository_of_problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems).

An agent proposes programs, an evaluator scores them, and every step is logged. Every so often a
**sleep** phase rewrites the run history into a compact consolidated block, which the agent reads
in place of the raw history. Sleep policies are compared at a **matched total token budget**: sleep
tokens and agent tokens draw from the same `Budget`.

This is a scaffold: one real path end to end (problem 36 + `no_sleep`/`summarize`), with mocks for
everything else. `DEVIN_TASKS.md` lists what remains.

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
| `llm/client.py` | `LLMClient.complete(messages, *, role, max_tokens)`: the only path to the API |
| `llm/mock.py` | deterministic mock transport with fixed usage |
| `harness/budget.py` | `Budget`: one total for agent + sleep, raises `BudgetExhausted` |
| `harness/logging.py` | JSONL history schema (read by sleep policies and analysis, never by the agent) |
| `harness/program_db.py` | all programs; parent = best w.p. 0.7, else uniform over the top 10 |
| `harness/loop.py` | the evolution loop |
| `harness/config.py` | `RunConfig` (YAML) |
| `sleep/` | `SleepPolicy` protocol; `NoSleep`, `Summarize`; `FullSleep` (stubs) |
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
