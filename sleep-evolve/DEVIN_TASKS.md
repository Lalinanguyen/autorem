# Devin tasks

Each section below is self-contained. Read `README.md` ("Design decisions worth knowing") first.

## Ground rules (apply to every task)

**Frozen contracts.** Do not change the signatures or semantics of:
- `evaluators/base.py`: `EvalResult`, `Evaluator`, `run_program`, `SubprocessEvaluator.evaluate`, `normalized_score`
- `llm/client.py`: `LLMClient.complete(messages, *, role, max_tokens) -> LLMResponse`
- `harness/budget.py`: `Budget`, `BudgetExhausted`
- `harness/logging.py`: the JSONL schema (`FIELDS`). New information goes inside `details`.
- `sleep/base.py`: `SleepPolicy`

If a task truly needs a contract change, stop and open a question instead of changing it.

**Invariants.**
- Every LLM call goes through `LLMClient.complete` and is charged to the run's `Budget`. Sleep work
  uses `role="sleep"`. Never import `openai` outside `llm/client.py` (`tests/test_llm_client.py` enforces this).
- Candidate programs only ever run through `evaluators.base.run_program` (subprocess + timeout) or
  on Modal. Never `exec`/`eval`/`import` them in the harness process.
- Minimization problems negate internally: `score = -value` and `best_known = -published_value`.
  `normalized_score` then handles the direction.
- Secrets come only from env vars. No new hard dependencies without a one-line justification in the PR.
  Optional extras go in a `[dependency-groups]` entry.
- `uv run pytest` passes before and after your change. Existing tests may not be weakened.

**Evaluator rule (tasks 1–3).** Every evaluator needs a test that reproduces a published score from a
known construction. Store the construction as data in `tests/data/pNN_*.json` (extract numbers, do not
execute notebook code), and cite its source (repo path + cell, or paper section) in the module docstring.

---

## Task 1: Evaluators batch A (problems 2, 3, 5, 7)

**Goal.** Add evaluators for problems 2, 3, 5 and 7, following `evaluators/p36_circle_packing.py`.

| # | Title (problems/N.html) | Source of definition + construction |
|---|---|---|
| 2 | An autocorrelation problem related to Sidon sets | `experiments/autocorrelation_problems/autocorrelation_problems.ipynb` |
| 3 | How much can an autocorrelation of a non-negative function resemble an indicator function? | same notebook |
| 5 | Erdős minimum overlap problem | no notebook in this repo; the page links the AlphaEvolve results colab `google-deepmind/alphaevolve_results/mathematical_results.ipynb` |
| 7 | Difference Bases | `experiments/difference_bases/difference_bases.ipynb` |

Base the objective, validity checks, direction (min or max) and `best_known` on the notebook's verification
function and AlphaEvolve's construction, not on memory. Where a problem has several instances (sizes, n),
pick the one the notebook verifies an AlphaEvolve construction for, and record the choice in the docstring.
Write `description` as a complete task statement for the agent, including the `construct()` return format.

**Touch.** `evaluators/p02_*.py`, `p03_*.py`, `p05_*.py`, `p07_*.py`; add registry lines to
`evaluators/__init__.py` (additions only); `tests/test_evaluators_batch_a.py`; `tests/data/`.

**Do NOT touch.** `evaluators/base.py`, `evaluators/p36_circle_packing.py`, `harness/`, `llm/`, `sleep/`, `infra/`.

**Acceptance.**
- For each problem, a test evaluates a program that returns the published construction, and gets
  `valid=True` and the published score (sign-adjusted) within 1e-9 relative, or the notebook's stated precision.
- For each problem, at least one invalid construction (wrong shape, constraint violation) gives `valid=False`.
- `seed_program()` is valid for each problem and finishes in under 5 s.
- `make_evaluator(n)` works for n in {2, 3, 5, 7}.

## Task 2: Evaluators batch B (problems 34, 44, 59)

**Goal.** Same as task 1, for:

| # | Title | Source |
|---|---|---|
| 34 | Tammes problem | `experiments/tammes_problem/tammes_problem.ipynb` |
| 44 | Sum-difference problem III | `experiments/sums_differences_problems/sums_differences_problems.ipynb` |
| 59 | Subsets of the grid with no isosceles triangles | `experiments/subsets_of_the_grid_with_no_isosceles_triangles/subsets_of_the_grid_with_no_isosceles_triangles.ipynb` |

**Touch.** `evaluators/p34_*.py`, `p44_*.py`, `p59_*.py`; registry additions in `evaluators/__init__.py`;
`tests/test_evaluators_batch_b.py`; `tests/data/`.

**Do NOT touch.** The same list as task 1, plus the batch A evaluator files.

**Acceptance.** Same as task 1, for n in {34, 44, 59}. For combinatorial problems (59), the validity
check must be exact (integer arithmetic), not floating-point.

## Task 3: Evaluators batch C (problems 48, 50, adapted from OpenEvolve)

**Goal.** Add problem 48 (Heilbronn problem in a fixed bounding box) and problem 50 (max-to-min distance
ratios). Adapt them from OpenEvolve's `examples/alphaevolve_math_problems` (github.com/algorithmicsuperintelligence/openevolve).
The problems repo has no notebook for these; their pages link the AlphaEvolve results colab, so
cross-check each instance and `best_known` against that colab and arXiv:2511.02864 (Sections 6.29, 6.30).
Rewrite OpenEvolve's evaluator logic as a `SubprocessEvaluator.score_solution` and do not import
OpenEvolve. Note its license in the docstring.

**Touch.** `evaluators/p48_*.py`, `p50_*.py`; registry additions; `tests/test_evaluators_batch_c.py`; `tests/data/`.

**Do NOT touch.** The same list as task 1, plus the other evaluator files.

**Acceptance.** Same as task 1, for n in {48, 50}. Also, a test documents any place where the
OpenEvolve and AlphaEvolve definitions differ (normalization, bounding box, n), and the docstring says which one we follow.

## Task 4: FullSleep stages + replay tool

**Goal.** Implement `sleep/full.py` (`prioritize`, `replay`, `abstract`, `prune`, `integrate`; the
docstrings are the spec). Add a replay tool that runs probe code against an **evaluator snapshot**.

- `sleep/replay.py`: `ReplayTool(evaluator_factory, max_probes, timeout_s)`. It builds its own evaluator
  instance from `make_evaluator(problem_id, ...)` (the snapshot), so probes cannot touch the run's evaluator
  state. Probes run through `run_program` / `evaluate`, or on Modal when `eval_backend == "modal"`.
- `FullSleep` needs access to the replay tool. Pass it via constructor params, not via the `sleep()`
  signature. `make_policy` may be extended to inject it.
- Probe results are evidence only. They are not added to `ProgramDB`, not logged as `"step"` events,
  and not shown in the agent's last-K window. Record them in the sleep event's `details`
  (`{"policy", "block", "probes": [...], "replay_evals": n}`).
- All LLM calls use `role="sleep"`. `BudgetExhausted` inside any stage propagates, and the loop stops the run.
- Block length: `integrate` calls the LLM with `max_tokens=max_block_tokens`.

**Touch.** `sleep/full.py`, `sleep/replay.py` (new), `sleep/__init__.py`, `harness/loop.py` (only to put the
returned probe details into the sleep event; keep the change minimal), `tests/test_full_sleep.py`.

**Do NOT touch.** `sleep/base.py`, `harness/budget.py`, `llm/`, `evaluators/`.

**Acceptance (mock LLM + mock evaluator).**
- A run with `full_sleep` ends with `budget.used <= total` (`overshoot` at most one call) and `sleep_tokens > 0`.
- No `"step"` event or `ProgramDB` entry comes from a probe. Probe counts appear in `details`.
- A probe that hangs is killed by the timeout and recorded as a failed probe; the run continues.
- Each stage has a unit test with a scripted `MockTransport`.

## Task 5: Masked-prediction check

**Goal.** Measure how much of the history a consolidated block retains. For a finished run, pick a
sleep event at step `s`. Hide a subset of the step events that sleep covered (configurable fraction,
seeded), give a predictor LLM the block plus the unhidden steps, and ask it to predict each hidden
step's outcome: `valid`, a score bucket (quantiles of the run's valid scores), and an error class
(`timeout`, `exception`, `constraint`, `no_code`). Score the predictions against the truth.

- Offline: reads `history.jsonl`. It never runs inside an experiment and never uses the run's Budget.
  It uses its own `LLMClient(Budget(eval_budget), ...)`, and reports its token use separately.
- Baselines in the same report: majority-class, and "no block" (unhidden steps only).
- Output: `masked_prediction.json` per run with per-field accuracy, score-bucket accuracy and MAE,
  N hidden, and tokens.

**Touch.** `analysis/masked_prediction.py` (new package `analysis/`), `scripts/masked_prediction.py`,
`tests/test_masked_prediction.py`.

**Do NOT touch.** `harness/`, `sleep/`, `llm/`, `evaluators/`.

**Acceptance.**
- On a synthetic history, a scripted mock predictor that returns the truth scores 1.0, and the majority baseline matches a hand-computed value.
- Hidden steps never appear in the predictor's prompt (asserted on the mock's recorded messages).
- The same seed gives the same hidden set.

## Task 6: past.dev recall arm

**Goal.** Add a sleep policy `pastdev` that uses past.dev as external memory: ingest every step,
recall at wake, one identity per run, and `queryTimestamp` for leak-free checks. Follow past.dev's
current API docs (do not guess endpoints). Configure it via env vars `PASTDEV_API_KEY` (and
`PASTDEV_BASE_URL` if needed), and add them to `.env.example`.

- **Ingest.** On each `should_sleep(step, history)` call, push the step events that are not yet
  ingested (the rendered step plus score and validity) under identity `run_id`. This needs no
  contract change, because `should_sleep` sees the history every step.
- **Recall at wake.** When the policy decides to sleep, query past.dev for memories relevant to the
  task and current best, with `queryTimestamp` set to the timestamp of the latest ingested step, so
  nothing ingested later (e.g. from a re-run) can leak in. Format the result as the new block.
- **One identity per run.** The identity is derived from `run_id`. Runs never share or reuse one.
- **Matched budget.** Any LLM call we make (e.g. formatting the block) uses `llm.complete(role="sleep")`.
  If past.dev spends model tokens server-side, they cannot go through our Budget. Record whatever
  usage past.dev reports in the sleep event `details`, and add a README section that flags this
  arm as not strictly budget-matched unless that usage is accounted for.
- HTTP via `httpx` (already installed as an openai dependency) or the stdlib. Network errors must not
  crash the run: on failure, keep the old block and log the error in `details`.

**Touch.** `sleep/pastdev.py` (new), `sleep/__init__.py`, `.env.example`, `README.md` (new section only),
`tests/test_pastdev.py`.

**Do NOT touch.** `sleep/base.py`, `harness/`, `llm/`, `evaluators/`.

**Acceptance (HTTP mocked).**
- Every step is ingested exactly once, under one identity per run.
- Every recall request carries a `queryTimestamp` no later than the latest ingested step.
- A simulated past.dev outage leaves the block unchanged and the run finishes.

## Task 7: Background (non-blocking) sleep

**Goal.** Let the agent keep working while consolidation runs. When a sleep triggers at step `s`,
a consolidator copies a **snapshot** of the history up to `s` and runs `policy.sleep` on it in a
background thread. The agent keeps taking steps with the old block. When the consolidator finishes,
the new block is swapped in **at the next step boundary** (never mid-prompt).

- Config flag `background_sleep: bool = false` in `RunConfig`. The blocking path stays the default and unchanged.
- Sleep tokens are charged to the same Budget (`Budget` is already thread-safe). `BudgetExhausted`
  raised in the background thread must stop the run cleanly at the next boundary.
- At most one consolidation in flight. If a trigger arrives while one runs, skip it and record that.
- The sleep event is logged at swap time with `details: {"policy", "block", "trigger_step", "swap_step"}`.
- At the end of the run, wait for an in-flight consolidation or cancel it, and record which happened.

**Touch.** `harness/background.py` (new), `harness/loop.py`, `harness/config.py` (new field only),
`tests/test_background_sleep.py`.

**Do NOT touch.** `harness/budget.py`, `sleep/base.py`, `llm/`, `evaluators/`.

**Acceptance (mock transport with an artificial delay on sleep calls).**
- Agent steps continue while a sleep is in flight: `swap_step > trigger_step`.
- The block in each agent prompt is constant within a step and changes only at a boundary.
- The consolidator never sees steps after `trigger_step` (asserted on its prompt).
- `budget.used <= total` (overshoot at most one call) and agent tokens + sleep tokens = used.
- All existing loop tests still pass with `background_sleep: false`.

## Task 8: Analysis

**Goal.** Tools to compare policies across runs.

- `analysis/load.py`: load `runs/*/summary.json` and `history.jsonl` into tidy tables (numpy/stdlib; pandas
  allowed only in an optional `analysis` dependency group).
- Curves: best-so-far `normalized_score` (use `evaluators.base.normalized_score`) against agent steps
  and against cumulative total tokens (agent + sleep), per run, then mean ± 95% CI per (problem, policy).
- Steps (and tokens) to reach 95% and 99% of `best_known`, per run. "Not reached" is censored, not
  dropped, and reported as such.
- Paired comparisons: pair runs by (problem, seed) across two policies. Report the mean difference in
  final normalized score and in steps-to-95%, with a paired bootstrap CI and a sign test (numpy only).
- Notebooks in `analysis/notebooks/` that only call these functions and plot (matplotlib in the optional group).

**Touch.** `analysis/` (new), `pyproject.toml` (optional `analysis` dependency group only), `tests/test_analysis.py`.

**Do NOT touch.** `harness/`, `sleep/`, `llm/`, `evaluators/`, `infra/`.

**Acceptance.**
- On synthetic run directories with known curves, steps-to-95%/99% and the paired differences match hand-computed values.
- Censoring: a run that never reaches 95% is reported as censored, not as `max_steps`.
- Minimization problems (negative `best_known`) give normalized curves that rise toward 1.0.
