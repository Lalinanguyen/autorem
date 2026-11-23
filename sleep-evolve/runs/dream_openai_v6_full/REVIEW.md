# dream_openai_v6_full — run review

- Branch/commit: `feat/sleep-evolve-v2` @ `961f31a` (edit_mode `full`, SEARCH copy-fidelity)
- Config: `configs/dream_openai.yaml` — problems 7, 36, 50, 59 × arms
  `no_sleep, fixed_letta, summarize, dream_unvalidated, dream` × seed 0, 250k-token budget each
- Launched 2026-11-22 ~16:12, finished ~17:26 local (~74 min), 4 parallel per-problem processes
  (`runs/_matrix_driver.py`, skips runs that already have `summary.json`)
- **20/20 runs completed, stop=budget_exhausted, no errors. Total cost: $1.399**
  (per-run $0.051–$0.091; sleep arms cost more — gpt-5-mini sleep calls)

## Final best normalized score (higher = better; vs AlphaEvolve reference ceiling)

| problem | no_sleep | fixed_letta | summarize | dream_unvalidated | dream |
|---|---|---|---|---|---|
| 7  | 0.739 | 0.990 | 0.990 | 0.990 | 0.880 |
| 36 | 0.927 | 0.913 | 0.822 | 0.942 | 0.902 |
| 50 | 0.996 | 0.997 | 0.981 | 0.937 | 0.728 |
| 59 | 0.688 | 0.688 | 0.759 | 0.750 | 0.768 |

## Headline

- **No arm is significantly better than any other** — every paired contrast is
  "not significant". With 4 pairs the exact Wilcoxon test cannot reach p<0.05
  (minimum 0.125); the suite is underpowered by construction at 1 seed.
- `dream` (validated REM) is **below** `dream_unvalidated` (median paired diff
  -0.075, W/T/L 1/0/3) — the A/B validation evals look like they cost more
  budget than confirmed discoveries return at this run length. Same direction
  as the guide's "validation is not load-bearing / costs more than it returns".
- `fixed_letta` ≈ `no_sleep` (+0.000, p=0.75): NREM-only sleep is neutral.
- Best per-problem: p7 → fixed_letta/summarize/dream_unvalidated tie 0.990;
  p36 → dream_unvalidated 0.942; p50 → fixed_letta 0.997 ≈ no_sleep 0.996;
  p59 → dream 0.768 (lowest-scoring problem overall).
- v5→v6 cell differences swing ±0.2 in both directions on the same
  (problem, arm) — e.g. p50_dream 0.954→0.728, p50_summarize 0.716→0.981,
  p7_no_sleep 0.862→0.739 — i.e. single-seed noise dominates any arm effect.

## Anomalies worth a look

- `dream_unvalidated` on p50 (0.937) and p59 (0.750) landed below both
  `no_sleep` and `fixed_letta` — check its history for dream spam eating steps.
- `summarize` on p36 regressed hard vs v5 (0.944→0.822, worst arm on p36).
- `p7_no_sleep` (0.739) is the only run far below its arm-mates — either a
  genuinely lucky sleep batch or a bad no_sleep rollout; diff vs v5's 0.862
  suggests variance, not a code regression (config drift check passed).
- `dream` sleep_evals ~17–18/run vs ~10 for `dream_unvalidated` (report.txt):
  validation replays are the budget sink.

## Files

- `report/` — compare_baselines output (report.txt, compare_baselines.json,
  4 figures: score_vs_budget, fatigue_curves, perseveration_sleep,
  sleep_count_vs_gain)
- `report_alphaevolve/` — alphaevolve_report.md (scored table vs the
  AlphaEvolve/OpenEvolve/GEPA references; all 20 runs scored, 0 excluded)
