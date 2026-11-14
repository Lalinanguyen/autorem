"""Fatigue: when has the agent's accumulated context started hurting its performance?

Modeled on the two-process model of human sleep:
  load        homeostatic pressure. Rises with the AGENT tokens consumed since the last sleep;
              a sleep of depth d removes the fraction `load_relief[d]` of the accumulated tokens.
  impairment  measurable decline relative to the agent's own RESTED BASELINE, which is
              measured over its first `baseline_steps` agent steps after each wake (and after
              run start). Until the baseline exists, impairment is 0.

Impairment components (each in [0, 1], computed over the last `window` agent steps since wake;
None when the log lacks the data, e.g. no EXPECTED_SCORE header):
  calibration    mean |expected_score - score| / scale, as a ratio to the baseline gap
                 -> (ratio - 1) / (calibration_ratio_cap - 1). The main signal.
  perseveration  fraction of window steps whose approach tags match an EARLIER FAILED step
                 of the raw run log (overlap |tags & failed| / |failed| >= perseveration_match),
                 as excess over the baseline fraction: (x - b) / (1 - b).
  stagnation     agent steps since the best valid score last improved / stagnation_steps_norm.
  sloppiness     rate of invalid / timeout / parse-failure steps, as a ratio to baseline
                 -> (ratio - 1) / (sloppiness_ratio_cap - 1).
  narrowing      mix of (a) drop in approach-tag diversity (unique / mentions) vs baseline and
                 (b) rise in mean pairwise token-set Jaccard similarity of window programs.
  bloat          min(bloat_cap, context_tokens / context_soft_cap) / bloat_cap.
impairment = weighted mean of the available components (weights in `impairment_weights`).

fatigue = load_weight * load + impairment_weight * impairment + debt

Levels (rested < drowsy < tired < exhausted) from thresholds, with
  grace       always "rested" for the first `grace_steps` agent steps after a wake
  hysteresis  Schmitt trigger per level (enter at T, leave below T - hysteresis_margin), and after
              a sleep the agent cannot reach tired/exhausted again until fatigue has dropped below
              `wake_threshold` (or `rearm_max_steps` steps have passed): the threshold to fall
              asleep is higher than the level it must drop below after sleep.
  debt        `on_deferred()` adds `debt_increment` (capped); debt is added to fatigue, paid off by
              `debt_relief[depth]` on sleep, and bumps `recommended_depth()` one level deeper.

This module only MEASURES. It makes no LLM calls and never decides to sleep: policies read
`state` / `recommended_depth()` and tell the monitor what happened via on_sleep / on_deferred.
It consumes the standard JSONL step records (harness/logging.py), so it computes identically
for every arm and can be replayed offline over a history.jsonl (`replay_history`).

Online call order per agent step (replay_history mirrors it):
    state = monitor.observe(record)        # record = the step's JSONL dict (agent_tokens cumulative)
    ... policy reads state, decides ...
    monitor.on_sleep(depth) | monitor.on_deferred()
    log step with monitor.log_fields()     # observed values + sleep_triggered/deferred/depth
    (sleep event line, if any, with monitor.log_fields(monitor.current()))
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, model_validator

COMPONENTS = ("calibration", "perseveration", "stagnation", "sloppiness", "narrowing", "bloat")


class Level(IntEnum):
    RESTED = 0
    DROWSY = 1
    TIRED = 2
    EXHAUSTED = 3

    @property
    def label(self) -> str:
        return self.name.lower()


class SleepDepth(IntEnum):
    NONE = 0
    FORGET = 1      # code-only forgetting, zero LLM calls (drowsy response)
    NREM = 2        # prioritize, replay, integrate, decay, evict
    NREM_REM = 3    # NREM followed by REM recombination

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: "SleepDepth | int | str | None") -> "SleepDepth":
        if value is None:
            return cls.NONE
        if isinstance(value, SleepDepth):
            return value
        if isinstance(value, int):
            return cls(value)
        key = str(value).strip().lower().replace("+", "_").replace("-", "_")
        try:
            return cls[key.upper()]
        except KeyError:
            raise ValueError(f"unknown sleep depth {value!r}; known: {[d.label for d in cls]}") from None


# ---------------------------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------------------------


class ImpairmentWeights(BaseModel):
    """Relative weights of the impairment components (normalized at use; need not sum to 1)."""

    calibration: float = Field(0.35, ge=0)
    perseveration: float = Field(0.15, ge=0)
    stagnation: float = Field(0.15, ge=0)
    sloppiness: float = Field(0.15, ge=0)
    narrowing: float = Field(0.10, ge=0)
    bloat: float = Field(0.10, ge=0)

    model_config = {"extra": "forbid"}

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in COMPONENTS}


class DepthFractions(BaseModel):
    """A fraction in [0, 1] per sleep depth (none is always 0)."""

    forget: float = Field(ge=0, le=1)
    nrem: float = Field(ge=0, le=1)
    nrem_rem: float = Field(ge=0, le=1)

    model_config = {"extra": "forbid"}

    def of(self, depth: SleepDepth) -> float:
        return 0.0 if depth == SleepDepth.NONE else float(getattr(self, depth.label))


class FatigueValidationConfig(BaseModel):
    """Settings for the offline predictive check (analysis/fatigue_check.py)."""

    horizon_k: int = Field(5, gt=0, description="label: no improvement or an invalid step in the next k steps")
    auc_margin: float = Field(0.05, ge=0, le=0.5, description="flag when |AUC - 0.5| < auc_margin")
    arm: str = Field("no_sleep", description="arm (sleep_policy) the check runs on")
    min_samples: int = Field(10, gt=0, description="fewer labelled steps per problem -> AUC not reported")

    model_config = {"extra": "forbid"}


class FatigueConfig(BaseModel):
    """Every threshold, weight, window, cap and grace length of the fatigue system."""

    model_config = {"extra": "forbid"}

    enabled: bool = True
    skip_seed_step: bool = Field(True, description="step 0 (seed, no LLM call) is not an agent step")
    round_digits: int = Field(6, ge=0, description="rounding of logged floats")

    # ---- load (homeostatic pressure) ----
    load_tokens_scale: int = Field(100_000, gt=0, description="agent tokens since sleep that give load 1.0 (linear)")
    load_budget_fraction: float | None = Field(
        None, gt=0, description="if set and the monitor knows the total budget: scale = fraction * total")
    load_curve: Literal["linear", "saturating"] = Field(
        "linear", description="linear: t/scale; saturating: 1 - exp(-t/scale)")
    load_cap: float = Field(1.0, gt=0)
    load_relief: DepthFractions = Field(
        default_factory=lambda: DepthFractions(forget=0.2, nrem=0.6, nrem_rem=0.9),
        description="fraction of accumulated load tokens removed by a sleep of each depth")

    # ---- baseline and window ----
    baseline_steps: int = Field(5, gt=0, description="agent steps after each wake that form the rested baseline")
    window: int = Field(5, gt=0, description="rolling window (agent steps since wake) for impairment")
    forget_resets_baseline: bool = Field(False, description="does a FORGET-depth response count as a wake")
    improvement_min_delta: float = Field(0.0, ge=0, description="score must beat the best by more than this")

    # ---- calibration ----
    calibration_floor: float = Field(0.01, gt=0, description="minimum baseline gap (avoids /0)")
    calibration_ratio_cap: float = Field(4.0, gt=1, description="gap ratio that maps to component 1.0")
    calibration_invalid_gap: float | None = Field(
        None, ge=0, description="gap charged for an invalid step with a prediction; None = skip invalid steps")
    calibration_score_scale: float | None = Field(
        None, gt=0, description="divide gaps by this; None = |best_known| if known, else 1")
    calibration_min_samples: int = Field(2, gt=0, description="min predictions in baseline/window")

    # ---- perseveration ----
    perseveration_match: float = Field(0.5, gt=0, le=1, description="|tags & failed| / |failed| to count as a match")
    perseveration_failure: Literal["invalid", "no_improvement"] = Field(
        "no_improvement", description="what counts as a failed step (no_improvement includes invalid)")
    perseveration_lookback: int | None = Field(
        None, gt=0, description="compare against at most this many earlier failed steps; None = all")

    # ---- stagnation ----
    stagnation_steps_norm: int = Field(20, gt=0, description="steps since improvement that give 1.0")
    stagnant_threshold: float = Field(0.75, ge=0, le=1, description="stagnation >= this = 'stagnant'")

    # ---- sloppiness ----
    sloppiness_floor: float = Field(0.05, gt=0, le=1, description="minimum baseline failure rate (avoids /0)")
    sloppiness_ratio_cap: float = Field(4.0, gt=1)
    parse_failure_patterns: list[str] = Field(
        default_factory=lambda: ["code block"], description="error substrings (case-insensitive) = parse failure")
    timeout_patterns: list[str] = Field(default_factory=lambda: ["timeout", "timed out"])
    header_expected: Literal["auto", "always", "never"] = Field(
        "auto", description="missing awake header = parse failure; auto: once any step carried one")

    # ---- narrowing ----
    narrowing_tag_weight: float = Field(0.5, ge=0)
    narrowing_similarity_weight: float = Field(0.5, ge=0)
    narrowing_min_samples: int = Field(2, ge=2, description="min tagged steps / programs to compute a part")
    similarity_token_regex: str = Field(r"\w+", description="token pattern for program Jaccard similarity")

    # ---- bloat ----
    context_soft_cap: int | None = Field(8000, gt=0, description="None = bloat not measured")
    bloat_cap: float = Field(1.0, gt=0, description="context/soft_cap ratio that gives bloat 1.0")

    # ---- mix ----
    impairment_weights: ImpairmentWeights = Field(default_factory=ImpairmentWeights)
    renormalize_missing: bool = Field(True, description="unavailable components drop out of the weighted mean")
    load_weight: float = Field(0.4, ge=0)
    impairment_weight: float = Field(0.6, ge=0)

    # ---- levels ----
    drowsy_threshold: float = Field(0.3, ge=0)
    tired_threshold: float = Field(0.5, ge=0)
    exhausted_threshold: float = Field(0.75, ge=0)
    hysteresis_margin: float = Field(0.05, ge=0, description="leave a level only below its threshold - margin")
    wake_threshold: float = Field(0.25, ge=0, description="after sleep, must drop below this to re-arm tired+")
    rearm_max_steps: int | None = Field(20, gt=0, description="re-arm anyway after this many awake steps")
    grace_steps: int = Field(3, ge=0, description="always rested for this many agent steps after waking")

    # ---- sleep debt ----
    debt_increment: float = Field(0.1, ge=0, description="debt added per deferred sleep")
    debt_cap: float = Field(0.5, ge=0)
    debt_relief: DepthFractions = Field(
        default_factory=lambda: DepthFractions(forget=0.0, nrem=0.5, nrem_rem=1.0),
        description="fraction of debt paid off by a sleep of each depth")
    debt_deepen_threshold: float = Field(1e-9, ge=0, description="debt >= this makes the next sleep one level deeper")

    # ---- offline replay of logs without explicit fatigue fields ----
    offline_default_depth: Literal["forget", "nrem", "nrem_rem"] = Field(
        "nrem", description="depth assumed for a sleep event without sleep_depth/sleep_mode")

    validation: FatigueValidationConfig = Field(default_factory=FatigueValidationConfig)

    @model_validator(mode="after")
    def _check_order(self) -> "FatigueConfig":
        if not (self.drowsy_threshold < self.tired_threshold < self.exhausted_threshold):
            raise ValueError("need drowsy_threshold < tired_threshold < exhausted_threshold")
        if self.wake_threshold >= self.tired_threshold:
            raise ValueError("wake_threshold must be below tired_threshold (hysteresis)")
        if self.load_weight + self.impairment_weight <= 0:
            raise ValueError("load_weight + impairment_weight must be positive")
        re.compile(self.similarity_token_regex)
        return self

    def thresholds(self) -> dict[Level, float]:
        return {Level.DROWSY: self.drowsy_threshold, Level.TIRED: self.tired_threshold,
                Level.EXHAUSTED: self.exhausted_threshold}


# ---------------------------------------------------------------------------------------------
# State and log fields
# ---------------------------------------------------------------------------------------------

FATIGUE_FIELDS = (
    "fatigue", "fatigue_load", "fatigue_impairment",
    *(f"fatigue_{c}" for c in COMPONENTS),
    "fatigue_level", "fatigue_debt", "fatigue_grace", "fatigue_baseline_ready",
    "sleep_triggered", "sleep_deferred", "sleep_depth",
)
"""JSONL fields added to every line (null when fatigue is disabled). harness/logging.py appends these to FIELDS."""


@dataclass(frozen=True)
class FatigueState:
    step: int | None
    fatigue: float
    load: float
    impairment: float
    components: dict[str, float | None]
    level: Level
    debt: float
    grace: bool
    baseline_ready: bool
    awake_steps: int          # agent steps since the last wake
    armed: bool               # may reach tired/exhausted (post-sleep hysteresis)
    load_tokens: float        # accumulated agent tokens behind `load`
    triggered: bool = False   # a sleep (depth >= forget) was taken after this step
    deferred: bool = False    # a sleep was due but deferred after this step
    depth: SleepDepth = SleepDepth.NONE

    def to_log(self, digits: int = 6) -> dict[str, Any]:
        r = (lambda x: None if x is None else round(float(x), digits))
        out: dict[str, Any] = {
            "fatigue": r(self.fatigue), "fatigue_load": r(self.load), "fatigue_impairment": r(self.impairment),
        }
        for c in COMPONENTS:
            out[f"fatigue_{c}"] = r(self.components.get(c))
        out.update(
            fatigue_level=self.level.label, fatigue_debt=r(self.debt), fatigue_grace=self.grace,
            fatigue_baseline_ready=self.baseline_ready, sleep_triggered=self.triggered,
            sleep_deferred=self.deferred, sleep_depth=self.depth.label if self.triggered else None,
        )
        return out


@dataclass(frozen=True)
class FatigueReading:
    """Read-only view handed to policies before should_sleep (loop hook `set_fatigue(reading)`)."""

    state: FatigueState
    recommended_depth: SleepDepth
    stagnant: bool
    budget_remaining: int | None = None

    @property
    def level(self) -> Level:
        return self.state.level


def null_log_fields() -> dict[str, None]:
    return {k: None for k in FATIGUE_FIELDS}


# ---------------------------------------------------------------------------------------------
# Helpers (pure)
# ---------------------------------------------------------------------------------------------


def _clip01(x: float) -> float:
    return 0.0 if x <= 0 else 1.0 if x >= 1 else float(x)


def _finite(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x) if math.isfinite(x) else None


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def tag_overlap(tags: Iterable[str], failed: Iterable[str]) -> float:
    a, b = set(tags), set(failed)
    return len(a & b) / len(b) if b else 0.0


def tag_diversity(tag_lists: list[list[str]]) -> float | None:
    mentions = [t for tags in tag_lists for t in tags]
    return len(set(mentions)) / len(mentions) if mentions else None


def program_similarity(token_sets: list[frozenset]) -> float | None:
    """Mean pairwise Jaccard similarity of token sets (None for fewer than two)."""
    n = len(token_sets)
    if n < 2:
        return None
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = token_sets[i], token_sets[j]
            u = len(a | b)
            total += (len(a & b) / u) if u else 1.0
            pairs += 1
    return total / pairs


def ratio_component(recent: float, base: float, floor: float, cap: float) -> float:
    return _clip01((recent / max(base, floor) - 1.0) / (cap - 1.0))


def excess_component(recent: float, base: float) -> float:
    return _clip01((recent - base) / max(1.0 - base, 1e-9))


@dataclass
class _Step:
    """Per-agent-step facts extracted once from a record."""

    step: int
    gap: float | None
    failed: bool              # for perseveration
    sloppy: bool
    tags: list[str]
    persev: bool | None       # matched an earlier failed step (None: no tags)
    tokens: frozenset | None
    context_tokens: float | None


@dataclass
class _Baseline:
    gap: float | None = None
    sloppy: float = 0.0
    persev: float | None = None
    diversity: float | None = None
    similarity: float | None = None


# ---------------------------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------------------------


class FatigueMonitor:
    """Pure measurement over standard JSONL step records. See the module docstring."""

    def __init__(self, cfg: FatigueConfig | dict | None = None, *, best_known: float | None = None,
                 total_budget: int | None = None):
        self.cfg = cfg if isinstance(cfg, FatigueConfig) else FatigueConfig(**(cfg or {}))
        self.best_known = best_known
        c = self.cfg
        if c.load_budget_fraction is not None and total_budget:
            self.load_scale = max(1.0, c.load_budget_fraction * total_budget)
        else:
            self.load_scale = float(c.load_tokens_scale)
        if c.calibration_score_scale is not None:
            self.gap_scale = c.calibration_score_scale
        else:
            self.gap_scale = abs(best_known) if best_known else 1.0
        self._token_re = re.compile(c.similarity_token_regex)
        w = c.impairment_weights.as_dict()
        self._weights = w

        self._prev_agent_tokens = 0.0
        self._load_tokens = 0.0
        self._debt = 0.0
        self._best: float | None = None
        self._agent_steps_total = 0
        self._last_improve_idx: int | None = None  # agent-step index of the last improvement
        self._failed_tags: list[list[str]] = []
        self._header_seen = False
        self._awake: list[_Step] = []             # agent steps since the last wake
        self._baseline: _Baseline | None = None
        self._level = Level.RESTED
        self._armed = True
        self._last_context: float | None = None
        self._state: FatigueState = self._snapshot(None, {c: 0.0 for c in COMPONENTS}, 0.0)
        self.sleeps: list[tuple[int | None, SleepDepth]] = []
        self.deferrals: list[int | None] = []

    # ---- public API -----------------------------------------------------------------------
    @property
    def state(self) -> FatigueState:
        """State observed at the last step, with that step's sleep decision flags."""
        return self._state

    def current(self) -> FatigueState:
        """Live state now (reflects on_sleep / on_deferred since the last observe)."""
        comps = {c: 0.0 for c in COMPONENTS} if not self.baseline_ready else self._state.components
        imp = 0.0 if not self.baseline_ready else self._state.impairment
        st = self._snapshot(self._state.step, comps, imp)
        return replace(st, triggered=self._state.triggered, deferred=self._state.deferred, depth=self._state.depth)

    @property
    def baseline_ready(self) -> bool:
        return self._baseline is not None

    @property
    def load(self) -> float:
        t = self._load_tokens / self.load_scale
        raw = t if self.cfg.load_curve == "linear" else 1.0 - math.exp(-t)
        return min(self.cfg.load_cap, raw)

    @property
    def debt(self) -> float:
        return self._debt

    def is_stagnant(self, state: FatigueState | None = None) -> bool:
        s = state or self._state
        v = s.components.get("stagnation")
        return v is not None and s.baseline_ready and v >= self.cfg.stagnant_threshold

    def recommended_depth(self, state: FatigueState | None = None) -> SleepDepth:
        """Depth matching the impairment level (policies may use or ignore it).

        drowsy -> forget, tired -> nrem, exhausted or tired-while-stagnant -> nrem_rem;
        outstanding debt makes any sleep one level deeper (capped at nrem_rem).
        """
        s = state or self._state
        if s.level == Level.RESTED:
            return SleepDepth.NONE
        depth = {Level.DROWSY: SleepDepth.FORGET, Level.TIRED: SleepDepth.NREM,
                 Level.EXHAUSTED: SleepDepth.NREM_REM}[s.level]
        if s.level == Level.TIRED and self.is_stagnant(s):
            depth = SleepDepth.NREM_REM
        if s.debt >= self.cfg.debt_deepen_threshold and s.debt > 0:
            depth = SleepDepth(min(int(depth) + 1, int(SleepDepth.NREM_REM)))
        return depth

    def reading(self, budget_remaining: int | None = None) -> FatigueReading:
        s = self._state
        return FatigueReading(s, self.recommended_depth(s), self.is_stagnant(s), budget_remaining)

    def observe(self, record: dict) -> FatigueState:
        """Consume one step record (a JSONL dict). Non-step records are ignored (state returned)."""
        if record.get("event", "step") != "step":
            return self._state
        c = self.cfg
        step = record.get("step")
        agent_tokens = _finite(record.get("agent_tokens"))
        if agent_tokens is not None:
            self._load_tokens += max(0.0, agent_tokens - self._prev_agent_tokens)
            self._prev_agent_tokens = agent_tokens
        ctx = _finite(record.get("context_tokens"))
        if ctx is not None:
            self._last_context = ctx

        valid = bool(record.get("valid"))
        score = _finite(record.get("score")) if valid else None
        improved = score is not None and (self._best is None or score > self._best + c.improvement_min_delta)
        is_seed = c.skip_seed_step and step == 0

        if is_seed:
            if improved:
                self._best = score
            self._state = self._snapshot(step, self._components_or_zero(), self._impairment_now())
            return self._state

        self._agent_steps_total += 1
        if improved:
            self._last_improve_idx = self._agent_steps_total
        facts = self._extract(record, valid, score, improved)
        if improved:
            self._best = score
        if facts.failed and facts.tags:
            self._failed_tags.append(facts.tags)

        self._awake.append(facts)
        if self._baseline is None and len(self._awake) >= c.baseline_steps:
            self._baseline = self._measure_baseline(self._awake[: c.baseline_steps])

        comps = self._components()
        imp = self._impairment(comps)
        self._state = self._snapshot(step, comps, imp, update_level=True)
        return self._state

    def on_sleep(self, depth: SleepDepth | int | str) -> FatigueState:
        """The policy took a sleep of `depth` after the last observed step."""
        d = SleepDepth.parse(depth)
        if d == SleepDepth.NONE:
            return self.current()
        c = self.cfg
        self._load_tokens *= 1.0 - c.load_relief.of(d)
        self._debt *= 1.0 - c.debt_relief.of(d)
        if d >= SleepDepth.NREM or c.forget_resets_baseline:
            self._awake = []
            self._baseline = None
            self._level = Level.RESTED
            # post-sleep hysteresis: re-armed only once fatigue is below the wake threshold
            self._armed = self._fatigue(0.0) < c.wake_threshold
        self._state = replace(self._state, triggered=True, depth=max(d, self._state.depth))
        self.sleeps.append((self._state.step, d))
        return self.current()

    def on_deferred(self) -> FatigueState:
        """A sleep was due after the last observed step but deferred (e.g. too little budget left)."""
        self._debt = min(self.cfg.debt_cap, self._debt + self.cfg.debt_increment)
        self._state = replace(self._state, deferred=True)
        self.deferrals.append(self._state.step)
        return self.current()

    def log_fields(self, state: FatigueState | None = None) -> dict[str, Any]:
        """The exact JSONL fields (FATIGUE_FIELDS) for `state` (default: the last observed step)."""
        if not self.cfg.enabled:
            return null_log_fields()
        return (state or self._state).to_log(self.cfg.round_digits)

    # ---- internals ------------------------------------------------------------------------
    def _extract(self, record: dict, valid: bool, score: float | None, improved: bool) -> _Step:
        c = self.cfg
        tags = [str(t) for t in (record.get("approach_tags") or [])]
        expected = _finite(record.get("expected_score"))
        if tags or expected is not None:
            self._header_seen = True

        gap = None
        if expected is not None:
            if score is not None:
                gap = abs(score - expected) / self.gap_scale
            elif c.calibration_invalid_gap is not None:
                gap = c.calibration_invalid_gap

        err = (record.get("error") or "").lower()
        parse_fail = any(p.lower() in err for p in c.parse_failure_patterns)
        timeout = any(p.lower() in err for p in c.timeout_patterns)
        header_missing = not tags and expected is None and (
            c.header_expected == "always" or (c.header_expected == "auto" and self._header_seen))
        sloppy = (not valid) or parse_fail or timeout or header_missing

        failed = (not valid) if c.perseveration_failure == "invalid" else (not valid or not improved)
        persev = None
        if tags:
            pool = self._failed_tags if c.perseveration_lookback is None else self._failed_tags[-c.perseveration_lookback:]
            persev = any(tag_overlap(tags, f) >= c.perseveration_match for f in pool)

        prog = record.get("program")
        tokens = frozenset(self._token_re.findall(prog)) if isinstance(prog, str) and prog.strip() else None
        return _Step(step=record.get("step") or 0, gap=gap, failed=failed, sloppy=sloppy, tags=tags,
                     persev=persev, tokens=tokens, context_tokens=_finite(record.get("context_tokens")))

    def _stats(self, steps: list[_Step]) -> _Baseline:
        c = self.cfg
        gaps = [s.gap for s in steps if s.gap is not None]
        pv = [s.persev for s in steps if s.persev is not None]
        tagged = [s.tags for s in steps if s.tags]
        toks = [s.tokens for s in steps if s.tokens is not None]
        return _Baseline(
            gap=_mean(gaps) if len(gaps) >= c.calibration_min_samples else None,
            sloppy=sum(s.sloppy for s in steps) / len(steps) if steps else 0.0,
            persev=(sum(pv) / len(pv)) if pv else None,
            diversity=tag_diversity(tagged) if len(tagged) >= c.narrowing_min_samples else None,
            similarity=program_similarity(toks) if len(toks) >= c.narrowing_min_samples else None,
        )

    def _measure_baseline(self, steps: list[_Step]) -> _Baseline:
        return self._stats(steps)

    def _components(self) -> dict[str, float | None]:
        if self._baseline is None:
            return {k: 0.0 for k in COMPONENTS}
        c, b = self.cfg, self._baseline
        win = self._awake[-c.window:]
        r = self._stats(win)

        cal = None
        if r.gap is not None and b.gap is not None:
            cal = ratio_component(r.gap, b.gap, c.calibration_floor, c.calibration_ratio_cap)
        persev = excess_component(r.persev, b.persev) if (r.persev is not None and b.persev is not None) else None
        since = self._agent_steps_total - (self._last_improve_idx or 0)
        stag = _clip01(since / c.stagnation_steps_norm)
        slop = ratio_component(r.sloppy, b.sloppy, c.sloppiness_floor, c.sloppiness_ratio_cap)

        parts: list[tuple[float, float]] = []
        if r.diversity is not None and b.diversity is not None and c.narrowing_tag_weight > 0:
            parts.append((c.narrowing_tag_weight, _clip01((b.diversity - r.diversity) / max(b.diversity, 1e-9))))
        if r.similarity is not None and b.similarity is not None and c.narrowing_similarity_weight > 0:
            parts.append((c.narrowing_similarity_weight, excess_component(r.similarity, b.similarity)))
        narrow = (sum(w * v for w, v in parts) / sum(w for w, _ in parts)) if parts else None

        bloat = None
        if c.context_soft_cap is not None and self._last_context is not None:
            bloat = min(c.bloat_cap, self._last_context / c.context_soft_cap) / c.bloat_cap

        return {"calibration": cal, "perseveration": persev, "stagnation": stag, "sloppiness": slop,
                "narrowing": narrow, "bloat": bloat}

    def _components_or_zero(self) -> dict[str, float | None]:
        return self._state.components if self.baseline_ready else {k: 0.0 for k in COMPONENTS}

    def _impairment_now(self) -> float:
        return self._state.impairment if self.baseline_ready else 0.0

    def _impairment(self, comps: dict[str, float | None]) -> float:
        num = den = 0.0
        for k, v in comps.items():
            w = self._weights[k]
            if w <= 0:
                continue
            if v is None:
                if self.cfg.renormalize_missing:
                    continue
                v = 0.0
            num += w * v
            den += w
        return num / den if den > 0 else 0.0

    def _fatigue(self, impairment: float) -> float:
        c = self.cfg
        return c.load_weight * self.load + c.impairment_weight * impairment + self._debt

    def _snapshot(self, step: int | None, comps: dict, imp: float, update_level: bool = False) -> FatigueState:
        f = self._fatigue(imp)
        if update_level:
            self._update_level(f)
        grace = len(self._awake) <= self.cfg.grace_steps if not (step == 0 and self.cfg.skip_seed_step) else True
        return FatigueState(
            step=step, fatigue=f, load=self.load, impairment=imp, components=dict(comps),
            level=self._level, debt=self._debt, grace=grace, baseline_ready=self.baseline_ready,
            awake_steps=len(self._awake), armed=self._armed, load_tokens=self._load_tokens,
        )

    def _update_level(self, f: float) -> None:
        c = self.cfg
        th = c.thresholds()
        up = Level.RESTED
        for lvl in (Level.DROWSY, Level.TIRED, Level.EXHAUSTED):
            if f >= th[lvl]:
                up = lvl
        prev = self._level
        if up >= prev:
            new = up
        else:  # dropping: stay at a level until fatigue falls below threshold - margin
            new = Level.RESTED
            for lvl in (Level.DROWSY, Level.TIRED, Level.EXHAUSTED):
                if lvl <= prev and f >= th[lvl] - c.hysteresis_margin:
                    new = lvl
        if not self._armed and (f < c.wake_threshold or
                                (c.rearm_max_steps is not None and len(self._awake) >= c.rearm_max_steps)):
            self._armed = True
        if not self._armed:
            new = min(new, Level.DROWSY)
        if len(self._awake) <= c.grace_steps:
            new = Level.RESTED
        self._level = Level(new)


# ---------------------------------------------------------------------------------------------
# Offline replay
# ---------------------------------------------------------------------------------------------


def _depth_from_sleep_event(rec: dict, cfg: FatigueConfig) -> SleepDepth:
    if rec.get("sleep_depth"):
        return SleepDepth.parse(rec["sleep_depth"])
    mode = rec.get("sleep_mode")
    if mode in ("rem", "nrem+rem"):
        return SleepDepth.NREM_REM
    if mode == "nrem":
        return SleepDepth.NREM
    return SleepDepth.parse(cfg.offline_default_depth)


def replay_history(history: Iterable[dict], cfg: FatigueConfig | dict | None = None, *,
                   best_known: float | None = None, total_budget: int | None = None) -> list[FatigueState]:
    """Recompute fatigue over a run log. Returns one FatigueState per step record (in order).

    Sleeps/deferrals: if the log carries the fatigue decision fields (sleep_triggered present on
    step lines), they are applied after each step exactly as online. Otherwise (older logs, other
    loops) each "sleep" event is applied with its sleep_depth / sleep_mode / offline_default_depth.
    """
    records = list(history)
    mon = FatigueMonitor(cfg, best_known=best_known, total_budget=total_budget)
    explicit = any(r.get("event", "step") == "step" and r.get("sleep_triggered") is not None for r in records)
    out: list[FatigueState] = []
    for rec in records:
        if rec.get("event", "step") == "step":
            mon.observe(rec)
            if explicit:
                if rec.get("sleep_deferred"):
                    mon.on_deferred()
                if rec.get("sleep_triggered"):
                    mon.on_sleep(rec.get("sleep_depth") or SleepDepth.parse(mon.cfg.offline_default_depth))
            out.append(mon.state)
        elif rec.get("event") == "sleep" and not explicit:
            mon.on_sleep(_depth_from_sleep_event(rec, mon.cfg))
            if out:
                out[-1] = replace(out[-1], triggered=True, depth=mon.state.depth)
    return out


__all__ = [
    "COMPONENTS", "FATIGUE_FIELDS", "DepthFractions", "FatigueConfig", "FatigueMonitor", "FatigueReading",
    "FatigueState",
    "FatigueValidationConfig", "ImpairmentWeights", "Level", "SleepDepth", "null_log_fields", "replay_history",
]
