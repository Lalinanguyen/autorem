"""FullSleep: structured-memory consolidation with pressure gating, NREM/REM modes and coupled replay.

One sleep (`consolidate`):
  1. choose mode (scheduler): NREM (verify) or REM (plateau: generalize; schema update)
  2. refresh the goal item (best score so far)
  3. ONE consolidator LLM call (role="sleep"): PRIORITIZE / EXTRACT CLAIMS / PLAN REPLAYS
     (sleep/consolidator_io.py), parsed robustly; malformed output = zero edits, logged
  4. coupled replay (sleep/replay.py): each planned replay runs via evaluator.evaluate and
     decides exactly one Edit, or is counted in replays_wasted
  5. claims without a decided replay become tentative edits per `relation`
  6. decay (items not touched this sleep), evict, optional REM schema abstraction
The agent then sees `memory_section(cap)`, the rendered store, never raw history.

Every mechanism has a flag (see DEFAULTS). All LLM calls go through llm.complete(role="sleep");
BudgetExhausted propagates to the caller.

Optional loop hooks implemented here: bind(evaluator=...), observe(step, record), last_metrics,
awake_tags, context_cap_tokens, set_fatigue(reading), last_depth, last_deferred, select_recent.

Gating (`gating` param):
  fatigue   (default) the loop's FatigueMonitor (harness/fatigue.py) sets the level; this policy
            only READS it (set_fatigue) and responds by level:
              drowsy                       -> forget(): code only, ZERO LLM calls: drop recent raw
                                              steps memory already covers, shrink recent steps to
                                              one-line records, strengthen dead ends just matched
              tired                        -> NREM consolidate()
              exhausted / tired + stagnant -> NREM then REM consolidate() (recombination)
            (the monitor's recommended_depth; debt makes it one level deeper; capped at max_depth).
            A sleep of depth >= nrem with less than `min_sleep_budget_tokens` left is DEFERRED
            (last_deferred; the monitor adds debt), falling back to forget if allowed.
            Without a reading (e.g. a caller that never sets one) it falls back to pressure/fixed.
  pressure  the pressure gate (sleep/pressure.py) + the NREM/REM scheduler, as before.
  fixed     every `every_n` steps (Letta-style "every N steps").
  schedule  an explicit list of steps (`sleep_steps`), computed before the run (e.g. matched to a
            fatigue run's sleep count by harness/baselines.py). Never sleeps elsewhere.
  Depth for fixed / schedule (`timed_depth`):
    scheduler  (default) consolidate() with the NREM/REM scheduler picking the mode
    nrem       always one NREM consolidation (no REM, no forget tier): the baseline arms
    fatigue    the fatigue monitor's recommended depth at that step (fixed timing, tiered depth;
               rested counts as forget), capped at max_depth; needs a fatigue reading, else nrem
After any sleep of depth >= nrem the raw history memory now covers is removed from the agent's
context for good: the agent sees the task, goals, goal-relevant memory (strength >=
wake_min_strength) and only the last `wake_recent_steps` steps from before the sleep.
"""

from __future__ import annotations

import json
import math
from typing import Any

from harness.context import count_tokens, fit_lines
from harness.fatigue import FatigueReading, SleepDepth
from sleep import pressure as P
from sleep.consolidator_io import (
    CONSOLIDATOR_SYSTEM, CONSOLIDATOR_USER, NREM_NOTE, REM_NOTE, parse_json_object, step_view,
    validate_consolidation,
)
from sleep.memory import Edit, MemoryStore
from sleep.replay import run_replays
from sleep.scheduler import choose_mode
from sleep.schema import SchemaStore

DEFAULTS: dict[str, Any] = {
    # fixed schedule (used when pressure is off)
    "every_n": 10,
    # memory rendering / agent context
    "max_block_tokens": 2048,      # cap on the rendered memory section (counter tokens)
    "context_cap_tokens": None,    # optional cap on the whole agent context; min() with RunConfig's
    "awake_tags": True,            # ask the agent for APPROACH_TAGS/EXPECTED_SCORE/... header
    # memory dynamics
    "decay": True, "decay_factor": 0.9,
    "eviction": True, "min_strength": 0.05, "max_items": 40,
    "initial_strength": 0.3, "verified_strength": 0.7, "support_gain": 0.15, "contradict_loss": 0.25,
    "max_item_chars": 300,
    # pressure gating
    "pressure": True, "pressure_threshold": 0.5, "min_gap": 5, "max_gap": 25,
    "stall_window": 10, "clutter_norm": 20, "pressure_weights": None, "dead_end_match": 0.5,
    # NREM/REM scheduler
    "scheduler": True, "plateau_window": 15, "late_budget_fraction": 0.8,
    # consolidator call
    "k_steps": 8, "rem_k_steps": 12, "max_history_steps": 40, "max_claims": 8,
    "consolidator_max_tokens": 2048,
    # coupled replay
    "replay": True, "replay_budget": 2, "rem_replay_budget": 1, "replay_rel_tol": 1e-6,
    # schema store (REM only)
    "schema": False, "schema_min_strength": 0.6, "schema_min_items": 3, "max_schemas": 4,
    "schema_max_tokens": 512, "schema_path": None, "schema_persist": False,
    # gating: fatigue (the loop's FatigueMonitor level) | pressure | fixed | schedule
    "gating": "fatigue",
    "sleep_steps": None,            # schedule gating: explicit steps after which to sleep
    "timed_depth": "scheduler",     # fixed/schedule depth: scheduler | nrem | fatigue
    "max_depth": "nrem_rem",        # cap on the depth this policy runs: forget | nrem | nrem_rem
    "respond_drowsy": True,         # drowsy -> code-only forgetting
    "forget_min_gap": 3,            # min steps between two forgets
    "forget_error_chars": 80,       # one-line records: error text cut to this many chars
    "forget_dead_end_gain": 0.1,    # strength added to a dead end matched since the last forget/sleep
    "min_sleep_budget_tokens": 4096,  # defer a sleep of depth >= nrem when less budget remains
    "defer_fallback_forget": True,  # when deferring, forget instead (zero LLM calls)
    # waking: after a sleep of depth >= nrem
    "wake_recent_steps": 2,         # pre-sleep raw steps still shown ("the last couple"); None = keep all
    "wake_min_strength": 0.1,       # goal-relevant memory: items below this strength are not shown
}

GATINGS = ("fatigue", "pressure", "fixed", "schedule")
TIMED_DEPTHS = ("scheduler", "nrem", "fatigue")
SUM_METRICS = ("items_added", "items_evicted", "replays_wasted", "replays_run", "claims", "edits_applied",
               "edits_rejected")

METRIC_FIELDS = ("pressure", "sleep_mode", "block_tokens", "items_added", "items_evicted", "replays_wasted")


class FullSleep:
    name = "full_sleep"

    def __init__(self, **params: Any):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown full_sleep params: {sorted(unknown)}; known: {sorted(DEFAULTS)}")
        self.p = {**DEFAULTS, **params}
        if self.p["every_n"] <= 0:
            raise ValueError("every_n must be positive")
        if self.p["gating"] not in GATINGS:
            raise ValueError(f"gating must be one of {GATINGS}")
        if self.p["timed_depth"] not in TIMED_DEPTHS:
            raise ValueError(f"timed_depth must be one of {TIMED_DEPTHS}")
        steps = self.p["sleep_steps"]
        if self.p["gating"] == "schedule" and steps is None:
            raise ValueError("gating=schedule needs sleep_steps (a list of steps)")
        self.sleep_steps: frozenset[int] = frozenset(int(x) for x in (steps or []))
        if any(x <= 0 for x in self.sleep_steps):
            raise ValueError("sleep_steps must be positive agent steps")
        self.max_depth = SleepDepth.parse(self.p["max_depth"])
        if self.max_depth == SleepDepth.NONE:
            raise ValueError("max_depth must be forget, nrem or nrem_rem")
        self.every_n = self.p["every_n"]
        self.max_block_tokens = self.p["max_block_tokens"]
        self.awake_tags = bool(self.p["awake_tags"])
        self.context_cap_tokens = self.p["context_cap_tokens"]
        self.store = MemoryStore(
            initial_strength=self.p["initial_strength"], verified_strength=self.p["verified_strength"],
            support_gain=self.p["support_gain"], contradict_loss=self.p["contradict_loss"],
            max_item_chars=self.p["max_item_chars"],
        )
        self.schemas = SchemaStore(
            min_strength=self.p["schema_min_strength"], min_items=self.p["schema_min_items"],
            max_schemas=self.p["max_schemas"], max_tokens=self.p["schema_max_tokens"],
            max_chars=self.p["max_item_chars"], path=self.p["schema_path"], persist=self.p["schema_persist"],
        ) if self.p["schema"] else None
        self.evaluator = None
        self.last_metrics: dict = {}
        self._last_sleep_step = -1
        self._seen: list[dict] = []  # slim copies of observed step records (for pressure)
        self._last_pressure: tuple[int, float] | None = None
        # fatigue response state
        self.fatigue: FatigueReading | None = None  # read-only, set by the loop each step
        self.last_depth: str | None = None          # depth actually run by the last sleep()
        self.last_deferred: bool = False            # should_sleep deferred a due sleep
        self._planned: SleepDepth | None = None     # depth chosen by should_sleep (fatigue / timed depth)
        self._reason = "fatigue"
        self._last_forget_step = -10**9
        self._pending_hits: dict[str, list[int]] = {}  # dead-end id -> steps that matched it
        self._wake_floor = -1                       # steps <= this are never shown again
        self._slim_upto = -1                        # steps <= this are shown as one-line records
        self._dropped: set[int] = set()             # steps dropped by forgetting (memory covers them)

    # ---- loop hooks ---------------------------------------------------------------
    def bind(self, *, evaluator) -> None:
        self.evaluator = evaluator

    def set_fatigue(self, reading: FatigueReading) -> None:
        self.fatigue = reading

    def select_recent(self, recent: list[dict], step: int) -> list[dict]:
        """Recent step records as the agent should see them after forgetting / sleep."""
        out = []
        for h in recent:
            s = h.get("step") or 0
            if s <= self._wake_floor or s in self._dropped:
                continue
            if s <= self._slim_upto:
                err = h.get("error")
                n = self.p["forget_error_chars"]
                h = dict(h, side_info={}, rationale=None,
                         error=(err[:n] + "...") if isinstance(err, str) and len(err) > n else err)
            out.append(h)
        return out

    def observe(self, step: int, record: dict) -> dict:
        """Called by the loop before a step is logged. Returns fields to log with it."""
        hits = self.store.match_dead_ends(record.get("approach_tags") or [], self.p["dead_end_match"])
        for h in hits:
            self._pending_hits.setdefault(h.id, []).append(step)
        slim = {k: record.get(k) for k in ("step", "score", "valid", "surprise")}
        slim["perseveration_hit"] = bool(hits)
        slim["event"] = "step"
        self._seen.append(slim)
        out = {"perseveration_hit": bool(hits), "perseveration_items": [h.id for h in hits]}
        if self.p["pressure"]:
            value = P.combine(self._components(self._seen, step), self.p["pressure_weights"])
            self._last_pressure = (step, value)
            out["pressure"] = value
        return out

    def _components(self, records: list[dict], step: int) -> dict:
        return P.components(records, step=step, last_sleep_step=max(0, self._last_sleep_step),
                            stall_window=self.p["stall_window"], clutter_norm=self.p["clutter_norm"])

    @property
    def scheduled_sleeps(self) -> int | None:
        """Number of sleeps an explicit schedule asks for (schedule gating), else None."""
        return len(self.sleep_steps) if self.p["gating"] == "schedule" else None

    # ---- SleepPolicy ----------------------------------------------------------------
    def should_sleep(self, step: int, history: list[dict]) -> bool:
        self._planned = None
        if step <= 0:
            return False
        gating = self.p["gating"]
        if gating == "fatigue" and self.fatigue is not None:
            self._reason = "fatigue"
            return self._fatigue_gate(step)
        if gating in ("fixed", "schedule"):
            due = (step in self.sleep_steps) if gating == "schedule" else (step % self.every_n == 0)
            if due:
                self._plan_timed()
            return due
        return self._legacy_gate(step, history)

    def _plan_timed(self) -> None:
        """Depth of a fixed/schedule sleep (see `timed_depth` in the module docstring)."""
        mode = self.p["timed_depth"]
        self._reason = f"{self.p['gating']}_{mode}"
        if mode == "nrem":
            self._planned = SleepDepth.NREM
        elif mode == "fatigue":
            rec = self.fatigue.recommended_depth if self.fatigue is not None else SleepDepth.NREM
            self._planned = SleepDepth(min(max(int(rec), int(SleepDepth.FORGET)), int(self.max_depth)))

    def _fatigue_gate(self, step: int) -> bool:
        """Respond to the monitor's level. Reads the fatigue reading only; never modifies it."""
        p, r = self.p, self.fatigue
        depth = SleepDepth(min(int(r.recommended_depth), int(self.max_depth)))
        if depth == SleepDepth.NONE:
            return False
        if depth >= SleepDepth.NREM and r.budget_remaining is not None \
                and r.budget_remaining < p["min_sleep_budget_tokens"]:
            self.last_deferred = True
            if not p["defer_fallback_forget"]:
                return False
            depth = SleepDepth.FORGET
        if depth == SleepDepth.FORGET:
            if not p["respond_drowsy"] or step - self._last_forget_step < p["forget_min_gap"]:
                return False
        self._planned = depth
        return True

    def _legacy_gate(self, step: int, history: list[dict]) -> bool:
        if self.p["gating"] == "fixed" or not self.p["pressure"]:
            return step % self.every_n == 0
        if self._last_pressure and self._last_pressure[0] == step:
            value = self._last_pressure[1]
        else:
            value = P.combine(self._components(history, step), self.p["pressure_weights"])
        return P.gate(step, max(0, self._last_sleep_step), value, threshold=self.p["pressure_threshold"],
                      min_gap=self.p["min_gap"], max_gap=self.p["max_gap"])

    def sleep(self, task: str, block: str, history: list[dict], llm, budget) -> str:
        step = max((h.get("step", 0) for h in history), default=0)
        planned, self._planned = self._planned, None
        if planned == SleepDepth.FORGET:
            self.forget(history, step=step)
            return self.memory_section(self.max_block_tokens)
        reason = self._reason
        if planned == SleepDepth.NREM:
            m = self.consolidate(task, history, llm, budget, step=step, mode="nrem", reason=reason)
        elif planned == SleepDepth.NREM_REM:
            since = self._last_sleep_step
            m1 = self.consolidate(task, history, llm, budget, step=step, mode="nrem", reason=reason)
            m2 = self.consolidate(task, history, llm, budget, step=step, mode="rem", reason=reason, since=since)
            m = dict(m2)
            for k in SUM_METRICS:
                m[k] = (m1.get(k) or 0) + (m2.get(k) or 0)
            m["evicted_ids"] = list(m1.get("evicted_ids", [])) + list(m2.get("evicted_ids", []))
            m["sleep_mode"] = "nrem+rem"
            m["nrem_pass"] = {k: v for k, v in m1.items() if k != "policy"}
        else:  # pressure / fixed gating (or no fatigue reading): the scheduler picks the mode
            m = self.consolidate(task, history, llm, budget, step=step)
        depth = SleepDepth.NREM_REM if m.get("sleep_mode") in ("rem", "nrem+rem") else SleepDepth.NREM
        self._wake(step)
        self.last_depth = depth.label
        m["sleep_depth"] = depth.label
        m["block_tokens"] = count_tokens(self.memory_section(self.max_block_tokens))
        self.last_metrics = m
        return self.memory_section(self.max_block_tokens)

    def _wake(self, step: int) -> None:
        """After a real sleep, raw history memory now covers leaves the agent's context for good."""
        k = self.p["wake_recent_steps"]
        if k is not None:
            self._wake_floor = max(self._wake_floor, step - k)
        self._pending_hits.clear()

    def forget(self, history: list[dict], *, step: int) -> dict:
        """Drowsy response: code only, ZERO LLM calls.

        - drop recent raw steps the memory already covers (cited as evidence by a memory item),
          except the newest `wake_recent_steps`
        - shrink every step up to now to a one-line record (no side_info, short error)
        - strengthen dead-end items matched since the last forget/sleep (evidence = those steps)
        """
        p = self.p
        records = [h for h in history if h.get("event", "step") == "step"]
        self._refresh_goal(self._best_record(records), step)
        covered = {e for it in self.store.items.values() for e in it.evidence if isinstance(e, int)}
        keep = {h.get("step") for h in records[-p["wake_recent_steps"]:]} if p["wake_recent_steps"] else set()
        newly = {s for s in covered if s <= step and s not in keep} - self._dropped
        self._dropped |= newly
        self._slim_upto = step
        strengthened = []
        for item_id, steps in sorted(self._pending_hits.items()):
            it = self.store.get(item_id)
            if it is not None and it.kind == "dead_end":
                self.store.support(item_id, step, evidence=steps, amount=p["forget_dead_end_gain"])
                strengthened.append(item_id)
        self._pending_hits.clear()
        self._last_forget_step = step
        self.last_depth = SleepDepth.FORGET.label
        self.last_metrics = {
            "policy": self.name, "sleep_mode": None, "sleep_depth": "forget", "forgot_steps": sorted(newly),
            "slimmed_upto": step, "dead_ends_strengthened": strengthened, "items_added": 0, "items_evicted": 0,
            "replays_wasted": 0, "llm_calls": 0, "n_items": len(self.store),
            "block_tokens": count_tokens(self.memory_section(self.max_block_tokens)),
        }
        return self.last_metrics

    def memory_section(self, token_cap: int | None = None) -> str:
        """Goals, schemas, then items with strength >= wake_min_strength by rank, under the cap."""
        cap = self.max_block_tokens if token_cap is None else token_cap
        extra = self.schemas.lines() if self.schemas else []
        return self.store.render(cap, extra_lines=extra,
                                 min_strength=self.p["wake_min_strength"] or 0.0)

    @staticmethod
    def _best_record(records: list[dict]) -> dict | None:
        ok = [h for h in records if h.get("valid") and isinstance(h.get("score"), (int, float))
              and math.isfinite(h["score"])]
        return max(ok, key=lambda h: h["score"], default=None)

    # ---- one sleep ------------------------------------------------------------------
    def consolidate(self, task: str, history: list[dict], llm, budget, *, step: int, mode: str | None = None,
                    reason: str | None = None, since: int | None = None) -> dict:
        """One consolidation pass. `mode` forces nrem/rem (else the scheduler picks); `since`
        overrides the last sleep step that bounds the history window (NREM+REM second pass)."""
        p = self.p
        records = [h for h in history if h.get("event", "step") == "step"]
        since = self._last_sleep_step if since is None else since
        window = [h for h in records if (h.get("step") or 0) > since][-p["max_history_steps"]:]
        programs = {h.get("program_id"): h.get("program") for h in records if h.get("program_id")}
        step_scores = {h.get("step"): h.get("score") for h in records
                       if h.get("valid") and isinstance(h.get("score"), (int, float)) and math.isfinite(h["score"])}
        best_rec = max((h for h in records if h.get("step") in step_scores), key=lambda h: h["score"], default=None)
        run_best = best_rec["score"] if best_rec else None

        if mode is None:
            mode, mode_reason = choose_mode(records, step, budget, enabled=p["scheduler"],
                                            plateau_window=p["plateau_window"],
                                            late_budget_fraction=p["late_budget_fraction"])
        else:
            if mode not in ("nrem", "rem"):
                raise ValueError(f"mode must be nrem or rem, got {mode!r}")
            mode_reason = reason or "forced"
        k_steps = p["rem_k_steps"] if mode == "rem" else p["k_steps"]
        replay_budget = p["rem_replay_budget"] if mode == "rem" else p["replay_budget"]
        comp = self._components(records, step)
        before = set(self.store.items)
        self._refresh_goal(best_rec, step)

        m: dict[str, Any] = {
            "policy": self.name, "sleep_mode": mode, "mode_reason": mode_reason,
            "pressure": P.combine(comp, p["pressure_weights"]), "pressure_components": comp,
            "history_steps": len(window), "claims": 0, "edits_applied": 0, "edits_rejected": 0,
            "replays_run": 0, "replays_wasted": 0, "parse_error": None, "replays": [], "edits": [],
        }
        claims: list[dict] = []
        replays: list[dict] = []
        if window:
            user = CONSOLIDATOR_USER.substitute(
                task=task, n_items=len(self.store), memory_items_json=json.dumps(self.store.to_json_list(), indent=0),
                last_sleep_step=max(0, since), step=step,
                steps_jsonl="\n".join(json.dumps(step_view(h, programs), default=str) for h in window),
                stall=f"{comp['stall']:.2f}", persev=f"{comp['perseveration']:.2f}", surprise=f"{comp['surprise']:.2f}",
                mode_note=REM_NOTE if mode == "rem" else NREM_NOTE, k_steps=k_steps,
                replay_budget=replay_budget if p["replay"] else 0,
            )
            resp = llm.complete([{"role": "system", "content": CONSOLIDATOR_SYSTEM}, {"role": "user", "content": user}],
                                role="sleep", max_tokens=p["consolidator_max_tokens"])
            obj, err = parse_json_object(resp.text)
            claims, replays, problems = validate_consolidation(obj, max_claims=p["max_claims"],
                                                               max_chars=p["max_item_chars"])
            m["parse_error"] = err or ("; ".join(problems) if problems else None)
            if isinstance(obj, dict) and isinstance(obj.get("selected_steps"), list):
                m["selected_steps"] = obj["selected_steps"][:k_steps]
        m["claims"] = len(claims)
        by_cid = {c["cid"]: c for c in claims}

        decided: set = set()
        applied: list[Edit] = []
        if p["replay"] and replays:
            rep = run_replays(replays, by_cid, self.store, self.evaluator, step=step, max_replays=replay_budget,
                              step_scores=step_scores, run_best=run_best, rel_tol=p["replay_rel_tol"])
            decided, applied = rep.decided_cids, list(rep.edits_applied)
            m.update(replays=rep.records, replays_run=len(rep.records), replays_wasted=rep.wasted)

        tentative = self._tentative_edits([c for c in claims if c["cid"] not in decided])
        ok, rejected = self.store.apply_edits(tentative, step)
        applied += ok
        m["edits_applied"] = len(applied)
        m["edits_rejected"] = len(rejected)
        m["edits"] = [e.to_dict() for e in applied]

        touched = {e.target for e in applied if e.target} | {"g0"}
        if p["decay"]:
            self.store.decay(step, p["decay_factor"], exempt=touched)
        evicted = self.store.evict(min_strength=p["min_strength"], max_items=p["max_items"]) if p["eviction"] else []
        if self.schemas is not None and mode == "rem":
            m.update(self.schemas.update(task, self.store, llm, step=step))

        self._last_sleep_step = step
        m["items_added"] = len(set(self.store.items) - before) + sum(1 for i in evicted if i not in before)
        m["items_evicted"] = len(evicted)
        m["evicted_ids"] = evicted
        m["n_items"] = len(self.store)
        m["block_tokens"] = count_tokens(self.memory_section(self.max_block_tokens))
        self.last_metrics = m
        return m

    def _refresh_goal(self, best_rec: dict | None, step: int) -> None:
        if best_rec:
            text = (f"Maximize the score. Best valid score so far: {best_rec['score']:.6g} "
                    f"(step {best_rec.get('step')}, program {best_rec.get('program_id')}).")
        else:
            text = "Maximize the score. No valid program yet: first produce any valid construction."
        g = self.store.get("g0")
        if g is None:
            self.store.add(kind="goal", text=text, step=step, status="verified", strength=1.0, item_id="g0")
        else:
            g.text, g.last_supported_step = text[: self.store.max_item_chars], step

    def _tentative_edits(self, claims: list[dict]) -> list[Edit]:
        edits: list[Edit] = []
        for c in claims:
            item = {"kind": c["kind"], "text": c["text"], "approach_tags": c["approach_tags"]}
            targets = [t for t in c["relates_to"]]
            if c["relation"] == "new" or not targets:
                edits.append(Edit("add", item=item, status="tentative", evidence=c["evidence"]))
            else:
                op = "support" if c["relation"] == "supports" else "contradict"
                edits.extend(Edit(op, target=t, evidence=c["evidence"]) for t in targets)
        return edits
