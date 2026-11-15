"""Deterministic mock transport for tests and dry runs. Plug it into LLMClient like a real one.

Every call reports fixed usage (prompt_tokens + completion_tokens), so budget arithmetic
in tests is exact. Responses come from `script` (a list consumed in order, then cycled)
or, by default, from generators routed on a marker in the system prompt:

  "sleep-phase consolidator"  -> FullSleep consolidator JSON (claims, paired control/treatment
                                 replays, one goal, no merges)
  "memory schemas"            -> schema-store JSON
  "REM-phase dreamer"         -> dream JSON (sleep/dream_io.py): three dreams whose tests are a ref
                                 control + diff treatment, a full-program fallback, and a diff that
                                 does not match its base (a legacy full/full test when the prompt shows
                                 no base program). Before, dream calls fell through to the agent route.
  consolidator replays        -> diff-first when the prompt's <programs> block shows a base: ref control +
                                 diff treatment, ref control + full-treatment fallback (with a reason),
                                 then a non-matching diff; the old full/full shape otherwise
  "consolidat" (other)        -> a canned Summarize block
  otherwise (agent)           -> a random program for the mock evaluator; with the awake
                                 header (APPROACH_TAGS, ...) when the system prompt asks for it

`grow_lines` > 0 makes agent programs grow by that many comment lines per agent call, so
context truncation gets exercised in long runs.

Worsening-calibration scenario (fatigue tests): `calibration_drift` > 0 makes EXPECTED_SCORE the
mock evaluator's true score of the proposed program (evaluators/mock.py: 1 - mean((x - 0.7)^2))
plus `calibration_drift * max(0, agent_calls - calibration_drift_start)`, so the agent's
prediction error grows steadily. With the default 0 the output is unchanged.

Memory-helps scenario (baseline-suite tests): `memory_boost` in (0, 1] makes agent programs whose
prompt contains consolidated memory (harness/context.py MEMORY_TITLE) sample each coordinate from
MOCK_TARGET +- 0.3 * (1 - memory_boost) instead of uniform [0, 1], so sleeping arms can differ from
no_sleep. With the default 0 the output is unchanged.

Diff edits (harness/edits.py): when the agent system prompt allows SEARCH/REPLACE blocks (edit_mode
auto: "SEARCH/REPLACE" in the system prompt), each agent reply is a diff with probability `diff_rate`
(default 0.5), else a full rewrite; when it demands them (edit_mode diff: "ONLY SEARCH/REPLACE"), every
reply is a diff. A diff replaces the parent's single `return ...` line (parent read from the prompt's
"## Parent program" block) with the new values; with probability `bad_diff_rate` (default 0) its
SEARCH line is one that is not in the parent, so applying it fails. No parent, not exactly one return
line, or a truncated parent -> a full rewrite. Prompts that do not mention SEARCH/REPLACE (edit_mode
full) get exactly the old output and the random stream is not touched.
"""

from __future__ import annotations

import json
import random
import re

from llm.client import LLMResponse, Message

MOCK_TARGET = 0.7  # evaluators/mock.py TARGET (kept here so llm/ does not import evaluators/)
MEMORY_MARKER = "## Consolidated notes"  # harness/context.py MEMORY_TITLE prefix
PARENT_RE = re.compile(r"## Parent program \([^)]*\)\n```python\n(.*?)```", re.DOTALL)  # harness/context.py
TRUNCATION_MARK = "chars truncated to fit the context cap"  # harness/context.py truncate_middle
BASE_RE = re.compile(r"\[([\w.-]+)\] step \S+ score \S+\n```python\n(.*?)```", re.DOTALL)  # program_ref.render_bases
TAG_VOCAB = ("uniform-init", "near-target", "random-search", "perturb-parent", "grid-sweep", "anneal")


class MockTransport:
    def __init__(
        self,
        prompt_tokens: int = 60,
        completion_tokens: int = 40,
        seed: int = 0,
        script: list[str] | None = None,
        grow_lines: int = 0,
        calibration_drift: float = 0.0,
        calibration_drift_start: int = 0,
        memory_boost: float = 0.0,
        diff_rate: float = 0.5,
        bad_diff_rate: float = 0.0,
        full_reason: str = "the change restructures most of the program",
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.script = list(script or [])
        self.grow_lines = grow_lines
        self.calibration_drift = calibration_drift
        self.calibration_drift_start = calibration_drift_start
        self.memory_boost = memory_boost
        self.diff_rate = diff_rate
        self.bad_diff_rate = bad_diff_rate
        self.full_reason = full_reason
        self.diff_replies = 0       # SEARCH/REPLACE replies emitted (bad ones included)
        self.bad_diff_replies = 0   # of which the SEARCH line is not in the parent
        self.calls: list[list[Message]] = []
        self.agent_calls = 0
        self.sleep_calls = 0
        self._rng = random.Random(seed)

    def __call__(self, messages: list[Message], max_tokens: int) -> LLMResponse:
        self.calls.append(messages)
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""
        low = system.lower()
        if self.script:
            text = self.script[(len(self.calls) - 1) % len(self.script)]
        elif "sleep-phase consolidator" in low:
            self.sleep_calls += 1
            text = self._consolidator(user)
        elif "memory schemas" in low:
            self.sleep_calls += 1
            text = self._schemas(user)
        elif "rem-phase dreamer" in low:
            self.sleep_calls += 1
            text = self._dreamer(user)
        elif "consolidat" in low:
            self.sleep_calls += 1
            text = f"Consolidated notes after {len(self.calls)} calls: values near 0.7 score best."
        else:
            self.agent_calls += 1
            text = self._agent(user, header="APPROACH_TAGS" in system,
                               diffs="SEARCH/REPLACE" in system, diff_only="ONLY SEARCH/REPLACE" in system)
        return LLMResponse(text, self.prompt_tokens, self.completion_tokens)

    # ---- generators -----------------------------------------------------------------
    def _agent(self, user: str, header: bool, diffs: bool = False, diff_only: bool = False) -> str:
        if self.memory_boost > 0 and MEMORY_MARKER in user:
            w = 0.3 * (1.0 - min(1.0, self.memory_boost))
            xs = [round(min(1.0, max(0.0, MOCK_TARGET + self._rng.uniform(-w, w))), 4) for _ in range(5)]
        else:
            xs = [round(self._rng.uniform(0.0, 1.0), 4) for _ in range(5)]
        pad = "".join(f"# exploration note {i}: keep values in [0, 1]\n"
                      for i in range(self.grow_lines * self.agent_calls))
        code = f"{pad}def construct():\n    return {xs}\n"
        body = f"```python\n{code}```"
        if diffs:
            diff = self._diff(user, xs, force=diff_only)
            if diff is not None:
                body = diff
            elif not diff_only:  # auto asks for the reason whenever a full rewrite is chosen
                body = f"FULL_REWRITE_REASON: {self.full_reason}\n{body}"
        if not header:
            return f"Trying new values.\n{body}"
        tags = self._rng.sample(TAG_VOCAB, 2)
        mem = re.findall(r"\[(m\d+)\]", user)[:2]
        expected = round(self._rng.uniform(0.5, 1.0), 3)
        if self.calibration_drift:
            true = 1.0 - sum((x - MOCK_TARGET) ** 2 for x in xs) / len(xs)
            lag = max(0, self.agent_calls - self.calibration_drift_start)
            expected = round(true + self.calibration_drift * lag, 6)
        return (
            f"APPROACH_TAGS: {', '.join(tags)}\n"
            f"EXPECTED_SCORE: {expected}\n"
            f"RATIONALE: try {tags[0]} values around the target\n"
            f"RELEVANT_MEMORY: {', '.join(mem) or 'none'}\n"
            f"{body}"
        )

    def _diff(self, user: str, xs: list[float], force: bool) -> str | None:
        """A SEARCH/REPLACE reply editing the parent's return line, or None for a full rewrite."""
        if not force and self._rng.random() >= self.diff_rate:
            return None
        m = PARENT_RE.search(user)
        parent = m.group(1) if m else ""
        if not parent or TRUNCATION_MARK in parent:
            return None
        lines = parent.splitlines()
        rets = [ln for ln in lines if ln.lstrip().startswith("return")]
        if len(rets) != 1:
            return None
        indent = rets[0][: len(rets[0]) - len(rets[0].lstrip())]
        search = rets[0]
        if self.bad_diff_rate and self._rng.random() < self.bad_diff_rate:
            search = f"{indent}return [0.123456789]  # not in the parent"
            self.bad_diff_replies += 1
        self.diff_replies += 1
        return f"<<<<<<< SEARCH\n{search}\n=======\n{indent}return {xs}\n>>>>>>> REPLACE\n"

    def _consolidator(self, user: str) -> str:
        steps = [int(s) for s in re.findall(r'"id": (\d+)', user)]
        mems = [m for m in dict.fromkeys(re.findall(r'"id": "(m\d+)"', user))]
        tag_lists = re.findall(r'"approach_tags": \[([^\]]*)\]', user.split("<history", 1)[-1])
        tags = [t.strip().strip('"') for t in (tag_lists[0].split(",") if tag_lists and tag_lists[0] else [])]
        tags = [t for t in tags if t] or ["near-target"]
        ev = steps[-3:] or [0]
        claims = [
            {"cid": "c1", "kind": "rule", "text": "Values close to 0.7 in every coordinate raise the score",
             "evidence": ev[-1:], "approach_tags": ["near-target"], "relates_to": [], "relation": "new"},
            {"cid": "c2", "kind": "dead_end", "text": f"{tags[0]} alone rarely beats the parent",
             "evidence": ev[:1], "approach_tags": tags[:2], "relates_to": [], "relation": "new"},
        ]
        # Paired replays (control + treatment differing only by the claimed change). Diff-first when the
        # prompt shows a base program: ref control + diff treatment, then a full-treatment fallback, then a
        # diff whose SEARCH is not in the base (wasted); the old full/full shape otherwise.
        base = self._base_program(user)
        if base is not None:
            bid, ret = base
            indent = ret[: len(ret) - len(ret.lstrip())]
            replays = [
                {"cid": "c1", "change": "every coordinate -> 0.7", "control": {"base": bid},
                 "treatment": {"diff": f"<<<<<<< SEARCH\n{ret}\n=======\n{indent}return [0.7, 0.7, 0.7, 0.7, 0.7]\n"
                                       ">>>>>>> REPLACE\n"},
                 "decides": "add c1 as verified"},
                {"cid": "c2", "change": "uniform 0.5 probe", "control": {"base": bid},
                 "treatment": {"program": "def construct():\n    return [0.5, 0.5, 0.5, 0.5, 0.5]\n",
                               "reason": "the probe replaces the whole construction"},
                 "decides": "add c2 as verified"},
                {"cid": "c1", "change": "stale edit", "control": {"base": bid},
                 "treatment": {"diff": f"<<<<<<< SEARCH\n{indent}return [0.123456789]  # not in the control\n"
                                       f"=======\n{indent}return [0.7, 0.7, 0.7, 0.7, 0.7]\n>>>>>>> REPLACE\n"},
                 "decides": "add c1 as verified"},
            ]
        else:
            replays = [
                {"cid": "c1", "change": "every coordinate 0.5 -> 0.7",
                 "control": "def construct():\n    return [0.5, 0.5, 0.5, 0.5, 0.5]\n",
                 "treatment": "def construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.7]\n",
                 "decides": "add c1 as verified"},
            ]
        if mems:
            claims.append({"cid": "c3", "kind": "fact", "text": "Earlier memory still holds", "evidence": ev,
                           "approach_tags": [], "relates_to": [mems[0]], "relation": "supports"})
            # crashing control: no baseline, so the treatment is never run (wasted, 1 evaluation)
            replays.append({"cid": "c3", "change": "probe",
                            "control": "def construct():\n    raise RuntimeError('probe crashed')\n",
                            "treatment": "def construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.7]\n",
                            "decides": f"support {mems[0]}"})
        if len(mems) >= 2:
            claims.append({"cid": "c4", "kind": "fact", "text": "An older memory item is wrong", "evidence": ev,
                           "approach_tags": [], "relates_to": [mems[-1]], "relation": "contradicts"})
        old_goal = next(iter(re.findall(r'"id": "(g[1-9]\d*)"', user)), None)
        goals = [{"text": "Keep every coordinate near 0.7 and probe one coordinate at a time.",
                  "replaces": old_goal}]
        return json.dumps({"selected_steps": ev, "claims": claims, "replays": replays, "goals": goals,
                           "merges": []})

    @staticmethod
    def _base_program(user: str) -> tuple[str, str] | None:
        """(program id, its single `return` line) of the first program in the <programs> block, or None."""
        block = user.split("<programs", 1)[-1].split("</programs>", 1)[0] if "<programs" in user else ""
        m = BASE_RE.search(block)
        if not m:
            return None
        rets = [ln for ln in m.group(2).splitlines() if ln.lstrip().startswith("return")]
        return (m.group(1), rets[0]) if len(rets) == 1 else None

    def _dreamer(self, user: str) -> str:
        """Dreams with diff-first tests: ref + diff (valid), full fallback, non-matching diff; legacy
        full/full test when no base program is shown."""
        mems = list(dict.fromkeys(re.findall(r'"id": "(m\d+)"', user)))[:2] or ["m1"]
        base = self._base_program(user)
        half = "def construct():\n    return [0.5, 0.5, 0.5, 0.5, 0.5]\n"
        good = "def construct():\n    return [0.7, 0.7, 0.7, 0.7, 0.7]\n"
        if base is None:
            tests = [{"control": half, "treatment": good, "change": "0.5 -> 0.7"}, None, None]
        else:
            bid, ret = base
            ind = ret[: len(ret) - len(ret.lstrip())]
            tests = [
                {"control": {"base": bid}, "change": "every coordinate -> 0.7",
                 "treatment": {"diff": f"<<<<<<< SEARCH\n{ret}\n=======\n{ind}return [0.7, 0.7, 0.7, 0.7, 0.7]\n"
                                       ">>>>>>> REPLACE\n"}},
                {"control": {"program": half, "reason": "a neutral baseline, no logged program fits"},
                 "treatment": {"program": good, "reason": "the control is new"}, "change": "0.5 -> 0.7"},
                {"control": {"base": bid}, "change": "stale edit",
                 "treatment": {"diff": f"<<<<<<< SEARCH\n{ind}return [0.123456789]\n=======\n{ind}return [0.6] * 5\n"
                                       ">>>>>>> REPLACE\n"}},
            ]
        kinds = ("association", "counterfactual", "extrapolation")
        return json.dumps({"dreams": [
            {"did": f"d{i + 1}", "kind": kinds[i], "text": f"dream {i + 1}: coordinates near 0.7 raise the score",
             "sources": mems, "rationale": "recombines the sampled items", "test": t}
            for i, t in enumerate(tests)]})

    def _schemas(self, user: str) -> str:
        ids = re.findall(r'"id": "(m\d+)"', user)
        return json.dumps({"schemas": [{"text": "Stay near the target and verify with small probes",
                                        "approach_tags": ["near-target"], "members": ids[:3]}]})
