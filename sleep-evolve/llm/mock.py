"""Deterministic mock transport for tests and dry runs. Plug it into LLMClient like a real one.

Every call reports fixed usage (prompt_tokens + completion_tokens), so budget arithmetic
in tests is exact. Responses come from `script` (a list consumed in order, then cycled)
or, by default, from generators routed on a marker in the system prompt:

  "sleep-phase consolidator"  -> FullSleep consolidator JSON (claims, paired control/treatment
                                 replays, one goal, no merges)
  "memory schemas"            -> schema-store JSON
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
"""

from __future__ import annotations

import json
import random
import re

from llm.client import LLMResponse, Message

MOCK_TARGET = 0.7  # evaluators/mock.py TARGET (kept here so llm/ does not import evaluators/)
MEMORY_MARKER = "## Consolidated notes"  # harness/context.py MEMORY_TITLE prefix
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
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.script = list(script or [])
        self.grow_lines = grow_lines
        self.calibration_drift = calibration_drift
        self.calibration_drift_start = calibration_drift_start
        self.memory_boost = memory_boost
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
        elif "consolidat" in low:
            self.sleep_calls += 1
            text = f"Consolidated notes after {len(self.calls)} calls: values near 0.7 score best."
        else:
            self.agent_calls += 1
            text = self._agent(user, header="APPROACH_TAGS" in system)
        return LLMResponse(text, self.prompt_tokens, self.completion_tokens)

    # ---- generators -----------------------------------------------------------------
    def _agent(self, user: str, header: bool) -> str:
        if self.memory_boost > 0 and MEMORY_MARKER in user:
            w = 0.3 * (1.0 - min(1.0, self.memory_boost))
            xs = [round(min(1.0, max(0.0, MOCK_TARGET + self._rng.uniform(-w, w))), 4) for _ in range(5)]
        else:
            xs = [round(self._rng.uniform(0.0, 1.0), 4) for _ in range(5)]
        pad = "".join(f"# exploration note {i}: keep values in [0, 1]\n"
                      for i in range(self.grow_lines * self.agent_calls))
        code = f"{pad}def construct():\n    return {xs}\n"
        if not header:
            return f"Trying new values.\n```python\n{code}```"
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
            f"```python\n{code}```"
        )

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
        # Paired replays (control + treatment differing only by the claimed change).
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

    def _schemas(self, user: str) -> str:
        ids = re.findall(r'"id": "(m\d+)"', user)
        return json.dumps({"schemas": [{"text": "Stay near the target and verify with small probes",
                                        "approach_tags": ["near-target"], "members": ids[:3]}]})
