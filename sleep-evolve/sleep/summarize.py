"""Every N steps, one LLM call folds the previous block and the new steps into a fresh block."""

from __future__ import annotations

from sleep.base import render_steps, step_events, steps_since_last_sleep

SYSTEM = (
    "You consolidate the history of an evolutionary program search into compact notes for "
    "the agent that proposes the next programs. The agent will read ONLY your notes, not the "
    "raw history. Keep what helps it improve: what worked, what failed and why, best scores, "
    "promising directions, and dead ends to avoid. Be concise and concrete."
)


class Summarize:
    name = "summarize"

    def __init__(self, every_n: int = 10, max_block_tokens: int = 1024, include_best_program: bool = True,
                 sleep_max_tokens: int = 4096, **_unused):
        # _unused: in a baseline suite every sleeping arm carries the identical shared
        # full_sleep param block (drift check); this policy only reads its own keys.
        if every_n <= 0:
            raise ValueError("every_n must be positive")
        self.every_n = every_n
        self.max_block_tokens = max_block_tokens
        self.include_best_program = include_best_program
        # completion cap for the sleep call; separate from the notes budget because reasoning
        # models also spend completion tokens on reasoning (gpt-5 at low effort would starve at 1024)
        self.sleep_max_tokens = sleep_max_tokens

    def should_sleep(self, step: int, history: list[dict]) -> bool:
        return step > 0 and step % self.every_n == 0

    def sleep(self, task, block, history, llm, budget) -> str:
        new_steps = steps_since_last_sleep(history)
        parts = [
            f"## Task\n{task}",
            f"## Previous notes\n{block or '(none yet)'}",
            f"## New steps since the last consolidation\n{render_steps(new_steps)}",
        ]
        if self.include_best_program:
            valid = [h for h in step_events(history) if h["valid"]]
            if valid:
                best = max(valid, key=lambda h: h["score"])
                parts.append(
                    f"## Best program so far ({best['program_id']}, score {best['score']:.6g})\n"
                    f"```python\n{best['program']}\n```"
                )
        parts.append("Write the updated notes. Output only the notes.")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n\n".join(parts)}]
        resp = llm.complete(messages, role="sleep", max_tokens=self.sleep_max_tokens)
        return resp.text.strip()
