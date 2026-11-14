"""All programs and scores of a run, plus parent sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Program:
    id: str
    code: str
    score: float  # -inf when invalid
    valid: bool
    parent_id: str | None
    step: int


class ProgramDB:
    P_BEST = 0.7
    TOP_K = 10

    def __init__(self) -> None:
        self.programs: dict[str, Program] = {}

    def add(self, program: Program) -> None:
        if program.id in self.programs:
            raise ValueError(f"duplicate program id {program.id}")
        self.programs[program.id] = program

    def __len__(self) -> int:
        return len(self.programs)

    def top(self, k: int) -> list[Program]:
        valid = [p for p in self.programs.values() if p.valid]
        return sorted(valid, key=lambda p: (-p.score, p.step))[:k]

    def best(self) -> Program | None:
        top = self.top(1)
        return top[0] if top else None

    def sample_parent(self, rng: random.Random) -> Program:
        """Best with probability 0.7, else uniform over the top 10 valid programs.

        If nothing is valid yet, fall back to the earliest program (the seed).
        """
        top = self.top(self.TOP_K)
        if not top:
            return min(self.programs.values(), key=lambda p: p.step)
        if rng.random() < self.P_BEST:
            return top[0]
        return rng.choice(top)
