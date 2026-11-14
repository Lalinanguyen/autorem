"""Evaluator contract and the shared subprocess sandbox.

A candidate program is Python source that defines `construct()` and returns the
candidate solution. It is NEVER executed in the harness process: `run_program`
writes it to a temp dir and runs `_runner.py` in a fresh interpreter with a hard
timeout. The child sends back only JSON data (the return value of `construct()`),
and the evaluator scores that data in the parent.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

INVALID_SCORE = float("-inf")
_RUNNER = Path(__file__).with_name("_runner.py")
_OUTPUT_TAIL = 2000  # chars of child stdout/stderr kept in error messages


class EvalResult(BaseModel):
    score: float  # higher is better; negate internally if the problem minimizes
    valid: bool
    error: str | None = None
    details: dict = Field(default_factory=dict)
    # Actionable diagnostics for the proposer (violations, overlap/slack values, ASCII render, ...).
    # Shown to the agent and logged for every arm. Must be JSON-serializable.
    side_info: dict = Field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    problem_id: int
    name: str
    best_known: float  # used for normalized score = score / best_known
    timeout_s: int
    description: str  # task statement shown to the agent (extension to the spec'd contract)

    def seed_program(self) -> str: ...

    def evaluate(self, program: str) -> EvalResult: ...


class ProgramFailed(Exception):
    """The program crashed, timed out, or returned something that is not JSON data.

    `kind` classifies the failure for side_info: "timeout", "crash", "no_result" or
    "bad_output". `output_tail` is the end of the child's stderr/stdout (crash only).
    """

    def __init__(self, message: str, kind: str = "crash", output_tail: str = ""):
        super().__init__(message)
        self.kind = kind
        self.output_tail = output_tail


def normalized_score(score: float | None, best_known: float) -> float | None:
    """1.0 at best_known, below 1 when worse, above 1 when it beats best_known.

    Maximization (best_known > 0): score / best_known, as the spec defines it.
    Minimization (negated, so best_known < 0): best_known / score. Plain score / best_known
    would be inverted here, because both are negative and a worse score gives a larger ratio.
    """
    import math

    if score is None or not math.isfinite(score) or best_known == 0:
        return None
    if best_known > 0:
        return score / best_known
    return best_known / score if score != 0 else None


def invalid(error: str, *, side_info: dict | None = None, **details: Any) -> EvalResult:
    return EvalResult(score=INVALID_SCORE, valid=False, error=error, details=details, side_info=side_info or {})


_FAILURE_HINTS = {
    "timeout": "construct() did not finish in time: reduce iterations or use vectorized numpy.",
    "crash": "the program raised or exited non-zero: see stderr_tail for the traceback.",
    "no_result": "the program ran but wrote no result: construct() must return the solution.",
    "bad_output": "construct() returned something that is not JSON data (lists/numbers/dicts).",
}
_SIDE_INFO_TAIL = 800  # chars of child output kept in side_info


def failure_side_info(e: "ProgramFailed", timeout_s: float | None = None) -> dict:
    """Structured side_info for a program that never produced a scorable solution."""
    info: dict[str, Any] = {"stage": "run", "failure": e.kind, "hint": _FAILURE_HINTS.get(e.kind, "")}
    if e.kind == "timeout" and timeout_s is not None:
        info["timeout_s"] = timeout_s
    if e.output_tail:
        info["stderr_tail"] = e.output_tail[-_SIDE_INFO_TAIL:]
    return info


def run_program(program: str, timeout_s: float) -> Any:
    """Run `program` in a subprocess, call its construct(), return the JSON-decoded result.

    Raises ProgramFailed on crash, timeout, or bad output. The whole process tree
    is killed on timeout.
    """
    with tempfile.TemporaryDirectory(prefix="sleepevolve_") as tmp:
        prog_path = Path(tmp) / "program.py"
        out_path = Path(tmp) / "result.json"
        prog_path.write_text(program, encoding="utf-8")
        cmd = [sys.executable, "-I", str(_RUNNER), str(prog_path), str(out_path)]
        popen_kw: dict[str, Any] = {}
        if os.name == "posix":
            popen_kw["start_new_session"] = True
        else:
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        proc = subprocess.Popen(
            cmd,
            cwd=tmp,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kw,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.communicate()
            raise ProgramFailed(f"timeout after {timeout_s}s", kind="timeout")
        if proc.returncode != 0:
            tail = (stderr or stdout or "")[-_OUTPUT_TAIL:]
            raise ProgramFailed(f"exit code {proc.returncode}: {tail}", kind="crash", output_tail=tail)
        if not out_path.exists():
            raise ProgramFailed("program produced no result", kind="no_result")
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ProgramFailed(f"result is not valid JSON: {e}", kind="bad_output")


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
    except (ProcessLookupError, OSError):
        pass
    if proc.poll() is None:
        proc.kill()


class SubprocessEvaluator:
    """Base class: subclasses set the metadata and implement `score_solution`.

    `evaluate` never raises. Every failure becomes an invalid EvalResult.
    """

    problem_id: int
    name: str
    best_known: float
    timeout_s: int
    description: str

    def seed_program(self) -> str:
        raise NotImplementedError

    def score_solution(self, solution: Any) -> EvalResult:
        """Score the JSON-decoded return value of construct(). Runs in the harness process."""
        raise NotImplementedError

    def evaluate(self, program: str) -> EvalResult:
        try:
            solution = run_program(program, self.timeout_s)
        except ProgramFailed as e:
            return invalid(str(e), side_info=failure_side_info(e, self.timeout_s))
        try:
            return self.score_solution(solution)
        except Exception as e:  # malformed solution shapes etc.
            return invalid(
                f"scoring failed: {type(e).__name__}: {e}",
                side_info={"stage": "score", "failure": "exception", "exception": type(e).__name__,
                           "message": str(e)[:_SIDE_INFO_TAIL], "solution_type": type(solution).__name__},
            )
