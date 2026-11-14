"""Loader for benchmarks/alphaevolve.yaml (the AlphaEvolve benchmark registry).

The YAML is the single source of truth: every number carries a source citation, and a value that
could not be found is `value: null` plus a `missing_note`. `load_registry()` validates that
contract with pydantic, so a malformed edit fails at load time (and in tests/test_benchmark.py).

Instances: a problem lists every instance AlphaEvolve reported. A run is scored only against the
instance whose `params` equal the params its evaluator declares (see evaluator_instances.py);
`Problem.match_instance` returns None on any mismatch, and callers must refuse to compare.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REGISTRY_PATH = Path(__file__).with_name("alphaevolve.yaml")
STATUSES = ("world_record", "former_record", "worse_than_record", "matched_optimal", "unlisted")
PENDING = "pending"


class ValueRecord(BaseModel):
    """A sourced number. `value: null` is allowed only with a `missing_note`."""

    model_config = ConfigDict(extra="forbid")
    value: float | None = None
    source: str = Field(min_length=1)
    holder: str | None = None
    note: str | None = None
    missing_note: str | None = None

    @model_validator(mode="after")
    def _missing_needs_note(self):
        if self.value is None and not self.missing_note:
            raise ValueError(f"value is null but no missing_note (source: {self.source!r})")
        return self


class ThirdParty(BaseModel):
    model_config = ConfigDict(extra="forbid")
    system: str
    value: float | None = None
    comparable: bool = True  # same instance as the registry instance
    evaluations_or_cost: str | None = None
    verification: str | None = None
    source: str = Field(min_length=1)
    note: str | None = None
    missing_note: str | None = None

    @model_validator(mode="after")
    def _missing_needs_note(self):
        if self.value is None and not self.missing_note:
            raise ValueError(f"third-party {self.system!r}: value is null but no missing_note")
        return self


class OtherReported(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    source: str


class Instance(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    params: dict[str, Any] = Field(min_length=1)
    quantity: str
    direction: Literal["maximize", "minimize"]
    score_transform: Literal["identity", "negate"]
    alphaevolve_value: ValueRecord
    prior_best_value: ValueRecord
    current_best_value: ValueRecord
    match_tolerance: float = Field(ge=0)
    verification_tolerance: float = Field(ge=0)
    construction: str | None = None  # path (repo-relative) to the published construction as JSON
    published_scorer: str | None = None  # key in benchmarks.verification.PUBLISHED_SCORERS
    other_reported: list[OtherReported] = Field(default_factory=list)
    third_party: list[ThirdParty] = Field(default_factory=list)

    @model_validator(mode="after")
    def _direction_matches_transform(self):
        want = "identity" if self.direction == "maximize" else "negate"
        if self.score_transform != want:
            raise ValueError(f"instance {self.id}: direction {self.direction} needs score_transform {want}")
        return self

    # --- conversions between the published value and our higher-is-better score -----------------
    def to_score(self, value: float | None) -> float | None:
        if value is None:
            return None
        return float(value) if self.score_transform == "identity" else -float(value)

    def from_score(self, score: float | None) -> float | None:
        return self.to_score(score)  # both transforms are involutions

    @property
    def ae_value(self) -> float | None:
        return self.alphaevolve_value.value

    @property
    def ae_score(self) -> float | None:
        return self.to_score(self.alphaevolve_value.value)


class Problem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    problem_id: int
    name: str
    tier: Literal[1, 2]
    page: str
    paper_ref: str
    status: Literal["world_record", "former_record", "worse_than_record", "matched_optimal", "unlisted"]
    status_raw: str | None
    status_note: str | None = None
    evaluator: str  # "module:Class" or "pending"
    instance_note: str | None = None
    instances: list[Instance] = Field(min_length=1)

    @property
    def evaluator_pending(self) -> bool:
        return self.evaluator == PENDING

    def instance(self, instance_id: str) -> Instance:
        for inst in self.instances:
            if inst.id == instance_id:
                return inst
        raise KeyError(f"problem {self.problem_id} has no instance {instance_id!r}")

    def match_instance(self, params: dict[str, Any] | None) -> Instance | None:
        """The instance whose params equal `params` exactly, else None (refuse to compare)."""
        if not params:
            return None
        for inst in self.instances:
            if _norm(inst.params) == _norm(params):
                return inst
        return None


class Registry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    retrieved: str
    sources: dict[str, str]
    status_labels: dict[str, str]
    problems: list[Problem]
    sha256: str = ""

    @model_validator(mode="after")
    def _unique(self):
        ids = [p.problem_id for p in self.problems]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate problem_id in registry")
        for p in self.problems:
            iids = [i.id for i in p.instances]
            if len(iids) != len(set(iids)):
                raise ValueError(f"problem {p.problem_id}: duplicate instance ids")
        return self

    def get(self, problem_id: int) -> Problem | None:
        for p in self.problems:
            if p.problem_id == problem_id:
                return p
        return None

    def __contains__(self, problem_id: int) -> bool:
        return self.get(problem_id) is not None

    @property
    def ids(self) -> list[int]:
        return [p.problem_id for p in self.problems]


def _norm(params: dict[str, Any]) -> dict[str, Any]:
    """Compare params by value: ints/floats equal if numerically equal, lists as tuples."""
    out = {}
    for k, v in params.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            out[str(k)] = tuple(v) if isinstance(v, list) else v
        else:
            out[str(k)] = float(v)
    return out


def load_registry(path: str | Path | None = None) -> Registry:
    p = Path(path) if path else REGISTRY_PATH
    raw = p.read_bytes()
    reg = Registry.model_validate(yaml.safe_load(raw))
    reg.sha256 = hashlib.sha256(raw).hexdigest()
    return reg


@lru_cache(maxsize=1)
def default_registry() -> Registry:
    return load_registry()
