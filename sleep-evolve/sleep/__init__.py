"""Sleep policy registry. Add new policies to REGISTRY (name -> class taking **params)."""

from __future__ import annotations

from typing import Any

from sleep.base import SleepPolicy
from sleep.full import FullSleep
from sleep.lite import SleepLite
from sleep.no_sleep import NoSleep
from sleep.summarize import Summarize

REGISTRY: dict[str, type] = {
    NoSleep.name: NoSleep,
    Summarize.name: Summarize,
    FullSleep.name: FullSleep,
    SleepLite.name: SleepLite,
}


def make_policy(name: str, params: dict[str, Any] | None = None) -> SleepPolicy:
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown sleep policy {name!r}; known: {sorted(REGISTRY)}") from None
    return cls(**(params or {}))


__all__ = ["SleepPolicy", "REGISTRY", "make_policy"]
