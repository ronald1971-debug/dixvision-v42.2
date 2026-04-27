"""Liveness checker — classifies engines from heartbeat snapshots.

Pure function: takes a snapshot dict + thresholds + ``ts_ns`` and emits
a tuple of :class:`EngineLiveness` rows. Determinism guaranteed by
sorted-by-engine output ordering.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class LivenessState(StrEnum):
    ALIVE = "ALIVE"
    SUSPECT = "SUSPECT"
    DEAD = "DEAD"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class EngineLiveness:
    engine: str
    state: LivenessState
    last_seen_ns: int | None
    age_ns: int | None


class LivenessChecker:
    """Apply per-engine thresholds to a heartbeat snapshot."""

    name: str = "liveness_checker"
    spec_id: str = "SYS-HEALTH-LV-01"

    __slots__ = ("_suspect_after_ns", "_dead_after_ns")

    def __init__(
        self,
        *,
        suspect_after_ns: int = 1_500_000_000,
        dead_after_ns: int = 5_000_000_000,
    ) -> None:
        if suspect_after_ns <= 0 or dead_after_ns <= 0:
            raise ValueError("thresholds must be positive")
        if dead_after_ns < suspect_after_ns:
            raise ValueError("dead_after_ns must be >= suspect_after_ns")
        self._suspect_after_ns = suspect_after_ns
        self._dead_after_ns = dead_after_ns

    def classify(
        self,
        *,
        ts_ns: int,
        heartbeats: Mapping[str, int],
        engines: tuple[str, ...] | None = None,
    ) -> tuple[EngineLiveness, ...]:
        names = (
            tuple(sorted(engines))
            if engines is not None
            else tuple(sorted(heartbeats.keys()))
        )
        out: list[EngineLiveness] = []
        for engine in names:
            last = heartbeats.get(engine)
            if last is None:
                out.append(
                    EngineLiveness(
                        engine=engine,
                        state=LivenessState.UNKNOWN,
                        last_seen_ns=None,
                        age_ns=None,
                    )
                )
                continue
            age = ts_ns - last
            if age >= self._dead_after_ns:
                state = LivenessState.DEAD
            elif age >= self._suspect_after_ns:
                state = LivenessState.SUSPECT
            else:
                state = LivenessState.ALIVE
            out.append(
                EngineLiveness(
                    engine=engine,
                    state=state,
                    last_seen_ns=last,
                    age_ns=age,
                )
            )
        return tuple(out)


__all__ = ["EngineLiveness", "LivenessChecker", "LivenessState"]
