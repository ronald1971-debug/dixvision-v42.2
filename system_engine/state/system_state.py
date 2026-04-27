"""SystemState — Dyon's authoritative read-only "what is happening".

This object is *write-once-per-tick* from the system_engine internal
loop and *read-only* outside it. It exposes a deterministic snapshot
that Governance can consume via :class:`SystemStateSnapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.events import HazardEvent


@dataclass(frozen=True, slots=True)
class SystemStateSnapshot:
    ts_ns: int
    heartbeats: tuple[tuple[str, int], ...] = ()
    open_hazards: tuple[HazardEvent, ...] = ()
    hazard_count: int = 0


class SystemState:
    """Mutable holder; produces immutable snapshots."""

    name: str = "system_state"
    spec_id: str = "CORE-18"

    __slots__ = ("_heartbeats", "_open_hazards", "_hazard_count")

    def __init__(self) -> None:
        self._heartbeats: dict[str, int] = {}
        self._open_hazards: dict[str, HazardEvent] = {}
        self._hazard_count = 0

    def record_heartbeat(self, *, engine: str, ts_ns: int) -> None:
        self._heartbeats[engine] = ts_ns

    def record_hazard(self, hazard: HazardEvent) -> None:
        # Latest-wins per code; counter is monotonic.
        self._open_hazards[hazard.code] = hazard
        self._hazard_count += 1

    def clear_hazard(self, code: str) -> None:
        self._open_hazards.pop(code, None)

    def snapshot(self, ts_ns: int) -> SystemStateSnapshot:
        heartbeats = tuple(sorted(self._heartbeats.items()))
        hazards = tuple(
            self._open_hazards[k] for k in sorted(self._open_hazards.keys())
        )
        return SystemStateSnapshot(
            ts_ns=ts_ns,
            heartbeats=heartbeats,
            open_hazards=hazards,
            hazard_count=self._hazard_count,
        )


__all__ = ["SystemState", "SystemStateSnapshot"]
