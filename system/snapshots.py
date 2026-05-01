"""
system/snapshots.py

DIX VISION v42.2 — Tier-0 Step 0: Periodic in-memory snapshots.

The :class:`SnapshotEngine` is a light-weight orchestrator that pairs
a set of domain projectors (market / portfolio / system / governance,
plus the bounded :mod:`mind.knowledge_store`) with the ledger-driven
event stream. Every N events *or* every T wall-clock nanoseconds it
takes an atomic capture of every projector's read-model and stashes it
in a bounded ring so the :class:`system.state_reconstructor.StateReconstructor`
can resume from the nearest snapshot without replaying the whole
ledger.

Hard rules (see docs/ARCHITECTURE_V42_2_TIER0.md §1):

    1. Snapshots are read-only derived state. They are *never* the
       authoritative source — that is and remains the append-only
       ledger in :mod:`state.ledger.event_store`.
    2. Each snapshot captures a ``cursor`` (sequence and wall-clock
       nanoseconds) so replay can pick up exactly where the snapshot
       left off.
    3. Snapshot capture is deterministic: for a given ordered event
       feed and projector set the resulting snapshot payloads are
       bit-identical across runs.
    4. The engine never writes to the ledger, never mutates domain
       state, and never holds a lock across a projector's apply()
       call — it only calls ``projector.snapshot()`` which is
       required to return an immutable copy.

This module is deliberately transport-agnostic: on-disk durability is
handled by :mod:`state.ledger.snapshot_manager`. This engine exists
to serve :class:`StateReconstructor` cheaply and in-process.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

try:  # T0-4 (PR #7) exposes these as the canonical hot-path clock.
    from system.time_source import now_ns, wall_ns  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - before T0-4 merges
    def now_ns() -> int:
        """Monotonic nanoseconds; never goes backwards."""
        return time.monotonic_ns()

    def wall_ns() -> int:
        """Wall-clock nanoseconds since Unix epoch."""
        return time.time_ns()


@runtime_checkable
class Projector(Protocol):
    """Minimal projector contract used by the snapshot engine."""

    def apply(self, event: Mapping[str, Any]) -> None: ...

    def snapshot(self) -> Any: ...


@dataclass(frozen=True)
class SnapshotCursor:
    """Position in the event stream that a snapshot corresponds to."""

    sequence: int
    wall_ns: int


@dataclass(frozen=True)
class Snapshot:
    """Immutable, timestamped capture of every projector's read-model."""

    cursor: SnapshotCursor
    event_count: int
    projectors: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotPolicy:
    """Trigger rules. Either threshold fires a capture."""

    events_per_snapshot: int = 1_000
    nanoseconds_per_snapshot: int = 5 * 1_000_000_000  # 5 s wall
    max_ring_size: int = 32

    def validate(self) -> None:
        if self.events_per_snapshot <= 0:
            raise ValueError("events_per_snapshot must be > 0")
        if self.nanoseconds_per_snapshot <= 0:
            raise ValueError("nanoseconds_per_snapshot must be > 0")
        if self.max_ring_size <= 0:
            raise ValueError("max_ring_size must be > 0")


class SnapshotEngine:
    """Thread-safe periodic-snapshot controller.

    Typical use from the ledger writer side:

        engine = SnapshotEngine(
            projectors={"market": market_proj, ...},
            policy=SnapshotPolicy(),
        )
        for event in ledger_events:
            engine.on_event(event)

    Typical use from the reconstructor side:

        snap = engine.latest()
        if snap is not None:
            for event in ledger.events_after(snap.cursor):
                ...  # replay to present
    """

    def __init__(
        self,
        projectors: Mapping[str, Projector],
        policy: SnapshotPolicy | None = None,
        *,
        clock_wall_ns: Callable[[], int] = wall_ns,
        clock_monotonic_ns: Callable[[], int] = now_ns,
    ) -> None:
        if not projectors:
            raise ValueError("at least one projector is required")
        self._policy = policy or SnapshotPolicy()
        self._policy.validate()
        self._projectors: dict[str, Projector] = dict(projectors)
        self._clock_wall = clock_wall_ns
        self._clock_mono = clock_monotonic_ns
        self._lock = threading.RLock()
        self._ring: deque[Snapshot] = deque(maxlen=self._policy.max_ring_size)
        self._events_since_last: int = 0
        self._total_events: int = 0
        self._last_capture_mono_ns: int = self._clock_mono()
        self._last_sequence: int = 0
        # ``None`` means 'no event has supplied a wall_ns yet'; we cannot
        # use 0 as a sentinel because an event may legitimately stamp
        # wall_ns=0 (Unix epoch / test feeds) and we must preserve that
        # determinism — see determinism rule 3 in the module docstring.
        self._last_wall_ns: int | None = None

    # ─────── ingestion ──────────────────────────────────────────────

    def on_event(self, event: Mapping[str, Any]) -> Snapshot | None:
        """Feed one event through every projector and maybe snapshot.

        The snapshot is taken *after* the projectors apply the event,
        so replay from the returned cursor re-plays no events already
        captured. Returns the new :class:`Snapshot` if one was taken,
        else ``None``.
        """
        with self._lock:
            for p in self._projectors.values():
                p.apply(event)
            self._events_since_last += 1
            self._total_events += 1
            self._last_sequence = int(event.get("sequence", self._last_sequence))
            ts = event.get("wall_ns")
            if ts is not None:
                self._last_wall_ns = int(ts)
            return self._capture_if_due_locked()

    # ─────── snapshot access ────────────────────────────────────────

    def snapshot_now(self) -> Snapshot:
        """Force a capture regardless of policy and return it."""
        with self._lock:
            return self._capture_locked()

    def latest(self) -> Snapshot | None:
        with self._lock:
            return self._ring[-1] if self._ring else None

    def latest_at_or_before(self, sequence: int) -> Snapshot | None:
        """Nearest snapshot whose cursor.sequence ≤ ``sequence``."""
        with self._lock:
            best: Snapshot | None = None
            for snap in self._ring:
                if snap.cursor.sequence <= sequence:
                    if best is None or snap.cursor.sequence > best.cursor.sequence:
                        best = snap
            return best

    def latest_at_or_before_wall_ns(self, wall_ns: int) -> Snapshot | None:
        """Nearest snapshot whose cursor.wall_ns ≤ ``wall_ns``."""
        with self._lock:
            best: Snapshot | None = None
            for snap in self._ring:
                if snap.cursor.wall_ns <= wall_ns:
                    if best is None or snap.cursor.wall_ns > best.cursor.wall_ns:
                        best = snap
            return best

    def all(self) -> tuple[Snapshot, ...]:
        with self._lock:
            return tuple(self._ring)

    def __len__(self) -> int:
        with self._lock:
            return len(self._ring)

    # ─────── internals ──────────────────────────────────────────────

    def _capture_if_due_locked(self) -> Snapshot | None:
        policy = self._policy
        due = (
            self._events_since_last >= policy.events_per_snapshot
            or (self._clock_mono() - self._last_capture_mono_ns)
            >= policy.nanoseconds_per_snapshot
        )
        if not due:
            return None
        return self._capture_locked()

    def _capture_locked(self) -> Snapshot:
        projector_views = {
            name: proj.snapshot() for name, proj in self._projectors.items()
        }
        # Prefer the wall_ns carried on the latest event; only fall back
        # to the clock when no event has supplied one. ``or`` would be a
        # bug here because an event-supplied wall_ns=0 is valid.
        cursor_wall = (
            self._last_wall_ns
            if self._last_wall_ns is not None
            else self._clock_wall()
        )
        snap = Snapshot(
            cursor=SnapshotCursor(
                sequence=self._last_sequence,
                wall_ns=cursor_wall,
            ),
            event_count=self._total_events,
            projectors=projector_views,
        )
        self._ring.append(snap)
        self._events_since_last = 0
        self._last_capture_mono_ns = self._clock_mono()
        return snap


__all__ = [
    "Projector",
    "Snapshot",
    "SnapshotCursor",
    "SnapshotEngine",
    "SnapshotPolicy",
]
