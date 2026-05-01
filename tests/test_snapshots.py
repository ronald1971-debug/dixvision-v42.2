"""Tests for system.snapshots — T0-0 periodic in-memory snapshotting."""
from __future__ import annotations

from typing import Any

import pytest

from system.snapshots import (
    Snapshot,
    SnapshotEngine,
    SnapshotPolicy,
)


class CountingProjector:
    """Minimal projector that counts events and exposes a deep-copyable
    snapshot of its internal tally."""

    def __init__(self) -> None:
        self.count = 0
        self.last_payload: dict[str, Any] | None = None

    def apply(self, event: dict) -> None:
        self.count += 1
        self.last_payload = dict(event.get("payload", {}))

    def snapshot(self) -> dict[str, Any]:
        return {"count": self.count, "last_payload": dict(self.last_payload or {})}

    # deliberate: no restore() — tests fallback hydration path


class RestorableProjector(CountingProjector):
    def restore(self, view: dict) -> None:
        self.count = int(view.get("count", 0))
        self.last_payload = dict(view.get("last_payload") or {})


def _event(seq: int, payload: dict | None = None, wall_ns: int | None = None) -> dict:
    return {
        "sequence": seq,
        "wall_ns": wall_ns if wall_ns is not None else seq * 1_000_000,
        "event_type": "MARKET",
        "sub_type": "TICK",
        "payload": payload or {"seq": seq},
    }


# ─────── policy ─────────────────────────────────────────────────────────


def test_policy_rejects_non_positive_thresholds() -> None:
    with pytest.raises(ValueError):
        SnapshotPolicy(events_per_snapshot=0).validate()
    with pytest.raises(ValueError):
        SnapshotPolicy(nanoseconds_per_snapshot=0).validate()
    with pytest.raises(ValueError):
        SnapshotPolicy(max_ring_size=0).validate()


def test_engine_requires_at_least_one_projector() -> None:
    with pytest.raises(ValueError):
        SnapshotEngine(projectors={})


# ─────── capture cadence ────────────────────────────────────────────────


def test_captures_every_n_events() -> None:
    mono_counter = {"v": 0}

    def fake_mono() -> int:
        mono_counter["v"] += 1
        return mono_counter["v"]

    eng = SnapshotEngine(
        {"m": CountingProjector()},
        SnapshotPolicy(events_per_snapshot=3,
                       nanoseconds_per_snapshot=10**18,
                       max_ring_size=16),
        clock_monotonic_ns=fake_mono,
        clock_wall_ns=lambda: 42,
    )
    caps = [eng.on_event(_event(i)) for i in range(1, 7)]
    assert caps[0] is None
    assert caps[1] is None
    assert caps[2] is not None
    assert caps[3] is None
    assert caps[4] is None
    assert caps[5] is not None
    assert len(eng) == 2


def test_captures_when_time_elapses() -> None:
    clock = {"v": 0}

    def fake_mono() -> int:
        return clock["v"]

    eng = SnapshotEngine(
        {"m": CountingProjector()},
        SnapshotPolicy(events_per_snapshot=10**9,
                       nanoseconds_per_snapshot=100,
                       max_ring_size=4),
        clock_monotonic_ns=fake_mono,
        clock_wall_ns=lambda: 0,
    )
    # first event: no time passed -> no capture
    assert eng.on_event(_event(1)) is None
    # advance clock past the ns threshold
    clock["v"] = 1_000
    snap = eng.on_event(_event(2))
    assert snap is not None
    assert snap.cursor.sequence == 2


def test_force_snapshot_now_ignores_policy() -> None:
    eng = SnapshotEngine({"m": CountingProjector()}, SnapshotPolicy())
    eng.on_event(_event(1))
    snap = eng.snapshot_now()
    assert isinstance(snap, Snapshot)
    assert snap.cursor.sequence == 1


def test_ring_is_bounded() -> None:
    eng = SnapshotEngine(
        {"m": CountingProjector()},
        SnapshotPolicy(events_per_snapshot=1,
                       nanoseconds_per_snapshot=10**18,
                       max_ring_size=3),
    )
    for i in range(1, 11):
        eng.on_event(_event(i))
    assert len(eng) == 3
    latest = eng.latest()
    assert latest is not None
    assert latest.cursor.sequence == 10


def test_latest_at_or_before_picks_correct_snapshot() -> None:
    eng = SnapshotEngine(
        {"m": CountingProjector()},
        SnapshotPolicy(events_per_snapshot=1,
                       nanoseconds_per_snapshot=10**18,
                       max_ring_size=16),
    )
    for i in [1, 2, 5, 9]:
        eng.on_event(_event(i))
    assert eng.latest_at_or_before(4).cursor.sequence == 2
    assert eng.latest_at_or_before(5).cursor.sequence == 5
    assert eng.latest_at_or_before(100).cursor.sequence == 9
    assert eng.latest_at_or_before(0) is None


# ─────── determinism ────────────────────────────────────────────────────


def test_snapshot_payloads_are_deterministic() -> None:
    """Same event feed + same policy + same fake clock => identical snapshots."""

    def make_engine() -> SnapshotEngine:
        clock = {"v": 0}
        return SnapshotEngine(
            {"m": CountingProjector()},
            SnapshotPolicy(events_per_snapshot=2,
                           nanoseconds_per_snapshot=10**18,
                           max_ring_size=16),
            clock_monotonic_ns=lambda: clock["v"],
            clock_wall_ns=lambda: 777,
        )

    feed = [_event(i, {"x": i, "y": i * 2}) for i in range(1, 11)]
    a = make_engine()
    b = make_engine()
    for ev in feed:
        a.on_event(ev)
        b.on_event(ev)
    assert [s.projectors for s in a.all()] == [s.projectors for s in b.all()]
    assert [s.cursor for s in a.all()] == [s.cursor for s in b.all()]
    assert [s.event_count for s in a.all()] == [s.event_count for s in b.all()]


def test_snapshot_is_immutable_after_capture() -> None:
    """Mutating the projector after capture must not bleed into the stored view."""
    proj = CountingProjector()
    eng = SnapshotEngine(
        {"m": proj},
        SnapshotPolicy(events_per_snapshot=1,
                       nanoseconds_per_snapshot=10**18,
                       max_ring_size=4),
    )
    snap = eng.on_event(_event(1, {"v": 100}))
    assert snap is not None
    stored = snap.projectors["m"]
    proj.apply(_event(2, {"v": 999}))
    assert stored["count"] == 1
    assert stored["last_payload"]["v"] == 100


# ─────── regression: determinism with wall_ns=0 (Devin Review) ─────────


def test_capture_preserves_event_wall_ns_zero() -> None:
    """Regression for Devin Review PR #10: an event that stamps
    ``wall_ns=0`` must be honored verbatim on the snapshot cursor,
    not quietly replaced with a non-deterministic clock read.

    Hard rule 3 in the module docstring mandates bit-deterministic
    capture; relying on ``self._last_wall_ns or self._clock_wall()``
    used Python's falsy-zero behavior to drop a legitimate zero.
    """
    proj = CountingProjector()
    engine = SnapshotEngine(
        projectors={"m": proj},
        policy=SnapshotPolicy(events_per_snapshot=1, nanoseconds_per_snapshot=10**18),
        clock_wall_ns=lambda: 123_456_789,  # would be used if fallback kicked in
    )
    snap = engine.on_event(_event(1, wall_ns=0))
    assert snap is not None
    assert snap.cursor.wall_ns == 0

    # latest_at_or_before_wall_ns(0) must now locate the capture and
    # latest_at_or_before_wall_ns(-1) must not.
    assert engine.latest_at_or_before_wall_ns(0) is snap
    assert engine.latest_at_or_before_wall_ns(-1) is None


def test_capture_uses_clock_when_no_event_wall_ns_yet() -> None:
    """The other side of the same fix: when *no* event has stamped a
    wall_ns yet, the capture falls back to the injected clock."""
    proj = CountingProjector()
    engine = SnapshotEngine(
        projectors={"m": proj},
        policy=SnapshotPolicy(events_per_snapshot=10**9, nanoseconds_per_snapshot=10**18),
        clock_wall_ns=lambda: 7_777,
    )
    snap = engine.snapshot_now()
    assert snap.cursor.wall_ns == 7_777
