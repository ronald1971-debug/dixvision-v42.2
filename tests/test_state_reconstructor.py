"""Tests for system.state_reconstructor — T0-0 deterministic replay."""
from __future__ import annotations

from typing import Any

import pytest

from system.snapshots import SnapshotEngine, SnapshotPolicy
from system.state_reconstructor import (
    OutOfRangeError,
    StateReconstructor,
)


class TallyProjector:
    def __init__(self) -> None:
        self.total = 0
        self.last_key: str | None = None

    def apply(self, event: dict) -> None:
        p = event.get("payload", {}) or {}
        self.total += int(p.get("delta", 0))
        self.last_key = str(p.get("key", self.last_key))

    def snapshot(self) -> dict[str, Any]:
        return {"total": self.total, "last_key": self.last_key}


class RestorableTally(TallyProjector):
    def restore(self, view: dict) -> None:
        self.total = int(view.get("total", 0))
        self.last_key = view.get("last_key")


def _events() -> list[dict]:
    return [
        {"sequence": 1, "wall_ns": 1_000, "payload": {"delta": 1, "key": "a"}},
        {"sequence": 2, "wall_ns": 2_000, "payload": {"delta": 5, "key": "b"}},
        {"sequence": 3, "wall_ns": 3_000, "payload": {"delta": -2, "key": "c"}},
        {"sequence": 4, "wall_ns": 4_000, "payload": {"delta": 10, "key": "d"}},
        {"sequence": 5, "wall_ns": 5_000, "payload": {"delta": 100, "key": "e"}},
    ]


# ─────── rebuild without snapshots ──────────────────────────────────────


def test_rebuild_latest_from_genesis() -> None:
    events = _events()
    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: events,
    )
    state = rc.rebuild_latest()
    assert state.projectors["tally"]["total"] == 114
    assert state.projectors["tally"]["last_key"] == "e"
    assert state.sequence == 5
    assert state.event_count == 5
    assert state.resumed_from_snapshot is False


def test_rebuild_at_specific_sequence() -> None:
    events = _events()
    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: events,
    )
    state = rc.rebuild_at(3)
    # events 1..3: 1 + 5 + -2 = 4
    assert state.projectors["tally"]["total"] == 4
    assert state.projectors["tally"]["last_key"] == "c"
    assert state.sequence == 3


def test_rebuild_at_wall_time() -> None:
    events = _events()
    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: events,
    )
    # cutoff 3_500 ns includes events 1..3 only
    state = rc.rebuild(3_500)
    assert state.projectors["tally"]["total"] == 4
    assert state.projectors["tally"]["last_key"] == "c"


def test_rebuild_past_ledger_end_raises() -> None:
    events = _events()
    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: events,
    )
    with pytest.raises(OutOfRangeError):
        rc.rebuild_at(999)


def test_negative_inputs_rejected() -> None:
    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: [],
    )
    with pytest.raises(ValueError):
        rc.rebuild_at(-1)
    with pytest.raises(ValueError):
        rc.rebuild(-1)


def test_empty_factories_rejected() -> None:
    with pytest.raises(ValueError):
        StateReconstructor(projector_factories={}, event_feed=lambda: [])


# ─────── determinism ────────────────────────────────────────────────────


def test_rebuild_is_bit_identical_across_runs() -> None:
    events = _events()
    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: list(events),
    )
    a = rc.rebuild_latest()
    b = rc.rebuild_latest()
    assert a.projectors == b.projectors
    assert a.sequence == b.sequence
    assert a.event_count == b.event_count


def test_rebuild_does_not_mutate_live_projectors() -> None:
    """Factory must produce fresh projectors per rebuild; calling
    rebuild_latest() twice in a row must not double-count."""
    events = _events()
    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: list(events),
    )
    first = rc.rebuild_latest()
    second = rc.rebuild_latest()
    assert first.projectors["tally"]["total"] == 114
    assert second.projectors["tally"]["total"] == 114


# ─────── snapshot fast-forward ──────────────────────────────────────────


def test_rebuild_resumes_from_snapshot_when_projector_supports_restore() -> None:
    events = _events()
    eng = SnapshotEngine(
        projectors={"tally": RestorableTally()},
        policy=SnapshotPolicy(
            events_per_snapshot=2,
            nanoseconds_per_snapshot=10**18,
            max_ring_size=8,
        ),
    )
    for ev in events[:4]:  # feed first 4 events -> two snapshots at seq 2, 4
        eng.on_event(ev)

    rc = StateReconstructor(
        projector_factories={"tally": RestorableTally},
        event_feed=lambda: events,
        snapshot_engine=eng,
    )
    state = rc.rebuild_latest()
    assert state.resumed_from_snapshot is True
    # full-run total
    assert state.projectors["tally"]["total"] == 114
    assert state.sequence == 5


def test_rebuild_falls_back_to_genesis_when_projector_lacks_restore() -> None:
    """A projector without ``restore`` must still produce the correct
    final state — it just replays from genesis."""
    events = _events()
    eng = SnapshotEngine(
        projectors={"tally": TallyProjector()},
        policy=SnapshotPolicy(events_per_snapshot=2,
                              nanoseconds_per_snapshot=10**18,
                              max_ring_size=8),
    )
    for ev in events[:4]:
        eng.on_event(ev)

    rc = StateReconstructor(
        projector_factories={"tally": TallyProjector},
        event_feed=lambda: events,
        snapshot_engine=eng,
    )
    state = rc.rebuild_latest()
    assert state.projectors["tally"]["total"] == 114
    assert state.sequence == 5


def test_rebuild_at_uses_nearest_snapshot_before_target() -> None:
    events = _events()
    eng = SnapshotEngine(
        projectors={"tally": RestorableTally()},
        policy=SnapshotPolicy(events_per_snapshot=2,
                              nanoseconds_per_snapshot=10**18,
                              max_ring_size=8),
    )
    for ev in events:
        eng.on_event(ev)

    rc = StateReconstructor(
        projector_factories={"tally": RestorableTally},
        event_feed=lambda: events,
        snapshot_engine=eng,
    )
    state = rc.rebuild_at(3)
    assert state.sequence == 3
    assert state.projectors["tally"]["total"] == 4
    # best snapshot ≤ 3 is the seq-2 snapshot, so resumed_from_snapshot
    # must be true.
    assert state.resumed_from_snapshot is True


# ─────── regression: bugs reported on PR #10 ────────────────────────────


def test_rebuild_by_wall_time_never_uses_snapshot_past_target() -> None:
    """Regression for Devin Review BUG_0001: ``rebuild(at_timestamp_ns)``
    must not pick a snapshot whose cursor is past the wall-time target.

    Scenario: events land at wall 1000..5000, snapshots captured at
    seq 2 (wall=2000) and seq 4 (wall=4000). Caller asks for wall=2500.
    The reconstructor must NOT resume from the seq-4 snapshot; doing so
    would poison projectors with events 3 and 4 that sit past 2500ns.
    """
    events = _events()
    eng = SnapshotEngine(
        projectors={"tally": RestorableTally()},
        policy=SnapshotPolicy(
            events_per_snapshot=2,
            nanoseconds_per_snapshot=10**18,
            max_ring_size=8,
        ),
    )
    for ev in events:
        eng.on_event(ev)

    rc = StateReconstructor(
        projector_factories={"tally": RestorableTally},
        event_feed=lambda: events,
        snapshot_engine=eng,
    )
    # events 1 (delta=1) + 2 (delta=5) = 6; events 3+ are past 2500ns
    state = rc.rebuild(2_500)
    assert state.projectors["tally"]["total"] == 6
    assert state.projectors["tally"]["last_key"] == "b"
    assert state.wall_ns == 2_000
    assert state.sequence == 2


def test_rebuild_by_wall_time_uses_eligible_snapshot() -> None:
    """Sibling of the regression above: when the snapshot cursor DOES
    fit under the wall-time target, the reconstructor should still
    fast-forward — i.e. the fix for BUG_0001 must not regress the
    fast-path for wall-time rebuilds."""
    events = _events()
    eng = SnapshotEngine(
        projectors={"tally": RestorableTally()},
        policy=SnapshotPolicy(
            events_per_snapshot=2,
            nanoseconds_per_snapshot=10**18,
            max_ring_size=8,
        ),
    )
    for ev in events[:4]:  # snapshots at seq 2 (wall=2000), seq 4 (wall=4000)
        eng.on_event(ev)

    rc = StateReconstructor(
        projector_factories={"tally": RestorableTally},
        event_feed=lambda: events,
        snapshot_engine=eng,
    )
    # cutoff 4_500 includes events 1..4
    state = rc.rebuild(4_500)
    assert state.projectors["tally"]["total"] == 14
    assert state.resumed_from_snapshot is True


def test_partial_hydration_resets_restored_projectors_to_genesis() -> None:
    """Regression for Devin Review BUG_0002: when one projector cannot
    restore, any projector that DID restore must be reset back to
    genesis — otherwise the forced genesis replay double-counts
    pre-snapshot events on the restored projector.

    Scenario: ``a`` (RestorableTally) restores from the snapshot,
    ``b`` (TallyProjector, no restore) forces genesis replay.  Before
    the fix, ``a.total`` ended up at 114+14 = 128; after the fix both
    projectors reach 114 cleanly.
    """
    events = _events()
    eng = SnapshotEngine(
        projectors={"a": RestorableTally(), "b": TallyProjector()},
        policy=SnapshotPolicy(
            events_per_snapshot=2,
            nanoseconds_per_snapshot=10**18,
            max_ring_size=8,
        ),
    )
    for ev in events[:4]:
        eng.on_event(ev)

    rc = StateReconstructor(
        projector_factories={
            "a": RestorableTally,
            "b": TallyProjector,
        },
        event_feed=lambda: events,
        snapshot_engine=eng,
    )
    state = rc.rebuild_latest()
    assert state.resumed_from_snapshot is False  # b forced genesis
    assert state.projectors["a"]["total"] == 114
    assert state.projectors["b"]["total"] == 114
    assert state.projectors["a"]["last_key"] == "e"
    assert state.projectors["b"]["last_key"] == "e"
