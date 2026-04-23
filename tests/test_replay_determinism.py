"""
tests/test_replay_determinism.py

DIX VISION v42.2 — Tier-0 Step 14: deterministic replay.

Core invariant (``docs/ARCHITECTURE_V42_2_TIER0.md`` §15):

    Same input → identical output (bit-level where possible)

Every decision the system makes must be reproducible from the ledger
alone. This file is the executable specification of that guarantee.
It pins the following concrete properties:

    (a) A fresh event stream fed through two independent projector
        sets produces bit-identical projector snapshots.
    (b) Running the same events via the live :class:`SnapshotEngine`
        (with periodic captures) versus genesis-replay through the
        :class:`StateReconstructor` produces bit-identical read-models.
    (c) Rebuilding at an intermediate cursor (``rebuild_at``) is
        indistinguishable from rebuilding the whole feed and stopping
        at that cursor by hand.
    (d) :class:`FastRiskCache.version_id` is a pure function of
        ``(version, updated_at_ns)`` — two caches fed identical
        update sequences stamp identical ``version_id`` strings.
    (e) The snapshot cursor preserves an event-supplied ``wall_ns``
        exactly (including 0), so replay driven by wall-time matches
        replay driven by sequence.

These tests are deliberately strict: they compare full projector-view
dicts, not hashes — a silent reordering in any projector will fail.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping

import pytest

from system.fast_risk_cache import FastRiskCache, _compute_version_id
from system.snapshots import SnapshotEngine, SnapshotPolicy
from system.state_reconstructor import StateReconstructor


# ─────── test fixtures ──────────────────────────────────────────────────


class OrderBookProjector:
    """Tally payloads by ``event_type`` + ``sub_type``.

    Deliberately keeps ``last_payload`` around so a reordering would
    surface in the snapshot comparison.
    """

    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.last_payload: dict[str, Any] | None = None

    def apply(self, event: Mapping[str, Any]) -> None:
        key = (str(event.get("event_type", "")), str(event.get("sub_type", "")))
        self.counts[key] = self.counts.get(key, 0) + 1
        self.last_payload = dict(event.get("payload", {}))

    def snapshot(self) -> dict[str, Any]:
        # Sort the keys so the snapshot is canonical regardless of
        # dict-insertion order on different Python builds.
        return {
            "counts": dict(sorted(self.counts.items())),
            "last_payload": dict(self.last_payload or {}),
        }

    def restore(self, view: dict[str, Any]) -> None:
        self.counts = {
            tuple(k): int(v) for k, v in view.get("counts", {}).items()
        }
        self.last_payload = dict(view.get("last_payload") or {})


class RunningSumProjector:
    """Sums ``payload['qty']`` across every event."""

    def __init__(self) -> None:
        self.total = 0

    def apply(self, event: Mapping[str, Any]) -> None:
        self.total += int(event.get("payload", {}).get("qty", 0))

    def snapshot(self) -> dict[str, Any]:
        return {"total": self.total}

    def restore(self, view: dict[str, Any]) -> None:
        self.total = int(view.get("total", 0))


def _feed() -> list[dict[str, Any]]:
    """A stable, deterministic feed. Mixes event types and sub-types,
    walks wall_ns monotonically, and includes a wall_ns=0 event to
    exercise the regression guard on snapshot cursor preservation."""
    return [
        {"sequence": 1, "wall_ns": 0,         "event_type": "MARKET",  "sub_type": "TICK",   "payload": {"qty": 5}},
        {"sequence": 2, "wall_ns": 1_000,     "event_type": "MARKET",  "sub_type": "TICK",   "payload": {"qty": 7}},
        {"sequence": 3, "wall_ns": 2_500,     "event_type": "SYSTEM",  "sub_type": "HAZARD", "payload": {"qty": 0}},
        {"sequence": 4, "wall_ns": 3_000,     "event_type": "MARKET",  "sub_type": "QUOTE",  "payload": {"qty": 11}},
        {"sequence": 5, "wall_ns": 4_200,     "event_type": "MARKET",  "sub_type": "TICK",   "payload": {"qty": 2}},
        {"sequence": 6, "wall_ns": 5_000,     "event_type": "SYSTEM",  "sub_type": "HAZARD", "payload": {"qty": 0}},
        {"sequence": 7, "wall_ns": 6_100,     "event_type": "MARKET",  "sub_type": "QUOTE",  "payload": {"qty": 13}},
        {"sequence": 8, "wall_ns": 7_000,     "event_type": "MARKET",  "sub_type": "TICK",   "payload": {"qty": 3}},
        {"sequence": 9, "wall_ns": 8_400,     "event_type": "MARKET",  "sub_type": "TICK",   "payload": {"qty": 17}},
        {"sequence": 10,"wall_ns": 9_000,     "event_type": "SYSTEM",  "sub_type": "HAZARD", "payload": {"qty": 0}},
    ]


def _factories() -> dict[str, Any]:
    return {
        "book": OrderBookProjector,
        "sum":  RunningSumProjector,
    }


# ─────── (a) two fresh projector sets land on identical views ──────────


def test_two_independent_replays_produce_identical_snapshots() -> None:
    """Baseline determinism: the same event list applied to two fresh
    projector sets must produce the same snapshot() output dict."""
    events_a = _feed()
    events_b = copy.deepcopy(events_a)  # defensive; events must not mutate

    proj_a = {"book": OrderBookProjector(), "sum": RunningSumProjector()}
    proj_b = {"book": OrderBookProjector(), "sum": RunningSumProjector()}

    for e in events_a:
        for p in proj_a.values():
            p.apply(e)
    for e in events_b:
        for p in proj_b.values():
            p.apply(e)

    assert {k: v.snapshot() for k, v in proj_a.items()} == {k: v.snapshot() for k, v in proj_b.items()}
    # Events must not be mutated by projectors.
    assert events_a == events_b


# ─────── (b) snapshot-accelerated replay == genesis replay ─────────────


def test_snapshot_accelerated_replay_equals_genesis_replay() -> None:
    """If ``SnapshotEngine`` captured mid-stream and ``StateReconstructor``
    fast-forwarded past the snapshot, the final read-model must exactly
    equal a pure genesis replay."""
    events = _feed()

    # Live ingestion: feed the engine, triggering captures every 3 events.
    engine = SnapshotEngine(
        projectors={"book": OrderBookProjector(), "sum": RunningSumProjector()},
        policy=SnapshotPolicy(
            events_per_snapshot=3,
            nanoseconds_per_snapshot=10**18,  # effectively disable the wall trigger
        ),
    )
    for e in events:
        engine.on_event(e)

    # Now reconstruct through the SAME snapshot ring (fast-forward path).
    accelerated = StateReconstructor(
        projector_factories=_factories(),
        event_feed=lambda: iter(events),
        snapshot_engine=engine,
    ).rebuild_latest()

    # And reconstruct with no snapshot engine at all (pure genesis replay).
    genesis = StateReconstructor(
        projector_factories=_factories(),
        event_feed=lambda: iter(events),
        snapshot_engine=None,
    ).rebuild_latest()

    assert accelerated.projectors == genesis.projectors
    assert accelerated.sequence == genesis.sequence
    assert accelerated.event_count == genesis.event_count
    # The fast-forward path must actually have used a snapshot — otherwise
    # we aren't covering the interesting branch.
    assert accelerated.resumed_from_snapshot is True
    assert genesis.resumed_from_snapshot is False


# ─────── (c) rebuild_at matches hand-cut feed ──────────────────────────


@pytest.mark.parametrize("target_seq", [1, 3, 5, 7, 10])
def test_rebuild_at_matches_hand_truncated_feed(target_seq: int) -> None:
    """``rebuild_at(N)`` == replay(events[:N])."""
    events = _feed()
    truncated = [e for e in events if e["sequence"] <= target_seq]

    proj = {"book": OrderBookProjector(), "sum": RunningSumProjector()}
    for e in truncated:
        for p in proj.values():
            p.apply(e)
    expected_views = {k: v.snapshot() for k, v in proj.items()}

    got = StateReconstructor(
        projector_factories=_factories(),
        event_feed=lambda: iter(events),
        snapshot_engine=None,
    ).rebuild_at(target_seq)

    assert got.projectors == expected_views
    assert got.sequence == (target_seq if truncated else 0)
    assert got.event_count == len(truncated)


# ─────── (d) FastRiskCache version_id is a pure function ───────────────


def test_version_id_is_a_pure_function_of_inputs() -> None:
    """The decision-record contract (T0-1) requires every decision to
    carry a ``risk_version_used`` equal to ``cache.read().version_id``.
    The string MUST be a pure function of ``(version, updated_at_ns)``
    so a replay with identical updates produces identical stamps."""
    for v in (1, 2, 7, 42, 10_000):
        for t in (0, 1, 100, 999_999, 2**40):
            a = _compute_version_id(v, t)
            b = _compute_version_id(v, t)
            assert a == b

    # Any delta in inputs must produce a different id.
    assert _compute_version_id(5, 100) != _compute_version_id(5, 101)
    assert _compute_version_id(5, 100) != _compute_version_id(6, 100)


def test_two_caches_fed_identically_stamp_identical_version_ids() -> None:
    """End-to-end: two fresh caches driven through the same update
    sequence must converge to the same ``version_id`` at every step."""
    clock_a = {"v": 100}
    clock_b = {"v": 100}
    cache_a = FastRiskCache(clock_wall_ns=lambda: clock_a["v"])
    cache_b = FastRiskCache(clock_wall_ns=lambda: clock_b["v"])

    assert cache_a.read().version_id == cache_b.read().version_id

    updates = [
        (200, {"max_order_size_usd": 100.0}),
        (350, {"trading_allowed": False}),
        (500, {"trading_allowed": True, "max_order_size_usd": 250.0}),
        (650, {"safe_mode": True}),
    ]
    for ts, kwargs in updates:
        clock_a["v"] = ts
        clock_b["v"] = ts
        cache_a.update(**kwargs)
        cache_b.update(**kwargs)
        assert cache_a.read().version_id == cache_b.read().version_id, (
            f"divergence after update at ts={ts}: {cache_a.read().version_id} != {cache_b.read().version_id}"
        )


# ─────── (e) wall_ns=0 preserved across replay ─────────────────────────


def test_wall_ns_zero_is_preserved_in_snapshot_and_replay() -> None:
    """Regression for PR #10 determinism fix: an event stamping
    ``wall_ns=0`` must not be silently replaced with a clock read,
    because that would make wall-time-driven replay produce a
    different cursor than sequence-driven replay."""
    events = _feed()
    assert events[0]["wall_ns"] == 0

    engine = SnapshotEngine(
        projectors={"book": OrderBookProjector(), "sum": RunningSumProjector()},
        policy=SnapshotPolicy(events_per_snapshot=1, nanoseconds_per_snapshot=10**18),
        clock_wall_ns=lambda: 99_999_999,  # would leak in if the fix regressed
    )
    first_snap = engine.on_event(events[0])
    assert first_snap is not None
    assert first_snap.cursor.wall_ns == 0

    # Finish feeding.
    for e in events[1:]:
        engine.on_event(e)

    # The engine retains a snapshot at wall_ns=0 in its ring — the
    # cursor-at-or-before(0) query must locate it exactly.
    at_zero = engine.latest_at_or_before_wall_ns(0)
    assert at_zero is first_snap

    # Replaying via the reconstructor by wall-time must stop at the
    # same event as replaying by sequence.
    seq_rebuild = StateReconstructor(
        projector_factories=_factories(),
        event_feed=lambda: iter(events),
        snapshot_engine=engine,
    ).rebuild_at(1)

    wall_rebuild = StateReconstructor(
        projector_factories=_factories(),
        event_feed=lambda: iter(events),
        snapshot_engine=engine,
    ).rebuild(at_timestamp_ns=0)

    assert seq_rebuild.sequence == wall_rebuild.sequence == 1
    assert seq_rebuild.projectors == wall_rebuild.projectors
