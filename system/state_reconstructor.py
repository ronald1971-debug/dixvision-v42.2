"""
system/state_reconstructor.py

DIX VISION v42.2 — Tier-0 Step 0: StateReconstructor.

Rebuilds the full system read-state at any point in the ledger's
history by taking the nearest :class:`system.snapshots.Snapshot`
(if any) and replaying the delta of events that landed after it.

Contract (see docs/ARCHITECTURE_V42_2_TIER0.md §1):

    rebuild_latest()   -> ReconstructedState
    rebuild_at(seq)    -> ReconstructedState at sequence ``seq`` (inclusive)
    rebuild(ts_wall_ns)-> ReconstructedState at the last event whose
                          ``wall_ns`` ≤ ``ts_wall_ns``

Hard rules:

    1. The ledger is authoritative. If no snapshot is present the
       reconstructor replays every event from genesis.
    2. Reconstruction is deterministic — same ledger, same projector
       set, same snapshot ring → bit-identical output.
    3. The reconstructor *never* writes to the ledger, *never* mutates
       the live projectors, and *never* mutates the snapshot ring.
       It constructs fresh projector instances from a caller-supplied
       factory for each rebuild, so concurrent boot-time rebuilds
       cannot race live traffic.
    4. A rebuild whose requested cursor is past the end of the feed
       raises :class:`OutOfRangeError` — silent truncation would
       defeat the audit guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from system.snapshots import Projector, Snapshot, SnapshotEngine


class OutOfRangeError(ValueError):
    """Raised when the requested cursor lies past the ledger end."""


@dataclass(frozen=True)
class ReconstructedState:
    """Immutable aggregate of every projector's read-model at a cursor."""

    sequence: int
    wall_ns: int
    event_count: int
    projectors: dict[str, Any] = field(default_factory=dict)
    resumed_from_snapshot: bool = False


ProjectorFactories = Mapping[str, Callable[[], Projector]]
EventFeed = Callable[[], Iterable[Mapping[str, Any]]]


class StateReconstructor:
    """Replay engine for the event-sourced system state.

    Parameters
    ----------
    projector_factories:
        Map of projector-name → zero-arg factory that returns a FRESH
        projector instance. The reconstructor instantiates projectors
        per rebuild so rebuilds never share state with live traffic.
    event_feed:
        Zero-arg callable returning an iterable of ledger events in
        sequence order. The callable is invoked once per rebuild —
        pass a factory that opens a fresh cursor each time, not a
        one-shot generator.
    snapshot_engine:
        Optional :class:`SnapshotEngine` whose ring is consulted to
        fast-forward past events already captured. If ``None``, every
        rebuild replays from genesis.
    """

    def __init__(
        self,
        projector_factories: ProjectorFactories,
        event_feed: EventFeed,
        snapshot_engine: SnapshotEngine | None = None,
    ) -> None:
        if not projector_factories:
            raise ValueError("at least one projector factory is required")
        self._factories = dict(projector_factories)
        self._feed = event_feed
        self._snapshots = snapshot_engine

    # ─────── public API ─────────────────────────────────────────────

    def rebuild_latest(self) -> ReconstructedState:
        """Rebuild the state as of the most recent ledger event."""
        return self._rebuild(target_sequence=None, target_wall_ns=None)

    def rebuild_at(self, sequence: int) -> ReconstructedState:
        """Rebuild the state as of (and including) ``sequence``."""
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        return self._rebuild(target_sequence=sequence, target_wall_ns=None)

    def rebuild(self, at_timestamp_ns: int) -> ReconstructedState:
        """Rebuild the state as of the last event with wall_ns ≤ ``at_timestamp_ns``.

        Matches the canonical contract wording in the Tier-0 directive.
        """
        if at_timestamp_ns < 0:
            raise ValueError("at_timestamp_ns must be non-negative")
        return self._rebuild(target_sequence=None, target_wall_ns=at_timestamp_ns)

    # ─────── internals ──────────────────────────────────────────────

    def _rebuild(
        self,
        *,
        target_sequence: int | None,
        target_wall_ns: int | None,
    ) -> ReconstructedState:
        snap = self._pick_snapshot(target_sequence, target_wall_ns)
        projectors, fully_hydrated = self._hydrate_projectors(snap)
        # We can only fast-forward past the snapshot cursor if every
        # projector was restorable — otherwise the projectors that fell
        # back to genesis would miss the pre-snapshot events entirely.
        use_fast_forward = snap is not None and fully_hydrated
        start_seq = snap.cursor.sequence if use_fast_forward else -1
        base_count = snap.event_count if use_fast_forward else 0

        last_seq = start_seq
        last_wall = snap.cursor.wall_ns if use_fast_forward else 0
        applied = 0

        for event in self._feed():
            seq = int(event.get("sequence", last_seq + 1))
            if seq <= start_seq:
                # already captured by snapshot
                continue
            wall = int(event.get("wall_ns", last_wall))
            if target_sequence is not None and seq > target_sequence:
                break
            if target_wall_ns is not None and wall > target_wall_ns:
                break
            for p in projectors.values():
                p.apply(event)
            last_seq = seq
            last_wall = wall
            applied += 1

        if target_sequence is not None and last_seq < target_sequence:
            raise OutOfRangeError(
                f"ledger ends at sequence {last_seq}; cannot rebuild at {target_sequence}"
            )

        return ReconstructedState(
            sequence=last_seq if last_seq >= 0 else 0,
            wall_ns=last_wall,
            event_count=base_count + applied,
            projectors={name: p.snapshot() for name, p in projectors.items()},
            resumed_from_snapshot=use_fast_forward,
        )

    def _pick_snapshot(
        self,
        target_sequence: int | None,
        target_wall_ns: int | None,
    ) -> Snapshot | None:
        """Pick the most advanced snapshot whose cursor is not past the target.

        - ``target_sequence`` constrains ``cursor.sequence``.
        - ``target_wall_ns`` constrains ``cursor.wall_ns``.

        If both are ``None`` the caller wants the latest available snapshot.
        Returning a snapshot past *either* target would poison the replay:
        events already captured beyond the requested cursor cannot be undone.
        """
        if self._snapshots is None:
            return None
        if target_sequence is not None:
            return self._snapshots.latest_at_or_before(target_sequence)
        if target_wall_ns is not None:
            return self._snapshots.latest_at_or_before_wall_ns(target_wall_ns)
        return self._snapshots.latest()

    def _hydrate_projectors(
        self, snap: Snapshot | None
    ) -> tuple[dict[str, Projector], bool]:
        """Instantiate a fresh projector set and try to restore from ``snap``.

        Returns
        -------
        (projectors, fully_hydrated)
            ``fully_hydrated`` is True only if ``snap`` was provided AND
            every projector supplied a working ``restore`` that accepted
            its matching view. When it is False the caller MUST replay
            from genesis — otherwise any projector that fell back to
            defaults would silently miss the pre-snapshot events.
        """
        projectors = {name: factory() for name, factory in self._factories.items()}
        if snap is None:
            return projectors, False

        fully_hydrated = True
        restored_names: set[str] = set()
        for name, proj in projectors.items():
            view = snap.projectors.get(name)
            if view is None:
                fully_hydrated = False
                continue
            restore = getattr(proj, "restore", None)
            if not callable(restore):
                fully_hydrated = False
                continue
            try:
                restore(view)
            except Exception:
                # Reset this one back to genesis and force a full replay.
                projectors[name] = self._factories[name]()
                fully_hydrated = False
            else:
                restored_names.add(name)

        if not fully_hydrated and restored_names:
            # Partial hydration poisons the replay: restored projectors would
            # carry snapshot state AND then re-apply every pre-snapshot event
            # during the forced genesis replay, double-counting everything.
            # Drop every restored projector back to genesis so the replay is
            # consistent across the whole set.
            for name in restored_names:
                projectors[name] = self._factories[name]()

        return projectors, fully_hydrated


__all__ = [
    "EventFeed",
    "OutOfRangeError",
    "ProjectorFactories",
    "ReconstructedState",
    "StateReconstructor",
]
