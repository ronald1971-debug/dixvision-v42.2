"""
state/ledger/hot_store.py

DIX VISION v42.2 — Tier-0 Step 5: Hot-tier event cache.

Bounded in-memory ring of the most recent ledger events. The hot store
is a *read accelerator* layered on top of the authoritative
:mod:`state.ledger.event_store` SQLite ledger — it is never the source
of truth. Callers that need sub-millisecond access to the last N
events (cockpit dashboards, Indira fast path, the projector bus) read
here first and fall through to the cold store when a miss occurs.

Hard rules (see docs/ARCHITECTURE_V42_2_TIER0.md §5):

    1. Hot store NEVER drops an event on the write path; it only ages
       entries out of its ring. The authoritative SQLite ledger keeps
       every event forever.
    2. Writes are serialized under a single lock; reads are cheap
       (deque copy + filter).
    3. Content is deterministic for a given ordered feed — no
       reordering, no deduplication.
    4. The ring is bounded by count; callers requesting events older
       than the tail must fall through to the cold store.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping


DEFAULT_HOT_CAPACITY = 10_000


@dataclass(frozen=True)
class HotEvent:
    """Immutable copy of a ledger event held in the hot ring."""

    sequence: int
    event_type: str
    sub_type: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_hash: str = ""
    event_id: str = ""


class HotStore:
    """Bounded ring of most-recent ledger events."""

    def __init__(self, capacity: int = DEFAULT_HOT_CAPACITY) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._capacity = int(capacity)
        self._lock = threading.RLock()
        self._ring: deque[HotEvent] = deque(maxlen=self._capacity)
        self._last_sequence: int = -1

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._ring)

    # ─────── ingestion ──────────────────────────────────────────────

    def add(self, event: Mapping[str, Any]) -> HotEvent:
        """Append one event. Returns the stored :class:`HotEvent`.

        Events arriving out of sequence order are still accepted — the
        authoritative sequence is owned by the event store, not by this
        ring — but ``last_sequence()`` reports the highest sequence
        observed.
        """
        hot = HotEvent(
            sequence=int(event.get("sequence", 0)),
            event_type=str(event.get("event_type", "")),
            sub_type=str(event.get("sub_type", "")),
            source=str(event.get("source", "")),
            payload=dict(event.get("payload", {}) or {}),
            event_hash=str(event.get("event_hash", "")),
            event_id=str(event.get("event_id", "")),
        )
        with self._lock:
            self._ring.append(hot)
            if hot.sequence > self._last_sequence:
                self._last_sequence = hot.sequence
        return hot

    def clear(self) -> None:
        with self._lock:
            self._ring.clear()
            self._last_sequence = -1

    # ─────── queries ────────────────────────────────────────────────

    def last_sequence(self) -> int:
        with self._lock:
            return self._last_sequence

    def earliest_sequence(self) -> int:
        """Smallest sequence currently in the ring, or -1 if empty."""
        with self._lock:
            if not self._ring:
                return -1
            return min(e.sequence for e in self._ring)

    def contains_sequence(self, sequence: int) -> bool:
        with self._lock:
            return any(e.sequence == sequence for e in self._ring)

    def recent(
        self,
        *,
        limit: int = 100,
        event_type: str | None = None,
        source: str | None = None,
        sub_type: str | None = None,
    ) -> list[HotEvent]:
        """Return up to ``limit`` most-recent events, newest first."""
        if limit <= 0:
            return []
        with self._lock:
            out: list[HotEvent] = []
            for e in reversed(self._ring):
                if event_type is not None and e.event_type != event_type:
                    continue
                if source is not None and e.source != source:
                    continue
                if sub_type is not None and e.sub_type != sub_type:
                    continue
                out.append(e)
                if len(out) >= limit:
                    break
            return out

    def events_after(self, sequence: int) -> list[HotEvent]:
        """Return all events with sequence > ``sequence`` in original order.

        Raises :class:`LookupError` if the request cannot be satisfied
        because the ring has already aged past ``sequence``. Callers
        should fall through to the cold store in that case.
        """
        with self._lock:
            if not self._ring:
                return []
            earliest = min(e.sequence for e in self._ring)
            if sequence + 1 < earliest:
                raise LookupError(
                    f"hot store earliest sequence is {earliest}; "
                    f"cannot serve events after {sequence}"
                )
            return [e for e in self._ring if e.sequence > sequence]


_store: HotStore | None = None
_lock = threading.Lock()


def get_hot_store() -> HotStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = HotStore()
    return _store


__all__ = [
    "DEFAULT_HOT_CAPACITY",
    "HotEvent",
    "HotStore",
    "get_hot_store",
]
