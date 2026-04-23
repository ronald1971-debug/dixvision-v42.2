"""
state/ledger/indexer.py

DIX VISION v42.2 — Tier-0 Step 5: Ledger secondary indexer.

Maintains in-memory secondary indexes over hot-tier events so the
cockpit / mind / governance read paths can answer "give me the most
recent N SYSTEM.HAZARD events" without scanning the full ring. The
indexer is fed alongside the hot store from the stream router.

Hard rules:

    1. Indexes are *derived*; losing them never affects ledger
       integrity. They can always be rebuilt by replaying the hot
       ring.
    2. Indexes are bounded per key (default 1 024 entries) — no
       unbounded growth.
    3. Writes and reads are serialized under a single lock.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Mapping


DEFAULT_PER_KEY_CAPACITY = 1_024


class LedgerIndexer:
    """In-memory secondary indexer over hot-tier events."""

    def __init__(self, per_key_capacity: int = DEFAULT_PER_KEY_CAPACITY) -> None:
        if per_key_capacity <= 0:
            raise ValueError("per_key_capacity must be > 0")
        self._cap = int(per_key_capacity)
        self._lock = threading.RLock()
        self._by_type: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=self._cap)
        )
        self._by_source: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=self._cap)
        )
        self._by_composite: dict[tuple[str, str], deque[int]] = defaultdict(
            lambda: deque(maxlen=self._cap)
        )

    def index(self, event: Mapping) -> None:
        """Record the event's sequence against each index key."""
        seq = int(event.get("sequence", 0))
        et = str(event.get("event_type", ""))
        src = str(event.get("source", ""))
        st = str(event.get("sub_type", ""))
        with self._lock:
            self._by_type[et].append(seq)
            self._by_source[src].append(seq)
            self._by_composite[(et, st)].append(seq)

    def recent_sequences_by_type(self, event_type: str, *, limit: int = 100) -> list[int]:
        return self._recent(self._by_type.get(event_type), limit)

    def recent_sequences_by_source(self, source: str, *, limit: int = 100) -> list[int]:
        return self._recent(self._by_source.get(source), limit)

    def recent_sequences_for(
        self, event_type: str, sub_type: str, *, limit: int = 100
    ) -> list[int]:
        return self._recent(self._by_composite.get((event_type, sub_type)), limit)

    def clear(self) -> None:
        with self._lock:
            self._by_type.clear()
            self._by_source.clear()
            self._by_composite.clear()

    # ─────── internals ──────────────────────────────────────────────

    def _recent(self, dq: deque[int] | None, limit: int) -> list[int]:
        if limit <= 0 or not dq:
            return []
        with self._lock:
            return list(reversed(list(dq)))[:limit]


_indexer: LedgerIndexer | None = None
_lock = threading.Lock()


def get_indexer() -> LedgerIndexer:
    global _indexer
    if _indexer is None:
        with _lock:
            if _indexer is None:
                _indexer = LedgerIndexer()
    return _indexer


__all__ = [
    "DEFAULT_PER_KEY_CAPACITY",
    "LedgerIndexer",
    "get_indexer",
]
