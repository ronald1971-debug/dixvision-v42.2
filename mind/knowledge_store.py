"""
mind/knowledge_store.py

DIX VISION v42.2 — Tier-0 Step 11: Bounded knowledge-store primitive.

A process-local, in-memory, thread-safe key-value store for ephemeral
knowledge (ranked signals, recent fundamentals, curated research
fragments). Unlike ``mind.knowledge.trader_knowledge`` (SQLite-backed,
persistent), this store is:

    - BOUNDED: hard limits on entry count AND estimated byte size.
    - LRU-EVICTING: the least-recently-USED entry is dropped first
      when any bound is exceeded.
    - COMPACTABLE: manual ``compact()`` prunes low-confidence entries
      without touching recently-used ones.
    - SNAPSHOT-READY: ``snapshot()`` / ``restore()`` serialize the
      full state for T0-0 state-reconstruction replay.

Consumers:
    - mind.engine (retrieval context)
    - mind.fast_execute   (read-only, never mutates)
    - governance policies (read-only, advisory)

Hard rules:
    1. No unbounded growth. Hitting ``max_entries`` OR ``max_bytes``
       triggers LRU eviction synchronously.
    2. Eviction is DETERMINISTIC given the same ordered put() sequence.
    3. Writes are serialized under a single RLock; reads are cheap
       (dict lookup + touch).
    4. Entries carry a monotonic ``inserted_at_ns`` from
       ``system.time_source.wall_ns()`` for audit and for compaction
       ranking.

See docs/ARCHITECTURE_V42_2_TIER0.md §13.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any


def _wall_ns() -> int:
    """Wall-clock nanoseconds (Unix epoch).

    Uses ``time.time_ns()`` directly. T0-4 exposes this as
    ``system.time_source.wall_ns``; once that PR lands this module
    should be switched to the imported alias for consistency.
    """
    return time.time_ns()


DEFAULT_MAX_ENTRIES = 50_000
DEFAULT_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB
DEFAULT_COMPACT_MIN_CONFIDENCE = 0.10


@dataclass(frozen=True)
class KnowledgeEntry:
    """One immutable knowledge record."""

    key: str
    value: Any
    confidence: float = 0.5
    inserted_at_ns: int = 0
    size_bytes: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)


def _estimate_bytes(value: Any) -> int:
    """Conservative byte-size estimate, enough for eviction math.

    We intentionally do not walk the whole object graph — this is a
    cap, not a forensic accountant. Tune if real-world working sets
    need finer-grained measurement.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, str):
        return len(value.encode("utf-8", errors="ignore"))
    if isinstance(value, (list, tuple, set)):
        return sum(_estimate_bytes(v) for v in value) + 32
    if isinstance(value, dict):
        return sum(
            _estimate_bytes(k) + _estimate_bytes(v) for k, v in value.items()
        ) + 32
    if isinstance(value, (int, float, bool)) or value is None:
        return 16
    try:
        return len(str(value).encode("utf-8", errors="ignore"))
    except Exception:
        return 64


class KnowledgeStore:
    """Bounded, LRU-evicting, thread-safe knowledge store."""

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        self._max_entries = int(max_entries)
        self._max_bytes = int(max_bytes)
        self._lock = RLock()
        self._data: OrderedDict[str, KnowledgeEntry] = OrderedDict()
        self._bytes = 0
        self._evictions = 0

    # ─────── properties ──────────────────────────────────────────

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes

    def eviction_count(self) -> int:
        with self._lock:
            return self._evictions

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    # ─────── core API ────────────────────────────────────────────

    def put(
        self,
        key: str,
        value: Any,
        *,
        confidence: float = 0.5,
        tags: tuple[str, ...] = (),
    ) -> None:
        """Insert or replace. Enforces both entry and byte caps via LRU."""
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty str")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        size = _estimate_bytes(value)
        entry = KnowledgeEntry(
            key=key,
            value=value,
            confidence=float(confidence),
            inserted_at_ns=_wall_ns(),
            size_bytes=size,
            tags=tuple(tags),
        )
        with self._lock:
            if key in self._data:
                old = self._data.pop(key)
                self._bytes -= old.size_bytes
            self._data[key] = entry
            self._bytes += size
            self._enforce_caps_locked()

    def get(self, key: str) -> Any | None:
        """Return value and mark as most-recently-used, else None."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            self._data.move_to_end(key)
            return entry.value

    def peek(self, key: str) -> KnowledgeEntry | None:
        """Read without touching LRU order."""
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            entry = self._data.pop(key, None)
            if entry is None:
                return False
            self._bytes -= entry.size_bytes
            return True

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._bytes = 0

    # ─────── compaction ──────────────────────────────────────────

    def compact(
        self,
        *,
        min_confidence: float = DEFAULT_COMPACT_MIN_CONFIDENCE,
    ) -> int:
        """Drop entries below ``min_confidence``. Returns count removed.

        Unlike eviction this is explicit and policy-driven; governance /
        scheduler decide when it runs.
        """
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        removed = 0
        with self._lock:
            victims = [
                k for k, e in self._data.items() if e.confidence < min_confidence
            ]
            for k in victims:
                entry = self._data.pop(k)
                self._bytes -= entry.size_bytes
                removed += 1
        return removed

    # ─────── snapshot / restore ─────────────────────────────────

    def snapshot(self) -> list[dict[str, Any]]:
        """Serializable list-of-dicts for T0-0 reconstruction."""
        with self._lock:
            return [
                {
                    "key": e.key,
                    "value": e.value,
                    "confidence": e.confidence,
                    "inserted_at_ns": e.inserted_at_ns,
                    "size_bytes": e.size_bytes,
                    "tags": list(e.tags),
                }
                for e in self._data.values()
            ]

    def restore(self, rows: list[dict[str, Any]]) -> None:
        """Replace state from a prior snapshot (in insertion order)."""
        with self._lock:
            self._data.clear()
            self._bytes = 0
            for row in rows:
                entry = KnowledgeEntry(
                    key=str(row["key"]),
                    value=row["value"],
                    confidence=float(row.get("confidence", 0.5)),
                    inserted_at_ns=int(row.get("inserted_at_ns", 0)),
                    size_bytes=int(
                        row.get("size_bytes", _estimate_bytes(row["value"]))
                    ),
                    tags=tuple(row.get("tags", ())),
                )
                self._data[entry.key] = entry
                self._bytes += entry.size_bytes
            self._enforce_caps_locked()

    # ─────── internals ───────────────────────────────────────────

    def _enforce_caps_locked(self) -> None:
        """Evict LRU until both caps are satisfied. Caller holds the lock."""
        while (
            len(self._data) > self._max_entries or self._bytes > self._max_bytes
        ):
            if not self._data:
                break
            _, evicted = self._data.popitem(last=False)  # LRU = oldest end
            self._bytes -= evicted.size_bytes
            self._evictions += 1


__all__ = [
    "DEFAULT_COMPACT_MIN_CONFIDENCE",
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_ENTRIES",
    "KnowledgeEntry",
    "KnowledgeStore",
]
