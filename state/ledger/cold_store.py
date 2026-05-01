"""
state/ledger/cold_store.py

DIX VISION v42.2 — Tier-0 Step 5: Cold-tier (archival) read facade.

Thin read-only wrapper around the authoritative :class:`EventStore`
SQLite ledger. The cold store exists so callers have a single,
stable interface for archival queries regardless of what the hot
store has aged out. Write paths continue to go through
:func:`state.ledger.event_store.append_event` — this module never
appends.

Hard rules:

    1. Read-only. Any method that would mutate the ledger belongs in
       ``event_store.py``, not here.
    2. No caching — the point is to always go to authoritative
       storage. Callers that want caching use the hot store.
    3. Queries are bounded by ``limit`` to prevent a pathological
       caller from OOM-ing the process on a multi-year archive.
"""
from __future__ import annotations

import threading
from typing import Any, Protocol

from state.ledger.event_store import EventStore, get_event_store


DEFAULT_QUERY_LIMIT = 1_000
MAX_QUERY_LIMIT = 1_000_000


class EventStoreReader(Protocol):
    """Minimal read-only contract the cold store uses.

    The concrete :class:`EventStore` satisfies this; tests substitute
    a fake reader that records calls without touching SQLite.
    """

    def query(
        self,
        event_type: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict]: ...


class ColdStore:
    """Archival read facade over the authoritative event store."""

    def __init__(self, reader: EventStoreReader | None = None) -> None:
        self._reader = reader or get_event_store()

    def query(
        self,
        *,
        event_type: str | None = None,
        source: str | None = None,
        limit: int = DEFAULT_QUERY_LIMIT,
    ) -> list[dict[str, Any]]:
        """Fetch up to ``limit`` archival events, newest first.

        Parameters are forwarded to :meth:`EventStore.query`. ``limit``
        is clamped to :data:`MAX_QUERY_LIMIT` to prevent runaway reads.
        """
        if limit <= 0:
            raise ValueError("limit must be > 0")
        if limit > MAX_QUERY_LIMIT:
            limit = MAX_QUERY_LIMIT
        return self._reader.query(
            event_type=event_type,
            source=source,
            limit=limit,
        )


_store: ColdStore | None = None
_lock = threading.Lock()


def get_cold_store() -> ColdStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                _store = ColdStore()
    return _store


__all__ = [
    "DEFAULT_QUERY_LIMIT",
    "MAX_QUERY_LIMIT",
    "ColdStore",
    "EventStoreReader",
    "get_cold_store",
]
