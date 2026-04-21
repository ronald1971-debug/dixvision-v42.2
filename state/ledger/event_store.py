"""
state/ledger/event_store.py
DIX VISION v42.2 — Append-Only Event Store (Hash-Chained)

Single source of truth for all system events.
Events: MARKET, SYSTEM, GOVERNANCE, HAZARD.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from system.time_source import now


@dataclass
class LedgerEvent:
    """Immutable ledger event."""
    event_id: str
    event_type: str       # MARKET | SYSTEM | GOVERNANCE | HAZARD
    sub_type: str
    source: str           # INDIRA | DYON | GOVERNANCE | SYSTEM
    payload: dict[str, Any]
    timestamp_utc: str
    sequence: int
    prev_hash: str
    event_hash: str = ""

    def compute_hash(self) -> str:
        data = json.dumps({
            "event_id": self.event_id, "event_type": self.event_type,
            "sub_type": self.sub_type, "source": self.source,
            "payload": self.payload, "timestamp_utc": self.timestamp_utc,
            "sequence": self.sequence, "prev_hash": self.prev_hash,
        }, sort_keys=True, default=str)
        return hashlib.sha256(data.encode()).hexdigest()

class EventStore:
    """
    Append-only SQLite-backed event store with SHA-256 hash chaining.
    Thread-safe. All three execution planes write here.
    """
    def __init__(self, db_path: str = "data/sqlite/ledger.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._prev_hash = "GENESIS"
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        # Event-store tuning (manifest §7, §8). Durable + WAL + fast.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")
        self._conn.execute("PRAGMA mmap_size=268435456")       # 256MB
        self._conn.execute("PRAGMA cache_size=-8000")          # 8MB page cache
        self._conn.execute("PRAGMA wal_autocheckpoint=1000")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sub_type TEXT NOT NULL,
                source TEXT NOT NULL,
                payload TEXT NOT NULL,
                timestamp_utc TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                prev_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL,
                UNIQUE(event_hash)
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON events(source)")
        self._conn.commit()
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        cur = self._conn.execute(
            "SELECT event_hash FROM events ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            self._prev_hash = row[0]

    def append(self, event_type: str, sub_type: str, source: str,
               payload: dict[str, Any]) -> LedgerEvent:
        import uuid
        with self._lock:
            ts = now()
            event = LedgerEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                sub_type=sub_type,
                source=source,
                payload=payload,
                timestamp_utc=ts.utc_time.isoformat(),
                sequence=ts.sequence,
                prev_hash=self._prev_hash,
            )
            event.event_hash = event.compute_hash()
            self._conn.execute("""
                INSERT INTO events
                (event_id, event_type, sub_type, source, payload,
                 timestamp_utc, sequence, prev_hash, event_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id, event.event_type, event.sub_type,
                event.source, json.dumps(event.payload, default=str),
                event.timestamp_utc, event.sequence,
                event.prev_hash, event.event_hash,
            ))
            self._conn.commit()
            self._prev_hash = event.event_hash

        return event

    def query(self, event_type: str = None, source: str = None,
              limit: int = 100) -> list[dict]:
        parts, params = [], []
        if event_type:
            parts.append("event_type = ?")
            params.append(event_type)
        if source:
            parts.append("source = ?")
            params.append(source)
        where = f"WHERE {' AND '.join(parts)}" if parts else ""
        params.append(limit)
        cur = self._conn.execute(
            f"SELECT * FROM events {where} ORDER BY id DESC LIMIT ?", params
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=False)) for row in cur.fetchall()]

    def verify_chain(self) -> bool:
        """Replay and verify hash chain integrity."""
        cur = self._conn.execute("SELECT * FROM events ORDER BY id ASC")
        cols = [d[0] for d in cur.description]
        prev = "GENESIS"
        for row in cur:
            ev = dict(zip(cols, row, strict=False))
            if ev["prev_hash"] != prev:
                return False
            recomputed = hashlib.sha256(json.dumps({
                "event_id": ev["event_id"], "event_type": ev["event_type"],
                "sub_type": ev["sub_type"], "source": ev["source"],
                "payload": json.loads(ev["payload"]),
                "timestamp_utc": ev["timestamp_utc"],
                "sequence": ev["sequence"], "prev_hash": ev["prev_hash"],
            }, sort_keys=True, default=str).encode()).hexdigest()
            if recomputed != ev["event_hash"]:
                return False
            prev = ev["event_hash"]
        return True

_store: EventStore | None = None
_lock = threading.Lock()

def get_event_store() -> EventStore:
    global _store
    if _store is None:
        with _lock:
            if _store is None:
                from system.config import get
                _store = EventStore(get("ledger.db_path", "data/sqlite/ledger.db"))
    return _store

def append_event(event_type: str, sub_type: str, source: str,
                 payload: dict[str, Any]) -> LedgerEvent:
    return get_event_store().append(event_type, sub_type, source, payload)
