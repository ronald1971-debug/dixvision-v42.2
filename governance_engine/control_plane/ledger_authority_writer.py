"""GOV-CP-05 — Ledger Authority Writer.

The **only** module permitted to append to the authority ledger
(``manifest.md`` §0.5 GOV-CP-05). Every approved governance decision
flows through ``append`` and is recorded as a hash-chained
:class:`LedgerEntry`.

Storage: by default the chain is held in memory (``list[LedgerEntry]``)
which keeps unit tests fast and isolated. Sprint-1 (architectural-review
remediation, Class-B "Trust the Ledger") adds **optional** SQLite WAL
persistence: when the writer is constructed with ``db_path=...`` (or
the harness sets ``DIXVISION_LEDGER_PATH``), every append is persisted
under the same lock that protects the in-memory chain, and the writer
replays existing rows on construction. The hash-chain integrity check
(:meth:`verify`) then runs across the persisted rows so a tampered
on-disk file is detected at boot.

Determinism contract (INV-15 / TEST-01): given the same sequence of
``append`` calls, every entry — including the chain hash — is
bit-identical across runs and across the in-memory / SQLite backends.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Lock

from core.contracts.governance import LedgerEntry

# 32 zero hex digits — distinct from any sha256 output (which is 64 chars)
# so the genesis row is unambiguous when reading the chain.
GENESIS_PREV_HASH = "0" * 64


def _canonical_payload(payload: Mapping[str, str]) -> str:
    """Stable serialisation: sorted ``k=v`` pairs joined by ``\\x1f``.

    Used to build the bytes that go into the chain hash. ``\\x1f`` (Unit
    Separator) is a control character that cannot appear in a Python
    identifier or in a base16 digest, which keeps the encoding
    unambiguous.
    """

    parts = [f"{k}={payload[k]}" for k in sorted(payload)]
    return "\x1f".join(parts)


def _row_bytes(
    seq: int, ts_ns: int, kind: str, payload: Mapping[str, str], prev_hash: str
) -> bytes:
    canonical = (
        f"{seq}\x1e{ts_ns}\x1e{kind}\x1e{_canonical_payload(payload)}"
        f"\x1e{prev_hash}"
    )
    return canonical.encode("utf-8")


# ---------------------------------------------------------------------------
# Sprint-1 / Class-B "Trust the Ledger" — optional SQLite WAL durability.
# ---------------------------------------------------------------------------
#
# Table schema (one row per ledger entry, append-only):
#
#   seq         INTEGER PRIMARY KEY  — gap-free, matches LedgerEntry.seq
#   ts_ns       INTEGER NOT NULL     — wall-ns from system.time_source
#   kind        TEXT NOT NULL        — non-empty
#   payload     TEXT NOT NULL        — JSON of dict[str, str]
#   prev_hash   TEXT NOT NULL        — 64 hex
#   hash_chain  TEXT NOT NULL        — 64 hex
#
# The hash chain remains the source of truth; the schema only persists it.
# WAL mode + synchronous=NORMAL keeps boot-time replay durable across
# Ctrl+C / kill -9 without paying full-fsync cost on every append.

_DDL = """
CREATE TABLE IF NOT EXISTS authority_ledger (
    seq         INTEGER PRIMARY KEY,
    ts_ns       INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    prev_hash   TEXT NOT NULL,
    hash_chain  TEXT NOT NULL
)
"""


def _open_sqlite(db_path: Path) -> sqlite3.Connection:
    """Open the SQLite store in WAL mode with autocommit semantics."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # ``isolation_level=None`` puts sqlite3 in autocommit mode so each
    # ``execute`` is its own transaction; combined with WAL this gives
    # a per-row durability guarantee with low overhead.
    conn = sqlite3.connect(
        str(db_path),
        isolation_level=None,
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(_DDL)
    return conn


def _replay_from_sqlite(conn: sqlite3.Connection) -> list[LedgerEntry]:
    """Read every persisted row, oldest first, into ``LedgerEntry``s.

    The chain integrity check happens later via :meth:`verify`; this
    function only deserialises rows in order.
    """
    cur = conn.execute(
        "SELECT seq, ts_ns, kind, payload, prev_hash, hash_chain "
        "FROM authority_ledger ORDER BY seq ASC"
    )
    out: list[LedgerEntry] = []
    for seq, ts_ns, kind, payload_json, prev_hash, chain in cur.fetchall():
        out.append(
            LedgerEntry(
                seq=int(seq),
                ts_ns=int(ts_ns),
                kind=str(kind),
                payload=dict(json.loads(payload_json)),
                prev_hash=str(prev_hash),
                hash_chain=str(chain),
            )
        )
    return out


class LedgerAuthorityWriter:
    """Append-only hash-chained store of governance decisions.

    Thread-safe: ``append`` and ``read`` are guarded by a single lock,
    so multi-threaded operator bridges or hazard sensors cannot
    interleave rows. When ``db_path`` is provided, the same lock guards
    the SQLite write so the persisted file is always consistent with
    the in-memory mirror.
    """

    name: str = "ledger_authority_writer"
    spec_id: str = "GOV-CP-05"

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._rows: list[LedgerEntry] = []
        self._lock = Lock()
        self._conn: sqlite3.Connection | None = None
        self._db_path: Path | None = None
        if db_path is not None:
            path = Path(db_path)
            self._conn = _open_sqlite(path)
            self._db_path = path
            self._rows = _replay_from_sqlite(self._conn)
            # Boot-time integrity gate: a tampered file (manually edited
            # row, partially-written truncation that escaped WAL) shows
            # up here as a hash mismatch and aborts startup loudly.
            if not self._verify_locked():
                raise RuntimeError(
                    f"durable authority ledger at {path} failed hash-chain "
                    "verification on replay (tampered or corrupted file)"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        ts_ns: int,
        kind: str,
        payload: Mapping[str, str],
    ) -> LedgerEntry:
        """Append one row, returning the materialised :class:`LedgerEntry`.

        Empty ``kind`` is rejected — every authority row must declare
        what kind of decision it represents (e.g.
        ``"MODE_TRANSITION"``, ``"PLUGIN_LIFECYCLE"``,
        ``"OPERATOR_REJECTED"``).
        """

        if not kind:
            raise ValueError("kind must be non-empty")

        with self._lock:
            seq = len(self._rows)
            prev_hash = (
                self._rows[-1].hash_chain if self._rows else GENESIS_PREV_HASH
            )
            row_bytes = _row_bytes(seq, ts_ns, kind, payload, prev_hash)
            chain = hashlib.sha256(prev_hash.encode("ascii") + row_bytes).hexdigest()
            entry = LedgerEntry(
                seq=seq,
                ts_ns=ts_ns,
                kind=kind,
                payload=dict(payload),
                prev_hash=prev_hash,
                hash_chain=chain,
            )
            if self._conn is not None:
                # Persist before mirroring in memory: if the SQLite
                # write throws (e.g. disk full), the in-memory chain is
                # not advanced, so a retry is safe.
                self._conn.execute(
                    "INSERT INTO authority_ledger "
                    "(seq, ts_ns, kind, payload, prev_hash, hash_chain) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        seq,
                        ts_ns,
                        kind,
                        json.dumps(dict(payload), sort_keys=True),
                        prev_hash,
                        chain,
                    ),
                )
            self._rows.append(entry)
            return entry

    def read(self) -> Sequence[LedgerEntry]:
        """Snapshot of the current chain (oldest first)."""

        with self._lock:
            return tuple(self._rows)

    def head_hash(self) -> str:
        """Hash of the most recently appended row (or genesis)."""

        with self._lock:
            return self._rows[-1].hash_chain if self._rows else GENESIS_PREV_HASH

    def verify(self) -> bool:
        """Recompute the chain top-to-bottom; return ``False`` on tamper."""

        with self._lock:
            return self._verify_locked()

    def _verify_locked(self) -> bool:
        prev = GENESIS_PREV_HASH
        for entry in self._rows:
            row_bytes = _row_bytes(
                entry.seq, entry.ts_ns, entry.kind, entry.payload, prev
            )
            expected = hashlib.sha256(
                prev.encode("ascii") + row_bytes
            ).hexdigest()
            if expected != entry.hash_chain or entry.prev_hash != prev:
                return False
            prev = entry.hash_chain
        return True

    @property
    def db_path(self) -> Path | None:
        """Filesystem path of the durable backing store, if any."""
        return self._db_path

    def close(self) -> None:
        """Release the SQLite handle (idempotent)."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


__all__ = ["GENESIS_PREV_HASH", "LedgerAuthorityWriter"]
