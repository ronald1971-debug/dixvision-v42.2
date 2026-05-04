"""Read-only ledger interface (offline + dashboard surface).

This is the **only** ledger entry point that offline engines (Learning,
Evolution) and runtime engines other than Governance may import. Lint
rule **L2** depends on this allow-list.

Two surfaces live on the same reader:

* :meth:`LedgerReader.read` / :meth:`tail` — the legacy 4-event view
  consumed by the dashboard ``DecisionTracePanel``. Backed by an
  in-process buffer that tests seed via :meth:`_seed_for_tests`.
* :meth:`LedgerReader.authority_entries` /
  :meth:`authority_count` — AUDIT-P0.2: SQLite-backed read of the
  governance authority ledger that ``LedgerAuthorityWriter`` writes
  (PR #164). When the harness boots with ``DIXVISION_LEDGER_PATH``
  set, the reader opens the same SQLite file in **read-only** mode
  and exposes ``LedgerEntry`` rows in seq order. Without a path the
  authority surface is empty (callers degrade gracefully).

The read-only opener uses ``mode=ro&immutable=0`` so multiple readers
can coexist with the writer's WAL session without holding write
locks. Writes still flow through ``LedgerAuthorityWriter`` only —
this module never mutates the file.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from core.contracts.events import Event
from core.contracts.governance import LedgerEntry


@dataclass(frozen=True, slots=True)
class LedgerCursor:
    """Opaque cursor used to resume tailing a ledger range.

    For the legacy event view ``seq`` is an index into the in-process
    buffer. For the authority surface it is the next ``LedgerEntry.seq``
    to return — exclusive lower bound (rows with ``seq >= cursor.seq``
    are returned).
    """

    seq: int = 0


class LedgerReader:
    """Read-only ledger surface for offline engines and dashboard widgets.

    Construction modes:

    * ``LedgerReader()`` — in-process event buffer only (test default
      and pre-PR-#164 behaviour). ``authority_entries()`` returns an
      empty tuple in this mode.
    * ``LedgerReader(db_path=Path(...))`` — also opens the writer's
      SQLite store in **read-only** mode. ``authority_entries()`` then
      streams ``LedgerEntry`` rows from disk in ``seq`` order.

    Thread-safety: ``sqlite3`` connections are not shared across
    threads in CPython by default, but the read-only handle here
    enables ``check_same_thread=False`` because dashboard widgets and
    offline engines may run on different threads. Reads are stateless
    (each call issues its own ``SELECT``) so cross-thread interleaving
    is safe.
    """

    def __init__(self, *, db_path: str | Path | None = None) -> None:
        self._events: list[Event] = []
        self._db_path: Path | None = (
            Path(db_path) if db_path is not None else None
        )
        self._conn: sqlite3.Connection | None = None
        if self._db_path is not None:
            self._conn = _open_readonly(self._db_path)

    # ------------------------------------------------------------------
    # Test seam — legacy event view
    # ------------------------------------------------------------------

    def _seed_for_tests(self, events: Iterable[Event]) -> None:
        """Internal helper used by tests / fixtures only."""
        self._events.extend(events)

    # ------------------------------------------------------------------
    # Legacy 4-event view (DecisionTracePanel, etc.)
    # ------------------------------------------------------------------

    def read(
        self,
        cursor: LedgerCursor | None = None,
        *,
        limit: int | None = None,
    ) -> Sequence[Event]:
        """Read events starting at ``cursor`` (default: from the head).

        Returns at most ``limit`` events; ``None`` returns all.
        """
        start = 0 if cursor is None else cursor.seq
        end = (
            len(self._events)
            if limit is None
            else min(len(self._events), start + limit)
        )
        return tuple(self._events[start:end])

    def tail(self, cursor: LedgerCursor | None = None) -> Iterator[Event]:
        """Yield events from ``cursor`` to the current head.

        Real implementation will be a long-lived iterator that blocks
        on the next append; this surface yields the snapshot once.
        """
        yield from self.read(cursor)

    # ------------------------------------------------------------------
    # AUDIT-P0.2 — SQLite-backed authority ledger view
    # ------------------------------------------------------------------

    @property
    def db_path(self) -> Path | None:
        """Filesystem path of the durable authority ledger, if any."""
        return self._db_path

    def authority_entries(
        self,
        cursor: LedgerCursor | None = None,
        *,
        limit: int | None = None,
    ) -> Sequence[LedgerEntry]:
        """Stream authority ledger rows in ``seq`` order.

        Without a ``db_path`` (legacy in-memory mode) returns an empty
        tuple — callers degrade gracefully so the dashboard does not
        404 the AuditLedgerViewer when the operator is running an
        ephemeral test build. With a ``db_path`` this issues one
        ``SELECT ... ORDER BY seq`` against the writer's SQLite store
        and rehydrates each row into a :class:`LedgerEntry`.

        ``cursor`` filters to rows with ``seq >= cursor.seq``; ``limit``
        caps the number of rows returned.
        """
        if self._conn is None:
            return ()
        start_seq = 0 if cursor is None else int(cursor.seq)
        sql = (
            "SELECT seq, ts_ns, kind, payload, prev_hash, hash_chain "
            "FROM authority_ledger WHERE seq >= ? ORDER BY seq ASC"
        )
        params: tuple[int, ...] = (start_seq,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (start_seq, int(limit))
        cur = self._conn.execute(sql, params)
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
        return tuple(out)

    def authority_count(self) -> int:
        """Total number of rows in the authority ledger.

        Returns ``0`` when no ``db_path`` is bound. Used by the
        dashboard ``AuditLedgerViewer`` to render a "X rows" badge
        without paging through the whole chain.
        """
        if self._conn is None:
            return 0
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM authority_ledger"
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Release the SQLite read handle (idempotent)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open ``db_path`` in SQLite read-only mode.

    The writer's WAL session keeps the file open in write mode; this
    handle uses the URI ``mode=ro`` opener so SQLite refuses any
    accidental write attempt and so multiple offline readers can
    share the file with the writer concurrently.
    """

    # ``immutable=0`` is the default but we make it explicit so the
    # opener documents the contract: this file *does* change, but not
    # via this handle.
    uri = f"file:{db_path}?mode=ro&immutable=0"
    conn = sqlite3.connect(
        uri,
        uri=True,
        isolation_level=None,
        check_same_thread=False,
    )
    return conn


__all__ = ["LedgerCursor", "LedgerReader"]
