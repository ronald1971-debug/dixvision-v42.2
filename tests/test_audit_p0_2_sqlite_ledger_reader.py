"""AUDIT-P0.2 — SQLite-backed LedgerReader regression tests.

Pins the offline / dashboard read surface against the durable authority
ledger that ``LedgerAuthorityWriter`` writes (PR #164). Every test in
this module exercises the contract that:

* Without a ``db_path``, the reader degrades gracefully (empty rows,
  legacy event buffer still works).
* With a ``db_path`` pointing at the writer's SQLite store, every
  ``LedgerEntry`` the writer appended is visible in ``seq`` order
  with the original ``kind`` / ``payload`` / ``hash_chain`` intact.
* The reader is read-only — accidental write attempts are refused by
  SQLite (the URI opener uses ``mode=ro``).
* Cursor + limit arguments page through the chain deterministically.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.contracts.events import (
    EventKind,
    Side,
    SignalEvent,
    SignalTrust,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from state.ledger.reader import LedgerCursor, LedgerReader


def _writer(db_path: Path) -> LedgerAuthorityWriter:
    """Build a SQLite-backed writer rooted at ``db_path``."""
    return LedgerAuthorityWriter(db_path=db_path)


def test_legacy_event_buffer_still_works_without_db_path() -> None:
    """``LedgerReader()`` with no ``db_path`` keeps the legacy surface.

    DecisionTracePanel and the other Phase 6 widgets rely on the
    in-process event buffer; the AUDIT-P0.2 changes must not break it.
    """

    reader = LedgerReader()
    event = SignalEvent(
        ts_ns=1,
        kind=EventKind.SIGNAL,
        produced_by_engine="intelligence_engine",
        symbol="BTCUSD",
        side=Side.BUY,
        confidence=0.7,
        plugin_chain=("microstructure_v1",),
        signal_trust=SignalTrust.INTERNAL,
    )
    reader._seed_for_tests([event])
    assert reader.read() == (event,)
    assert reader.authority_entries() == ()
    assert reader.authority_count() == 0
    assert reader.db_path is None


def test_authority_entries_replay_writer_rows(tmp_path: Path) -> None:
    """Every ``LedgerEntry`` the writer persists is visible to the reader."""

    db_path = tmp_path / "authority.db"
    writer = _writer(db_path)
    e1 = writer.append(
        ts_ns=1_000, kind="MODE_TRANSITION", payload={"to": "PAPER"}
    )
    e2 = writer.append(
        ts_ns=2_000,
        kind="STRATEGY_LIFECYCLE",
        payload={"strategy": "breakout_v1", "to": "SHADOW"},
    )
    e3 = writer.append(
        ts_ns=3_000,
        kind="OPERATOR_SETTINGS_CHANGED",
        payload={"setting": "autonomy_mode", "next_json": '"FULL_AUTO"'},
    )

    reader = LedgerReader(db_path=db_path)
    rows = reader.authority_entries()
    assert len(rows) == 3
    assert rows[0].seq == e1.seq
    assert rows[0].kind == "MODE_TRANSITION"
    assert rows[0].payload == {"to": "PAPER"}
    assert rows[0].hash_chain == e1.hash_chain
    assert rows[1].seq == e2.seq
    assert rows[2].seq == e3.seq
    assert reader.authority_count() == 3


def test_authority_entries_cursor_filters_by_seq(tmp_path: Path) -> None:
    """``cursor.seq`` is an inclusive lower bound on ``LedgerEntry.seq``."""

    db_path = tmp_path / "authority.db"
    writer = _writer(db_path)
    for i in range(5):
        writer.append(
            ts_ns=1_000 + i,
            kind="MODE_TRANSITION",
            payload={"step": str(i)},
        )

    reader = LedgerReader(db_path=db_path)
    tail = reader.authority_entries(LedgerCursor(seq=3))
    assert tuple(row.payload["step"] for row in tail) == ("3", "4")


def test_authority_entries_limit_caps_rows(tmp_path: Path) -> None:
    """``limit=N`` returns at most N rows."""

    db_path = tmp_path / "authority.db"
    writer = _writer(db_path)
    for i in range(10):
        writer.append(
            ts_ns=1_000 + i,
            kind="MODE_TRANSITION",
            payload={"step": str(i)},
        )

    reader = LedgerReader(db_path=db_path)
    rows = reader.authority_entries(limit=4)
    assert len(rows) == 4
    assert tuple(row.payload["step"] for row in rows) == ("0", "1", "2", "3")


def test_authority_entries_cursor_and_limit_compose(tmp_path: Path) -> None:
    """Pagination contract: cursor advances + limit caps."""

    db_path = tmp_path / "authority.db"
    writer = _writer(db_path)
    for i in range(10):
        writer.append(
            ts_ns=1_000 + i,
            kind="MODE_TRANSITION",
            payload={"step": str(i)},
        )

    reader = LedgerReader(db_path=db_path)
    page = reader.authority_entries(LedgerCursor(seq=5), limit=3)
    assert tuple(row.payload["step"] for row in page) == ("5", "6", "7")


def test_reader_is_read_only(tmp_path: Path) -> None:
    """The reader's SQLite handle must refuse writes (``mode=ro``).

    This guards against a future regression where someone wires the
    reader into a code path that accidentally tries to ``INSERT`` /
    ``UPDATE`` / ``DELETE``. SQLite raises ``OperationalError`` for
    any DML on a ``mode=ro`` URI handle.
    """

    db_path = tmp_path / "authority.db"
    writer = _writer(db_path)
    writer.append(ts_ns=1_000, kind="MODE_TRANSITION", payload={"to": "PAPER"})
    reader = LedgerReader(db_path=db_path)
    # Use the private attribute deliberately — this test is a contract
    # check on the read-only opener, not a public API exercise.
    conn = reader._conn
    assert conn is not None
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO authority_ledger "
            "(seq, ts_ns, kind, payload, prev_hash, hash_chain) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (99, 1, "x", "{}", "0" * 64, "a" * 64),
        )


def test_reader_close_releases_handle(tmp_path: Path) -> None:
    """``close()`` is idempotent and releases the SQLite handle."""

    db_path = tmp_path / "authority.db"
    _writer(db_path).append(
        ts_ns=1_000, kind="MODE_TRANSITION", payload={"to": "PAPER"}
    )
    reader = LedgerReader(db_path=db_path)
    assert reader._conn is not None
    reader.close()
    assert reader._conn is None
    # Idempotent — second close must not raise.
    reader.close()
    # After close, the legacy in-memory surface still works.
    assert reader.read() == ()


def test_reader_db_path_property_exposes_resolved_path(tmp_path: Path) -> None:
    """``reader.db_path`` round-trips the constructor argument."""

    db_path = tmp_path / "authority.db"
    _writer(db_path)
    reader = LedgerReader(db_path=db_path)
    assert reader.db_path == db_path


def test_authority_entries_match_writer_chain_hashes(tmp_path: Path) -> None:
    """The reader rehydrates ``hash_chain`` bit-identical to the writer.

    This pins the on-disk schema contract: writer-side hash-chain
    bytes survive serialise -> sqlite -> deserialise without
    corruption. If a future PR changes the canonical row encoding,
    this test fails before it can ship.
    """

    db_path = tmp_path / "authority.db"
    writer = _writer(db_path)
    written = [
        writer.append(
            ts_ns=1_000 + i,
            kind="STRATEGY_LIFECYCLE",
            payload={"to": "SHADOW", "strategy": f"s{i}"},
        )
        for i in range(4)
    ]
    reader = LedgerReader(db_path=db_path)
    for written_row, read_row in zip(
        written, reader.authority_entries(), strict=True
    ):
        assert written_row.seq == read_row.seq
        assert written_row.ts_ns == read_row.ts_ns
        assert written_row.kind == read_row.kind
        assert written_row.payload == read_row.payload
        assert written_row.prev_hash == read_row.prev_hash
        assert written_row.hash_chain == read_row.hash_chain
