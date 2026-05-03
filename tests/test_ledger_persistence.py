"""Sprint-1 / Class-B "Trust the Ledger" — durability + replay tests.

Covers the three crash-recovery contracts the architectural review
flagged as P0:

1. ``LedgerAuthorityWriter`` persists to SQLite WAL when constructed
   with ``db_path=...`` and a fresh writer pointed at the same file
   replays the rows in seq order with the hash chain intact.
2. ``derive_system_intent`` over the replayed rows reproduces the
   ``SystemIntent`` that was active just before "crash".
3. ``rehydrate_approval_queue_from_ledger`` over the replayed rows
   reproduces the ``ApprovalQueue`` projection (PENDING / APPROVED /
   REJECTED entries by ``proposal_id``).

Together these prove the runtime can survive a ``Ctrl+C`` / ``kill -9``
between any two ledger appends and boot back into the same governance
state, which is the whole point of the SQLite backing store.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.coherence.system_intent import (
    DEFAULT_SYSTEM_INTENT,
    INTENT_KEY_FOCUS,
    INTENT_KEY_HORIZON,
    INTENT_KEY_OBJECTIVE,
    INTENT_KEY_RISK_MODE,
    INTENT_TRANSITION_KIND,
    derive_system_intent,
)
from governance_engine.control_plane.ledger_authority_writer import (
    GENESIS_PREV_HASH,
    LedgerAuthorityWriter,
)


def _intent_payload(
    objective: str, risk_mode: str, horizon: str, focus: str
) -> dict[str, str]:
    return {
        INTENT_KEY_OBJECTIVE: objective,
        INTENT_KEY_RISK_MODE: risk_mode,
        INTENT_KEY_HORIZON: horizon,
        INTENT_KEY_FOCUS: focus,
    }


def test_sqlite_backend_survives_restart_with_chain_intact(tmp_path: Path) -> None:
    """First writer persists rows; a fresh writer replays them verbatim."""

    db = tmp_path / "authority.db"
    first = LedgerAuthorityWriter(db_path=db)
    e0 = first.append(
        ts_ns=1_000,
        kind="MODE_TRANSITION",
        payload={"from": "SAFE", "to": "PAPER"},
    )
    e1 = first.append(
        ts_ns=1_001,
        kind=INTENT_TRANSITION_KIND,
        payload=_intent_payload(
            "CAPITAL_PRESERVATION", "DEFENSIVE", "INTRADAY", ""
        ),
    )
    head_before = first.head_hash()
    assert e0.prev_hash == GENESIS_PREV_HASH
    assert e1.prev_hash == e0.hash_chain
    first.close()

    # New process — same file, same chain.
    second = LedgerAuthorityWriter(db_path=db)
    rows = second.read()
    assert len(rows) == 2
    assert rows[0] == e0
    assert rows[1] == e1
    assert second.head_hash() == head_before
    assert second.verify() is True
    second.close()


def test_replayed_chain_supports_continued_appends(tmp_path: Path) -> None:
    """A replayed writer continues the chain seamlessly."""

    db = tmp_path / "authority.db"
    first = LedgerAuthorityWriter(db_path=db)
    first.append(ts_ns=10, kind="ALPHA", payload={"k": "v0"})
    first.close()

    second = LedgerAuthorityWriter(db_path=db)
    e1 = second.append(ts_ns=20, kind="BETA", payload={"k": "v1"})
    assert e1.seq == 1
    assert e1.prev_hash == second.read()[0].hash_chain
    assert second.verify() is True
    second.close()

    third = LedgerAuthorityWriter(db_path=db)
    rows = third.read()
    assert [r.kind for r in rows] == ["ALPHA", "BETA"]
    assert third.verify() is True
    third.close()


def test_tampered_sqlite_file_aborts_boot(tmp_path: Path) -> None:
    """Manually corrupting a stored row breaks chain verification at boot."""

    import sqlite3

    db = tmp_path / "authority.db"
    first = LedgerAuthorityWriter(db_path=db)
    first.append(ts_ns=1, kind="MODE_TRANSITION", payload={"from": "SAFE"})
    first.append(ts_ns=2, kind="MODE_TRANSITION", payload={"from": "PAPER"})
    first.close()

    # Tamper with payload of seq=0 — chain hash for seq=1 must now mismatch.
    conn = sqlite3.connect(str(db))
    conn.execute(
        "UPDATE authority_ledger SET payload=? WHERE seq=?",
        ('{"from":"TAMPERED"}', 0),
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="hash-chain verification"):
        LedgerAuthorityWriter(db_path=db)


def test_system_intent_recovers_across_restart(tmp_path: Path) -> None:
    """``derive_system_intent`` over replayed rows returns the last intent."""

    db = tmp_path / "authority.db"
    first = LedgerAuthorityWriter(db_path=db)
    first.append(
        ts_ns=100,
        kind=INTENT_TRANSITION_KIND,
        payload=_intent_payload(
            "YIELD_OPTIMIZATION", "AGGRESSIVE", "SWING", ""
        ),
    )
    first.append(
        ts_ns=200,
        kind="MODE_TRANSITION",
        payload={"from": "SAFE", "to": "PAPER"},
    )
    first.append(
        ts_ns=300,
        kind=INTENT_TRANSITION_KIND,
        payload=_intent_payload(
            "CAPITAL_PRESERVATION", "DEFENSIVE", "INTRADAY", ""
        ),
    )
    expected_intent = derive_system_intent(first.read())
    assert expected_intent != DEFAULT_SYSTEM_INTENT
    assert expected_intent.objective.value == "CAPITAL_PRESERVATION"
    first.close()

    second = LedgerAuthorityWriter(db_path=db)
    recovered = derive_system_intent(second.read())
    assert recovered == expected_intent
    second.close()


def test_approval_queue_rehydrates_across_restart(tmp_path: Path) -> None:
    """The ledger-backed approval projection survives restart unchanged."""

    from intelligence_engine.cognitive.approval_projection import (
        APPROVED_KIND,
        PENDING_KIND,
        REJECTED_KIND,
    )
    from intelligence_engine.cognitive.approval_queue import ApprovalQueue
    from ui.cognitive_chat_runtime import rehydrate_approval_queue_from_ledger

    db = tmp_path / "authority.db"
    first = LedgerAuthorityWriter(db_path=db)
    pending_p1 = {
        "approval_id": "P-1",
        "thread_id": "T-1",
        "ts_ns": "10",
        "symbol": "BTC-USD",
        "side": "BUY",
        "confidence": "0.8",
        "rationale": "",
    }
    pending_p2 = {
        "approval_id": "P-2",
        "thread_id": "T-2",
        "ts_ns": "20",
        "symbol": "ETH-USD",
        "side": "SELL",
        "confidence": "0.6",
        "rationale": "",
    }
    first.append(ts_ns=10, kind=PENDING_KIND, payload=pending_p1)
    first.append(ts_ns=20, kind=PENDING_KIND, payload=pending_p2)
    first.append(
        ts_ns=30,
        kind=APPROVED_KIND,
        payload={
            "approval_id": "P-1",
            "ts_ns": "30",
            "decided_by": "operator",
        },
    )
    first.append(
        ts_ns=40,
        kind=REJECTED_KIND,
        payload={
            "approval_id": "P-2",
            "ts_ns": "40",
            "decided_by": "operator",
            "reason": "size_cap",
        },
    )
    first.close()

    second = LedgerAuthorityWriter(db_path=db)
    queue = ApprovalQueue()
    n = rehydrate_approval_queue_from_ledger(queue, second)
    # Both proposals should be terminal — none pending — but visible
    # in the full audit trail.
    assert n == 2
    assert queue.list() == ()
    full = queue.list(include_decided=True)
    assert {row.request_id for row in full} == {"P-1", "P-2"}
    second.close()
