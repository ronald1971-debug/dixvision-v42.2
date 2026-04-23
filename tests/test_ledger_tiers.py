"""Tests for state.ledger.hot_store / cold_store / indexer — T0-5."""
from __future__ import annotations

from typing import Any

import pytest

from state.ledger.cold_store import (
    MAX_QUERY_LIMIT,
    ColdStore,
)
from state.ledger.hot_store import HotStore
from state.ledger.indexer import LedgerIndexer


def _event(seq: int, et: str = "MARKET", st: str = "TICK", src: str = "INDIRA",
           **extra: Any) -> dict:
    return {
        "sequence": seq,
        "event_type": et,
        "sub_type": st,
        "source": src,
        "payload": {"seq": seq, **extra},
        "event_hash": f"h{seq}",
        "event_id": f"id{seq}",
    }


# ─────── hot store ──────────────────────────────────────────────────────


def test_hot_store_capacity_is_enforced() -> None:
    hs = HotStore(capacity=3)
    for i in range(1, 8):
        hs.add(_event(i))
    assert len(hs) == 3
    assert hs.earliest_sequence() == 5
    assert hs.last_sequence() == 7


def test_hot_store_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError):
        HotStore(capacity=0)


def test_hot_store_recent_filters_by_type_and_source() -> None:
    hs = HotStore(capacity=100)
    hs.add(_event(1, et="MARKET", src="INDIRA"))
    hs.add(_event(2, et="HAZARD", src="DYON"))
    hs.add(_event(3, et="MARKET", src="INDIRA"))
    hs.add(_event(4, et="GOVERNANCE", src="GOVERNANCE"))
    market = hs.recent(event_type="MARKET")
    assert [e.sequence for e in market] == [3, 1]
    dyon = hs.recent(source="DYON")
    assert [e.sequence for e in dyon] == [2]


def test_hot_store_events_after_returns_only_newer() -> None:
    hs = HotStore(capacity=100)
    for i in range(1, 6):
        hs.add(_event(i))
    after = hs.events_after(3)
    assert [e.sequence for e in after] == [4, 5]


def test_hot_store_events_after_raises_on_aged_gap() -> None:
    """If the caller asks for events the ring has already aged out, we
    must raise — silently returning an incomplete list would let a
    projector skip events."""
    hs = HotStore(capacity=3)
    for i in range(1, 10):
        hs.add(_event(i))
    # ring now holds sequences 7, 8, 9
    with pytest.raises(LookupError):
        hs.events_after(2)


def test_hot_store_clear_resets_sequences() -> None:
    hs = HotStore(capacity=10)
    hs.add(_event(1))
    hs.clear()
    assert len(hs) == 0
    assert hs.last_sequence() == -1
    assert hs.earliest_sequence() == -1


def test_hot_store_contains_sequence() -> None:
    hs = HotStore(capacity=10)
    hs.add(_event(7))
    assert hs.contains_sequence(7) is True
    assert hs.contains_sequence(8) is False


# ─────── cold store ─────────────────────────────────────────────────────


class _RecordingReader:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[dict] = []

    def query(
        self,
        event_type: str | None = None,
        source: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        self.calls.append(
            {"event_type": event_type, "source": source, "limit": limit}
        )
        return list(self.rows[:limit])


def test_cold_store_forwards_arguments_to_reader() -> None:
    reader = _RecordingReader([_event(i) for i in range(5)])
    cs = ColdStore(reader=reader)
    cs.query(event_type="HAZARD", source="DYON", limit=2)
    assert reader.calls == [
        {"event_type": "HAZARD", "source": "DYON", "limit": 2},
    ]


def test_cold_store_rejects_non_positive_limit() -> None:
    cs = ColdStore(reader=_RecordingReader([]))
    with pytest.raises(ValueError):
        cs.query(limit=0)


def test_cold_store_clamps_runaway_limit() -> None:
    reader = _RecordingReader([])
    cs = ColdStore(reader=reader)
    cs.query(limit=MAX_QUERY_LIMIT * 10)
    assert reader.calls[-1]["limit"] == MAX_QUERY_LIMIT


# ─────── indexer ────────────────────────────────────────────────────────


def test_indexer_records_sequences_per_dimension() -> None:
    idx = LedgerIndexer(per_key_capacity=100)
    idx.index(_event(1, et="MARKET", st="TICK", src="INDIRA"))
    idx.index(_event(2, et="HAZARD", st="SEVERE", src="DYON"))
    idx.index(_event(3, et="MARKET", st="TRADE", src="INDIRA"))
    assert idx.recent_sequences_by_type("MARKET") == [3, 1]
    assert idx.recent_sequences_by_source("DYON") == [2]
    assert idx.recent_sequences_for("MARKET", "TICK") == [1]


def test_indexer_bounded_per_key() -> None:
    idx = LedgerIndexer(per_key_capacity=3)
    for i in range(1, 8):
        idx.index(_event(i, et="MARKET"))
    seqs = idx.recent_sequences_by_type("MARKET")
    # only the last 3 retained, newest first
    assert seqs == [7, 6, 5]


def test_indexer_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError):
        LedgerIndexer(per_key_capacity=0)


def test_indexer_clear_resets_all_dimensions() -> None:
    idx = LedgerIndexer()
    idx.index(_event(1, et="MARKET"))
    idx.clear()
    assert idx.recent_sequences_by_type("MARKET") == []
    assert idx.recent_sequences_by_source("INDIRA") == []
    assert idx.recent_sequences_for("MARKET", "TICK") == []
