"""
tests/test_knowledge_store.py — T0-11 bounded knowledge-store tests.
"""
from __future__ import annotations

import threading

import pytest

from mind.knowledge_store import KnowledgeEntry, KnowledgeStore


def test_put_and_get_roundtrip() -> None:
    s = KnowledgeStore()
    s.put("a", "hello")
    assert s.get("a") == "hello"
    assert "a" in s
    assert len(s) == 1


def test_get_missing_returns_none() -> None:
    s = KnowledgeStore()
    assert s.get("nope") is None
    assert "nope" not in s


def test_entry_count_cap_triggers_lru_eviction() -> None:
    s = KnowledgeStore(max_entries=3, max_bytes=10 ** 9)
    s.put("a", 1)
    s.put("b", 2)
    s.put("c", 3)
    s.put("d", 4)  # evicts "a" (oldest)
    assert s.get("a") is None
    assert s.get("b") == 2
    assert s.get("c") == 3
    assert s.get("d") == 4
    assert s.eviction_count() == 1


def test_byte_cap_triggers_lru_eviction() -> None:
    # tiny cap so string values overflow quickly
    s = KnowledgeStore(max_entries=1000, max_bytes=128)
    s.put("a", "x" * 80)
    s.put("b", "y" * 80)  # must evict "a"
    assert "a" not in s
    assert s.get("b") is not None
    assert s.bytes_used() <= 128
    assert s.eviction_count() >= 1


def test_get_touches_lru_order() -> None:
    s = KnowledgeStore(max_entries=3, max_bytes=10 ** 9)
    s.put("a", 1)
    s.put("b", 2)
    s.put("c", 3)
    _ = s.get("a")          # a is now MRU
    s.put("d", 4)           # should evict b (oldest untouched)
    assert "a" in s
    assert "b" not in s
    assert "c" in s
    assert "d" in s


def test_peek_does_not_touch_lru_order() -> None:
    s = KnowledgeStore(max_entries=3, max_bytes=10 ** 9)
    s.put("a", 1)
    s.put("b", 2)
    s.put("c", 3)
    assert s.peek("a") is not None
    s.put("d", 4)           # a should still be evicted
    assert "a" not in s


def test_delete() -> None:
    s = KnowledgeStore()
    s.put("k", "v")
    assert s.delete("k") is True
    assert "k" not in s
    assert s.delete("k") is False


def test_clear() -> None:
    s = KnowledgeStore()
    for i in range(10):
        s.put(f"k{i}", i)
    s.clear()
    assert len(s) == 0
    assert s.bytes_used() == 0


def test_compact_removes_low_confidence_entries() -> None:
    s = KnowledgeStore()
    s.put("keep", "yes", confidence=0.9)
    s.put("drop_1", "low", confidence=0.05)
    s.put("drop_2", "low", confidence=0.09)
    s.put("edge", "edge", confidence=0.10)  # equal to threshold = kept
    removed = s.compact(min_confidence=0.10)
    assert removed == 2
    assert "keep" in s
    assert "drop_1" not in s
    assert "drop_2" not in s
    assert "edge" in s


def test_snapshot_restore_roundtrip() -> None:
    s = KnowledgeStore()
    s.put("a", {"x": 1}, confidence=0.7, tags=("macro",))
    s.put("b", [1, 2, 3], confidence=0.4)
    snap = s.snapshot()

    s2 = KnowledgeStore()
    s2.restore(snap)
    assert len(s2) == 2
    assert s2.get("a") == {"x": 1}
    assert s2.get("b") == [1, 2, 3]
    entry_a = s2.peek("a")
    assert entry_a is not None
    assert entry_a.confidence == pytest.approx(0.7)
    assert entry_a.tags == ("macro",)


def test_invalid_confidence_rejected() -> None:
    s = KnowledgeStore()
    with pytest.raises(ValueError):
        s.put("x", 1, confidence=1.5)
    with pytest.raises(ValueError):
        s.put("x", 1, confidence=-0.1)


def test_empty_key_rejected() -> None:
    s = KnowledgeStore()
    with pytest.raises(ValueError):
        s.put("", 1)


def test_invalid_caps_rejected() -> None:
    with pytest.raises(ValueError):
        KnowledgeStore(max_entries=0)
    with pytest.raises(ValueError):
        KnowledgeStore(max_bytes=0)


def test_concurrent_writes_stay_bounded() -> None:
    """Hammer the store from 8 threads and verify it never exceeds cap."""
    s = KnowledgeStore(max_entries=100, max_bytes=10 ** 9)

    def worker(offset: int) -> None:
        for i in range(1000):
            s.put(f"k-{offset}-{i}", i)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(s) <= 100
    assert s.eviction_count() > 0


def test_entry_dataclass_is_frozen() -> None:
    e = KnowledgeEntry(key="k", value=1)
    with pytest.raises(Exception):
        e.key = "mutated"  # type: ignore[misc]


def test_oversized_put_is_rejected_without_wiping_store() -> None:
    """Regression guard: a single oversized put() must NOT cascade-evict
    the entire store. Before the fix, putting a value larger than
    max_bytes would (a) push out every existing entry and (b) finally
    evict itself, leaving the store empty."""
    s = KnowledgeStore(max_entries=1000, max_bytes=100)
    s.put("a", "x" * 30)
    s.put("b", "y" * 30)
    s.put("c", "z" * 20)
    assert len(s) == 3
    pre_evictions = s.eviction_count()

    with pytest.raises(ValueError, match="exceeds max_bytes"):
        s.put("huge", "Q" * 500)

    assert len(s) == 3, "existing entries must survive rejected oversized put"
    assert s.get("a") is not None
    assert s.get("b") is not None
    assert s.get("c") is not None
    assert "huge" not in s
    assert s.eviction_count() == pre_evictions


def test_oversized_restore_row_is_skipped() -> None:
    """Regression guard: restore() must not admit rows larger than the
    byte cap, otherwise the same cascading eviction destroys the rest
    of the replayed snapshot."""
    snap = [
        {"key": "ok1", "value": "x" * 10, "confidence": 0.5,
         "inserted_at_ns": 0, "size_bytes": 10, "tags": []},
        {"key": "huge", "value": "Q" * 500, "confidence": 0.5,
         "inserted_at_ns": 0, "size_bytes": 500, "tags": []},
        {"key": "ok2", "value": "y" * 10, "confidence": 0.5,
         "inserted_at_ns": 0, "size_bytes": 10, "tags": []},
    ]
    s = KnowledgeStore(max_entries=1000, max_bytes=100)
    s.restore(snap)
    assert s.get("ok1") is not None
    assert s.get("ok2") is not None
    assert "huge" not in s


def test_replace_existing_key_updates_value_and_size() -> None:
    s = KnowledgeStore()
    s.put("k", "short")
    bytes_before = s.bytes_used()
    s.put("k", "a" * 200)
    assert s.bytes_used() > bytes_before
    assert s.get("k") == "a" * 200
    assert len(s) == 1
