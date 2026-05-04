"""Unit tests for sensory.web_autolearn.pending_buffer."""

from __future__ import annotations

import pytest

from sensory.web_autolearn.contracts import CuratedItem
from sensory.web_autolearn.pending_buffer import (
    HitlBufferFull,
    PendingBuffer,
)


def _curated(
    seed_id: str = "s",
    url: str = "https://x",
    ts_ns: int = 1,
) -> CuratedItem:
    return CuratedItem(
        ts_ns=ts_ns,
        seed_id=seed_id,
        url=url,
        title="t",
        body="b",
        score=0.5,
        seed_topic="crypto",
    )


def test_capacity_rejected_below_one() -> None:
    with pytest.raises(ValueError, match="capacity"):
        PendingBuffer(capacity=0)


def test_add_returns_true_for_new_item() -> None:
    buf = PendingBuffer(capacity=3)
    assert buf.add(_curated(url="https://a"))
    assert len(buf) == 1


def test_add_returns_false_on_duplicate() -> None:
    buf = PendingBuffer(capacity=3)
    assert buf.add(_curated())
    assert not buf.add(_curated())  # same id
    assert len(buf) == 1


def test_strict_full_raises() -> None:
    buf = PendingBuffer(capacity=2)
    buf.add(_curated(url="https://a"))
    buf.add(_curated(url="https://b"))
    with pytest.raises(HitlBufferFull):
        buf.add(_curated(url="https://c"))


def test_evict_oldest_when_full_drops_first_item() -> None:
    buf = PendingBuffer(capacity=2, evict_oldest_when_full=True)
    buf.add(_curated(url="https://a"))
    buf.add(_curated(url="https://b"))
    buf.add(_curated(url="https://c"))

    urls = [item.curated.url for item in buf.pending()]
    assert urls == ["https://b", "https://c"]


def test_pending_returns_fifo_order() -> None:
    buf = PendingBuffer(capacity=5)
    buf.add(_curated(url="https://1"))
    buf.add(_curated(url="https://2"))
    buf.add(_curated(url="https://3"))

    urls = [item.curated.url for item in buf.pending()]
    assert urls == ["https://1", "https://2", "https://3"]


def test_take_removes_and_returns_row() -> None:
    buf = PendingBuffer(capacity=2)
    buf.add(_curated(url="https://a"))
    buf.add(_curated(url="https://b"))
    [a, _] = buf.pending()

    taken = buf.take(a.hitl_id)
    assert taken is not None
    assert taken.curated.url == "https://a"
    assert len(buf) == 1


def test_take_returns_none_for_unknown() -> None:
    buf = PendingBuffer(capacity=2)
    assert buf.take("unknown") is None


def test_hitl_id_is_stable_per_curated_triplet() -> None:
    buf = PendingBuffer(capacity=2)
    a = _curated(url="https://x", ts_ns=1)
    b = _curated(url="https://x", ts_ns=1)  # same triplet
    c = _curated(url="https://x", ts_ns=2)  # different ts_ns

    assert buf.add(a)
    assert not buf.add(b)  # idempotent
    assert buf.add(c)  # distinct
    assert len(buf) == 2


def test_buffer_thread_safety_smoke() -> None:
    """Lock prevents corrupted state under concurrent ``add`` callers.

    Smoke-only: launches 8 threads each adding distinct URLs and
    asserts the final count matches.
    """

    import threading

    buf = PendingBuffer(capacity=200)

    def worker(prefix: int) -> None:
        for i in range(20):
            buf.add(_curated(url=f"https://{prefix}/{i}"))

    threads = [
        threading.Thread(target=worker, args=(p,)) for p in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(buf) == 8 * 20
