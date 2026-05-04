"""WEBLEARN-04 (HITL-07) — bounded pending buffer for HITL review.

The pending buffer is the *only* stateful component of the web
autolearn pipeline. Curated items land here; the operator dashboard
reads pending items, approves or rejects them, and approved items
become :class:`SignalEvent` instances on the canonical bus (wiring
landed by a separate ``ui/server.py`` change in a follow-up PR).

Design constraints:

  * **Bounded** — the buffer caps at ``capacity`` items. Once full,
    new items either raise :class:`HitlBufferFull` (strict mode,
    default) or evict the oldest entry (FIFO mode). Strict is the
    safe default because silently dropping curator output would lose
    information without an audit trail. FIFO mode exists for tests
    that exercise eviction.
  * **FIFO ordering** — :meth:`pending` returns items in the order
    they were added. The operator sees the oldest first.
  * **Idempotent enqueue** — adding the same ``(seed_id, url)`` pair
    twice is rejected (returns False) so the dashboard never shows
    duplicates from re-crawls.
  * **Deterministic-replay safe** — no clock reads. Every method is
    pure given the buffer state.

Authority discipline: no engine imports, no FSM mutation, no ledger
writes. Approval / rejection decisions performed by the operator are
*not* recorded here — the harness is responsible for emitting an
``OPERATOR_*`` system event on the canonical bus when it dequeues.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock

from sensory.web_autolearn.contracts import CuratedItem


class HitlBufferFull(RuntimeError):
    """Raised by :meth:`PendingBuffer.add` in strict mode when full."""


@dataclass(frozen=True, slots=True)
class PendingItem:
    """One row in the pending buffer.

    Wraps :class:`CuratedItem` with a stable ``hitl_id`` so the
    operator dashboard can address individual rows by id rather than
    by ``(seed_id, url)`` (which would be lossy if the same URL is
    re-curated under a different seed).
    """

    hitl_id: str
    curated: CuratedItem


@dataclass
class PendingBuffer:
    """Bounded FIFO holder of :class:`CuratedItem` awaiting HITL.

    Args:
        capacity: Max items the buffer holds. Must be >= 1.
        evict_oldest_when_full: If True, ``add`` silently FIFO-evicts
            the oldest item when full. If False (default), ``add``
            raises :class:`HitlBufferFull`. Strict mode is the safe
            default.
    """

    capacity: int
    evict_oldest_when_full: bool = False
    _items: OrderedDict[str, PendingItem] = field(
        default_factory=OrderedDict, repr=False, compare=False
    )
    _lock: Lock = field(
        default_factory=Lock, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("PendingBuffer.capacity must be >= 1")

    @staticmethod
    def _hitl_id_for(curated: CuratedItem) -> str:
        # Stable, deterministic-replay-safe id derived from
        # (seed_id, url, ts_ns). Tab-separated to avoid ambiguity.
        return f"{curated.seed_id}\t{curated.url}\t{curated.ts_ns}"

    def add(self, curated: CuratedItem) -> bool:
        """Enqueue ``curated``; return True iff it was added.

        Returns False if a row with the same ``hitl_id`` already
        exists (idempotent re-enqueue).

        Raises :class:`HitlBufferFull` when full and
        ``evict_oldest_when_full`` is False.
        """

        hitl_id = self._hitl_id_for(curated)
        with self._lock:
            if hitl_id in self._items:
                return False
            if len(self._items) >= self.capacity:
                if not self.evict_oldest_when_full:
                    raise HitlBufferFull(
                        f"PendingBuffer full at capacity"
                        f" {self.capacity}"
                    )
                # FIFO evict
                self._items.popitem(last=False)
            self._items[hitl_id] = PendingItem(
                hitl_id=hitl_id, curated=curated
            )
            return True

    def pending(self) -> tuple[PendingItem, ...]:
        """Return all pending rows in insertion order."""

        with self._lock:
            return tuple(self._items.values())

    def take(self, hitl_id: str) -> PendingItem | None:
        """Remove and return the row addressed by ``hitl_id``.

        Returns None if no such row exists.
        """

        with self._lock:
            return self._items.pop(hitl_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


__all__ = [
    "HitlBufferFull",
    "PendingBuffer",
    "PendingItem",
]
