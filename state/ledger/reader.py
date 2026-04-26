"""Read-only ledger interface (Phase E0 stub).

This is the **only** ledger entry point that offline engines (Learning,
Evolution) and runtime engines other than Governance may import. Lint rule
**L2** depends on this allow-list.

The full implementation lands in Phase E1+ (event-sourced reconstruction,
hash-chain verification, snapshot replay). For Phase E0 we expose a minimal
typed surface so engines can be wired without depending on storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

from core.contracts.events import Event


@dataclass(frozen=True, slots=True)
class LedgerCursor:
    """Opaque cursor used to resume tailing a ledger range."""

    seq: int = 0


class LedgerReader:
    """Phase E0 in-memory ledger reader stub.

    Real implementation: hash-chained append-only log (CORE-05) +
    periodic snapshots (T0-0). This stub returns whatever events were
    appended to it via :meth:`_seed_for_tests`, in order.
    """

    def __init__(self) -> None:
        self._events: list[Event] = []

    # Internal helper used by tests / fixtures only — not a public API.
    def _seed_for_tests(self, events: Iterable[Event]) -> None:
        self._events.extend(events)

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
        end = len(self._events) if limit is None else min(
            len(self._events), start + limit
        )
        return tuple(self._events[start:end])

    def tail(self, cursor: LedgerCursor | None = None) -> Iterator[Event]:
        """Yield events from ``cursor`` to the current head.

        Real implementation will be a long-lived iterator that blocks on
        the next append; this stub yields the snapshot once.
        """
        yield from self.read(cursor)


__all__ = ["LedgerCursor", "LedgerReader"]
