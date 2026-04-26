"""Broker adapter Protocol (Phase E1).

A :class:`BrokerAdapter` is a side-effect boundary: it is the only place
the Execution Engine talks to a venue (real or simulated). Adapters MUST
be deterministic given identical inputs (TEST-01), or — for live adapters
— must be exercised behind a recording layer that re-emits identical
:class:`ExecutionEvent` objects on replay.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core.contracts.events import ExecutionEvent, SignalEvent


@runtime_checkable
class BrokerAdapter(Protocol):
    """Adapter contract.

    Attributes:
        name: Short, stable adapter identifier (e.g. ``"paper"``,
            ``"binance_spot"``). Logged into ``ExecutionEvent.venue`` and
            referenced from ``registry/plugins.yaml``.
    """

    name: str

    def submit(self, signal: SignalEvent, mark_price: float) -> ExecutionEvent:
        """Convert one approved signal into one ``ExecutionEvent``.

        Implementations must be pure functions of ``(signal, mark_price)``
        plus their internal monotonic counters; no external IO on the
        canonical hot path (INV-17, T1).
        """
        ...


__all__ = ["BrokerAdapter"]
