"""Shared scaffold for live BrokerAdapter implementations.

Live adapters (Hummingbot, Pump.fun, UniswapX, …) all share the same
operational lifecycle:

* ``DISCONNECTED`` — no credentials / no socket
* ``CONNECTING`` — handshake in progress
* ``READY`` — venue session healthy, accepting orders
* ``DEGRADED`` — partial outage (e.g. WS reconnect loop)
* ``HALTED`` — operator pulled the plug or kill_switch fired

Until the credentials are wired and the socket is alive, these adapters
must remain operationally honest: ``submit()`` returns an
``ExecutionEvent`` with ``status=REJECTED`` and a structured ``meta``
explanation rather than silently dropping the order or pretending to
fill. This keeps the Triad Lock invariant (INV-56) intact — the
adapter is the *Executor*, and an Executor that fakes fills is a hard
violation of the Approver/Executor decoupling.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    SignalEvent,
)


class AdapterState(StrEnum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    HALTED = "HALTED"


@dataclass(frozen=True)
class AdapterStatus:
    """Snapshot of one adapter's health for the operator dashboard."""

    name: str
    venue: str
    state: AdapterState
    detail: str
    last_heartbeat_ns: int = 0


class LiveAdapterBase:
    """Default scaffolding for a venue-bound adapter.

    Subclasses override :meth:`connect`, :meth:`disconnect`, and
    :meth:`_submit_live` to wire the actual venue. Until the venue is
    reachable, :meth:`submit` returns a structured ``REJECTED``
    ``ExecutionEvent`` so callers (and the audit ledger) can tell the
    difference between a real fill and a "scaffold only" path.

    Attributes:
        name: Stable adapter identifier (logged into ``ExecutionEvent.venue``
            and surfaced in ``AdapterStatus``).
        venue: Human-readable venue tag (e.g. ``"hummingbot:binance_paper"``).
    """

    name: str = "live_base"
    venue: str = ""

    def __init__(self, *, name: str, venue: str) -> None:
        if not name:
            raise ValueError("name required")
        if not venue:
            raise ValueError("venue required")
        self.name = name
        self.venue = venue
        self._state: AdapterState = AdapterState.DISCONNECTED
        self._detail: str = "scaffold — credentials not yet wired"
        self._last_heartbeat_ns: int = 0

    # ---- lifecycle -------------------------------------------------------

    def connect(self) -> None:
        """Establish the venue session.

        Default implementation is a no-op so unit tests don't need to
        stub a socket. Live subclasses override this and flip ``_state``
        to ``READY`` on success.
        """
        self._state = AdapterState.DISCONNECTED
        self._detail = "scaffold — connect() not implemented for live mode"

    def disconnect(self) -> None:
        self._state = AdapterState.DISCONNECTED
        self._detail = "operator disconnect"

    def halt(self, reason: str) -> None:
        self._state = AdapterState.HALTED
        self._detail = f"halted: {reason}"

    # ---- BrokerAdapter Protocol -----------------------------------------

    def submit(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        if self._state is not AdapterState.READY:
            return self._reject(signal, mark_price)
        return self._submit_live(signal, mark_price)

    def _submit_live(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        """Override in live subclasses.

        Default implementation rejects so a half-implemented subclass
        cannot accidentally execute.
        """
        return self._reject(signal, mark_price)

    # ---- introspection ---------------------------------------------------

    def status(self) -> AdapterStatus:
        return AdapterStatus(
            name=self.name,
            venue=self.venue,
            state=self._state,
            detail=self._detail,
            last_heartbeat_ns=self._last_heartbeat_ns,
        )

    # ---- helpers ---------------------------------------------------------

    def _reject(
        self,
        signal: SignalEvent,
        mark_price: float,
    ) -> ExecutionEvent:
        meta: Mapping[str, str] = {
            "reason": "adapter_not_ready",
            "adapter_state": self._state.value,
            "adapter_detail": self._detail,
        }
        return ExecutionEvent(
            ts_ns=signal.ts_ns,
            symbol=signal.symbol,
            side=signal.side,
            qty=0.0,
            price=mark_price,
            status=ExecutionStatus.REJECTED,
            venue=self.venue,
            order_id="",
            meta=meta,
            produced_by_engine="execution_engine",
        )


__all__ = ["AdapterState", "AdapterStatus", "LiveAdapterBase"]
