"""ExecutionEngine — RUNTIME-ENGINE-02 (Phase E1).

Phase E1 wires a :class:`BrokerAdapter` (default :class:`PaperBroker`)
behind the canonical event bus. The engine consumes ``SignalEvent`` and
emits exactly one ``ExecutionEvent`` per consumed signal.

Determinism contract (INV-15 / TEST-01):

* :meth:`process` is a pure function of (event, internal mark cache,
  adapter state). No clocks, no randomness, no external IO.
* :meth:`on_market` updates the internal mark cache with the last trade
  price; market ticks are *inputs*, not bus events (see
  ``core/contracts/market.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    Plugin,
    RuntimeEngine,
)
from core.contracts.events import (
    Event,
    ExecutionEvent,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from core.contracts.market import MarketTick
from execution_engine.adapters import BrokerAdapter, PaperBroker


class ExecutionEngine(RuntimeEngine):
    name: str = "execution"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        adapter: BrokerAdapter | None = None,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
    ) -> None:
        self._adapter: BrokerAdapter = adapter or PaperBroker()
        self._marks: dict[str, float] = {}
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )

    @property
    def adapter(self) -> BrokerAdapter:
        return self._adapter

    def on_market(self, tick: MarketTick) -> None:
        """Update the internal mark cache.

        Market ticks are inputs, not bus events. This keeps the canonical
        bus restricted to the four typed events (INV-08).
        """
        if tick.last > 0.0:
            self._marks[tick.symbol] = tick.last

    def process(self, event: Event) -> Sequence[Event]:
        if not isinstance(event, SignalEvent):
            return ()

        mark = self._marks.get(event.symbol, 0.0)
        if mark <= 0.0:
            return (
                ExecutionEvent(
                    ts_ns=event.ts_ns,
                    symbol=event.symbol,
                    side=event.side,
                    qty=0.0,
                    price=0.0,
                    status=ExecutionStatus.FAILED,
                    venue=self._adapter.name,
                    order_id="",
                    meta={"reason": "no mark for symbol"},
                ),
            )

        if event.side is Side.HOLD:
            return (
                self._adapter.submit(event, mark),
            )

        return (self._adapter.submit(event, mark),)

    def check_self(self) -> HealthStatus:
        plugin_states = {
            slot: {p.name: HealthState.OK for p in plugins}
            for slot, plugins in self.plugin_slots.items()
        }
        return HealthStatus(
            state=HealthState.OK,
            detail=f"Phase E1 — adapter={self._adapter.name}",
            plugin_states=plugin_states,
        )


__all__ = ["ExecutionEngine"]
