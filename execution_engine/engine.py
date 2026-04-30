"""ExecutionEngine — RUNTIME-ENGINE-02 (Phase E1, hardened in HARDEN-02).

The Execution Gate (INV-68) makes :meth:`ExecutionEngine.execute` the
*only* runtime path to a venue. The legacy :meth:`process` path that
accepted a bare :class:`SignalEvent` still works for backwards
compatibility (the operator UI fixture harness still uses it) but
emits a :class:`DeprecationWarning` so any production caller is
visible in CI / pytest output.

Determinism contract (INV-15 / TEST-01):

* :meth:`execute` and :meth:`process` are pure functions of (intent /
  event, internal mark cache, adapter state). No clocks, no
  randomness, no external IO.
* :meth:`on_market` updates the internal mark cache with the last
  trade price; market ticks are *inputs*, not bus events (see
  ``core/contracts/market.py``).
"""

from __future__ import annotations

import warnings
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
from core.contracts.execution_intent import ExecutionIntent
from core.contracts.market import MarketTick
from execution_engine.adapters import BrokerAdapter, PaperBroker
from execution_engine.execution_gate import AuthorityGuard


class ExecutionEngine(RuntimeEngine):
    name: str = "execution"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        adapter: BrokerAdapter | None = None,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        *,
        guard: AuthorityGuard | None = None,
    ) -> None:
        self._adapter: BrokerAdapter = adapter or PaperBroker()
        self._marks: dict[str, float] = {}
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )
        # The AuthorityGuard is constructed lazily so tests that only
        # exercise :meth:`process` (legacy path) don't need a matrix
        # YAML on disk. ``execute`` materialises the guard on first
        # use unless the caller injected one explicitly.
        self._guard: AuthorityGuard | None = guard

    @property
    def adapter(self) -> BrokerAdapter:
        return self._adapter

    @property
    def guard(self) -> AuthorityGuard:
        if self._guard is None:
            self._guard = AuthorityGuard()
        return self._guard

    def on_market(self, tick: MarketTick) -> None:
        """Update the internal mark cache.

        Market ticks are inputs, not bus events. This keeps the canonical
        bus restricted to the four typed events (INV-08).
        """
        if tick.last > 0.0:
            self._marks[tick.symbol] = tick.last

    # ------------------------------------------------------------------
    # HARDEN-02 chokepoint
    # ------------------------------------------------------------------

    def execute(
        self,
        intent: ExecutionIntent,
        *,
        caller: str = "execution_engine",
    ) -> Sequence[ExecutionEvent]:
        """The Execution Gate — single runtime path to a venue.

        The :class:`AuthorityGuard` validates the intent (governance
        approval, content-hash integrity, registered origin) before
        any adapter side effect. On guard failure the call raises
        :class:`UnauthorizedActorError` and a synthetic
        ``HAZ-AUTHORITY`` :class:`HazardEvent` is emitted via the
        guard's hazard sink (when configured).

        Args:
            intent: A frozen, governance-approved
                :class:`ExecutionIntent`.
            caller: Runtime label of the engine invoking the gate.
                Defaults to ``"execution_engine"`` — the only value
                accepted by the default matrix.

        Returns:
            A short, ordered sequence of :class:`ExecutionEvent`
            envelopes — typically one fill per signal, possibly more
            if the underlying adapter splits.
        """

        self.guard.assert_can_execute(intent, caller=caller, ts_ns=intent.ts_ns)
        return self._execute_signal(intent.signal)

    # ------------------------------------------------------------------
    # Legacy path (SignalEvent → ExecutionEvent), retained for backwards
    # compatibility with the UI fixture harness. Emits a runtime
    # DeprecationWarning so any production caller is loud in CI.
    # ------------------------------------------------------------------

    def process(self, event: Event) -> Sequence[Event]:
        if not isinstance(event, SignalEvent):
            return ()
        warnings.warn(
            "ExecutionEngine.process(SignalEvent) is deprecated; route "
            "trades through ExecutionEngine.execute(ExecutionIntent) "
            "(HARDEN-02 / INV-68). The legacy path remains for the "
            "operator UI fixture harness only.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._execute_signal(event)

    # ------------------------------------------------------------------
    # Shared signal → execution kernel. Both paths reuse this so the
    # only difference between execute() and process() is the gate.
    # ------------------------------------------------------------------

    def _execute_signal(self, event: SignalEvent) -> Sequence[ExecutionEvent]:
        # Phase E2: SHADOW signals from intelligence_engine are observed
        # but never reach a broker. Returning a REJECTED ExecutionEvent
        # keeps the audit trail visible without producing a live trade.
        if event.meta.get("shadow") == "true":
            return (
                ExecutionEvent(
                    ts_ns=event.ts_ns,
                    symbol=event.symbol,
                    side=event.side,
                    qty=0.0,
                    price=0.0,
                    status=ExecutionStatus.REJECTED,
                    venue=self._adapter.name,
                    order_id="",
                    meta={"reason": "shadow signal"},
                    produced_by_engine="execution_engine",
                ),
            )

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
                    produced_by_engine="execution_engine",
                ),
            )

        if event.side is Side.HOLD:
            return (self._adapter.submit(event, mark),)

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
