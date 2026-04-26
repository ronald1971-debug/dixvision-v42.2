"""IntelligenceEngine — RUNTIME-ENGINE-01 (Phase E2).

Phase E2 wires the first concrete intelligence plugin (IND-L02 market
microstructure) under the ``microstructure`` slot. The engine still
satisfies :class:`RuntimeEngine`:

* :meth:`process` is bus-side. It is a pure passthrough for
  ``SignalEvent``s today (Phase E0 behaviour preserved); other event
  kinds are silently ignored at the contract layer.
* :meth:`on_market` is the input-side. ``MarketTick`` is **not** a
  canonical bus event (INV-08); it flows from a data feed into the
  engine, drives the active microstructure plugins, and the engine
  collects their :class:`SignalEvent` outputs.

A plugin in :attr:`PluginLifecycle.SHADOW` has its emitted signals
tagged ``meta["shadow"] = "true"`` so the Execution Engine rejects them
without contacting any broker (Phase E2 exit: "Shadow mode wired (no
live trades)").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    MicrostructurePlugin,
    Plugin,
    PluginLifecycle,
    RuntimeEngine,
)
from core.contracts.events import Event, SignalEvent
from core.contracts.market import MarketTick


class IntelligenceEngine(RuntimeEngine):
    name: str = "intelligence"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        microstructure_plugins: Sequence[MicrostructurePlugin] | None = None,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
    ) -> None:
        self._microstructure: tuple[MicrostructurePlugin, ...] = tuple(
            microstructure_plugins or ()
        )
        slots: dict[str, Sequence[object]] = dict(plugin_slots or {})
        # Surface the typed microstructure plugins under the same slot
        # exposed in registry/plugins.yaml so check_self() reports them.
        slots["microstructure"] = self._microstructure
        self.plugin_slots = slots  # type: ignore[assignment]

    @property
    def microstructure_plugins(self) -> tuple[MicrostructurePlugin, ...]:
        return self._microstructure

    def on_market(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        """Run all enabled microstructure plugins against ``tick``.

        Returns the concatenated, in-order tuple of emitted signals.
        SHADOW signals are tagged with ``meta["shadow"] = "true"`` so the
        Execution Engine refuses to fill them.
        """
        out: list[SignalEvent] = []
        for plugin in self._microstructure:
            if plugin.lifecycle is PluginLifecycle.DISABLED:
                continue
            for sig in plugin.on_tick(tick):
                if plugin.lifecycle is PluginLifecycle.SHADOW:
                    meta = dict(sig.meta)
                    meta["shadow"] = "true"
                    sig = SignalEvent(
                        ts_ns=sig.ts_ns,
                        symbol=sig.symbol,
                        side=sig.side,
                        confidence=sig.confidence,
                        plugin_chain=sig.plugin_chain,
                        meta=meta,
                    )
                out.append(sig)
        return tuple(out)

    def process(self, event: Event) -> Sequence[Event]:
        # Bus-side passthrough; SignalEvents flow on the canonical bus.
        if isinstance(event, SignalEvent):
            return (event,)
        return ()

    def check_self(self) -> HealthStatus:
        plugin_states: dict[str, dict[str, HealthState]] = {}
        for slot, plugins in self.plugin_slots.items():
            slot_states: dict[str, HealthState] = {}
            for p in plugins:
                try:
                    slot_states[p.name] = p.check_self().state
                except Exception:  # pragma: no cover - defensive
                    slot_states[p.name] = HealthState.FAIL
            plugin_states[slot] = slot_states

        if not self._microstructure:
            detail = "Phase E2 — no microstructure plugins loaded"
        else:
            modes = ",".join(
                f"{p.name}:{p.lifecycle}" for p in self._microstructure
            )
            detail = f"Phase E2 — microstructure=[{modes}]"

        return HealthStatus(
            state=HealthState.OK,
            detail=detail,
            plugin_states=plugin_states,
        )


__all__ = ["IntelligenceEngine"]
