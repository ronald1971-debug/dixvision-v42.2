"""IntelligenceEngine — RUNTIME-ENGINE-01 (Phase E2 + Wave 1 wiring).

Phase E2 wired the first concrete intelligence plugin (IND-L02 market
microstructure) under the ``microstructure`` slot. Wave 1 adds the
optional :class:`MetaControllerHotPath` integration so that a single
:meth:`run_meta_tick` call:

1. drives all enabled microstructure plugins from a :class:`MarketTick`,
2. appends the emitted signals to a bounded rolling window owned by
   the engine,
3. invokes :meth:`MetaControllerHotPath.step` with the rolling window
   plus a caller-supplied :class:`RuntimeContext` (perf / risk /
   drift / latency / ``vol_spike_z`` / ``elapsed_ns``),
4. returns ``(signals, decision, ledger)``.

The engine still satisfies :class:`RuntimeEngine`:

* :meth:`process` is bus-side. It is a pure passthrough for
  ``SignalEvent``s today (Phase E0 behaviour preserved); other event
  kinds are silently ignored at the contract layer.
* :meth:`on_market` is the input-side. ``MarketTick`` is **not** a
  canonical bus event (INV-08); it flows from a data feed into the
  engine, drives the active microstructure plugins, and the engine
  collects their :class:`SignalEvent` outputs.
* :meth:`run_meta_tick` is opt-in — it requires a
  :class:`MetaControllerHotPath` to have been passed at construction.

Plugin-level SHADOW was demolished by SHADOW-DEMOLITION-01: a plugin
is either ``DISABLED`` (skipped) or ``ACTIVE`` (its signals flow
into the conflict resolver). Signals-on/execution-off behaviour now
lives at the system-mode layer only.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    MicrostructurePlugin,
    Plugin,
    PluginLifecycle,
    RuntimeEngine,
)
from core.contracts.events import Event, SignalEvent, SystemEvent
from core.contracts.market import MarketTick
from intelligence_engine.meta_controller import MetaControllerHotPath
from intelligence_engine.meta_controller.policy import ExecutionDecision
from intelligence_engine.runtime_context import RuntimeContext

DEFAULT_SIGNAL_WINDOW_SIZE = 32


class IntelligenceEngine(RuntimeEngine):
    name: str = "intelligence"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        microstructure_plugins: Sequence[MicrostructurePlugin] | None = None,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        *,
        meta_controller_hot_path: MetaControllerHotPath | None = None,
        signal_window_size: int = DEFAULT_SIGNAL_WINDOW_SIZE,
    ) -> None:
        if signal_window_size <= 0:
            raise ValueError("signal_window_size must be > 0")
        self._microstructure: tuple[MicrostructurePlugin, ...] = tuple(
            microstructure_plugins or ()
        )
        slots: dict[str, Sequence[object]] = dict(plugin_slots or {})
        # Surface the typed microstructure plugins under the same slot
        # exposed in registry/plugins.yaml so check_self() reports them.
        slots["microstructure"] = self._microstructure
        self.plugin_slots = slots  # type: ignore[assignment]

        self._meta_controller_hot_path = meta_controller_hot_path
        self._signal_window: deque[SignalEvent] = deque(
            maxlen=signal_window_size
        )

    @property
    def microstructure_plugins(self) -> tuple[MicrostructurePlugin, ...]:
        return self._microstructure

    @property
    def meta_controller_hot_path(self) -> MetaControllerHotPath | None:
        return self._meta_controller_hot_path

    @property
    def signal_window(self) -> tuple[SignalEvent, ...]:
        """Snapshot of the rolling signal window. Read-only."""
        return tuple(self._signal_window)

    def on_market(self, tick: MarketTick) -> tuple[SignalEvent, ...]:
        """Run all enabled microstructure plugins against ``tick``.

        Returns the concatenated, in-order tuple of emitted signals.

        Wave 1: the engine also appends the emitted signals to its
        rolling window so a subsequent :meth:`run_meta_tick` sees a
        coherent recent-signal context.
        """
        out: list[SignalEvent] = []
        for plugin in self._microstructure:
            if plugin.lifecycle is PluginLifecycle.DISABLED:
                continue
            for sig in plugin.on_tick(tick):
                out.append(sig)
        emitted = tuple(out)
        for sig in emitted:
            self._signal_window.append(sig)
        return emitted

    def run_meta_tick(
        self,
        *,
        tick: MarketTick,
        context: RuntimeContext,
        extra_signals: Iterable[SignalEvent] = (),
    ) -> tuple[
        tuple[SignalEvent, ...],
        ExecutionDecision,
        tuple[SystemEvent, ...],
    ]:
        """Drive plugins, advance the meta-controller, and return the
        full per-tick triple ``(signals, decision, ledger)``.

        The engine itself does not consult any clock; ``elapsed_ns``
        and ``tick.ts_ns`` are caller-supplied so replay determinism
        (INV-15) is preserved.

        Args:
            tick: The :class:`MarketTick` to drive plugins from.
            context: Per-tick runtime scalars
                (:class:`RuntimeContext`).
            extra_signals: Optional additional signals from non-
                microstructure intelligence plugins (e.g. plugins owned
                by a higher-level orchestrator) to be appended to the
                rolling window before the meta-controller step. They
                are returned as part of the emitted signals tuple as
                well, after the microstructure-emitted ones.

        Returns:
            ``(signals, decision, ledger)`` where:

            * ``signals`` are the freshly emitted signals (microstructure
              + ``extra_signals``) in deterministic order.
            * ``decision`` is the primary :class:`ExecutionDecision` from
              the meta-controller.
            * ``ledger`` is the four-event :class:`SystemEvent` ledger
              (BELIEF_STATE_SNAPSHOT → PRESSURE_VECTOR_SNAPSHOT →
              META_AUDIT → optional META_DIVERGENCE).

        Raises:
            RuntimeError: if no :class:`MetaControllerHotPath` was
                passed at construction.
        """
        hot = self._meta_controller_hot_path
        if hot is None:
            raise RuntimeError(
                "IntelligenceEngine.run_meta_tick requires "
                "meta_controller_hot_path to be configured at "
                "construction time."
            )

        emitted = self.on_market(tick)
        extras = tuple(extra_signals)
        for sig in extras:
            self._signal_window.append(sig)

        decision, ledger = hot.step(
            ts_ns=tick.ts_ns,
            signals=tuple(self._signal_window),
            perf=context.perf,
            risk=context.risk,
            drift=context.drift,
            latency=context.latency,
            vol_spike_z=context.vol_spike_z,
            elapsed_ns=context.elapsed_ns,
        )
        return emitted + extras, decision, ledger

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

        if self._meta_controller_hot_path is not None:
            detail = f"{detail} meta_controller=wired"

        return HealthStatus(
            state=HealthState.OK,
            detail=detail,
            plugin_states=plugin_states,
        )


__all__ = ["DEFAULT_SIGNAL_WINDOW_SIZE", "IntelligenceEngine"]
