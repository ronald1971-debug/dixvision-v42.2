"""Meta-Controller hot-path harness — Phase 6.T1c (engine wiring).

Stateful runtime wrapper around the pure
:func:`step_meta_controller_hot_path` adapter. The harness owns:

* the rolling :class:`MetaControllerState` (router state, version
  pin),
* the :class:`MetaControllerConfig` bundle (regime / confidence /
  sizer + latency budget),
* the :class:`PressureConfig` (entropy α / β + high-water modifier).

A single ``.step(...)`` call:

1. Derives a fresh :class:`BeliefState` from the signal window
   (:func:`derive_belief_state`).
2. Derives a fresh :class:`PressureVector` (:func:`derive_pressure_vector`)
   from the signal window plus the four caller-supplied scalars
   (``perf`` / ``risk`` / ``drift`` / ``latency``) sourced from the
   appropriate runtime monitors.
3. Calls :func:`step_meta_controller_hot_path` to run the orchestrator
   and build the four-event ledger.
4. Advances the harness's internal ``MetaControllerState``.
5. Returns the :class:`ExecutionDecision` (the value the
   PolicyEngine will validate) plus the ledger tuple.

Authority constraints:

* B1 — depends only on :mod:`core.contracts`, :mod:`core.coherence`,
  and the meta-controller package.
* INV-15 — pure step semantics; identical inputs ⇒ identical
  outputs / ledgers / state transitions.
* INV-48 — fallback path is owned by the inner adapter; the harness
  emits the same four-event ledger under degrade.
* INV-52 — the ``META_DIVERGENCE`` event is emitted only by the inner
  adapter when primary != shadow; the harness adds nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.coherence.belief_state import BeliefState, derive_belief_state
from core.coherence.performance_pressure import (
    PressureConfig,
    PressureVector,
    derive_pressure_vector,
)
from core.contracts.events import SignalEvent, SystemEvent
from intelligence_engine.meta_controller.orchestrator import (
    MetaControllerConfig,
    MetaControllerOutput,
    MetaControllerState,
    initial_meta_controller_state,
)
from intelligence_engine.meta_controller.policy import ExecutionDecision
from intelligence_engine.meta_controller.runtime_adapter import (
    step_meta_controller_hot_path,
)


class MetaControllerHotPath:
    """Stateful wrapper around the pure runtime adapter.

    One instance per engine. Per-tick, callers feed in the signal
    window (built upstream by the intelligence pipeline) plus the
    four pressure scalars from the runtime monitors, and receive
    the :class:`ExecutionDecision` and the audit ledger that should
    be appended to the bus / SystemEvent log.
    """

    __slots__ = (
        "_meta_config",
        "_pressure_config",
        "_state",
    )

    def __init__(
        self,
        *,
        meta_config: MetaControllerConfig,
        pressure_config: PressureConfig,
        initial_state: MetaControllerState | None = None,
    ) -> None:
        self._meta_config = meta_config
        self._pressure_config = pressure_config
        self._state = initial_state or initial_meta_controller_state()

    # ------------------------------------------------------------------
    # Read-only introspection
    # ------------------------------------------------------------------

    @property
    def state(self) -> MetaControllerState:
        return self._state

    @property
    def meta_config(self) -> MetaControllerConfig:
        return self._meta_config

    @property
    def pressure_config(self) -> PressureConfig:
        return self._pressure_config

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    def derive_inputs(
        self,
        *,
        ts_ns: int,
        signals: Sequence[SignalEvent],
        perf: float,
        risk: float,
        drift: float,
        latency: float,
        vol_spike_z: float,
    ) -> tuple[BeliefState, PressureVector]:
        """Compute belief + pressure for the current tick.

        Pure — exposed for tests + dashboards that want the projected
        inputs without driving a full meta-controller step.
        """
        belief = derive_belief_state(
            ts_ns=ts_ns,
            signals=signals,
            vol_spike_z=vol_spike_z,
        )
        pressure = derive_pressure_vector(
            ts_ns=ts_ns,
            signals=signals,
            perf=perf,
            risk=risk,
            drift=drift,
            latency=latency,
            config=self._pressure_config,
        )
        return belief, pressure

    def step(
        self,
        *,
        ts_ns: int,
        signals: Sequence[SignalEvent],
        perf: float,
        risk: float,
        drift: float,
        latency: float,
        vol_spike_z: float,
        elapsed_ns: int,
    ) -> tuple[ExecutionDecision, tuple[SystemEvent, ...]]:
        """Run one composed tick.

        Returns ``(primary_decision, ledger)``. The four-event ledger
        order is fixed by the inner adapter
        (BELIEF_STATE_SNAPSHOT → PRESSURE_VECTOR_SNAPSHOT →
        META_AUDIT → optional META_DIVERGENCE).

        State advances via the inner adapter's pure transition;
        callers can observe the new state via :attr:`state` after
        the call.
        """
        belief, pressure = self.derive_inputs(
            ts_ns=ts_ns,
            signals=signals,
            perf=perf,
            risk=risk,
            drift=drift,
            latency=latency,
            vol_spike_z=vol_spike_z,
        )
        output, ledger = step_meta_controller_hot_path(
            state=self._state,
            signals=signals,
            belief=belief,
            pressure=pressure,
            config=self._meta_config,
            elapsed_ns=elapsed_ns,
            ts_ns=ts_ns,
        )
        self._state = output.state
        return output.primary_decision, ledger

    def step_full(
        self,
        *,
        ts_ns: int,
        signals: Sequence[SignalEvent],
        perf: float,
        risk: float,
        drift: float,
        latency: float,
        vol_spike_z: float,
        elapsed_ns: int,
    ) -> tuple[MetaControllerOutput, tuple[SystemEvent, ...]]:
        """Like :meth:`step` but returns the full
        :class:`MetaControllerOutput` (J3 components + shadow + …)
        rather than just the primary decision. Used by tests and
        the calibrator harness."""
        belief, pressure = self.derive_inputs(
            ts_ns=ts_ns,
            signals=signals,
            perf=perf,
            risk=risk,
            drift=drift,
            latency=latency,
            vol_spike_z=vol_spike_z,
        )
        output, ledger = step_meta_controller_hot_path(
            state=self._state,
            signals=signals,
            belief=belief,
            pressure=pressure,
            config=self._meta_config,
            elapsed_ns=elapsed_ns,
            ts_ns=ts_ns,
        )
        self._state = output.state
        return output, ledger


__all__ = ["MetaControllerHotPath"]
