"""Meta-Controller runtime adapter — Phase 6.T1c.

Pure wrapper around :func:`run_meta_controller_tick` that builds the
per-tick **ledger of SystemEvents** the orchestrator does not own
directly:

* ``BELIEF_STATE_SNAPSHOT`` — INV-53 calibration hook.
* ``PRESSURE_VECTOR_SNAPSHOT`` — INV-53 calibration hook.
* ``META_AUDIT`` — J3 per-tick audit (confidence / sizing components
  + final decision summary). The offline calibrator reads this to
  attribute drift to specific components.
* ``META_DIVERGENCE`` — INV-52 shadow path; passed through unchanged
  from the orchestrator (only present when primary != shadow).

The adapter is **pure**: callers pass ``elapsed_ns`` (typically
``time_authority.now_ns() - tick_start_ns``) and ``ts_ns`` in. The
orchestrator does not touch the system clock; the adapter does not
either.

Authority constraints:

* B1 — no cross-runtime-engine imports. The adapter only depends on
  :mod:`core.contracts`, :mod:`core.coherence`, and the
  meta-controller package itself.
* B17 — shadow path remains divergence-only (passthrough; we do not
  modify the orchestrator's shadow output).
* INV-15 — deterministic; same inputs produce identical outputs and
  identical ledgers across replays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.coherence.belief_state import BeliefState
from core.coherence.performance_pressure import PressureVector
from core.contracts.events import (
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from intelligence_engine.meta_controller.orchestrator import (
    META_CONTROLLER_VERSION,
    MetaControllerConfig,
    MetaControllerOutput,
    MetaControllerState,
    run_meta_controller_tick,
)

RUNTIME_ADAPTER_SOURCE = "intelligence_engine.meta_controller.runtime_adapter"


def build_meta_audit_event(
    *,
    ts_ns: int,
    output: MetaControllerOutput,
    source: str = RUNTIME_ADAPTER_SOURCE,
) -> SystemEvent:
    """Project a :class:`MetaControllerOutput` into a J3 audit event.

    Captures, in a single ledgerable record:

    * The proposed side resolved from the signal window.
    * The composite confidence + per-component breakdown.
    * The composite size + per-component breakdown + rationale.
    * The committed regime (post-hysteresis) and whether it just
      transitioned.
    * The primary decision summary.
    * The elapsed_ns the orchestrator observed for this tick (so the
      calibrator can correlate latency with INV-48 fallback usage).
    """
    confidence = output.confidence_components
    sizing = output.sizing_components
    primary = output.primary_decision
    committed_regime = output.state.router_state.current_regime
    payload: Mapping[str, str] = {
        "version": output.version,
        "proposed_side": output.proposed_side.value,
        "regime": committed_regime.value,
        "regime_transitioned": "true" if output.regime_transitioned else "false",
        "elapsed_ns": str(output.elapsed_ns),
        # Confidence J3 components.
        "confidence_consensus": f"{confidence.consensus:.6f}",
        "confidence_strength": f"{confidence.strength:.6f}",
        "confidence_coverage": f"{confidence.coverage:.6f}",
        "confidence_composite": f"{confidence.composite:.6f}",
        "confidence_signal_count": str(confidence.signal_count),
        # Sizing J3 components.
        "sizing_confidence_factor": f"{sizing.confidence_factor:.6f}",
        "sizing_regime_factor": f"{sizing.regime_factor:.6f}",
        "sizing_risk_factor": f"{sizing.risk_factor:.6f}",
        "sizing_pre_cap": f"{sizing.pre_cap_size:.6f}",
        "sizing_final": f"{sizing.final_size:.6f}",
        "sizing_rationale": sizing.rationale,
        # Decision summary.
        "decision_side": primary.side.value,
        "decision_size": f"{primary.size_fraction:.6f}",
        "decision_confidence": f"{primary.confidence:.6f}",
        "decision_fallback": "true" if primary.fallback else "false",
    }
    return SystemEvent(
        ts_ns=ts_ns,
        sub_kind=SystemEventKind.META_AUDIT,
        source=source,
        payload=payload,
    )


def step_meta_controller_hot_path(
    *,
    state: MetaControllerState,
    signals: Sequence[SignalEvent],
    belief: BeliefState,
    pressure: PressureVector,
    config: MetaControllerConfig,
    elapsed_ns: int,
    ts_ns: int,
) -> tuple[MetaControllerOutput, tuple[SystemEvent, ...]]:
    """Run one orchestrator tick + emit the calibration / audit ledger.

    Returns a tuple ``(output, ledger)``:

    * ``output``: the :class:`MetaControllerOutput` from the
      orchestrator. Callers update their stored
      :class:`MetaControllerState` via ``output.state``.
    * ``ledger``: an in-order tuple of :class:`SystemEvent`s the
      caller must publish to the bus / append to the ledger. The
      ledger always contains, in this order:

        1. ``BELIEF_STATE_SNAPSHOT`` (INV-53)
        2. ``PRESSURE_VECTOR_SNAPSHOT`` (INV-53)
        3. ``META_AUDIT`` (J3 per-tick audit)
        4. ``META_DIVERGENCE`` (INV-52, only when primary != shadow)

    The function is pure: same ``(state, signals, belief, pressure,
    config, elapsed_ns, ts_ns)`` ⇒ same ``(output, ledger)``. INV-15.

    The function is also O(1) in latency-fallback mode: the
    orchestrator returns the constant ``FALLBACK_POLICY`` and the
    adapter still emits the four-event ledger so calibration / audit
    are unaffected by the fast-path degrade (INV-48).
    """
    output = run_meta_controller_tick(
        state=state,
        signals=signals,
        belief=belief,
        pressure=pressure,
        config=config,
        elapsed_ns=elapsed_ns,
        ts_ns=ts_ns,
    )
    ledger: list[SystemEvent] = [
        belief.to_event(),
        pressure.to_event(),
        build_meta_audit_event(ts_ns=ts_ns, output=output),
    ]
    if output.divergence_event is not None:
        ledger.append(output.divergence_event)
    return output, tuple(ledger)


__all__ = [
    "META_CONTROLLER_VERSION",
    "RUNTIME_ADAPTER_SOURCE",
    "build_meta_audit_event",
    "step_meta_controller_hot_path",
]
