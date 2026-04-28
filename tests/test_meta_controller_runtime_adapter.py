"""Phase 6.T1c — Meta-Controller runtime adapter tests.

Covers:

* Pure wrapper around :func:`run_meta_controller_tick`.
* Ledger contains BELIEF_STATE_SNAPSHOT, PRESSURE_VECTOR_SNAPSHOT,
  META_AUDIT (always) and META_DIVERGENCE (only when primary !=
  shadow).
* Ledger order is fixed (snapshot pair → audit → optional divergence).
* INV-48 fallback emits the same ledger shape (audit + snapshots) so
  calibration is not blinded under degrade.
* J3 audit payload carries confidence + sizing components and the
  decision summary.
* INV-15 replay determinism.
"""

from __future__ import annotations

import pytest

from core.coherence.belief_state import BeliefState, Regime
from core.coherence.performance_pressure import PressureVector
from core.contracts.events import Side, SignalEvent, SystemEventKind
from intelligence_engine.meta_controller import (
    MetaControllerConfig,
    initial_meta_controller_state,
    load_meta_controller_config,
    step_meta_controller_hot_path,
)
from intelligence_engine.meta_controller.allocation import PositionSizerConfig
from intelligence_engine.meta_controller.evaluation import ConfidenceEngineConfig
from intelligence_engine.meta_controller.perception.regime_router import (
    RegimeRouterConfig,
)
from intelligence_engine.meta_controller.policy import FALLBACK_POLICY
from intelligence_engine.meta_controller.runtime_adapter import (
    RUNTIME_ADAPTER_SOURCE,
    build_meta_audit_event,
)

# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_meta_controller_orchestrator.py shapes)
# ---------------------------------------------------------------------------


def _signal(side: Side, confidence: float = 0.8, ts_ns: int = 1) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="X",
        side=side,
        confidence=confidence,
    )


def _belief(
    *,
    regime: Regime = Regime.TREND_UP,
    regime_confidence: float = 0.9,
    ts_ns: int = 100,
) -> BeliefState:
    return BeliefState(
        ts_ns=ts_ns,
        regime=regime,
        regime_confidence=regime_confidence,
        consensus_side=Side.BUY,
        signal_count=4,
        avg_confidence=0.8,
        symbols=("X",),
    )


def _pressure(
    *,
    risk: float = 0.0,
    safety_modifier: float = 1.0,
    uncertainty: float = 0.1,
    ts_ns: int = 100,
) -> PressureVector:
    return PressureVector(
        ts_ns=ts_ns,
        perf=0.0,
        risk=risk,
        drift=0.0,
        latency=0.0,
        uncertainty=uncertainty,
        safety_modifier=safety_modifier,
        cross_signal_entropy=0.0,
        signal_count=4,
    )


def _config(
    *,
    persistence_ticks: int = 4,
    confidence_delta_threshold: float = 0.4,
    latency_budget_ns: int = 500_000,
) -> MetaControllerConfig:
    return MetaControllerConfig(
        router_config=RegimeRouterConfig(
            persistence_ticks=persistence_ticks,
            confidence_delta_threshold=confidence_delta_threshold,
        ),
        confidence_config=ConfidenceEngineConfig(
            consensus_weight=0.5,
            strength_weight=0.3,
            coverage_weight=0.2,
            saturation_count=8,
        ),
        sizer_config=PositionSizerConfig(
            base_fraction=1.0,
            kelly_cap=0.25,
            trend_multiplier=1.0,
            range_multiplier=0.5,
            vol_spike_multiplier=0.0,
            confidence_floor=0.2,
            risk_damping=0.5,
        ),
        latency_budget_ns=latency_budget_ns,
    )


# ---------------------------------------------------------------------------
# Step semantics
# ---------------------------------------------------------------------------


def test_step_returns_output_and_ledger_tuple() -> None:
    output, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=100,
    )
    assert output.elapsed_ns == 1_000
    assert isinstance(ledger, tuple)
    assert all(ev.ts_ns == 100 for ev in ledger)


def test_ledger_always_contains_three_baseline_events_in_order() -> None:
    """Snapshot pair + audit, in that exact order, every tick."""
    _, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=100,
    )
    assert ledger[0].sub_kind is SystemEventKind.BELIEF_STATE_SNAPSHOT
    assert ledger[1].sub_kind is SystemEventKind.PRESSURE_VECTOR_SNAPSHOT
    assert ledger[2].sub_kind is SystemEventKind.META_AUDIT


def test_ledger_appends_divergence_when_primary_disagrees_with_shadow() -> None:
    """safety_modifier=0.5 damps primary but not shadow → divergence."""
    _, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(safety_modifier=0.5),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=100,
    )
    assert len(ledger) == 4
    assert ledger[3].sub_kind is SystemEventKind.META_DIVERGENCE


def test_ledger_omits_divergence_when_primary_agrees_with_shadow() -> None:
    """safety_modifier=1.0 → primary == shadow → no divergence event."""
    _, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(safety_modifier=1.0),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=100,
    )
    kinds = {ev.sub_kind for ev in ledger}
    assert SystemEventKind.META_DIVERGENCE not in kinds
    assert len(ledger) == 3


def test_state_advances_through_step() -> None:
    state0 = initial_meta_controller_state()
    output, _ = step_meta_controller_hot_path(
        state=state0,
        signals=[_signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=100,
    )
    # A first tick commits the initial regime; router state changes.
    assert output.state is not state0
    assert output.state.router_state.current_regime is Regime.TREND_UP


# ---------------------------------------------------------------------------
# INV-48 latency fallback
# ---------------------------------------------------------------------------


def test_latency_fallback_emits_full_ledger() -> None:
    """Even when primary is FALLBACK_POLICY, calibration is not
    blinded — the ledger still carries the snapshot pair + audit."""
    output, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(),
        config=_config(latency_budget_ns=10),
        elapsed_ns=1_000_000,
        ts_ns=100,
    )
    assert output.primary_decision is FALLBACK_POLICY
    assert ledger[0].sub_kind is SystemEventKind.BELIEF_STATE_SNAPSHOT
    assert ledger[1].sub_kind is SystemEventKind.PRESSURE_VECTOR_SNAPSHOT
    assert ledger[2].sub_kind is SystemEventKind.META_AUDIT
    audit_payload = ledger[2].payload
    assert audit_payload["decision_fallback"] == "true"


# ---------------------------------------------------------------------------
# Audit payload (J3)
# ---------------------------------------------------------------------------


def test_meta_audit_payload_carries_j3_components() -> None:
    output, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=100,
    )
    audit = ledger[2]
    assert audit.sub_kind is SystemEventKind.META_AUDIT
    assert audit.source == RUNTIME_ADAPTER_SOURCE
    p = audit.payload
    # Confidence J3 components present.
    assert "confidence_consensus" in p
    assert "confidence_strength" in p
    assert "confidence_coverage" in p
    assert "confidence_composite" in p
    # Sizing J3 components present.
    assert "sizing_confidence_factor" in p
    assert "sizing_regime_factor" in p
    assert "sizing_risk_factor" in p
    assert "sizing_pre_cap" in p
    assert "sizing_final" in p
    assert "sizing_rationale" in p
    # Decision summary present.
    assert p["decision_side"] == output.primary_decision.side.value
    assert p["decision_size"] == f"{output.primary_decision.size_fraction:.6f}"
    assert p["proposed_side"] == output.proposed_side.value
    assert p["regime"] == output.state.router_state.current_regime.value
    assert p["regime_transitioned"] == "true"  # first tick commits
    assert p["elapsed_ns"] == "1000"


def test_meta_audit_payload_is_string_only() -> None:
    """SystemEvent payloads must be Mapping[str, str]."""
    _, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=100,
    )
    for v in ledger[2].payload.values():
        assert isinstance(v, str)


def test_build_meta_audit_event_helper_matches_step() -> None:
    """``build_meta_audit_event`` is the projection function. The
    adapter MUST emit exactly that record at index 2."""
    from intelligence_engine.meta_controller.orchestrator import (
        run_meta_controller_tick,
    )

    state = initial_meta_controller_state()
    signals = [_signal(Side.BUY), _signal(Side.BUY)]
    belief = _belief()
    pressure = _pressure()
    config = _config()
    output_pure = run_meta_controller_tick(
        state=state,
        signals=signals,
        belief=belief,
        pressure=pressure,
        config=config,
        elapsed_ns=1_000,
        ts_ns=100,
    )
    expected_audit = build_meta_audit_event(ts_ns=100, output=output_pure)
    _, ledger = step_meta_controller_hot_path(
        state=state,
        signals=signals,
        belief=belief,
        pressure=pressure,
        config=config,
        elapsed_ns=1_000,
        ts_ns=100,
    )
    assert ledger[2] == expected_audit


# ---------------------------------------------------------------------------
# INV-15 replay determinism
# ---------------------------------------------------------------------------


def test_step_is_replay_deterministic() -> None:
    """Same inputs → same output + same ledger across runs."""
    runs = []
    for _ in range(10):
        out, ledger = step_meta_controller_hot_path(
            state=initial_meta_controller_state(),
            signals=[_signal(Side.BUY), _signal(Side.SELL)],
            belief=_belief(),
            pressure=_pressure(safety_modifier=0.5),
            config=_config(),
            elapsed_ns=1_000,
            ts_ns=100,
        )
        runs.append((out, ledger))
    first_out, first_ledger = runs[0]
    for out, ledger in runs[1:]:
        assert out == first_out
        assert ledger == first_ledger


# ---------------------------------------------------------------------------
# Config loader (registry round-trip)
# ---------------------------------------------------------------------------


def test_load_meta_controller_config_round_trip() -> None:
    cfg = load_meta_controller_config(
        regime_path="registry/regime.yaml",
        confidence_path="registry/confidence.yaml",
        sizer_path="registry/position_sizer.yaml",
        latency_budget_ns=750_000,
    )
    assert cfg.latency_budget_ns == 750_000
    # Sub-config validation: each loader returned a non-zero record.
    assert cfg.router_config.persistence_ticks > 0
    assert cfg.confidence_config.saturation_count > 0
    assert cfg.sizer_config.kelly_cap > 0


def test_load_meta_controller_config_default_latency_budget() -> None:
    """Default latency budget is the manifest's 500 µs T1 ceiling."""
    cfg = load_meta_controller_config()
    assert cfg.latency_budget_ns == 500_000


def test_load_meta_controller_config_rejects_zero_budget() -> None:
    with pytest.raises(ValueError, match="latency_budget_ns"):
        load_meta_controller_config(latency_budget_ns=0)


def test_loaded_config_drives_step() -> None:
    """End-to-end: registry → loader → step_meta_controller_hot_path."""
    cfg = load_meta_controller_config(latency_budget_ns=500_000)
    output, ledger = step_meta_controller_hot_path(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        belief=_belief(),
        pressure=_pressure(),
        config=cfg,
        elapsed_ns=1_000,
        ts_ns=100,
    )
    assert ledger[0].sub_kind is SystemEventKind.BELIEF_STATE_SNAPSHOT
    assert ledger[2].sub_kind is SystemEventKind.META_AUDIT
    assert output.elapsed_ns == 1_000
