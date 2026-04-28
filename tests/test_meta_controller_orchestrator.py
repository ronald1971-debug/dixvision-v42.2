"""Phase 6.T1b — Meta-Controller orchestrator tests.

Covers:

* Wiring contract: regime_router output is what evaluation /
  allocation / policy see.
* INV-49 hysteresis is honoured (raw belief.regime is *not* used by
  downstream stages until the router commits the transition).
* INV-48 latency-budget fallback flows through the orchestrator.
* INV-52 shadow path runs unconditionally and surfaces a
  ``META_DIVERGENCE`` SystemEvent when primary != shadow.
* J3 audit components (confidence + sizing) reach the output.
* Replay-determinism (INV-15).
"""

from __future__ import annotations

import dataclasses

import pytest

from core.coherence.belief_state import BeliefState, Regime
from core.coherence.performance_pressure import PressureVector
from core.contracts.events import Side, SignalEvent, SystemEventKind
from intelligence_engine.meta_controller import (
    META_CONTROLLER_VERSION,
    MetaControllerConfig,
    MetaControllerOutput,
    initial_meta_controller_state,
    run_meta_controller_tick,
)
from intelligence_engine.meta_controller.allocation import PositionSizerConfig
from intelligence_engine.meta_controller.evaluation import ConfidenceEngineConfig
from intelligence_engine.meta_controller.perception.regime_router import (
    RegimeRouterConfig,
    initial_router_state,
)
from intelligence_engine.meta_controller.policy import FALLBACK_POLICY


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
) -> PressureVector:
    return PressureVector(
        ts_ns=100,
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
# Records — frozen + validation
# ---------------------------------------------------------------------------


def test_state_and_output_are_frozen() -> None:
    s = initial_meta_controller_state()
    assert dataclasses.is_dataclass(s)
    assert s.version == META_CONTROLLER_VERSION
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.router_state = initial_router_state()  # type: ignore[misc]


def test_config_rejects_zero_latency_budget() -> None:
    with pytest.raises(ValueError, match="latency_budget_ns"):
        MetaControllerConfig(
            router_config=RegimeRouterConfig(
                persistence_ticks=4,
                confidence_delta_threshold=0.4,
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
            latency_budget_ns=0,
        )


def test_run_rejects_negative_elapsed_ns() -> None:
    with pytest.raises(ValueError, match="elapsed_ns"):
        run_meta_controller_tick(
            state=initial_meta_controller_state(),
            signals=[_signal(Side.BUY)],
            belief=_belief(),
            pressure=_pressure(),
            config=_config(),
            elapsed_ns=-1,
            ts_ns=200,
        )


# ---------------------------------------------------------------------------
# Wiring — regime_router output flows into downstream stages
# ---------------------------------------------------------------------------


def test_first_tick_router_locks_in_belief_regime() -> None:
    """A fresh UNKNOWN router takes belief.regime on the first tick;
    downstream stages immediately see TREND_UP, not UNKNOWN."""
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(regime=Regime.TREND_UP, regime_confidence=0.95),
        pressure=_pressure(risk=0.0),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    assert out.state.router_state.current_regime is Regime.TREND_UP
    assert out.regime_transitioned is True
    # Sizer must have used TREND_UP, not UNKNOWN.
    assert out.sizing_components.regime_factor == pytest.approx(1.0)
    assert out.sizing_components.final_size > 0.0
    # Primary policy honours the consensus side.
    assert out.primary_decision.side is Side.BUY


def test_hysteresis_blocks_premature_transition() -> None:
    """Belief flips regime but the candidate hasn't persisted long
    enough; the orchestrator must keep the old regime and downstream
    sizer / policy must use the OLD regime."""
    state0 = initial_meta_controller_state()
    # Tick 1: lock in TREND_UP.
    out1 = run_meta_controller_tick(
        state=state0,
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(regime=Regime.TREND_UP, regime_confidence=0.95),
        pressure=_pressure(),
        config=_config(persistence_ticks=4, confidence_delta_threshold=0.5),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    # First tick: UNKNOWN → TREND_UP; delta 0.95 - 0.0 ≥ 0.5 fast-path.
    assert out1.state.router_state.current_regime is Regime.TREND_UP

    # Tick 2: belief flips to RANGE with mid confidence; Δ-confidence
    # (0.6 - 0.95 = -0.35) is below the 0.5 fast-path threshold and
    # persistence is only 1 < 4. The router must NOT transition.
    out2 = run_meta_controller_tick(
        state=out1.state,
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(regime=Regime.RANGE, regime_confidence=0.6),
        pressure=_pressure(),
        config=_config(persistence_ticks=4, confidence_delta_threshold=0.5),
        elapsed_ns=1_000,
        ts_ns=300,
    )
    assert out2.regime_transitioned is False
    assert out2.state.router_state.current_regime is Regime.TREND_UP
    # Sizer used TREND_UP -> regime_factor = 1.0 (NOT RANGE's 0.5).
    assert out2.sizing_components.regime_factor == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Side / confidence / size flow
# ---------------------------------------------------------------------------


def test_consensus_side_drives_primary_side() -> None:
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.SELL)] * 3 + [_signal(Side.BUY)],
        belief=_belief(regime=Regime.TREND_DOWN, regime_confidence=0.9),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    assert out.proposed_side is Side.SELL
    assert out.primary_decision.side is Side.SELL


def test_confidence_components_surface_in_output() -> None:
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY, confidence=0.9)] * 4,
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    assert out.confidence_components.consensus == pytest.approx(1.0)
    assert out.confidence_components.strength == pytest.approx(0.9)
    assert out.confidence_components.signal_count == 4
    # Policy sees the composite confidence (modulo safety_modifier=1.0 here).
    assert out.primary_decision.confidence == pytest.approx(
        out.confidence_components.composite
    )


def test_sizer_kelly_cap_flows_through_policy() -> None:
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY, confidence=1.0)] * 8,
        belief=_belief(),
        pressure=_pressure(risk=0.0),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    # cap=0.25
    assert out.sizing_components.rationale == "kelly_capped"
    assert out.sizing_components.final_size == pytest.approx(0.25)
    assert out.primary_decision.size_fraction == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# INV-48 latency budget
# ---------------------------------------------------------------------------


def test_latency_budget_exceeded_returns_fallback() -> None:
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(),
        pressure=_pressure(),
        config=_config(latency_budget_ns=1_000),
        elapsed_ns=2_000,  # over budget
        ts_ns=200,
    )
    assert out.primary_decision is FALLBACK_POLICY
    # Shadow ignores latency by design.
    assert out.shadow_decision is not FALLBACK_POLICY
    assert out.shadow_decision.fallback is False


# ---------------------------------------------------------------------------
# INV-52 shadow path & divergence event
# ---------------------------------------------------------------------------


def test_shadow_runs_unconditionally() -> None:
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(),
        pressure=_pressure(safety_modifier=0.5),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    # safety_modifier=0.5 dampens primary; shadow forces it to 1.0.
    assert out.primary_decision.size_fraction != out.shadow_decision.size_fraction


def test_divergence_event_surfaces_when_decisions_differ() -> None:
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(),
        pressure=_pressure(safety_modifier=0.5),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    assert out.divergence_event is not None
    assert out.divergence_event.sub_kind is SystemEventKind.META_DIVERGENCE
    assert out.divergence_event.ts_ns == 200


def test_no_divergence_event_when_decisions_agree() -> None:
    """Identical primary / shadow → no event."""
    out = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        # Tie signals → side=HOLD; under HOLD the sizer collapses to 0.0
        # for both primary and shadow (no opportunity to diverge).
        signals=[_signal(Side.BUY), _signal(Side.SELL)],
        belief=_belief(),
        pressure=_pressure(safety_modifier=1.0),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    assert out.primary_decision == out.shadow_decision
    assert out.divergence_event is None


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_run_is_replay_deterministic() -> None:
    kwargs: dict = dict(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=1_000,
        ts_ns=200,
    )
    a = run_meta_controller_tick(**kwargs)
    b = run_meta_controller_tick(**kwargs)
    assert a == b


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_output_carries_elapsed_ns_and_version() -> None:
    out: MetaControllerOutput = run_meta_controller_tick(
        state=initial_meta_controller_state(),
        signals=[_signal(Side.BUY)] * 4,
        belief=_belief(),
        pressure=_pressure(),
        config=_config(),
        elapsed_ns=12_345,
        ts_ns=200,
    )
    assert out.elapsed_ns == 12_345
    assert out.version == META_CONTROLLER_VERSION


def test_initial_meta_controller_state_starts_unknown() -> None:
    s = initial_meta_controller_state()
    assert s.router_state.current_regime is Regime.UNKNOWN
    assert s.router_state.candidate_regime is Regime.UNKNOWN
    assert s.router_state.candidate_persistence_ticks == 0
