"""Phase 6.T1c — :class:`MetaControllerHotPath` engine-wiring tests.

Covers:

* Per-tick derivation of belief + pressure from signals + scalars.
* State advances through the inner adapter; same instance retains
  router persistence across calls.
* Returned ledger has the fixed four-event shape.
* INV-15 replay determinism — two harnesses fed identical inputs
  produce identical state + identical ledgers.
* INV-48 fallback flows through unchanged.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.coherence.belief_state import Regime
from core.coherence.performance_pressure import (
    PressureConfig,
    load_pressure_config,
)
from core.contracts.events import Side, SignalEvent, SystemEventKind
from intelligence_engine.meta_controller import (
    MetaControllerConfig,
    MetaControllerHotPath,
    MetaControllerState,
    initial_meta_controller_state,
    load_meta_controller_config,
)
from intelligence_engine.meta_controller.allocation import PositionSizerConfig
from intelligence_engine.meta_controller.evaluation import ConfidenceEngineConfig
from intelligence_engine.meta_controller.perception.regime_router import (
    RegimeRouterConfig,
)
from intelligence_engine.meta_controller.policy import FALLBACK_POLICY

REPO_ROOT = Path(__file__).resolve().parent.parent


def _signal(side: Side, confidence: float = 0.8, ts_ns: int = 1) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol="X",
        side=side,
        confidence=confidence,
    )


def _meta_config(latency_budget_ns: int = 500_000) -> MetaControllerConfig:
    return MetaControllerConfig(
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
        latency_budget_ns=latency_budget_ns,
    )


def _pressure_config() -> PressureConfig:
    return load_pressure_config(REPO_ROOT / "registry" / "pressure.yaml")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_default_initial_state() -> None:
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
    )
    assert h.state == initial_meta_controller_state()
    assert h.meta_config.latency_budget_ns == 500_000


def test_explicit_initial_state_is_used() -> None:
    custom = initial_meta_controller_state()
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
        initial_state=custom,
    )
    assert h.state is custom


# ---------------------------------------------------------------------------
# Step semantics
# ---------------------------------------------------------------------------


def _step_kwargs(ts_ns: int = 100, elapsed_ns: int = 1_000) -> dict[str, object]:
    return dict(
        ts_ns=ts_ns,
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        perf=0.5,
        risk=0.0,
        drift=0.0,
        latency=0.0,
        vol_spike_z=0.0,
        elapsed_ns=elapsed_ns,
    )


def test_step_returns_decision_and_ledger() -> None:
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
    )
    decision, ledger = h.step(**_step_kwargs())
    assert decision.side is Side.BUY
    assert decision.size_fraction > 0
    assert ledger[0].sub_kind is SystemEventKind.BELIEF_STATE_SNAPSHOT
    assert ledger[1].sub_kind is SystemEventKind.PRESSURE_VECTOR_SNAPSHOT
    assert ledger[2].sub_kind is SystemEventKind.META_AUDIT


def test_step_advances_state() -> None:
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
    )
    state0 = h.state
    h.step(**_step_kwargs())
    state1 = h.state
    assert state1 is not state0
    # First commit transitions into TREND_UP for unanimous BUY signals.
    assert state1.router_state.current_regime is Regime.TREND_UP


def test_step_full_returns_orchestrator_output() -> None:
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
    )
    output, ledger = h.step_full(**_step_kwargs())
    # J3 components surface on the full output.
    assert output.confidence_components.composite > 0
    assert output.sizing_components.final_size > 0
    assert output.proposed_side is Side.BUY
    assert len(ledger) >= 3


# ---------------------------------------------------------------------------
# Derivation surface
# ---------------------------------------------------------------------------


def test_derive_inputs_is_pure_and_does_not_advance_state() -> None:
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
    )
    state_before = h.state
    belief, pressure = h.derive_inputs(
        ts_ns=100,
        signals=[_signal(Side.BUY), _signal(Side.SELL)],
        perf=0.5,
        risk=0.2,
        drift=0.1,
        latency=0.0,
        vol_spike_z=0.0,
    )
    assert h.state is state_before
    assert belief.signal_count == 2
    assert pressure.signal_count == 2


# ---------------------------------------------------------------------------
# INV-48 latency fallback
# ---------------------------------------------------------------------------


def test_latency_fallback_propagates_through_harness() -> None:
    h = MetaControllerHotPath(
        meta_config=_meta_config(latency_budget_ns=10),
        pressure_config=_pressure_config(),
    )
    decision, ledger = h.step(
        ts_ns=100,
        signals=[_signal(Side.BUY), _signal(Side.BUY)],
        perf=0.5,
        risk=0.0,
        drift=0.0,
        latency=0.0,
        vol_spike_z=0.0,
        elapsed_ns=1_000_000,
    )
    assert decision is FALLBACK_POLICY
    # Calibration ledger is still emitted under degrade.
    assert ledger[0].sub_kind is SystemEventKind.BELIEF_STATE_SNAPSHOT
    assert ledger[1].sub_kind is SystemEventKind.PRESSURE_VECTOR_SNAPSHOT
    assert ledger[2].sub_kind is SystemEventKind.META_AUDIT
    assert ledger[2].payload["decision_fallback"] == "true"


# ---------------------------------------------------------------------------
# INV-15 replay determinism
# ---------------------------------------------------------------------------


def test_two_harnesses_produce_identical_streams() -> None:
    """Same config + same input sequence ⇒ same state + same ledger."""
    inputs = [
        _step_kwargs(ts_ns=100 + i, elapsed_ns=1_000) for i in range(5)
    ]

    runs = []
    for _ in range(2):
        h = MetaControllerHotPath(
            meta_config=_meta_config(),
            pressure_config=_pressure_config(),
        )
        ledgers: list[tuple[object, ...]] = []
        decisions = []
        for kw in inputs:
            d, ledger = h.step(**kw)
            decisions.append(d)
            ledgers.append(ledger)
        runs.append((tuple(decisions), tuple(ledgers), h.state))

    assert runs[0] == runs[1]


# ---------------------------------------------------------------------------
# Hysteresis through the harness
# ---------------------------------------------------------------------------


def test_hysteresis_state_persists_across_steps() -> None:
    """Single regime-flip in the middle of a run is suppressed by INV-49."""
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
    )

    # 2 BUY ticks, 1 SELL tick (single-tick noise), 2 more BUY ticks.
    sequence = [
        [_signal(Side.BUY), _signal(Side.BUY)],
        [_signal(Side.BUY), _signal(Side.BUY)],
        [_signal(Side.SELL), _signal(Side.SELL)],
        [_signal(Side.BUY), _signal(Side.BUY)],
        [_signal(Side.BUY), _signal(Side.BUY)],
    ]
    final_regime = None
    for i, sigs in enumerate(sequence):
        _, _ = h.step(
            ts_ns=100 + i,
            signals=sigs,
            perf=0.5,
            risk=0.0,
            drift=0.0,
            latency=0.0,
            vol_spike_z=0.0,
            elapsed_ns=1_000,
        )
        final_regime = h.state.router_state.current_regime

    # Hysteresis: the lone SELL tick does not flip the committed regime.
    assert final_regime is Regime.TREND_UP


# ---------------------------------------------------------------------------
# Registry round-trip
# ---------------------------------------------------------------------------


def test_harness_drives_with_registry_loaded_configs() -> None:
    h = MetaControllerHotPath(
        meta_config=load_meta_controller_config(),
        pressure_config=_pressure_config(),
    )
    decision, ledger = h.step(**_step_kwargs())
    assert decision.side is Side.BUY
    assert ledger[2].sub_kind is SystemEventKind.META_AUDIT


# ---------------------------------------------------------------------------
# Type sanity
# ---------------------------------------------------------------------------


def test_state_is_immutable_meta_controller_state() -> None:
    h = MetaControllerHotPath(
        meta_config=_meta_config(),
        pressure_config=_pressure_config(),
    )
    s = h.state
    assert isinstance(s, MetaControllerState)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.router_state = None  # type: ignore[misc]
