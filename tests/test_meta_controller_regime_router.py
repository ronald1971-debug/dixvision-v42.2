"""Phase 6.T1e — Regime Router (INV-49 hysteresis) tests.

Exercises the three transition cases:

* current regime reaffirmed → no transition, candidate cleared
* candidate persists for N ticks → transition
* confidence-delta fast-path → immediate transition

Plus determinism, config validation, and registry YAML load.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from core.coherence.belief_state import BeliefState, Regime
from core.contracts.events import Side
from intelligence_engine.meta_controller.perception import (
    RegimeRouterConfig,
    RegimeRouterState,
    initial_router_state,
    load_regime_router_config,
    step_regime_router,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _belief(
    regime: Regime,
    *,
    confidence: float = 0.7,
    ts_ns: int = 1,
) -> BeliefState:
    return BeliefState(
        ts_ns=ts_ns,
        regime=regime,
        regime_confidence=confidence,
        consensus_side=Side.HOLD,
        signal_count=0,
        avg_confidence=0.0,
    )


def _cfg(*, persistence: int = 4, threshold: float = 0.4) -> RegimeRouterConfig:
    return RegimeRouterConfig(
        persistence_ticks=persistence,
        confidence_delta_threshold=threshold,
    )


# ---------------------------------------------------------------------------
# State / determinism
# ---------------------------------------------------------------------------


def test_router_state_is_frozen_dataclass() -> None:
    state = initial_router_state()
    assert dataclasses.is_dataclass(state)
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.current_regime = Regime.RANGE  # type: ignore[misc]


def test_initial_state_defaults_to_unknown() -> None:
    state = initial_router_state()
    assert state.current_regime is Regime.UNKNOWN
    assert state.current_confidence == 0.0
    assert state.candidate_regime is Regime.UNKNOWN
    assert state.candidate_persistence_ticks == 0


def test_step_replay_determinism() -> None:
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.5)
    belief = _belief(Regime.TREND_UP, confidence=0.7)
    cfg = _cfg()
    a, ta = step_regime_router(state=state, belief=belief, config=cfg)
    b, tb = step_regime_router(state=state, belief=belief, config=cfg)
    assert a == b
    assert ta == tb


# ---------------------------------------------------------------------------
# Case 1 — current regime reaffirmed
# ---------------------------------------------------------------------------


def test_current_regime_reaffirmed_clears_candidate() -> None:
    state = RegimeRouterState(
        current_regime=Regime.RANGE,
        current_confidence=0.6,
        candidate_regime=Regime.TREND_UP,
        candidate_confidence=0.55,
        candidate_persistence_ticks=2,
        last_belief_ts_ns=10,
    )
    belief = _belief(Regime.RANGE, confidence=0.8, ts_ns=20)
    new_state, transitioned = step_regime_router(
        state=state,
        belief=belief,
        config=_cfg(),
    )
    assert transitioned is False
    assert new_state.current_regime is Regime.RANGE
    assert new_state.current_confidence == 0.8  # refreshed
    assert new_state.candidate_regime is Regime.UNKNOWN
    assert new_state.candidate_persistence_ticks == 0
    assert new_state.last_belief_ts_ns == 20


# ---------------------------------------------------------------------------
# Case 2 — candidate persists for N ticks
# ---------------------------------------------------------------------------


def test_candidate_persists_for_n_ticks_then_transitions() -> None:
    """INV-49 persistence path: only after N consecutive ticks."""
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.6)
    cfg = _cfg(persistence=4, threshold=0.99)  # threshold disabled (high)

    # Drive 4 ticks of TREND_UP at modest confidence (delta < θ).
    for tick in range(1, 4):
        belief = _belief(Regime.TREND_UP, confidence=0.65, ts_ns=tick)
        state, transitioned = step_regime_router(
            state=state,
            belief=belief,
            config=cfg,
        )
        assert transitioned is False
        assert state.current_regime is Regime.RANGE
        assert state.candidate_regime is Regime.TREND_UP
        assert state.candidate_persistence_ticks == tick

    # 4th tick triggers transition (persistence_ticks == 4).
    belief = _belief(Regime.TREND_UP, confidence=0.65, ts_ns=4)
    state, transitioned = step_regime_router(
        state=state,
        belief=belief,
        config=cfg,
    )
    assert transitioned is True
    assert state.current_regime is Regime.TREND_UP
    assert state.candidate_regime is Regime.UNKNOWN
    assert state.candidate_persistence_ticks == 0


def test_candidate_interrupted_resets_persistence() -> None:
    """A different challenger replaces the previous candidate."""
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.6)
    cfg = _cfg(persistence=4, threshold=0.99)

    state, _ = step_regime_router(
        state=state,
        belief=_belief(Regime.TREND_UP, confidence=0.65, ts_ns=1),
        config=cfg,
    )
    assert state.candidate_regime is Regime.TREND_UP
    assert state.candidate_persistence_ticks == 1

    # New challenger TREND_DOWN — replaces TREND_UP, persistence resets.
    state, transitioned = step_regime_router(
        state=state,
        belief=_belief(Regime.TREND_DOWN, confidence=0.65, ts_ns=2),
        config=cfg,
    )
    assert transitioned is False
    assert state.current_regime is Regime.RANGE
    assert state.candidate_regime is Regime.TREND_DOWN
    assert state.candidate_persistence_ticks == 1


def test_current_regime_reasserts_clears_candidate() -> None:
    """Flapping noise: candidate appears for 2 ticks, then current returns."""
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.6)
    cfg = _cfg(persistence=4, threshold=0.99)

    state, _ = step_regime_router(
        state=state,
        belief=_belief(Regime.TREND_UP, confidence=0.65, ts_ns=1),
        config=cfg,
    )
    state, _ = step_regime_router(
        state=state,
        belief=_belief(Regime.TREND_UP, confidence=0.65, ts_ns=2),
        config=cfg,
    )
    assert state.candidate_persistence_ticks == 2

    state, transitioned = step_regime_router(
        state=state,
        belief=_belief(Regime.RANGE, confidence=0.7, ts_ns=3),
        config=cfg,
    )
    assert transitioned is False
    assert state.current_regime is Regime.RANGE
    assert state.candidate_regime is Regime.UNKNOWN
    assert state.candidate_persistence_ticks == 0


# ---------------------------------------------------------------------------
# Case 3 — confidence-delta fast-path
# ---------------------------------------------------------------------------


def test_confidence_delta_fast_path_transitions_immediately() -> None:
    """A single tick with delta >= θ transitions without waiting for N."""
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.3)
    cfg = _cfg(persistence=10, threshold=0.4)

    belief = _belief(Regime.VOL_SPIKE, confidence=0.95, ts_ns=1)
    new_state, transitioned = step_regime_router(
        state=state,
        belief=belief,
        config=cfg,
    )
    # delta = 0.95 - 0.3 = 0.65 >= threshold 0.4 → immediate.
    assert transitioned is True
    assert new_state.current_regime is Regime.VOL_SPIKE
    assert new_state.current_confidence == 0.95
    assert new_state.candidate_regime is Regime.UNKNOWN
    assert new_state.candidate_persistence_ticks == 0


def test_fast_path_blocked_for_unknown_target() -> None:
    """A challenger of UNKNOWN must never trigger a transition."""
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.0)
    cfg = _cfg(persistence=1, threshold=0.0)

    belief = _belief(Regime.UNKNOWN, confidence=0.5, ts_ns=1)
    new_state, transitioned = step_regime_router(
        state=state,
        belief=belief,
        config=cfg,
    )
    assert transitioned is False
    assert new_state.current_regime is Regime.RANGE


# ---------------------------------------------------------------------------
# Persistence floor of 1 — config-allowed instant transitions
# ---------------------------------------------------------------------------


def test_persistence_one_transitions_on_first_tick() -> None:
    """If operator picks N=1, the router transitions immediately."""
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.6)
    cfg = _cfg(persistence=1, threshold=0.99)
    belief = _belief(Regime.TREND_UP, confidence=0.5, ts_ns=1)
    new_state, transitioned = step_regime_router(
        state=state,
        belief=belief,
        config=cfg,
    )
    assert transitioned is True
    assert new_state.current_regime is Regime.TREND_UP


# ---------------------------------------------------------------------------
# Config validation + YAML load
# ---------------------------------------------------------------------------


def test_config_rejects_zero_persistence() -> None:
    with pytest.raises(ValueError, match="persistence_ticks"):
        RegimeRouterConfig(persistence_ticks=0, confidence_delta_threshold=0.4)


def test_config_rejects_threshold_out_of_range() -> None:
    with pytest.raises(ValueError, match="confidence_delta_threshold"):
        RegimeRouterConfig(persistence_ticks=4, confidence_delta_threshold=1.5)


def test_load_regime_router_config_from_registry() -> None:
    cfg = load_regime_router_config(REPO_ROOT / "registry" / "regime.yaml")
    assert cfg.persistence_ticks >= 1
    assert 0.0 <= cfg.confidence_delta_threshold <= 1.0


# ---------------------------------------------------------------------------
# Anti-flap stress
# ---------------------------------------------------------------------------


def test_alternating_regimes_never_transition_under_persistence() -> None:
    """Pure flapping (RANGE/TREND_UP alternating) must not transition.

    With N >= 2 and θ disabled, no candidate ever accumulates 2 in a row.
    """
    state = initial_router_state(initial_regime=Regime.RANGE, initial_confidence=0.6)
    cfg = _cfg(persistence=2, threshold=0.99)
    transitions = 0
    for tick in range(1, 21):
        regime = Regime.TREND_UP if tick % 2 == 1 else Regime.RANGE
        belief = _belief(regime, confidence=0.65, ts_ns=tick)
        state, t = step_regime_router(state=state, belief=belief, config=cfg)
        transitions += int(t)
    assert transitions == 0
    assert state.current_regime is Regime.RANGE
