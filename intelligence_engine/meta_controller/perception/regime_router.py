"""Regime Router — INV-49 hysteresis-bounded transition gate.

Phase 6.T1e. Consumes :class:`core.coherence.BeliefState` (read-only)
and decides when to move from a *current* regime to a new *candidate*
regime. The gate is hysteresis-bounded so the router does not flap
on every tick of noise:

    transition iff persistence_ticks ≥ N
              OR  (candidate_confidence - current_confidence) ≥ θ

Both N and θ are versioned in ``registry/regime.yaml``. Belief State
itself is NOT modified; this module is a pure transition function on
(state, belief, config) → (state', transitioned).

Authority constraints (manifest §H1, INV-49):

* This module imports only from :mod:`core.contracts`,
  :mod:`core.coherence`, and the standard library.
* No engine cross-imports.
* No clock, no PRNG, no IO outside config load.
* Replay-deterministic: same inputs in the same order always
  produce the same output (INV-15).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from core.coherence.belief_state import BeliefState, Regime

REGIME_ROUTER_VERSION = "v3.3-T1e"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegimeRouterConfig:
    """Hysteresis coefficients for :func:`step_regime_router`.

    Loaded from ``registry/regime.yaml`` (single source of truth).

    Fields:
        persistence_ticks: ``N`` — minimum number of consecutive ticks
            the candidate regime must dominate before the router will
            transition.
        confidence_delta_threshold: ``θ`` in ``[0, 1]`` — alternative
            fast-path: a single tick whose ``regime_confidence``
            exceeds the current confidence by at least ``θ`` is
            allowed to transition immediately. This handles legitimate
            regime breaks (vol spikes, structural shifts) while the
            persistence floor handles noisy oscillations.
        version: Config schema version; recorded into emitted state.
    """

    persistence_ticks: int
    confidence_delta_threshold: float
    version: str = REGIME_ROUTER_VERSION

    def __post_init__(self) -> None:
        if self.persistence_ticks < 1:
            raise ValueError(
                "RegimeRouterConfig.persistence_ticks must be >= 1: "
                f"{self.persistence_ticks}"
            )
        if not (0.0 <= self.confidence_delta_threshold <= 1.0):
            raise ValueError(
                "RegimeRouterConfig.confidence_delta_threshold must be "
                f"in [0, 1]: {self.confidence_delta_threshold}"
            )


def load_regime_router_config(path: str | Path) -> RegimeRouterConfig:
    """Load :class:`RegimeRouterConfig` from a YAML file."""
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"regime config at {path} is not a mapping")
    return RegimeRouterConfig(
        persistence_ticks=int(raw["persistence_ticks"]),
        confidence_delta_threshold=float(raw["confidence_delta_threshold"]),
        version=str(raw.get("version", REGIME_ROUTER_VERSION)),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RegimeRouterState:
    """Immutable router state snapshot.

    The state is updated by *replacing the whole tuple* (frozen
    dataclass + :func:`dataclasses.replace`). This preserves replay
    determinism and makes every transition observable as a value
    diff.

    Fields:
        current_regime: The active regime that downstream consumers
            see right now.
        current_confidence: Confidence of the active regime as
            published when it was committed.
        candidate_regime: The challenger regime under observation.
            ``Regime.UNKNOWN`` means there is no challenger yet (the
            current regime is dominant in the latest tick).
        candidate_confidence: Latest belief confidence for the
            candidate regime.
        candidate_persistence_ticks: How many *consecutive* ticks the
            candidate has been observed since it became the
            challenger. Resets to 0 when the candidate is replaced or
            the current regime reasserts itself.
        last_belief_ts_ns: Timestamp of the most recent belief
            consumed; useful for staleness checks higher up the
            stack.
    """

    current_regime: Regime
    current_confidence: float
    candidate_regime: Regime
    candidate_confidence: float
    candidate_persistence_ticks: int
    last_belief_ts_ns: int


def initial_router_state(
    *,
    initial_regime: Regime = Regime.UNKNOWN,
    initial_confidence: float = 0.0,
) -> RegimeRouterState:
    """Construct a fresh :class:`RegimeRouterState` at boot."""
    if not (0.0 <= initial_confidence <= 1.0):
        raise ValueError(
            f"initial_confidence must be in [0, 1]: {initial_confidence}"
        )
    return RegimeRouterState(
        current_regime=initial_regime,
        current_confidence=initial_confidence,
        candidate_regime=Regime.UNKNOWN,
        candidate_confidence=0.0,
        candidate_persistence_ticks=0,
        last_belief_ts_ns=0,
    )


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


def step_regime_router(
    *,
    state: RegimeRouterState,
    belief: BeliefState,
    config: RegimeRouterConfig,
) -> tuple[RegimeRouterState, bool]:
    """Pure transition function — INV-49 hysteresis gate.

    Three cases:

    1. ``belief.regime == state.current_regime``: the active regime
       is reaffirmed. Any candidate is cleared, current confidence is
       refreshed. **No transition.**

    2. ``belief.regime == state.candidate_regime``: the challenger is
       persisting. Increment the persistence counter and refresh its
       confidence. Transition iff
       ``candidate_persistence_ticks >= config.persistence_ticks`` or
       the confidence-delta fast-path
       ``(belief.regime_confidence - state.current_confidence) >=
       config.confidence_delta_threshold`` fires.

    3. Otherwise: a *new* challenger has appeared. Replace the
       candidate slot with belief.regime, reset persistence to 1.
       Same fast-path: a single tick whose confidence delta exceeds
       ``θ`` is enough to transition immediately.

    Returns:
        ``(new_state, transitioned)`` where ``transitioned`` is
        ``True`` iff the router changed its current regime on this
        tick.
    """
    new_ts = belief.ts_ns

    # Case 1 — current regime reaffirmed.
    if belief.regime is state.current_regime:
        return (
            replace(
                state,
                current_confidence=belief.regime_confidence,
                candidate_regime=Regime.UNKNOWN,
                candidate_confidence=0.0,
                candidate_persistence_ticks=0,
                last_belief_ts_ns=new_ts,
            ),
            False,
        )

    # Case 2 — candidate persists.
    if belief.regime is state.candidate_regime and belief.regime is not Regime.UNKNOWN:
        new_persistence = state.candidate_persistence_ticks + 1
        delta = belief.regime_confidence - state.current_confidence
        if (
            new_persistence >= config.persistence_ticks
            or delta >= config.confidence_delta_threshold
        ):
            return (
                RegimeRouterState(
                    current_regime=belief.regime,
                    current_confidence=belief.regime_confidence,
                    candidate_regime=Regime.UNKNOWN,
                    candidate_confidence=0.0,
                    candidate_persistence_ticks=0,
                    last_belief_ts_ns=new_ts,
                ),
                True,
            )
        return (
            replace(
                state,
                candidate_confidence=belief.regime_confidence,
                candidate_persistence_ticks=new_persistence,
                last_belief_ts_ns=new_ts,
            ),
            False,
        )

    # Case 3 — new challenger.
    delta = belief.regime_confidence - state.current_confidence
    if delta >= config.confidence_delta_threshold and belief.regime is not Regime.UNKNOWN:
        # Confidence-delta fast-path: immediate transition without
        # waiting for persistence.
        return (
            RegimeRouterState(
                current_regime=belief.regime,
                current_confidence=belief.regime_confidence,
                candidate_regime=Regime.UNKNOWN,
                candidate_confidence=0.0,
                candidate_persistence_ticks=0,
                last_belief_ts_ns=new_ts,
            ),
            True,
        )

    # Persistence path: install new candidate, persistence = 1.
    # If persistence_ticks == 1 the new candidate transitions on this
    # very tick (config-controlled).
    new_persistence = 1
    if new_persistence >= config.persistence_ticks and belief.regime is not Regime.UNKNOWN:
        return (
            RegimeRouterState(
                current_regime=belief.regime,
                current_confidence=belief.regime_confidence,
                candidate_regime=Regime.UNKNOWN,
                candidate_confidence=0.0,
                candidate_persistence_ticks=0,
                last_belief_ts_ns=new_ts,
            ),
            True,
        )
    return (
        RegimeRouterState(
            current_regime=state.current_regime,
            current_confidence=state.current_confidence,
            candidate_regime=belief.regime,
            candidate_confidence=belief.regime_confidence,
            candidate_persistence_ticks=new_persistence,
            last_belief_ts_ns=new_ts,
        ),
        False,
    )


__all__ = [
    "REGIME_ROUTER_VERSION",
    "RegimeRouterConfig",
    "RegimeRouterState",
    "initial_router_state",
    "load_regime_router_config",
    "step_regime_router",
]
