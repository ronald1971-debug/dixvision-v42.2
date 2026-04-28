"""Meta-Controller orchestrator — composes the H1 sub-packages.

Phase 6.T1b. Single pure entry point that runs one tick through:

    perception/  →  evaluation/  →  allocation/  →  policy/

Wiring contract:

* ``regime_router`` (perception) — INV-49 hysteresis gate over
  :class:`BeliefState`. Produces the *committed* regime that
  evaluation / allocation / policy must use.
* ``confidence_engine`` (evaluation) — composite confidence over the
  signal sequence + consensus-side resolution.
* ``position_sizer`` (allocation) — composite size over
  ``(confidence, regime, pressure)`` with kelly cap.
* ``execution_policy`` (policy, primary) — INV-48 latency-budget
  fallback + regime-aware side resolution + safety_modifier damping.
* ``shadow_policy`` (policy, INV-52) — non-acting alternative
  decision; emits a ``META_DIVERGENCE`` SystemEvent when primary and
  shadow disagree.

Authority constraints:

* Pure deterministic function. No clock, no PRNG; the caller passes
  ``elapsed_ns`` and ``ts_ns`` in.
* Imports only :mod:`core.contracts`, :mod:`core.coherence`, and the
  already-built H1 sub-packages.
* No Governance writes; the orchestrator does not call
  :mod:`governance_engine`. Wiring into the runtime hot path lands
  in Phase 6.T1c — this module just composes the pure functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from core.coherence.belief_state import BeliefState
from core.coherence.performance_pressure import PressureVector
from core.contracts.events import Side, SignalEvent, SystemEvent
from intelligence_engine.meta_controller.allocation import (
    PositionSizerConfig,
    SizingComponents,
    compute_position_size,
)
from intelligence_engine.meta_controller.evaluation import (
    ConfidenceComponents,
    ConfidenceEngineConfig,
    compute_confidence,
    resolve_proposed_side,
)
from intelligence_engine.meta_controller.perception.regime_router import (
    RegimeRouterConfig,
    RegimeRouterState,
    initial_router_state,
    step_regime_router,
)
from intelligence_engine.meta_controller.policy import (
    ExecutionDecision,
    compute_shadow_decision,
    decide_execution_policy,
    emit_divergence_event,
)

META_CONTROLLER_VERSION = "v3.3-T1b"


# ---------------------------------------------------------------------------
# Config + state + output
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetaControllerConfig:
    """Bundle of sub-package configs + the latency budget."""

    router_config: RegimeRouterConfig
    confidence_config: ConfidenceEngineConfig
    sizer_config: PositionSizerConfig
    latency_budget_ns: int
    version: str = META_CONTROLLER_VERSION

    def __post_init__(self) -> None:
        if self.latency_budget_ns <= 0:
            raise ValueError(
                f"MetaControllerConfig.latency_budget_ns must be > 0: "
                f"{self.latency_budget_ns}"
            )


@dataclass(frozen=True, slots=True)
class MetaControllerState:
    """Immutable orchestrator state — currently only the router."""

    router_state: RegimeRouterState
    version: str = META_CONTROLLER_VERSION


def initial_meta_controller_state() -> MetaControllerState:
    """Construct a fresh :class:`MetaControllerState` at boot."""
    return MetaControllerState(router_state=initial_router_state())


@dataclass(frozen=True, slots=True)
class MetaControllerOutput:
    """One-tick result bundle.

    ``primary_decision`` is the value that will be handed to the
    PolicyEngine. ``shadow_decision`` and ``divergence_event`` are
    consumed by the offline calibrator only (INV-52).

    The four ``*_components`` / ``*_state`` fields are intentionally
    exposed for J3 reward attribution in T1c.
    """

    state: MetaControllerState
    primary_decision: ExecutionDecision
    shadow_decision: ExecutionDecision
    divergence_event: SystemEvent | None
    confidence_components: ConfidenceComponents
    sizing_components: SizingComponents
    proposed_side: Side
    regime_transitioned: bool
    elapsed_ns: int
    version: str = META_CONTROLLER_VERSION


# ---------------------------------------------------------------------------
# Tick
# ---------------------------------------------------------------------------


def run_meta_controller_tick(
    *,
    state: MetaControllerState,
    signals: Sequence[SignalEvent],
    belief: BeliefState,
    pressure: PressureVector,
    config: MetaControllerConfig,
    elapsed_ns: int,
    ts_ns: int,
) -> MetaControllerOutput:
    """Run one composed pass through the H1 sub-packages.

    Wiring order:

    1. ``step_regime_router(belief)`` → committed regime.
    2. ``resolve_proposed_side(signals)`` → consensus direction.
    3. ``compute_confidence(signals)`` → composite confidence.
    4. ``compute_position_size(...)`` → composite size.
    5. ``decide_execution_policy(...)`` → primary decision (INV-48).
    6. ``compute_shadow_decision(...)`` → INV-52 shadow.
    7. ``emit_divergence_event(...)`` → ``META_DIVERGENCE`` or ``None``.

    The committed regime from step 1 (i.e. the post-hysteresis
    regime in ``new_router_state.current_regime``) is what flows
    into 4 / 5 / 6 — never the raw ``belief.regime``. This is the
    whole point of the H1 split: hysteresis applies before sizing
    and execution decide anything.
    """
    if elapsed_ns < 0:
        raise ValueError(
            f"run_meta_controller_tick: elapsed_ns must be >= 0: {elapsed_ns}"
        )

    new_router_state, regime_transitioned = step_regime_router(
        state=state.router_state,
        belief=belief,
        config=config.router_config,
    )
    committed_regime = new_router_state.current_regime

    proposed_side = resolve_proposed_side(signals)
    confidence_components = compute_confidence(signals, config.confidence_config)
    sizing_components = compute_position_size(
        confidence=confidence_components.composite,
        regime=committed_regime,
        pressure=pressure,
        config=config.sizer_config,
    )

    primary_decision = decide_execution_policy(
        regime=committed_regime,
        pressure=pressure,
        proposed_side=proposed_side,
        proposed_size=sizing_components.final_size,
        proposed_confidence=confidence_components.composite,
        latency_budget_ns=config.latency_budget_ns,
        elapsed_ns=elapsed_ns,
    )
    shadow_decision = compute_shadow_decision(
        regime=committed_regime,
        pressure=pressure,
        proposed_side=proposed_side,
        proposed_size=sizing_components.final_size,
        proposed_confidence=confidence_components.composite,
        latency_budget_ns=config.latency_budget_ns,
        elapsed_ns=elapsed_ns,
    )
    divergence_event = emit_divergence_event(
        ts_ns=ts_ns,
        primary=primary_decision,
        shadow=shadow_decision,
    )

    new_state = replace(state, router_state=new_router_state)
    return MetaControllerOutput(
        state=new_state,
        primary_decision=primary_decision,
        shadow_decision=shadow_decision,
        divergence_event=divergence_event,
        confidence_components=confidence_components,
        sizing_components=sizing_components,
        proposed_side=proposed_side,
        regime_transitioned=regime_transitioned,
        elapsed_ns=elapsed_ns,
    )


__all__ = [
    "META_CONTROLLER_VERSION",
    "MetaControllerConfig",
    "MetaControllerOutput",
    "MetaControllerState",
    "initial_meta_controller_state",
    "run_meta_controller_tick",
]
