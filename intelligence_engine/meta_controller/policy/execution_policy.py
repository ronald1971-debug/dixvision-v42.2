"""Execution Policy — primary final gate with INV-48 fallback.

Phase 6.T1b. Pure decision function consuming the (regime, pressure,
proposed sizing, latency budget) tuple and returning a single
:class:`ExecutionDecision`.

INV-48 (manifest_v3.2_delta.md §1.1):

    Meta-controller must degrade to O(1) under latency pressure.
    If ``elapsed_ns > latency_budget_ns`` the policy must return a
    precomputed fallback decision with no dependency on Belief State,
    Pressure Vector, or any non-trivial computation.

The fallback (:data:`FALLBACK_POLICY`) is a constant; constructing it
allocates nothing. Governance retains the hard-override path
(``safety_modifier == 0``) outside this module — see PolicyEngine.

Authority constraints:

* Imports only from :mod:`core.contracts`,
  :mod:`core.coherence`, and the standard library.
* No clock, no PRNG; replay-deterministic per INV-15.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from core.coherence.belief_state import Regime
from core.coherence.performance_pressure import PressureVector
from core.contracts.events import Side

EXECUTION_POLICY_VERSION = "v3.3-T1b"


# ---------------------------------------------------------------------------
# Decision record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExecutionDecision:
    """Final decision emitted by the meta-controller policy layer.

    Read-only by construction. Downstream PolicyEngine validates and
    ledgers; this record is the only thing the meta-controller hands
    to it.

    Fields:
        side: ``BUY``, ``SELL``, or ``HOLD``.
        size_fraction: Fraction of allocation budget in ``[0, 1]``.
            ``0.0`` means do nothing (HOLD or zero-sized).
        confidence: Decision confidence in ``[0, 1]``. Combines belief
            confidence, pressure damping, and proposed-size signal
            strength.
        rationale: Short audit string. Stable values:
            * ``"primary"`` — primary path produced this decision.
            * ``"latency_budget_exceeded:fallback"`` — INV-48.
            * ``"safety_modifier_zero:fallback"`` — pressure damping
              forced HOLD without governance involvement.
            * ``"unknown_regime:fallback"`` — UNKNOWN regime cannot
              produce a directional bias.
        fallback: ``True`` iff the decision is a fallback (any of the
            above non-``"primary"`` rationales).
        version: Schema version stamp.
    """

    side: Side
    size_fraction: float
    confidence: float
    rationale: str
    fallback: bool
    version: str = EXECUTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not (0.0 <= self.size_fraction <= 1.0):
            raise ValueError(
                f"ExecutionDecision.size_fraction must be in [0, 1]: "
                f"{self.size_fraction}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"ExecutionDecision.confidence must be in [0, 1]: "
                f"{self.confidence}"
            )


# ---------------------------------------------------------------------------
# Fallback (INV-48)
# ---------------------------------------------------------------------------


FALLBACK_POLICY: Final[ExecutionDecision] = ExecutionDecision(
    side=Side.HOLD,
    size_fraction=0.0,
    confidence=0.0,
    rationale="latency_budget_exceeded:fallback",
    fallback=True,
)
"""INV-48 constant-time fallback. Zero allocation per call site."""


# Internal sentinels for the other fallback rationales. They share
# zero-size HOLD semantics with FALLBACK_POLICY but distinguish the
# audit string so calibration / debug can attribute the cause.
_SAFETY_ZERO_FALLBACK = ExecutionDecision(
    side=Side.HOLD,
    size_fraction=0.0,
    confidence=0.0,
    rationale="safety_modifier_zero:fallback",
    fallback=True,
)
_UNKNOWN_REGIME_FALLBACK = ExecutionDecision(
    side=Side.HOLD,
    size_fraction=0.0,
    confidence=0.0,
    rationale="unknown_regime:fallback",
    fallback=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _side_for_regime(regime: Regime, proposed_side: Side) -> Side:
    """Resolve the directional bias from regime + proposed side.

    Rule of thumb (T1b placeholder):

    * ``TREND_UP``: prefer ``BUY``; reject SELL by collapsing to HOLD.
    * ``TREND_DOWN``: prefer ``SELL``; reject BUY by collapsing to HOLD.
    * ``RANGE``: pass-through (mean-reversion strategies own the
      direction).
    * ``VOL_SPIKE``: HOLD — no directional bias under volatility
      shock.
    * ``UNKNOWN``: caller must short-circuit; we still safe-collapse
      to HOLD here as defence-in-depth.
    """
    if regime is Regime.TREND_UP:
        return proposed_side if proposed_side is Side.BUY else Side.HOLD
    if regime is Regime.TREND_DOWN:
        return proposed_side if proposed_side is Side.SELL else Side.HOLD
    if regime is Regime.RANGE:
        return proposed_side
    if regime is Regime.VOL_SPIKE:
        return Side.HOLD
    return Side.HOLD  # UNKNOWN


# ---------------------------------------------------------------------------
# Primary policy
# ---------------------------------------------------------------------------


def decide_execution_policy(
    *,
    regime: Regime,
    pressure: PressureVector,
    proposed_side: Side,
    proposed_size: float,
    proposed_confidence: float,
    latency_budget_ns: int,
    elapsed_ns: int,
) -> ExecutionDecision:
    """Pure decision function for the meta-controller's final gate.

    Order of evaluation (each step is constant-time):

    1. **INV-48 latency guard** — if the elapsed processing time has
       already exceeded the budget, return :data:`FALLBACK_POLICY`
       *without inspecting any of the other inputs*. This is the
       degraded-mode contract.
    2. **Pressure hard-damp** — if ``pressure.safety_modifier == 0``,
       short-circuit to :data:`_SAFETY_ZERO_FALLBACK` (HOLD, zero
       size). Any positive damping is folded into sizing instead.
    3. **Unknown regime guard** — without a regime there is no
       directional thesis; return HOLD with rationale
       ``"unknown_regime:fallback"``.
    4. **Primary path**:
       * resolve side via :func:`_side_for_regime`;
       * size = ``clamp01(proposed_size * pressure.safety_modifier)``;
       * confidence = ``clamp01(proposed_confidence *
         pressure.safety_modifier)``;
       * if the resolved side is HOLD, force size and confidence to
         0 (HOLD has no magnitude).
    """
    if elapsed_ns > latency_budget_ns:
        return FALLBACK_POLICY

    if pressure.safety_modifier <= 0.0:
        return _SAFETY_ZERO_FALLBACK

    if regime is Regime.UNKNOWN:
        return _UNKNOWN_REGIME_FALLBACK

    side = _side_for_regime(regime, proposed_side)
    if side is Side.HOLD:
        return ExecutionDecision(
            side=Side.HOLD,
            size_fraction=0.0,
            confidence=0.0,
            rationale="primary",
            fallback=False,
        )

    size = _clamp01(proposed_size * pressure.safety_modifier)
    confidence = _clamp01(proposed_confidence * pressure.safety_modifier)
    return ExecutionDecision(
        side=side,
        size_fraction=size,
        confidence=confidence,
        rationale="primary",
        fallback=False,
    )


__all__ = [
    "EXECUTION_POLICY_VERSION",
    "FALLBACK_POLICY",
    "ExecutionDecision",
    "decide_execution_policy",
]
