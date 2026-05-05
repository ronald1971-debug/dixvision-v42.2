"""AGT-03 — macro-regime gating agent (3rd AGT-XX agent).

Closes drift item H2.2 from the canonical-rebuild walk (5 of 5 agents
now needed; AGT-01 scalper + AGT-02 swing on disk; this PR ships
AGT-03 macro). Unlike the tick-driven scalper / swing agents, this
agent ingests macro classifications (RISK_ON / RISK_OFF / NEUTRAL /
CRISIS / UNKNOWN) from
:mod:`core.contracts.macro_regime.MacroRegime` and gates upstream
:class:`SignalEvent` records based on regime-direction alignment.

Behaviour matrix:

* RISK_ON  → BUY signals pass; SELL signals → HOLD.
* RISK_OFF → SELL signals pass; BUY  signals → HOLD.
* NEUTRAL  → both directions pass at reduced confidence.
* CRISIS   → all signals → HOLD (defensive).
* UNKNOWN  → all signals → HOLD (warmup).

INV-54 invariants enforced via
:class:`~intelligence_engine.agents._base.AgentBase`:

* :meth:`state_snapshot` is pure (no clock, no PRNG, no IO).
* :meth:`recent_decisions` is O(1) per call (bounded ring buffer).
* ``state_snapshot`` keys subset
  ``registry/agent_state_keys.yaml#AGT-03-macro``.
* ``rationale_tags`` drawn from
  ``registry/agent_rationale_tags.yaml`` (introduces
  ``regime_risk_on`` / ``regime_risk_off`` / ``regime_neutral`` /
  ``regime_crisis`` / ``regime_unknown`` /
  ``macro_aligned_buy`` / ``macro_aligned_sell``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from core.contracts.macro_regime import MacroRegime
from intelligence_engine.agents._base import AgentBase

_BUY = "BUY"
_SELL = "SELL"
_HOLD = "HOLD"


@dataclass
class MacroAgent(AgentBase, AgentIntrospection):
    """Macro-regime gating agent (AGT-03).

    Attributes:
        agent_id: Stable identifier matching
            ``registry/agent_state_keys.yaml`` and
            :class:`AgentDecisionTrace.signal_id` audit rows.
        neutral_confidence_scale: Multiplier applied to passing-signal
            confidence under NEUTRAL regime (default 0.5 = halved).
        min_confidence: Floor below which an otherwise-aligned signal
            is downgraded to HOLD.
        ring_capacity: Max recent decisions retained (O(1)).
    """

    agent_id: str = "AGT-03-macro"
    neutral_confidence_scale: float = 0.5
    min_confidence: float = 0.05
    ring_capacity: int = 64

    _regime: MacroRegime = field(default=MacroRegime.UNKNOWN, init=False, repr=False)
    _last_decision_direction: str = field(default=_HOLD, init=False, repr=False)
    _last_decision_confidence: float = field(default=0.0, init=False, repr=False)
    _last_decision_ts_ns: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.neutral_confidence_scale <= 1.0:
            raise ValueError("neutral_confidence_scale must be in [0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        AgentBase.__init__(self, self.agent_id, self.ring_capacity)

    # --- inputs ----------------------------------------------------

    def observe_regime(self, regime: MacroRegime) -> None:
        """Update the cached macro regime.

        Pure: stores the latest classification verbatim. Callers are
        expected to feed regimes produced by
        :class:`~intelligence_engine.macro.regime_engine.MacroRegimeEngine`.
        """

        self._regime = regime

    def decide(self, signal: SignalEvent) -> AgentDecisionTrace:
        """Gate ``signal`` through the cached macro regime.

        Returns an :class:`AgentDecisionTrace` (BUY / SELL / HOLD) and
        appends it to the bounded ring buffer.
        """

        rationale: list[str] = []
        regime = self._regime
        direction: str
        confidence: float

        if regime is MacroRegime.UNKNOWN:
            rationale.append("regime_unknown")
            direction = _HOLD
            confidence = 0.0
        elif regime is MacroRegime.CRISIS:
            rationale.append("regime_crisis")
            direction = _HOLD
            confidence = 0.0
        elif regime is MacroRegime.RISK_ON:
            rationale.append("regime_risk_on")
            if signal.side is Side.BUY:
                direction = _BUY
                confidence = float(signal.confidence)
                rationale.append("macro_aligned_buy")
            else:
                direction = _HOLD
                confidence = 0.0
        elif regime is MacroRegime.RISK_OFF:
            rationale.append("regime_risk_off")
            if signal.side is Side.SELL:
                direction = _SELL
                confidence = float(signal.confidence)
                rationale.append("macro_aligned_sell")
            else:
                direction = _HOLD
                confidence = 0.0
        elif regime is MacroRegime.NEUTRAL:
            rationale.append("regime_neutral")
            if signal.side is Side.BUY:
                direction = _BUY
                confidence = float(signal.confidence) * self.neutral_confidence_scale
                rationale.append("macro_aligned_buy")
            elif signal.side is Side.SELL:
                direction = _SELL
                confidence = float(signal.confidence) * self.neutral_confidence_scale
                rationale.append("macro_aligned_sell")
            else:
                direction = _HOLD
                confidence = 0.0
        else:  # defensive: unknown enum member → HOLD
            rationale.append("regime_unknown")
            direction = _HOLD
            confidence = 0.0

        if direction != _HOLD and confidence < self.min_confidence:
            direction = _HOLD
            confidence = 0.0
            rationale.append("confidence_below_floor")

        trace = AgentDecisionTrace(
            ts_ns=int(signal.ts_ns),
            signal_id=str(signal.meta.get("signal_id", "")),
            direction=direction,
            confidence=confidence,
            rationale_tags=tuple(rationale),
            memory_refs=(),
        )
        self._last_decision_direction = direction
        self._last_decision_confidence = confidence
        self._last_decision_ts_ns = int(signal.ts_ns)
        self._record_decision(trace)
        return trace

    # --- INV-54 introspection -------------------------------------

    def state_snapshot(self) -> Mapping[str, str]:
        return {
            "agent_id": self.agent_id,
            "lifecycle": "ACTIVE",
            "current_regime": self._regime.value,
            "last_decision_direction": self._last_decision_direction,
            "last_decision_confidence": f"{self._last_decision_confidence:.6f}",
            "last_decision_ts_ns": str(self._last_decision_ts_ns),
            "decisions_in_window": str(len(self._decision_buffer)),
            "neutral_confidence_scale": f"{self.neutral_confidence_scale:.6f}",
        }


__all__ = ["MacroAgent"]
