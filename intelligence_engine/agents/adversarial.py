"""AGT-05 — adversarial / fade-the-crowd agent (5th and final AGT-XX agent).

Closes drift item H2.4 from the canonical-rebuild walk (5 of 5
AGT-XX agents now on disk: scalper / swing / macro / lp /
adversarial). The adversarial agent is the canonical contrarian
probe: when the upstream :class:`SignalEvent` carries a confidence
above a configurable ``fade_threshold``, the agent emits the
**opposite** direction at a (typically dampened) confidence. Its
purpose is to act as a built-in devil's advocate — useful as a
shadow / canary lane to stress-test confident signals against a
deterministic counter-stance.

* ``signal.confidence >= fade_threshold`` and ``side == BUY``  → emit SELL.
* ``signal.confidence >= fade_threshold`` and ``side == SELL`` → emit BUY.
* otherwise → HOLD (signal not strong enough to fade).

Output confidence is ``signal.confidence * fade_confidence_scale``
(default 0.5) so the contrarian view is structurally weaker than
the original — the agent is meant to be a probe, not a primary
gate.

INV-54 invariants enforced via
:class:`~intelligence_engine.agents._base.AgentBase`:

* :meth:`state_snapshot` is pure (no clock, no PRNG, no IO).
* :meth:`recent_decisions` is O(1) per call (bounded ring buffer).
* ``state_snapshot`` keys subset
  ``registry/agent_state_keys.yaml#AGT-05-adversarial``.
* ``rationale_tags`` drawn from
  ``registry/agent_rationale_tags.yaml`` (introduces
  ``adversarial_fade_buy`` / ``adversarial_fade_sell`` /
  ``adversarial_below_threshold``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from intelligence_engine.agents._base import AgentBase

_BUY = "BUY"
_SELL = "SELL"
_HOLD = "HOLD"


@dataclass
class AdversarialAgent(AgentBase, AgentIntrospection):
    """Contrarian / fade-the-crowd agent (AGT-05).

    Attributes:
        agent_id: Stable identifier matching
            ``registry/agent_state_keys.yaml`` and audit ledger rows.
        fade_threshold: Minimum upstream-signal confidence below
            which the agent will not fade (returns HOLD instead).
        fade_confidence_scale: Multiplier applied to the upstream
            confidence to produce the contrarian-output confidence
            (default 0.5 = halve).
        min_confidence: Floor below which a faded direction is
            downgraded to HOLD.
        ring_capacity: Max recent decisions retained (O(1)).
    """

    agent_id: str = "AGT-05-adversarial"
    fade_threshold: float = 0.5
    fade_confidence_scale: float = 0.5
    min_confidence: float = 0.05
    ring_capacity: int = 64

    _last_decision_direction: str = field(default=_HOLD, init=False, repr=False)
    _last_decision_confidence: float = field(default=0.0, init=False, repr=False)
    _last_decision_ts_ns: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.fade_threshold <= 1.0:
            raise ValueError("fade_threshold must be in [0, 1]")
        if not 0.0 <= self.fade_confidence_scale <= 1.0:
            raise ValueError("fade_confidence_scale must be in [0, 1]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        AgentBase.__init__(self, self.agent_id, self.ring_capacity)

    # --- decide ----------------------------------------------------

    def decide(self, signal: SignalEvent) -> AgentDecisionTrace:
        rationale: list[str] = []
        signal_confidence = float(signal.confidence)

        if signal_confidence < self.fade_threshold:
            rationale.append("adversarial_below_threshold")
            direction = _HOLD
            confidence = 0.0
        elif signal.side is Side.BUY:
            rationale.append("adversarial_fade_buy")
            direction = _SELL
            confidence = signal_confidence * self.fade_confidence_scale
        elif signal.side is Side.SELL:
            rationale.append("adversarial_fade_sell")
            direction = _BUY
            confidence = signal_confidence * self.fade_confidence_scale
        else:
            # Side.HOLD has nothing to fade.
            rationale.append("adversarial_below_threshold")
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
            "last_decision_direction": self._last_decision_direction,
            "last_decision_confidence": f"{self._last_decision_confidence:.6f}",
            "last_decision_ts_ns": str(self._last_decision_ts_ns),
            "decisions_in_window": str(len(self._decision_buffer)),
            "fade_threshold": f"{self.fade_threshold:.6f}",
            "fade_confidence_scale": f"{self.fade_confidence_scale:.6f}",
        }


__all__ = ["AdversarialAgent"]
