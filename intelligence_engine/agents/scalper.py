"""AGT-01 — high-frequency intra-bar scalper (1st AGT-XX agent).

Closes drift item C from the canonical-rebuild walk:
``intelligence_engine/agents/`` was empty on ``main`` even though
the spec calls for 5 AGT-XX agents (scalper / swing / macro / LP /
adversarial). This module ships the first one (AGT-01) along with
the :class:`AgentIntrospection` Protocol (INV-54).

Behaviour:

The scalper consumes upstream :class:`SignalEvent` records (typically
from the order-flow imbalance plugin or microstructure_v1) and
gates them through a short rolling-window momentum filter on the
mid price. A signal whose direction agrees with the recent mid
trajectory passes through; one that fights it is downgraded to
HOLD. The agent is intentionally simple — the goal is to land the
``agents/`` namespace + the introspection contract in production,
not to ship a profitable scalping strategy.

INV-54 invariants enforced:

* :meth:`state_snapshot` is pure (no clock, no PRNG, no IO; reads
  only ``self`` state and returns string values).
* :meth:`recent_decisions` is O(1) per call (bounded ring buffer in
  :class:`~intelligence_engine.agents._base.AgentBase`).
* :meth:`state_snapshot` keys subset
  ``registry/agent_state_keys.yaml`` (validated in tests + asserted
  at construction time).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from core.contracts.agent import AgentDecisionTrace, AgentIntrospection
from core.contracts.events import Side, SignalEvent
from core.contracts.market import MarketTick
from intelligence_engine.agents._base import AgentBase

_BUY = "BUY"
_SELL = "SELL"
_HOLD = "HOLD"


@dataclass
class ScalperAgent(AgentBase, AgentIntrospection):
    """High-frequency intra-bar scalper agent (AGT-01)."""

    agent_id: str = "AGT-01-scalper"
    mid_window_size: int = 8
    momentum_threshold_bps: float = 1.0
    min_confidence: float = 0.05
    ring_capacity: int = 64

    _mid_window: deque[float] = field(init=False, repr=False)
    _last_decision_direction: str = field(default=_HOLD, init=False, repr=False)
    _last_decision_confidence: float = field(default=0.0, init=False, repr=False)
    _last_decision_ts_ns: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.mid_window_size < 2:
            raise ValueError("mid_window_size must be >= 2")
        if self.momentum_threshold_bps < 0.0:
            raise ValueError("momentum_threshold_bps must be >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        # Bypass dataclass-generated __init__ field shadowing for the
        # AgentBase machinery: the dataclass declares ``agent_id`` /
        # ``ring_capacity`` so they appear as ctor kwargs, but the
        # base owns the validated mirrors.
        AgentBase.__init__(self, self.agent_id, self.ring_capacity)
        self._mid_window = deque(maxlen=self.mid_window_size)

    # --- inputs ----------------------------------------------------

    def observe_tick(self, tick: MarketTick) -> None:
        """Update the rolling mid-price window from a market tick.

        Pure with respect to outputs (no return value, no side effects
        beyond the agent's own buffer).
        """

        if tick.bid <= 0.0 or tick.ask <= 0.0 or tick.ask < tick.bid:
            return
        mid = 0.5 * (tick.bid + tick.ask)
        if mid <= 0.0:
            return
        self._mid_window.append(mid)

    def decide(self, signal: SignalEvent) -> AgentDecisionTrace:
        """Gate ``signal`` through the rolling-mid momentum filter.

        Returns a :class:`AgentDecisionTrace` describing the gated
        decision (BUY / SELL / HOLD) and appends it to the ring
        buffer. The original ``SignalEvent`` is **not** mutated; the
        scalper only emits its decision via the returned trace +
        introspection surface.
        """

        rationale: list[str] = []

        if len(self._mid_window) < self.mid_window_size:
            direction = _HOLD
            confidence = 0.0
            rationale.append("momentum_neutral")
        else:
            first = self._mid_window[0]
            last = self._mid_window[-1]
            if first <= 0.0:
                direction = _HOLD
                confidence = 0.0
                rationale.append("book_invalid")
            else:
                drift_bps = (last - first) / first * 10_000.0
                if drift_bps > self.momentum_threshold_bps:
                    rationale.append("momentum_up")
                    if signal.side is Side.BUY:
                        direction = _BUY
                        rationale.append("mid_drift_buy")
                    else:
                        direction = _HOLD
                elif drift_bps < -self.momentum_threshold_bps:
                    rationale.append("momentum_down")
                    if signal.side is Side.SELL:
                        direction = _SELL
                        rationale.append("mid_drift_sell")
                    else:
                        direction = _HOLD
                else:
                    rationale.append("momentum_neutral")
                    direction = _HOLD

                if direction == _HOLD:
                    confidence = 0.0
                else:
                    confidence = float(signal.confidence)
                    if confidence < self.min_confidence:
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
            "mid_window_size": str(self.mid_window_size),
            "momentum_threshold": f"{self.momentum_threshold_bps:.6f}",
        }


__all__ = ["ScalperAgent"]
