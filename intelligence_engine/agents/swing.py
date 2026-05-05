"""AGT-02 — multi-bar swing agent (2nd AGT-XX agent).

Closes drift item H2.1 from the canonical-rebuild walk: only AGT-01
(scalper) was on disk against the spec's 5 AGT-XX agents
(scalper / swing / macro / LP / adversarial). This module ships
AGT-02 (swing) as a leaf-pure agent that reuses the
:class:`AgentIntrospection` Protocol shipped with AGT-01.

Behaviour:

The swing agent gates upstream :class:`SignalEvent` records through
a two-window simple-moving-average crossover on the mid price. The
fast window is short (default 8 ticks); the slow window is longer
(default 24 ticks) and covers the typical swing horizon. A signal
whose direction agrees with the fast-vs-slow crossover passes
through; one that fights it is downgraded to HOLD.

Compared to :class:`~intelligence_engine.agents.scalper.ScalperAgent`:

* Scalper uses a single rolling window + first-vs-last drift on the
  mid; swing uses fast/slow SMA crossover on the mid and gates on
  the *spread* between the two means rather than a raw drift.
* Scalper's typical horizon is intra-bar (window of 8 mids); swing
  defaults to a slow window of 24 mids and a wider crossover gate
  (5 bps default vs scalper's 1 bps).

INV-54 invariants enforced:

* :meth:`state_snapshot` is pure (no clock, no PRNG, no IO).
* :meth:`recent_decisions` is O(1) per call (bounded ring buffer in
  :class:`~intelligence_engine.agents._base.AgentBase`).
* ``state_snapshot`` keys subset
  ``registry/agent_state_keys.yaml#AGT-02-swing``.
* ``rationale_tags`` drawn from
  ``registry/agent_rationale_tags.yaml`` (no new vocabulary
  introduced; reuses ``momentum_up``/``momentum_down``/
  ``momentum_neutral``/``book_invalid``/``confidence_below_floor`` +
  the swing-specific ``ma_crossover_buy``/``ma_crossover_sell``).
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
class SwingAgent(AgentBase, AgentIntrospection):
    """Multi-bar swing agent (AGT-02).

    Two-window SMA crossover on the mid price, gated by signal-side
    alignment and a confidence floor.
    """

    agent_id: str = "AGT-02-swing"
    fast_window_size: int = 8
    slow_window_size: int = 24
    crossover_threshold_bps: float = 5.0
    min_confidence: float = 0.05
    ring_capacity: int = 64

    _mid_window: deque[float] = field(init=False, repr=False)
    _last_decision_direction: str = field(default=_HOLD, init=False, repr=False)
    _last_decision_confidence: float = field(default=0.0, init=False, repr=False)
    _last_decision_ts_ns: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fast_window_size < 2:
            raise ValueError("fast_window_size must be >= 2")
        if self.slow_window_size <= self.fast_window_size:
            raise ValueError("slow_window_size must be > fast_window_size")
        if self.crossover_threshold_bps < 0.0:
            raise ValueError("crossover_threshold_bps must be >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        AgentBase.__init__(self, self.agent_id, self.ring_capacity)
        self._mid_window = deque(maxlen=self.slow_window_size)

    # --- inputs ----------------------------------------------------

    def observe_tick(self, tick: MarketTick) -> None:
        """Update the rolling mid-price buffer from a market tick."""

        if tick.bid <= 0.0 or tick.ask <= 0.0 or tick.ask < tick.bid:
            return
        mid = 0.5 * (tick.bid + tick.ask)
        if mid <= 0.0:
            return
        self._mid_window.append(mid)

    def _fast_sma(self) -> float:
        n = self.fast_window_size
        items = list(self._mid_window)
        recent = items[-n:]
        return sum(recent) / float(len(recent))

    def _slow_sma(self) -> float:
        items = list(self._mid_window)
        return sum(items) / float(len(items))

    def decide(self, signal: SignalEvent) -> AgentDecisionTrace:
        """Gate ``signal`` through the SMA crossover filter.

        Returns an :class:`AgentDecisionTrace` (BUY / SELL / HOLD)
        and appends it to the bounded ring buffer.
        """

        rationale: list[str] = []

        if len(self._mid_window) < self.slow_window_size:
            direction = _HOLD
            confidence = 0.0
            rationale.append("momentum_neutral")
        else:
            slow = self._slow_sma()
            if slow <= 0.0:
                direction = _HOLD
                confidence = 0.0
                rationale.append("book_invalid")
            else:
                fast = self._fast_sma()
                spread_bps = (fast - slow) / slow * 10_000.0
                if spread_bps > self.crossover_threshold_bps:
                    rationale.append("momentum_up")
                    if signal.side is Side.BUY:
                        direction = _BUY
                        rationale.append("ma_crossover_buy")
                    else:
                        direction = _HOLD
                elif spread_bps < -self.crossover_threshold_bps:
                    rationale.append("momentum_down")
                    if signal.side is Side.SELL:
                        direction = _SELL
                        rationale.append("ma_crossover_sell")
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
            "fast_window_size": str(self.fast_window_size),
            "slow_window_size": str(self.slow_window_size),
            "crossover_threshold": f"{self.crossover_threshold_bps:.6f}",
        }


__all__ = ["SwingAgent"]
