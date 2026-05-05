"""AGT-04 — liquidity-provider mean-reversion agent (4th AGT-XX agent).

Closes drift item H2.3 from the canonical-rebuild walk (4 of 5
AGT-XX agents now on disk: scalper / swing / macro / lp). Unlike
the trend-following scalper / swing agents, the LP agent gates on
*mean reversion*: it favours BUY signals when the current mid sits
below a rolling fair-value mean by more than a configured band,
and favours SELL signals when the current mid sits above by the
same band. Inside the band the agent is neutral.

This is the canonical retail / market-making stance — provide
liquidity into dips, take it back into rallies — and is
deliberately complementary to the swing agent's trend-continuation
gate so a meta-controller could blend the two.

INV-54 invariants enforced via
:class:`~intelligence_engine.agents._base.AgentBase`:

* :meth:`state_snapshot` is pure (no clock, no PRNG, no IO).
* :meth:`recent_decisions` is O(1) per call (bounded ring buffer).
* ``state_snapshot`` keys subset
  ``registry/agent_state_keys.yaml#AGT-04-lp``.
* ``rationale_tags`` drawn from
  ``registry/agent_rationale_tags.yaml`` (introduces
  ``lp_quote_buy`` / ``lp_quote_sell`` /
  ``mean_reversion_neutral``).
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
class LiquidityProviderAgent(AgentBase, AgentIntrospection):
    """LP / market-making mean-reversion agent (AGT-04).

    Compares the latest mid against a rolling fair-value mean:

    * If ``(mid - fair) / fair * 1e4 < -band_bps`` → mean-reversion
      BUY (price dipped below band; provide liquidity at the bid).
    * If ``(mid - fair) / fair * 1e4 > +band_bps`` → mean-reversion
      SELL (price popped above band; provide liquidity at the ask).
    * Otherwise → HOLD (inside the no-quote band).
    """

    agent_id: str = "AGT-04-lp"
    fair_value_window: int = 16
    band_bps: float = 8.0
    min_confidence: float = 0.05
    ring_capacity: int = 64

    _mid_window: deque[float] = field(init=False, repr=False)
    _last_decision_direction: str = field(default=_HOLD, init=False, repr=False)
    _last_decision_confidence: float = field(default=0.0, init=False, repr=False)
    _last_decision_ts_ns: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.fair_value_window < 2:
            raise ValueError("fair_value_window must be >= 2")
        if self.band_bps < 0.0:
            raise ValueError("band_bps must be >= 0")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        AgentBase.__init__(self, self.agent_id, self.ring_capacity)
        self._mid_window = deque(maxlen=self.fair_value_window)

    # --- inputs ----------------------------------------------------

    def observe_tick(self, tick: MarketTick) -> None:
        if tick.bid <= 0.0 or tick.ask <= 0.0 or tick.ask < tick.bid:
            return
        mid = 0.5 * (tick.bid + tick.ask)
        if mid <= 0.0:
            return
        self._mid_window.append(mid)

    def _fair_value(self) -> float:
        items = list(self._mid_window)
        return sum(items) / float(len(items))

    def decide(self, signal: SignalEvent) -> AgentDecisionTrace:
        rationale: list[str] = []

        if len(self._mid_window) < self.fair_value_window:
            direction = _HOLD
            confidence = 0.0
            rationale.append("mean_reversion_neutral")
        else:
            fair = self._fair_value()
            if fair <= 0.0:
                direction = _HOLD
                confidence = 0.0
                rationale.append("book_invalid")
            else:
                latest = self._mid_window[-1]
                deviation_bps = (latest - fair) / fair * 10_000.0
                if deviation_bps < -self.band_bps:
                    # Price below fair → LP wants to bid → BUY stance.
                    if signal.side is Side.BUY:
                        direction = _BUY
                        rationale.append("lp_quote_buy")
                    else:
                        direction = _HOLD
                        rationale.append("lp_quote_buy")
                elif deviation_bps > self.band_bps:
                    if signal.side is Side.SELL:
                        direction = _SELL
                        rationale.append("lp_quote_sell")
                    else:
                        direction = _HOLD
                        rationale.append("lp_quote_sell")
                else:
                    direction = _HOLD
                    rationale.append("mean_reversion_neutral")

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
            "fair_value_window": str(self.fair_value_window),
            "band_threshold": f"{self.band_bps:.6f}",
        }


__all__ = ["LiquidityProviderAgent"]
