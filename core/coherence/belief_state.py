"""Belief State — read-only projection of system regime + market view.

Phase 6.T1a — first half. Companion to
:mod:`core.coherence.performance_pressure`.

Design (v3.1 §B1, v3.3 §1.2):

* :class:`BeliefState` is a frozen dataclass; all fields are derived
  from inputs only. There is no setter, no in-place mutation, and no
  feedback path. Other modules consume the snapshot read-only.
* :func:`derive_belief_state` is a pure function — given the same
  inputs in the same order it always returns the same output (replay
  determinism, INV-15).
* The snapshot is emitted to the ledger via :meth:`BeliefState.to_event`
  so that the offline calibrator (INV-53) can score belief-vs-reality
  per window.

Authority constraints:

* No imports from any ``*_engine`` package.
* No imports from ``state.ledger`` writers.
* Only :mod:`core.contracts` is imported.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from core.contracts.events import Side, SignalEvent, SystemEvent, SystemEventKind

# Module version — bumped when the projection function changes shape.
# Recorded in every snapshot payload so the calibrator can disambiguate
# windows produced by different derivation versions.
BELIEF_STATE_VERSION = "v3.3-T1a"


class Regime(StrEnum):
    """Regime label set produced by :func:`derive_belief_state`.

    The label set is intentionally small at T1a. Phase 6.T1e
    (``regime_router`` activation, INV-49 hysteresis) is the consumer
    that collapses these into actionable buckets.
    """

    UNKNOWN = "UNKNOWN"
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    VOL_SPIKE = "VOL_SPIKE"


@dataclass(frozen=True, slots=True)
class BeliefState:
    """Read-only system view of regime + market context.

    Fields:
        ts_ns: Snapshot timestamp (nanoseconds, TimeAuthority).
        regime: Current regime label (see :class:`Regime`).
        regime_confidence: Confidence of the regime classification,
            ``[0.0, 1.0]``.
        consensus_side: Majority side across the input signal window
            (``HOLD`` if no signals or the BUY/SELL counts tie).
        signal_count: Number of input signals contributing.
        avg_confidence: Mean confidence across input signals,
            ``[0.0, 1.0]``.
        symbols: Distinct symbols seen in the window, sorted (stable).
        version: Derivation version (``BELIEF_STATE_VERSION``).
    """

    ts_ns: int
    regime: Regime
    regime_confidence: float
    consensus_side: Side
    signal_count: int
    avg_confidence: float
    symbols: tuple[str, ...] = ()
    version: str = BELIEF_STATE_VERSION

    def to_event(self, source: str = "core.coherence.belief_state") -> SystemEvent:
        """Project the snapshot into a ledgerable :class:`SystemEvent`.

        The resulting event is the **only** way the BeliefState reaches
        any other engine — there is no in-memory cross-engine handle.
        Calibrator (INV-53) consumes this from the ledger.
        """
        payload: Mapping[str, str] = {
            "regime": self.regime.value,
            "regime_confidence": f"{self.regime_confidence:.6f}",
            "consensus_side": self.consensus_side.value,
            "signal_count": str(self.signal_count),
            "avg_confidence": f"{self.avg_confidence:.6f}",
            "symbols": ",".join(self.symbols),
            "version": self.version,
        }
        return SystemEvent(
            ts_ns=self.ts_ns,
            sub_kind=SystemEventKind.BELIEF_STATE_SNAPSHOT,
            source=source,
            payload=payload,
        )


def _consensus_side(signals: Sequence[SignalEvent]) -> Side:
    """Pure majority vote across signal sides; ties → :attr:`Side.HOLD`."""
    counts: dict[Side, int] = {Side.BUY: 0, Side.SELL: 0, Side.HOLD: 0}
    for s in signals:
        counts[s.side] = counts.get(s.side, 0) + 1
    buy = counts[Side.BUY]
    sell = counts[Side.SELL]
    if buy == 0 and sell == 0:
        return Side.HOLD
    if buy > sell:
        return Side.BUY
    if sell > buy:
        return Side.SELL
    return Side.HOLD


def _classify_regime(
    signals: Sequence[SignalEvent],
    *,
    vol_spike_z: float,
) -> tuple[Regime, float]:
    """Deterministic regime classification.

    The classifier is intentionally **rule-based** and cheap — Phase
    6.T1a does not yet ship a learned model. The rules are:

    * No signals → ``UNKNOWN`` at confidence 0.0.
    * Volatility z-score above ``vol_spike_z`` → ``VOL_SPIKE`` at the
      caller-supplied vol confidence.
    * Strong directional consensus (≥80% one side) → ``TREND_UP`` or
      ``TREND_DOWN`` proportional to the consensus fraction.
    * Otherwise → ``RANGE`` with confidence equal to the inverse of
      the BUY/SELL spread.
    """
    n = len(signals)
    if n == 0:
        return Regime.UNKNOWN, 0.0

    if vol_spike_z >= 3.0:
        # Vol confidence is bounded so a runaway z does not produce
        # a confidence above 1.0 — preserves [0, 1] domain.
        conf = min(1.0, (vol_spike_z - 3.0) / 5.0 + 0.6)
        return Regime.VOL_SPIKE, conf

    buy = sum(1 for s in signals if s.side is Side.BUY)
    sell = sum(1 for s in signals if s.side is Side.SELL)
    directional = buy + sell
    if directional == 0:
        return Regime.RANGE, 1.0  # only HOLD signals → strong range

    buy_frac = buy / directional
    sell_frac = sell / directional
    if buy_frac >= 0.8:
        return Regime.TREND_UP, buy_frac
    if sell_frac >= 0.8:
        return Regime.TREND_DOWN, sell_frac

    spread = abs(buy_frac - sell_frac)
    return Regime.RANGE, max(0.0, 1.0 - spread)


def derive_belief_state(
    *,
    ts_ns: int,
    signals: Sequence[SignalEvent],
    vol_spike_z: float = 0.0,
) -> BeliefState:
    """Pure derivation of :class:`BeliefState` from a signal window.

    Inputs:
        ts_ns: Snapshot timestamp.
        signals: Window of recent :class:`SignalEvent`s. Order is
            preserved but does not affect determinism — the function
            uses only commutative aggregations (counts / averages /
            sorted unique symbol set).
        vol_spike_z: Externally-computed volatility z-score for the
            window. Caller (Phase 6.T1a wiring) sources this from
            ``intelligence_engine.signal_pipeline``. Default 0.0
            during T1a stand-alone tests.

    Returns:
        A frozen :class:`BeliefState` snapshot.
    """
    n = len(signals)
    if n == 0:
        avg_confidence = 0.0
    else:
        avg_confidence = sum(s.confidence for s in signals) / n

    regime, regime_confidence = _classify_regime(
        signals,
        vol_spike_z=vol_spike_z,
    )
    consensus = _consensus_side(signals)
    symbols = tuple(sorted({s.symbol for s in signals}))

    return BeliefState(
        ts_ns=ts_ns,
        regime=regime,
        regime_confidence=regime_confidence,
        consensus_side=consensus,
        signal_count=n,
        avg_confidence=avg_confidence,
        symbols=symbols,
    )


__all__ = [
    "BELIEF_STATE_VERSION",
    "BeliefState",
    "Regime",
    "derive_belief_state",
]


# Silence ``F401`` on the ``field`` re-export used by future T1a follow-ons
# (kept available so the module mirrors the style of ``core.contracts.events``
# without an import churn when extended).
_ = field
