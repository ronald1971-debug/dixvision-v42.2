"""Strategy decomposition primitives — five value-object components.

Each component is a frozen, slotted dataclass with structural
equality. The fields are intentionally coarse — the goal is
*reusable, recombinable building blocks*, not faithful per-strategy
reproduction. Finer-grained parameter knobs ride on the
``parameters`` mapping each component carries.

The five components map directly onto the spec from the Wave-04
trader-intelligence brief:

* "entry logic" → :class:`EntryLogic`
* "exit logic" → :class:`ExitLogic`
* "risk model" → :class:`RiskModel`
* "timeframe" → :class:`Timeframe`
* "market conditions" → :class:`MarketCondition`

INV-15: every field is a deterministic primitive (StrEnum / str /
float / Mapping[str, str]). Float parameter values are stringified
before they enter :attr:`parameters` so byte-identical replay
(:func:`json.dumps(..., sort_keys=True, separators=(",", ":"))`) is
unaffected by float-formatting drift across runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Discriminator enums (StrEnum — replay-deterministic, not free-form text)
# ---------------------------------------------------------------------------


class EntryStyle(StrEnum):
    """How a position is opened."""

    BREAKOUT = "BREAKOUT"  # enter on confirmed range break
    PULLBACK = "PULLBACK"  # enter on retrace within a trend
    MEAN_REVERSION = "MEAN_REVERSION"  # fade overshoots back to mean
    MOMENTUM = "MOMENTUM"  # enter on accelerating move
    MICROSTRUCTURE = "MICROSTRUCTURE"  # bid/ask imbalance / mid-deviation
    NEWS_DRIVEN = "NEWS_DRIVEN"  # event-window entries
    CARRY = "CARRY"  # spread / funding harvest
    UNKNOWN = "UNKNOWN"


class ExitStyle(StrEnum):
    """How a position is closed."""

    FIXED_TARGET = "FIXED_TARGET"  # take-profit at price level
    TRAILING_STOP = "TRAILING_STOP"  # ratchet stop with adverse move
    TIME_STOP = "TIME_STOP"  # close after N bars
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"  # close when entry signal flips
    VOLATILITY_TARGET = "VOLATILITY_TARGET"  # close when realised vol exits band
    UNKNOWN = "UNKNOWN"


class StopStyle(StrEnum):
    """How protective stops are placed (subset of risk model)."""

    TIGHT = "TIGHT"  # ≤ 0.5% adverse move
    NORMAL = "NORMAL"  # ~1–2% adverse move
    WIDE = "WIDE"  # ≥ 3% adverse move
    NONE = "NONE"  # no hard stop (rare; e.g. carry)
    UNKNOWN = "UNKNOWN"


class SizingStyle(StrEnum):
    """How position size is computed."""

    FIXED_NOTIONAL = "FIXED_NOTIONAL"  # constant USD per trade
    FIXED_RISK = "FIXED_RISK"  # constant USD-at-risk per trade
    KELLY_FRACTION = "KELLY_FRACTION"  # fractional-Kelly on edge estimate
    VOLATILITY_TARGET = "VOLATILITY_TARGET"  # size to a portfolio vol budget
    UNKNOWN = "UNKNOWN"


class MarketRegime(StrEnum):
    """Coarse regime taxonomy the strategy is designed for."""

    TRENDING = "TRENDING"  # clear directional drift
    RANGING = "RANGING"  # mean-reverting, no drift
    HIGH_VOL = "HIGH_VOL"  # realised vol > long-run avg
    LOW_VOL = "LOW_VOL"  # realised vol < long-run avg
    EVENT_DRIVEN = "EVENT_DRIVEN"  # macro / earnings windows
    ANY = "ANY"  # regime-agnostic
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Five reusable components
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntryLogic:
    """When and why to open a position.

    Attributes:
        component_id: Stable handle (``"breakout_v1"``, ``"midpoint_deviation_v1"``,
            …). Used by the canonical registry; freeform IDs are allowed
            for offline experimentation but only registry-listed IDs flow
            into :class:`~core.contracts.trader_intelligence.TraderModel.strategy_signatures`.
        style: Coarse :class:`EntryStyle` discriminator.
        parameters: Component-specific knobs as ``str``-valued mapping
            (e.g. ``{"breakout_lookback_bars": "20",
            "tolerance_bps": "2.0"}``). Values are strings to preserve
            replay determinism — see module docstring.
    """

    component_id: str
    style: EntryStyle = EntryStyle.UNKNOWN
    parameters: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExitLogic:
    """When to close a position.

    Attributes:
        component_id: Stable handle.
        style: Coarse :class:`ExitStyle` discriminator.
        parameters: Component-specific knobs (``str``-valued).
    """

    component_id: str
    style: ExitStyle = ExitStyle.UNKNOWN
    parameters: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RiskModel:
    """How risk is sized + protected for one position.

    Attributes:
        component_id: Stable handle.
        sizing: Coarse :class:`SizingStyle` discriminator.
        stop: Coarse :class:`StopStyle` discriminator.
        max_position_size_pct: Hard cap as a percent of account equity
            (``"1.0"`` for 1%, etc.). Stored as a string for replay
            determinism.
        parameters: Component-specific knobs (``str``-valued).
    """

    component_id: str
    sizing: SizingStyle = SizingStyle.UNKNOWN
    stop: StopStyle = StopStyle.UNKNOWN
    max_position_size_pct: str = "0.0"
    parameters: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Timeframe:
    """Holding-period bucket + bar interval.

    Distinct from :class:`core.contracts.trader_intelligence.TimeHorizon`:
    that enum buckets a *trader's typical holding period* (a fact about
    the trader); this one buckets a *strategy's design timeframe* (a
    fact about the strategy). The two often align (a SCALP strategy is
    usually run by a SCALP-horizon trader) but the composition engine
    must reason about them separately.

    Attributes:
        component_id: Stable handle.
        bar_interval: Canonical bar size — ``"1s"``, ``"1m"``, ``"5m"``,
            ``"1h"``, ``"1d"``. Free-form for forward-compatibility but
            registry-listed values are recommended.
        holding_period_bars: Typical position duration measured in
            ``bar_interval`` units, as a string (``"1"``, ``"20"``,
            ``"unbounded"``).
    """

    component_id: str
    bar_interval: str = "1s"
    holding_period_bars: str = "1"


@dataclass(frozen=True, slots=True)
class MarketCondition:
    """Regime / environment the strategy is designed to run in.

    Attributes:
        component_id: Stable handle.
        regime: Coarse :class:`MarketRegime` discriminator.
        symbol_universe: Tuple of symbol patterns the strategy applies
            to (``"BTCUSDT"``, ``"*USDT"``, ``"SPX_constituent"``, …).
            Empty tuple = applies to anything.
        notes: Free-form structural metadata (``"requires_l2_book"``,
            ``"funding_window_only"``, …).
    """

    component_id: str
    regime: MarketRegime = MarketRegime.ANY
    symbol_universe: tuple[str, ...] = ()
    notes: Mapping[str, str] = field(default_factory=dict)


__all__ = [
    "EntryLogic",
    "EntryStyle",
    "ExitLogic",
    "ExitStyle",
    "MarketCondition",
    "MarketRegime",
    "RiskModel",
    "SizingStyle",
    "StopStyle",
    "Timeframe",
]
