"""Frozen registry of canonical components + reference decompositions.

The registry is **append-only by convention**: removing a component
ID changes any signature hash that referenced it (the hash is over
the full component, not the ID), but it also breaks the contract
:class:`core.contracts.trader_intelligence.TraderModel.strategy_signatures`
relies on. Adding a new component is safe and does not affect any
existing signatures.

The first canonical decomposition — ``microstructure_v1`` — mirrors
the live :class:`intelligence_engine.plugins.microstructure.microstructure_v1.MicrostructureV1`
plugin field-for-field. Its presence here is the structural seam that
lets the composition engine (Wave-04 PR-4) recombine
``microstructure_v1.entry`` with another strategy's ``risk`` model
without coupling to the plugin module itself.

INV-15: every value here is constructed from deterministic primitives;
re-importing this module produces byte-identical objects across runs.
"""

from __future__ import annotations

from types import MappingProxyType

from intelligence_engine.strategy_library.components import (
    EntryLogic,
    EntryStyle,
    ExitLogic,
    ExitStyle,
    MarketCondition,
    MarketRegime,
    RiskModel,
    SizingStyle,
    StopStyle,
    Timeframe,
)
from intelligence_engine.strategy_library.decomposition import (
    StrategyDecomposition,
)

# ---------------------------------------------------------------------------
# EntryLogic catalogue
# ---------------------------------------------------------------------------

_ENTRY_LOGIC: dict[str, EntryLogic] = {
    "midpoint_deviation_v1": EntryLogic(
        component_id="midpoint_deviation_v1",
        style=EntryStyle.MICROSTRUCTURE,
        parameters=MappingProxyType(
            {
                "tolerance_bps": "2.0",
                "confidence_scale_bps": "50.0",
            }
        ),
    ),
    "breakout_v1": EntryLogic(
        component_id="breakout_v1",
        style=EntryStyle.BREAKOUT,
        parameters=MappingProxyType(
            {
                "lookback_bars": "20",
                "confirmation_bars": "1",
            }
        ),
    ),
    "pullback_v1": EntryLogic(
        component_id="pullback_v1",
        style=EntryStyle.PULLBACK,
        parameters=MappingProxyType(
            {
                "trend_lookback_bars": "50",
                "retrace_pct": "38.2",
            }
        ),
    ),
    "mean_reversion_zscore_v1": EntryLogic(
        component_id="mean_reversion_zscore_v1",
        style=EntryStyle.MEAN_REVERSION,
        parameters=MappingProxyType(
            {
                "lookback_bars": "100",
                "z_threshold": "2.0",
            }
        ),
    ),
}

CANONICAL_ENTRY_LOGIC: MappingProxyType[str, EntryLogic] = MappingProxyType(
    _ENTRY_LOGIC
)

# ---------------------------------------------------------------------------
# ExitLogic catalogue
# ---------------------------------------------------------------------------

_EXIT_LOGIC: dict[str, ExitLogic] = {
    "signal_reversal_v1": ExitLogic(
        component_id="signal_reversal_v1",
        style=ExitStyle.SIGNAL_REVERSAL,
        parameters=MappingProxyType({}),
    ),
    "fixed_target_v1": ExitLogic(
        component_id="fixed_target_v1",
        style=ExitStyle.FIXED_TARGET,
        parameters=MappingProxyType(
            {
                "target_bps": "20.0",
            }
        ),
    ),
    "trailing_stop_v1": ExitLogic(
        component_id="trailing_stop_v1",
        style=ExitStyle.TRAILING_STOP,
        parameters=MappingProxyType(
            {
                "trail_bps": "10.0",
            }
        ),
    ),
    "time_stop_5m_v1": ExitLogic(
        component_id="time_stop_5m_v1",
        style=ExitStyle.TIME_STOP,
        parameters=MappingProxyType(
            {
                "max_holding_bars": "5",
            }
        ),
    ),
}

CANONICAL_EXIT_LOGIC: MappingProxyType[str, ExitLogic] = MappingProxyType(
    _EXIT_LOGIC
)

# ---------------------------------------------------------------------------
# RiskModel catalogue
# ---------------------------------------------------------------------------

_RISK_MODELS: dict[str, RiskModel] = {
    "observe_zero_size_v1": RiskModel(
        component_id="observe_zero_size_v1",
        sizing=SizingStyle.FIXED_NOTIONAL,
        stop=StopStyle.NONE,
        max_position_size_pct="0.0",
        parameters=MappingProxyType({}),
    ),
    "fixed_risk_tight_v1": RiskModel(
        component_id="fixed_risk_tight_v1",
        sizing=SizingStyle.FIXED_RISK,
        stop=StopStyle.TIGHT,
        max_position_size_pct="0.5",
        parameters=MappingProxyType(
            {
                "risk_per_trade_pct": "0.25",
            }
        ),
    ),
    "fixed_risk_normal_v1": RiskModel(
        component_id="fixed_risk_normal_v1",
        sizing=SizingStyle.FIXED_RISK,
        stop=StopStyle.NORMAL,
        max_position_size_pct="1.0",
        parameters=MappingProxyType(
            {
                "risk_per_trade_pct": "0.5",
            }
        ),
    ),
    "vol_target_wide_v1": RiskModel(
        component_id="vol_target_wide_v1",
        sizing=SizingStyle.VOLATILITY_TARGET,
        stop=StopStyle.WIDE,
        max_position_size_pct="2.0",
        parameters=MappingProxyType(
            {
                "annualised_vol_target_pct": "10.0",
            }
        ),
    ),
}

CANONICAL_RISK_MODELS: MappingProxyType[str, RiskModel] = MappingProxyType(
    _RISK_MODELS
)

# ---------------------------------------------------------------------------
# Timeframe catalogue
# ---------------------------------------------------------------------------

_TIMEFRAMES: dict[str, Timeframe] = {
    "tick_scalp_v1": Timeframe(
        component_id="tick_scalp_v1",
        bar_interval="1s",
        holding_period_bars="1",
    ),
    "intraday_5m_v1": Timeframe(
        component_id="intraday_5m_v1",
        bar_interval="5m",
        holding_period_bars="20",
    ),
    "swing_1h_v1": Timeframe(
        component_id="swing_1h_v1",
        bar_interval="1h",
        holding_period_bars="48",
    ),
    "position_1d_v1": Timeframe(
        component_id="position_1d_v1",
        bar_interval="1d",
        holding_period_bars="20",
    ),
}

CANONICAL_TIMEFRAMES: MappingProxyType[str, Timeframe] = MappingProxyType(
    _TIMEFRAMES
)

# ---------------------------------------------------------------------------
# MarketCondition catalogue
# ---------------------------------------------------------------------------

_MARKET_CONDITIONS: dict[str, MarketCondition] = {
    "any_liquid_v1": MarketCondition(
        component_id="any_liquid_v1",
        regime=MarketRegime.ANY,
        symbol_universe=(),
        notes=MappingProxyType(
            {
                "min_avg_daily_volume_usd": "1e7",
            }
        ),
    ),
    "trending_only_v1": MarketCondition(
        component_id="trending_only_v1",
        regime=MarketRegime.TRENDING,
        symbol_universe=(),
        notes=MappingProxyType({}),
    ),
    "ranging_only_v1": MarketCondition(
        component_id="ranging_only_v1",
        regime=MarketRegime.RANGING,
        symbol_universe=(),
        notes=MappingProxyType({}),
    ),
    "high_vol_event_v1": MarketCondition(
        component_id="high_vol_event_v1",
        regime=MarketRegime.HIGH_VOL,
        symbol_universe=(),
        notes=MappingProxyType(
            {
                "requires_event_window": "true",
            }
        ),
    ),
}

CANONICAL_MARKET_CONDITIONS: MappingProxyType[str, MarketCondition] = (
    MappingProxyType(_MARKET_CONDITIONS)
)

# ---------------------------------------------------------------------------
# Reference decompositions
# ---------------------------------------------------------------------------
#
# ``microstructure_v1`` mirrors
# :class:`intelligence_engine.plugins.microstructure.microstructure_v1.MicrostructureV1`
# with a zero-size, no-stop risk model. The plugin is currently the
# *only* live intelligence plugin in the system; once Wave-04 PR-4
# wires the composition engine, additional decompositions get minted
# by recombining catalogued components, not by hand-writing them here.

_DECOMPOSITIONS: dict[str, StrategyDecomposition] = {
    "microstructure_v1": StrategyDecomposition(
        decomposition_id="microstructure_v1",
        entry=_ENTRY_LOGIC["midpoint_deviation_v1"],
        exit_=_EXIT_LOGIC["signal_reversal_v1"],
        risk=_RISK_MODELS["observe_zero_size_v1"],
        timeframe=_TIMEFRAMES["tick_scalp_v1"],
        market_condition=_MARKET_CONDITIONS["any_liquid_v1"],
    ),
}

CANONICAL_DECOMPOSITIONS: MappingProxyType[str, StrategyDecomposition] = (
    MappingProxyType(_DECOMPOSITIONS)
)


__all__ = [
    "CANONICAL_DECOMPOSITIONS",
    "CANONICAL_ENTRY_LOGIC",
    "CANONICAL_EXIT_LOGIC",
    "CANONICAL_MARKET_CONDITIONS",
    "CANONICAL_RISK_MODELS",
    "CANONICAL_TIMEFRAMES",
]
