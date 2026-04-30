"""Tests for the composition engine + compatibility constraint table.

Covers (Wave-04 PR-4):

* Happy-path composition with and without a philosophy profile.
* Signature determinism (matches :func:`signature_for`).
* All five constraint blocklists (philosophy↔entry, philosophy↔exit,
  philosophy↔risk-attitude, horizon↔timeframe, regime↔entry).
* The belief-strength threshold filter (weak signals don't trip
  constraints).
* UNKNOWN-discriminator rejection.
* Multiple findings collected in one error.
* Error semantics (immutability, message carries every finding).
* Catalogue interplay (canonical components compose without errors).
"""

from __future__ import annotations

import pytest

from core.contracts.trader_intelligence import (
    ConvictionStyle,
    PhilosophyProfile,
    RiskAttitude,
    TimeHorizon,
)
from intelligence_engine.strategy_library import (
    BELIEF_THRESHOLD,
    CANONICAL_DECOMPOSITIONS,
    ComposedStrategy,
    EntryLogic,
    EntryStyle,
    ExitLogic,
    ExitStyle,
    IncompatibilityFinding,
    IncompatibilityReason,
    IncompatibleCompositionError,
    MarketCondition,
    MarketRegime,
    RiskModel,
    SizingStyle,
    StopStyle,
    Timeframe,
    compose,
    signature_for,
)

# ---------------------------------------------------------------------------
# Test fixtures — well-formed components
# ---------------------------------------------------------------------------


def _entry(style: EntryStyle = EntryStyle.PULLBACK) -> EntryLogic:
    return EntryLogic(component_id="entry_test", style=style)


def _exit(style: ExitStyle = ExitStyle.FIXED_TARGET) -> ExitLogic:
    return ExitLogic(component_id="exit_test", style=style)


def _risk(
    sizing: SizingStyle = SizingStyle.FIXED_RISK,
    stop: StopStyle = StopStyle.NORMAL,
) -> RiskModel:
    return RiskModel(
        component_id="risk_test",
        sizing=sizing,
        stop=stop,
        max_position_size_pct="1.0",
    )


def _timeframe(bar_interval: str = "5m") -> Timeframe:
    return Timeframe(
        component_id="timeframe_test",
        bar_interval=bar_interval,
        holding_period_bars="20",
    )


def _market(regime: MarketRegime = MarketRegime.ANY) -> MarketCondition:
    return MarketCondition(component_id="market_test", regime=regime)


def _philosophy(
    *,
    belief_system: dict[str, float] | None = None,
    risk_attitude: RiskAttitude = RiskAttitude.BALANCED,
    time_horizon: TimeHorizon = TimeHorizon.SWING,
    conviction_style: ConvictionStyle = ConvictionStyle.SYSTEMATIC,
) -> PhilosophyProfile:
    return PhilosophyProfile(
        trader_id="trader-test",
        belief_system=belief_system or {"trend_following": 0.7},
        risk_attitude=risk_attitude,
        time_horizon=time_horizon,
        conviction_style=conviction_style,
    )


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_compose_without_philosophy_returns_proposal() -> None:
    result = compose(
        composition_id="comp1",
        decomposition_id="decomp1",
        entry=_entry(),
        exit_=_exit(),
        risk=_risk(),
        timeframe=_timeframe(),
        market_condition=_market(),
    )
    assert isinstance(result, ComposedStrategy)
    assert result.composition_id == "comp1"
    assert result.decomposition.decomposition_id == "decomp1"
    assert result.philosophy is None
    assert result.notes == {}


def test_compose_with_philosophy_returns_proposal() -> None:
    phil = _philosophy()
    result = compose(
        composition_id="comp2",
        decomposition_id="decomp2",
        entry=_entry(EntryStyle.PULLBACK),
        exit_=_exit(),
        risk=_risk(),
        timeframe=_timeframe(),
        market_condition=_market(),
        philosophy=phil,
    )
    assert result.philosophy is phil


def test_compose_signature_matches_signature_for() -> None:
    """Composed signature must equal signature_for(decomposition).

    Structural seam: composed strategies have byte-identical signatures
    to hand-constructed StrategyDecompositions, so they can flow into
    TraderModel.strategy_signatures interchangeably.
    """
    result = compose(
        composition_id="comp",
        decomposition_id="decomp_sig",
        entry=_entry(),
        exit_=_exit(),
        risk=_risk(),
        timeframe=_timeframe(),
        market_condition=_market(),
    )
    assert result.signature == signature_for(result.decomposition)
    assert len(result.signature) == 64
    assert all(c in "0123456789abcdef" for c in result.signature)


def test_compose_notes_are_copied() -> None:
    notes = {"source": "manual"}
    result = compose(
        composition_id="comp",
        decomposition_id="decomp",
        entry=_entry(),
        exit_=_exit(),
        risk=_risk(),
        timeframe=_timeframe(),
        market_condition=_market(),
        notes=notes,
    )
    assert result.notes == notes
    notes["source"] = "mutated"
    assert result.notes == {"source": "manual"}


def test_compose_canonical_microstructure_v1_replay() -> None:
    """The PR-3 reference decomposition replays through compose() unchanged."""
    canonical = CANONICAL_DECOMPOSITIONS["microstructure_v1"]
    result = compose(
        composition_id="microstructure_v1_proposal",
        decomposition_id=canonical.decomposition_id,
        entry=canonical.entry,
        exit_=canonical.exit_,
        risk=canonical.risk,
        timeframe=canonical.timeframe,
        market_condition=canonical.market_condition,
    )
    assert result.signature == signature_for(canonical)


# ---------------------------------------------------------------------------
# Belief vs entry
# ---------------------------------------------------------------------------


def test_mean_reversion_belief_blocks_breakout_entry() -> None:
    phil = _philosophy(belief_system={"mean_reversion": 0.9})
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.BREAKOUT),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    reasons = {f.reason for f in exc.value.findings}
    assert IncompatibilityReason.PHILOSOPHY_VS_ENTRY_STYLE in reasons


def test_mean_reversion_belief_blocks_momentum_entry() -> None:
    phil = _philosophy(belief_system={"mean_reversion": 0.8})
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.MOMENTUM),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.PHILOSOPHY_VS_ENTRY_STYLE
        for f in exc.value.findings
    )


def test_trend_following_belief_blocks_mean_reversion_entry() -> None:
    phil = _philosophy(belief_system={"trend_following": 0.9})
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.MEAN_REVERSION),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.PHILOSOPHY_VS_ENTRY_STYLE
        for f in exc.value.findings
    )


def test_volatility_premium_belief_blocks_breakout_entry() -> None:
    phil = _philosophy(belief_system={"volatility_premium": 0.7})
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.BREAKOUT),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.PHILOSOPHY_VS_ENTRY_STYLE
        for f in exc.value.findings
    )


def test_weak_belief_does_not_block() -> None:
    """Belief strength below BELIEF_THRESHOLD is treated as noise."""
    weak = BELIEF_THRESHOLD - 0.01
    phil = _philosophy(belief_system={"mean_reversion": weak})
    result = compose(
        composition_id="c",
        decomposition_id="d",
        entry=_entry(EntryStyle.BREAKOUT),
        exit_=_exit(),
        risk=_risk(),
        timeframe=_timeframe(),
        market_condition=_market(),
        philosophy=phil,
    )
    assert isinstance(result, ComposedStrategy)


def test_belief_threshold_constant_is_documented() -> None:
    """Pin the threshold so changing it without test churn is impossible."""
    assert BELIEF_THRESHOLD == 0.5


# ---------------------------------------------------------------------------
# Belief vs exit
# ---------------------------------------------------------------------------


def test_mean_reversion_belief_blocks_trailing_stop_exit() -> None:
    phil = _philosophy(belief_system={"mean_reversion": 0.9})
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.PULLBACK),
            exit_=_exit(ExitStyle.TRAILING_STOP),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.PHILOSOPHY_VS_EXIT_STYLE
        for f in exc.value.findings
    )


# ---------------------------------------------------------------------------
# Risk attitude vs stop
# ---------------------------------------------------------------------------


def test_conservative_with_no_stop_blocked() -> None:
    phil = _philosophy(risk_attitude=RiskAttitude.CONSERVATIVE)
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(stop=StopStyle.NONE),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.PHILOSOPHY_VS_RISK_ATTITUDE
        for f in exc.value.findings
    )


def test_conservative_with_wide_stop_blocked() -> None:
    phil = _philosophy(risk_attitude=RiskAttitude.CONSERVATIVE)
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(stop=StopStyle.WIDE),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.PHILOSOPHY_VS_RISK_ATTITUDE
        for f in exc.value.findings
    )


def test_leveraged_with_wide_stop_blocked() -> None:
    phil = _philosophy(risk_attitude=RiskAttitude.LEVERAGED)
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(stop=StopStyle.WIDE),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.PHILOSOPHY_VS_RISK_ATTITUDE
        for f in exc.value.findings
    )


# ---------------------------------------------------------------------------
# Horizon vs timeframe
# ---------------------------------------------------------------------------


def test_scalp_horizon_blocked_on_daily_bars() -> None:
    phil = _philosophy(time_horizon=TimeHorizon.SCALP)
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(bar_interval="1d"),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.HORIZON_VS_TIMEFRAME
        for f in exc.value.findings
    )


def test_scalp_horizon_blocked_on_hourly_bars() -> None:
    phil = _philosophy(time_horizon=TimeHorizon.SCALP)
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(bar_interval="1h"),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.HORIZON_VS_TIMEFRAME
        for f in exc.value.findings
    )


def test_macro_horizon_blocked_on_tick_bars() -> None:
    phil = _philosophy(time_horizon=TimeHorizon.MACRO)
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(bar_interval="1s"),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.HORIZON_VS_TIMEFRAME
        for f in exc.value.findings
    )


def test_intraday_horizon_blocked_on_daily_bars() -> None:
    phil = _philosophy(time_horizon=TimeHorizon.INTRADAY)
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(bar_interval="1d"),
            market_condition=_market(),
            philosophy=phil,
        )
    assert any(
        f.reason is IncompatibilityReason.HORIZON_VS_TIMEFRAME
        for f in exc.value.findings
    )


# ---------------------------------------------------------------------------
# Regime vs entry (philosophy-free)
# ---------------------------------------------------------------------------


def test_trending_regime_blocks_mean_reversion_entry() -> None:
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.MEAN_REVERSION),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(MarketRegime.TRENDING),
        )
    assert any(
        f.reason is IncompatibilityReason.REGIME_VS_ENTRY_STYLE
        for f in exc.value.findings
    )


def test_ranging_regime_blocks_breakout_entry() -> None:
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.BREAKOUT),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(MarketRegime.RANGING),
        )
    assert any(
        f.reason is IncompatibilityReason.REGIME_VS_ENTRY_STYLE
        for f in exc.value.findings
    )


def test_ranging_regime_blocks_momentum_entry() -> None:
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.MOMENTUM),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(MarketRegime.RANGING),
        )
    assert any(
        f.reason is IncompatibilityReason.REGIME_VS_ENTRY_STYLE
        for f in exc.value.findings
    )


# ---------------------------------------------------------------------------
# UNKNOWN discriminators
# ---------------------------------------------------------------------------


def test_unknown_entry_style_rejected() -> None:
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.UNKNOWN),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
        )
    assert any(
        f.reason is IncompatibilityReason.UNKNOWN_DISCRIMINATOR
        for f in exc.value.findings
    )


def test_unknown_exit_style_rejected() -> None:
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(ExitStyle.UNKNOWN),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
        )
    assert any(
        f.reason is IncompatibilityReason.UNKNOWN_DISCRIMINATOR
        for f in exc.value.findings
    )


def test_unknown_stop_style_rejected() -> None:
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(),
            exit_=_exit(),
            risk=_risk(stop=StopStyle.UNKNOWN),
            timeframe=_timeframe(),
            market_condition=_market(),
        )
    assert any(
        f.reason is IncompatibilityReason.UNKNOWN_DISCRIMINATOR
        for f in exc.value.findings
    )


# ---------------------------------------------------------------------------
# Multiple findings collected at once
# ---------------------------------------------------------------------------


def test_multiple_findings_all_collected() -> None:
    """A composition that violates 3 constraints must surface all 3."""
    phil = _philosophy(
        belief_system={"mean_reversion": 0.9},
        risk_attitude=RiskAttitude.CONSERVATIVE,
        time_horizon=TimeHorizon.SCALP,
    )
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.BREAKOUT),  # vs belief + vs regime
            exit_=_exit(ExitStyle.TRAILING_STOP),  # vs belief
            risk=_risk(stop=StopStyle.NONE),  # vs risk_attitude
            timeframe=_timeframe(bar_interval="1d"),  # vs horizon
            market_condition=_market(MarketRegime.RANGING),
            philosophy=phil,
        )
    reasons = {f.reason for f in exc.value.findings}
    assert IncompatibilityReason.PHILOSOPHY_VS_ENTRY_STYLE in reasons
    assert IncompatibilityReason.PHILOSOPHY_VS_EXIT_STYLE in reasons
    assert IncompatibilityReason.PHILOSOPHY_VS_RISK_ATTITUDE in reasons
    assert IncompatibilityReason.HORIZON_VS_TIMEFRAME in reasons
    assert IncompatibilityReason.REGIME_VS_ENTRY_STYLE in reasons


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


def test_error_message_carries_all_findings() -> None:
    phil = _philosophy(belief_system={"mean_reversion": 0.9})
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.BREAKOUT),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    msg = str(exc.value)
    assert "composition rejected" in msg
    assert "PHILOSOPHY_VS_ENTRY_STYLE" in msg


def test_error_findings_tuple_is_immutable() -> None:
    phil = _philosophy(belief_system={"mean_reversion": 0.9})
    with pytest.raises(IncompatibleCompositionError) as exc:
        compose(
            composition_id="c",
            decomposition_id="d",
            entry=_entry(EntryStyle.BREAKOUT),
            exit_=_exit(),
            risk=_risk(),
            timeframe=_timeframe(),
            market_condition=_market(),
            philosophy=phil,
        )
    findings = exc.value.findings
    assert isinstance(findings, tuple)
    assert all(isinstance(f, IncompatibilityFinding) for f in findings)


def test_error_requires_at_least_one_finding() -> None:
    with pytest.raises(ValueError, match="at least one finding"):
        IncompatibleCompositionError(())


def test_finding_is_frozen() -> None:
    f = IncompatibilityFinding(
        reason=IncompatibilityReason.UNKNOWN_DISCRIMINATOR,
        detail="x",
    )
    with pytest.raises((AttributeError, TypeError)):
        f.detail = "mutated"  # type: ignore[misc]


def test_composed_strategy_is_frozen() -> None:
    result = compose(
        composition_id="c",
        decomposition_id="d",
        entry=_entry(),
        exit_=_exit(),
        risk=_risk(),
        timeframe=_timeframe(),
        market_condition=_market(),
    )
    with pytest.raises((AttributeError, TypeError)):
        result.composition_id = "mutated"  # type: ignore[misc]
