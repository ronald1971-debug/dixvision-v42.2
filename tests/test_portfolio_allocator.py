"""Unit tests for ``intelligence_engine.portfolio.allocator``."""

from __future__ import annotations

import dataclasses

import pytest

from core.contracts.portfolio import (
    AllocationCandidate,
    AllocationDecision,
    ExposureSnapshot,
)
from intelligence_engine.portfolio import (
    PortfolioAllocator,
    PortfolioAllocatorConfig,
    load_portfolio_allocator_config,
)


def _cand(
    cid: str,
    *,
    symbol: str = "BTC-USD",
    archetype: str = "TA-TREND-001",
    confidence: float = 0.5,
    side: str = "BUY",
) -> AllocationCandidate:
    return AllocationCandidate(
        candidate_id=cid,
        symbol=symbol,
        archetype_id=archetype,
        confidence=confidence,
        side=side,
    )


def _empty_exposure(ts_ns: int = 1_700_000_000_000_000_000) -> ExposureSnapshot:
    return ExposureSnapshot(ts_ns=ts_ns, by_symbol={})


def _engine() -> PortfolioAllocator:
    return PortfolioAllocator(load_portfolio_allocator_config())


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def test_allocation_candidate_rejects_bad_inputs() -> None:
    base = _cand("c1")
    with pytest.raises(ValueError):
        dataclasses.replace(base, candidate_id="")
    with pytest.raises(ValueError):
        dataclasses.replace(base, symbol="")
    with pytest.raises(ValueError):
        dataclasses.replace(base, archetype_id="")
    with pytest.raises(ValueError):
        dataclasses.replace(base, confidence=1.5)
    with pytest.raises(ValueError):
        dataclasses.replace(base, side="HOLD")


def test_allocation_decision_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        AllocationDecision(candidate_id="", share=0.5, rule_fired="ok")
    with pytest.raises(ValueError):
        AllocationDecision(candidate_id="c", share=1.5, rule_fired="ok")
    with pytest.raises(ValueError):
        AllocationDecision(candidate_id="c", share=0.5, rule_fired="")


def test_exposure_snapshot_rejects_empty_symbol() -> None:
    with pytest.raises(ValueError):
        ExposureSnapshot(ts_ns=1, by_symbol={"": 100.0})
    with pytest.raises(ValueError):
        ExposureSnapshot(ts_ns=0, by_symbol={"BTC-USD": 100.0})


def test_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        PortfolioAllocatorConfig(
            confidence_floor=1.5,
            max_symbol_notional_usd=10_000.0,
            max_total_share=0.5,
        )
    with pytest.raises(ValueError):
        PortfolioAllocatorConfig(
            confidence_floor=0.2,
            max_symbol_notional_usd=-1.0,
            max_total_share=0.5,
        )
    with pytest.raises(ValueError):
        PortfolioAllocatorConfig(
            confidence_floor=0.2,
            max_symbol_notional_usd=10_000.0,
            max_total_share=1.5,
        )


def test_config_rejects_nan_max_symbol_notional() -> None:
    """Regression: NaN must not pass `<= 0.0` and silently disable allocation.

    Devin Review on PR #234 noted that ``self.max_symbol_notional_usd <= 0.0``
    is False for NaN under IEEE 754, so a config row containing ``.nan``
    (parsed by ``yaml.safe_load`` into ``float('nan')``) would be accepted
    and then cause the symbol-cap headroom calculation to clamp every
    allocation to 0.0 — silently rejecting every candidate. The validator
    is now phrased as ``not (x > 0.0)`` so NaN, -inf, 0.0 and negatives are
    all rejected at construction time.
    """

    with pytest.raises(ValueError, match="max_symbol_notional_usd"):
        PortfolioAllocatorConfig(
            confidence_floor=0.2,
            max_symbol_notional_usd=float("nan"),
            max_total_share=0.5,
        )
    with pytest.raises(ValueError, match="max_symbol_notional_usd"):
        PortfolioAllocatorConfig(
            confidence_floor=0.2,
            max_symbol_notional_usd=float("-inf"),
            max_total_share=0.5,
        )
    with pytest.raises(ValueError, match="max_symbol_notional_usd"):
        PortfolioAllocatorConfig(
            confidence_floor=0.2,
            max_symbol_notional_usd=0.0,
            max_total_share=0.5,
        )


def test_load_portfolio_allocator_config_from_registry() -> None:
    cfg = load_portfolio_allocator_config()
    assert isinstance(cfg, PortfolioAllocatorConfig)
    assert cfg.max_symbol_notional_usd > 0.0
    assert 0.0 < cfg.max_total_share <= 1.0


# ---------------------------------------------------------------------------
# Allocator behaviour
# ---------------------------------------------------------------------------


def test_below_floor_candidates_get_zero_share() -> None:
    eng = _engine()
    floor = eng.config.confidence_floor
    decisions = eng.allocate(
        [_cand("c1", confidence=floor - 0.01)],
        _empty_exposure(),
        available_capital_usd=10_000.0,
    )
    assert decisions == (
        AllocationDecision(candidate_id="c1", share=0.0, rule_fired="below_floor"),
    )


def test_single_candidate_gets_full_max_total_share() -> None:
    eng = _engine()
    decisions = eng.allocate(
        [_cand("c1", confidence=0.9)],
        _empty_exposure(),
        available_capital_usd=10_000.0,
    )
    assert len(decisions) == 1
    assert decisions[0].candidate_id == "c1"
    assert decisions[0].rule_fired == "ok"
    assert decisions[0].share == pytest.approx(eng.config.max_total_share)


def test_two_candidates_split_proportional_to_confidence() -> None:
    eng = _engine()
    decisions = eng.allocate(
        [
            _cand("c1", symbol="BTC-USD", confidence=0.9),
            _cand("c2", symbol="ETH-USD", confidence=0.3),
        ],
        _empty_exposure(),
        available_capital_usd=10_000.0,
    )
    by_id = {d.candidate_id: d for d in decisions}
    total = by_id["c1"].share + by_id["c2"].share
    assert total == pytest.approx(eng.config.max_total_share)
    assert by_id["c1"].share == pytest.approx(
        (0.9 / 1.2) * eng.config.max_total_share
    )
    assert by_id["c2"].share == pytest.approx(
        (0.3 / 1.2) * eng.config.max_total_share
    )


def test_symbol_cap_rejects_when_room_zero() -> None:
    eng = _engine()
    # Symbol already at full cap -> next candidate must be rejected.
    snap = ExposureSnapshot(
        ts_ns=1, by_symbol={"BTC-USD": eng.config.max_symbol_notional_usd}
    )
    decisions = eng.allocate(
        [_cand("c1", confidence=0.9, symbol="BTC-USD")],
        snap,
        available_capital_usd=10_000.0,
    )
    assert decisions[0].rule_fired == "symbol_cap_rejected"
    assert decisions[0].share == 0.0


def test_symbol_cap_clamps_when_partial_room() -> None:
    eng = _engine()
    # Half the cap is used, requesting more than half -> clamp.
    half = eng.config.max_symbol_notional_usd / 2.0
    snap = ExposureSnapshot(ts_ns=1, by_symbol={"BTC-USD": half})
    # Only one candidate, so it would otherwise get max_total_share
    # of 1_000_000 capital, which dwarfs the residual cap room.
    decisions = eng.allocate(
        [_cand("c1", confidence=0.9, symbol="BTC-USD")],
        snap,
        available_capital_usd=1_000_000.0,
    )
    assert decisions[0].rule_fired == "symbol_cap_clamped"
    assert decisions[0].share > 0.0
    # Resulting notional must equal residual room.
    notional = decisions[0].share * 1_000_000.0
    assert notional == pytest.approx(half)


def test_zero_capital_yields_no_capital_rule() -> None:
    eng = _engine()
    decisions = eng.allocate(
        [_cand("c1", confidence=0.9)],
        _empty_exposure(),
        available_capital_usd=0.0,
    )
    assert decisions[0].rule_fired == "no_capital"
    assert decisions[0].share == 0.0


def test_negative_capital_rejected() -> None:
    eng = _engine()
    with pytest.raises(ValueError):
        eng.allocate(
            [_cand("c1", confidence=0.9)], _empty_exposure(), -1.0
        )


def test_empty_candidates_returns_empty() -> None:
    eng = _engine()
    decisions = eng.allocate([], _empty_exposure(), 10_000.0)
    assert decisions == ()


def test_decisions_sorted_by_candidate_id() -> None:
    eng = _engine()
    decisions = eng.allocate(
        [
            _cand("c-z", symbol="A", confidence=0.9),
            _cand("c-a", symbol="B", confidence=0.9),
            _cand("c-m", symbol="C", confidence=0.9),
        ],
        _empty_exposure(),
        10_000.0,
    )
    ids = [d.candidate_id for d in decisions]
    assert ids == sorted(ids)


def test_replay_determinism_same_input_same_output() -> None:
    eng = _engine()
    cands = [
        _cand("c1", symbol="BTC-USD", confidence=0.7),
        _cand("c2", symbol="ETH-USD", confidence=0.4),
        _cand("c3", symbol="SOL-USD", confidence=0.9),
    ]
    snap = ExposureSnapshot(ts_ns=1, by_symbol={"BTC-USD": 10_000.0})
    a = eng.allocate(cands, snap, 50_000.0)
    b = eng.allocate(cands, snap, 50_000.0)
    assert a == b
