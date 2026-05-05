"""Tests for SIM-16 order_book_decay."""

from __future__ import annotations

from typing import Any

import pytest

from core.contracts.simulation import RealityOutcome, RealityScenario
from simulation.order_book_decay import OrderBookDecay, OrderBookDecayConfig


def _scenario(
    *,
    scenario_id: str = "S",
    reference_price: float = 100.0,
    order_size_usd: float = 50_000.0,
    side: str = "buy",
    num_levels: int = 10,
    level_spacing_bps: float = 1.0,
    level_depth_usd: float = 10_000.0,
    decay_rate: float = 0.5,
    elapsed_seconds: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> RealityScenario:
    meta: dict[str, Any] = {
        "reference_price": reference_price,
        "order_size_usd": order_size_usd,
        "side": side,
        "num_levels": num_levels,
        "level_spacing_bps": level_spacing_bps,
        "level_depth_usd": level_depth_usd,
        "decay_rate": decay_rate,
        "elapsed_seconds": elapsed_seconds,
    }
    if extra:
        meta.update(extra)
    return RealityScenario(
        scenario_id=scenario_id,
        ts_ns=42,
        initial_state_hash="h",
        meta=meta,
    )


def test_replay_determinism() -> None:
    s = _scenario()
    a = OrderBookDecay().step(seed=7, scenario=s)
    b = OrderBookDecay().step(seed=7, scenario=s)
    assert a == b


def test_pnl_always_non_positive() -> None:
    runner = OrderBookDecay()
    for k in range(40):
        s = _scenario(side="buy" if k % 2 == 0 else "sell")
        out = runner.step(seed=k, scenario=s)
        assert out.pnl_usd <= 0.0
        assert out.terminal_drawdown_usd >= 0.0
        assert out.terminal_drawdown_usd == pytest.approx(-out.pnl_usd)


def test_zero_decay_zero_jitter_is_deepest_book() -> None:
    # decay=0 means every level is fully alive. With 10 levels of 10k
    # depth each = 100k available. 50k order should sweep through ~5
    # levels. Cost is the per-level offset times eaten size.
    s = _scenario(decay_rate=0.0, elapsed_seconds=0.0)
    out = OrderBookDecay().step(seed=0, scenario=s)
    assert out.rule_fired in ("buy_fully_swept", "sell_fully_swept")
    assert out.fills_count >= 1


def test_higher_decay_higher_cost() -> None:
    runner = OrderBookDecay()
    # Order small enough that both books fully sweep — we want to
    # compare *which* levels are touched, not whether the sweep
    # completes. At decay=0.05/elapsed=10 the inside still has ~6k
    # USD alive (cheap level 0). At decay=0.5/elapsed=10 only the
    # outer levels survive, so the same 5k order has to walk to
    # higher offsets and pay more.
    s_low = _scenario(
        decay_rate=0.05, elapsed_seconds=10.0, order_size_usd=5_000.0
    )
    s_high = _scenario(
        decay_rate=0.5, elapsed_seconds=10.0, order_size_usd=5_000.0
    )
    cost_low = sum(
        -runner.step(seed=k, scenario=s_low).pnl_usd for k in range(20)
    )
    cost_high = sum(
        -runner.step(seed=k, scenario=s_high).pnl_usd for k in range(20)
    )
    assert cost_high > cost_low


def test_book_too_thin_when_decay_consumes_inventory() -> None:
    # Decay 8 over 100 seconds is essentially total annihilation;
    # only the deepest level (rate ~ 0) survives, but its capacity
    # (10_000 USD) is far less than the 50_000 USD order.
    s = _scenario(decay_rate=8.0, elapsed_seconds=100.0)
    out = OrderBookDecay().step(seed=0, scenario=s)
    assert out.rule_fired in ("buy_book_too_thin", "sell_book_too_thin")


def test_more_levels_more_capacity() -> None:
    runner = OrderBookDecay()
    s_thin = _scenario(num_levels=2, level_depth_usd=5_000.0)
    s_deep = _scenario(num_levels=20, level_depth_usd=5_000.0)
    out_thin = runner.step(seed=0, scenario=s_thin)
    out_deep = runner.step(seed=0, scenario=s_deep)
    assert out_thin.fills_count <= out_deep.fills_count


def test_outcome_round_trip() -> None:
    s = _scenario(scenario_id="round-trip-16")
    out = OrderBookDecay().step(seed=3, scenario=s)
    assert isinstance(out, RealityOutcome)
    assert out.scenario_id == "round-trip-16"
    assert out.seed == 3


def test_invalid_seed_rejected() -> None:
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=-1, scenario=_scenario())


def test_invalid_side_rejected() -> None:
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(side="long"))


def test_invalid_num_levels_rejected() -> None:
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(num_levels=0))
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(num_levels=-1))
    with pytest.raises(ValueError):
        OrderBookDecay().step(
            seed=0,
            scenario=_scenario(extra={"num_levels": 1.5}),  # type: ignore[arg-type]
        )


def test_num_levels_above_cap_rejected() -> None:
    cfg = OrderBookDecayConfig(max_levels=5)
    with pytest.raises(ValueError):
        OrderBookDecay(cfg).step(seed=0, scenario=_scenario(num_levels=10))


def test_invalid_spacing_or_depth_rejected() -> None:
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(level_spacing_bps=0.0))
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(level_spacing_bps=200.0))
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(level_depth_usd=-1.0))


def test_invalid_decay_or_elapsed_rejected() -> None:
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(decay_rate=-0.5))
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(decay_rate=20.0))
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(elapsed_seconds=-1.0))
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(elapsed_seconds=1e6))


def test_invalid_reference_or_size_rejected() -> None:
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(reference_price=0.0))
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=_scenario(order_size_usd=-1.0))


def test_missing_meta_keys_rejected() -> None:
    bad = RealityScenario(
        scenario_id="X",
        ts_ns=1,
        initial_state_hash="h",
        meta={"reference_price": 100.0},
    )
    with pytest.raises(ValueError):
        OrderBookDecay().step(seed=0, scenario=bad)


def test_nan_inputs_rejected() -> None:
    nan = float("nan")
    for key in (
        "reference_price",
        "order_size_usd",
        "level_spacing_bps",
        "level_depth_usd",
        "decay_rate",
        "elapsed_seconds",
    ):
        with pytest.raises(ValueError):
            OrderBookDecay().step(seed=0, scenario=_scenario(extra={key: nan}))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        OrderBookDecayConfig(max_levels=0)
    with pytest.raises(ValueError):
        OrderBookDecayConfig(residual_epsilon_usd=-0.01)
    with pytest.raises(ValueError):
        OrderBookDecayConfig(residual_epsilon_usd=2.0)


def test_distribution_over_seeds_varies() -> None:
    s = _scenario()
    runner = OrderBookDecay()
    pnls = {
        runner.step(seed=k, scenario=s).pnl_usd for k in range(50)
    }
    # 5% jitter over 50 seeds must produce > 1 distinct outcome.
    assert len(pnls) >= 5


def test_inside_decays_faster_than_outside() -> None:
    # Validate the inside-fast / outside-slow invariant by comparing
    # the rule_fired outcome at a decay/elapsed setting tuned so:
    #   - a deep book (many levels) survives because its outer
    #     levels barely decay, and
    #   - a shallow book (few levels) collapses entirely because
    #     all of its levels are in the inside-decay zone.
    runner = OrderBookDecay()
    s_shallow = _scenario(
        num_levels=2,
        level_depth_usd=10_000.0,
        decay_rate=2.0,
        elapsed_seconds=8.0,
        order_size_usd=10_000.0,
    )
    s_deep = _scenario(
        num_levels=40,
        level_depth_usd=10_000.0,
        decay_rate=2.0,
        elapsed_seconds=8.0,
        order_size_usd=10_000.0,
    )
    out_shallow = runner.step(seed=0, scenario=s_shallow)
    out_deep = runner.step(seed=0, scenario=s_deep)
    # The shallow book is entirely in the inside-decay zone and
    # cannot satisfy the order; the deep book has many outer levels
    # surviving and does fully sweep.
    assert out_shallow.rule_fired.endswith("book_too_thin")
    assert out_deep.rule_fired.endswith("fully_swept")
