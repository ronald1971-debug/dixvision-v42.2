"""B-12 finrl_env — authority pins + correctness + INV-15 determinism."""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import pytest

from learning_engine.lanes import finrl_env
from learning_engine.lanes.finrl_env import (
    MAX_ASSETS,
    NEW_PIP_DEPENDENCIES,
    Bar,
    EpisodeConfig,
    EpisodeNotStartedError,
    FinRLEnvError,
    FinRLPortfolioEnv,
    PortfolioAction,
    PortfolioObservation,
    StepResult,
    observation_to_tuple,
    run_episode,
)

MODULE_PATH = Path(finrl_env.__file__)
MODULE_SOURCE = MODULE_PATH.read_text()
MODULE_AST = ast.parse(MODULE_SOURCE)


# ---------------------------------------------------------------- AST pins


def test_authority_adapted_from_header() -> None:
    assert MODULE_SOURCE.startswith("# ADAPTED FROM: AI4Finance-Foundation/FinRL")


def test_authority_no_finrl_import() -> None:
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "finrl" not in alias.name.lower()
        elif isinstance(node, ast.ImportFrom):
            assert node.module is None or "finrl" not in node.module.lower()


def test_authority_no_runtime_imports() -> None:
    forbidden = {
        "pandas",
        "numpy",
        "polars",
        "torch",
        "scipy",
        "gym",
        "gymnasium",
        "stable_baselines3",
        "cleanrl",
        "elegantrl",
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "socket",
        "secrets",
        "uuid",
        "requests",
        "httpx",
        "aiohttp",
        "websockets",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in forbidden, f"forbidden import: {node.module}"


def test_authority_no_engine_cross_imports() -> None:
    forbidden_roots = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
        "intelligence_engine",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root not in forbidden_roots, f"forbidden engine import: {node.module}"


def test_authority_no_typed_event_construction() -> None:
    forbidden_call_names = {
        "SignalEvent",
        "ExecutionIntent",
        "HazardEvent",
        "GovernanceDecision",
        "PatchProposal",
        "TradeOutcome",
    }
    for node in ast.walk(MODULE_AST):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden_call_names:
                raise AssertionError(f"forbidden typed event construction: {name}")


def test_authority_no_top_level_io() -> None:
    forbidden = {"open", "print", "input", "exec", "eval"}
    for node in MODULE_AST.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                raise AssertionError(f"forbidden top-level call: {func.id}")


def test_authority_pip_dependencies_empty() -> None:
    assert NEW_PIP_DEPENDENCIES == ()


def test_authority_module_reimport_clean() -> None:
    mod = importlib.import_module("learning_engine.lanes.finrl_env")
    assert mod is finrl_env


# ---------------------------------------------------------------- Bar value object


def test_bar_basic_ok() -> None:
    bar = Bar(ts_ns=1_000, symbol="AAPL", close=150.0)
    assert bar.symbol == "AAPL"
    assert bar.close == 150.0


def test_bar_rejects_negative_ts() -> None:
    with pytest.raises(FinRLEnvError):
        Bar(ts_ns=-1, symbol="AAPL", close=150.0)


def test_bar_rejects_empty_symbol() -> None:
    with pytest.raises(FinRLEnvError):
        Bar(ts_ns=0, symbol="", close=150.0)


def test_bar_rejects_nonpositive_close() -> None:
    with pytest.raises(FinRLEnvError):
        Bar(ts_ns=0, symbol="AAPL", close=0.0)
    with pytest.raises(FinRLEnvError):
        Bar(ts_ns=0, symbol="AAPL", close=-1.0)


def test_bar_rejects_nan_close() -> None:
    with pytest.raises(FinRLEnvError):
        Bar(ts_ns=0, symbol="AAPL", close=float("nan"))


def test_bar_rejects_negative_volume() -> None:
    with pytest.raises(FinRLEnvError):
        Bar(ts_ns=0, symbol="AAPL", close=150.0, volume=-1.0)


def test_bar_is_frozen() -> None:
    bar = Bar(ts_ns=0, symbol="AAPL", close=150.0)
    with pytest.raises((AttributeError, TypeError)):
        bar.close = 200.0  # type: ignore[misc]


# ---------------------------------------------------------------- EpisodeConfig


def test_episode_config_ok() -> None:
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"))
    assert cfg.symbols == ("AAPL", "MSFT")
    assert cfg.initial_cash == 1_000_000.0
    assert cfg.hmax == 100


def test_episode_config_rejects_empty_symbols() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=())


def test_episode_config_rejects_duplicate_symbols() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("AAPL", "AAPL"))


def test_episode_config_rejects_unsorted_symbols() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("MSFT", "AAPL"))


def test_episode_config_rejects_too_many_symbols() -> None:
    syms = tuple(f"S{i:05d}" for i in range(MAX_ASSETS + 1))
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=syms)


def test_episode_config_rejects_zero_initial_cash() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("AAPL",), initial_cash=0.0)


def test_episode_config_rejects_negative_hmax() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("AAPL",), hmax=0)


def test_episode_config_rejects_out_of_range_cost() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("AAPL",), buy_cost_pct=1.0)
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("AAPL",), sell_cost_pct=-0.001)


def test_episode_config_rejects_non_finite_reward_scaling() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("AAPL",), reward_scaling=float("inf"))


def test_episode_config_rejects_zero_max_steps() -> None:
    with pytest.raises(FinRLEnvError):
        EpisodeConfig(symbols=("AAPL",), max_steps=0)


# ---------------------------------------------------------------- PortfolioAction


def test_action_ok() -> None:
    a = PortfolioAction(targets=(0.5, -0.25))
    assert a.targets == (0.5, -0.25)


def test_action_rejects_empty() -> None:
    with pytest.raises(FinRLEnvError):
        PortfolioAction(targets=())


def test_action_rejects_nan() -> None:
    with pytest.raises(FinRLEnvError):
        PortfolioAction(targets=(float("nan"),))


def test_action_rejects_out_of_range() -> None:
    with pytest.raises(FinRLEnvError):
        PortfolioAction(targets=(1.5,))
    with pytest.raises(FinRLEnvError):
        PortfolioAction(targets=(-1.5,))


def test_action_is_frozen() -> None:
    a = PortfolioAction(targets=(0.5,))
    with pytest.raises((AttributeError, TypeError)):
        a.targets = (0.1,)  # type: ignore[misc]


# ----------------------------------------------- _validate_bars guard rails (via constructor)


def _two_step_aapl_msft_bars() -> tuple[Bar, ...]:
    return (
        Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
        Bar(ts_ns=1_000, symbol="MSFT", close=200.0),
        Bar(ts_ns=2_000, symbol="AAPL", close=110.0),
        Bar(ts_ns=2_000, symbol="MSFT", close=190.0),
    )


def test_env_rejects_empty_bars() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",))
    with pytest.raises(FinRLEnvError):
        FinRLPortfolioEnv(bars=(), config=cfg)


def test_env_rejects_unknown_symbol_in_bars() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",))
    bars = (Bar(ts_ns=1_000, symbol="GOOG", close=100.0),)
    with pytest.raises(FinRLEnvError):
        FinRLPortfolioEnv(bars=bars, config=cfg)


def test_env_rejects_duplicate_bar_for_symbol() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",))
    bars = (
        Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
        Bar(ts_ns=1_000, symbol="AAPL", close=101.0),
    )
    with pytest.raises(FinRLEnvError):
        FinRLPortfolioEnv(bars=bars, config=cfg)


def test_env_rejects_missing_symbol_for_timestamp() -> None:
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"))
    bars = (Bar(ts_ns=1_000, symbol="AAPL", close=100.0),)
    with pytest.raises(FinRLEnvError):
        FinRLPortfolioEnv(bars=bars, config=cfg)


def test_env_rejects_non_bar_entries() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",))
    with pytest.raises(TypeError):
        FinRLPortfolioEnv(bars=("not a bar",), config=cfg)  # type: ignore[arg-type]


def test_env_rejects_non_config() -> None:
    with pytest.raises(TypeError):
        FinRLPortfolioEnv(bars=(), config="not a config")  # type: ignore[arg-type]


# ---------------------------------------------------------------- reset / step contract


def test_reset_returns_initial_observation() -> None:
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"), initial_cash=10_000.0)
    env = FinRLPortfolioEnv(bars=_two_step_aapl_msft_bars(), config=cfg)
    obs, info = env.reset(seed=42)
    assert isinstance(obs, PortfolioObservation)
    assert obs.cash == 10_000.0
    assert obs.holdings == (0, 0)
    assert obs.prices == (100.0, 200.0)
    assert obs.portfolio_value == 10_000.0
    assert info["seed"] == "42"
    assert info["n_assets"] == "2"
    assert info["n_bars"] == "2"


def test_step_before_reset_raises() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",))
    env = FinRLPortfolioEnv(bars=(Bar(ts_ns=0, symbol="AAPL", close=100.0),), config=cfg)
    with pytest.raises(EpisodeNotStartedError):
        env.step(PortfolioAction(targets=(0.0,)))


def test_step_rejects_wrong_action_length() -> None:
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"))
    env = FinRLPortfolioEnv(bars=_two_step_aapl_msft_bars(), config=cfg)
    env.reset(seed=0)
    with pytest.raises(FinRLEnvError):
        env.step(PortfolioAction(targets=(0.0,)))


def test_step_rejects_non_action() -> None:
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"))
    env = FinRLPortfolioEnv(bars=_two_step_aapl_msft_bars(), config=cfg)
    env.reset(seed=0)
    with pytest.raises(TypeError):
        env.step((0.0, 0.0))  # type: ignore[arg-type]


def test_step_returns_5_tuple_shape() -> None:
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"), initial_cash=100_000.0)
    env = FinRLPortfolioEnv(bars=_two_step_aapl_msft_bars(), config=cfg)
    env.reset(seed=0)
    result = env.step(PortfolioAction(targets=(0.5, 0.0)))
    obs, reward, terminated, truncated, info = result.as_tuple()
    assert isinstance(obs, PortfolioObservation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------- Trading semantics


def test_buy_then_hold_updates_holdings_and_cash() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL",),
        initial_cash=100_000.0,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=110.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    result = env.step(PortfolioAction(targets=(1.0,)))
    # hmax=100, action=+1 -> buy 100 shares at $100 = $10,000 outlay.
    assert result.observation.holdings == (100,)
    assert result.observation.cash == pytest.approx(90_000.0)
    # New close is $110 -> portfolio_value = 90,000 + 100*110 = 101,000
    assert result.observation.portfolio_value == pytest.approx(101_000.0)


def test_buy_cost_reduces_purchasing_power() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL",),
        initial_cash=10_100.0,
        hmax=100,
        buy_cost_pct=0.01,
        sell_cost_pct=0.0,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    result = env.step(PortfolioAction(targets=(1.0,)))
    # Each share costs 100*(1.01) = 101; cash=10,100 affords 100 shares
    # exactly. Spending 100*101 = 10,100.
    assert result.observation.holdings == (100,)
    assert result.observation.cash == pytest.approx(0.0, abs=1e-6)


def test_buy_clipped_by_cash() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL",),
        initial_cash=500.0,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    result = env.step(PortfolioAction(targets=(1.0,)))
    # Wanted 100 shares; can afford only 5.
    assert result.observation.holdings == (5,)
    assert result.observation.cash == pytest.approx(0.0, abs=1e-6)


def test_sell_only_what_we_hold() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL",),
        initial_cash=100_000.0,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=3_000, symbol="AAPL", close=100.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    env.step(PortfolioAction(targets=(0.5,)))  # buy 50 shares
    result = env.step(PortfolioAction(targets=(-1.0,)))  # try to sell 100
    # We had 50; cannot oversell.
    assert result.observation.holdings == (0,)


def test_sell_cost_reduces_proceeds() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL",),
        initial_cash=10_000.0,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.01,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=3_000, symbol="AAPL", close=100.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    env.step(PortfolioAction(targets=(1.0,)))  # buy 100 shares for 10,000
    result = env.step(PortfolioAction(targets=(-1.0,)))  # sell 100 shares
    # Gross 100*100 = 10,000; cost 100 -> net 9,900 added back to cash.
    assert result.observation.holdings == (0,)
    assert result.observation.cash == pytest.approx(9_900.0)


def test_sells_free_cash_for_buys_in_same_step() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL", "MSFT"),
        initial_cash=0.0 + 1.0,  # almost no starting cash
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=1_000, symbol="MSFT", close=200.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="MSFT", close=200.0),
            Bar(ts_ns=3_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=3_000, symbol="MSFT", close=200.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    # Force a holding of MSFT via a fake constructor path: buy MSFT first.
    # First step: starting cash is ~1, no buy fits. Skip directly.
    # Instead, take cfg2 with bigger cash.
    cfg2 = EpisodeConfig(
        symbols=("AAPL", "MSFT"),
        initial_cash=20_000.0,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
    )
    env2 = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=1_000, symbol="MSFT", close=200.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="MSFT", close=200.0),
            Bar(ts_ns=3_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=3_000, symbol="MSFT", close=200.0),
        ),
        config=cfg2,
    )
    env2.reset(seed=0)
    # Step 1: buy 100 MSFT -> 100*200 = 20,000 outlay.
    env2.step(PortfolioAction(targets=(0.0, 1.0)))
    # Step 2: sell all MSFT (frees 20,000) THEN buy 100 AAPL (10,000).
    result = env2.step(PortfolioAction(targets=(1.0, -1.0)))
    assert result.observation.holdings == (100, 0)
    assert result.observation.cash == pytest.approx(10_000.0)


def test_zero_action_holds_position() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL",),
        initial_cash=10_000.0,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=120.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    result = env.step(PortfolioAction(targets=(0.0,)))
    assert result.observation.holdings == (0,)
    assert result.observation.cash == 10_000.0


# ---------------------------------------------------------------- Reward / termination


def test_reward_is_delta_portfolio_value_scaled() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL",),
        initial_cash=10_000.0,
        hmax=100,
        buy_cost_pct=0.0,
        sell_cost_pct=0.0,
        reward_scaling=2.0,
    )
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=110.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    result = env.step(PortfolioAction(targets=(1.0,)))
    # Step 0: bought 100 shares at 100 = 10,000 outlay, cash=0
    # Step 1: holdings_value 100*110 = 11,000 + cash 0 = 11,000
    # Delta vs initial 10,000 = +1,000; scaled by 2.0 = 2,000.
    assert result.reward == pytest.approx(2_000.0)


def test_terminated_at_last_bar() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",), initial_cash=10_000.0)
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    r1 = env.step(PortfolioAction(targets=(0.0,)))
    assert r1.terminated is False
    r2 = env.step(PortfolioAction(targets=(0.0,)))
    assert r2.terminated is True
    assert r2.truncated is False


def test_truncated_at_max_steps() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",), initial_cash=10_000.0, max_steps=2)
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=3_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=4_000, symbol="AAPL", close=100.0),
        ),
        config=cfg,
    )
    env.reset(seed=0)
    env.step(PortfolioAction(targets=(0.0,)))
    r = env.step(PortfolioAction(targets=(0.0,)))
    assert r.terminated is False
    assert r.truncated is True


# ---------------------------------------------------------------- observation_to_tuple flatten


def test_observation_to_tuple_layout() -> None:
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"), initial_cash=10_000.0)
    env = FinRLPortfolioEnv(bars=_two_step_aapl_msft_bars(), config=cfg)
    obs, _ = env.reset(seed=0)
    flat = observation_to_tuple(obs)
    # [cash, holdings_AAPL, holdings_MSFT, price_AAPL, price_MSFT]
    assert flat == (10_000.0, 0.0, 0.0, 100.0, 200.0)


# ---------------------------------------------------------------- INV-15 determinism


def test_replay_byte_identical_three_runs() -> None:
    cfg = EpisodeConfig(
        symbols=("AAPL", "MSFT"),
        initial_cash=100_000.0,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
    )
    actions = (
        PortfolioAction(targets=(0.5, -0.5)),
        PortfolioAction(targets=(1.0, 0.0)),
        PortfolioAction(targets=(-1.0, 0.5)),
    )

    def replay() -> tuple[PortfolioObservation, ...]:
        env = FinRLPortfolioEnv(
            bars=_two_step_aapl_msft_bars()
            + (
                Bar(ts_ns=3_000, symbol="AAPL", close=120.0),
                Bar(ts_ns=3_000, symbol="MSFT", close=180.0),
                Bar(ts_ns=4_000, symbol="AAPL", close=130.0),
                Bar(ts_ns=4_000, symbol="MSFT", close=170.0),
            ),
            config=cfg,
        )
        return run_episode(env, actions, seed=99)

    r1 = replay()
    r2 = replay()
    r3 = replay()
    assert r1 == r2 == r3


def test_replay_bar_order_independent() -> None:
    """Bars in different input order produce identical observations."""
    cfg = EpisodeConfig(symbols=("AAPL", "MSFT"), initial_cash=50_000.0)
    bars_a = _two_step_aapl_msft_bars()
    bars_b = (
        _two_step_aapl_msft_bars()[3],
        _two_step_aapl_msft_bars()[0],
        _two_step_aapl_msft_bars()[2],
        _two_step_aapl_msft_bars()[1],
    )
    actions = (PortfolioAction(targets=(0.25, 0.0)),)
    env_a = FinRLPortfolioEnv(bars=bars_a, config=cfg)
    env_b = FinRLPortfolioEnv(bars=bars_b, config=cfg)
    obs_a = run_episode(env_a, actions, seed=0)
    obs_b = run_episode(env_b, actions, seed=0)
    assert obs_a == obs_b


def test_replay_seed_independent_when_no_random() -> None:
    """No PRNG -> different seeds produce identical trajectories given identical actions."""
    cfg = EpisodeConfig(symbols=("AAPL",), initial_cash=10_000.0)
    bars = (
        Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
        Bar(ts_ns=2_000, symbol="AAPL", close=110.0),
    )
    actions = (PortfolioAction(targets=(0.5,)),)
    env1 = FinRLPortfolioEnv(bars=bars, config=cfg)
    env2 = FinRLPortfolioEnv(bars=bars, config=cfg)
    obs1 = run_episode(env1, actions, seed=1)
    obs2 = run_episode(env2, actions, seed=999_999)
    assert obs1 == obs2


# ---------------------------------------------------------------- run_episode helper


def test_run_episode_stops_at_termination() -> None:
    cfg = EpisodeConfig(symbols=("AAPL",), initial_cash=10_000.0)
    env = FinRLPortfolioEnv(
        bars=(
            Bar(ts_ns=1_000, symbol="AAPL", close=100.0),
            Bar(ts_ns=2_000, symbol="AAPL", close=110.0),
        ),
        config=cfg,
    )
    actions = tuple(PortfolioAction(targets=(0.0,)) for _ in range(10))
    traj = run_episode(env, actions, seed=0)
    # Initial obs + at most 2 step obs (env terminates after 1 step, since
    # there are only 2 bars and step 1 reaches the last bar).
    assert len(traj) <= 3
    assert traj[0].step_idx == 0


# ---------------------------------------------------------------- StepResult value object


def test_step_result_rejects_non_finite_reward() -> None:
    obs = PortfolioObservation(
        step_idx=0,
        ts_ns=0,
        cash=0.0,
        holdings=(),
        prices=(),
        portfolio_value=0.0,
    )
    with pytest.raises(FinRLEnvError):
        StepResult(
            observation=obs,
            reward=math.nan,
            terminated=False,
            truncated=False,
        )


def test_step_result_info_must_be_str_str() -> None:
    obs = PortfolioObservation(
        step_idx=0,
        ts_ns=0,
        cash=0.0,
        holdings=(),
        prices=(),
        portfolio_value=0.0,
    )
    with pytest.raises(TypeError):
        StepResult(
            observation=obs,
            reward=0.0,
            terminated=False,
            truncated=False,
            info={"k": 123},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------- PortfolioObservation guards


def test_observation_rejects_mismatched_lengths() -> None:
    with pytest.raises(FinRLEnvError):
        PortfolioObservation(
            step_idx=0,
            ts_ns=0,
            cash=0.0,
            holdings=(1, 2),
            prices=(100.0,),
            portfolio_value=0.0,
        )


def test_observation_rejects_negative_holdings() -> None:
    with pytest.raises(FinRLEnvError):
        PortfolioObservation(
            step_idx=0,
            ts_ns=0,
            cash=0.0,
            holdings=(-1,),
            prices=(100.0,),
            portfolio_value=0.0,
        )


def test_observation_rejects_nonpositive_prices() -> None:
    with pytest.raises(FinRLEnvError):
        PortfolioObservation(
            step_idx=0,
            ts_ns=0,
            cash=0.0,
            holdings=(0,),
            prices=(0.0,),
            portfolio_value=0.0,
        )
