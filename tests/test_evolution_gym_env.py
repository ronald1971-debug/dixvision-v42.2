"""Tests for ``evolution_engine.gym_env`` — A-01.1 stable-baselines3 adapter.

Pinning the contract:

* Module metadata + AST authority pins (no top-level gymnasium /
  stable_baselines3 / numpy / asyncio / clock imports).
* Frozen value-object validators.
* :class:`DIXStrategyEnv.reset` / ``step`` API shape (Gymnasium 0.26+
  5-tuple).
* Episode-budget enforcement (max_steps + global ceiling).
* INV-15 byte-identical 3-run replay equality.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import inspect
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

env_mod = importlib.import_module("evolution_engine.gym_env")
DIXStrategyEnv = env_mod.DIXStrategyEnv
EpisodeBudgetExceededError = env_mod.EpisodeBudgetExceededError
EpisodeConfig = env_mod.EpisodeConfig
EpisodeNotStartedError = env_mod.EpisodeNotStartedError
MarketDynamics = env_mod.MarketDynamics
MAX_EPISODE_STEPS = env_mod.MAX_EPISODE_STEPS
MIN_INITIAL_NOTIONAL_USD = env_mod.MIN_INITIAL_NOTIONAL_USD
NEW_PIP_DEPENDENCIES = env_mod.NEW_PIP_DEPENDENCIES
Observation = env_mod.Observation
TradeAction = env_mod.TradeAction
Transition = env_mod.Transition

_MOD_PATH = Path(env_mod.__file__)


# ---------------------------------------------------------------------------
# Module metadata + AST authority pins
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_is_frozen_tuple() -> None:
    assert isinstance(NEW_PIP_DEPENDENCIES, tuple)
    assert NEW_PIP_DEPENDENCIES == ("gymnasium", "stable-baselines3")


def test_module_has_adapted_from_header() -> None:
    head = _MOD_PATH.read_text(encoding="utf-8").splitlines()[:6]
    assert any(line.startswith("# ADAPTED FROM:") for line in head), head


def test_max_episode_steps_is_finite_positive() -> None:
    assert isinstance(MAX_EPISODE_STEPS, int)
    assert MAX_EPISODE_STEPS > 0
    assert MAX_EPISODE_STEPS <= 1_000_000


def test_min_initial_notional_usd_is_positive() -> None:
    assert isinstance(MIN_INITIAL_NOTIONAL_USD, float)
    assert math.isfinite(MIN_INITIAL_NOTIONAL_USD)
    assert MIN_INITIAL_NOTIONAL_USD > 0.0


def _iter_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_top_level_gymnasium_import() -> None:
    """gymnasium MUST be lazy-imported only inside the optional factory."""

    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                top_level.append(node.module)
    forbidden = {"gymnasium", "gym", "stable_baselines3", "numpy", "torch"}
    assert not (set(top_level) & forbidden), (
        f"top-level imports leaked forbidden packages: {set(top_level) & forbidden}"
    )


def test_no_runtime_clock_or_io_imports_anywhere() -> None:
    """Authority pin: tier=OFFLINE but the env class is wall-clock-free
    and IO-free everywhere — INV-15 forbids hidden clock reads."""

    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    all_imports = set(_iter_imports(tree))
    forbidden = {
        "time",
        "datetime",
        "asyncio",
        "websockets",
        "os",
        "random",
        "secrets",
        "uuid",
    }
    assert not (all_imports & forbidden), f"forbidden imports: {all_imports & forbidden}"


def test_no_engine_cross_imports() -> None:
    src = _MOD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    all_imports = set(_iter_imports(tree))
    forbidden_prefixes = (
        "execution_engine.",
        "governance_engine.",
        "system_engine.",
        "intelligence_engine.",
        "registry.",
        "ui.",
    )
    leaked = {m for m in all_imports if m.startswith(forbidden_prefixes)}
    # The env is OFFLINE and leaf-pure: it must not import any other
    # engine. core.* imports would be allowed here but currently none
    # are needed.
    assert not leaked, f"engine cross-imports leaked: {leaked}"


def test_value_object_dataclasses_are_frozen_and_slotted() -> None:
    for cls in (Observation, Transition, EpisodeConfig):
        params = cls.__dataclass_params__  # type: ignore[attr-defined]
        assert params.frozen, f"{cls.__name__} must be frozen"
        assert getattr(cls, "__slots__", None) is not None, f"{cls.__name__} must declare __slots__"


def test_dixstrategyenv_uses_slots() -> None:
    assert getattr(DIXStrategyEnv, "__slots__", None) is not None
    # No __dict__ on instances — frozen runtime shape.
    sig = inspect.signature(DIXStrategyEnv.__init__)
    assert "dynamics" in sig.parameters


# ---------------------------------------------------------------------------
# TradeAction
# ---------------------------------------------------------------------------


def test_trade_action_values_match_gym_discrete_3_convention() -> None:
    assert int(TradeAction.HOLD) == 0
    assert int(TradeAction.BUY) == 1
    assert int(TradeAction.SELL) == 2
    assert {a.value for a in TradeAction} == {0, 1, 2}


# ---------------------------------------------------------------------------
# EpisodeConfig
# ---------------------------------------------------------------------------


def test_episode_config_happy_path() -> None:
    cfg = EpisodeConfig(initial_notional_usd=10_000.0, max_steps=128)
    assert cfg.initial_notional_usd == 10_000.0
    assert cfg.max_steps == 128
    assert cfg.reward_scale == 1.0
    assert cfg.drawdown_penalty_weight == 0.5


def test_episode_config_rejects_non_finite_initial_notional() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            EpisodeConfig(initial_notional_usd=bad, max_steps=10)


def test_episode_config_rejects_below_min_initial_notional() -> None:
    with pytest.raises(ValueError):
        EpisodeConfig(initial_notional_usd=0.0, max_steps=10)
    with pytest.raises(ValueError):
        EpisodeConfig(initial_notional_usd=0.5, max_steps=10)


def test_episode_config_rejects_non_positive_max_steps() -> None:
    with pytest.raises(ValueError):
        EpisodeConfig(initial_notional_usd=100.0, max_steps=0)
    with pytest.raises(ValueError):
        EpisodeConfig(initial_notional_usd=100.0, max_steps=-1)


def test_episode_config_rejects_max_steps_above_global_ceiling() -> None:
    with pytest.raises(ValueError):
        EpisodeConfig(
            initial_notional_usd=100.0,
            max_steps=MAX_EPISODE_STEPS + 1,
        )


def test_episode_config_rejects_non_positive_reward_scale() -> None:
    for bad in (0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            EpisodeConfig(
                initial_notional_usd=100.0,
                max_steps=10,
                reward_scale=bad,
            )


def test_episode_config_rejects_negative_drawdown_penalty_weight() -> None:
    with pytest.raises(ValueError):
        EpisodeConfig(
            initial_notional_usd=100.0,
            max_steps=10,
            drawdown_penalty_weight=-0.1,
        )
    with pytest.raises(ValueError):
        EpisodeConfig(
            initial_notional_usd=100.0,
            max_steps=10,
            drawdown_penalty_weight=math.nan,
        )


def test_episode_config_is_frozen() -> None:
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.max_steps = 20  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def _make_obs(**kwargs: Any) -> Observation:
    base: dict[str, Any] = {
        "step_idx": 0,
        "mid_price": 100.0,
        "inventory_signed": 0,
        "cumulative_pnl_usd": 0.0,
        "state_hash": "0123456789abcdef",
    }
    base.update(kwargs)
    return Observation(**base)


def test_observation_happy_path() -> None:
    obs = _make_obs()
    assert obs.step_idx == 0
    assert obs.mid_price == 100.0
    assert obs.inventory_signed == 0


def test_observation_rejects_negative_step_idx() -> None:
    with pytest.raises(ValueError):
        _make_obs(step_idx=-1)


def test_observation_rejects_non_positive_mid_price() -> None:
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            _make_obs(mid_price=bad)


def test_observation_rejects_invalid_inventory() -> None:
    for bad in (-2, 2, 5):
        with pytest.raises(ValueError):
            _make_obs(inventory_signed=bad)


def test_observation_rejects_non_finite_pnl() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            _make_obs(cumulative_pnl_usd=bad)


def test_observation_rejects_bad_hash_length() -> None:
    with pytest.raises(ValueError):
        _make_obs(state_hash="abc")
    with pytest.raises(ValueError):
        _make_obs(state_hash="0" * 32)


def test_observation_rejects_non_hex_hash() -> None:
    with pytest.raises(ValueError):
        _make_obs(state_hash="ZZZZZZZZZZZZZZZZ")
    with pytest.raises(ValueError):
        _make_obs(state_hash="0123456789ABCDEF")  # uppercase rejected


def test_observation_is_frozen() -> None:
    obs = _make_obs()
    with pytest.raises(dataclasses.FrozenInstanceError):
        obs.step_idx = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


def _make_transition(**kwargs: Any) -> Transition:
    base: dict[str, Any] = {
        "next_mid_price": 101.0,
        "realised_pnl_usd": 1.0,
        "drawdown_usd": 0.0,
        "next_inventory_signed": 1,
        "terminated": False,
        "truncated": False,
    }
    base.update(kwargs)
    return Transition(**base)


def test_transition_happy_path() -> None:
    t = _make_transition()
    assert t.next_mid_price == 101.0
    assert t.realised_pnl_usd == 1.0
    assert not t.terminated


def test_transition_rejects_non_positive_mid() -> None:
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            _make_transition(next_mid_price=bad)


def test_transition_rejects_non_finite_pnl() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            _make_transition(realised_pnl_usd=bad)


def test_transition_rejects_negative_drawdown() -> None:
    with pytest.raises(ValueError):
        _make_transition(drawdown_usd=-0.5)
    with pytest.raises(ValueError):
        _make_transition(drawdown_usd=math.nan)


def test_transition_rejects_invalid_inventory() -> None:
    with pytest.raises(ValueError):
        _make_transition(next_inventory_signed=2)
    with pytest.raises(ValueError):
        _make_transition(next_inventory_signed=-3)


def test_transition_is_frozen() -> None:
    t = _make_transition()
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.terminated = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Deterministic in-test dynamics
# ---------------------------------------------------------------------------


class _ConstantUpDynamics:
    """Toy dynamics: price drifts up by $1 per step regardless of action.

    BUY makes +1 per step (long captures drift). SELL makes -1 (short
    against drift). HOLD makes 0. Drawdown is always 0.
    """

    def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
        return 100.0

    def step(
        self,
        prev_obs: Observation,
        action: TradeAction,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> Transition:
        if action == TradeAction.BUY:
            pnl = 1.0
            inv = 1
        elif action == TradeAction.SELL:
            pnl = -1.0
            inv = -1
        else:
            pnl = 0.0
            inv = 0
        return Transition(
            next_mid_price=prev_obs.mid_price + 1.0,
            realised_pnl_usd=pnl,
            drawdown_usd=0.0,
            next_inventory_signed=inv,
            terminated=False,
            truncated=False,
        )


class _SeededDynamics:
    """Toy dynamics whose pnl depends on the per-step seed — used to
    pin INV-15 (same episode_seed → same per-step seeds → same
    transitions)."""

    def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
        return 50.0 + (seed % 7)

    def step(
        self,
        prev_obs: Observation,
        action: TradeAction,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> Transition:
        # Map seed to a deterministic small float in [-2, 2].
        bucket = (seed >> 7) & 0x3F  # 6-bit slice
        delta = (bucket / 63.0) * 4.0 - 2.0
        action_sign = {
            TradeAction.HOLD: 0,
            TradeAction.BUY: 1,
            TradeAction.SELL: -1,
        }[action]
        return Transition(
            next_mid_price=prev_obs.mid_price + delta * 0.1 + 0.001,
            realised_pnl_usd=delta * action_sign,
            drawdown_usd=max(0.0, -delta * action_sign),
            next_inventory_signed=action_sign,
            terminated=False,
            truncated=False,
        )


class _TerminatingDynamics:
    """Terminates after exactly N steps."""

    def __init__(self, terminate_after: int) -> None:
        self._terminate_after = terminate_after

    def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
        return 100.0

    def step(
        self,
        prev_obs: Observation,
        action: TradeAction,
        *,
        seed: int,
        config: EpisodeConfig,
    ) -> Transition:
        will_terminate = prev_obs.step_idx + 1 >= self._terminate_after
        return Transition(
            next_mid_price=prev_obs.mid_price + 1.0,
            realised_pnl_usd=0.5,
            drawdown_usd=0.1,
            next_inventory_signed=0,
            terminated=will_terminate,
            truncated=False,
        )


# ---------------------------------------------------------------------------
# DIXStrategyEnv: construction + reset + step
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_protocol_dynamics() -> None:
    with pytest.raises(TypeError):
        DIXStrategyEnv(dynamics="not-a-protocol")  # type: ignore[arg-type]


def test_action_space_n_matches_trade_action_count() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    assert env.action_space_n == 3


def test_observation_keys_are_stable_order() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    assert env.observation_keys == (
        "step_idx",
        "mid_price",
        "inventory_signed",
        "cumulative_pnl_usd",
    )


def test_reset_returns_initial_observation_and_info() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=10_000.0, max_steps=10)
    obs, info = env.reset(seed=42, config=cfg)
    assert obs.step_idx == 0
    assert obs.mid_price == 100.0
    assert obs.inventory_signed == 0
    assert obs.cumulative_pnl_usd == 0.0
    assert info["episode_seed"] == 42
    assert info["max_steps"] == 10


def test_reset_rejects_non_int_seed() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    with pytest.raises(TypeError):
        env.reset(seed="42", config=cfg)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        env.reset(seed=True, config=cfg)


def test_reset_rejects_negative_seed() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    with pytest.raises(ValueError):
        env.reset(seed=-1, config=cfg)


def test_reset_rejects_non_episode_config() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    with pytest.raises(TypeError):
        env.reset(seed=0, config={"max_steps": 10})  # type: ignore[arg-type]


def test_reset_rejects_dynamics_returning_non_positive_initial_mid() -> None:
    class _BadDyn:
        def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
            return -1.0

        def step(
            self,
            prev_obs: Observation,
            action: TradeAction,
            *,
            seed: int,
            config: EpisodeConfig,
        ) -> Transition:
            raise AssertionError("not reached")

    env = DIXStrategyEnv(_BadDyn())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    with pytest.raises(ValueError):
        env.reset(seed=0, config=cfg)


def test_step_before_reset_raises() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    with pytest.raises(EpisodeNotStartedError):
        env.step(TradeAction.HOLD)


def test_step_returns_canonical_5_tuple() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    env.reset(seed=0, config=cfg)
    out = env.step(TradeAction.BUY)
    assert isinstance(out, tuple) and len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(obs, Observation)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, Mapping)


def test_step_advances_pnl_and_step_idx() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    env.reset(seed=0, config=cfg)
    obs1, r1, *_ = env.step(TradeAction.BUY)
    obs2, r2, *_ = env.step(TradeAction.BUY)
    assert obs1.step_idx == 1 and obs2.step_idx == 2
    assert obs1.cumulative_pnl_usd == 1.0
    assert obs2.cumulative_pnl_usd == 2.0
    assert r1 == 1.0 and r2 == 1.0


def test_step_rejects_uncoercible_action() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    env.reset(seed=0, config=cfg)
    with pytest.raises(TypeError):
        env.step("BUY")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        env.step(99)  # value out of TradeAction range


def test_step_coerces_int_action_to_trade_action() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    env.reset(seed=0, config=cfg)
    obs, reward, *_ = env.step(0)  # type: ignore[arg-type]  # HOLD
    assert reward == 0.0
    assert obs.inventory_signed == 0


def test_step_rejects_dynamics_returning_non_transition() -> None:
    class _BadStepDyn:
        def initial_mid_price(self, *, seed: int, config: EpisodeConfig) -> float:
            return 100.0

        def step(
            self,
            prev_obs: Observation,
            action: TradeAction,
            *,
            seed: int,
            config: EpisodeConfig,
        ) -> Any:
            return {"this": "is not a Transition"}

    env = DIXStrategyEnv(_BadStepDyn())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    env.reset(seed=0, config=cfg)
    with pytest.raises(TypeError):
        env.step(TradeAction.HOLD)


def test_reward_combines_pnl_and_drawdown_penalty() -> None:
    env = DIXStrategyEnv(_TerminatingDynamics(terminate_after=5))
    cfg = EpisodeConfig(
        initial_notional_usd=100.0,
        max_steps=10,
        reward_scale=2.0,
        drawdown_penalty_weight=3.0,
    )
    env.reset(seed=0, config=cfg)
    _, reward, *_ = env.step(TradeAction.HOLD)
    # 2.0 * 0.5 - 3.0 * 0.1 = 1.0 - 0.3 = 0.7
    assert reward == pytest.approx(0.7)


def test_step_emits_step_seed_and_step_idx_in_info() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    env.reset(seed=42, config=cfg)
    _, _, _, _, info = env.step(TradeAction.HOLD)
    assert "step_seed" in info
    assert isinstance(info["step_seed"], int)
    assert info["step_idx"] == 1


# ---------------------------------------------------------------------------
# Termination + truncation + budget
# ---------------------------------------------------------------------------


def test_episode_terminates_when_dynamics_signals_terminated() -> None:
    env = DIXStrategyEnv(_TerminatingDynamics(terminate_after=3))
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=10)
    env.reset(seed=0, config=cfg)
    env.step(TradeAction.HOLD)
    env.step(TradeAction.HOLD)
    obs, _, terminated, truncated, _ = env.step(TradeAction.HOLD)
    assert terminated is True
    assert truncated is False
    assert obs.step_idx == 3
    assert env.is_episode_done


def test_episode_truncates_at_max_steps() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=3)
    env.reset(seed=0, config=cfg)
    env.step(TradeAction.HOLD)
    env.step(TradeAction.HOLD)
    obs, _, terminated, truncated, _ = env.step(TradeAction.HOLD)
    assert terminated is False
    assert truncated is True
    assert obs.step_idx == 3
    assert env.is_episode_done


def test_step_after_done_raises() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=2)
    env.reset(seed=0, config=cfg)
    env.step(TradeAction.HOLD)
    env.step(TradeAction.HOLD)  # truncated
    with pytest.raises(EpisodeBudgetExceededError):
        env.step(TradeAction.HOLD)


def test_reset_after_done_starts_a_fresh_episode() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=2)
    env.reset(seed=0, config=cfg)
    env.step(TradeAction.BUY)
    env.step(TradeAction.BUY)
    obs, _ = env.reset(seed=1, config=cfg)
    assert obs.step_idx == 0
    assert obs.cumulative_pnl_usd == 0.0
    assert not env.is_episode_done


def test_current_observation_starts_none_then_tracks_state() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    assert env.current_observation is None
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=5)
    env.reset(seed=0, config=cfg)
    assert env.current_observation is not None
    assert env.current_observation.step_idx == 0
    env.step(TradeAction.HOLD)
    assert env.current_observation.step_idx == 1


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def _run_episode(seed: int) -> list[tuple[Any, ...]]:
    env = DIXStrategyEnv(_SeededDynamics())
    cfg = EpisodeConfig(initial_notional_usd=1_000.0, max_steps=12)
    env.reset(seed=seed, config=cfg)
    actions = [
        TradeAction.BUY,
        TradeAction.HOLD,
        TradeAction.SELL,
        TradeAction.BUY,
        TradeAction.SELL,
        TradeAction.HOLD,
        TradeAction.BUY,
        TradeAction.BUY,
    ]
    out: list[tuple[Any, ...]] = []
    for action in actions:
        obs, reward, terminated, truncated, info = env.step(action)
        out.append(
            (
                obs.step_idx,
                obs.mid_price,
                obs.inventory_signed,
                obs.cumulative_pnl_usd,
                obs.state_hash,
                reward,
                terminated,
                truncated,
                info["step_seed"],
            )
        )
    return out


def test_inv15_byte_identical_three_run_replay() -> None:
    a = _run_episode(seed=12345)
    b = _run_episode(seed=12345)
    c = _run_episode(seed=12345)
    assert a == b == c


def test_inv15_different_seeds_produce_different_episodes() -> None:
    a = _run_episode(seed=1)
    b = _run_episode(seed=2)
    assert a != b


def test_state_hash_changes_when_pnl_changes() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=5)
    obs0, _ = env.reset(seed=0, config=cfg)
    obs1, *_ = env.step(TradeAction.BUY)
    obs2, *_ = env.step(TradeAction.BUY)
    assert obs0.state_hash != obs1.state_hash != obs2.state_hash


def test_state_hash_is_lowercase_16_hex() -> None:
    env = DIXStrategyEnv(_ConstantUpDynamics())
    cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=5)
    obs0, _ = env.reset(seed=99, config=cfg)
    assert len(obs0.state_hash) == 16
    assert all(c in "0123456789abcdef" for c in obs0.state_hash)


# ---------------------------------------------------------------------------
# Optional gymnasium wrapper factory
# ---------------------------------------------------------------------------


def test_gymnasium_factory_either_wraps_or_raises_import_error() -> None:
    """If gymnasium is installed, the factory returns a real Env;
    otherwise it raises ImportError. Either branch is acceptable —
    the AST contract above already pins that gymnasium is NOT
    imported at module top-level."""

    factory = env_mod.gymnasium_dix_strategy_env
    try:
        env = factory(_ConstantUpDynamics())
    except ImportError:
        pytest.skip("gymnasium not installed in this venv")
    else:
        # Smoke-test: reset + step round-trips through the wrapper.
        cfg = EpisodeConfig(initial_notional_usd=100.0, max_steps=3)
        obs, info = env.reset(seed=0, options={"config": cfg})
        assert isinstance(obs, tuple) and len(obs) == 4
        out = env.step(int(TradeAction.HOLD))
        assert isinstance(out, tuple) and len(out) == 5
