"""Tests for evolution_engine/environments/anytrading_env.py (C-29)."""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

from evolution_engine.environments import anytrading_env
from evolution_engine.environments.anytrading_env import (
    ANYTRADING_ENV_VERSION,
    MAX_EPISODE_STEPS,
    MIN_WINDOW_SIZE,
    NEW_PIP_DEPENDENCIES,
    AnytradingAction,
    AnytradingConfig,
    AnytradingEpisodeBudgetExceededError,
    AnytradingObservation,
    AnytradingPosition,
    AnytradingStepResult,
    DIXAnytradingEnv,
    gymnasium_anytrading_env_factory,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_identity_pip_deps() -> None:
    assert NEW_PIP_DEPENDENCIES == ("gym-anytrading", "gymnasium")


def test_module_identity_version() -> None:
    assert ANYTRADING_ENV_VERSION == "c-29-anytrading-1"


def test_module_identity_min_window_size() -> None:
    assert MIN_WINDOW_SIZE == 2


def test_module_identity_max_episode_steps() -> None:
    assert MAX_EPISODE_STEPS == 1_000_000


def test_module_canonical_exports() -> None:
    expected = {
        "NEW_PIP_DEPENDENCIES",
        "ANYTRADING_ENV_VERSION",
        "MIN_WINDOW_SIZE",
        "MAX_EPISODE_STEPS",
        "AnytradingAction",
        "AnytradingPosition",
        "AnytradingConfig",
        "AnytradingObservation",
        "AnytradingStepResult",
        "AnytradingEpisodeBudgetExceededError",
        "DIXAnytradingEnv",
        "gymnasium_anytrading_env_factory",
    }
    assert set(anytrading_env.__all__) == expected


# ---------------------------------------------------------------------------
# Enum identity (mirrors upstream gym-anytrading)
# ---------------------------------------------------------------------------


def test_action_enum_values_match_upstream() -> None:
    assert AnytradingAction.SELL.value == 0
    assert AnytradingAction.BUY.value == 1


def test_position_enum_values_match_upstream() -> None:
    assert AnytradingPosition.SHORT.value == 0
    assert AnytradingPosition.LONG.value == 1


# ---------------------------------------------------------------------------
# AnytradingConfig validation
# ---------------------------------------------------------------------------


def test_config_rejects_empty_prices() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        AnytradingConfig(prices=(), window_size=2)


def test_config_rejects_non_float_price() -> None:
    with pytest.raises(TypeError, match="must be float"):
        AnytradingConfig(prices=(1.0, "x", 3.0), window_size=2)  # type: ignore[arg-type]


def test_config_rejects_non_finite_price() -> None:
    with pytest.raises(ValueError, match="finite"):
        AnytradingConfig(prices=(1.0, float("nan"), 3.0), window_size=2)


def test_config_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError, match="positive"):
        AnytradingConfig(prices=(1.0, 0.0, 3.0), window_size=2)


def test_config_rejects_window_below_min() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        AnytradingConfig(prices=(1.0, 2.0, 3.0), window_size=1)


def test_config_rejects_window_above_price_count() -> None:
    with pytest.raises(ValueError, match="<= len"):
        AnytradingConfig(prices=(1.0, 2.0, 3.0), window_size=4)


def test_config_rejects_non_finite_fee() -> None:
    with pytest.raises(ValueError, match="finite"):
        AnytradingConfig(
            prices=(1.0, 2.0, 3.0),
            window_size=2,
            trade_fee_bid_percent=float("inf"),
        )


def test_config_rejects_negative_fee() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        AnytradingConfig(
            prices=(1.0, 2.0, 3.0),
            window_size=2,
            trade_fee_ask_percent=-0.01,
        )


def test_config_accepts_minimum_valid() -> None:
    cfg = AnytradingConfig(prices=(1.0, 2.0), window_size=2)
    assert cfg.prices == (1.0, 2.0)
    assert cfg.window_size == 2
    assert cfg.initial_position is AnytradingPosition.SHORT


# ---------------------------------------------------------------------------
# AnytradingObservation validation
# ---------------------------------------------------------------------------


def test_observation_rejects_negative_step_idx() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        AnytradingObservation(
            step_idx=-1,
            window=(1.0, 2.0),
            deltas=(1.0,),
            position=AnytradingPosition.SHORT,
            state_hash="0" * 16,
        )


def test_observation_rejects_short_window() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        AnytradingObservation(
            step_idx=0,
            window=(1.0,),
            deltas=(),
            position=AnytradingPosition.SHORT,
            state_hash="0" * 16,
        )


def test_observation_rejects_delta_mismatch() -> None:
    with pytest.raises(ValueError, match="deltas must have len"):
        AnytradingObservation(
            step_idx=0,
            window=(1.0, 2.0, 3.0),
            deltas=(1.0,),
            position=AnytradingPosition.SHORT,
            state_hash="0" * 16,
        )


def test_observation_rejects_bad_hash_length() -> None:
    with pytest.raises(ValueError, match="16 hex"):
        AnytradingObservation(
            step_idx=0,
            window=(1.0, 2.0),
            deltas=(1.0,),
            position=AnytradingPosition.SHORT,
            state_hash="abc",
        )


# ---------------------------------------------------------------------------
# AnytradingStepResult validation
# ---------------------------------------------------------------------------


def _dummy_obs() -> AnytradingObservation:
    return AnytradingObservation(
        step_idx=0,
        window=(1.0, 2.0),
        deltas=(1.0,),
        position=AnytradingPosition.SHORT,
        state_hash="0" * 16,
    )


def test_step_result_rejects_nonfinite_reward() -> None:
    with pytest.raises(ValueError, match="reward must be finite"):
        AnytradingStepResult(
            observation=_dummy_obs(),
            reward=float("nan"),
            terminated=False,
            truncated=False,
            total_reward=0.0,
            total_profit=1.0,
            position_changed=False,
        )


def test_step_result_rejects_nonfinite_total_reward() -> None:
    with pytest.raises(ValueError, match="total_reward must be finite"):
        AnytradingStepResult(
            observation=_dummy_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
            total_reward=float("inf"),
            total_profit=1.0,
            position_changed=False,
        )


def test_step_result_rejects_nonfinite_total_profit() -> None:
    with pytest.raises(ValueError, match="total_profit must be finite"):
        AnytradingStepResult(
            observation=_dummy_obs(),
            reward=0.0,
            terminated=False,
            truncated=False,
            total_reward=0.0,
            total_profit=float("-inf"),
            position_changed=False,
        )


# ---------------------------------------------------------------------------
# DIXAnytradingEnv reset behaviour
# ---------------------------------------------------------------------------


def _make_env(prices: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)) -> DIXAnytradingEnv:
    cfg = AnytradingConfig(prices=prices, window_size=2)
    return DIXAnytradingEnv(config=cfg)


def test_reset_returns_initial_observation() -> None:
    env = _make_env()
    obs, info = env.reset(seed=42)
    assert obs.step_idx == 1
    assert obs.window == (1.0, 2.0)
    assert obs.deltas == (1.0,)
    assert obs.position is AnytradingPosition.SHORT
    assert info["total_reward"] == 0.0
    assert info["total_profit"] == 1.0
    assert info["position"] == AnytradingPosition.SHORT.value


def test_reset_rejects_non_int_seed() -> None:
    env = _make_env()
    with pytest.raises(TypeError, match="must be int"):
        env.reset(seed="0")  # type: ignore[arg-type]


def test_reset_resets_internal_state_between_episodes() -> None:
    env = _make_env()
    env.reset(seed=0)
    env.step(AnytradingAction.BUY.value)
    env.step(AnytradingAction.SELL.value)
    obs, _ = env.reset(seed=0)
    assert env.total_reward == 0.0
    assert env.total_profit == 1.0
    assert obs.position is AnytradingPosition.SHORT
    assert obs.step_idx == 1


# ---------------------------------------------------------------------------
# DIXAnytradingEnv step behaviour
# ---------------------------------------------------------------------------


def test_step_rejects_non_int_action() -> None:
    env = _make_env()
    env.reset()
    with pytest.raises(TypeError, match="must be int"):
        env.step("buy")  # type: ignore[arg-type]


def test_step_rejects_out_of_range_action() -> None:
    env = _make_env()
    env.reset()
    with pytest.raises(ValueError, match=r"must be in"):
        env.step(2)


def test_step_no_action_change_yields_zero_reward() -> None:
    env = _make_env()
    env.reset()
    result = env.step(AnytradingAction.SELL.value)  # already SHORT
    assert result.reward == 0.0
    assert not result.position_changed
    assert result.observation.position is AnytradingPosition.SHORT


def test_step_buy_from_short_changes_position_zero_reward() -> None:
    env = _make_env()
    env.reset()
    result = env.step(AnytradingAction.BUY.value)
    assert result.reward == 0.0
    assert result.position_changed
    assert result.observation.position is AnytradingPosition.LONG


def test_step_long_to_short_realises_reward() -> None:
    env = _make_env(prices=(1.0, 2.0, 3.0, 4.0, 5.0))
    env.reset()
    env.step(AnytradingAction.BUY.value)
    result = env.step(AnytradingAction.SELL.value)
    assert result.reward == pytest.approx(4.0 - 3.0)
    assert result.position_changed
    assert result.observation.position is AnytradingPosition.SHORT
    assert result.total_profit == pytest.approx(4.0 / 3.0)


def test_step_terminates_at_final_price() -> None:
    env = _make_env(prices=(1.0, 2.0, 3.0, 4.0))
    env.reset()
    result_first = env.step(AnytradingAction.SELL.value)
    assert not result_first.terminated
    result_last = env.step(AnytradingAction.SELL.value)
    assert result_last.terminated
    assert not result_last.truncated


def test_step_after_termination_raises() -> None:
    env = _make_env(prices=(1.0, 2.0, 3.0, 4.0))
    env.reset()
    env.step(AnytradingAction.SELL.value)
    env.step(AnytradingAction.SELL.value)
    with pytest.raises(RuntimeError, match="termination"):
        env.step(AnytradingAction.SELL.value)


def test_step_total_reward_accumulates() -> None:
    env = _make_env(prices=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
    env.reset()
    env.step(AnytradingAction.BUY.value)
    env.step(AnytradingAction.BUY.value)
    result = env.step(AnytradingAction.SELL.value)
    assert env.total_reward == result.total_reward
    assert result.total_reward > 0.0


# ---------------------------------------------------------------------------
# INV-15: byte-identical replay determinism
# ---------------------------------------------------------------------------


def _replay_episode(actions: tuple[int, ...], seed: int) -> list[str]:
    env = _make_env(prices=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
    obs, _ = env.reset(seed=seed)
    hashes: list[str] = [obs.state_hash]
    for action in actions:
        result = env.step(action)
        hashes.append(result.observation.state_hash)
        if result.terminated or result.truncated:
            break
    return hashes


def test_replay_byte_identical_three_runs() -> None:
    actions = (
        AnytradingAction.BUY.value,
        AnytradingAction.BUY.value,
        AnytradingAction.SELL.value,
        AnytradingAction.BUY.value,
        AnytradingAction.SELL.value,
    )
    run_a = _replay_episode(actions, seed=42)
    run_b = _replay_episode(actions, seed=42)
    run_c = _replay_episode(actions, seed=42)
    assert run_a == run_b == run_c


def test_replay_seed_changes_hash() -> None:
    actions = (AnytradingAction.BUY.value, AnytradingAction.SELL.value)
    run_a = _replay_episode(actions, seed=1)
    run_b = _replay_episode(actions, seed=2)
    assert run_a != run_b


# ---------------------------------------------------------------------------
# Episode budget cap
# ---------------------------------------------------------------------------


def test_episode_budget_cap_raises() -> None:
    cfg = AnytradingConfig(
        prices=tuple(float(i + 1) for i in range(MAX_EPISODE_STEPS + 10)),
        window_size=2,
    )
    env = DIXAnytradingEnv(config=cfg)
    env.reset()
    for _ in range(MAX_EPISODE_STEPS):
        env.step(AnytradingAction.SELL.value)
    with pytest.raises(AnytradingEpisodeBudgetExceededError):
        env.step(AnytradingAction.SELL.value)


# ---------------------------------------------------------------------------
# Value object frozen + slotted guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        AnytradingConfig(prices=(1.0, 2.0), window_size=2),
        AnytradingObservation(
            step_idx=0,
            window=(1.0, 2.0),
            deltas=(1.0,),
            position=AnytradingPosition.SHORT,
            state_hash="0" * 16,
        ),
        AnytradingStepResult(
            observation=AnytradingObservation(
                step_idx=0,
                window=(1.0, 2.0),
                deltas=(1.0,),
                position=AnytradingPosition.SHORT,
                state_hash="0" * 16,
            ),
            reward=0.0,
            terminated=False,
            truncated=False,
            total_reward=0.0,
            total_profit=1.0,
            position_changed=False,
        ),
    ],
)
def test_value_objects_frozen_and_slotted(obj: object) -> None:
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(obj, "_arbitrary_attr_for_test", "y")
    assert not hasattr(obj, "__dict__")


# ---------------------------------------------------------------------------
# Render is a no-op
# ---------------------------------------------------------------------------


def test_render_is_noop() -> None:
    env = _make_env()
    env.reset()
    assert env.render() is None
    assert env.render(mode=None) is None


# ---------------------------------------------------------------------------
# gymnasium_anytrading_env_factory lazy seam
# ---------------------------------------------------------------------------


def test_gymnasium_factory_requires_gymnasium(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AnytradingConfig(prices=(1.0, 2.0, 3.0), window_size=2)
    monkeypatch.setitem(sys.modules, "gymnasium", None)
    with pytest.raises(RuntimeError, match="gymnasium"):
        gymnasium_anytrading_env_factory(config=cfg)


def test_gymnasium_factory_only_imports_gymnasium_inside_body() -> None:
    src = pathlib.Path(anytrading_env.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level_imports.add(node.module)
    forbidden = {"gymnasium", "gym", "gym_anytrading", "numpy"}
    leaks = top_level_imports & forbidden
    assert leaks == set(), f"Top-level vendor leaks: {leaks!r}"


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


def _collect_imports(path: pathlib.Path) -> tuple[set[str], dict[str, set[str]]]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_level: set[str] = set()
    function_locals: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            top_level.add(node.module)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names: set[str] = set()
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        names.add(alias.name)
                if isinstance(sub, ast.ImportFrom) and sub.module is not None:
                    names.add(sub.module)
            if names:
                function_locals[node.name] = names
    return top_level, function_locals


def test_no_top_level_engine_cross_imports() -> None:
    top_level, _ = _collect_imports(pathlib.Path(anytrading_env.__file__))
    forbidden = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "learning_engine",
    }
    leaks = {m for m in top_level if any(m.startswith(p) for p in forbidden)}
    assert leaks == set(), f"Engine cross-imports at top level: {leaks!r}"


def test_no_top_level_io_or_subprocess_imports() -> None:
    top_level, _ = _collect_imports(pathlib.Path(anytrading_env.__file__))
    forbidden = {"subprocess", "os.path", "pathlib", "socket", "urllib", "requests"}
    leaks = top_level & forbidden
    assert leaks == set(), f"IO module leaks at top level: {leaks!r}"


def test_gymnasium_factory_holds_gymnasium_import_inside_body() -> None:
    _, function_locals = _collect_imports(pathlib.Path(anytrading_env.__file__))
    assert "gymnasium_anytrading_env_factory" in function_locals
    factory_imports = function_locals["gymnasium_anytrading_env_factory"]
    assert "gymnasium" in factory_imports


def test_module_reload_idempotent() -> None:
    importlib.reload(anytrading_env)
    assert anytrading_env.ANYTRADING_ENV_VERSION == "c-29-anytrading-1"
