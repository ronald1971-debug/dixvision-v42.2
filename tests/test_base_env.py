"""Tests for ``evolution_engine.environments.base_env`` (C-31)."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib
import pathlib
import sys
from typing import Any

import pytest

from evolution_engine.environments import base_env
from evolution_engine.environments.base_env import (
    DIX_BASE_ENV_VERSION,
    MAX_EPISODE_STEPS,
    NEW_PIP_DEPENDENCIES,
    DIXBaseEnv,
    DIXBaseEnvNotResetError,
    DIXBaseEpisodeBudgetExceededError,
    DIXBaseObservation,
    DIXBaseStepResult,
    DIXBoxSpace,
    DIXDiscreteSpace,
)

# ----- module identity ------------------------------------------------------


def test_new_pip_dependencies_is_gymnasium() -> None:
    assert NEW_PIP_DEPENDENCIES == ("gymnasium",)


def test_version_string_pinned() -> None:
    assert DIX_BASE_ENV_VERSION == "c-31-base-env-1"


def test_max_episode_steps_pinned() -> None:
    assert MAX_EPISODE_STEPS == 1_000_000


def test_exports_pinned() -> None:
    assert set(base_env.__all__) == {
        "NEW_PIP_DEPENDENCIES",
        "DIX_BASE_ENV_VERSION",
        "MAX_EPISODE_STEPS",
        "DIXBaseEpisodeBudgetExceededError",
        "DIXBaseEnvNotResetError",
        "DIXBoxSpace",
        "DIXDiscreteSpace",
        "DIXBaseObservation",
        "DIXBaseStepResult",
        "DIXBaseEnv",
        "gymnasium_dix_base_env_factory",
    }


# ----- DIXBoxSpace ----------------------------------------------------------


def test_box_space_valid() -> None:
    space = DIXBoxSpace(low=0.0, high=1.0, shape=(3,))
    assert space.low == 0.0
    assert space.high == 1.0
    assert space.shape == (3,)


def test_box_space_rejects_infinite_low() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        DIXBoxSpace(low=float("-inf"), high=1.0, shape=(3,))


def test_box_space_rejects_infinite_high() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        DIXBoxSpace(low=0.0, high=float("inf"), shape=(3,))


def test_box_space_rejects_low_ge_high() -> None:
    with pytest.raises(ValueError, match="must be < high"):
        DIXBoxSpace(low=1.0, high=1.0, shape=(3,))


def test_box_space_rejects_empty_shape() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        DIXBoxSpace(low=0.0, high=1.0, shape=())


def test_box_space_rejects_non_int_shape() -> None:
    with pytest.raises(TypeError, match="must be int"):
        DIXBoxSpace(low=0.0, high=1.0, shape=(3.0,))  # type: ignore[arg-type]


def test_box_space_rejects_non_positive_shape() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DIXBoxSpace(low=0.0, high=1.0, shape=(0,))


def test_box_space_contains_valid() -> None:
    space = DIXBoxSpace(low=0.0, high=1.0, shape=(3,))
    assert space.contains((0.0, 0.5, 1.0)) is True
    assert space.contains((-0.1, 0.5, 0.9)) is False
    assert space.contains((0.0, 0.5)) is False
    assert space.contains((0.0, 0.5, float("nan"))) is False
    assert space.contains((0.0, 0.5, float("inf"))) is False
    assert space.contains([0.0, 0.5, 1.0]) is False  # type: ignore[arg-type]


def test_box_space_sample_deterministic() -> None:
    space = DIXBoxSpace(low=0.0, high=1.0, shape=(3,))
    s1 = space.sample(seed=42)
    s2 = space.sample(seed=42)
    s3 = space.sample(seed=43)
    assert s1 == s2
    assert s1 != s3
    assert len(s1) == 3
    assert all(0.0 <= v <= 1.0 for v in s1)


def test_box_space_frozen() -> None:
    space = DIXBoxSpace(low=0.0, high=1.0, shape=(3,))
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        space.low = 0.5  # type: ignore[misc]


# ----- DIXDiscreteSpace -----------------------------------------------------


def test_discrete_space_valid() -> None:
    space = DIXDiscreteSpace(n=3)
    assert space.n == 3


def test_discrete_space_rejects_non_int() -> None:
    with pytest.raises(TypeError, match="must be int"):
        DIXDiscreteSpace(n=3.0)  # type: ignore[arg-type]


def test_discrete_space_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DIXDiscreteSpace(n=0)


def test_discrete_space_contains_valid() -> None:
    space = DIXDiscreteSpace(n=3)
    assert space.contains(0) is True
    assert space.contains(2) is True
    assert space.contains(3) is False
    assert space.contains(-1) is False
    assert space.contains(True) is False
    assert space.contains(1.5) is False  # type: ignore[arg-type]


def test_discrete_space_sample_deterministic() -> None:
    space = DIXDiscreteSpace(n=5)
    s1 = space.sample(seed=42)
    s2 = space.sample(seed=42)
    assert s1 == s2
    assert 0 <= s1 < 5


def test_discrete_space_frozen() -> None:
    space = DIXDiscreteSpace(n=3)
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        space.n = 5  # type: ignore[misc]


# ----- DIXBaseObservation ---------------------------------------------------


def test_observation_valid() -> None:
    obs = DIXBaseObservation(
        step_idx=0,
        payload=(1.0, 2.0),
        state_hash="0" * 16,
    )
    assert obs.step_idx == 0
    assert obs.payload == (1.0, 2.0)
    assert len(obs.state_hash) == 16


def test_observation_rejects_negative_step_idx() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        DIXBaseObservation(step_idx=-1, payload=None, state_hash="0" * 16)


def test_observation_rejects_wrong_hash_length() -> None:
    with pytest.raises(ValueError, match="16 hex chars"):
        DIXBaseObservation(step_idx=0, payload=None, state_hash="abc")


def test_observation_frozen() -> None:
    obs = DIXBaseObservation(step_idx=0, payload=None, state_hash="0" * 16)
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        obs.step_idx = 5  # type: ignore[misc]


# ----- DIXBaseStepResult ----------------------------------------------------


def test_step_result_valid() -> None:
    obs = DIXBaseObservation(step_idx=1, payload=None, state_hash="a" * 16)
    result = DIXBaseStepResult(
        observation=obs,
        reward=0.5,
        terminated=False,
        truncated=False,
        info=(("k", "v"),),
    )
    assert result.reward == 0.5
    assert result.info_dict() == {"k": "v"}


def test_step_result_rejects_non_finite_reward() -> None:
    obs = DIXBaseObservation(step_idx=1, payload=None, state_hash="a" * 16)
    with pytest.raises(ValueError, match="must be finite"):
        DIXBaseStepResult(
            observation=obs,
            reward=float("nan"),
            terminated=False,
            truncated=False,
            info=(),
        )


def test_step_result_frozen() -> None:
    obs = DIXBaseObservation(step_idx=1, payload=None, state_hash="a" * 16)
    result = DIXBaseStepResult(
        observation=obs,
        reward=0.5,
        terminated=False,
        truncated=False,
        info=(),
    )
    with pytest.raises((AttributeError, TypeError, dataclasses.FrozenInstanceError)):
        result.reward = 1.0  # type: ignore[misc]


# ----- DIXBaseEnv (via concrete subclass) -----------------------------------


class _CountingEnv(DIXBaseEnv):
    """Concrete subclass for testing base behavior — counts upward."""

    observation_space = DIXBoxSpace(low=0.0, high=1_000_000.0, shape=(1,))
    action_space = DIXDiscreteSpace(n=3)

    def __init__(self, *, max_steps: int = 5) -> None:
        super().__init__()
        self._max_steps = max_steps
        self._counter = 0

    def _reset_payload(self, *, options: dict[str, Any] | None) -> tuple[float, ...]:
        self._counter = 0
        return (float(self._counter),)

    def _step_payload(
        self, action: Any
    ) -> tuple[tuple[float, ...], float, bool, bool, dict[str, Any]]:
        if not isinstance(action, int) or not (0 <= action < 3):
            raise ValueError(f"invalid action {action!r}")
        self._counter += 1
        terminated = self._step_idx >= self._max_steps
        truncated = False
        reward = float(action)
        return (
            (float(self._counter),),
            reward,
            terminated,
            truncated,
            {"action": action},
        )


def test_env_reset_returns_obs_and_info() -> None:
    env = _CountingEnv()
    obs, info = env.reset(seed=42)
    assert obs.step_idx == 0
    assert obs.payload == (0.0,)
    assert len(obs.state_hash) == 16
    assert info == {}
    assert env.seed == 42
    assert env.step_idx == 0
    assert env.is_terminated is False


def test_env_reset_rejects_non_int_seed() -> None:
    env = _CountingEnv()
    with pytest.raises(TypeError, match="must be int"):
        env.reset(seed=42.0)  # type: ignore[arg-type]


def test_env_step_before_reset_raises() -> None:
    env = _CountingEnv()
    with pytest.raises(DIXBaseEnvNotResetError, match="before reset"):
        env.step(0)


def test_env_step_returns_5_tuple_shape() -> None:
    env = _CountingEnv()
    env.reset(seed=42)
    result = env.step(1)
    assert isinstance(result, DIXBaseStepResult)
    assert result.observation.step_idx == 1
    assert result.observation.payload == (1.0,)
    assert result.reward == 1.0
    assert result.terminated is False
    assert result.truncated is False
    assert result.info_dict() == {"action": 1}


def test_env_step_advances_step_idx() -> None:
    env = _CountingEnv()
    env.reset(seed=42)
    for i in range(1, 4):
        result = env.step(0)
        assert result.observation.step_idx == i


def test_env_step_terminates_at_max_steps() -> None:
    env = _CountingEnv(max_steps=3)
    env.reset(seed=42)
    for _ in range(2):
        result = env.step(0)
        assert result.terminated is False
    result = env.step(0)
    assert result.terminated is True
    assert env.is_terminated is True


def test_env_step_after_termination_raises() -> None:
    env = _CountingEnv(max_steps=1)
    env.reset(seed=42)
    env.step(0)
    with pytest.raises(RuntimeError, match="after termination"):
        env.step(0)


def test_env_episode_budget_cap() -> None:
    env = _CountingEnv(max_steps=2_000_000)
    env.reset(seed=42)
    # Patch the limit down for the test
    saved_limit = base_env.MAX_EPISODE_STEPS
    try:
        base_env.MAX_EPISODE_STEPS = 5
        # Note: the constant is captured at function call time in the
        # step method, so monkey-patching works only if we re-read.
        # The env actually reads MAX_EPISODE_STEPS at step time via
        # the module-level constant.
        for _ in range(5):
            env.step(0)
        with pytest.raises(DIXBaseEpisodeBudgetExceededError):
            env.step(0)
    finally:
        base_env.MAX_EPISODE_STEPS = saved_limit


def test_env_render_returns_none() -> None:
    env = _CountingEnv()
    env.reset(seed=42)
    assert env.render() is None
    assert env.render(mode=None) is None


def test_env_close_returns_none() -> None:
    env = _CountingEnv()
    assert env.close() is None


# ----- INV-15 byte-identical replay ----------------------------------------


def _run_episode(seed: int, max_steps: int = 5) -> tuple[str, ...]:
    env = _CountingEnv(max_steps=max_steps)
    env.reset(seed=seed)
    hashes: list[str] = []
    for i in range(max_steps):
        result = env.step(i % 3)
        hashes.append(result.observation.state_hash)
    return tuple(hashes)


def test_replay_byte_identical_three_runs() -> None:
    """INV-15 — three replays of the same episode produce byte-identical
    observation hashes."""

    h1 = _run_episode(seed=42)
    h2 = _run_episode(seed=42)
    h3 = _run_episode(seed=42)
    assert h1 == h2 == h3
    for h in h1:
        assert len(h) == 16


def test_replay_seed_changes_hash() -> None:
    h1 = _run_episode(seed=42)
    h2 = _run_episode(seed=43)
    assert h1 != h2


def test_replay_step_count_changes_hash() -> None:
    h1 = _run_episode(seed=42, max_steps=3)
    h2 = _run_episode(seed=42, max_steps=4)
    # Same first 3 hashes; h2 longer.
    assert h1 == h2[:3]


# ----- Hash computation ----------------------------------------------------


def test_state_hash_uses_blake2b_16() -> None:
    env = _CountingEnv()
    env.reset(seed=42)
    result = env.step(1)
    expected = hashlib.blake2b(
        "|".join(
            (
                f"v={DIX_BASE_ENV_VERSION}",
                "seed=42",
                "step=1",
                "payload=(1.0,)",
            )
        ).encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    assert result.observation.state_hash == expected


# ----- AST guards ----------------------------------------------------------


def _module_ast() -> ast.Module:
    src = pathlib.Path(base_env.__file__).read_text(encoding="utf-8")
    return ast.parse(src)


def test_no_top_level_gymnasium_import() -> None:
    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("gymnasium")
                assert not alias.name.startswith("gym")
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("gymnasium")
            assert node.module is None or not node.module.startswith("gym")


def test_no_top_level_numpy_import() -> None:
    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("numpy")
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("numpy")


def test_no_top_level_io_imports() -> None:
    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("subprocess")
                assert not alias.name.startswith("socket")
                assert not alias.name.startswith("urllib")
                assert not alias.name.startswith("requests")
                assert not alias.name.startswith("httpx")
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith(
                ("subprocess", "socket", "urllib", "requests", "httpx")
            )


def test_no_engine_cross_imports() -> None:
    tree = _module_ast()
    forbidden = (
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "learning_engine",
    )
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith(forbidden)
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith(forbidden)


def test_gymnasium_only_inside_factory() -> None:
    tree = _module_ast()
    factory_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "gymnasium_dix_base_env_factory":
            factory_node = node
            break
    assert factory_node is not None
    found_gym_import = False
    for inner in ast.walk(factory_node):
        if isinstance(inner, ast.Import):
            for alias in inner.names:
                if alias.name == "gymnasium":
                    found_gym_import = True
        if isinstance(inner, ast.ImportFrom):
            if inner.module == "gymnasium":
                found_gym_import = True
    assert found_gym_import, "gymnasium import must live inside factory body"


# ----- Reload idempotency --------------------------------------------------


def test_module_reload_idempotent() -> None:
    mod1 = sys.modules["evolution_engine.environments.base_env"]
    mod2 = importlib.reload(mod1)
    assert mod2.DIX_BASE_ENV_VERSION == DIX_BASE_ENV_VERSION
    assert mod2.MAX_EPISODE_STEPS == MAX_EPISODE_STEPS
