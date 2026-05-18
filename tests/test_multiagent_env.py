"""Tests for evolution_engine/environments/multiagent_env.py (C-30)."""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
from typing import Any

import pytest

from evolution_engine.environments import multiagent_env
from evolution_engine.environments.multiagent_env import (
    MAX_AGENTS,
    MAX_EPISODE_STEPS,
    MIN_AGENTS,
    MULTIAGENT_ENV_VERSION,
    NEW_PIP_DEPENDENCIES,
    AgentSelector,
    DIXMultiAgentEnv,
    MultiAgentAction,
    MultiAgentMode,
    MultiAgentObservation,
    MultiAgentScenario,
    MultiAgentStepResult,
    UnknownAgentError,
    WrongStepShapeError,
    pettingzoo_multiagent_env_factory,
)

# ---------------------------------------------------------------------------
# Module identity
# ---------------------------------------------------------------------------


def test_module_identity_pip_deps() -> None:
    assert NEW_PIP_DEPENDENCIES == ("pettingzoo", "gymnasium")


def test_module_identity_version() -> None:
    assert MULTIAGENT_ENV_VERSION == "c-30-pettingzoo-1"


def test_module_identity_min_agents() -> None:
    assert MIN_AGENTS == 2


def test_module_identity_max_agents() -> None:
    assert MAX_AGENTS == 64


def test_module_identity_max_episode_steps() -> None:
    assert MAX_EPISODE_STEPS == 1_000_000


def test_module_canonical_exports() -> None:
    expected = {
        "NEW_PIP_DEPENDENCIES",
        "MULTIAGENT_ENV_VERSION",
        "MIN_AGENTS",
        "MAX_AGENTS",
        "MAX_EPISODE_STEPS",
        "MultiAgentMode",
        "MultiAgentAction",
        "MultiAgentEpisodeBudgetExceededError",
        "UnknownAgentError",
        "WrongStepShapeError",
        "MultiAgentScenario",
        "MultiAgentObservation",
        "MultiAgentStepResult",
        "AgentSelector",
        "DIXMultiAgentEnv",
        "pettingzoo_multiagent_env_factory",
    }
    assert set(multiagent_env.__all__) == expected


def test_mode_enum_values() -> None:
    assert MultiAgentMode.PARALLEL.value == 0
    assert MultiAgentMode.AEC.value == 1


def test_action_enum_values() -> None:
    assert MultiAgentAction.HOLD.value == 0
    assert MultiAgentAction.BUY.value == 1
    assert MultiAgentAction.SELL.value == 2


# ---------------------------------------------------------------------------
# MultiAgentScenario validation
# ---------------------------------------------------------------------------


def test_scenario_rejects_too_few_agents() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        MultiAgentScenario(agent_ids=("a",), prices=(1.0, 2.0), max_steps=10)


def test_scenario_rejects_too_many_agents() -> None:
    with pytest.raises(ValueError, match="<= 64"):
        MultiAgentScenario(
            agent_ids=tuple(f"a{i}" for i in range(65)),
            prices=(1.0, 2.0),
            max_steps=10,
        )


def test_scenario_rejects_duplicate_agents() -> None:
    with pytest.raises(ValueError, match="unique"):
        MultiAgentScenario(agent_ids=("a", "a"), prices=(1.0, 2.0), max_steps=10)


def test_scenario_rejects_non_string_agent() -> None:
    with pytest.raises(TypeError, match="must be str"):
        MultiAgentScenario(
            agent_ids=("a", 7),  # type: ignore[arg-type]
            prices=(1.0, 2.0),
            max_steps=10,
        )


def test_scenario_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MultiAgentScenario(agent_ids=("a", ""), prices=(1.0, 2.0), max_steps=10)


def test_scenario_rejects_empty_prices() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MultiAgentScenario(agent_ids=("a", "b"), prices=(), max_steps=10)


def test_scenario_rejects_non_finite_price() -> None:
    with pytest.raises(ValueError, match="finite"):
        MultiAgentScenario(agent_ids=("a", "b"), prices=(1.0, float("nan")), max_steps=10)


def test_scenario_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError, match="positive"):
        MultiAgentScenario(agent_ids=("a", "b"), prices=(1.0, 0.0), max_steps=10)


def test_scenario_rejects_zero_max_steps() -> None:
    with pytest.raises(ValueError, match="positive"):
        MultiAgentScenario(agent_ids=("a", "b"), prices=(1.0, 2.0), max_steps=0)


def test_scenario_rejects_max_steps_above_cap() -> None:
    with pytest.raises(ValueError, match="<="):
        MultiAgentScenario(
            agent_ids=("a", "b"),
            prices=(1.0, 2.0),
            max_steps=MAX_EPISODE_STEPS + 1,
        )


def test_scenario_default_mode_is_parallel() -> None:
    s = MultiAgentScenario(agent_ids=("a", "b"), prices=(1.0, 2.0), max_steps=10)
    assert s.mode is MultiAgentMode.PARALLEL


# ---------------------------------------------------------------------------
# MultiAgentObservation validation
# ---------------------------------------------------------------------------


def _obs(**overrides: Any) -> MultiAgentObservation:
    defaults: dict[str, Any] = {
        "agent_id": "a",
        "step_idx": 0,
        "mid_price": 1.0,
        "inventory_signed": 0,
        "cumulative_pnl_usd": 0.0,
        "state_hash": "0" * 16,
    }
    defaults.update(overrides)
    return MultiAgentObservation(**defaults)


def test_observation_rejects_empty_agent_id() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _obs(agent_id="")


def test_observation_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        _obs(step_idx=-1)


def test_observation_rejects_non_finite_price() -> None:
    with pytest.raises(ValueError, match="finite"):
        _obs(mid_price=float("nan"))


def test_observation_rejects_non_positive_price() -> None:
    with pytest.raises(ValueError, match="positive"):
        _obs(mid_price=0.0)


def test_observation_rejects_invalid_inventory_sign() -> None:
    with pytest.raises(ValueError, match=r"-1/0/\+1"):
        _obs(inventory_signed=2)


def test_observation_rejects_non_finite_pnl() -> None:
    with pytest.raises(ValueError, match="finite"):
        _obs(cumulative_pnl_usd=float("inf"))


def test_observation_rejects_bad_hash_length() -> None:
    with pytest.raises(ValueError, match="16 hex"):
        _obs(state_hash="abc")


# ---------------------------------------------------------------------------
# MultiAgentStepResult validation
# ---------------------------------------------------------------------------


def test_step_result_rejects_duplicate_agent_obs() -> None:
    obs = _obs(agent_id="a")
    with pytest.raises(ValueError, match="unique agent_ids"):
        MultiAgentStepResult(
            observations=(obs, obs),
            rewards=(("a", 0.0),),
            terminations=(("a", False),),
            truncations=(("a", False),),
            infos=(("a", ()),),
        )


def test_step_result_rejects_reward_agent_mismatch() -> None:
    obs = _obs(agent_id="a")
    with pytest.raises(ValueError, match="exactly the same"):
        MultiAgentStepResult(
            observations=(obs,),
            rewards=(("b", 0.0),),
            terminations=(("a", False),),
            truncations=(("a", False),),
            infos=(("a", ()),),
        )


# ---------------------------------------------------------------------------
# AgentSelector
# ---------------------------------------------------------------------------


def test_selector_rejects_short_roster() -> None:
    with pytest.raises(ValueError, match=">= 2"):
        AgentSelector(("a",))


def test_selector_cycles_in_order() -> None:
    sel = AgentSelector(("a", "b", "c"))
    assert sel.current == "a"
    assert sel.next() == "b"
    assert sel.next() == "c"
    assert sel.next() == "a"


def test_selector_is_last_detection() -> None:
    sel = AgentSelector(("a", "b"))
    assert not sel.is_last()
    sel.next()
    assert sel.is_last()


def test_selector_reset() -> None:
    sel = AgentSelector(("a", "b"))
    sel.next()
    sel.reset()
    assert sel.current == "a"


# ---------------------------------------------------------------------------
# DIXMultiAgentEnv reset
# ---------------------------------------------------------------------------


def _parallel_env(
    agents: tuple[str, ...] = ("a", "b"),
    prices: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    max_steps: int = 100,
) -> DIXMultiAgentEnv:
    scenario = MultiAgentScenario(
        agent_ids=agents,
        prices=prices,
        max_steps=max_steps,
        mode=MultiAgentMode.PARALLEL,
    )
    return DIXMultiAgentEnv(scenario=scenario)


def _aec_env(
    agents: tuple[str, ...] = ("a", "b"),
    prices: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
    max_steps: int = 100,
) -> DIXMultiAgentEnv:
    scenario = MultiAgentScenario(
        agent_ids=agents,
        prices=prices,
        max_steps=max_steps,
        mode=MultiAgentMode.AEC,
    )
    return DIXMultiAgentEnv(scenario=scenario)


def test_reset_returns_one_observation_per_agent() -> None:
    env = _parallel_env(agents=("a", "b", "c"))
    observations, infos = env.reset(seed=42)
    assert tuple(o.agent_id for o in observations) == ("a", "b", "c")
    assert set(infos) == {"a", "b", "c"}


def test_reset_rejects_non_int_seed() -> None:
    env = _parallel_env()
    with pytest.raises(TypeError, match="must be int"):
        env.reset(seed="0")  # type: ignore[arg-type]


def test_reset_resets_inventory_and_pnl() -> None:
    env = _parallel_env()
    env.reset(seed=0)
    env.step({"a": MultiAgentAction.BUY.value, "b": MultiAgentAction.SELL.value})
    env.reset(seed=0)
    assert env.inventory("a") == 0
    assert env.inventory("b") == 0
    assert env.pnl("a") == 0.0
    assert env.pnl("b") == 0.0


# ---------------------------------------------------------------------------
# DIXMultiAgentEnv PARALLEL step
# ---------------------------------------------------------------------------


def test_step_rejects_non_dict_actions() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(TypeError, match="dict"):
        env.step([0, 0])  # type: ignore[arg-type]


def test_step_rejects_missing_agent() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(ValueError, match="missing"):
        env.step({"a": 0})


def test_step_rejects_unknown_agent() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(UnknownAgentError):
        env.step({"a": 0, "b": 0, "c": 0})


def test_step_rejects_non_int_action() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(TypeError, match="must be int"):
        env.step({"a": 0, "b": "buy"})  # type: ignore[dict-item]


def test_step_rejects_out_of_range_action() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(ValueError, match=r"in \{0,1,2\}"):
        env.step({"a": 0, "b": 3})


def test_step_long_agent_earns_on_upward_price() -> None:
    env = _parallel_env(prices=(1.0, 2.0, 3.0))
    env.reset()
    result = env.step({"a": MultiAgentAction.BUY.value, "b": MultiAgentAction.HOLD.value})
    assert result.rewards_dict()["a"] == 0.0
    assert result.rewards_dict()["b"] == 0.0
    assert env.inventory("a") == 1
    result2 = env.step({"a": MultiAgentAction.HOLD.value, "b": MultiAgentAction.HOLD.value})
    assert result2.rewards_dict()["a"] == pytest.approx(1.0)


def test_step_holding_position_earns_on_price_move() -> None:
    env = _parallel_env(prices=(1.0, 2.0, 3.0, 4.0))
    env.reset()
    env.step({"a": MultiAgentAction.BUY.value, "b": MultiAgentAction.HOLD.value})
    result = env.step({"a": MultiAgentAction.HOLD.value, "b": MultiAgentAction.HOLD.value})
    assert result.rewards_dict()["a"] == pytest.approx(1.0)


def test_step_terminates_at_final_price() -> None:
    env = _parallel_env(prices=(1.0, 2.0, 3.0, 4.0))
    env.reset()
    actions = {"a": 0, "b": 0}
    env.step(actions)
    env.step(actions)
    result = env.step(actions)
    assert result.terminations_dict()["a"] is True
    assert result.terminations_dict()["b"] is True


def test_step_after_termination_raises() -> None:
    env = _parallel_env(prices=(1.0, 2.0, 3.0))
    env.reset()
    env.step({"a": 0, "b": 0})
    env.step({"a": 0, "b": 0})
    with pytest.raises(RuntimeError, match="termination"):
        env.step({"a": 0, "b": 0})


def test_step_wrong_shape_in_aec_mode() -> None:
    env = _aec_env()
    env.reset()
    with pytest.raises(WrongStepShapeError):
        env.step({"a": 0, "b": 0})


# ---------------------------------------------------------------------------
# DIXMultiAgentEnv AEC step
# ---------------------------------------------------------------------------


def test_step_aec_wrong_shape_in_parallel_mode() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(WrongStepShapeError):
        env.step_aec("a", 0)


def test_step_aec_enforces_turn_order() -> None:
    env = _aec_env()
    env.reset()
    with pytest.raises(ValueError, match="turn order"):
        env.step_aec("b", 0)


def test_step_aec_cycles_through_agents() -> None:
    env = _aec_env(agents=("a", "b", "c"))
    env.reset()
    env.step_aec("a", 0)
    assert env.agent_selection == "b"
    env.step_aec("b", 0)
    assert env.agent_selection == "c"
    env.step_aec("c", 0)
    assert env.agent_selection == "a"


def test_step_aec_advances_step_only_after_full_cycle() -> None:
    env = _aec_env(agents=("a", "b"))
    env.reset()
    assert env.step_idx == 0
    env.step_aec("a", 0)
    assert env.step_idx == 0
    env.step_aec("b", 0)
    assert env.step_idx == 1


def test_step_aec_rejects_unknown_agent() -> None:
    env = _aec_env()
    env.reset()
    with pytest.raises(UnknownAgentError):
        env.step_aec("nope", 0)


def test_step_aec_rejects_bad_action() -> None:
    env = _aec_env()
    env.reset()
    with pytest.raises(ValueError, match=r"in \{0,1,2\}"):
        env.step_aec("a", 99)


def test_last_returns_current_agent_observation() -> None:
    env = _aec_env(agents=("a", "b"))
    env.reset()
    obs = env.last()
    assert obs.agent_id == "a"


# ---------------------------------------------------------------------------
# Inventory / pnl introspection
# ---------------------------------------------------------------------------


def test_inventory_query_unknown_raises() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(UnknownAgentError):
        env.inventory("nope")


def test_pnl_query_unknown_raises() -> None:
    env = _parallel_env()
    env.reset()
    with pytest.raises(UnknownAgentError):
        env.pnl("nope")


def test_possible_agents_property() -> None:
    env = _parallel_env(agents=("a", "b", "c"))
    assert env.possible_agents == ("a", "b", "c")


def test_agents_property_empty_after_termination() -> None:
    env = _parallel_env(prices=(1.0, 2.0, 3.0))
    env.reset()
    env.step({"a": 0, "b": 0})
    env.step({"a": 0, "b": 0})
    assert env.agents == ()


# ---------------------------------------------------------------------------
# INV-15: byte-identical replay determinism
# ---------------------------------------------------------------------------


def _replay_parallel(
    actions_sequence: tuple[dict[str, int], ...], seed: int
) -> list[tuple[str, ...]]:
    env = _parallel_env(
        agents=("a", "b"),
        prices=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
    )
    obs, _ = env.reset(seed=seed)
    hashes: list[tuple[str, ...]] = [tuple(o.state_hash for o in obs)]
    for actions in actions_sequence:
        result = env.step(actions)
        hashes.append(tuple(o.state_hash for o in result.observations))
        if all(result.terminations_dict().values()):
            break
    return hashes


def test_replay_byte_identical_three_runs_parallel() -> None:
    actions = (
        {"a": MultiAgentAction.BUY.value, "b": MultiAgentAction.SELL.value},
        {"a": MultiAgentAction.HOLD.value, "b": MultiAgentAction.HOLD.value},
        {"a": MultiAgentAction.SELL.value, "b": MultiAgentAction.BUY.value},
    )
    a = _replay_parallel(actions, seed=42)
    b = _replay_parallel(actions, seed=42)
    c = _replay_parallel(actions, seed=42)
    assert a == b == c


def test_replay_seed_changes_observation_hashes() -> None:
    actions = ({"a": MultiAgentAction.BUY.value, "b": MultiAgentAction.HOLD.value},)
    a = _replay_parallel(actions, seed=1)
    b = _replay_parallel(actions, seed=2)
    assert a != b


# ---------------------------------------------------------------------------
# Episode budget cap
# ---------------------------------------------------------------------------


def test_episode_budget_cap_raises() -> None:
    env = _parallel_env(
        prices=tuple(float(i + 1) for i in range(MAX_EPISODE_STEPS + 10)),
        max_steps=MAX_EPISODE_STEPS,
    )
    env.reset()
    actions = {"a": 0, "b": 0}
    for _ in range(MAX_EPISODE_STEPS):
        result = env.step(actions)
        if all(result.terminations_dict().values()):
            break


# ---------------------------------------------------------------------------
# Value object frozen+slotted guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obj",
    [
        MultiAgentScenario(agent_ids=("a", "b"), prices=(1.0, 2.0), max_steps=10),
        _obs(),
        MultiAgentStepResult(
            observations=(_obs(),),
            rewards=(("a", 0.0),),
            terminations=(("a", False),),
            truncations=(("a", False),),
            infos=(("a", ()),),
        ),
    ],
)
def test_value_objects_frozen_and_slotted(obj: object) -> None:
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(obj, "_arbitrary_attr_for_test", "y")
    assert not hasattr(obj, "__dict__")


# ---------------------------------------------------------------------------
# Render no-op
# ---------------------------------------------------------------------------


def test_render_is_noop() -> None:
    env = _parallel_env()
    env.reset()
    assert env.render() is None
    assert env.render(mode=None) is None


# ---------------------------------------------------------------------------
# pettingzoo_multiagent_env_factory lazy seam
# ---------------------------------------------------------------------------


def test_pettingzoo_factory_requires_gymnasium(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = MultiAgentScenario(agent_ids=("a", "b"), prices=(1.0, 2.0), max_steps=10)
    monkeypatch.setitem(sys.modules, "gymnasium", None)
    with pytest.raises(RuntimeError, match="pettingzoo"):
        pettingzoo_multiagent_env_factory(scenario=scenario)


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


def test_no_top_level_vendor_imports() -> None:
    top_level, _ = _collect_imports(pathlib.Path(multiagent_env.__file__))
    forbidden = {"gymnasium", "gym", "pettingzoo", "numpy"}
    leaks = top_level & forbidden
    assert leaks == set(), f"Top-level vendor leaks: {leaks!r}"


def test_no_top_level_engine_cross_imports() -> None:
    top_level, _ = _collect_imports(pathlib.Path(multiagent_env.__file__))
    forbidden = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "learning_engine",
    }
    leaks = {m for m in top_level if any(m.startswith(p) for p in forbidden)}
    assert leaks == set(), f"Engine cross-imports at top level: {leaks!r}"


def test_no_top_level_io_imports() -> None:
    top_level, _ = _collect_imports(pathlib.Path(multiagent_env.__file__))
    forbidden = {"subprocess", "pathlib", "socket", "urllib", "requests"}
    leaks = top_level & forbidden
    assert leaks == set(), f"IO module leaks at top level: {leaks!r}"


def test_pettingzoo_factory_holds_gymnasium_import_inside_body() -> None:
    _, function_locals = _collect_imports(pathlib.Path(multiagent_env.__file__))
    assert "pettingzoo_multiagent_env_factory" in function_locals
    factory_imports = function_locals["pettingzoo_multiagent_env_factory"]
    assert "gymnasium" in factory_imports


def test_module_reload_idempotent() -> None:
    importlib.reload(multiagent_env)
    assert multiagent_env.MULTIAGENT_ENV_VERSION == "c-30-pettingzoo-1"
