"""Tests for B-01.1 distributed runner.

Coverage:

* :class:`DistributedRunnerConfig` validation
* :class:`InProcessWorkerExecutor` drops in for the Ray executor
* :class:`DistributedRunner` matches :class:`ParallelRunner` byte-for-byte
* Seed-validation parity with parallel runner (negative / duplicate)
* Min / max reality bounds
* Executor scheduling order MUST NOT leak into outcome tuple
* Outcome ``scenario_id`` and ``seed`` validation
* Executor returning the wrong count is rejected
* INV-15 byte-identical replay equality (3 runs)
* AST guards: no engine cross-imports, no typed bus event
  construction, no top-level ``ray`` import, no forbidden imports,
  ``# ADAPTED FROM`` header present
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest

from core.contracts.simulation import (
    RealityOutcome,
    RealityScenario,
)
from simulation.distributed_runner import (
    NEW_PIP_DEPENDENCIES,
    DistributedRunner,
    DistributedRunnerConfig,
    InProcessWorkerExecutor,
    WorkerExecutor,
)
from simulation.parallel_runner import (
    ParallelRunner,
    ParallelRunnerConfig,
    StepFn,
)

_MODULE_PATH = Path(__file__).resolve().parent.parent / "simulation" / "distributed_runner.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario(scenario_id: str = "scn-1") -> RealityScenario:
    return RealityScenario(
        scenario_id=scenario_id,
        ts_ns=1,
        initial_state_hash="abc123",
    )


def _step(seed: int, scenario: RealityScenario) -> RealityOutcome:
    """Deterministic step — pnl scales linearly with seed."""
    return RealityOutcome(
        scenario_id=scenario.scenario_id,
        seed=seed,
        pnl_usd=float(seed - 5),
        terminal_drawdown_usd=float(seed % 3),
        fills_count=seed,
        rule_fired="test-rule",
    )


def _bad_scenario_step(seed: int, scenario: RealityScenario) -> RealityOutcome:
    return RealityOutcome(
        scenario_id="wrong",
        seed=seed,
        pnl_usd=0.0,
        terminal_drawdown_usd=0.0,
        fills_count=0,
        rule_fired="test-rule",
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_defaults_valid() -> None:
    cfg = DistributedRunnerConfig()
    assert cfg.min_realities == 1
    assert cfg.num_workers >= 1


def test_config_rejects_min_realities_zero() -> None:
    with pytest.raises(ValueError):
        DistributedRunnerConfig(min_realities=0)


def test_config_rejects_max_below_min() -> None:
    with pytest.raises(ValueError):
        DistributedRunnerConfig(min_realities=5, max_realities=3)


def test_config_rejects_workers_zero() -> None:
    with pytest.raises(ValueError):
        DistributedRunnerConfig(num_workers=0)


def test_config_as_parallel_projection() -> None:
    cfg = DistributedRunnerConfig(
        min_realities=2,
        max_realities=42,
        win_threshold_usd=1.5,
        num_workers=8,
    )
    proj = cfg.as_parallel()
    assert isinstance(proj, ParallelRunnerConfig)
    assert proj.min_realities == 2
    assert proj.max_realities == 42
    assert proj.win_threshold_usd == 1.5


# ---------------------------------------------------------------------------
# In-process executor + Protocol
# ---------------------------------------------------------------------------


def test_in_process_executor_satisfies_protocol() -> None:
    executor = InProcessWorkerExecutor()
    assert isinstance(executor, WorkerExecutor)


def test_in_process_executor_maps_in_order() -> None:
    executor = InProcessWorkerExecutor()
    result = list(executor.map(_step, _scenario(), [0, 1, 2]))
    assert [o.seed for o in result] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Drop-in parity with ParallelRunner
# ---------------------------------------------------------------------------


def test_distributed_runner_matches_parallel_runner_in_process() -> None:
    seeds = [3, 1, 7, 5, 2]
    p_outcomes, p_summary = ParallelRunner().run(_scenario(), seeds, _step)
    d_outcomes, d_summary = DistributedRunner().run(_scenario(), seeds, _step)
    assert p_outcomes == d_outcomes
    assert p_summary == d_summary


def test_distributed_runner_sorts_outcomes_by_seed() -> None:
    outcomes, _ = DistributedRunner().run(_scenario(), [9, 1, 4, 2], _step)
    assert [o.seed for o in outcomes] == [1, 2, 4, 9]


# ---------------------------------------------------------------------------
# Seed validation
# ---------------------------------------------------------------------------


def test_empty_seeds_rejected() -> None:
    with pytest.raises(ValueError):
        DistributedRunner().run(_scenario(), [], _step)


def test_negative_seed_rejected() -> None:
    with pytest.raises(ValueError):
        DistributedRunner().run(_scenario(), [1, -1, 2], _step)


def test_duplicate_seed_rejected() -> None:
    with pytest.raises(ValueError):
        DistributedRunner().run(_scenario(), [1, 1, 2], _step)


def test_too_few_seeds_rejected() -> None:
    runner = DistributedRunner(DistributedRunnerConfig(min_realities=5))
    with pytest.raises(ValueError):
        runner.run(_scenario(), [1, 2], _step)


def test_too_many_seeds_rejected() -> None:
    runner = DistributedRunner(DistributedRunnerConfig(max_realities=2))
    with pytest.raises(ValueError):
        runner.run(_scenario(), [1, 2, 3], _step)


# ---------------------------------------------------------------------------
# Outcome validation
# ---------------------------------------------------------------------------


def test_step_returning_wrong_scenario_id_rejected() -> None:
    with pytest.raises(ValueError):
        DistributedRunner().run(_scenario(), [1, 2], _bad_scenario_step)


def test_step_returning_wrong_seed_rejected() -> None:
    def wrong_seed_step(seed: int, scenario: RealityScenario) -> RealityOutcome:
        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed + 100,  # wrong
            pnl_usd=0.0,
            terminal_drawdown_usd=0.0,
            fills_count=0,
            rule_fired="test-rule",
        )

    with pytest.raises(ValueError):
        DistributedRunner().run(_scenario(), [1, 2], wrong_seed_step)


# ---------------------------------------------------------------------------
# Executor scheduling-order independence
# ---------------------------------------------------------------------------


class _ShuffleExecutor:
    """Returns outcomes in reverse order to model out-of-order Ray scheduling."""

    def map(
        self,
        step: StepFn,
        scenario: RealityScenario,
        seeds: Sequence[int],
    ) -> Iterable[RealityOutcome]:
        return list(reversed([step(s, scenario) for s in seeds]))


class _ShortExecutor:
    """Drops the last outcome — runner must detect the mismatch."""

    def map(
        self,
        step: StepFn,
        scenario: RealityScenario,
        seeds: Sequence[int],
    ) -> Iterable[RealityOutcome]:
        outs = [step(s, scenario) for s in seeds]
        return outs[:-1]


class _DuplicateExecutor:
    """Returns one duplicate seed — runner must detect."""

    def map(
        self,
        step: StepFn,
        scenario: RealityScenario,
        seeds: Sequence[int],
    ) -> Iterable[RealityOutcome]:
        outs = [step(s, scenario) for s in seeds]
        # Replace last with a clone of first to introduce a duplicate seed
        outs[-1] = outs[0]
        return outs


def test_out_of_order_executor_yields_sorted_output() -> None:
    runner = DistributedRunner(executor=_ShuffleExecutor())
    outcomes, _ = runner.run(_scenario(), [3, 1, 4, 2], _step)
    assert [o.seed for o in outcomes] == [1, 2, 3, 4]


def test_executor_count_mismatch_rejected() -> None:
    runner = DistributedRunner(executor=_ShortExecutor())
    with pytest.raises(ValueError):
        runner.run(_scenario(), [1, 2, 3], _step)


def test_executor_duplicate_seed_rejected() -> None:
    runner = DistributedRunner(executor=_DuplicateExecutor())
    with pytest.raises(ValueError):
        runner.run(_scenario(), [1, 2, 3], _step)


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def test_three_run_byte_identical_replay() -> None:
    runs = [DistributedRunner().run(_scenario(), [3, 1, 7, 5, 2], _step) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_replay_under_shuffle_executor_byte_identical() -> None:
    in_order = DistributedRunner().run(_scenario(), [3, 1, 7, 5, 2], _step)
    shuffled = DistributedRunner(executor=_ShuffleExecutor()).run(
        _scenario(), [3, 1, 7, 5, 2], _step
    )
    assert in_order == shuffled


# ---------------------------------------------------------------------------
# Ray factory surface (lazy)
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declares_ray() -> None:
    assert NEW_PIP_DEPENDENCIES == ("ray[default]",)


def test_ray_factory_does_not_import_ray_at_module_level() -> None:
    # Loading the module must succeed without ray installed; the
    # factory itself is the only place that imports it.
    import importlib

    mod = importlib.import_module("simulation.distributed_runner")
    assert hasattr(mod, "ray_worker_executor_factory")


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


def _module_tree() -> ast.AST:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _top_level_imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def _all_imports(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
    return out


def _call_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


def test_no_top_level_ray_import() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    top = _top_level_imports(tree)
    for forbidden in ("ray", "ray.rllib", "ray.actor", "ray.remote_function"):
        assert forbidden not in top, f"ray must be lazy-imported, found top-level: {forbidden}"


def test_ray_only_imported_inside_factory_body() -> None:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    # Find the factory function
    factory: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "ray_worker_executor_factory":
            factory = node
            break
    assert factory is not None
    body_imports: set[str] = set()
    for node in ast.walk(factory):
        if isinstance(node, ast.Import):
            for alias in node.names:
                body_imports.add(alias.name)
    assert "ray" in body_imports


def test_no_forbidden_imports() -> None:
    mods = _all_imports(_module_tree())
    for forbidden in (
        "random",
        "time",
        "datetime",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
        "pandas",
    ):
        assert forbidden not in mods, f"forbidden import: {forbidden}"


def test_no_engine_cross_imports() -> None:
    mods = _all_imports(_module_tree())
    for forbidden in (
        "governance_engine",
        "system_engine",
        "intelligence_engine",
        "evolution_engine",
        "learning_engine",
        "execution_engine",
    ):
        for m in mods:
            assert not m.startswith(forbidden), f"forbidden engine import: {m}"


def test_no_typed_bus_event_construction() -> None:
    calls = _call_names(_module_tree())
    for forbidden in (
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "GovernanceDecision",
        "LearningUpdate",
        "PatchProposal",
        "TraderObservation",
    ):
        assert forbidden not in calls, f"forbidden constructor call: {forbidden}"


def test_adapted_from_header_present() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "# ADAPTED FROM: ray-project/ray" in src
