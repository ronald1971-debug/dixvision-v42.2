"""Unit tests for the SIM-07 ParallelRunner harness.

Pure backend tests — no clock, no IO, no network. The step functions are
deterministic on (seed, scenario) so the whole suite is replay-stable.
"""

from __future__ import annotations

import math

import pytest

from core.contracts.simulation import (
    RealityOutcome,
    RealityScenario,
    RealitySummary,
)
from simulation.parallel_runner import (
    ParallelRunner,
    ParallelRunnerConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scenario(scenario_id: str = "scn-1", ts_ns: int = 1_000_000) -> RealityScenario:
    return RealityScenario(
        scenario_id=scenario_id,
        ts_ns=ts_ns,
        initial_state_hash="abc123",
        meta={"symbol": "BTC-USD"},
    )


def _step_constant(pnl: float, drawdown: float = 0.0, fills: int = 1):
    """Step function whose outcome ignores the seed (for sanity tests)."""

    def step(seed: int, scenario: RealityScenario) -> RealityOutcome:
        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed,
            pnl_usd=pnl,
            terminal_drawdown_usd=drawdown,
            fills_count=fills,
            rule_fired="constant",
        )

    return step


def _step_seed_pnl(seed: int, scenario: RealityScenario) -> RealityOutcome:
    """Step function whose pnl is a deterministic function of the seed.

    Allows distributional assertions without involving any PRNG.
    """

    return RealityOutcome(
        scenario_id=scenario.scenario_id,
        seed=seed,
        pnl_usd=float(seed),
        terminal_drawdown_usd=float(seed) * 0.1,
        fills_count=1,
        rule_fired="seed_pnl",
    )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def test_scenario_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        RealityScenario(
            scenario_id="",
            ts_ns=1,
            initial_state_hash="x",
        )
    with pytest.raises(ValueError, match="ts_ns"):
        RealityScenario(
            scenario_id="s",
            ts_ns=0,
            initial_state_hash="x",
        )
    with pytest.raises(ValueError, match="initial_state_hash"):
        RealityScenario(
            scenario_id="s",
            ts_ns=1,
            initial_state_hash="",
        )


def test_outcome_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="scenario_id"):
        RealityOutcome(
            scenario_id="",
            seed=0,
            pnl_usd=0.0,
            terminal_drawdown_usd=0.0,
            fills_count=0,
            rule_fired="r",
        )
    with pytest.raises(ValueError, match="seed"):
        RealityOutcome(
            scenario_id="s",
            seed=-1,
            pnl_usd=0.0,
            terminal_drawdown_usd=0.0,
            fills_count=0,
            rule_fired="r",
        )
    with pytest.raises(ValueError, match="terminal_drawdown_usd"):
        RealityOutcome(
            scenario_id="s",
            seed=0,
            pnl_usd=0.0,
            terminal_drawdown_usd=-1.0,
            fills_count=0,
            rule_fired="r",
        )
    with pytest.raises(ValueError, match="fills_count"):
        RealityOutcome(
            scenario_id="s",
            seed=0,
            pnl_usd=0.0,
            terminal_drawdown_usd=0.0,
            fills_count=-1,
            rule_fired="r",
        )
    with pytest.raises(ValueError, match="rule_fired"):
        RealityOutcome(
            scenario_id="s",
            seed=0,
            pnl_usd=0.0,
            terminal_drawdown_usd=0.0,
            fills_count=0,
            rule_fired="",
        )


def test_summary_rejects_unsorted_quantiles() -> None:
    with pytest.raises(ValueError, match="p05 <= pnl_median <= pnl_p95"):
        RealitySummary(
            scenario_id="s",
            n_realities=1,
            pnl_mean_usd=0.0,
            pnl_median_usd=10.0,
            pnl_p05_usd=20.0,
            pnl_p95_usd=5.0,
            win_rate=0.5,
            max_drawdown_usd=0.0,
        )


def test_runner_config_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="min_realities"):
        ParallelRunnerConfig(min_realities=0)
    with pytest.raises(ValueError, match="max_realities"):
        ParallelRunnerConfig(min_realities=10, max_realities=5)


# ---------------------------------------------------------------------------
# Runner behaviour
# ---------------------------------------------------------------------------


def test_run_rejects_empty_seeds() -> None:
    runner = ParallelRunner()
    with pytest.raises(ValueError, match="at least one seed"):
        runner.run(_scenario(), [], _step_constant(1.0))


def test_run_rejects_negative_seed() -> None:
    runner = ParallelRunner()
    with pytest.raises(ValueError, match="non-negative"):
        runner.run(_scenario(), [0, -1], _step_constant(1.0))


def test_run_rejects_duplicate_seeds() -> None:
    runner = ParallelRunner()
    with pytest.raises(ValueError, match="duplicate seed"):
        runner.run(_scenario(), [0, 1, 0], _step_constant(1.0))


def test_run_rejects_step_returning_wrong_scenario_id() -> None:
    def bad_step(seed: int, scenario: RealityScenario) -> RealityOutcome:
        return RealityOutcome(
            scenario_id="wrong-id",
            seed=seed,
            pnl_usd=0.0,
            terminal_drawdown_usd=0.0,
            fills_count=0,
            rule_fired="x",
        )

    runner = ParallelRunner()
    with pytest.raises(ValueError, match="scenario_id"):
        runner.run(_scenario(), [0], bad_step)


def test_run_rejects_step_returning_wrong_seed() -> None:
    def bad_step(seed: int, scenario: RealityScenario) -> RealityOutcome:
        return RealityOutcome(
            scenario_id=scenario.scenario_id,
            seed=seed + 1,
            pnl_usd=0.0,
            terminal_drawdown_usd=0.0,
            fills_count=0,
            rule_fired="x",
        )

    runner = ParallelRunner()
    with pytest.raises(ValueError, match="seed"):
        runner.run(_scenario(), [0], bad_step)


def test_run_rejects_too_few_seeds_below_min() -> None:
    runner = ParallelRunner(ParallelRunnerConfig(min_realities=3, max_realities=10))
    with pytest.raises(ValueError, match="too few seeds"):
        runner.run(_scenario(), [0, 1], _step_constant(1.0))


def test_run_rejects_too_many_seeds_above_max() -> None:
    runner = ParallelRunner(ParallelRunnerConfig(max_realities=2))
    with pytest.raises(ValueError, match="too many seeds"):
        runner.run(_scenario(), [0, 1, 2], _step_constant(1.0))


def test_run_outcomes_sorted_by_seed() -> None:
    runner = ParallelRunner()
    outcomes, _ = runner.run(_scenario(), [42, 7, 99, 0], _step_seed_pnl)
    assert [o.seed for o in outcomes] == [0, 7, 42, 99]


def test_run_summary_constant_pnl() -> None:
    runner = ParallelRunner()
    _, summary = runner.run(_scenario(), [0, 1, 2, 3, 4], _step_constant(10.0, 5.0))
    assert summary.n_realities == 5
    assert summary.pnl_mean_usd == 10.0
    assert summary.pnl_median_usd == 10.0
    assert summary.pnl_p05_usd == 10.0
    assert summary.pnl_p95_usd == 10.0
    assert summary.win_rate == 1.0
    assert summary.max_drawdown_usd == 5.0


def test_run_summary_quantiles_match_seed_pnl() -> None:
    runner = ParallelRunner()
    seeds = list(range(101))  # pnl == seed, so [0..100]
    _, summary = runner.run(_scenario(), seeds, _step_seed_pnl)
    assert summary.n_realities == 101
    assert math.isclose(summary.pnl_mean_usd, 50.0)
    assert math.isclose(summary.pnl_median_usd, 50.0)
    # p05 over [0..100] linear-interp lands at 5.0 exactly.
    assert math.isclose(summary.pnl_p05_usd, 5.0)
    assert math.isclose(summary.pnl_p95_usd, 95.0)
    # Wins = strictly greater than win_threshold_usd (0.0) → seeds 1..100.
    assert math.isclose(summary.win_rate, 100.0 / 101.0)
    assert math.isclose(summary.max_drawdown_usd, 100.0 * 0.1)


def test_run_win_rate_uses_threshold() -> None:
    runner = ParallelRunner(
        ParallelRunnerConfig(win_threshold_usd=50.0),
    )
    seeds = list(range(101))
    _, summary = runner.run(_scenario(), seeds, _step_seed_pnl)
    # pnls 51..100 strictly above 50 → 50 wins / 101.
    assert math.isclose(summary.win_rate, 50.0 / 101.0)


def test_run_replay_determinism() -> None:
    """Same scenario + same seeds + same step → identical outputs."""

    runner_a = ParallelRunner()
    runner_b = ParallelRunner()
    seeds = [11, 3, 7, 5]
    out_a, sum_a = runner_a.run(_scenario(), seeds, _step_seed_pnl)
    out_b, sum_b = runner_b.run(_scenario(), list(seeds), _step_seed_pnl)
    assert out_a == out_b
    assert sum_a == sum_b


def test_run_seed_order_invariant() -> None:
    """Reordering the input seed list must not change the outputs."""

    runner = ParallelRunner()
    out_a, sum_a = runner.run(_scenario(), [0, 1, 2], _step_seed_pnl)
    out_b, sum_b = runner.run(_scenario(), [2, 1, 0], _step_seed_pnl)
    assert out_a == out_b
    assert sum_a == sum_b


def test_run_single_reality() -> None:
    runner = ParallelRunner()
    outcomes, summary = runner.run(_scenario(), [42], _step_seed_pnl)
    assert len(outcomes) == 1
    assert summary.n_realities == 1
    assert summary.pnl_mean_usd == 42.0
    assert summary.pnl_median_usd == 42.0
    assert summary.pnl_p05_usd == 42.0
    assert summary.pnl_p95_usd == 42.0
