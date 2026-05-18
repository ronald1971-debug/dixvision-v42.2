# ADAPTED FROM: HypothesisWorks/hypothesis
#   - hypothesis/strategies/_internal/strategies.py — `@given`
#   - hypothesis/strategies/_internal/numbers.py — `integers`/`floats`
#   - hypothesis/strategies/_internal/collections.py — `lists`
# MPL-2.0 license; only the public strategy/decorator contract is used.
"""A-13 hypothesis → property-based replay-determinism invariants for
:class:`simulation.parallel_runner.ParallelRunner`.

Properties pinned:

1. **INV-15 byte-identical replay.** Two runs of the runner with the
   same scenario + same seed sequence + same step function produce
   tuples of :class:`RealityOutcome` that are equal field-for-field.
2. **Seed-order independence.** Two runs with the same scenario +
   permuted seed sequence + same step function produce the same
   sorted-by-seed outcome tuple (the runner sorts before returning,
   so this is the externally observable determinism contract).
3. **Summary aggregation determinism.** The :class:`RealitySummary`
   value is identical across replays.
4. **Step-fn purity contract.** When the step function is a pure
   function of ``(seed, scenario)``, every replayed outcome has the
   same ``terminal_pnl_usd`` / ``terminal_drawdown_usd`` /
   ``fills_count`` / ``rule_fired`` as the first run.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from core.contracts.simulation import (
    RealityOutcome,
    RealityScenario,
)
from simulation.parallel_runner import ParallelRunner, ParallelRunnerConfig

# ---------------------------------------------------------------------------
# Deterministic step function used by every property below.
# ---------------------------------------------------------------------------


def _pure_step(seed: int, scenario: RealityScenario) -> RealityOutcome:
    """Pure function: outcome depends only on ``(seed, scenario)``.

    Uses a local ``random.Random(seed)`` so the function is pure — it
    does not touch the global RNG and does not read the clock. INV-15
    replay is guaranteed by construction.
    """

    rng = random.Random((seed * 1_000_003) ^ scenario.ts_ns)
    pnl = rng.uniform(-100.0, 100.0)
    drawdown = abs(min(0.0, pnl)) + rng.uniform(0.0, 20.0)
    fills = rng.randint(0, 32)
    rule = "rule_a" if pnl >= 0.0 else "rule_b"
    return RealityOutcome(
        scenario_id=scenario.scenario_id,
        seed=seed,
        pnl_usd=pnl,
        terminal_drawdown_usd=drawdown,
        fills_count=fills,
        rule_fired=rule,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def _scenarios(draw: st.DrawFn) -> RealityScenario:
    return RealityScenario(
        scenario_id=draw(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("Ll", "Lu", "Nd"),
                    whitelist_characters="_-",
                ),
                min_size=1,
                max_size=16,
            )
        ),
        ts_ns=draw(st.integers(min_value=1, max_value=2**62)),
        initial_state_hash=draw(st.text(min_size=1, max_size=32, alphabet="0123456789abcdef")),
    )


_SEED_LISTS = st.lists(
    st.integers(min_value=0, max_value=2**31 - 1),
    min_size=1,
    max_size=24,
    unique=True,
)


# ---------------------------------------------------------------------------
# Property 1: identical inputs → identical outcomes
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(scenario=_scenarios(), seeds=_SEED_LISTS)
def test_parallel_runner_replay_is_byte_identical(
    scenario: RealityScenario, seeds: Sequence[int]
) -> None:
    runner = ParallelRunner(ParallelRunnerConfig())
    outcomes_a, summary_a = runner.run(scenario, list(seeds), _pure_step)
    outcomes_b, summary_b = runner.run(scenario, list(seeds), _pure_step)
    assert outcomes_a == outcomes_b
    assert summary_a == summary_b


# ---------------------------------------------------------------------------
# Property 2: permuted seed list → same sorted outcome tuple
# ---------------------------------------------------------------------------


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(scenario=_scenarios(), seeds=_SEED_LISTS)
def test_parallel_runner_is_insensitive_to_seed_order(
    scenario: RealityScenario, seeds: Sequence[int]
) -> None:
    forward = list(seeds)
    reverse = list(reversed(seeds))

    runner = ParallelRunner(ParallelRunnerConfig())
    out_fwd, sum_fwd = runner.run(scenario, forward, _pure_step)
    out_rev, sum_rev = runner.run(scenario, reverse, _pure_step)

    # ``ParallelRunner.run`` returns outcomes sorted by seed; the
    # two runs must therefore agree element-for-element.
    assert out_fwd == out_rev
    assert sum_fwd == sum_rev


# ---------------------------------------------------------------------------
# Property 3: summary fields are deterministic across replays
# ---------------------------------------------------------------------------


@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(scenario=_scenarios(), seeds=_SEED_LISTS)
def test_parallel_runner_summary_is_deterministic(
    scenario: RealityScenario, seeds: Sequence[int]
) -> None:
    runner = ParallelRunner(ParallelRunnerConfig())
    _, summary_a = runner.run(scenario, list(seeds), _pure_step)
    _, summary_b = runner.run(scenario, list(seeds), _pure_step)
    assert summary_a == summary_b
