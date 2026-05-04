"""Tests for the slow-loop continuous learner (D4)."""

from __future__ import annotations

import random

import pytest

from core.contracts.governance import SystemMode
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)
from intelligence_engine.learning import (
    SLOW_LOOP_VERSION,
    FeedbackSample,
    ParameterBounds,
    SlowLoopLearner,
)


def _bounds(
    *, lo: float = -1.0, hi: float = 1.0, step: float = 0.1, init: float = 0.0
) -> ParameterBounds:
    return ParameterBounds(lo=lo, hi=hi, step=step, initial=init)


def test_construction_requires_non_empty_bounds() -> None:
    with pytest.raises(ValueError):
        SlowLoopLearner({})


def test_initial_snapshot_carries_initial_values() -> None:
    learner = SlowLoopLearner({"alpha": _bounds(init=0.25)})
    snap = learner.tick()
    assert snap.values["alpha"] == 0.25
    assert snap.sample_counts["alpha"] == 0
    assert snap.version == SLOW_LOOP_VERSION
    # None freeze policy is the migration-default "unfrozen" sentinel.
    assert snap.frozen is False


def test_default_freeze_policy_is_none_so_updates_run() -> None:
    learner = SlowLoopLearner({"alpha": _bounds()})
    snap = learner.tick()
    assert snap.frozen is False


def test_positive_reward_pushes_value_upward_within_step() -> None:
    learner = SlowLoopLearner(
        {"alpha": _bounds(step=0.1, init=0.0)},
        ema_alpha=1.0,  # collapse EMA → straight reward
    )
    learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="alpha", reward=5.0)
    )
    snap = learner.tick()
    # ema = 5.0, magnitude clamped to step=0.1, sign positive ⇒ 0.0+0.1=0.1
    assert snap.values["alpha"] == pytest.approx(0.1)
    assert snap.ema["alpha"] == pytest.approx(5.0)
    assert snap.sample_counts["alpha"] == 1


def test_negative_reward_pushes_value_downward() -> None:
    learner = SlowLoopLearner(
        {"alpha": _bounds(step=0.1, init=0.0)},
        ema_alpha=1.0,
    )
    learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="alpha", reward=-2.0)
    )
    snap = learner.tick()
    assert snap.values["alpha"] == pytest.approx(-0.1)


def test_value_clamps_at_high_bound() -> None:
    learner = SlowLoopLearner(
        {"alpha": _bounds(lo=0.0, hi=0.5, step=1.0, init=0.4)},
        ema_alpha=1.0,
    )
    learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="alpha", reward=10.0)
    )
    snap = learner.tick()
    assert snap.values["alpha"] == 0.5


def test_value_clamps_at_low_bound() -> None:
    learner = SlowLoopLearner(
        {"alpha": _bounds(lo=-0.5, hi=0.5, step=1.0, init=-0.4)},
        ema_alpha=1.0,
    )
    learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="alpha", reward=-10.0)
    )
    snap = learner.tick()
    assert snap.values["alpha"] == -0.5


def test_freeze_policy_blocks_updates_but_drains_buffer() -> None:
    policy = LearningEvolutionFreezePolicy(
        mode=SystemMode.PAPER, operator_override=False
    )
    learner = SlowLoopLearner(
        {"alpha": _bounds(step=0.1, init=0.0)},
        freeze_policy=policy,
        ema_alpha=1.0,
    )
    learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="alpha", reward=5.0)
    )
    snap = learner.tick()
    assert snap.frozen is True
    assert snap.values["alpha"] == 0.0  # unchanged
    # Sample WAS counted — EMA is updated even when frozen so the
    # learner can resume the right gradient when unfrozen.
    assert snap.sample_counts["alpha"] == 1
    assert snap.ema["alpha"] == pytest.approx(5.0)


def test_live_mode_with_override_unfreezes() -> None:
    policy = LearningEvolutionFreezePolicy(
        mode=SystemMode.LIVE, operator_override=True
    )
    learner = SlowLoopLearner(
        {"alpha": _bounds(step=0.1, init=0.0)},
        freeze_policy=policy,
        ema_alpha=1.0,
    )
    learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="alpha", reward=5.0)
    )
    snap = learner.tick()
    assert snap.frozen is False
    assert snap.values["alpha"] == pytest.approx(0.1)


def test_unknown_parameter_is_rejected() -> None:
    learner = SlowLoopLearner({"alpha": _bounds()})
    accepted = learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="bogus", reward=1.0)
    )
    assert accepted is False


def test_replay_is_deterministic() -> None:
    bounds = {
        "alpha": _bounds(step=0.05, init=0.1),
        "beta": _bounds(step=0.05, init=-0.1),
    }
    samples = [
        FeedbackSample(ts_unix_s=10, parameter="alpha", reward=1.0),
        FeedbackSample(ts_unix_s=11, parameter="beta", reward=-0.5),
        FeedbackSample(ts_unix_s=12, parameter="alpha", reward=-2.0),
    ]

    def run() -> tuple[float, float]:
        learner = SlowLoopLearner(
            bounds,
            time_unix_s_provider=lambda: 100,
            rng=random.Random(7),
            ema_alpha=0.5,
            exploration_eps=0.1,  # exercise the PRNG path
        )
        learner.submit_many(samples)
        snap = learner.tick()
        return snap.values["alpha"], snap.values["beta"]

    assert run() == run()


def test_max_samples_cap_truncates_buffer_fifo() -> None:
    learner = SlowLoopLearner(
        {"alpha": _bounds()},
        max_samples_per_param=2,
    )
    for i in range(5):
        learner.submit(
            FeedbackSample(
                ts_unix_s=i, parameter="alpha", reward=float(i)
            )
        )
    snap = learner.tick()
    # Only the last 2 retained samples were folded into EMA; the
    # buffer must have been bounded.
    assert snap.sample_counts["alpha"] == 2


def test_invalid_ema_alpha_rejected() -> None:
    with pytest.raises(ValueError):
        SlowLoopLearner({"alpha": _bounds()}, ema_alpha=0.0)
    with pytest.raises(ValueError):
        SlowLoopLearner({"alpha": _bounds()}, ema_alpha=1.5)


def test_invalid_exploration_eps_rejected() -> None:
    with pytest.raises(ValueError):
        SlowLoopLearner({"alpha": _bounds()}, exploration_eps=-0.1)


def test_feedback_sample_rejects_bad_reward() -> None:
    with pytest.raises(ValueError):
        FeedbackSample(
            ts_unix_s=1, parameter="alpha", reward=float("inf")
        )


def test_feedback_sample_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError):
        FeedbackSample(
            ts_unix_s=1, parameter="alpha", reward=0.0, weight=0.0
        )


def test_parameter_bounds_validation() -> None:
    with pytest.raises(ValueError):
        ParameterBounds(lo=1.0, hi=0.0, step=0.1, initial=0.5)
    with pytest.raises(ValueError):
        ParameterBounds(lo=0.0, hi=1.0, step=0.0, initial=0.5)
    with pytest.raises(ValueError):
        ParameterBounds(lo=0.0, hi=1.0, step=0.1, initial=2.0)


def test_reset_restores_initial_state() -> None:
    learner = SlowLoopLearner(
        {"alpha": _bounds(init=0.2, step=0.1)},
        ema_alpha=1.0,
    )
    learner.submit(
        FeedbackSample(ts_unix_s=1, parameter="alpha", reward=5.0)
    )
    learner.tick()
    learner.reset()
    snap = learner.tick()
    assert snap.values["alpha"] == 0.2
    assert snap.ema["alpha"] == 0.0
    assert snap.sample_counts["alpha"] == 0
