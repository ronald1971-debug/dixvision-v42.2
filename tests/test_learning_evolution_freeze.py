"""HARDEN-04 — LearningEvolutionFreeze policy (INV-70).

Locks the contract that adaptive mutations are refused unless the
operator has explicitly armed the override flag. Pairs with the
HARDEN-01 / 02 / 03 runtime defences of the Triad Lock.

Contract version: ``v42.2-P0-RELAX``. Under direct operator directive
the ``mode is LIVE`` half of the previous dual gate was dropped —
the single freeze predicate is now ``operator_override is True``.
This test module pins the relaxed semantics. The execution-side
safety chain (kill switch, ``RiskSnapshot.halted``, hazard throttle,
FSM consent envelopes) is unchanged by this relaxation and is
pinned in its own dedicated test modules.
"""

from __future__ import annotations

import pytest

from core.contracts.governance import SystemMode
from core.contracts.learning import LearningUpdate
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
    LearningEvolutionFrozenError,
    assert_unfrozen,
    is_unfrozen,
)
from evolution_engine.intelligence_loops.mutation_proposer import (
    MutationProposer,
    MutationThresholds,
)
from learning_engine.update_emitter import UpdateEmitter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _live_unfrozen() -> LearningEvolutionFreezePolicy:
    return LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=True)


def _learning_update(*, parameter: str = "consensus_weight") -> LearningUpdate:
    return LearningUpdate(
        ts_ns=1_000_000_000,
        strategy_id="strat-1",
        parameter=parameter,
        old_value="0.50",
        new_value="0.51",
        reason="reward_corr",
        meta={"source": "weight_adjuster"},
    )


def _stats_breaching(*, strategy_id: str = "strat-1"):
    from core.contracts.learning import StrategyStats

    return StrategyStats(
        ts_ns=1_000_000_000,
        strategy_id=strategy_id,
        n_trades=100,
        n_wins=10,
        n_losses=90,
        total_pnl=-1.0,
        mean_pnl=-0.01,
        win_rate=0.10,
    )


# ---------------------------------------------------------------------------
# Policy semantics — ``v42.2-P0-RELAX`` operator-gated freeze predicate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.AUTO,
        SystemMode.LOCKED,
    ],
)
def test_policy_is_unfrozen_in_every_non_live_mode_with_override(mode: SystemMode) -> None:
    """Relaxed predicate: override=True unfreezes regardless of mode.

    Under ``v42.2-P0-RELAX`` the ``mode is LIVE`` predicate was
    dropped per direct operator directive; the freeze gate is
    ``operator_override`` alone.
    """

    policy = LearningEvolutionFreezePolicy(mode=mode, operator_override=True)
    assert policy.is_frozen() is False
    assert policy.is_unfrozen() is True


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
        SystemMode.LOCKED,
    ],
)
def test_policy_is_frozen_in_every_mode_without_override(mode: SystemMode) -> None:
    policy = LearningEvolutionFreezePolicy(mode=mode, operator_override=False)
    assert policy.is_frozen() is True
    assert policy.is_unfrozen() is False


def test_policy_is_unfrozen_in_live_with_explicit_override() -> None:
    """LIVE + override=True is still the canonical unfrozen state.

    The relaxation extends unfreezing to non-LIVE modes; it does not
    revoke unfreezing in LIVE.
    """

    policy = LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=True)
    assert policy.is_frozen() is False
    assert policy.is_unfrozen() is True


def test_policy_default_operator_override_is_false() -> None:
    # Defensive — operator_override must default to False so unfreezing
    # is always an explicit act.
    policy = LearningEvolutionFreezePolicy(mode=SystemMode.LIVE)
    assert policy.operator_override is False
    assert policy.is_frozen() is True


def test_policy_is_immutable() -> None:
    from dataclasses import FrozenInstanceError

    policy = LearningEvolutionFreezePolicy(mode=SystemMode.LIVE)
    with pytest.raises(FrozenInstanceError):
        policy.operator_override = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# assert_unfrozen / is_unfrozen
# ---------------------------------------------------------------------------


def test_assert_unfrozen_passes_with_unfrozen_policy() -> None:
    assert_unfrozen(_live_unfrozen(), action="emit_update")


def test_assert_unfrozen_passes_with_none_policy_for_backwards_compat() -> None:
    # None = "no policy wired yet" migration sentinel; intentionally permissive.
    assert_unfrozen(None, action="emit_update")


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,  # without override
        SystemMode.AUTO,
        SystemMode.LOCKED,
    ],
)
def test_assert_unfrozen_raises_for_frozen_policy(mode: SystemMode) -> None:
    policy = LearningEvolutionFreezePolicy(mode=mode, operator_override=False)
    with pytest.raises(LearningEvolutionFrozenError) as excinfo:
        assert_unfrozen(policy, action="emit_update")
    msg = str(excinfo.value)
    assert "emit_update" in msg
    assert mode.name in msg
    assert "operator_override=False" in msg


def test_assert_unfrozen_raises_for_live_without_override() -> None:
    policy = LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=False)
    with pytest.raises(LearningEvolutionFrozenError):
        assert_unfrozen(policy, action="propose_patch")


def test_is_unfrozen_returns_true_for_unfrozen() -> None:
    assert is_unfrozen(_live_unfrozen()) is True


def test_is_unfrozen_returns_true_for_none_policy() -> None:
    assert is_unfrozen(None) is True


def test_is_unfrozen_returns_false_for_frozen_policy() -> None:
    assert is_unfrozen(LearningEvolutionFreezePolicy(mode=SystemMode.PAPER)) is False


# ---------------------------------------------------------------------------
# UpdateEmitter integration
# ---------------------------------------------------------------------------


def test_update_emitter_without_freeze_policy_preserves_backwards_compat() -> None:
    emitter = UpdateEmitter()
    event = emitter.emit(_learning_update())
    assert event.payload["strategy_id"] == "strat-1"


def test_update_emitter_with_unfrozen_policy_emits() -> None:
    emitter = UpdateEmitter(freeze=_live_unfrozen())
    event = emitter.emit(_learning_update())
    assert event.payload["strategy_id"] == "strat-1"


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
        SystemMode.LOCKED,
    ],
)
def test_update_emitter_with_frozen_policy_raises(mode: SystemMode) -> None:
    """Frozen ⇔ ``operator_override is False`` — every mode raises."""

    emitter = UpdateEmitter(
        freeze=LearningEvolutionFreezePolicy(mode=mode, operator_override=False)
    )
    with pytest.raises(LearningEvolutionFrozenError) as excinfo:
        emitter.emit(_learning_update())
    assert "emit_update" in str(excinfo.value)


def test_update_emitter_emit_many_short_circuits_on_freeze() -> None:
    emitter = UpdateEmitter(freeze=LearningEvolutionFreezePolicy(mode=SystemMode.PAPER))
    with pytest.raises(LearningEvolutionFrozenError):
        emitter.emit_many((_learning_update(), _learning_update(parameter="b")))


# ---------------------------------------------------------------------------
# MutationProposer integration
# ---------------------------------------------------------------------------


def test_mutation_proposer_without_freeze_policy_preserves_backwards_compat() -> None:
    proposer = MutationProposer(thresholds=MutationThresholds(min_trades=1, min_win_rate=0.5))
    out = proposer.evaluate(_stats_breaching())
    assert len(out) >= 1


def test_mutation_proposer_with_unfrozen_policy_proposes() -> None:
    proposer = MutationProposer(
        thresholds=MutationThresholds(min_trades=1, min_win_rate=0.5),
        freeze=_live_unfrozen(),
    )
    out = proposer.evaluate(_stats_breaching())
    assert len(out) >= 1


@pytest.mark.parametrize(
    "mode",
    [
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
        SystemMode.LOCKED,
    ],
)
def test_mutation_proposer_with_frozen_policy_raises(mode: SystemMode) -> None:
    """Frozen ⇔ ``operator_override is False`` — every mode raises."""

    proposer = MutationProposer(
        thresholds=MutationThresholds(min_trades=1, min_win_rate=0.5),
        freeze=LearningEvolutionFreezePolicy(mode=mode, operator_override=False),
    )
    with pytest.raises(LearningEvolutionFrozenError) as excinfo:
        proposer.evaluate(_stats_breaching())
    assert "propose_patch" in str(excinfo.value)


def test_mutation_proposer_with_live_no_override_raises() -> None:
    proposer = MutationProposer(
        thresholds=MutationThresholds(min_trades=1, min_win_rate=0.5),
        freeze=LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=False),
    )
    with pytest.raises(LearningEvolutionFrozenError):
        proposer.evaluate(_stats_breaching())
