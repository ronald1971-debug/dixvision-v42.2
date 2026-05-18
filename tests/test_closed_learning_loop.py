"""Tests for :class:`ClosedLearningLoop` (P0-A).

Pins:

* HARDEN-04 / INV-70 — the loop refuses to drive the learner / emitter
  when the live :class:`LearningEvolutionFreezePolicy` is frozen.
* The loop is the **single freeze gate** — the inner
  :class:`SlowLoopLearner` + :class:`UpdateEmitter` must be constructed
  with ``freeze=None``; otherwise the constructor raises ``ValueError``.
* INV-15 byte-identical replay: same drained outcomes + same policy
  snapshots → byte-identical :class:`LoopTickResult` sequence.
* B27 / B28 / INV-71 — the loop never constructs typed events
  directly; emissions go through the :class:`UpdateEmitter` (pinned by
  an AST scan over ``learning_engine/loops/closed_loop.py``).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from core.contracts.events import ExecutionStatus, SystemEventKind
from core.contracts.governance import SystemMode
from core.contracts.learning import LearningUpdate, TradeOutcome
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)
from execution_engine.protections.feedback import FeedbackCollector
from intelligence_engine.learning.slow_loop import (
    FeedbackSample,
    ParameterBounds,
    ParameterSnapshot,
    SlowLoopLearner,
)
from learning_engine.loops.closed_loop import (
    ClosedLearningLoop,
    LoopTickResult,
)
from learning_engine.update_emitter import UpdateEmitter

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "learning_engine" / "loops" / "closed_loop.py"
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _frozen_policy() -> LearningEvolutionFreezePolicy:
    return LearningEvolutionFreezePolicy(mode=SystemMode.PAPER, operator_override=False)


def _unfrozen_policy() -> LearningEvolutionFreezePolicy:
    return LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=True)


def _live_no_override() -> LearningEvolutionFreezePolicy:
    return LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=False)


def _make_collector_with(
    outcomes: tuple[TradeOutcome, ...],
) -> FeedbackCollector:
    fc = FeedbackCollector()
    fc.extend(outcomes)
    return fc


def _outcome(*, ts_ns: int, pnl: float, strategy_id: str = "alpha") -> TradeOutcome:
    return TradeOutcome(
        ts_ns=ts_ns,
        strategy_id=strategy_id,
        symbol="BTCUSD",
        qty=1.0,
        pnl=pnl,
        status=ExecutionStatus.FILLED,
        venue="paper",
        order_id=f"o-{ts_ns}",
        meta={},
    )


def _learner() -> SlowLoopLearner:
    return SlowLoopLearner(
        bounds={"alpha_threshold": ParameterBounds(lo=0.0, hi=1.0, step=0.1, initial=0.5)},
        freeze_policy=None,
    )


def _sample_builder(
    outcomes: tuple[TradeOutcome, ...],
) -> tuple[FeedbackSample, ...]:
    return tuple(
        FeedbackSample(
            ts_unix_s=int(o.ts_ns // 1_000_000_000),
            parameter="alpha_threshold",
            reward=o.pnl,
        )
        for o in outcomes
    )


def _update_builder(
    previous: ParameterSnapshot | None,
    current: ParameterSnapshot,
    ts_ns: int,
) -> tuple[LearningUpdate, ...]:
    prev_value = previous.values.get("alpha_threshold", 0.5) if previous is not None else 0.5
    cur_value = current.values["alpha_threshold"]
    if prev_value == cur_value:
        return ()
    return (
        LearningUpdate(
            ts_ns=ts_ns,
            strategy_id="alpha",
            parameter="alpha_threshold",
            old_value=f"{prev_value:.6f}",
            new_value=f"{cur_value:.6f}",
            reason="ema_gradient",
            meta={},
        ),
    )


def _make_loop(
    *,
    feedback: FeedbackCollector,
    policy: LearningEvolutionFreezePolicy,
    learner: SlowLoopLearner | None = None,
    emitter: UpdateEmitter | None = None,
) -> ClosedLearningLoop:
    return ClosedLearningLoop(
        feedback_collector=feedback,
        learner=learner or _learner(),
        emitter=emitter or UpdateEmitter(source="learning_test"),
        policy_supplier=lambda: policy,
        sample_builder=_sample_builder,
        update_builder=_update_builder,
    )


# ---------------------------------------------------------------------------
# Constructor invariants
# ---------------------------------------------------------------------------


def test_loop_refuses_learner_with_inner_freeze_policy() -> None:
    bad_learner = SlowLoopLearner(
        bounds={"alpha_threshold": ParameterBounds(lo=0.0, hi=1.0, step=0.1, initial=0.5)},
        freeze_policy=_unfrozen_policy(),
    )
    with pytest.raises(ValueError, match="learner.freeze_policy=None"):
        ClosedLearningLoop(
            feedback_collector=FeedbackCollector(),
            learner=bad_learner,
            emitter=UpdateEmitter(source="learning_test"),
            policy_supplier=_unfrozen_policy,
        )


def test_loop_refuses_emitter_with_inner_freeze_policy() -> None:
    bad_emitter = UpdateEmitter(source="learning_test", freeze=_unfrozen_policy())
    with pytest.raises(ValueError, match="emitter.freeze=None"):
        ClosedLearningLoop(
            feedback_collector=FeedbackCollector(),
            learner=_learner(),
            emitter=bad_emitter,
            policy_supplier=_unfrozen_policy,
        )


# ---------------------------------------------------------------------------
# Frozen path
# ---------------------------------------------------------------------------


def test_frozen_default_paper_mode_is_noop() -> None:
    collector = _make_collector_with((_outcome(ts_ns=1, pnl=1.0),))
    loop = _make_loop(feedback=collector, policy=_frozen_policy())
    result = loop.tick(ts_ns=10_000)
    assert isinstance(result, LoopTickResult)
    assert result.frozen is True
    assert result.policy_mode_name == "PAPER"
    assert result.operator_override is False
    assert len(result.drained_outcomes) == 1
    assert result.submitted_samples == ()
    assert result.snapshot is None
    assert result.emitted_events == ()
    assert len(collector) == 0
    assert loop.previous_snapshot is None


def test_frozen_live_without_override_is_noop() -> None:
    collector = _make_collector_with((_outcome(ts_ns=1, pnl=1.0), _outcome(ts_ns=2, pnl=-0.5)))
    loop = _make_loop(feedback=collector, policy=_live_no_override())
    result = loop.tick(ts_ns=10_000)
    assert result.frozen is True
    assert result.policy_mode_name == "LIVE"
    assert result.operator_override is False
    assert len(result.drained_outcomes) == 2
    assert result.emitted_events == ()


# ---------------------------------------------------------------------------
# Unfrozen path
# ---------------------------------------------------------------------------


def test_live_plus_override_drains_and_emits() -> None:
    collector = _make_collector_with(
        (
            _outcome(ts_ns=1_000_000_000, pnl=1.5),
            _outcome(ts_ns=2_000_000_000, pnl=2.0),
        )
    )
    loop = _make_loop(feedback=collector, policy=_unfrozen_policy())
    result = loop.tick(ts_ns=10_000)
    assert result.frozen is False
    assert result.policy_mode_name == "LIVE"
    assert result.operator_override is True
    assert len(result.drained_outcomes) == 2
    assert len(result.submitted_samples) == 2
    assert result.snapshot is not None
    assert result.snapshot.frozen is False
    assert len(result.emitted_events) == 1
    assert result.emitted_events[0].sub_kind == SystemEventKind.UPDATE_PROPOSED
    assert result.emitted_events[0].ts_ns == 10_000
    assert loop.previous_snapshot is not None


def test_empty_drain_unfrozen_still_ticks_learner() -> None:
    loop = _make_loop(feedback=FeedbackCollector(), policy=_unfrozen_policy())
    result = loop.tick(ts_ns=42)
    assert result.frozen is False
    assert result.drained_outcomes == ()
    assert result.submitted_samples == ()
    assert result.snapshot is not None
    assert result.emitted_events == ()
    assert loop.previous_snapshot is not None


# ---------------------------------------------------------------------------
# Mode flip mid-loop
# ---------------------------------------------------------------------------


def test_mode_flip_mid_loop() -> None:
    collector = _make_collector_with((_outcome(ts_ns=1_000_000_000, pnl=2.0),))
    policies: list[LearningEvolutionFreezePolicy] = [
        _frozen_policy(),
        _unfrozen_policy(),
        _frozen_policy(),
    ]
    iter_policies = iter(policies)
    loop = ClosedLearningLoop(
        feedback_collector=collector,
        learner=_learner(),
        emitter=UpdateEmitter(source="learning_test"),
        policy_supplier=lambda: next(iter_policies),
        sample_builder=_sample_builder,
        update_builder=_update_builder,
    )
    r1 = loop.tick(ts_ns=100)
    assert r1.frozen is True
    assert len(r1.drained_outcomes) == 1
    collector.extend((_outcome(ts_ns=3_000_000_000, pnl=1.0),))
    r2 = loop.tick(ts_ns=200)
    assert r2.frozen is False
    assert len(r2.submitted_samples) == 1
    r3 = loop.tick(ts_ns=300)
    assert r3.frozen is True
    assert r3.emitted_events == ()


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay
# ---------------------------------------------------------------------------


def _run_three_ticks() -> tuple[LoopTickResult, ...]:
    collector = _make_collector_with(
        (
            _outcome(ts_ns=1_000_000_000, pnl=1.5),
            _outcome(ts_ns=2_000_000_000, pnl=2.0),
            _outcome(ts_ns=3_000_000_000, pnl=-0.25),
        )
    )
    loop = _make_loop(feedback=collector, policy=_unfrozen_policy())
    return tuple(loop.tick(ts_ns=t) for t in (10, 20, 30))


def test_byte_identical_replay() -> None:
    a = _run_three_ticks()
    b = _run_three_ticks()
    c = _run_three_ticks()
    assert a == b == c


# ---------------------------------------------------------------------------
# B27 / B28 / INV-71 — AST scan of the loop module
# ---------------------------------------------------------------------------


def _module_ast() -> ast.AST:
    return ast.parse(_MODULE_PATH.read_text())


def test_loop_module_constructs_no_typed_events() -> None:
    tree = _module_ast()
    bad_constructors = {
        "PatchProposal",
        "GovernanceDecision",
        "SignalEvent",
        "ExecutionEvent",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = (
                fn.id
                if isinstance(fn, ast.Name)
                else (fn.attr if isinstance(fn, ast.Attribute) else None)
            )
            assert name not in bad_constructors, f"closed_loop must not construct {name}"


def test_loop_module_does_not_import_engines() -> None:
    tree = _module_ast()
    forbidden_prefixes = (
        "governance_engine.",
        "system_engine.",
        "evolution_engine.",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith(forbidden_prefixes), (
                f"closed_loop must not import {node.module}"
            )
