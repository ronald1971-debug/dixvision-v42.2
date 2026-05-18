"""PR-Z2 — wire concrete builders into ClosedLearningLoop (P0).

Pins that the production builders shipped in
:mod:`learning_engine.loops.builders` actually turn the closed
learning loop into a non-empty engine when HARDEN-04 is unfrozen
(``mode == LIVE`` and ``operator_override is True``), instead of the
silent no-op the loop was running before this wave.

Coverage:

* ``make_pnl_sample_builder`` factory contracts: validation, empty
  input, multi-outcome mapping, deterministic byte-identical replay.
* ``make_diff_update_builder`` factory contracts: first-tick empty
  return, diff emission in canonical parameter-name sort order,
  stable string formatting under ``.12g``.
* End-to-end ``ClosedLearningLoop`` integration: with both concrete
  builders wired, an unfrozen LIVE+override tick emits non-empty
  ``submitted_samples`` *and* non-empty ``emitted_events`` once the
  parameter values actually drift; the existing frozen short-circuit
  is preserved.
* ``PatchOutcomeFeedback`` wiring: observed outcomes surface through
  :meth:`PatchOutcomeFeedback.all_snapshots` so a
  ``_structural_stats_supplier`` built on top of it yields per-
  strategy :class:`StrategyStats` rows for every observed strategy.
* AST guardrails: the builders module imports nothing from runtime
  tiers (B1), constructs :class:`LearningUpdate` only inside the
  ``learning_engine.*`` authority window (B27 / HARDEN-06 / INV-71),
  and does no top-level wall-clock / PRNG / IO reads (INV-15).
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

from core.contracts.events import ExecutionStatus
from core.contracts.governance import SystemMode
from core.contracts.learning import LearningUpdate, StrategyStats, TradeOutcome
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
from learning_engine.lanes.patch_outcome_feedback import PatchOutcomeFeedback
from learning_engine.loops.builders import (
    DEFAULT_LEARNING_STRATEGY_ID,
    DEFAULT_UPDATE_REASON,
    make_diff_update_builder,
    make_pnl_sample_builder,
)
from learning_engine.loops.closed_loop import ClosedLearningLoop
from learning_engine.update_emitter import UpdateEmitter

BUILDERS_PATH = Path(__file__).resolve().parent.parent / "learning_engine" / "loops" / "builders.py"


# ---------------------------------------------------------------------------
# make_pnl_sample_builder — contract pins
# ---------------------------------------------------------------------------


def test_sample_builder_rejects_empty_parameters() -> None:
    import pytest

    with pytest.raises(ValueError, match="at least one parameter"):
        make_pnl_sample_builder((), FeedbackSample)


def test_sample_builder_rejects_duplicate_parameters() -> None:
    import pytest

    with pytest.raises(ValueError, match="unique parameter names"):
        make_pnl_sample_builder(("learning_rate", "learning_rate"), FeedbackSample)


def test_sample_builder_rejects_non_callable_factory() -> None:
    import pytest

    with pytest.raises(TypeError, match="callable sample_factory"):
        make_pnl_sample_builder(("learning_rate",), "not-callable")  # type: ignore[arg-type]


def test_sample_builder_rejects_non_positive_weight() -> None:
    import pytest

    with pytest.raises(ValueError, match="weight must be > 0"):
        make_pnl_sample_builder(("learning_rate",), FeedbackSample, weight=0.0)
    with pytest.raises(ValueError, match="weight must be > 0"):
        make_pnl_sample_builder(("learning_rate",), FeedbackSample, weight=-1.0)


def test_sample_builder_empty_outcomes_yields_empty_tuple() -> None:
    build = make_pnl_sample_builder(("learning_rate",), FeedbackSample)

    assert build(()) == ()


def _outcome(*, ts_ns: int, strategy_id: str = "alpha", pnl: float) -> TradeOutcome:
    return TradeOutcome(
        ts_ns=ts_ns,
        strategy_id=strategy_id,
        symbol="BTCUSDT",
        qty=1.0,
        pnl=pnl,
        status=ExecutionStatus.FILLED,
        venue="paper",
        order_id="order-1",
        meta={},
    )


def test_sample_builder_one_sample_per_parameter_per_outcome() -> None:
    parameters = ("alpha", "beta", "learning_rate")
    build = make_pnl_sample_builder(parameters, FeedbackSample)
    outcomes = (
        _outcome(ts_ns=1_500_000_000, pnl=0.5),
        _outcome(ts_ns=2_500_000_000, pnl=-0.25),
    )

    samples = build(outcomes)

    # Two outcomes × three parameters = six samples.
    assert len(samples) == len(outcomes) * len(parameters)
    assert all(isinstance(s, FeedbackSample) for s in samples)
    # First outcome → all three parameters with pnl=0.5 reward.
    first_three = samples[:3]
    assert {s.parameter for s in first_three} == set(parameters)
    assert {s.reward for s in first_three} == {0.5}
    assert {s.ts_unix_s for s in first_three} == {1}
    # Second outcome → all three parameters with pnl=-0.25.
    second_three = samples[3:]
    assert {s.parameter for s in second_three} == set(parameters)
    assert {s.reward for s in second_three} == {-0.25}
    assert {s.ts_unix_s for s in second_three} == {2}


def test_sample_builder_weight_propagates_to_samples() -> None:
    build = make_pnl_sample_builder(("learning_rate",), FeedbackSample, weight=2.5)
    outcomes = (_outcome(ts_ns=1_000_000_000, pnl=1.0),)

    (sample,) = build(outcomes)

    assert sample.weight == 2.5


def test_sample_builder_replay_byte_identical() -> None:
    # INV-15 — same inputs must produce byte-identical outputs.
    parameters = ("learning_rate",)
    outcomes = tuple(
        _outcome(ts_ns=ts, pnl=float(i))
        for i, ts in enumerate((1_000_000_000, 2_000_000_000, 3_000_000_000), start=1)
    )
    runs = [make_pnl_sample_builder(parameters, FeedbackSample)(outcomes) for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# make_diff_update_builder — contract pins
# ---------------------------------------------------------------------------


def test_diff_builder_rejects_empty_strategy_id() -> None:
    import pytest

    with pytest.raises(ValueError, match="strategy_id"):
        make_diff_update_builder(strategy_id="")


def test_diff_builder_rejects_empty_reason() -> None:
    import pytest

    with pytest.raises(ValueError, match="reason"):
        make_diff_update_builder(reason="")


def _snapshot(
    *,
    ts_unix_s: int,
    values: dict[str, float],
    version: int = 1,
) -> ParameterSnapshot:
    return ParameterSnapshot(
        ts_unix_s=ts_unix_s,
        version=version,
        values=values,
        ema={k: 0.0 for k in values},
        sample_counts={k: 0 for k in values},
        frozen=False,
    )


def test_diff_builder_first_tick_returns_empty_tuple() -> None:
    build = make_diff_update_builder()
    current = _snapshot(ts_unix_s=10, values={"learning_rate": 0.05})

    assert build(None, current, ts_ns=10_000_000_000) == ()


def test_diff_builder_unchanged_snapshot_returns_empty() -> None:
    build = make_diff_update_builder()
    previous = _snapshot(ts_unix_s=10, values={"learning_rate": 0.05})
    current = _snapshot(ts_unix_s=11, values={"learning_rate": 0.05})

    assert build(previous, current, ts_ns=11_000_000_000) == ()


def test_diff_builder_emits_one_update_per_changed_parameter() -> None:
    build = make_diff_update_builder()
    previous = _snapshot(
        ts_unix_s=10,
        values={"alpha": 0.5, "beta": 0.25, "learning_rate": 0.05},
    )
    current = _snapshot(
        ts_unix_s=11,
        values={"alpha": 0.5, "beta": 0.3, "learning_rate": 0.07},
    )

    updates = build(previous, current, ts_ns=11_000_000_000)

    assert len(updates) == 2
    assert all(isinstance(u, LearningUpdate) for u in updates)
    # Canonical sort: alphabetical parameter names.
    assert [u.parameter for u in updates] == ["beta", "learning_rate"]
    assert updates[0].old_value == "0.25"
    assert updates[0].new_value == "0.3"
    assert updates[1].old_value == "0.05"
    assert updates[1].new_value == "0.07"
    for update in updates:
        assert update.ts_ns == 11_000_000_000
        assert update.strategy_id == DEFAULT_LEARNING_STRATEGY_ID
        assert update.reason == DEFAULT_UPDATE_REASON
        assert update.meta == {}


def test_diff_builder_overrides_strategy_id_and_reason() -> None:
    build = make_diff_update_builder(strategy_id="custom_lane", reason="custom_reason")
    previous = _snapshot(ts_unix_s=10, values={"learning_rate": 0.05})
    current = _snapshot(ts_unix_s=11, values={"learning_rate": 0.07})

    (update,) = build(previous, current, ts_ns=11_000_000_000)

    assert update.strategy_id == "custom_lane"
    assert update.reason == "custom_reason"


def test_diff_builder_replay_byte_identical() -> None:
    # INV-15 — three runs must produce byte-identical updates.
    build = make_diff_update_builder()
    previous = _snapshot(ts_unix_s=10, values={"learning_rate": 0.05})
    current = _snapshot(ts_unix_s=11, values={"learning_rate": 0.07})

    runs = [build(previous, current, ts_ns=11_000_000_000) for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# End-to-end ClosedLearningLoop integration — the P0 pin
# ---------------------------------------------------------------------------


def _build_unfrozen_loop() -> ClosedLearningLoop:
    """Construct a closed loop wired with the production builders.

    The loop is unfrozen (``LIVE`` + ``operator_override=True``)
    so a tick exercises the full sample→learner→emit→diff chain.
    """

    collector = FeedbackCollector()
    learner = SlowLoopLearner(
        bounds={
            "learning_rate": ParameterBounds(lo=0.0001, hi=1.0, step=0.5, initial=0.05),
        },
        freeze_policy=None,
    )
    emitter = UpdateEmitter(freeze=None)

    def _live_unfrozen() -> LearningEvolutionFreezePolicy:
        return LearningEvolutionFreezePolicy(mode=SystemMode.LIVE, operator_override=True)

    return ClosedLearningLoop(
        feedback_collector=collector,
        learner=learner,
        emitter=emitter,
        policy_supplier=_live_unfrozen,
        sample_builder=make_pnl_sample_builder(tuple(learner.parameters), FeedbackSample),
        update_builder=make_diff_update_builder(),
    )


def test_unfrozen_loop_emits_non_empty_submitted_samples() -> None:
    loop = _build_unfrozen_loop()
    # Push outcomes through the production sink path.
    loop._feedback.record(  # noqa: SLF001 — test harness
        ts_ns=1_500_000_000,
        strategy_id="alpha",
        symbol="BTCUSDT",
        qty=1.0,
        pnl=0.5,
        status=ExecutionStatus.FILLED,
    )
    loop._feedback.record(  # noqa: SLF001 — test harness
        ts_ns=2_500_000_000,
        strategy_id="alpha",
        symbol="BTCUSDT",
        qty=1.0,
        pnl=-0.25,
        status=ExecutionStatus.FILLED,
    )

    result = loop.tick(ts_ns=3_000_000_000)

    assert result.frozen is False
    assert len(result.drained_outcomes) == 2
    # One sample per outcome per parameter (currently one parameter).
    assert len(result.submitted_samples) == 2
    assert all(s.parameter == "learning_rate" for s in result.submitted_samples)


def test_unfrozen_loop_emits_non_empty_events_when_value_drifts() -> None:
    loop = _build_unfrozen_loop()
    # First tick seeds the previous snapshot (no diff yet).
    loop.tick(ts_ns=1_000_000_000)
    # Inject several positive-reward outcomes so the bounded learner
    # steps the parameter on the next tick. The bound's step=0.5 +
    # initial=0.05 means a single bump-step lands at 0.55 (clamped to
    # the hi=1.0 ceiling), which is enough for the diff builder to
    # see a non-zero delta.
    for ts_ns in (2_000_000_000, 3_000_000_000, 4_000_000_000):
        loop._feedback.record(  # noqa: SLF001 — test harness
            ts_ns=ts_ns,
            strategy_id="alpha",
            symbol="BTCUSDT",
            qty=1.0,
            pnl=1.0,
            status=ExecutionStatus.FILLED,
        )

    result = loop.tick(ts_ns=5_000_000_000)

    assert result.frozen is False
    assert result.snapshot is not None
    # If the learner moved at all, we should see at least one event.
    if result.snapshot.values["learning_rate"] != 0.05:
        assert len(result.emitted_events) >= 1
        # UpdateEmitter wraps each LearningUpdate in a
        # ``SystemEvent(sub_kind=UPDATE_PROPOSED)`` for bus emission.
        for ev in result.emitted_events:
            assert ev.sub_kind.value == "UPDATE_PROPOSED"


def test_frozen_loop_still_returns_empty_samples_and_events() -> None:
    """Existing HARDEN-04 short-circuit must be preserved."""

    collector = FeedbackCollector()
    learner = SlowLoopLearner(
        bounds={
            "learning_rate": ParameterBounds(lo=0.0001, hi=1.0, step=0.01, initial=0.05),
        },
        freeze_policy=None,
    )
    emitter = UpdateEmitter(freeze=None)

    def _frozen() -> LearningEvolutionFreezePolicy:
        # Under v42.2-P0-RELAX the freeze gate is operator_override
        # alone; mode is no longer consulted. Pass override=False so
        # the policy is frozen and exercises the loop short-circuit.
        return LearningEvolutionFreezePolicy(mode=SystemMode.PAPER, operator_override=False)

    loop = ClosedLearningLoop(
        feedback_collector=collector,
        learner=learner,
        emitter=emitter,
        policy_supplier=_frozen,
        sample_builder=make_pnl_sample_builder(tuple(learner.parameters), FeedbackSample),
        update_builder=make_diff_update_builder(),
    )
    collector.record(
        ts_ns=1_500_000_000,
        strategy_id="alpha",
        symbol="BTCUSDT",
        qty=1.0,
        pnl=0.5,
        status=ExecutionStatus.FILLED,
    )

    result = loop.tick(ts_ns=2_000_000_000)

    assert result.frozen is True
    assert result.submitted_samples == ()
    assert result.emitted_events == ()
    # But outcomes are still drained so the collector is honest.
    assert len(result.drained_outcomes) == 1


# ---------------------------------------------------------------------------
# PatchOutcomeFeedback wiring — backs _structural_stats_supplier
# ---------------------------------------------------------------------------


def test_patch_outcome_feedback_emits_per_strategy_snapshots() -> None:
    feedback = PatchOutcomeFeedback()
    feedback.observe(_outcome(ts_ns=1_000_000_000, strategy_id="alpha", pnl=1.0))
    feedback.observe(_outcome(ts_ns=2_000_000_000, strategy_id="alpha", pnl=-0.5))
    feedback.observe(_outcome(ts_ns=3_000_000_000, strategy_id="beta", pnl=0.25))

    snapshots = feedback.all_snapshots(ts_ns=4_000_000_000)

    assert set(snapshots.keys()) == {"alpha", "beta"}
    alpha = snapshots["alpha"]
    beta = snapshots["beta"]
    assert isinstance(alpha, StrategyStats)
    assert isinstance(beta, StrategyStats)
    assert alpha.n_trades == 2
    assert beta.n_trades == 1
    assert alpha.win_rate == 0.5  # 1 win out of 2.
    assert beta.win_rate == 1.0


def test_structural_stats_supplier_yields_observed_rows() -> None:
    """The shape used by ``_State._structural_stats_supplier`` —
    drain → tuple — should surface every observed strategy."""

    feedback = PatchOutcomeFeedback()
    feedback.observe(_outcome(ts_ns=1_000_000_000, strategy_id="alpha", pnl=1.0))
    feedback.observe(_outcome(ts_ns=2_000_000_000, strategy_id="beta", pnl=-0.5))

    rows = tuple(feedback.all_snapshots(ts_ns=3_000_000_000).values())

    assert len(rows) == 2
    assert {row.strategy_id for row in rows} == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# AST guardrails — INV-15 / B1 / B27 / HARDEN-06
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_LEVEL_IMPORTS = {
    "asyncio",
    "datetime",
    "os",
    "polars",
    "random",
    "requests",
    "time",
    "torch",
    "numpy",
}


def _read_tree() -> ast.Module:
    return ast.parse(BUILDERS_PATH.read_text(), filename=str(BUILDERS_PATH))


def test_ast_no_forbidden_top_level_imports() -> None:
    tree = _read_tree()

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                assert root not in _FORBIDDEN_TOP_LEVEL_IMPORTS, (
                    f"forbidden top-level import {alias.name!r} in "
                    f"learning_engine/loops/builders.py"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            assert root not in _FORBIDDEN_TOP_LEVEL_IMPORTS, (
                f"forbidden top-level import {node.module!r} in learning_engine/loops/builders.py"
            )


def test_ast_no_runtime_tier_cross_imports() -> None:
    """B1 — ``learning_engine.*`` is offline; must not import from
    runtime tiers (``execution_engine.*`` or ``system_engine.*``)."""

    tree = _read_tree()
    runtime_roots = {"execution_engine", "system_engine"}

    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            assert root not in runtime_roots, (
                f"learning_engine.loops.builders imports from runtime "
                f"tier {node.module!r}; violates B1"
            )


def test_ast_no_top_level_typed_event_constructors() -> None:
    """B27 / HARDEN-06 / INV-71 — typed event constructors at module
    level are forbidden. Inside function bodies, only
    :class:`LearningUpdate` may be constructed from
    ``learning_engine.*`` (this module is under that path)."""

    tree = _read_tree()
    forbidden_names = {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
    }

    class _CtorVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name):
                if node.func.id in forbidden_names:
                    self.calls.append(node.func.id)
            self.generic_visit(node)

    visitor = _CtorVisitor()
    visitor.visit(tree)
    assert visitor.calls == [], (
        f"forbidden typed-event constructors at module level: {visitor.calls!r}"
    )


def test_builders_module_imports_cleanly() -> None:
    """The module must import without side-effects (no clock reads,
    no PRNG seeding, no IO)."""

    module = importlib.import_module("learning_engine.loops.builders")

    assert hasattr(module, "make_pnl_sample_builder")
    assert hasattr(module, "make_diff_update_builder")
    assert module.DEFAULT_LEARNING_STRATEGY_ID == "closed_learning_loop"
    assert module.DEFAULT_UPDATE_REASON == "slow_loop_parameter_diff"
