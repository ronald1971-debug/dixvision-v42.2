"""DASH-SLP-01 — StrategyLifecyclePanel widget tests."""

from __future__ import annotations

from dashboard_backend.control_plane.strategy_lifecycle_panel import (
    LIFECYCLE_COLUMNS,
    StrategyLifecyclePanel,
)
from intelligence_engine.strategy_runtime.state_machine import (
    StrategyState,
    StrategyStateMachine,
)


def _build_fsm() -> StrategyStateMachine:
    fsm = StrategyStateMachine()
    fsm.propose(strategy_id="alpha", ts_ns=1)
    fsm.propose(strategy_id="beta", ts_ns=2)
    fsm.propose(strategy_id="gamma", ts_ns=3)
    fsm.transition(
        strategy_id="alpha",
        new_state=StrategyState.CANARY,
        ts_ns=5,
        reason="canary promote",
    )
    fsm.transition(
        strategy_id="beta",
        new_state=StrategyState.FAILED,
        ts_ns=6,
        reason="invariant breach",
    )
    return fsm


def test_panel_groups_by_canonical_state_order():
    panel = StrategyLifecyclePanel(fsm=_build_fsm())
    grouped = panel.by_state()
    assert tuple(grouped.keys()) == tuple(s.value for s in LIFECYCLE_COLUMNS)
    assert tuple(r.strategy_id for r in grouped["PROPOSED"]) == ("gamma",)
    assert tuple(r.strategy_id for r in grouped["CANARY"]) == ("alpha",)
    assert tuple(r.strategy_id for r in grouped["FAILED"]) == ("beta",)


def test_panel_emits_history_entries_in_order():
    panel = StrategyLifecyclePanel(fsm=_build_fsm())
    rows = {r.strategy_id: r for r in panel.all_rows()}
    alpha = rows["alpha"]
    assert tuple((h.prev, h.new) for h in alpha.history) == (
        ("PROPOSED", "PROPOSED"),
        ("PROPOSED", "CANARY"),
    )


def test_panel_marks_terminal_strategies():
    panel = StrategyLifecyclePanel(fsm=_build_fsm())
    rows = {r.strategy_id: r for r in panel.all_rows()}
    assert rows["beta"].is_terminal is True
    assert rows["alpha"].is_terminal is False
    assert rows["gamma"].is_terminal is False
