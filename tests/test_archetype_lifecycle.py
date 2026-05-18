"""Tests for ``intelligence_engine.strategy_runtime.archetype_lifecycle`` (B-09).

Five test groups:
* AST authority pins
* Value-object validation
* FSM transition behaviour (IDLE / PENDING_* / OPEN_*)
* Strategy-Protocol contract
* INV-15 byte-identical replay determinism
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import pytest

from intelligence_engine.strategy_runtime.archetype_lifecycle import (
    NEW_PIP_DEPENDENCIES,
    ArchetypeContext,
    ArchetypeDecision,
    ArchetypeLifecycle,
    ArchetypeLifecycleError,
    ArchetypeStateError,
    ArchetypeStrategy,
    DecisionKind,
    EntryDecision,
    LifecycleState,
    PendingEntry,
    PositionAction,
    PositionSnapshot,
    PositionUpdate,
    Side,
    advance_lifecycle,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "intelligence_engine" / "strategy_runtime" / "archetype_lifecycle.py"
MODULE_SRC = MODULE_PATH.read_text(encoding="utf-8")
MODULE_TREE = ast.parse(MODULE_SRC)


# ---------------------------------------------------------------------------
# AST authority pins (B-09 archetype-lifecycle)
# ---------------------------------------------------------------------------


def _imported_modules(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                mods.add(node.module.split(".", 1)[0])
    return mods


def _call_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def test_authority_no_jesse_import() -> None:
    """B-09 PATTERN_ONLY: no upstream jesse classes in production."""
    assert "jesse" not in _imported_modules(MODULE_TREE)


def test_authority_no_typed_event_construction() -> None:
    """B27/B28/INV-71: intelligence tier never constructs typed
    bus events."""
    names = _call_names(MODULE_TREE)
    forbidden = {
        "SignalEvent",
        "ExecutionIntent",
        "PatchProposal",
        "GovernanceDecision",
    }
    assert not (forbidden & names), (
        f"archetype_lifecycle must not construct typed bus events; found: {forbidden & names}"
    )


def test_authority_no_engine_cross_imports() -> None:
    """INV-71: never import from owning engines / system."""
    mods = _imported_modules(MODULE_TREE)
    forbidden = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "evolution_engine",
    }
    assert not (forbidden & mods), (
        f"archetype_lifecycle must not import engines; found: {forbidden & mods}"
    )


def test_authority_no_clock_or_random() -> None:
    """INV-15: never import datetime / time / random / asyncio / os."""
    mods = _imported_modules(MODULE_TREE)
    forbidden = {
        "datetime",
        "time",
        "random",
        "asyncio",
        "os",
        "secrets",
        "uuid",
    }
    assert not (forbidden & mods), f"archetype_lifecycle must not import: {forbidden & mods}"


def test_authority_no_numpy_torch_pandas() -> None:
    """OFFLINE-tier coordinator: no heavy deps."""
    mods = _imported_modules(MODULE_TREE)
    forbidden = {"numpy", "torch", "pandas", "polars", "scipy"}
    assert not (forbidden & mods)


def test_authority_adapted_from_header() -> None:
    """B-09 PATTERN_ONLY: explicit attribution header required."""
    assert "ADAPTED FROM" in MODULE_SRC.split("\n", 1)[0] or any(
        "ADAPTED FROM" in line for line in MODULE_SRC.splitlines()[:5]
    )
    assert "jesse" in MODULE_SRC.lower()


def test_authority_pip_dependencies_empty() -> None:
    """B-09 PATTERN_ONLY: pure stdlib."""
    assert NEW_PIP_DEPENDENCIES == ()


def test_authority_no_top_level_io() -> None:
    """Pure module: no open() / write() / print() at top level."""
    forbidden = {"open", "print", "input", "exec", "eval"}
    found: set[str] = set()
    for node in MODULE_TREE.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in forbidden:
                    found.add(sub.func.id)
    assert not found, f"archetype_lifecycle has top-level IO: {found}"


def test_authority_module_reimport_clean() -> None:
    """Module must reimport without side effects."""
    mod_name = "intelligence_engine.strategy_runtime.archetype_lifecycle"
    sys.modules.pop(mod_name, None)
    importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Value-object validation
# ---------------------------------------------------------------------------


def test_position_snapshot_long_ok() -> None:
    p = PositionSnapshot(
        side=Side.LONG,
        qty=1.0,
        entry_price=100.0,
        unrealised_pnl_usd=5.0,
        bars_held=3,
    )
    assert p.side is Side.LONG


def test_position_snapshot_flat_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PositionSnapshot(
            side=Side.FLAT,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=0.0,
            bars_held=0,
        )


def test_position_snapshot_negative_qty_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PositionSnapshot(
            side=Side.LONG,
            qty=-1.0,
            entry_price=100.0,
            unrealised_pnl_usd=0.0,
            bars_held=0,
        )


def test_position_snapshot_nonfinite_entry_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PositionSnapshot(
            side=Side.LONG,
            qty=1.0,
            entry_price=float("inf"),
            unrealised_pnl_usd=0.0,
            bars_held=0,
        )


def test_position_snapshot_negative_bars_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PositionSnapshot(
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=0.0,
            bars_held=-1,
        )


def test_position_snapshot_frozen() -> None:
    p = PositionSnapshot(
        side=Side.LONG,
        qty=1.0,
        entry_price=100.0,
        unrealised_pnl_usd=0.0,
        bars_held=0,
    )
    with pytest.raises((AttributeError, TypeError)):
        p.qty = 2.0  # type: ignore[misc]


def test_pending_entry_long_ok() -> None:
    p = PendingEntry(side=Side.LONG, qty=1.0, limit_price=100.0, bars_pending=0)
    assert p.side is Side.LONG


def test_pending_entry_flat_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PendingEntry(side=Side.FLAT, qty=1.0, limit_price=100.0, bars_pending=0)


def test_pending_entry_zero_qty_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PendingEntry(side=Side.LONG, qty=0.0, limit_price=100.0, bars_pending=0)


def test_entry_decision_long_ok() -> None:
    e = EntryDecision(
        side=Side.LONG,
        qty=1.0,
        limit_price=100.0,
        stop_loss_price=95.0,
        take_profit_price=110.0,
    )
    assert e.side is Side.LONG


def test_entry_decision_long_stops_above_limit_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        EntryDecision(
            side=Side.LONG,
            qty=1.0,
            limit_price=100.0,
            stop_loss_price=105.0,
            take_profit_price=110.0,
        )


def test_entry_decision_long_tp_below_limit_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        EntryDecision(
            side=Side.LONG,
            qty=1.0,
            limit_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=99.0,
        )


def test_entry_decision_short_ok() -> None:
    e = EntryDecision(
        side=Side.SHORT,
        qty=1.0,
        limit_price=100.0,
        stop_loss_price=105.0,
        take_profit_price=90.0,
    )
    assert e.side is Side.SHORT


def test_entry_decision_short_stops_below_limit_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        EntryDecision(
            side=Side.SHORT,
            qty=1.0,
            limit_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=90.0,
        )


def test_entry_decision_too_many_tags_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        EntryDecision(
            side=Side.LONG,
            qty=1.0,
            limit_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            rationale_tags=tuple(f"t{i}" for i in range(17)),
        )


def test_entry_decision_too_long_rationale_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        EntryDecision(
            side=Side.LONG,
            qty=1.0,
            limit_price=100.0,
            stop_loss_price=95.0,
            take_profit_price=110.0,
            rationale="x" * 257,
        )


def test_position_update_hold_ok() -> None:
    u = PositionUpdate(action=PositionAction.HOLD)
    assert u.action is PositionAction.HOLD


def test_position_update_close_ok() -> None:
    u = PositionUpdate(action=PositionAction.CLOSE)
    assert u.action is PositionAction.CLOSE


def test_position_update_adjust_requires_nonzero() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PositionUpdate(action=PositionAction.ADJUST_STOPS)


def test_position_update_negative_price_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        PositionUpdate(
            action=PositionAction.ADJUST_STOPS,
            new_stop_loss_price=-1.0,
        )


def test_archetype_context_negative_ts_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        ArchetypeContext(
            ts_ns=-1,
            archetype_id="TA-001",
            symbol="BTC-USD",
            bar_index=0,
            last_price=100.0,
        )


def test_archetype_context_empty_id_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        ArchetypeContext(
            ts_ns=0,
            archetype_id="",
            symbol="BTC-USD",
            bar_index=0,
            last_price=100.0,
        )


def test_archetype_context_oversized_id_rejected() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        ArchetypeContext(
            ts_ns=0,
            archetype_id="x" * 33,
            symbol="BTC-USD",
            bar_index=0,
            last_price=100.0,
        )


def test_archetype_context_features_typed() -> None:
    with pytest.raises(ArchetypeLifecycleError):
        ArchetypeContext(
            ts_ns=0,
            archetype_id="TA-001",
            symbol="BTC-USD",
            bar_index=0,
            last_price=100.0,
            features={"x": [1, 2]},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# Strategy Protocol test doubles
# ---------------------------------------------------------------------------


@dataclass
class _HoldStrategy:
    """Strategy that never enters or exits — used to assert NO_OP paths."""

    before_calls: int = 0
    after_calls: int = 0

    def before(self, ctx: ArchetypeContext) -> None:
        self.before_calls += 1

    def go_long(self, ctx: ArchetypeContext) -> EntryDecision | None:
        return None

    def go_short(self, ctx: ArchetypeContext) -> EntryDecision | None:
        return None

    def should_cancel_entry(self, ctx: ArchetypeContext, pending: PendingEntry) -> bool:
        return False

    def update_position(self, ctx: ArchetypeContext, position: PositionSnapshot) -> PositionUpdate:
        return PositionUpdate(action=PositionAction.HOLD)

    def after(self, ctx: ArchetypeContext) -> None:
        self.after_calls += 1


class _LongStrategy(_HoldStrategy):
    def go_long(self, ctx: ArchetypeContext) -> EntryDecision | None:
        return EntryDecision(
            side=Side.LONG,
            qty=1.0,
            limit_price=ctx.last_price,
            stop_loss_price=ctx.last_price * 0.95,
            take_profit_price=ctx.last_price * 1.10,
            rationale_tags=("trend_up",),
            rationale="moving avg crossover",
        )


class _ShortStrategy(_HoldStrategy):
    def go_short(self, ctx: ArchetypeContext) -> EntryDecision | None:
        return EntryDecision(
            side=Side.SHORT,
            qty=1.0,
            limit_price=ctx.last_price,
            stop_loss_price=ctx.last_price * 1.05,
            take_profit_price=ctx.last_price * 0.90,
        )


class _AlwaysCancelStrategy(_HoldStrategy):
    def should_cancel_entry(self, ctx: ArchetypeContext, pending: PendingEntry) -> bool:
        return True


class _CloseStrategy(_HoldStrategy):
    def update_position(self, ctx: ArchetypeContext, position: PositionSnapshot) -> PositionUpdate:
        return PositionUpdate(
            action=PositionAction.CLOSE,
            rationale_tags=("tp_hit",),
            rationale="take-profit reached",
        )


class _AdjustStopsStrategy(_HoldStrategy):
    def update_position(self, ctx: ArchetypeContext, position: PositionSnapshot) -> PositionUpdate:
        return PositionUpdate(
            action=PositionAction.ADJUST_STOPS,
            new_stop_loss_price=ctx.last_price * 0.97,
            new_take_profit_price=ctx.last_price * 1.05,
        )


# ---------------------------------------------------------------------------
# FSM transition behaviour
# ---------------------------------------------------------------------------


def _idle_ctx(ts_ns: int = 1_000_000_000) -> ArchetypeContext:
    return ArchetypeContext(
        ts_ns=ts_ns,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
    )


def _idle_lifecycle() -> ArchetypeLifecycle:
    return ArchetypeLifecycle(
        state=LifecycleState.IDLE,
        archetype_id="TA-001",
        symbol="BTC-USD",
    )


def test_idle_no_signal_returns_noop() -> None:
    strat = _HoldStrategy()
    nxt, dec = advance_lifecycle(
        lifecycle=_idle_lifecycle(),
        strategy=strat,
        ctx=_idle_ctx(),
    )
    assert dec.kind is DecisionKind.NO_OP
    assert nxt.state is LifecycleState.IDLE
    assert strat.before_calls == 1
    assert strat.after_calls == 1


def test_idle_go_long_transitions_to_pending_long() -> None:
    strat = _LongStrategy()
    nxt, dec = advance_lifecycle(
        lifecycle=_idle_lifecycle(),
        strategy=strat,
        ctx=_idle_ctx(),
    )
    assert dec.kind is DecisionKind.OPEN_ENTRY
    assert dec.state_before is LifecycleState.IDLE
    assert dec.state_after is LifecycleState.PENDING_LONG
    assert dec.entry is not None
    assert dec.entry.side is Side.LONG
    assert nxt.state is LifecycleState.PENDING_LONG
    assert dec.rationale_tags == ("trend_up",)


def test_idle_go_short_transitions_to_pending_short() -> None:
    strat = _ShortStrategy()
    nxt, dec = advance_lifecycle(
        lifecycle=_idle_lifecycle(),
        strategy=strat,
        ctx=_idle_ctx(),
    )
    assert dec.kind is DecisionKind.OPEN_ENTRY
    assert dec.state_after is LifecycleState.PENDING_SHORT
    assert nxt.state is LifecycleState.PENDING_SHORT


def test_idle_with_position_raises() -> None:
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
        position=PositionSnapshot(
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=0.0,
            bars_held=0,
        ),
    )
    with pytest.raises(ArchetypeStateError):
        advance_lifecycle(
            lifecycle=_idle_lifecycle(),
            strategy=_HoldStrategy(),
            ctx=ctx,
        )


def test_idle_with_pending_raises() -> None:
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
        pending=PendingEntry(
            side=Side.LONG,
            qty=1.0,
            limit_price=100.0,
            bars_pending=0,
        ),
    )
    with pytest.raises(ArchetypeStateError):
        advance_lifecycle(
            lifecycle=_idle_lifecycle(),
            strategy=_HoldStrategy(),
            ctx=ctx,
        )


def test_pending_long_no_cancel_stays() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.PENDING_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=1,
        last_price=100.0,
        pending=PendingEntry(
            side=Side.LONG,
            qty=1.0,
            limit_price=100.0,
            bars_pending=1,
        ),
    )
    nxt, dec = advance_lifecycle(lifecycle=lifecycle, strategy=_HoldStrategy(), ctx=ctx)
    assert dec.kind is DecisionKind.NO_OP
    assert nxt.state is LifecycleState.PENDING_LONG


def test_pending_long_cancel_transitions_to_idle() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.PENDING_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=1,
        last_price=100.0,
        pending=PendingEntry(
            side=Side.LONG,
            qty=1.0,
            limit_price=100.0,
            bars_pending=5,
        ),
    )
    nxt, dec = advance_lifecycle(lifecycle=lifecycle, strategy=_AlwaysCancelStrategy(), ctx=ctx)
    assert dec.kind is DecisionKind.CANCEL_ENTRY
    assert nxt.state is LifecycleState.IDLE


def test_pending_long_without_pending_raises() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.PENDING_LONG)
    with pytest.raises(ArchetypeStateError):
        advance_lifecycle(lifecycle=lifecycle, strategy=_HoldStrategy(), ctx=_idle_ctx())


def test_pending_long_with_wrong_side_raises() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.PENDING_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=1,
        last_price=100.0,
        pending=PendingEntry(
            side=Side.SHORT,
            qty=1.0,
            limit_price=100.0,
            bars_pending=0,
        ),
    )
    with pytest.raises(ArchetypeStateError):
        advance_lifecycle(lifecycle=lifecycle, strategy=_HoldStrategy(), ctx=ctx)


def test_open_long_hold_keeps_state() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.OPEN_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=2,
        last_price=102.0,
        position=PositionSnapshot(
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=2.0,
            bars_held=2,
        ),
    )
    nxt, dec = advance_lifecycle(lifecycle=lifecycle, strategy=_HoldStrategy(), ctx=ctx)
    assert dec.kind is DecisionKind.HOLD_POSITION
    assert nxt.state is LifecycleState.OPEN_LONG


def test_open_long_close_transitions_to_idle() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.OPEN_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=10,
        last_price=110.0,
        position=PositionSnapshot(
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=10.0,
            bars_held=10,
        ),
    )
    nxt, dec = advance_lifecycle(lifecycle=lifecycle, strategy=_CloseStrategy(), ctx=ctx)
    assert dec.kind is DecisionKind.CLOSE_POSITION
    assert nxt.state is LifecycleState.IDLE
    assert dec.rationale_tags == ("tp_hit",)


def test_open_long_adjust_stops_keeps_state() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.OPEN_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=5,
        last_price=105.0,
        position=PositionSnapshot(
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=5.0,
            bars_held=5,
        ),
    )
    nxt, dec = advance_lifecycle(lifecycle=lifecycle, strategy=_AdjustStopsStrategy(), ctx=ctx)
    assert dec.kind is DecisionKind.ADJUST_STOPS
    assert nxt.state is LifecycleState.OPEN_LONG
    assert dec.update is not None
    assert dec.update.new_stop_loss_price == pytest.approx(105.0 * 0.97)


def test_open_short_close_transitions_to_idle() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.OPEN_SHORT)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=4,
        last_price=95.0,
        position=PositionSnapshot(
            side=Side.SHORT,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=5.0,
            bars_held=4,
        ),
    )
    nxt, dec = advance_lifecycle(lifecycle=lifecycle, strategy=_CloseStrategy(), ctx=ctx)
    assert dec.kind is DecisionKind.CLOSE_POSITION
    assert nxt.state is LifecycleState.IDLE


def test_open_long_without_position_raises() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.OPEN_LONG)
    with pytest.raises(ArchetypeStateError):
        advance_lifecycle(lifecycle=lifecycle, strategy=_HoldStrategy(), ctx=_idle_ctx())


def test_open_long_with_wrong_side_position_raises() -> None:
    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.OPEN_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=1,
        last_price=100.0,
        position=PositionSnapshot(
            side=Side.SHORT,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=0.0,
            bars_held=0,
        ),
    )
    with pytest.raises(ArchetypeStateError):
        advance_lifecycle(lifecycle=lifecycle, strategy=_HoldStrategy(), ctx=ctx)


def test_ctx_archetype_id_mismatch_raises() -> None:
    ctx = ArchetypeContext(
        ts_ns=0,
        archetype_id="TA-OTHER",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
    )
    with pytest.raises(ArchetypeLifecycleError):
        advance_lifecycle(lifecycle=_idle_lifecycle(), strategy=_HoldStrategy(), ctx=ctx)


def test_ctx_symbol_mismatch_raises() -> None:
    ctx = ArchetypeContext(
        ts_ns=0,
        archetype_id="TA-001",
        symbol="ETH-USD",
        bar_index=0,
        last_price=100.0,
    )
    with pytest.raises(ArchetypeLifecycleError):
        advance_lifecycle(lifecycle=_idle_lifecycle(), strategy=_HoldStrategy(), ctx=ctx)


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


def test_strategy_protocol_runtime_checkable() -> None:
    assert isinstance(_HoldStrategy(), ArchetypeStrategy)


def test_non_protocol_object_rejected() -> None:
    class _Bare:
        pass

    with pytest.raises(ArchetypeLifecycleError):
        advance_lifecycle(
            lifecycle=_idle_lifecycle(),
            strategy=_Bare(),  # type: ignore[arg-type]
            ctx=_idle_ctx(),
        )


def test_should_cancel_must_return_bool() -> None:
    class _BadCancel(_HoldStrategy):
        def should_cancel_entry(self, ctx: ArchetypeContext, pending: PendingEntry) -> bool:
            return "yes"  # type: ignore[return-value]

    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.PENDING_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=1,
        last_price=100.0,
        pending=PendingEntry(side=Side.LONG, qty=1.0, limit_price=100.0, bars_pending=1),
    )
    with pytest.raises(ArchetypeLifecycleError):
        advance_lifecycle(lifecycle=lifecycle, strategy=_BadCancel(), ctx=ctx)


def test_update_position_must_return_position_update() -> None:
    class _BadUpdate(_HoldStrategy):
        def update_position(
            self, ctx: ArchetypeContext, position: PositionSnapshot
        ) -> PositionUpdate:
            return "hold"  # type: ignore[return-value]

    lifecycle = replace(_idle_lifecycle(), state=LifecycleState.OPEN_LONG)
    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=1,
        last_price=100.0,
        position=PositionSnapshot(
            side=Side.LONG,
            qty=1.0,
            entry_price=100.0,
            unrealised_pnl_usd=0.0,
            bars_held=0,
        ),
    )
    with pytest.raises(ArchetypeLifecycleError):
        advance_lifecycle(lifecycle=lifecycle, strategy=_BadUpdate(), ctx=ctx)


def test_go_long_returning_short_decision_raises() -> None:
    class _MislabeledLong(_HoldStrategy):
        def go_long(self, ctx: ArchetypeContext) -> EntryDecision | None:
            return EntryDecision(
                side=Side.SHORT,
                qty=1.0,
                limit_price=100.0,
                stop_loss_price=105.0,
                take_profit_price=90.0,
            )

    with pytest.raises(ArchetypeLifecycleError):
        advance_lifecycle(
            lifecycle=_idle_lifecycle(),
            strategy=_MislabeledLong(),
            ctx=_idle_ctx(),
        )


def test_go_short_returning_long_decision_raises() -> None:
    class _MislabeledShort(_HoldStrategy):
        def go_short(self, ctx: ArchetypeContext) -> EntryDecision | None:
            return EntryDecision(
                side=Side.LONG,
                qty=1.0,
                limit_price=100.0,
                stop_loss_price=95.0,
                take_profit_price=110.0,
            )

    with pytest.raises(ArchetypeLifecycleError):
        advance_lifecycle(
            lifecycle=_idle_lifecycle(),
            strategy=_MislabeledShort(),
            ctx=_idle_ctx(),
        )


def test_long_strategy_calls_before_and_after_in_order() -> None:
    """Hook order: before() always runs first; after() always last."""
    order: list[str] = []

    class _OrderedStrategy(_HoldStrategy):
        def before(self, ctx: ArchetypeContext) -> None:
            order.append("before")

        def go_long(self, ctx: ArchetypeContext) -> EntryDecision | None:
            order.append("go_long")
            return None

        def go_short(self, ctx: ArchetypeContext) -> EntryDecision | None:
            order.append("go_short")
            return None

        def after(self, ctx: ArchetypeContext) -> None:
            order.append("after")

    advance_lifecycle(
        lifecycle=_idle_lifecycle(),
        strategy=_OrderedStrategy(),
        ctx=_idle_ctx(),
    )
    assert order == ["before", "go_long", "go_short", "after"]


# ---------------------------------------------------------------------------
# INV-15 byte-identical replay determinism
# ---------------------------------------------------------------------------


def _run_three_bars(
    strat: ArchetypeStrategy,
) -> tuple[ArchetypeDecision, ArchetypeDecision, ArchetypeDecision]:
    """Single deterministic 3-bar driver for replay tests."""
    lifecycle = _idle_lifecycle()
    decs: list[ArchetypeDecision] = []
    # bar 0: IDLE → PENDING_LONG
    ctx0 = ArchetypeContext(
        ts_ns=1_000_000_000,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
    )
    lifecycle, dec = advance_lifecycle(lifecycle=lifecycle, strategy=strat, ctx=ctx0)
    decs.append(dec)
    # bar 1: PENDING_LONG → still pending (simulate not filled)
    ctx1 = ArchetypeContext(
        ts_ns=2_000_000_000,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=1,
        last_price=101.0,
        pending=PendingEntry(side=Side.LONG, qty=1.0, limit_price=100.0, bars_pending=1),
    )
    lifecycle, dec = advance_lifecycle(lifecycle=lifecycle, strategy=strat, ctx=ctx1)
    decs.append(dec)
    # bar 2: PENDING_LONG → cancel
    ctx2 = ArchetypeContext(
        ts_ns=3_000_000_000,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=2,
        last_price=99.0,
        pending=PendingEntry(side=Side.LONG, qty=1.0, limit_price=100.0, bars_pending=2),
    )
    lifecycle, dec = advance_lifecycle(lifecycle=lifecycle, strategy=strat, ctx=ctx2)
    decs.append(dec)
    return decs[0], decs[1], decs[2]


@dataclass
class _ReplayStrategy(_HoldStrategy):
    """Long entry then cancel after 2 bars pending. Pure, no PRNG."""

    def go_long(self, ctx: ArchetypeContext) -> EntryDecision | None:
        return EntryDecision(
            side=Side.LONG,
            qty=1.0,
            limit_price=ctx.last_price,
            stop_loss_price=ctx.last_price * 0.95,
            take_profit_price=ctx.last_price * 1.10,
        )

    def should_cancel_entry(self, ctx: ArchetypeContext, pending: PendingEntry) -> bool:
        return pending.bars_pending >= 2


def test_replay_three_runs_byte_identical() -> None:
    """INV-15: 3 independent passes produce identical decisions."""
    run_a = _run_three_bars(_ReplayStrategy())
    run_b = _run_three_bars(_ReplayStrategy())
    run_c = _run_three_bars(_ReplayStrategy())
    assert run_a == run_b == run_c


def test_replay_kinds_in_sequence() -> None:
    bar0, bar1, bar2 = _run_three_bars(_ReplayStrategy())
    assert bar0.kind is DecisionKind.OPEN_ENTRY
    assert bar1.kind is DecisionKind.NO_OP
    assert bar2.kind is DecisionKind.CANCEL_ENTRY


def test_features_dict_order_independence() -> None:
    """Two ctx instances with the same features in different dict
    insertion order produce equal decisions."""
    ctx_a = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
        features={"a": 1.0, "b": 2.0, "c": 3.0},
    )
    ctx_b = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
        features={"c": 3.0, "a": 1.0, "b": 2.0},
    )
    _, dec_a = advance_lifecycle(lifecycle=_idle_lifecycle(), strategy=_LongStrategy(), ctx=ctx_a)
    _, dec_b = advance_lifecycle(lifecycle=_idle_lifecycle(), strategy=_LongStrategy(), ctx=ctx_b)
    assert dec_a == dec_b


def test_archetype_decision_frozen() -> None:
    _, dec = advance_lifecycle(
        lifecycle=_idle_lifecycle(),
        strategy=_LongStrategy(),
        ctx=_idle_ctx(),
    )
    with pytest.raises((AttributeError, TypeError)):
        dec.kind = DecisionKind.NO_OP  # type: ignore[misc]


def test_archetype_lifecycle_frozen() -> None:
    lifecycle = _idle_lifecycle()
    with pytest.raises((AttributeError, TypeError)):
        lifecycle.state = LifecycleState.OPEN_LONG  # type: ignore[misc]


def test_features_mapping_accepted_via_dict() -> None:
    """Caller-supplied features may be any read-only Mapping[str, ...]."""

    class _ReadOnlyMap(Mapping[str, Any]):
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __iter__(self) -> Any:
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

    ctx = ArchetypeContext(
        ts_ns=1,
        archetype_id="TA-001",
        symbol="BTC-USD",
        bar_index=0,
        last_price=100.0,
        features=_ReadOnlyMap({"x": 1.0}),
    )
    _, dec = advance_lifecycle(lifecycle=_idle_lifecycle(), strategy=_HoldStrategy(), ctx=ctx)
    assert dec.kind is DecisionKind.NO_OP
