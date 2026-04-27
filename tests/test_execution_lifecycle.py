"""Phase 2 / v2-C — execution lifecycle unit tests."""

from __future__ import annotations

import pytest

from core.contracts.events import Side
from execution_engine.lifecycle import (
    LEGAL_ORDER_TRANSITIONS,
    Bracket,
    BracketTrigger,
    FillEvent,
    FillHandler,
    OrderState,
    OrderStateMachine,
    PartialFillResolution,
    PartialFillResolver,
    RetryClassification,
    RetryPolicy,
    SLTPManager,
    StateTransitionError,
)
from execution_engine.lifecycle.partial_fill_resolver import ResolutionContext

# ---------------------------------------------------------------------------
# OrderStateMachine
# ---------------------------------------------------------------------------


def test_order_open_starts_in_new_with_seed_history():
    fsm = OrderStateMachine()
    record = fsm.open(order_id="o1", ts_ns=100)
    assert record.state is OrderState.NEW
    assert len(record.history) == 1
    assert record.history[0].reason == "open"
    assert record.history[0].ts_ns == 100


def test_order_open_duplicate_raises():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    with pytest.raises(ValueError):
        fsm.open(order_id="o1", ts_ns=2)


def test_order_open_blank_raises():
    fsm = OrderStateMachine()
    with pytest.raises(ValueError):
        fsm.open(order_id="", ts_ns=1)


def test_legal_transition_pending_then_filled_then_closed():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=10)
    fsm.transition(
        order_id="o1",
        new_state=OrderState.PENDING,
        ts_ns=20,
        reason="submitted",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.FILLED,
        ts_ns=30,
        reason="venue_fill",
    )
    record = fsm.transition(
        order_id="o1",
        new_state=OrderState.CLOSED,
        ts_ns=40,
        reason="closed",
    )
    assert record.is_terminal()
    assert [h.new for h in record.history] == [
        OrderState.NEW,
        OrderState.PENDING,
        OrderState.FILLED,
        OrderState.CLOSED,
    ]


def test_illegal_transition_new_to_filled_raises():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=10)
    with pytest.raises(StateTransitionError):
        fsm.transition(
            order_id="o1",
            new_state=OrderState.FILLED,
            ts_ns=20,
            reason="bad",
        )


def test_terminal_closed_has_no_outgoing_edges():
    assert LEGAL_ORDER_TRANSITIONS[OrderState.CLOSED] == frozenset()


def test_transition_unknown_order_raises_keyerror():
    fsm = OrderStateMachine()
    with pytest.raises(KeyError):
        fsm.transition(
            order_id="ghost",
            new_state=OrderState.PENDING,
            ts_ns=1,
            reason="x",
        )


def test_partial_fill_can_repeat_then_close_via_filled():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    fsm.transition(
        order_id="o1",
        new_state=OrderState.PENDING,
        ts_ns=2,
        reason="submitted",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.PARTIALLY_FILLED,
        ts_ns=3,
        reason="p1",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.PARTIALLY_FILLED,
        ts_ns=4,
        reason="p2",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.FILLED,
        ts_ns=5,
        reason="final",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.CLOSED,
        ts_ns=6,
        reason="closed",
    )
    record = fsm.get("o1")
    assert record is not None
    assert record.state is OrderState.CLOSED


def test_cancel_from_pending_then_closed():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    fsm.transition(
        order_id="o1",
        new_state=OrderState.PENDING,
        ts_ns=2,
        reason="submitted",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.CANCELLED,
        ts_ns=3,
        reason="user",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.CLOSED,
        ts_ns=4,
        reason="closed",
    )
    assert fsm.get("o1").state is OrderState.CLOSED


def test_error_from_partial_then_closed():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    fsm.transition(
        order_id="o1",
        new_state=OrderState.PENDING,
        ts_ns=2,
        reason="submitted",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.PARTIALLY_FILLED,
        ts_ns=3,
        reason="partial",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.ERROR,
        ts_ns=4,
        reason="venue_error",
    )
    fsm.transition(
        order_id="o1",
        new_state=OrderState.CLOSED,
        ts_ns=5,
        reason="closed",
    )
    assert fsm.get("o1").state is OrderState.CLOSED


def test_fsm_replay_determinism_same_inputs_same_history():
    def run() -> tuple:
        fsm = OrderStateMachine()
        fsm.open(order_id="o1", ts_ns=10)
        fsm.transition(
            order_id="o1",
            new_state=OrderState.PENDING,
            ts_ns=20,
            reason="r",
        )
        fsm.transition(
            order_id="o1",
            new_state=OrderState.PARTIALLY_FILLED,
            ts_ns=30,
            reason="p",
        )
        fsm.transition(
            order_id="o1",
            new_state=OrderState.FILLED,
            ts_ns=40,
            reason="f",
        )
        fsm.transition(
            order_id="o1",
            new_state=OrderState.CLOSED,
            ts_ns=50,
            reason="c",
        )
        return tuple(
            (h.ts_ns, h.prev, h.new, h.reason)
            for h in fsm.get("o1").history
        )

    assert run() == run()


# ---------------------------------------------------------------------------
# FillHandler
# ---------------------------------------------------------------------------


def test_fill_handler_register_drives_fsm_to_pending():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=10.0, ts_ns=2)
    assert fsm.get("o1").state is OrderState.PENDING


def test_fill_handler_full_fill_drives_to_filled_with_avg_price():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=10.0, ts_ns=2)
    state = handler.apply(
        FillEvent(ts_ns=10, order_id="o1", qty=10.0, price=100.0)
    )
    assert fsm.get("o1").state is OrderState.FILLED
    assert state.filled_qty == 10.0
    assert state.avg_price == 100.0


def test_fill_handler_partial_fills_compute_vwap():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=10.0, ts_ns=2)
    handler.apply(FillEvent(ts_ns=10, order_id="o1", qty=4.0, price=100.0))
    handler.apply(FillEvent(ts_ns=11, order_id="o1", qty=6.0, price=110.0))
    state = handler.state("o1")
    assert state is not None
    assert state.filled_qty == 10.0
    # vwap = (4*100 + 6*110) / 10 = 1060 / 10
    assert state.avg_price == pytest.approx(106.0)
    assert fsm.get("o1").state is OrderState.FILLED


def test_fill_handler_intermediate_partial_state():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=10.0, ts_ns=2)
    handler.apply(FillEvent(ts_ns=10, order_id="o1", qty=3.0, price=100.0))
    assert fsm.get("o1").state is OrderState.PARTIALLY_FILLED


def test_fill_handler_overfill_raises():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=10.0, ts_ns=2)
    with pytest.raises(ValueError):
        handler.apply(FillEvent(ts_ns=10, order_id="o1", qty=11.0, price=100.0))


def test_fill_handler_duplicate_fill_idempotent():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=10.0, ts_ns=2)
    fill = FillEvent(ts_ns=10, order_id="o1", qty=4.0, price=100.0)
    handler.apply(fill)
    handler.apply(fill)
    state = handler.state("o1")
    assert state is not None
    assert state.filled_qty == 4.0


def test_fill_handler_unknown_order_raises():
    fsm = OrderStateMachine()
    handler = FillHandler(fsm)
    with pytest.raises(KeyError):
        handler.apply(FillEvent(ts_ns=1, order_id="ghost", qty=1.0, price=1.0))


def test_fill_handler_register_failure_does_not_create_book_entry():
    """If FSM rejects the transition, no book entry must remain."""
    fsm = OrderStateMachine()
    handler = FillHandler(fsm)
    # Order was never opened on the FSM.
    with pytest.raises(KeyError):
        handler.register(order_id="ghost", target_qty=10.0, ts_ns=1)
    assert handler.state("ghost") is None
    # Re-trying after opening the order must succeed cleanly.
    fsm.open(order_id="ghost", ts_ns=2)
    state = handler.register(order_id="ghost", target_qty=10.0, ts_ns=3)
    assert state.target_qty == 10.0
    assert fsm.get("ghost").state is OrderState.PENDING


def test_fill_handler_zero_qty_or_price_raises():
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=10.0, ts_ns=2)
    with pytest.raises(ValueError):
        handler.apply(FillEvent(ts_ns=10, order_id="o1", qty=0.0, price=100.0))
    with pytest.raises(ValueError):
        handler.apply(FillEvent(ts_ns=10, order_id="o1", qty=1.0, price=0.0))


# ---------------------------------------------------------------------------
# PartialFillResolver
# ---------------------------------------------------------------------------


def _state(filled: float, target: float):
    fsm = OrderStateMachine()
    fsm.open(order_id="o1", ts_ns=1)
    handler = FillHandler(fsm)
    handler.register(order_id="o1", target_qty=target, ts_ns=2)
    if filled > 0.0:
        handler.apply(FillEvent(ts_ns=3, order_id="o1", qty=filled, price=100.0))
    state = handler.state("o1")
    assert state is not None
    return state


def test_resolver_fully_filled_returns_mark_filled():
    resolver = PartialFillResolver()
    assert resolver.resolve(_state(10.0, 10.0)) is PartialFillResolution.MARK_FILLED


def test_resolver_above_threshold_returns_mark_filled():
    resolver = PartialFillResolver(ResolutionContext(min_fill_ratio=0.95))
    assert (
        resolver.resolve(_state(9.7, 10.0))
        is PartialFillResolution.MARK_FILLED
    )


def test_resolver_venue_done_cancels_remainder():
    resolver = PartialFillResolver()
    assert (
        resolver.resolve(_state(4.0, 10.0), venue_says_done=True)
        is PartialFillResolution.CANCEL_REMAINDER
    )


def test_resolver_below_threshold_leaves_open():
    resolver = PartialFillResolver()
    assert (
        resolver.resolve(_state(4.0, 10.0))
        is PartialFillResolution.LEAVE_OPEN
    )


def test_resolver_cancel_after_ratio_triggers_cancel():
    resolver = PartialFillResolver(
        ResolutionContext(min_fill_ratio=0.99, cancel_after_ratio=0.5)
    )
    assert (
        resolver.resolve(_state(6.0, 10.0))
        is PartialFillResolution.CANCEL_REMAINDER
    )


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


def test_retry_classify_buckets():
    p = RetryPolicy()
    assert p.classify("TIMEOUT") is RetryClassification.TRANSIENT
    assert p.classify("RATE_LIMIT") is RetryClassification.THROTTLED
    assert p.classify("INSUFFICIENT_BALANCE") is RetryClassification.PERMANENT
    assert p.classify("UNHEARD_OF") is RetryClassification.PERMANENT


def test_retry_decide_transient_within_attempts():
    p = RetryPolicy(max_attempts=3, base_backoff_ns=100, backoff_factor=2.0)
    d = p.decide(error_code="TIMEOUT", attempt=1)
    assert d.should_retry
    assert d.classification is RetryClassification.TRANSIENT
    assert d.backoff_ns == 100


def test_retry_decide_attempt_grows_backoff():
    p = RetryPolicy(max_attempts=4, base_backoff_ns=100, backoff_factor=2.0)
    d1 = p.decide(error_code="TIMEOUT", attempt=1)
    d2 = p.decide(error_code="TIMEOUT", attempt=2)
    assert d2.backoff_ns == d1.backoff_ns * 2


def test_retry_decide_throttled_uses_higher_floor():
    p = RetryPolicy(base_backoff_ns=100)
    d_t = p.decide(error_code="TIMEOUT", attempt=1)
    d_th = p.decide(error_code="RATE_LIMIT", attempt=1)
    assert d_th.backoff_ns > d_t.backoff_ns


def test_retry_decide_permanent_never_retries():
    p = RetryPolicy()
    d = p.decide(error_code="INSUFFICIENT_BALANCE", attempt=1)
    assert not d.should_retry
    assert d.classification is RetryClassification.PERMANENT


def test_retry_decide_max_attempts_caps_retry():
    p = RetryPolicy(max_attempts=2)
    d = p.decide(error_code="TIMEOUT", attempt=2)
    assert not d.should_retry
    assert d.reason == "max_attempts_exceeded"


def test_retry_decide_rejects_invalid_attempt():
    p = RetryPolicy()
    with pytest.raises(ValueError):
        p.decide(error_code="TIMEOUT", attempt=0)


def test_retry_backoff_clamped_to_max():
    p = RetryPolicy(
        max_attempts=10,
        base_backoff_ns=1_000_000_000,
        backoff_factor=10.0,
        max_backoff_ns=1_500_000_000,
    )
    d = p.decide(error_code="TIMEOUT", attempt=3)
    assert d.backoff_ns == 1_500_000_000


# ---------------------------------------------------------------------------
# SLTPManager
# ---------------------------------------------------------------------------


def test_bracket_buy_take_profit_triggers_above():
    mgr = SLTPManager()
    mgr.attach(
        Bracket(
            order_id="o1",
            side=Side.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
    )
    assert (
        mgr.evaluate(order_id="o1", mark=111.0).trigger
        is BracketTrigger.TAKE_PROFIT
    )


def test_bracket_buy_stop_loss_triggers_below():
    mgr = SLTPManager()
    mgr.attach(
        Bracket(
            order_id="o1",
            side=Side.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
    )
    assert (
        mgr.evaluate(order_id="o1", mark=94.0).trigger
        is BracketTrigger.STOP_LOSS
    )


def test_bracket_sell_take_profit_triggers_below():
    mgr = SLTPManager()
    mgr.attach(
        Bracket(
            order_id="o1",
            side=Side.SELL,
            entry_price=100.0,
            stop_loss=110.0,
            take_profit=90.0,
        )
    )
    assert (
        mgr.evaluate(order_id="o1", mark=89.0).trigger
        is BracketTrigger.TAKE_PROFIT
    )


def test_bracket_sell_stop_loss_triggers_above():
    mgr = SLTPManager()
    mgr.attach(
        Bracket(
            order_id="o1",
            side=Side.SELL,
            entry_price=100.0,
            stop_loss=110.0,
            take_profit=90.0,
        )
    )
    assert (
        mgr.evaluate(order_id="o1", mark=111.0).trigger
        is BracketTrigger.STOP_LOSS
    )


def test_bracket_no_trigger_inside_range():
    mgr = SLTPManager()
    mgr.attach(
        Bracket(
            order_id="o1",
            side=Side.BUY,
            entry_price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
    )
    assert (
        mgr.evaluate(order_id="o1", mark=100.5).trigger
        is BracketTrigger.NONE
    )


def test_bracket_unknown_order_returns_none_trigger():
    mgr = SLTPManager()
    assert (
        mgr.evaluate(order_id="ghost", mark=100.0).trigger
        is BracketTrigger.NONE
    )


def test_bracket_invalid_buy_levels_rejected():
    mgr = SLTPManager()
    with pytest.raises(ValueError):
        mgr.attach(
            Bracket(
                order_id="o1",
                side=Side.BUY,
                entry_price=100.0,
                stop_loss=110.0,  # SL above entry on a BUY → invalid
                take_profit=120.0,
            )
        )


def test_bracket_hold_side_rejected():
    mgr = SLTPManager()
    with pytest.raises(ValueError):
        mgr.attach(
            Bracket(
                order_id="o1",
                side=Side.HOLD,
                entry_price=100.0,
                stop_loss=95.0,
                take_profit=110.0,
            )
        )


def test_bracket_double_attach_rejected():
    mgr = SLTPManager()
    b = Bracket(order_id="o1", side=Side.BUY, entry_price=100.0, stop_loss=95.0)
    mgr.attach(b)
    with pytest.raises(ValueError):
        mgr.attach(b)


def test_bracket_detach_then_reattach():
    mgr = SLTPManager()
    b = Bracket(order_id="o1", side=Side.BUY, entry_price=100.0, stop_loss=95.0)
    mgr.attach(b)
    mgr.detach("o1")
    mgr.attach(b)
    assert mgr.get("o1") is not None


def test_bracket_evaluate_invalid_mark_raises():
    mgr = SLTPManager()
    mgr.attach(
        Bracket(order_id="o1", side=Side.BUY, entry_price=100.0, stop_loss=95.0)
    )
    with pytest.raises(ValueError):
        mgr.evaluate(order_id="o1", mark=0.0)
