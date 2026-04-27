"""Execution lifecycle (Phase 2 / v2-C).

The lifecycle layer owns the *post-submit* world of a single order — the
deterministic state machine that an :class:`ExecutionEvent` traverses
between ``PROPOSED`` and ``CLOSED`` once it leaves
:class:`ExecutionEngine.process`.

Build Compiler Spec §2 PHASE 2 deliverables:

* ``order_state_machine``      — FSM + legal-edge set
* ``fill_handler``             — fill bookkeeping (qty, avg_price)
* ``partial_fill_resolver``    — resolve partial vs final fills
* ``retry_logic``              — deterministic transient/permanent classifier
* ``sl_tp_manager``            — stop-loss / take-profit bracket lifecycle

Every module here is a pure-Python, IO-free building block consumed by
``ExecutionEngine``. No clocks, no randomness, no network — replay
determinism (INV-15) is preserved by construction.
"""

from execution_engine.lifecycle.fill_handler import (
    FillEvent,
    FillHandler,
    OrderFillState,
)
from execution_engine.lifecycle.order_state_machine import (
    LEGAL_ORDER_TRANSITIONS,
    OrderRecord,
    OrderState,
    OrderStateMachine,
    StateTransitionError,
)
from execution_engine.lifecycle.partial_fill_resolver import (
    PartialFillResolution,
    PartialFillResolver,
)
from execution_engine.lifecycle.retry_logic import (
    RetryClassification,
    RetryDecision,
    RetryPolicy,
)
from execution_engine.lifecycle.sl_tp_manager import (
    Bracket,
    BracketTrigger,
    SLTPManager,
)

__all__ = [
    "Bracket",
    "BracketTrigger",
    "FillEvent",
    "FillHandler",
    "LEGAL_ORDER_TRANSITIONS",
    "OrderFillState",
    "OrderRecord",
    "OrderState",
    "OrderStateMachine",
    "PartialFillResolution",
    "PartialFillResolver",
    "RetryClassification",
    "RetryDecision",
    "RetryPolicy",
    "SLTPManager",
    "StateTransitionError",
]
