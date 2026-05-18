# ADAPTED FROM: jesse-ai/jesse (jesse/strategies/Strategy.py) patterns —
# extracted lifecycle hooks (before / go_long / go_short /
# update_position / should_cancel_entry / after) and rewritten as a
# pure-Python deterministic FSM. No jesse import.
"""Archetype-driven strategy lifecycle FSM (B-09).

Wraps the existing :class:`~intelligence_engine.meta.trader_archetypes
.TraderArchetypeRegistry` rows (TI-CONS — 300 hand-built archetypes)
with a Jesse-style strategy lifecycle. Each archetype carries its own
:class:`ArchetypeStrategy` (the pattern that fires entries / manages
exits); the lifecycle FSM coordinates the canonical 5-hook sequence
once per bar:

::

    before()
        -> go_long() / go_short()         (only when flat)
            -> should_cancel_entry()      (only when pending entry)
                -> update_position()       (only when in a position)
                    -> after()             (post-tick read-only summary)

Authority constraints
---------------------
* OFFLINE-tier intelligence runtime: this module is the
  archetype-driven coordinator on the **intelligence** side of the
  authority boundary. It does **not** construct
  :class:`~core.contracts.events.SignalEvent` /
  :class:`~core.contracts.execution.ExecutionIntent` /
  :class:`~core.contracts.learning.PatchProposal` /
  :class:`~core.contracts.governance.GovernanceDecision` — those
  remain the exclusive responsibility of their owning engines
  (B27 / B28 / INV-71).
* No clock: every timestamp comes from the caller-supplied
  :class:`ArchetypeContext`; the module never imports ``datetime`` /
  ``time`` / :mod:`system.time_source`.
* No PRNG: lifecycle decisions are pure functions of the caller's
  context.
* No IO: pure in-memory state machine.
* No engine cross-imports: pinned by an AST authority test.

INV-15 (replay determinism)
---------------------------
Two callers that pass the same ``(state, ctx, strategy)`` produce
byte-identical :class:`ArchetypeDecision` outputs:

* Hooks are called in the canonical order documented above.
* :func:`advance_lifecycle` is a pure transition: it takes the
  current state and returns the next state plus the emitted decision
  — never mutates caller objects.
* All value-objects are frozen + slotted; tuple-of-strings rationale
  tags are kept in insertion order (caller responsibility).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""Pure stdlib — no jesse, no numpy, no pandas."""


_MAX_RATIONALE_LEN = 256
_MAX_RATIONALE_TAGS = 16
_MAX_ARCHETYPE_ID_LEN = 32
_MAX_SYMBOL_LEN = 32


class ArchetypeLifecycleError(ValueError):
    """Raised when lifecycle inputs violate the contract."""


class ArchetypeStateError(RuntimeError):
    """Raised when the FSM is asked to fire a hook in an invalid state."""


class Side(StrEnum):
    """Long / short / flat directional state."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class LifecycleState(StrEnum):
    """Five-state archetype lifecycle FSM.

    ADAPTED FROM: jesse/strategies/Strategy.py — TradingMode names.
    Renamed for clarity; same semantics.
    """

    IDLE = "IDLE"
    """Flat, no pending entry; next hook is ``go_long`` / ``go_short``."""

    PENDING_LONG = "PENDING_LONG"
    """Pending long entry awaiting fill; next hook is
    ``should_cancel_entry``."""

    PENDING_SHORT = "PENDING_SHORT"
    """Pending short entry awaiting fill; next hook is
    ``should_cancel_entry``."""

    OPEN_LONG = "OPEN_LONG"
    """Open long position; next hook is ``update_position``."""

    OPEN_SHORT = "OPEN_SHORT"
    """Open short position; next hook is ``update_position``."""


@dataclass(frozen=True, slots=True)
class PositionSnapshot:
    """Read-only snapshot of the open position passed to
    ``update_position``."""

    side: Side
    qty: float
    entry_price: float
    unrealised_pnl_usd: float
    bars_held: int

    def __post_init__(self) -> None:
        if self.side is Side.FLAT:
            raise ArchetypeLifecycleError(
                "PositionSnapshot.side must be LONG or SHORT (never FLAT)"
            )
        if not math.isfinite(self.qty) or self.qty <= 0.0:
            raise ArchetypeLifecycleError(f"PositionSnapshot.qty must be > 0; got {self.qty!r}")
        if not math.isfinite(self.entry_price) or self.entry_price < 0.0:
            raise ArchetypeLifecycleError("PositionSnapshot.entry_price must be finite and >= 0")
        if not math.isfinite(self.unrealised_pnl_usd):
            raise ArchetypeLifecycleError("PositionSnapshot.unrealised_pnl_usd must be finite")
        if self.bars_held < 0:
            raise ArchetypeLifecycleError("PositionSnapshot.bars_held must be >= 0")


@dataclass(frozen=True, slots=True)
class PendingEntry:
    """Read-only snapshot of a pending entry passed to
    ``should_cancel_entry``."""

    side: Side
    qty: float
    limit_price: float
    bars_pending: int

    def __post_init__(self) -> None:
        if self.side is Side.FLAT:
            raise ArchetypeLifecycleError("PendingEntry.side must be LONG or SHORT (never FLAT)")
        if not math.isfinite(self.qty) or self.qty <= 0.0:
            raise ArchetypeLifecycleError(f"PendingEntry.qty must be > 0; got {self.qty!r}")
        if not math.isfinite(self.limit_price) or self.limit_price < 0.0:
            raise ArchetypeLifecycleError("PendingEntry.limit_price must be finite and >= 0")
        if self.bars_pending < 0:
            raise ArchetypeLifecycleError("PendingEntry.bars_pending must be >= 0")


@dataclass(frozen=True, slots=True)
class ArchetypeContext:
    """Per-bar context fed to every lifecycle hook.

    Args:
        ts_ns: Bar timestamp (caller-supplied, INV-15).
        archetype_id: TI-CONS archetype id (e.g. ``TA-TREND-001``).
        symbol: Instrument symbol.
        bar_index: Monotonic bar counter inside this run.
        last_price: Closing price of the current bar (for distance /
            unrealised-P&L math).
        position: Optional open position (None when flat).
        pending: Optional pending entry awaiting fill.
        features: Caller-supplied read-only feature map. Keys must be
            ``str``; values must be ``float`` / ``int`` / ``bool`` /
            ``str``.
    """

    ts_ns: int
    archetype_id: str
    symbol: str
    bar_index: int
    last_price: float
    position: PositionSnapshot | None = None
    pending: PendingEntry | None = None
    features: Mapping[str, float | int | bool | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ArchetypeLifecycleError("ts_ns must be >= 0")
        if not self.archetype_id:
            raise ArchetypeLifecycleError("archetype_id must be non-empty")
        if len(self.archetype_id) > _MAX_ARCHETYPE_ID_LEN:
            raise ArchetypeLifecycleError(f"archetype_id must be <= {_MAX_ARCHETYPE_ID_LEN} chars")
        if not self.symbol:
            raise ArchetypeLifecycleError("symbol must be non-empty")
        if len(self.symbol) > _MAX_SYMBOL_LEN:
            raise ArchetypeLifecycleError(f"symbol must be <= {_MAX_SYMBOL_LEN} chars")
        if self.bar_index < 0:
            raise ArchetypeLifecycleError("bar_index must be >= 0")
        if not math.isfinite(self.last_price) or self.last_price < 0.0:
            raise ArchetypeLifecycleError("last_price must be finite and >= 0")
        for key, value in self.features.items():
            if not isinstance(key, str):
                raise ArchetypeLifecycleError("ArchetypeContext.features keys must be str")
            if not isinstance(value, (float, int, bool, str)):
                raise ArchetypeLifecycleError(
                    "ArchetypeContext.features values must be float/int/bool/str"
                )


@dataclass(frozen=True, slots=True)
class EntryDecision:
    """Long / short entry probe result.

    ADAPTED FROM: jesse/strategies/Strategy.go_long / .go_short — the
    Jesse hooks set ``self.buy = (qty, price)`` / ``self.sell = ...``.
    DIX renames this into an explicit frozen return value so the seam
    is pure.
    """

    side: Side
    qty: float
    limit_price: float
    stop_loss_price: float
    take_profit_price: float
    rationale_tags: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.side is Side.FLAT:
            raise ArchetypeLifecycleError("EntryDecision.side must be LONG or SHORT (never FLAT)")
        for name, value in (
            ("qty", self.qty),
            ("limit_price", self.limit_price),
            ("stop_loss_price", self.stop_loss_price),
            ("take_profit_price", self.take_profit_price),
        ):
            if not math.isfinite(value):
                raise ArchetypeLifecycleError(f"EntryDecision.{name} must be finite; got {value!r}")
        if self.qty <= 0.0:
            raise ArchetypeLifecycleError("EntryDecision.qty must be > 0")
        if self.limit_price < 0.0:
            raise ArchetypeLifecycleError("EntryDecision.limit_price must be >= 0")
        if self.stop_loss_price < 0.0:
            raise ArchetypeLifecycleError("EntryDecision.stop_loss_price must be >= 0")
        if self.take_profit_price < 0.0:
            raise ArchetypeLifecycleError("EntryDecision.take_profit_price must be >= 0")
        if self.side is Side.LONG:
            if self.stop_loss_price >= self.limit_price:
                raise ArchetypeLifecycleError("LONG entry: stop_loss_price must be < limit_price")
            if self.take_profit_price <= self.limit_price:
                raise ArchetypeLifecycleError("LONG entry: take_profit_price must be > limit_price")
        else:  # SHORT
            if self.stop_loss_price <= self.limit_price:
                raise ArchetypeLifecycleError("SHORT entry: stop_loss_price must be > limit_price")
            if self.take_profit_price >= self.limit_price:
                raise ArchetypeLifecycleError(
                    "SHORT entry: take_profit_price must be < limit_price"
                )
        if len(self.rationale_tags) > _MAX_RATIONALE_TAGS:
            raise ArchetypeLifecycleError(
                f"rationale_tags must be <= {_MAX_RATIONALE_TAGS} entries"
            )
        for tag in self.rationale_tags:
            if not isinstance(tag, str):
                raise ArchetypeLifecycleError("rationale_tags entries must be str")
        if len(self.rationale) > _MAX_RATIONALE_LEN:
            raise ArchetypeLifecycleError(f"rationale must be <= {_MAX_RATIONALE_LEN} chars")


class PositionAction(StrEnum):
    """Three actions ``update_position`` may emit."""

    HOLD = "HOLD"
    CLOSE = "CLOSE"
    ADJUST_STOPS = "ADJUST_STOPS"


@dataclass(frozen=True, slots=True)
class PositionUpdate:
    """Result of a single ``update_position`` call.

    ADAPTED FROM: jesse/strategies/Strategy.update_position — Jesse
    mutates ``self.stop_loss`` / ``self.take_profit`` or calls
    ``self.liquidate()``. DIX renames into an explicit frozen value
    object.
    """

    action: PositionAction
    new_stop_loss_price: float = 0.0
    new_take_profit_price: float = 0.0
    rationale_tags: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            ("new_stop_loss_price", self.new_stop_loss_price),
            ("new_take_profit_price", self.new_take_profit_price),
        ):
            if not math.isfinite(value):
                raise ArchetypeLifecycleError(f"PositionUpdate.{name} must be finite")
            if value < 0.0:
                raise ArchetypeLifecycleError(f"PositionUpdate.{name} must be >= 0")
        if self.action is PositionAction.ADJUST_STOPS:
            if self.new_stop_loss_price <= 0.0 and self.new_take_profit_price <= 0.0:
                raise ArchetypeLifecycleError(
                    "ADJUST_STOPS requires a positive new_stop_loss_price or new_take_profit_price"
                )
        if len(self.rationale_tags) > _MAX_RATIONALE_TAGS:
            raise ArchetypeLifecycleError(
                f"rationale_tags must be <= {_MAX_RATIONALE_TAGS} entries"
            )
        for tag in self.rationale_tags:
            if not isinstance(tag, str):
                raise ArchetypeLifecycleError("rationale_tags entries must be str")
        if len(self.rationale) > _MAX_RATIONALE_LEN:
            raise ArchetypeLifecycleError(f"rationale must be <= {_MAX_RATIONALE_LEN} chars")


class DecisionKind(StrEnum):
    """Five flavours of lifecycle outcome."""

    NO_OP = "NO_OP"
    OPEN_ENTRY = "OPEN_ENTRY"
    CANCEL_ENTRY = "CANCEL_ENTRY"
    HOLD_POSITION = "HOLD_POSITION"
    CLOSE_POSITION = "CLOSE_POSITION"
    ADJUST_STOPS = "ADJUST_STOPS"


@dataclass(frozen=True, slots=True)
class ArchetypeDecision:
    """Frozen advisory record returned by :func:`advance_lifecycle`.

    Downstream callers translate this into a typed bus event through
    the existing governance / execution surface. The lifecycle module
    itself never reaches into either engine.
    """

    kind: DecisionKind
    ts_ns: int
    archetype_id: str
    symbol: str
    bar_index: int
    state_before: LifecycleState
    state_after: LifecycleState
    entry: EntryDecision | None = None
    update: PositionUpdate | None = None
    rationale_tags: tuple[str, ...] = ()
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ArchetypeLifecycleError("ArchetypeDecision.ts_ns must be >= 0")
        if not self.archetype_id:
            raise ArchetypeLifecycleError("ArchetypeDecision.archetype_id must be non-empty")
        if not self.symbol:
            raise ArchetypeLifecycleError("ArchetypeDecision.symbol must be non-empty")
        if self.bar_index < 0:
            raise ArchetypeLifecycleError("ArchetypeDecision.bar_index must be >= 0")
        if self.kind is DecisionKind.OPEN_ENTRY and self.entry is None:
            raise ArchetypeLifecycleError("OPEN_ENTRY requires an EntryDecision")
        if self.kind is DecisionKind.ADJUST_STOPS and self.update is None:
            raise ArchetypeLifecycleError("ADJUST_STOPS requires a PositionUpdate")


@runtime_checkable
class ArchetypeStrategy(Protocol):
    """Pure-function Strategy seam.

    ADAPTED FROM: jesse/strategies/Strategy.py — Jesse's base class
    exposes mutable ``self`` hooks (``before`` / ``go_long`` /
    ``go_short`` / ``should_cancel_entry`` / ``update_position`` /
    ``after``). DIX makes them pure ``(ctx) -> ...`` mappings so
    INV-15 replay determinism holds across the whole intelligence
    tier.

    Concrete archetype classes implement this protocol; the lifecycle
    coordinator calls hooks in canonical order based on the current
    :class:`LifecycleState`.
    """

    def before(self, ctx: ArchetypeContext) -> None:  # pragma: no cover
        """Read-only pre-tick hook. MUST NOT mutate ``ctx``."""

    def go_long(self, ctx: ArchetypeContext) -> EntryDecision | None:  # pragma: no cover
        """Return a LONG :class:`EntryDecision`, or ``None`` to skip."""

    def go_short(self, ctx: ArchetypeContext) -> EntryDecision | None:  # pragma: no cover
        """Return a SHORT :class:`EntryDecision`, or ``None`` to skip."""

    def should_cancel_entry(
        self, ctx: ArchetypeContext, pending: PendingEntry
    ) -> bool:  # pragma: no cover
        """Return ``True`` to cancel the pending entry."""

    def update_position(
        self, ctx: ArchetypeContext, position: PositionSnapshot
    ) -> PositionUpdate:  # pragma: no cover
        """Return a :class:`PositionUpdate` (HOLD / CLOSE / ADJUST_STOPS)."""

    def after(self, ctx: ArchetypeContext) -> None:  # pragma: no cover
        """Read-only post-tick hook. MUST NOT mutate ``ctx``."""


@dataclass(frozen=True, slots=True)
class ArchetypeLifecycle:
    """Frozen lifecycle handle: ``(state, archetype_id, symbol)``.

    Use :func:`advance_lifecycle` to compute the next state.
    """

    state: LifecycleState
    archetype_id: str
    symbol: str

    def __post_init__(self) -> None:
        if not self.archetype_id:
            raise ArchetypeLifecycleError("ArchetypeLifecycle.archetype_id must be non-empty")
        if not self.symbol:
            raise ArchetypeLifecycleError("ArchetypeLifecycle.symbol must be non-empty")


def _check_ctx_matches(lifecycle: ArchetypeLifecycle, ctx: ArchetypeContext) -> None:
    if ctx.archetype_id != lifecycle.archetype_id:
        raise ArchetypeLifecycleError(
            f"ctx.archetype_id {ctx.archetype_id!r} != "
            f"lifecycle.archetype_id {lifecycle.archetype_id!r}"
        )
    if ctx.symbol != lifecycle.symbol:
        raise ArchetypeLifecycleError(
            f"ctx.symbol {ctx.symbol!r} != lifecycle.symbol {lifecycle.symbol!r}"
        )


def _check_entry_matches_ctx(entry: EntryDecision, ctx: ArchetypeContext, expected: Side) -> None:
    if entry.side is not expected:
        raise ArchetypeLifecycleError(
            f"strategy.{expected.value.lower()} returned side {entry.side!r}; expected {expected!r}"
        )


def _next_state_after_entry(entry: EntryDecision) -> LifecycleState:
    if entry.side is Side.LONG:
        return LifecycleState.PENDING_LONG
    return LifecycleState.PENDING_SHORT


def advance_lifecycle(
    *,
    lifecycle: ArchetypeLifecycle,
    strategy: ArchetypeStrategy,
    ctx: ArchetypeContext,
) -> tuple[ArchetypeLifecycle, ArchetypeDecision]:
    """Pure FSM transition.

    Calls the strategy hooks in canonical order based on
    ``lifecycle.state`` and returns the next lifecycle handle plus the
    emitted :class:`ArchetypeDecision`.

    Hook routing:
        * ``IDLE``           → ``before`` → ``go_long`` → ``go_short`` → ``after``
        * ``PENDING_LONG``   → ``before`` → ``should_cancel_entry`` → ``after``
        * ``PENDING_SHORT``  → ``before`` → ``should_cancel_entry`` → ``after``
        * ``OPEN_LONG``      → ``before`` → ``update_position`` → ``after``
        * ``OPEN_SHORT``     → ``before`` → ``update_position`` → ``after``

    Args:
        lifecycle: Current lifecycle handle.
        strategy: Object implementing :class:`ArchetypeStrategy`.
        ctx: Per-bar :class:`ArchetypeContext`. Must carry a
            :class:`PositionSnapshot` iff the state is OPEN_*, and a
            :class:`PendingEntry` iff the state is PENDING_*.

    Returns:
        Tuple ``(next_lifecycle, decision)``.

    Raises:
        ArchetypeLifecycleError: Inputs violate the contract.
        ArchetypeStateError: A hook returned a value inconsistent with
            the current state (e.g. a SHORT EntryDecision while the
            state was already PENDING_LONG).
    """
    if not isinstance(strategy, ArchetypeStrategy):
        raise ArchetypeLifecycleError("strategy must implement the ArchetypeStrategy Protocol")
    _check_ctx_matches(lifecycle, ctx)
    strategy.before(ctx)

    state = lifecycle.state
    if state is LifecycleState.IDLE:
        decision = _advance_idle(lifecycle, strategy, ctx)
    elif state in (LifecycleState.PENDING_LONG, LifecycleState.PENDING_SHORT):
        decision = _advance_pending(lifecycle, strategy, ctx)
    elif state in (LifecycleState.OPEN_LONG, LifecycleState.OPEN_SHORT):
        decision = _advance_open(lifecycle, strategy, ctx)
    else:  # pragma: no cover — exhaustive
        raise ArchetypeStateError(f"unhandled lifecycle state: {state!r}")

    strategy.after(ctx)
    next_lifecycle = ArchetypeLifecycle(
        state=decision.state_after,
        archetype_id=lifecycle.archetype_id,
        symbol=lifecycle.symbol,
    )
    return next_lifecycle, decision


def _advance_idle(
    lifecycle: ArchetypeLifecycle,
    strategy: ArchetypeStrategy,
    ctx: ArchetypeContext,
) -> ArchetypeDecision:
    if ctx.position is not None:
        raise ArchetypeStateError("IDLE state requires ctx.position is None")
    if ctx.pending is not None:
        raise ArchetypeStateError("IDLE state requires ctx.pending is None")
    long_entry = strategy.go_long(ctx)
    if long_entry is not None:
        _check_entry_matches_ctx(long_entry, ctx, Side.LONG)
        return ArchetypeDecision(
            kind=DecisionKind.OPEN_ENTRY,
            ts_ns=ctx.ts_ns,
            archetype_id=lifecycle.archetype_id,
            symbol=lifecycle.symbol,
            bar_index=ctx.bar_index,
            state_before=lifecycle.state,
            state_after=LifecycleState.PENDING_LONG,
            entry=long_entry,
            rationale_tags=long_entry.rationale_tags,
            rationale=long_entry.rationale,
        )
    short_entry = strategy.go_short(ctx)
    if short_entry is not None:
        _check_entry_matches_ctx(short_entry, ctx, Side.SHORT)
        return ArchetypeDecision(
            kind=DecisionKind.OPEN_ENTRY,
            ts_ns=ctx.ts_ns,
            archetype_id=lifecycle.archetype_id,
            symbol=lifecycle.symbol,
            bar_index=ctx.bar_index,
            state_before=lifecycle.state,
            state_after=LifecycleState.PENDING_SHORT,
            entry=short_entry,
            rationale_tags=short_entry.rationale_tags,
            rationale=short_entry.rationale,
        )
    return ArchetypeDecision(
        kind=DecisionKind.NO_OP,
        ts_ns=ctx.ts_ns,
        archetype_id=lifecycle.archetype_id,
        symbol=lifecycle.symbol,
        bar_index=ctx.bar_index,
        state_before=lifecycle.state,
        state_after=LifecycleState.IDLE,
    )


def _advance_pending(
    lifecycle: ArchetypeLifecycle,
    strategy: ArchetypeStrategy,
    ctx: ArchetypeContext,
) -> ArchetypeDecision:
    if ctx.pending is None:
        raise ArchetypeStateError(f"{lifecycle.state} requires ctx.pending != None")
    expected_side = Side.LONG if lifecycle.state is LifecycleState.PENDING_LONG else Side.SHORT
    if ctx.pending.side is not expected_side:
        raise ArchetypeStateError(
            f"{lifecycle.state} requires pending.side={expected_side!r}; got {ctx.pending.side!r}"
        )
    if ctx.position is not None:
        raise ArchetypeStateError(f"{lifecycle.state} requires ctx.position is None")
    cancel = strategy.should_cancel_entry(ctx, ctx.pending)
    if not isinstance(cancel, bool):
        raise ArchetypeLifecycleError("should_cancel_entry must return bool")
    if cancel:
        return ArchetypeDecision(
            kind=DecisionKind.CANCEL_ENTRY,
            ts_ns=ctx.ts_ns,
            archetype_id=lifecycle.archetype_id,
            symbol=lifecycle.symbol,
            bar_index=ctx.bar_index,
            state_before=lifecycle.state,
            state_after=LifecycleState.IDLE,
        )
    return ArchetypeDecision(
        kind=DecisionKind.NO_OP,
        ts_ns=ctx.ts_ns,
        archetype_id=lifecycle.archetype_id,
        symbol=lifecycle.symbol,
        bar_index=ctx.bar_index,
        state_before=lifecycle.state,
        state_after=lifecycle.state,
    )


def _advance_open(
    lifecycle: ArchetypeLifecycle,
    strategy: ArchetypeStrategy,
    ctx: ArchetypeContext,
) -> ArchetypeDecision:
    if ctx.position is None:
        raise ArchetypeStateError(f"{lifecycle.state} requires ctx.position != None")
    expected_side = Side.LONG if lifecycle.state is LifecycleState.OPEN_LONG else Side.SHORT
    if ctx.position.side is not expected_side:
        raise ArchetypeStateError(
            f"{lifecycle.state} requires position.side={expected_side!r}; got {ctx.position.side!r}"
        )
    if ctx.pending is not None:
        raise ArchetypeStateError(f"{lifecycle.state} requires ctx.pending is None")
    update = strategy.update_position(ctx, ctx.position)
    if not isinstance(update, PositionUpdate):
        raise ArchetypeLifecycleError("update_position must return a PositionUpdate")
    if update.action is PositionAction.CLOSE:
        return ArchetypeDecision(
            kind=DecisionKind.CLOSE_POSITION,
            ts_ns=ctx.ts_ns,
            archetype_id=lifecycle.archetype_id,
            symbol=lifecycle.symbol,
            bar_index=ctx.bar_index,
            state_before=lifecycle.state,
            state_after=LifecycleState.IDLE,
            update=update,
            rationale_tags=update.rationale_tags,
            rationale=update.rationale,
        )
    if update.action is PositionAction.ADJUST_STOPS:
        return ArchetypeDecision(
            kind=DecisionKind.ADJUST_STOPS,
            ts_ns=ctx.ts_ns,
            archetype_id=lifecycle.archetype_id,
            symbol=lifecycle.symbol,
            bar_index=ctx.bar_index,
            state_before=lifecycle.state,
            state_after=lifecycle.state,
            update=update,
            rationale_tags=update.rationale_tags,
            rationale=update.rationale,
        )
    return ArchetypeDecision(
        kind=DecisionKind.HOLD_POSITION,
        ts_ns=ctx.ts_ns,
        archetype_id=lifecycle.archetype_id,
        symbol=lifecycle.symbol,
        bar_index=ctx.bar_index,
        state_before=lifecycle.state,
        state_after=lifecycle.state,
        update=update,
        rationale_tags=update.rationale_tags,
        rationale=update.rationale,
    )


__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "ArchetypeContext",
    "ArchetypeDecision",
    "ArchetypeLifecycle",
    "ArchetypeLifecycleError",
    "ArchetypeStateError",
    "ArchetypeStrategy",
    "DecisionKind",
    "EntryDecision",
    "LifecycleState",
    "PendingEntry",
    "PositionAction",
    "PositionSnapshot",
    "PositionUpdate",
    "Side",
    "advance_lifecycle",
)
