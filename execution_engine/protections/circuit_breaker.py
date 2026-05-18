"""Circuit breaker — A-20.2 / EXEC-protections.

Deterministic consecutive-loss circuit breaker over recorded
:class:`StoplossEvent` records, plus a post-trip cooldown timer. The
breaker is a pure value-object emitter; it never mutates the ledger and
never constructs typed bus events directly.

# ADAPTED FROM: freqtrade/plugins/protections/stoploss_guard.py (math + counting)
# ADAPTED FROM: freqtrade/plugins/protections/cooldown_period.py (cooldown timer math)
# GPL-3.0 mitigation: only the *counting + lockout-window math* is reused;
# no freqtrade class is imported, subclassed or referenced. The DIX-native
# implementation is a pure Python deterministic state machine.

Spec line 1547 (A-20.2) calls for the breaker to "emit HazardEvent to
governance". Per the :class:`HazardEvent` authority docstring
(``core/contracts/events.py``) HazardEvent is the **system_engine**
producer's typed event. The breaker therefore returns a
:class:`CircuitBreakerVerdict` (frozen value object) that the
integrating system engine materialises into a HazardEvent — the
production seam mirrors the runtime_monitor pattern.

Tier discipline
---------------

* OFFLINE_ONLY counting math (caller supplies ``ts_ns`` from each
  source event; the breaker never reads a clock).
* INV-15 byte-identical replay: deterministic ``BLAKE2b-16``
  ``policy_digest`` over the policy's canonical text projection;
  3-run replay equality + lookup-window monotonicity pinned in tests.
* B27 / B28 / INV-71 authority symmetry: this module returns value
  objects only — no ``HazardEvent`` / ``SignalEvent`` /
  ``ExecutionEvent`` / ``GovernanceDecision`` / ``LearningUpdate`` /
  ``PatchProposal`` / ``TraderObservation`` constructor calls.
  Pinned by AST tests.
* No ``governance_engine`` / ``system_engine`` / ``intelligence_engine``
  / ``evolution_engine`` imports (B1).
* No ``random`` / ``asyncio`` / ``os`` / ``datetime`` / ``time`` /
  ``numpy`` / ``torch`` / ``polars`` / ``pandas`` imports.

``NEW_PIP_DEPENDENCIES = ()`` — pure stdlib only.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from typing import Final

# ---------------------------------------------------------------------------
# Constants (canonical defaults mirror freqtrade plugin defaults).
# ---------------------------------------------------------------------------

DEFAULT_TRADE_LIMIT: Final[int] = 4
"""Number of qualifying stoplosses inside ``lookback_ns`` that trip the breaker."""

DEFAULT_LOOKBACK_NS: Final[int] = 60 * 60 * 1_000_000_000  # 60 minutes
"""Rolling lookback window (nanoseconds) for the loss counter."""

DEFAULT_PROFIT_LIMIT: Final[float] = 0.0
"""Stoplosses with ``close_profit < profit_limit`` are counted as qualifying."""

DEFAULT_STOP_DURATION_NS: Final[int] = 60 * 60 * 1_000_000_000  # 60 minutes
"""Post-trip cooldown (nanoseconds) during which new entries are locked."""

MAX_EVENT_BUFFER: Final[int] = 4096
"""Hard cap on retained ``StoplossEvent`` history (FIFO eviction)."""

NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StoplossExitReason(enum.Enum):
    """Subset of freqtrade ``ExitType`` values that count toward the breaker.

    Mirrors freqtrade's ``stoploss_guard._stoploss_guard`` filter; any
    other exit reason (``ROI``, ``EXIT_SIGNAL``, manual close, ...) is
    *not* counted as a qualifying stoploss.
    """

    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP_LOSS = "TRAILING_STOP_LOSS"
    STOPLOSS_ON_EXCHANGE = "STOPLOSS_ON_EXCHANGE"
    LIQUIDATION = "LIQUIDATION"


class Side(enum.Enum):
    """Trade direction (matches freqtrade ``LongShort``)."""

    LONG = "LONG"
    SHORT = "SHORT"


class CircuitBreakerState(enum.Enum):
    """High-level breaker state."""

    ARMED = "ARMED"  # not locked, counting events
    TRIPPED = "TRIPPED"  # locked, in cooldown until ``locked_until_ns``


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class StoplossEvent:
    """One recorded stoploss exit, fed to the breaker by the caller.

    All fields are caller-supplied — the breaker never queries any
    external store and never reads a clock.
    """

    ts_ns: int
    pair: str
    side: Side
    close_profit: float
    exit_reason: StoplossExitReason
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")
        if not self.pair:
            raise ValueError("pair must be non-empty")
        if not isinstance(self.side, Side):
            raise TypeError("side must be a Side enum")
        if not isinstance(self.exit_reason, StoplossExitReason):
            raise TypeError("exit_reason must be a StoplossExitReason enum")


@dataclasses.dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    """Frozen breaker configuration.

    Adapted from freqtrade ``StoplossGuard`` + ``CooldownPeriod``
    constants. All time fields are nanoseconds (DIX canonical unit).
    """

    trade_limit: int = DEFAULT_TRADE_LIMIT
    lookback_ns: int = DEFAULT_LOOKBACK_NS
    profit_limit: float = DEFAULT_PROFIT_LIMIT
    stop_duration_ns: int = DEFAULT_STOP_DURATION_NS
    only_per_pair: bool = False
    only_per_side: bool = False

    def __post_init__(self) -> None:
        if self.trade_limit < 1:
            raise ValueError("trade_limit must be >= 1")
        if self.lookback_ns <= 0:
            raise ValueError("lookback_ns must be > 0")
        if self.stop_duration_ns < 0:
            raise ValueError("stop_duration_ns must be >= 0")
        if not isinstance(self.only_per_pair, bool):
            raise TypeError("only_per_pair must be bool")
        if not isinstance(self.only_per_side, bool):
            raise TypeError("only_per_side must be bool")

    def canonical_text(self) -> str:
        """Sorted-key text projection used by ``policy_digest``."""
        return (
            f"trade_limit={self.trade_limit}|"
            f"lookback_ns={self.lookback_ns}|"
            f"profit_limit={self.profit_limit!r}|"
            f"stop_duration_ns={self.stop_duration_ns}|"
            f"only_per_pair={self.only_per_pair}|"
            f"only_per_side={self.only_per_side}"
        )

    def policy_digest(self) -> str:
        """BLAKE2b-16 hex digest over ``canonical_text``."""
        return hashlib.blake2b(self.canonical_text().encode("utf-8"), digest_size=16).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class CircuitBreakerVerdict:
    """Verdict returned by :meth:`CircuitBreaker.evaluate`.

    A caller (typically the system engine) materialises this into a
    typed ``HazardEvent`` and forwards it to governance. The verdict
    is a pure value object — no IO, no event construction.
    """

    state: CircuitBreakerState
    is_locked: bool
    now_ns: int
    locked_until_ns: int
    lock_side: str  # "*" | "LONG" | "SHORT"
    lock_pair: str  # "*" | "BTC/USDT" | ...
    qualifying_count: int
    trade_limit: int
    reason: str
    policy_digest: str
    meta: Mapping[str, str] = dataclasses.field(default_factory=dict)


# ---------------------------------------------------------------------------
# Breaker
# ---------------------------------------------------------------------------


class CircuitBreaker:
    """Deterministic consecutive-loss circuit breaker.

    Usage::

        breaker = CircuitBreaker(policy=CircuitBreakerPolicy(...))
        breaker.record(stoploss_event)
        verdict = breaker.evaluate(now_ns=now, pair="BTC/USDT", side=Side.LONG)
        if verdict.is_locked:
            # caller materialises HazardEvent to governance
            ...

    The breaker is stateful only in the FIFO event buffer (bounded by
    ``MAX_EVENT_BUFFER``) and the rolling lock-until timestamp. All
    decisions are derived from ``policy`` + ``now_ns`` so two breakers
    fed identical inputs produce identical outputs (INV-15).
    """

    name: Final[str] = "circuit_breaker"
    spec_id: Final[str] = "A-20.2"

    def __init__(
        self,
        *,
        policy: CircuitBreakerPolicy | None = None,
        max_buffer: int = MAX_EVENT_BUFFER,
    ) -> None:
        if max_buffer <= 0:
            raise ValueError("max_buffer must be > 0")
        self._policy = policy if policy is not None else CircuitBreakerPolicy()
        self._events: deque[StoplossEvent] = deque(maxlen=max_buffer)
        self._locked_until_ns: int = 0
        self._last_lock_pair: str = "*"
        self._last_lock_side: str = "*"
        self._last_qualifying_count: int = 0

    # -- queries -----------------------------------------------------------

    @property
    def policy(self) -> CircuitBreakerPolicy:
        return self._policy

    @property
    def event_count(self) -> int:
        return len(self._events)

    def events(self) -> Iterator[StoplossEvent]:
        """Yield retained events (FIFO order, oldest → newest)."""
        return iter(tuple(self._events))

    # -- mutations ---------------------------------------------------------

    def record(self, event: StoplossEvent) -> None:
        """Append one stoploss outcome (FIFO, monotonic in ``ts_ns``)."""
        if self._events and event.ts_ns < self._events[-1].ts_ns:
            raise ValueError("ts_ns must be monotonic non-decreasing across recorded events")
        self._events.append(event)

    def record_batch(self, events: Iterable[StoplossEvent]) -> None:
        for ev in events:
            self.record(ev)

    # -- decisions ---------------------------------------------------------

    def evaluate(
        self,
        *,
        now_ns: int,
        pair: str = "",
        side: Side | None = None,
        meta: Mapping[str, str] | None = None,
    ) -> CircuitBreakerVerdict:
        """Return the current breaker verdict.

        ``pair`` and ``side`` filter the count when ``only_per_pair`` /
        ``only_per_side`` are enabled on the policy. ``now_ns`` is
        caller-supplied (no clock reads).
        """
        if now_ns < 0:
            raise ValueError("now_ns must be >= 0")
        policy = self._policy
        # ── 1. Honour an in-flight cooldown ────────────────────────────
        if now_ns < self._locked_until_ns:
            return self._build_verdict(
                state=CircuitBreakerState.TRIPPED,
                is_locked=True,
                now_ns=now_ns,
                locked_until_ns=self._locked_until_ns,
                lock_side=self._last_lock_side,
                lock_pair=self._last_lock_pair,
                qualifying_count=self._last_qualifying_count,
                reason=(
                    f"cooldown active: locked until {self._locked_until_ns}ns "
                    f"({self._last_qualifying_count}/{policy.trade_limit} stoplosses)"
                ),
                meta=meta,
            )
        # ── 2. Count qualifying events inside the lookback window ──────
        window_start_ns = now_ns - policy.lookback_ns
        qualifying = self._count_qualifying(
            window_start_ns=window_start_ns,
            pair=pair if policy.only_per_pair else "",
            side=side if policy.only_per_side else None,
        )
        # ── 3. Trip if threshold met ──────────────────────────────────
        if qualifying >= policy.trade_limit:
            locked_until_ns = now_ns + policy.stop_duration_ns
            self._locked_until_ns = locked_until_ns
            self._last_lock_pair = pair if policy.only_per_pair else "*"
            self._last_lock_side = side.value if policy.only_per_side and side else "*"
            self._last_qualifying_count = qualifying
            return self._build_verdict(
                state=CircuitBreakerState.TRIPPED,
                is_locked=True,
                now_ns=now_ns,
                locked_until_ns=locked_until_ns,
                lock_side=self._last_lock_side,
                lock_pair=self._last_lock_pair,
                qualifying_count=qualifying,
                reason=(
                    f"tripped: {qualifying}/{policy.trade_limit} qualifying "
                    f"stoplosses within {policy.lookback_ns}ns"
                ),
                meta=meta,
            )
        # ── 4. Armed (under threshold) ────────────────────────────────
        return self._build_verdict(
            state=CircuitBreakerState.ARMED,
            is_locked=False,
            now_ns=now_ns,
            locked_until_ns=0,
            lock_side="*",
            lock_pair="*",
            qualifying_count=qualifying,
            reason=(
                f"armed: {qualifying}/{policy.trade_limit} qualifying "
                f"stoplosses within {policy.lookback_ns}ns"
            ),
            meta=meta,
        )

    # -- internals ---------------------------------------------------------

    def _count_qualifying(
        self,
        *,
        window_start_ns: int,
        pair: str,
        side: Side | None,
    ) -> int:
        policy = self._policy
        count = 0
        for ev in self._events:
            if ev.ts_ns < window_start_ns:
                continue
            if ev.close_profit >= policy.profit_limit:
                continue
            if pair and ev.pair != pair:
                continue
            if side is not None and ev.side is not side:
                continue
            count += 1
        return count

    def _build_verdict(
        self,
        *,
        state: CircuitBreakerState,
        is_locked: bool,
        now_ns: int,
        locked_until_ns: int,
        lock_side: str,
        lock_pair: str,
        qualifying_count: int,
        reason: str,
        meta: Mapping[str, str] | None,
    ) -> CircuitBreakerVerdict:
        merged_meta: Mapping[str, str]
        if meta:
            merged_meta = dict(sorted(meta.items()))
        else:
            merged_meta = {}
        return CircuitBreakerVerdict(
            state=state,
            is_locked=is_locked,
            now_ns=now_ns,
            locked_until_ns=locked_until_ns,
            lock_side=lock_side,
            lock_pair=lock_pair,
            qualifying_count=qualifying_count,
            trade_limit=self._policy.trade_limit,
            reason=reason,
            policy_digest=self._policy.policy_digest(),
            meta=merged_meta,
        )


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerPolicy",
    "CircuitBreakerState",
    "CircuitBreakerVerdict",
    "DEFAULT_LOOKBACK_NS",
    "DEFAULT_PROFIT_LIMIT",
    "DEFAULT_STOP_DURATION_NS",
    "DEFAULT_TRADE_LIMIT",
    "MAX_EVENT_BUFFER",
    "NEW_PIP_DEPENDENCIES",
    "Side",
    "StoplossEvent",
    "StoplossExitReason",
]
