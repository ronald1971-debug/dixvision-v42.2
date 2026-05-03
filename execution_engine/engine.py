"""ExecutionEngine — RUNTIME-ENGINE-02 (Phase E1, hardened in HARDEN-02 / 05).

The Execution Gate (INV-68) makes :meth:`ExecutionEngine.execute` the
*only* runtime path to a venue. HARDEN-05 removed the legacy
:meth:`process` shim that previously accepted a bare
:class:`SignalEvent` — the method now raises
:class:`LegacyExecutionPathRemovedError` to make any caller that still
relies on the old contract loud at runtime, instead of silently
degrading via a :class:`DeprecationWarning`.

Determinism contract (INV-15 / TEST-01):

* :meth:`execute` is a pure function of (intent, internal mark cache,
  adapter state). No clocks, no randomness, no external IO.
* :meth:`on_market` updates the internal mark cache with the last
  trade price; market ticks are *inputs*, not bus events (see
  ``core/contracts/market.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from core.contracts.engine import (
    EngineTier,
    HealthState,
    HealthStatus,
    Plugin,
    RuntimeEngine,
)
from core.contracts.events import (
    Event,
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    Side,
    SignalEvent,
)
from core.contracts.execution_intent import ExecutionIntent
from core.contracts.governance import SystemMode
from core.contracts.learning_sink import IntelligenceFeedbackSink
from core.contracts.market import MarketTick
from core.contracts.mode_effects import effect_for
from core.contracts.risk import RiskSnapshot
from execution_engine.adapters import BrokerAdapter, PaperBroker
from execution_engine.execution_gate import AuthorityGuard
from execution_engine.protections.feedback import FeedbackCollector
from system_engine.coupling import HazardThrottleAdapter

__all__ = ["ExecutionEngine", "LegacyExecutionPathRemovedError"]


class LegacyExecutionPathRemovedError(RuntimeError):
    """Raised when something calls the removed ``ExecutionEngine.process``.

    HARDEN-05 deleted the deprecated ``process(SignalEvent)`` path
    that bypassed the :class:`AuthorityGuard`. All trade-producing
    callers must now construct an :class:`ExecutionIntent` (via the
    ``governance_engine.harness_approver`` shim or the live governance
    pipeline) and call :meth:`ExecutionEngine.execute`.
    """


class ExecutionEngine(RuntimeEngine):
    name: str = "execution"
    tier: EngineTier = EngineTier.RUNTIME

    def __init__(
        self,
        adapter: BrokerAdapter | None = None,
        plugin_slots: Mapping[str, Sequence[Plugin]] | None = None,
        *,
        guard: AuthorityGuard | None = None,
        feedback_collector: FeedbackCollector | None = None,
        intelligence_feedback: IntelligenceFeedbackSink | None = None,
        throttle_adapter: HazardThrottleAdapter | None = None,
        risk_baseline: RiskSnapshot | None = None,
    ) -> None:
        self._adapter: BrokerAdapter = adapter or PaperBroker()
        self._marks: dict[str, float] = {}
        self.plugin_slots: Mapping[str, Sequence[Plugin]] = dict(
            plugin_slots or {}
        )
        # The AuthorityGuard is constructed lazily so unit tests that
        # only exercise broker plumbing don't need a matrix YAML on
        # disk. ``execute`` materialises the guard on first use
        # unless the caller injected one explicitly.
        self._guard: AuthorityGuard | None = guard
        # P0-3 — closed learning loop. Both sinks are optional so
        # offline harness flows that never construct them retain their
        # pre-P0-3 dispatch shape. Production wiring (cockpit /
        # bootstrap_kernel) injects real instances so terminal
        # ExecutionEvents flow into the learning + evolution
        # pipeline (Build Compiler Spec §8). The intelligence sink is
        # duck-typed via ``IntelligenceFeedbackSink`` so the import
        # arrow stays one-way: execution_engine → core.contracts
        # only (B1).
        self._feedback_collector: FeedbackCollector | None = (
            feedback_collector
        )
        self._intelligence_feedback: IntelligenceFeedbackSink | None = (
            intelligence_feedback
        )
        # P0-2: hazard throttle chain closure. The adapter is the
        # single seam through which observed HazardEvents tighten the
        # hot-path RiskSnapshot via apply_throttle(). Both attributes
        # are optional so existing harness flows that never touch the
        # hazard bus retain their pre-P0-2 behaviour.
        self._throttle_adapter: HazardThrottleAdapter | None = throttle_adapter
        self._risk_baseline: RiskSnapshot | None = risk_baseline

    @property
    def adapter(self) -> BrokerAdapter:
        return self._adapter

    @property
    def guard(self) -> AuthorityGuard:
        if self._guard is None:
            self._guard = AuthorityGuard()
        return self._guard

    def set_risk_baseline(self, snapshot: RiskSnapshot) -> None:
        """Replace the baseline :class:`RiskSnapshot`.

        The baseline is the un-throttled view from the FastRiskCache.
        On every :meth:`execute` call the throttle adapter projects
        the active hazard window onto this baseline.
        """

        self._risk_baseline = snapshot

    def on_hazard(self, event: HazardEvent) -> None:
        """Feed a :class:`HazardEvent` into the throttle adapter.

        No-op when no adapter was injected -- preserves the
        pre-P0-2 behaviour for harness flows that never touch the
        hazard bus.
        """

        if self._throttle_adapter is None:
            return
        self._throttle_adapter.observe(event)

    def on_market(self, tick: MarketTick) -> None:
        """Update the internal mark cache.

        Market ticks are inputs, not bus events. This keeps the canonical
        bus restricted to the four typed events (INV-08).
        """
        if tick.last > 0.0:
            self._marks[tick.symbol] = tick.last

    # ------------------------------------------------------------------
    # HARDEN-02 chokepoint
    # ------------------------------------------------------------------

    def execute(
        self,
        intent: ExecutionIntent,
        *,
        caller: str = "execution_engine",
        current_mode: SystemMode | None = None,
    ) -> Sequence[ExecutionEvent]:
        """The Execution Gate — single runtime path to a venue.

        The :class:`AuthorityGuard` validates the intent (governance
        approval, content-hash integrity, registered origin) before
        any adapter side effect. On guard failure the call raises
        :class:`UnauthorizedActorError` and a synthetic
        ``HAZ-AUTHORITY`` :class:`HazardEvent` is emitted via the
        guard's hazard sink (when configured).

        Wave-04.6 PR-B — when ``current_mode`` is supplied and the
        canonical mode-effect table reports ``executions_dispatch=False``
        for that mode (today: ``SAFE``, ``SHADOW``, ``LOCKED``), the
        gate passes the AuthorityGuard normally but **suppresses the
        broker side effect**, returning a single synthetic
        :class:`ExecutionEvent` with ``status=REJECTED`` and a
        machine-readable reason in ``meta``. This is the canonical
        SHADOW behaviour: signals are observed, ledgered, and audited
        without ever reaching a venue. Callers that omit
        ``current_mode`` retain the legacy unconditional-dispatch shape
        used by replay tests and harness flows that have already been
        gated upstream.

        Args:
            intent: A frozen, governance-approved
                :class:`ExecutionIntent`.
            caller: Runtime label of the engine invoking the gate.
                Defaults to ``"execution_engine"`` — the only value
                accepted by the default matrix.
            current_mode: Optional canonical :class:`SystemMode` for
                Wave-04.6 dispatch gating. ``None`` (default) preserves
                the pre-Wave-04.6 unconditional-dispatch behaviour.

        Returns:
            A short, ordered sequence of :class:`ExecutionEvent`
            envelopes — typically one fill per signal, possibly more
            if the underlying adapter splits, or a single mode-suppressed
            REJECTED event when ``current_mode`` denies dispatch.
        """

        self.guard.assert_can_execute(intent, caller=caller, ts_ns=intent.ts_ns)

        # P0-2: consult the throttle adapter (if configured) before
        # any side effect. ``halted=True`` short-circuits to a
        # REJECTED ExecutionEvent with reason ``hazard_throttled``;
        # ``qty_multiplier == 0`` is the same outcome via a different
        # path. The non-blocking ``confidence_floor`` /
        # ``qty_multiplier`` projections are consumed downstream by
        # the hot-path FastExecutor and are not enforced here.
        if (
            self._throttle_adapter is not None
            and self._risk_baseline is not None
        ):
            throttled = self._throttle_adapter.project(
                snapshot=self._risk_baseline,
                now_ns=intent.ts_ns,
            )
            if throttled.halted:
                signal = intent.signal
                throttled_events: tuple[ExecutionEvent, ...] = (
                    ExecutionEvent(
                        ts_ns=signal.ts_ns,
                        symbol=signal.symbol,
                        side=signal.side,
                        qty=0.0,
                        price=0.0,
                        status=ExecutionStatus.REJECTED,
                        venue=self._adapter.name,
                        order_id="",
                        meta={"reason": "hazard_throttled"},
                        produced_by_engine="execution_engine",
                    ),
                )
                # P0-3 — every terminal ExecutionEvent must reach the
                # learning loop, including hazard-throttled REJECTs.
                # Without this the freeze policy + outcome buffers
                # silently lose entire windows of refusal data.
                self._feed_learning_loop(intent.signal, throttled_events)
                return throttled_events

        if current_mode is not None and not effect_for(
            current_mode
        ).executions_dispatch:
            signal = intent.signal
            suppressed = (
                ExecutionEvent(
                    ts_ns=signal.ts_ns,
                    symbol=signal.symbol,
                    side=signal.side,
                    qty=0.0,
                    price=0.0,
                    status=ExecutionStatus.REJECTED,
                    venue=self._adapter.name,
                    order_id="",
                    meta={
                        "reason": "mode_effect_suppressed",
                        "mode": current_mode.name,
                    },
                    produced_by_engine="execution_engine",
                ),
            )
            self._feed_learning_loop(intent.signal, suppressed)
            return suppressed
        events = self._execute_signal(intent.signal)
        self._feed_learning_loop(intent.signal, events)
        return events

    # ------------------------------------------------------------------
    # HARDEN-05 — the legacy ``process(SignalEvent)`` path is gone. The
    # method is retained as a hard-fail tripwire so any code that still
    # carries the old call shape lights up at runtime instead of silently
    # smuggling a SignalEvent past the AuthorityGuard.
    # ------------------------------------------------------------------

    def process(self, event: Event) -> Sequence[Event]:
        raise LegacyExecutionPathRemovedError(
            "ExecutionEngine.process is removed (HARDEN-05). Construct "
            "an ExecutionIntent (via governance_engine.harness_approver "
            ".approve_signal_for_execution or the live governance "
            "pipeline) and call ExecutionEngine.execute(intent)."
        )

    # ------------------------------------------------------------------
    # Internal signal → execution kernel. Reachable only through
    # :meth:`execute` (post-AuthorityGuard).
    # ------------------------------------------------------------------

    def _execute_signal(self, event: SignalEvent) -> Sequence[ExecutionEvent]:
        # Phase E2: SHADOW signals from intelligence_engine are observed
        # but never reach a broker. Returning a REJECTED ExecutionEvent
        # keeps the audit trail visible without producing a live trade.
        if event.meta.get("shadow") == "true":
            return (
                ExecutionEvent(
                    ts_ns=event.ts_ns,
                    symbol=event.symbol,
                    side=event.side,
                    qty=0.0,
                    price=0.0,
                    status=ExecutionStatus.REJECTED,
                    venue=self._adapter.name,
                    order_id="",
                    meta={"reason": "shadow signal"},
                    produced_by_engine="execution_engine",
                ),
            )

        mark = self._marks.get(event.symbol, 0.0)
        if mark <= 0.0:
            return (
                ExecutionEvent(
                    ts_ns=event.ts_ns,
                    symbol=event.symbol,
                    side=event.side,
                    qty=0.0,
                    price=0.0,
                    status=ExecutionStatus.FAILED,
                    venue=self._adapter.name,
                    order_id="",
                    meta={"reason": "no mark for symbol"},
                    produced_by_engine="execution_engine",
                ),
            )

        if event.side is Side.HOLD:
            return (self._adapter.submit(event, mark),)

        return (self._adapter.submit(event, mark),)

    # ------------------------------------------------------------------
    # P0-3 — closed learning loop. Pure dispatch helper. Pushes terminal
    # ExecutionEvents into the configured FeedbackCollector and
    # IntelligenceFeedbackSink. Empty when neither sink is wired so
    # offline harness flows are unchanged.
    # ------------------------------------------------------------------

    def _feed_learning_loop(
        self,
        signal: SignalEvent,
        events: Sequence[ExecutionEvent],
    ) -> None:
        if (
            self._feedback_collector is None
            and self._intelligence_feedback is None
        ):
            return
        mark = self._marks.get(signal.symbol)
        strategy_id = signal.plugin_chain[0] if signal.plugin_chain else ""
        for event in events:
            if self._feedback_collector is not None and strategy_id:
                self._feedback_collector.record(
                    ts_ns=event.ts_ns,
                    strategy_id=strategy_id,
                    symbol=event.symbol,
                    qty=event.qty,
                    pnl=self._realised_pnl(
                        side=event.side,
                        qty=event.qty,
                        entry_price=event.price,
                        mark_price=mark,
                    ),
                    status=event.status,
                    venue=event.venue,
                    order_id=event.order_id,
                    meta=event.meta,
                )
            if self._intelligence_feedback is not None:
                self._intelligence_feedback.record(
                    signal=signal,
                    execution=event,
                    mark_price=mark,
                )

    @staticmethod
    def _realised_pnl(
        *,
        side: Side,
        qty: float,
        entry_price: float,
        mark_price: float | None,
    ) -> float:
        if mark_price is None or qty == 0.0 or entry_price == 0.0:
            return 0.0
        if side is Side.BUY:
            return qty * (mark_price - entry_price)
        if side is Side.SELL:
            return qty * (entry_price - mark_price)
        return 0.0

    def check_self(self) -> HealthStatus:
        plugin_states = {
            slot: {p.name: HealthState.OK for p in plugins}
            for slot, plugins in self.plugin_slots.items()
        }
        return HealthStatus(
            state=HealthState.OK,
            detail=f"Phase E1 — adapter={self._adapter.name}",
            plugin_states=plugin_states,
        )
