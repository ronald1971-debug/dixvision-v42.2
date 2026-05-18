"""ClosedLearningLoop — P0-A closed-loop driver.

Wires :class:`FeedbackCollector` → :class:`SlowLoopLearner` →
:class:`UpdateEmitter` together under the live
:class:`LearningEvolutionFreezePolicy` supplied by the runtime.

Lifecycle on every :meth:`tick`:

1. Snapshot the live freeze policy via ``policy_supplier()``.
2. Drain the feedback collector (always, even when frozen — the queue
   must not back up indefinitely).
3. If frozen: return :class:`LoopTickResult` with ``frozen=True`` and
   ``drained_outcomes`` recorded so the operator can see the loop is
   doing its bookkeeping. No samples are submitted, no learner tick is
   driven, no updates are emitted.
4. If unfrozen: build :class:`FeedbackSample` rows from drained
   outcomes via the caller-supplied ``sample_builder``; submit them
   to the learner; call ``learner.tick()``; diff the new snapshot
   against the previous one via the caller-supplied ``update_builder``;
   emit each resulting :class:`LearningUpdate` via :class:`UpdateEmitter`.

The loop is the **single freeze-policy enforcement point** for this
chain. The inner ``SlowLoopLearner`` and ``UpdateEmitter`` are
constructed with ``freeze_policy=None`` / ``freeze=None`` because the
loop already gates every invocation. Tests pin this invariant.

Pure / deterministic: same ``(ts_ns, drained outcomes, policy, learner
state)`` → same :class:`LoopTickResult`. No clocks, no PRNG, no IO at
this seam (the learner may use a caller-supplied :class:`random.Random`,
but the loop never reads one).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from core.contracts.events import SystemEvent
from core.contracts.learning import LearningUpdate, TradeOutcome
from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
)
from learning_engine.update_emitter import UpdateEmitter

# B1/L2: ``learning_engine`` is an OFFLINE tier under
# ``authority_lint`` and MUST NOT import ``execution_engine`` /
# ``intelligence_engine`` (which are RUNTIME tiers) — not even under
# ``TYPE_CHECKING``, because the lint walks the full AST. The loop
# never constructs either type (instances are always caller-supplied
# from ``ui.server._State``), so we duck-type the two seams via local
# :class:`typing.Protocol` definitions and treat the sample / snapshot
# value objects as ``Any``. This keeps the offline-engine boundary
# intact under L2 while preserving the loop's runtime contract.
FeedbackSample = Any
ParameterSnapshot = Any


class FeedbackCollector(Protocol):
    """Duck-typed contract for the execution-tier feedback collector.

    The concrete production type is
    :class:`execution_engine.protections.feedback.FeedbackCollector`,
    but learning_engine cannot import it under L2. Any object whose
    ``drain()`` returns a tuple of :class:`TradeOutcome` is accepted.
    """

    def drain(self) -> tuple[TradeOutcome, ...]: ...


class SlowLoopLearner(Protocol):
    """Duck-typed contract for the intelligence-tier slow-loop learner.

    The concrete production type is
    :class:`intelligence_engine.learning.slow_loop.SlowLoopLearner`,
    but learning_engine cannot import it under L2. Any object that
    exposes :meth:`submit` and :meth:`tick` is accepted.
    """

    def submit(self, sample: Any) -> bool: ...

    def tick(self) -> Any: ...


#: Type alias: maps drained outcomes → feedback samples. Pure function.
SampleBuilder = Callable[[tuple[TradeOutcome, ...]], tuple[FeedbackSample, ...]]

#: Type alias: maps (previous snapshot, current snapshot, ts_ns) → updates.
#: Pure function. ``previous`` is ``None`` on the first tick.
UpdateBuilder = Callable[
    [ParameterSnapshot | None, ParameterSnapshot, int],
    tuple[LearningUpdate, ...],
]

#: Type alias: zero-arg supplier returning the live freeze policy. The
#: runtime closure typically reads ``SystemMode`` + the operator-override
#: flag under ``STATE.lock`` and constructs a fresh frozen policy each
#: call so the supplier itself is pure with respect to its callers.
FreezePolicySupplier = Callable[[], LearningEvolutionFreezePolicy]


def _empty_sample_builder(
    _outcomes: tuple[TradeOutcome, ...],
) -> tuple[FeedbackSample, ...]:
    return ()


def _empty_update_builder(
    _previous: ParameterSnapshot | None,
    _current: ParameterSnapshot,
    _ts_ns: int,
) -> tuple[LearningUpdate, ...]:
    return ()


@dataclass(frozen=True, slots=True)
class LoopTickResult:
    """Frozen summary of one :meth:`ClosedLearningLoop.tick` call.

    Attributes:
        ts_ns: Caller-supplied tick timestamp.
        frozen: ``True`` iff the live freeze policy refused the tick.
        drained_outcomes: Tuple of outcomes drained from the feedback
            collector on this tick (always populated — the collector is
            drained regardless of freeze state).
        submitted_samples: Tuple of samples actually submitted to the
            learner. Empty when frozen.
        snapshot: ``ParameterSnapshot`` returned by the learner after
            the tick. ``None`` when frozen.
        emitted_events: Tuple of :class:`SystemEvent` rows emitted by
            the :class:`UpdateEmitter`, one per :class:`LearningUpdate`.
            Empty when frozen.
        policy_mode_name: Snapshot of ``policy.mode.name`` at tick time
            (handy for tests and the operator audit trail).
        operator_override: Snapshot of ``policy.operator_override`` at
            tick time.
    """

    ts_ns: int
    frozen: bool
    drained_outcomes: tuple[TradeOutcome, ...]
    submitted_samples: tuple[FeedbackSample, ...]
    snapshot: ParameterSnapshot | None
    emitted_events: tuple[SystemEvent, ...]
    policy_mode_name: str
    operator_override: bool


class ClosedLearningLoop:
    """Deterministic closed-loop driver (P0-A).

    Args:
        feedback_collector: Source of :class:`TradeOutcome` rows. Drained
            every tick (queue must not back up).
        learner: Bounded :class:`SlowLoopLearner`. Must be constructed
            with ``freeze_policy=None`` — the loop owns the gate.
        emitter: :class:`UpdateEmitter`. Must be constructed with
            ``freeze=None`` — the loop owns the gate.
        policy_supplier: Zero-arg callable returning the live
            :class:`LearningEvolutionFreezePolicy`. The supplier is
            invoked exactly once per :meth:`tick` so the loop sees a
            consistent policy snapshot for the duration of the tick.
        sample_builder: Pure function mapping drained outcomes →
            feedback samples. Defaults to the no-op builder so this
            seam can be unit-tested without a real strategy mapping.
        update_builder: Pure function mapping ``(previous_snapshot,
            current_snapshot, ts_ns)`` → learning updates. Defaults to
            the no-op builder.

    Raises:
        ValueError: if ``learner`` or ``emitter`` was constructed with
            a non-``None`` freeze policy. The loop is the single gate
            and refuses to nest the contract.
    """

    name: str = "closed_learning_loop"
    spec_id: str = "P0-A"

    __slots__ = (
        "_feedback",
        "_learner",
        "_emitter",
        "_policy_supplier",
        "_sample_builder",
        "_update_builder",
        "_previous_snapshot",
    )

    def __init__(
        self,
        *,
        feedback_collector: FeedbackCollector,
        learner: SlowLoopLearner,
        emitter: UpdateEmitter,
        policy_supplier: FreezePolicySupplier,
        sample_builder: SampleBuilder | None = None,
        update_builder: UpdateBuilder | None = None,
    ) -> None:
        if getattr(learner, "_freeze", None) is not None:
            raise ValueError(
                "ClosedLearningLoop requires learner.freeze_policy=None "
                "(the loop is the single freeze gate)"
            )
        if getattr(emitter, "_freeze", None) is not None:
            raise ValueError(
                "ClosedLearningLoop requires emitter.freeze=None "
                "(the loop is the single freeze gate)"
            )
        self._feedback = feedback_collector
        self._learner = learner
        self._emitter = emitter
        self._policy_supplier = policy_supplier
        self._sample_builder = sample_builder or _empty_sample_builder
        self._update_builder = update_builder or _empty_update_builder
        self._previous_snapshot: ParameterSnapshot | None = None

    @property
    def previous_snapshot(self) -> ParameterSnapshot | None:
        """Latest snapshot kept for diff'ing on the next unfrozen tick."""
        return self._previous_snapshot

    def tick(self, *, ts_ns: int) -> LoopTickResult:
        """Drive one closed-loop tick.

        Args:
            ts_ns: Tick timestamp. Forwarded to every emitted
                :class:`LearningUpdate` and recorded on the result.
        """

        policy = self._policy_supplier()
        drained = self._feedback.drain()
        if policy.is_frozen():
            return LoopTickResult(
                ts_ns=ts_ns,
                frozen=True,
                drained_outcomes=drained,
                submitted_samples=(),
                snapshot=None,
                emitted_events=(),
                policy_mode_name=policy.mode.name,
                operator_override=policy.operator_override,
            )
        samples = tuple(self._sample_builder(drained))
        for s in samples:
            self._learner.submit(s)
        snapshot = self._learner.tick()
        updates = tuple(self._update_builder(self._previous_snapshot, snapshot, ts_ns))
        events = self._emitter.emit_many(updates)
        self._previous_snapshot = snapshot
        return LoopTickResult(
            ts_ns=ts_ns,
            frozen=False,
            drained_outcomes=drained,
            submitted_samples=samples,
            snapshot=snapshot,
            emitted_events=events,
            policy_mode_name=policy.mode.name,
            operator_override=policy.operator_override,
        )


__all__ = [
    "ClosedLearningLoop",
    "FreezePolicySupplier",
    "LoopTickResult",
    "SampleBuilder",
    "UpdateBuilder",
]
