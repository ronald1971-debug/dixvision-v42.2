"""learning_engine.loops — closed learning loop drivers (P0-A).

Closed loop: ``FeedbackCollector`` → ``SlowLoopLearner`` → ``UpdateEmitter``
under the live :class:`LearningEvolutionFreezePolicy` supplied by the
runtime context. The loop is the **outer gate** — when the live policy
is frozen (any non-``LIVE`` mode or ``LIVE`` without operator override),
the loop drains and discards inputs without invoking any inner
component. The inner ``SlowLoopLearner`` / ``UpdateEmitter`` are
constructed with ``freeze_policy=None`` because the loop already enforces
the gate. AST tests pin the inner-component-freeze=None invariant + that
the loop never mutates the SystemMode.

INV-15 byte-identical replay: ``tick(ts_ns)`` is a pure function of
``(ts_ns, drained outcomes, policy snapshot, learner internal state)``.
No clocks, no PRNG (jitter disabled by default at this seam).
"""

from learning_engine.loops.closed_loop import (
    ClosedLearningLoop,
    LoopTickResult,
    SampleBuilder,
    UpdateBuilder,
)

__all__ = [
    "ClosedLearningLoop",
    "LoopTickResult",
    "SampleBuilder",
    "UpdateBuilder",
]
