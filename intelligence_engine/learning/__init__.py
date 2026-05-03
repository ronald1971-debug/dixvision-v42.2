"""Slow-loop continuous learner (D4).

The slow-loop runs at a much lower frequency than the hot path
(MetaControllerHotPath) and is responsible for evolving free
parameters — confidence floors, news sentiment weights, regime
hysteresis bounds — based on accumulated feedback. It is gated by
:class:`core.contracts.learning_evolution_freeze.FreezeStatus` so a
hazardous mode (DEGRADED, FROZEN) cannot drift parameters.

Authority constraints:

* Pure / deterministic. Time + PRNG are caller-injected. Two replays
  of the same feedback sequence produce identical parameter snapshots.
* Bounded. The learner never proposes a parameter outside the
  ``ParameterBounds`` declared at construction. Out-of-bounds proposals
  are clamped and logged in the snapshot.
* Freeze-aware. While ``FreezeStatus.frozen`` is true, ``tick()``
  returns the previous snapshot unchanged and emits no updates.
"""

from intelligence_engine.learning.slow_loop import (
    SLOW_LOOP_VERSION,
    FeedbackSample,
    ParameterBounds,
    ParameterSnapshot,
    SlowLoopLearner,
)

__all__ = [
    "SLOW_LOOP_VERSION",
    "FeedbackSample",
    "ParameterBounds",
    "ParameterSnapshot",
    "SlowLoopLearner",
]
