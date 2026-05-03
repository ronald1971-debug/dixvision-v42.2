"""Slow-loop continuous learner — feedback → parameter evolution (D4).

The learner accumulates :class:`FeedbackSample` rows (one per closed
trade or per replayed window), applies a windowed exponential moving
average to extract a smoothed reward gradient, and proposes bounded
adjustments to a small set of free parameters. Output is a
:class:`ParameterSnapshot` that downstream code can persist or feed
back into the meta-controller.

Determinism contract:

* All time-of-tick reads go through ``time_unix_s_provider`` (defaults
  to a constant lambda — callers MUST inject their TimeAuthority).
* The sole stochastic axis is the parameter "exploration jitter",
  drawn from a caller-injected :class:`random.Random` seeded by the
  caller. Replays construct a fresh ``Random(seed)`` and the snapshot
  is byte-identical.
* The learner respects :class:`LearningEvolutionFreezePolicy`. While
  frozen, ``tick()`` returns the previous snapshot unchanged and the
  internal feedback buffer drains without effect.

Bounds:

* Every tracked parameter is paired with a :class:`ParameterBounds`
  ``(lo, hi, step)``. Proposals are clamped to ``[lo, hi]``; ``step``
  caps the magnitude of any single adjustment so a single noisy
  feedback batch cannot swing the parameter end-to-end.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from core.contracts.learning_evolution_freeze import (
    LearningEvolutionFreezePolicy,
    is_unfrozen,
)

#: Bumped on any change to the smoothing / proposal algorithm.
SLOW_LOOP_VERSION = "v1"

#: Reasonable default smoothing factor for the EMA (0 < alpha <= 1).
DEFAULT_EMA_ALPHA = 0.2

#: Default upper bound on samples retained per parameter; older samples
#: are discarded FIFO so the working set is bounded.
DEFAULT_MAX_SAMPLES = 512


@dataclass(frozen=True, slots=True)
class FeedbackSample:
    """One reward observation feeding the learner.

    Attributes:
        ts_unix_s: Wall time of the observation (caller-supplied).
        parameter: Name of the parameter the sample is associated with.
        reward: Real-valued reward in any range. The learner only
            looks at the *gradient* of the EMA, so absolute scaling
            doesn't matter, but consistency across samples does.
        weight: Optional positive scalar weight (default ``1.0``).
            Heavier samples pull the EMA harder.
    """

    ts_unix_s: int
    parameter: str
    reward: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight <= 0.0 or not math.isfinite(self.weight):
            raise ValueError(
                f"FeedbackSample.weight must be > 0 and finite, "
                f"got {self.weight!r}"
            )
        if not math.isfinite(self.reward):
            raise ValueError(
                f"FeedbackSample.reward must be finite, "
                f"got {self.reward!r}"
            )


@dataclass(frozen=True, slots=True)
class ParameterBounds:
    """Bounds for one tracked parameter.

    Attributes:
        lo: Hard floor.
        hi: Hard ceiling. Must satisfy ``hi > lo``.
        step: Maximum magnitude of one update. Must be ``> 0``.
        initial: Starting value. Must satisfy ``lo <= initial <= hi``.
    """

    lo: float
    hi: float
    step: float
    initial: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError("ParameterBounds.lo / hi must be finite")
        if self.hi <= self.lo:
            raise ValueError(
                f"ParameterBounds.hi ({self.hi}) must be > lo ({self.lo})"
            )
        if self.step <= 0.0 or not math.isfinite(self.step):
            raise ValueError(
                f"ParameterBounds.step must be > 0, got {self.step}"
            )
        if not (self.lo <= self.initial <= self.hi):
            raise ValueError(
                f"ParameterBounds.initial ({self.initial}) outside "
                f"[{self.lo}, {self.hi}]"
            )


@dataclass(frozen=True, slots=True)
class ParameterSnapshot:
    """One frozen view of every tracked parameter after a tick.

    Attributes:
        ts_unix_s: ``time_unix_s_provider()`` at the moment of the tick.
        version: :data:`SLOW_LOOP_VERSION` at construction time.
        values: Parameter name → current value (post-update, clamped).
        ema: Parameter name → most recent reward EMA.
        sample_counts: Parameter name → cumulative samples ingested.
        frozen: ``True`` if the freeze policy blocked updates this tick.
    """

    ts_unix_s: int
    version: str
    values: Mapping[str, float]
    ema: Mapping[str, float]
    sample_counts: Mapping[str, int]
    frozen: bool


@dataclass(slots=True)
class _ParamState:
    bounds: ParameterBounds
    value: float
    ema: float = 0.0
    weight_sum: float = 0.0
    samples_seen: int = 0
    pending: list[FeedbackSample] = field(default_factory=list)


class SlowLoopLearner:
    """Bounded, freeze-aware parameter evolver.

    Args:
        bounds: Mapping of ``parameter_name`` → :class:`ParameterBounds`.
            The set of tracked parameters is fixed at construction.
        time_unix_s_provider: Injectable wall clock. The default is
            ``lambda: 0`` so unit tests can construct a learner without
            wiring a real clock; production wiring MUST inject the
            TimeAuthority.
        rng: Injectable :class:`random.Random` for exploration jitter.
            Defaults to ``random.Random(0)``.
        freeze_policy: Optional freeze policy. ``None`` means "always
            allowed" (migration window default).
        ema_alpha: Smoothing factor for the reward EMA.
        max_samples_per_param: FIFO cap on retained samples.
        exploration_eps: Std-dev of the exploration jitter, expressed
            as a fraction of ``ParameterBounds.step``. ``0.0`` disables
            jitter entirely (fully deterministic gradient descent).
    """

    def __init__(
        self,
        bounds: Mapping[str, ParameterBounds],
        *,
        time_unix_s_provider: Callable[[], int] | None = None,
        rng: random.Random | None = None,
        freeze_policy: LearningEvolutionFreezePolicy | None = None,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        max_samples_per_param: int = DEFAULT_MAX_SAMPLES,
        exploration_eps: float = 0.0,
    ) -> None:
        if not bounds:
            raise ValueError("bounds must be non-empty")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1], got {ema_alpha}"
            )
        if max_samples_per_param <= 0:
            raise ValueError(
                f"max_samples_per_param must be > 0, "
                f"got {max_samples_per_param}"
            )
        if exploration_eps < 0.0:
            raise ValueError(
                f"exploration_eps must be >= 0, got {exploration_eps}"
            )
        self._time_unix_s = (
            time_unix_s_provider
            if time_unix_s_provider is not None
            else (lambda: 0)
        )
        self._rng = rng if rng is not None else random.Random(0)
        self._freeze = freeze_policy
        self._alpha = ema_alpha
        self._max_samples = max_samples_per_param
        self._eps = exploration_eps
        self._params: dict[str, _ParamState] = {
            name: _ParamState(bounds=b, value=b.initial)
            for name, b in bounds.items()
        }

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(self._params.keys())

    def submit(self, sample: FeedbackSample) -> bool:
        """Buffer one feedback sample. Returns ``True`` if accepted."""

        st = self._params.get(sample.parameter)
        if st is None:
            return False
        st.pending.append(sample)
        if len(st.pending) > self._max_samples:
            st.pending.pop(0)
        return True

    def submit_many(self, samples: list[FeedbackSample]) -> int:
        accepted = 0
        for s in samples:
            if self.submit(s):
                accepted += 1
        return accepted

    def tick(self) -> ParameterSnapshot:
        """Drain pending samples, fold into EMA, propose bounded updates."""

        ts = int(self._time_unix_s())
        unfrozen = is_unfrozen(self._freeze)
        for st in self._params.values():
            if not st.pending:
                continue
            for s in st.pending:
                # Weighted EMA — equivalent to repeating the sample
                # ``weight`` times for fractional weights too.
                eff_alpha = 1.0 - (1.0 - self._alpha) ** s.weight
                st.ema = st.ema + eff_alpha * (s.reward - st.ema)
                st.weight_sum += s.weight
                st.samples_seen += 1
            st.pending.clear()
            if not unfrozen:
                continue
            # Gradient sign: positive EMA → reward says we should push
            # the value upward toward ``hi``; negative → downward.
            direction = 1.0 if st.ema >= 0.0 else -1.0
            magnitude = min(abs(st.ema), st.bounds.step)
            jitter = 0.0
            if self._eps > 0.0:
                jitter = self._rng.gauss(0.0, self._eps * st.bounds.step)
            proposed = st.value + direction * magnitude + jitter
            st.value = max(st.bounds.lo, min(st.bounds.hi, proposed))
        return ParameterSnapshot(
            ts_unix_s=ts,
            version=SLOW_LOOP_VERSION,
            values={n: s.value for n, s in self._params.items()},
            ema={n: s.ema for n, s in self._params.items()},
            sample_counts={
                n: s.samples_seen for n, s in self._params.items()
            },
            frozen=not unfrozen,
        )

    def reset(self) -> None:
        for st in self._params.values():
            st.value = st.bounds.initial
            st.ema = 0.0
            st.weight_sum = 0.0
            st.samples_seen = 0
            st.pending.clear()


__all__ = [
    "DEFAULT_EMA_ALPHA",
    "DEFAULT_MAX_SAMPLES",
    "FeedbackSample",
    "ParameterBounds",
    "ParameterSnapshot",
    "SLOW_LOOP_VERSION",
    "SlowLoopLearner",
]
