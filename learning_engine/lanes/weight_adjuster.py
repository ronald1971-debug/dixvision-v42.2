"""Closed-loop weight adjuster — Trade Result → Score → Adjust Weights.

This is the third leg of the closed learning loop. Pieces 1 + 2 already
existed in the repo:

* **Trade Result.** :class:`core.contracts.learning.TradeOutcome` —
  emitted by ``execution_engine.protections.feedback`` once an order
  reaches a terminal status.
* **Score.** :class:`learning_engine.lanes.reward_shaping.RewardBreakdown`
  — pure J3-aligned attribution of the realised PnL into named
  components, ledgered as ``REWARD_BREAKDOWN``.

What was missing was the third piece — feeding the score back into the
runtime's weights. This module closes that loop:

* read a window of ledgered ``RewardBreakdown`` records,
* compute, per *tracked weight* (e.g. ``consensus_weight``,
  ``strength_weight``, ``coverage_weight`` of the confidence engine),
  the Pearson correlation between the component's contribution and
  ``shaped_reward``,
* propose bounded nudges to the weights as
  :class:`core.contracts.learning.LearningUpdate` records,
* hand them to :class:`learning_engine.update_emitter.UpdateEmitter`
  which materialises them as ``SystemEvent(UPDATE_PROPOSED)`` on the
  bus.

Authority + safety constraints:

* **L2.** Lives in the offline ``learning_engine``. Imports only
  :mod:`core.contracts` and the standard library — no runtime engine.
* **INV-15.** Pure / deterministic / no clock / no PRNG / no I/O.
  Every ``LearningUpdate`` carries a caller-supplied ``ts_ns``.
* **INV-47.** Operates on the ``RewardBreakdown`` records only. The
  raw PnL is preserved upstream; this module never recomputes it.
* **SAFE-65 (new in this delta).** Per-step nudges are bounded by
  ``max_nudge_per_step``; absolute weights are clipped into
  ``[min_weight, max_weight]``. The loop cannot drive a weight to
  zero (or past the cap) in a single step.
* **SAFE-66 (new in this delta).** The output is a *proposal*, not a
  mutation. Nothing in this module touches the runtime config.
  Governance promotes a proposal via the existing
  ``UPDATE_PROPOSED`` → ``UPDATE_APPLIED`` ratchet.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from core.contracts.learning import LearningUpdate
from learning_engine.lanes.reward_shaping import RewardBreakdown

WEIGHT_ADJUSTER_VERSION = "v3.6-P2"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeightAdjustmentConfig:
    """Bounds and gain on the weight-adjuster's per-window output.

    Attributes:
        learning_rate:
            Scales the per-window nudge size. The actual nudge for a
            given weight is
            ``sign(corr) · learning_rate · |corr|``,
            clipped further into ``[-max_nudge_per_step,
            +max_nudge_per_step]``. ``learning_rate`` is therefore an
            *upper bound* on the nudge for a perfectly correlated
            window, before the per-step clip is applied.
        max_nudge_per_step:
            Hard absolute cap on a single update's delta. Even with a
            large ``learning_rate`` and a perfectly correlated window,
            no single update can exceed this magnitude.
        min_weight:
            Floor on the post-update weight. Clipping happens after the
            nudge.
        max_weight:
            Ceiling on the post-update weight.
        min_samples:
            Below this many breakdowns the adjuster returns nothing for
            that weight (statistically meaningless to compute Pearson).
        correlation_floor:
            Magnitude floor on Pearson |r|. Below this, no update is
            proposed for that weight (treats the window as
            uninformative noise).
        version:
            Tag stamped on every ``LearningUpdate.meta`` so calibrators
            can attribute drift to the adjuster version that produced
            it.
    """

    learning_rate: float
    max_nudge_per_step: float
    min_weight: float
    max_weight: float
    min_samples: int
    correlation_floor: float
    version: str = WEIGHT_ADJUSTER_VERSION

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError(
                f"learning_rate must be a positive finite number, "
                f"got {self.learning_rate}"
            )
        if (
            self.max_nudge_per_step <= 0
            or not math.isfinite(self.max_nudge_per_step)
        ):
            raise ValueError(
                f"max_nudge_per_step must be a positive finite number, "
                f"got {self.max_nudge_per_step}"
            )
        if self.min_weight < 0 or not math.isfinite(self.min_weight):
            raise ValueError(
                f"min_weight must be a non-negative finite number, "
                f"got {self.min_weight}"
            )
        if not math.isfinite(self.max_weight) or self.max_weight <= self.min_weight:
            raise ValueError(
                f"max_weight must be > min_weight, got "
                f"min={self.min_weight}, max={self.max_weight}"
            )
        if self.min_samples < 2:
            raise ValueError(
                f"min_samples must be >= 2 (Pearson is undefined for "
                f"fewer samples), got {self.min_samples}"
            )
        if not (0.0 <= self.correlation_floor <= 1.0):
            raise ValueError(
                f"correlation_floor must lie in [0, 1], got "
                f"{self.correlation_floor}"
            )


# ---------------------------------------------------------------------------
# Per-weight binding
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeightBinding:
    """A single weight the adjuster knows how to nudge.

    Attributes:
        parameter:
            Free-form parameter id matching the runtime's config key
            (e.g. ``"consensus_weight"``). Forwarded as
            ``LearningUpdate.parameter``.
        component_name:
            Name of the breakdown component whose contribution is
            correlated against ``shaped_reward`` (e.g.
            ``"confidence_consensus"``). Must appear on every
            breakdown that participates in the window — breakdowns
            missing the component are skipped.
        current_value:
            The runtime's current weight. The proposed ``new_value`` is
            ``clip(current_value + nudge, min_weight, max_weight)``.
        strategy_id:
            Forwarded as ``LearningUpdate.strategy_id``. Identifies the
            owner of the parameter on the runtime side. Required by
            the existing ``UpdateEmitter`` validation.
    """

    parameter: str
    component_name: str
    current_value: float
    strategy_id: str

    def __post_init__(self) -> None:
        if not self.parameter:
            raise ValueError("WeightBinding.parameter must be non-empty")
        if not self.component_name:
            raise ValueError("WeightBinding.component_name must be non-empty")
        if not self.strategy_id:
            raise ValueError("WeightBinding.strategy_id must be non-empty")
        if not math.isfinite(self.current_value):
            raise ValueError(
                f"WeightBinding.current_value must be finite, got "
                f"{self.current_value}"
            )


# ---------------------------------------------------------------------------
# Pure stats — Pearson correlation
# ---------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation of two equal-length sequences.

    Returns ``None`` when either side has zero variance (the correlation
    is undefined and the adjuster must treat the window as
    uninformative).
    """

    n = len(xs)
    if n != len(ys) or n < 2:
        return None
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(ys) / n
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    syy = math.fsum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    denom = math.sqrt(sxx * syy)
    if denom <= 0.0:
        return None
    r = sxy / denom
    # Clamp into [-1, +1] to absorb any final rounding noise.
    if r > 1.0:
        return 1.0
    if r < -1.0:
        return -1.0
    return r


def _component_value(breakdown: RewardBreakdown, name: str) -> float | None:
    """Return the component contribution if present, else ``None``."""

    for cname, cval in breakdown.components:
        if cname == name:
            return cval
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeightAdjustment:
    """Per-weight diagnostic record for one window evaluation.

    Returned alongside the ``LearningUpdate`` so callers (and tests)
    can inspect *why* a particular nudge was proposed without having
    to re-run the math. Replay-stable.
    """

    parameter: str
    component_name: str
    sample_count: int
    correlation: float | None  # None when undefined / under-sampled
    raw_nudge: float           # before clipping
    clipped_nudge: float       # after [-max_nudge, +max_nudge] clamp
    old_value: float
    new_value: float           # after [min_weight, max_weight] clamp
    proposed: bool             # True iff a LearningUpdate was emitted


def propose_weight_updates(
    *,
    ts_ns: int,
    breakdowns: Sequence[RewardBreakdown],
    bindings: Sequence[WeightBinding],
    config: WeightAdjustmentConfig,
) -> tuple[tuple[LearningUpdate, ...], tuple[WeightAdjustment, ...]]:
    """Propose bounded weight nudges from a window of reward breakdowns.

    For each ``WeightBinding`` whose component is present on at least
    ``config.min_samples`` breakdowns and whose Pearson correlation
    with ``shaped_reward`` exceeds ``config.correlation_floor`` in
    magnitude, one ``LearningUpdate`` is produced. The proposed
    ``new_value`` is

    ``clip(current + clip(learning_rate · r, ±max_nudge), [min, max])``

    Returns a pair ``(updates, diagnostics)`` where ``diagnostics`` has
    one row per binding (whether or not an update was proposed). Both
    tuples are ordered by binding-input order so replay is stable.

    Raises:
        ValueError: if ``ts_ns`` is negative, or if any binding's
        ``current_value`` is already outside ``[min_weight,
        max_weight]`` (caller is expected to keep weights inside the
        envelope; rejecting fail-fast prevents silent saturation).
    """

    if ts_ns < 0:
        raise ValueError(f"ts_ns must be >= 0, got {ts_ns}")

    diagnostics: list[WeightAdjustment] = []
    updates: list[LearningUpdate] = []

    for binding in bindings:
        if not (config.min_weight <= binding.current_value <= config.max_weight):
            raise ValueError(
                f"WeightBinding({binding.parameter}).current_value "
                f"{binding.current_value} is outside the configured "
                f"envelope [{config.min_weight}, {config.max_weight}]"
            )

        xs: list[float] = []
        ys: list[float] = []
        for b in breakdowns:
            v = _component_value(b, binding.component_name)
            if v is None:
                continue
            xs.append(v)
            ys.append(b.shaped_reward)

        sample_count = len(xs)
        r = (
            _pearson(xs, ys)
            if sample_count >= config.min_samples
            else None
        )

        if r is None or abs(r) < config.correlation_floor:
            diagnostics.append(
                WeightAdjustment(
                    parameter=binding.parameter,
                    component_name=binding.component_name,
                    sample_count=sample_count,
                    correlation=r,
                    raw_nudge=0.0,
                    clipped_nudge=0.0,
                    old_value=binding.current_value,
                    new_value=binding.current_value,
                    proposed=False,
                )
            )
            continue

        raw_nudge = config.learning_rate * r
        clipped_nudge = max(
            -config.max_nudge_per_step,
            min(config.max_nudge_per_step, raw_nudge),
        )
        candidate = binding.current_value + clipped_nudge
        new_value = max(config.min_weight, min(config.max_weight, candidate))

        # If the clip pinned the new value exactly to the old value
        # (e.g. already at the ceiling and the nudge is positive), no
        # actual change happens — emit a diagnostic but no update.
        if new_value == binding.current_value:
            diagnostics.append(
                WeightAdjustment(
                    parameter=binding.parameter,
                    component_name=binding.component_name,
                    sample_count=sample_count,
                    correlation=r,
                    raw_nudge=raw_nudge,
                    clipped_nudge=clipped_nudge,
                    old_value=binding.current_value,
                    new_value=new_value,
                    proposed=False,
                )
            )
            continue

        diagnostics.append(
            WeightAdjustment(
                parameter=binding.parameter,
                component_name=binding.component_name,
                sample_count=sample_count,
                correlation=r,
                raw_nudge=raw_nudge,
                clipped_nudge=clipped_nudge,
                old_value=binding.current_value,
                new_value=new_value,
                proposed=True,
            )
        )
        meta: Mapping[str, str] = {
            "adjuster_version": config.version,
            "component": binding.component_name,
            "correlation": f"{r:.6f}",
            "sample_count": str(sample_count),
            "raw_nudge": f"{raw_nudge:.6f}",
            "clipped_nudge": f"{clipped_nudge:.6f}",
        }
        updates.append(
            LearningUpdate(
                ts_ns=ts_ns,
                strategy_id=binding.strategy_id,
                parameter=binding.parameter,
                old_value=f"{binding.current_value:.6f}",
                new_value=f"{new_value:.6f}",
                reason=(
                    f"weight_adjuster correlation={r:+.4f} over "
                    f"{sample_count} breakdowns"
                ),
                meta=meta,
            )
        )

    return tuple(updates), tuple(diagnostics)


__all__ = [
    "WEIGHT_ADJUSTER_VERSION",
    "WeightAdjustment",
    "WeightAdjustmentConfig",
    "WeightBinding",
    "propose_weight_updates",
]
