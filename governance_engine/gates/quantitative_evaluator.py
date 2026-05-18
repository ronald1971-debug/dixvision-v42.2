"""P0-B — quantitative promotion-gate evaluator.

This module gates the ``CANARY → APPROVED`` edge of the patch pipeline
FSM on a small, conservative set of quantitative thresholds drawn from
``docs/promotion_gates.yaml``:

* ``sharpe_ratio   >= sharpe_ratio_min``        (default 1.0)
* ``max_drawdown   <= max_drawdown_max``        (default 0.05 = 5%)
* ``samples        >= samples_min``             (default 200)
* ``|is - oos|     <= is_oos_divergence_max``   (default 0.5σ)

The evaluator is *pure*: same inputs → same verdict (INV-15). It never
constructs typed bus events, never mutates state, and never reads a
clock, PRNG, or environment variable. The
:class:`PatchApprovalBridge` consumes the verdict on the
``CANARY → APPROVED`` edge; production wiring is supplied by the
harness so unit tests can drop the bridge in standalone.

The thresholds are defensible *defaults*, deliberately mirrored against
the ``shadow_to_canary`` performance floors of ``promotion_gates.yaml``
so operators only need to remember one canonical set. They are *not*
meant to be accepted blindly — operators may supply tighter
:class:`QuantitativeThresholds` per patch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

#: Sentinel: rejection codes used in :class:`QuantitativeVerdict.rejection_codes`.
REJECTION_CODE_INSUFFICIENT_SAMPLES = "QUANT_INSUFFICIENT_SAMPLES"
REJECTION_CODE_SHARPE_BELOW_FLOOR = "QUANT_SHARPE_BELOW_FLOOR"
REJECTION_CODE_DRAWDOWN_EXCEEDS_CEILING = "QUANT_DRAWDOWN_EXCEEDS_CEILING"
REJECTION_CODE_IS_OOS_DIVERGENCE = "QUANT_IS_OOS_DIVERGENCE"


class QuantitativeVerdictKind(StrEnum):
    """Three-valued verdict over a patch's quantitative metrics."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True, slots=True)
class QuantitativeThresholds:
    """Configurable thresholds for the quantitative gate.

    All thresholds are inclusive (``>=`` / ``<=``). ``samples_min`` is a
    hard floor — fewer samples → ``INSUFFICIENT_DATA``, not a numeric
    rejection. ``is_oos_divergence_max`` is expressed in σ of the IS
    distribution (caller-supplied ``is_std``); when ``is_std == 0`` the
    raw absolute divergence is compared directly against the threshold.
    """

    sharpe_ratio_min: float = 1.0
    max_drawdown_max: float = 0.05
    samples_min: int = 200
    is_oos_divergence_max_sigma: float = 0.5

    def __post_init__(self) -> None:
        # Hard contract: thresholds must be sane. Caller errors here
        # surface immediately at wiring time, not silently weaken the
        # gate at evaluation time.
        if self.sharpe_ratio_min < 0.0:
            raise ValueError("sharpe_ratio_min must be >= 0.0")
        if not (0.0 <= self.max_drawdown_max <= 1.0):
            raise ValueError("max_drawdown_max must be in [0.0, 1.0]")
        if self.samples_min < 0:
            raise ValueError("samples_min must be >= 0")
        if self.is_oos_divergence_max_sigma < 0.0:
            raise ValueError("is_oos_divergence_max_sigma must be >= 0.0")


#: Default conservative thresholds, mirrored against
#: ``docs/promotion_gates.yaml`` ``shadow_to_canary.performance``.
DEFAULT_QUANTITATIVE_THRESHOLDS: QuantitativeThresholds = QuantitativeThresholds()


@dataclass(frozen=True, slots=True)
class QuantitativeMetrics:
    """Frozen view of a patch's quantitative evaluation snapshot.

    Pure value object — caller supplies the values from whatever
    offline backtester / replay runner produced them. The evaluator
    never derives these itself; it only compares them against the
    thresholds.

    Fields:

    * ``sharpe_ratio`` — annualised Sharpe over the evaluation window.
    * ``max_drawdown`` — peak-to-trough drawdown, fractional (e.g. 0.04
      means 4%).
    * ``samples`` — number of independent trade-level observations.
    * ``is_score`` — in-sample objective (typically annualised return).
    * ``oos_score`` — out-of-sample objective on the same scale.
    * ``is_std`` — standard deviation of the IS distribution (used to
      normalise the IS/OOS divergence into σ-units).
    """

    sharpe_ratio: float
    max_drawdown: float
    samples: int
    is_score: float
    oos_score: float
    is_std: float

    def __post_init__(self) -> None:
        if self.samples < 0:
            raise ValueError("samples must be >= 0")
        if self.max_drawdown < 0.0:
            raise ValueError("max_drawdown must be >= 0.0")
        if self.is_std < 0.0:
            raise ValueError("is_std must be >= 0.0")


@dataclass(frozen=True, slots=True)
class QuantitativeVerdict:
    """Frozen quantitative verdict.

    ``passed`` is convenient sugar for ``kind == APPROVED``.
    ``rejection_codes`` is order-stable and lexicographically sorted so
    serialisation is byte-identical across replays (INV-15).
    """

    kind: QuantitativeVerdictKind
    passed: bool
    rejection_codes: tuple[str, ...] = ()
    detail: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)


class QuantitativeEvaluator:
    """Pure evaluator over :class:`QuantitativeMetrics`.

    Stateless — the only configuration is the
    :class:`QuantitativeThresholds` passed at construction. Re-using a
    single evaluator across patches is therefore safe and idiomatic.
    """

    __slots__ = ("_thresholds",)

    def __init__(
        self,
        *,
        thresholds: QuantitativeThresholds = DEFAULT_QUANTITATIVE_THRESHOLDS,
    ) -> None:
        self._thresholds = thresholds

    @property
    def thresholds(self) -> QuantitativeThresholds:
        return self._thresholds

    # ------------------------------------------------------------------
    def evaluate(self, metrics: QuantitativeMetrics) -> QuantitativeVerdict:
        """Pure verdict over ``metrics`` w.r.t. the configured thresholds."""

        t = self._thresholds

        # Insufficient-data fast path. Returning INSUFFICIENT_DATA is
        # *not* an APPROVAL — the bridge treats it as a block — but it
        # is also not a numeric rejection. The distinction matters for
        # rollout policy: an under-sampled patch can be re-evaluated
        # once more samples accumulate; a numerically-rejected patch
        # cannot be revived without re-running the offline pipeline.
        if metrics.samples < t.samples_min:
            return QuantitativeVerdict(
                kind=QuantitativeVerdictKind.INSUFFICIENT_DATA,
                passed=False,
                rejection_codes=(REJECTION_CODE_INSUFFICIENT_SAMPLES,),
                detail=(f"samples={metrics.samples} < samples_min={t.samples_min}"),
            )

        codes: list[str] = []
        details: list[str] = []

        if metrics.sharpe_ratio < t.sharpe_ratio_min:
            codes.append(REJECTION_CODE_SHARPE_BELOW_FLOOR)
            details.append(
                f"sharpe_ratio={metrics.sharpe_ratio:.4f} < "
                f"sharpe_ratio_min={t.sharpe_ratio_min:.4f}"
            )

        if metrics.max_drawdown > t.max_drawdown_max:
            codes.append(REJECTION_CODE_DRAWDOWN_EXCEEDS_CEILING)
            details.append(
                f"max_drawdown={metrics.max_drawdown:.4f} > "
                f"max_drawdown_max={t.max_drawdown_max:.4f}"
            )

        divergence_abs = abs(metrics.is_score - metrics.oos_score)
        if metrics.is_std > 0.0:
            divergence_sigma = divergence_abs / metrics.is_std
            divergence_repr = f"{divergence_sigma:.4f}σ"
        else:
            # No IS dispersion — compare absolute divergence directly.
            divergence_sigma = divergence_abs
            divergence_repr = f"{divergence_sigma:.4f} (abs)"
        if divergence_sigma > t.is_oos_divergence_max_sigma:
            codes.append(REJECTION_CODE_IS_OOS_DIVERGENCE)
            details.append(
                f"|is-oos|={divergence_repr} > "
                f"is_oos_divergence_max={t.is_oos_divergence_max_sigma:.4f}σ"
            )

        if codes:
            return QuantitativeVerdict(
                kind=QuantitativeVerdictKind.REJECTED,
                passed=False,
                rejection_codes=tuple(sorted(codes)),
                detail="; ".join(details),
            )

        return QuantitativeVerdict(
            kind=QuantitativeVerdictKind.APPROVED,
            passed=True,
            rejection_codes=(),
            detail=(
                f"sharpe={metrics.sharpe_ratio:.4f} "
                f"max_dd={metrics.max_drawdown:.4f} "
                f"samples={metrics.samples} "
                f"|is-oos|={divergence_repr}"
            ),
        )


__all__ = [
    "DEFAULT_QUANTITATIVE_THRESHOLDS",
    "QuantitativeEvaluator",
    "QuantitativeMetrics",
    "QuantitativeThresholds",
    "QuantitativeVerdict",
    "QuantitativeVerdictKind",
    "REJECTION_CODE_DRAWDOWN_EXCEEDS_CEILING",
    "REJECTION_CODE_INSUFFICIENT_SAMPLES",
    "REJECTION_CODE_IS_OOS_DIVERGENCE",
    "REJECTION_CODE_SHARPE_BELOW_FLOOR",
]
