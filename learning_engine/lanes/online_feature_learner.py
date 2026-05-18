# ADAPTED FROM: river/drift/adwin.py + river/linear_model/pa.py
"""S-07 — Online incremental feature learning lane (ADWIN drift + PA-I update).

This module adapts two well-known online-learning algorithms from the
`river` project (https://github.com/online-ml/river, BSD-3-Clause) into
a single deterministic, OFFLINE-tier learning lane behind the DIX
``LearningUpdate`` contract:

1. **ADWIN concept-drift detection** (Bifet & Gavaldà, 2007). A
   bucket-based variant of the Adaptive-Windowing algorithm: each new
   value extends the right-most window; on every step, every possible
   split is checked against the Hoeffding bound

   .. math::

      \\epsilon_{\\text{cut}} =
      \\sqrt{\\tfrac{1}{2m}\\ln\\!\\big(\\tfrac{2}{\\delta'}\\big)}
      + \\tfrac{2}{3m}\\ln\\!\\big(\\tfrac{2}{\\delta'}\\big)

   with :math:`m = (|W_0|\\,|W_1|)/(|W_0|+|W_1|)` (harmonic mean) and
   :math:`\\delta' = \\delta / |W|` (Bonferroni-style correction). When
   :math:`|\\bar{W_0}-\\bar{W_1}| > \\epsilon_{\\text{cut}}`, the older
   half is dropped and a :class:`DriftReport` is emitted.

2. **Passive-Aggressive online classifier** (Crammer et al., 2006), the
   PA-I variant:

   .. math::

      \\tau = \\min\\big(C,\\;\\ell_t / \\|x_t\\|^2\\big),
      \\quad
      w_{t+1} = w_t + \\tau\\, y_t\\, x_t,
      \\quad
      b_{t+1} = b_t + \\tau\\, y_t

   where :math:`\\ell_t = \\max(0,\\; 1 - y_t(w_t \\cdot x_t + b_t))`
   is the hinge loss.

DIX integration rules — verbatim from PART 1 + the S-07 spec:

* OFFLINE-tier only — never on the hot path. The lane is a pure
  reducer: ``state' = step(state, observation, config)``. No clock
  reads, no PRNG state, no global mutable state. Caller supplies
  ``ts_ns`` on every observation; the lane only forwards it.
* The state is **fully serializable** as a frozen dataclass of
  primitives (counts, floats, tuples). Callers can ledger a snapshot
  and replay exactly via :func:`load_state` /
  :func:`OnlineFeatureLearnerState.to_payload`.
* Drift detection produces a :class:`DriftReport` record. The lane
  itself does **not** write to the ledger; the wiring code lifts the
  record onto a :class:`SystemEvent` (same pattern as
  :mod:`learning_engine.lanes.reward_shaping`).
* Proposed parameter mutations are emitted as
  :class:`core.contracts.learning.LearningUpdate` records and **must**
  be routed through the governance approval queue — INV-12. The lane
  never applies a mutation directly.
* INV-15 (replay determinism): same observation sequence + same config
  + same seed → byte-identical state, byte-identical drift report,
  byte-identical proposed update. Tests pin this across three runs.

Public surface (see ``__all__``):

* :class:`OnlineLearnerConfig`, :class:`OnlineLearnerObservation`,
  :class:`OnlineLearnerState`, :class:`OnlineLearnerStepOutcome`
* :class:`Prediction`, :class:`DriftReport`
* :func:`make_initial_state` — deterministic zero-state factory
* :func:`predict` — pure, no state mutation
* :func:`step` — pure, returns ``StepOutcome`` carrying the new state
* :func:`build_drift_update` — wraps a :class:`DriftReport` into a
  :class:`core.contracts.learning.LearningUpdate` proposal
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Final

from core.contracts.learning import LearningUpdate

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
"""S-07 adapts the *algorithms* from river verbatim in pure Python.

We deliberately do **not** add ``river`` as a runtime dependency — the
ADWIN cut math and PA-I update are reproduced in this module so the
OFFLINE tier stays standalone (matches the S-02 / S-03 / S-04
adaptation pattern).
"""

ONLINE_LEARNER_VERSION: Final[str] = "s-07.v1"
"""Version tag carried on every state and proposed update."""

PA_VARIANT_PA: Final[str] = "PA"
PA_VARIANT_PA_I: Final[str] = "PA-I"
PA_VARIANT_PA_II: Final[str] = "PA-II"

_KNOWN_PA_VARIANTS: Final[frozenset[str]] = frozenset(
    {PA_VARIANT_PA, PA_VARIANT_PA_I, PA_VARIANT_PA_II}
)

VALID_LABELS: Final[tuple[int, int]] = (-1, 1)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class OnlineLearnerError(ValueError):
    """Base class for typed errors raised by this lane."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class OnlineLearnerConfig:
    """Configuration for the lane.

    Fields:
        dim: Dimensionality of the feature vector. Must match
            ``len(observation.x)`` on every step.
        adwin_delta: ADWIN confidence parameter. Smaller ``delta``
            means stricter Hoeffding bound → fewer false positives,
            slower drift response. River's default is ``0.002``.
        pa_C: Aggressiveness cap for PA-I / PA-II. Larger ``C`` allows
            larger step sizes per observation. River's default is
            ``1.0``.
        pa_variant: One of ``"PA"``, ``"PA-I"``, ``"PA-II"``.
        adwin_min_window: Minimum window length before any cut is
            even attempted. Prevents trivial early-stream splits.
        version: Tag carried on every emitted record.
    """

    dim: int
    adwin_delta: float = 0.002
    pa_C: float = 1.0
    pa_variant: str = PA_VARIANT_PA_I
    adwin_min_window: int = 10
    version: str = ONLINE_LEARNER_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.dim, int) or isinstance(self.dim, bool):
            raise TypeError(f"dim must be int, got {type(self.dim).__name__}")
        if self.dim <= 0:
            raise OnlineLearnerError(f"dim must be > 0, got {self.dim}")

        if not isinstance(self.adwin_delta, float):
            raise TypeError(f"adwin_delta must be float, got {type(self.adwin_delta).__name__}")
        if not (0.0 < self.adwin_delta < 1.0):
            raise OnlineLearnerError(f"adwin_delta must be in (0, 1), got {self.adwin_delta}")

        if not isinstance(self.pa_C, float):
            raise TypeError(f"pa_C must be float, got {type(self.pa_C).__name__}")
        if not (self.pa_C > 0.0):
            raise OnlineLearnerError(f"pa_C must be > 0 (non-NaN), got {self.pa_C}")

        if self.pa_variant not in _KNOWN_PA_VARIANTS:
            raise OnlineLearnerError(
                f"pa_variant must be one of {sorted(_KNOWN_PA_VARIANTS)}, got {self.pa_variant!r}"
            )

        if not isinstance(self.adwin_min_window, int) or isinstance(self.adwin_min_window, bool):
            raise TypeError(
                f"adwin_min_window must be int, got {type(self.adwin_min_window).__name__}"
            )
        if self.adwin_min_window < 2:
            raise OnlineLearnerError(f"adwin_min_window must be >= 2, got {self.adwin_min_window}")

        if not isinstance(self.version, str) or not self.version:
            raise OnlineLearnerError("version must be non-empty str")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class OnlineLearnerObservation:
    """A single labelled observation fed to :func:`step`.

    Fields:
        ts_ns: Caller-supplied timestamp; forwarded into
            :class:`DriftReport` / :class:`LearningUpdate` records but
            never read as a clock by the lane.
        learner_id: Stable identifier (e.g. ``"alpha-momentum-v3"``)
            so a single ledger can host many independent lanes.
        x: Feature vector; ``len(x)`` must equal ``config.dim``.
        y: Binary label in :data:`VALID_LABELS` (``-1`` / ``+1``).
    """

    ts_ns: int
    learner_id: str
    x: tuple[float, ...]
    y: int

    def __post_init__(self) -> None:
        if not isinstance(self.ts_ns, int) or isinstance(self.ts_ns, bool):
            raise TypeError(f"ts_ns must be int, got {type(self.ts_ns).__name__}")
        if self.ts_ns <= 0:
            raise OnlineLearnerError(f"ts_ns must be positive, got {self.ts_ns}")

        if not isinstance(self.learner_id, str) or not self.learner_id:
            raise OnlineLearnerError("learner_id must be non-empty str")

        if not isinstance(self.x, tuple):
            raise TypeError(f"x must be tuple[float, ...], got {type(self.x).__name__}")
        for i, v in enumerate(self.x):
            if not isinstance(v, float) or isinstance(v, bool):
                raise TypeError(f"x[{i}] must be float, got {type(v).__name__}")
            if math.isnan(v) or math.isinf(v):
                raise OnlineLearnerError(f"x[{i}] must be finite, got {v}")

        if not isinstance(self.y, int) or isinstance(self.y, bool):
            raise TypeError(f"y must be int, got {type(self.y).__name__}")
        if self.y not in VALID_LABELS:
            raise OnlineLearnerError(f"y must be in {VALID_LABELS}, got {self.y}")


@dataclasses.dataclass(frozen=True, slots=True)
class Prediction:
    """Output of :func:`predict`.

    ``score`` is the raw ``w·x + b``; ``label`` is ``+1`` if
    ``score >= 0`` else ``-1`` (sign convention matches river).
    """

    score: float
    label: int


@dataclasses.dataclass(frozen=True, slots=True)
class DriftReport:
    """Concept-drift trigger emitted by ADWIN.

    Fields:
        ts_ns: Observation timestamp at which the cut was detected.
        learner_id: Same as the originating observation.
        change_point: Number of observations *kept* in the new window
            (i.e. observations from index ``change_point`` onward
            survived the cut). Equal to ``|W_1|`` at cut time.
        magnitude: ``|mean(W_0) - mean(W_1)|`` at cut time.
        n_observations_before: Total ``|W|`` immediately before the cut.
        epsilon_cut: The Hoeffding bound at cut time (audit field).
    """

    ts_ns: int
    learner_id: str
    change_point: int
    magnitude: float
    n_observations_before: int
    epsilon_cut: float

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise OnlineLearnerError(f"ts_ns must be positive, got {self.ts_ns}")
        if not self.learner_id:
            raise OnlineLearnerError("learner_id must be non-empty")
        if self.change_point < 0:
            raise OnlineLearnerError("change_point must be >= 0")
        if self.n_observations_before <= 0:
            raise OnlineLearnerError("n_observations_before must be > 0")
        if self.magnitude < 0.0 or math.isnan(self.magnitude):
            raise OnlineLearnerError("magnitude must be finite and >= 0")
        if self.epsilon_cut < 0.0 or math.isnan(self.epsilon_cut):
            raise OnlineLearnerError("epsilon_cut must be finite and >= 0")


@dataclasses.dataclass(frozen=True, slots=True)
class OnlineLearnerState:
    """Fully serializable, replayable lane state.

    Fields:
        learner_id: Stable identifier for the lane instance.
        version: Schema tag (``s-07.v1``); checkpoint readers should
            refuse mismatched versions.
        weights: Tuple of model weights, ``len == config.dim``.
        bias: Scalar bias term.
        n_observations: Number of observations consumed since reset.
        adwin_window: Tuple of margin values
            (``y · (w·x + b)``) currently in the ADWIN window.
        last_drift_ts_ns: Timestamp of the most recent drift cut, or
            ``0`` if none.
    """

    learner_id: str
    version: str
    weights: tuple[float, ...]
    bias: float
    n_observations: int
    adwin_window: tuple[float, ...]
    last_drift_ts_ns: int

    def __post_init__(self) -> None:
        if not self.learner_id:
            raise OnlineLearnerError("learner_id must be non-empty")
        if not self.version:
            raise OnlineLearnerError("version must be non-empty")
        if not isinstance(self.weights, tuple):
            raise TypeError("weights must be tuple")
        for i, w in enumerate(self.weights):
            if not isinstance(w, float) or isinstance(w, bool):
                raise TypeError(f"weights[{i}] must be float")
            if math.isnan(w) or math.isinf(w):
                raise OnlineLearnerError(f"weights[{i}] must be finite")
        if not isinstance(self.bias, float) or isinstance(self.bias, bool):
            raise TypeError("bias must be float")
        if math.isnan(self.bias) or math.isinf(self.bias):
            raise OnlineLearnerError("bias must be finite")
        if self.n_observations < 0:
            raise OnlineLearnerError("n_observations must be >= 0")
        if not isinstance(self.adwin_window, tuple):
            raise TypeError("adwin_window must be tuple")
        for i, v in enumerate(self.adwin_window):
            if not isinstance(v, float) or isinstance(v, bool):
                raise TypeError(f"adwin_window[{i}] must be float")
            if math.isnan(v) or math.isinf(v):
                raise OnlineLearnerError(f"adwin_window[{i}] must be finite")
        if self.last_drift_ts_ns < 0:
            raise OnlineLearnerError("last_drift_ts_ns must be >= 0")

    def to_payload(self) -> dict[str, str]:
        """Project state to a stringified payload (ledger-friendly)."""
        return {
            "learner_id": self.learner_id,
            "version": self.version,
            "weights": ",".join(f"{w:.12g}" for w in self.weights),
            "bias": f"{self.bias:.12g}",
            "n_observations": str(self.n_observations),
            "adwin_window": ",".join(f"{v:.12g}" for v in self.adwin_window),
            "last_drift_ts_ns": str(self.last_drift_ts_ns),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class OnlineLearnerStepOutcome:
    """Output of :func:`step`.

    Fields:
        new_state: Post-update state.
        prediction: The pre-update prediction for ``observation.x``.
        drift_report: A :class:`DriftReport` if ADWIN cut on this
            step, else ``None``.
        proposed_update: A :class:`core.contracts.learning.LearningUpdate`
            for governance approval if drift was detected, else
            ``None``. The lane never applies it directly (INV-12).
    """

    new_state: OnlineLearnerState
    prediction: Prediction
    drift_report: DriftReport | None
    proposed_update: LearningUpdate | None


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def make_initial_state(*, learner_id: str, dim: int) -> OnlineLearnerState:
    """Construct the canonical zero state for a fresh learner."""
    if not isinstance(learner_id, str) or not learner_id:
        raise OnlineLearnerError("learner_id must be non-empty str")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"dim must be int, got {type(dim).__name__}")
    if dim <= 0:
        raise OnlineLearnerError(f"dim must be > 0, got {dim}")
    return OnlineLearnerState(
        learner_id=learner_id,
        version=ONLINE_LEARNER_VERSION,
        weights=tuple(0.0 for _ in range(dim)),
        bias=0.0,
        n_observations=0,
        adwin_window=(),
        last_drift_ts_ns=0,
    )


# ---------------------------------------------------------------------------
# Pure prediction
# ---------------------------------------------------------------------------


def predict(state: OnlineLearnerState, x: tuple[float, ...]) -> Prediction:
    """Return ``w·x + b``-based prediction; pure, no state mutation."""
    if len(x) != len(state.weights):
        raise OnlineLearnerError(
            f"x dim {len(x)} does not match state.weights dim {len(state.weights)}"
        )
    score = math.fsum(w * v for w, v in zip(state.weights, x, strict=True))
    score += state.bias
    label = 1 if score >= 0.0 else -1
    return Prediction(score=score, label=label)


# ---------------------------------------------------------------------------
# Passive-Aggressive update
# ---------------------------------------------------------------------------


def _hinge_loss(margin: float) -> float:
    """``max(0, 1 - margin)`` — the standard PA hinge."""
    return max(0.0, 1.0 - margin)


def _pa_tau(*, loss: float, x_norm_sq: float, C: float, variant: str) -> float:
    """Compute the PA step size τ for the given variant.

    Returns 0.0 if ``loss == 0`` (no update needed) or if
    ``x_norm_sq == 0`` for variants that would divide by zero.
    """
    if loss <= 0.0:
        return 0.0
    if variant == PA_VARIANT_PA:
        if x_norm_sq <= 0.0:
            return 0.0
        return loss / x_norm_sq
    if variant == PA_VARIANT_PA_I:
        if x_norm_sq <= 0.0:
            return 0.0
        return min(C, loss / x_norm_sq)
    if variant == PA_VARIANT_PA_II:
        # tau = loss / (||x||^2 + 1/(2C))
        denom = x_norm_sq + 1.0 / (2.0 * C)
        if denom <= 0.0:
            return 0.0
        return loss / denom
    raise OnlineLearnerError(f"unknown pa_variant: {variant!r}")


def _pa_update(
    weights: tuple[float, ...],
    bias: float,
    x: tuple[float, ...],
    y: int,
    config: OnlineLearnerConfig,
    *,
    margin: float,
) -> tuple[tuple[float, ...], float]:
    """Apply one PA step. Returns new (weights, bias)."""
    loss = _hinge_loss(margin)
    if loss == 0.0:
        return weights, bias
    x_norm_sq = math.fsum(v * v for v in x)
    tau = _pa_tau(loss=loss, x_norm_sq=x_norm_sq, C=config.pa_C, variant=config.pa_variant)
    if tau == 0.0:
        return weights, bias
    new_weights = tuple(w + tau * y * v for w, v in zip(weights, x, strict=True))
    new_bias = bias + tau * y
    return new_weights, new_bias


# ---------------------------------------------------------------------------
# ADWIN cut detection
# ---------------------------------------------------------------------------


def _epsilon_cut(*, n0: int, n1: int, delta_prime: float) -> float:
    """Hoeffding bound used by ADWIN to test a candidate split.

    Reproduces the bound from river's ADWIN: ::

        m = (n0 * n1) / (n0 + n1)             # harmonic mean
        eps = sqrt((1 / (2m)) * ln(2/delta')) + (2/3m) * ln(2/delta')
    """
    if n0 <= 0 or n1 <= 0:
        return math.inf
    m = (n0 * n1) / (n0 + n1)
    log_term = math.log(2.0 / delta_prime)
    return math.sqrt((1.0 / (2.0 * m)) * log_term) + (2.0 / (3.0 * m)) * log_term


def _try_adwin_cut(
    window: tuple[float, ...], delta: float, min_window: int
) -> tuple[int, float, float] | None:
    """Search for an ADWIN cut. Returns ``(split_index, magnitude,
    epsilon)`` or ``None``.

    Iterates split indices ``1 <= split < len(window)`` and returns
    the *first* (smallest split index → drops the most data) for
    which the Hoeffding bound is exceeded. River's reference
    implementation uses bucketed scans for performance; the math is
    identical and the smallest passing split is the canonical cut.
    """
    n = len(window)
    if n < min_window:
        return None
    delta_prime = delta / n
    # Precompute prefix sums for O(n) per split.
    prefix: list[float] = [0.0]
    running = 0.0
    for v in window:
        running += v
        prefix.append(running)
    total_sum = prefix[-1]
    for split in range(1, n):
        n0 = split
        n1 = n - split
        if n0 < 1 or n1 < 1:
            continue
        sum0 = prefix[split]
        sum1 = total_sum - sum0
        mean0 = sum0 / n0
        mean1 = sum1 / n1
        magnitude = abs(mean0 - mean1)
        eps = _epsilon_cut(n0=n0, n1=n1, delta_prime=delta_prime)
        if magnitude > eps:
            return split, magnitude, eps
    return None


# ---------------------------------------------------------------------------
# step
# ---------------------------------------------------------------------------


def step(
    state: OnlineLearnerState,
    observation: OnlineLearnerObservation,
    config: OnlineLearnerConfig,
) -> OnlineLearnerStepOutcome:
    """Consume one observation; return new state + side-effect records.

    Pipeline:

    1. Predict on the current state.
    2. Apply a PA-I (or PA / PA-II) update using the predicted margin.
    3. Append the *new* margin to the ADWIN window.
    4. Search for a Hoeffding-bound cut. If found:

       * drop the older half from the window,
       * stamp ``last_drift_ts_ns``,
       * emit a :class:`DriftReport` and a
         :class:`core.contracts.learning.LearningUpdate` proposal.

    Returns an :class:`OnlineLearnerStepOutcome` carrying the new
    state. The lane is pure: no clock reads, no global mutation.
    """
    if not isinstance(state, OnlineLearnerState):
        raise TypeError(f"state must be OnlineLearnerState, got {type(state).__name__}")
    if not isinstance(observation, OnlineLearnerObservation):
        raise TypeError(
            f"observation must be OnlineLearnerObservation, got {type(observation).__name__}"
        )
    if not isinstance(config, OnlineLearnerConfig):
        raise TypeError(f"config must be OnlineLearnerConfig, got {type(config).__name__}")

    if observation.learner_id != state.learner_id:
        raise OnlineLearnerError(
            f"observation.learner_id {observation.learner_id!r} does not "
            f"match state.learner_id {state.learner_id!r}"
        )
    if len(observation.x) != len(state.weights):
        raise OnlineLearnerError(
            f"observation.x dim {len(observation.x)} does not match "
            f"state.weights dim {len(state.weights)}"
        )
    if len(state.weights) != config.dim:
        raise OnlineLearnerError(
            f"state.weights dim {len(state.weights)} does not match config.dim {config.dim}"
        )

    pre_pred = predict(state, observation.x)
    margin = observation.y * pre_pred.score
    new_weights, new_bias = _pa_update(
        weights=state.weights,
        bias=state.bias,
        x=observation.x,
        y=observation.y,
        config=config,
        margin=margin,
    )

    # Re-score under the new model and append the post-update *raw
    # score* to the drift window. We track the unsigned score (not
    # ``y · score``) so the window registers a regime flip — when
    # the population label distribution swings, the model's score on
    # a fixed feature pattern shifts sign and ADWIN's mean test
    # fires. Tracking the signed-margin would be invariant to label
    # flips (the model adapts by inverting weights, leaving margin
    # near +1 in both regimes) and so would never see drift.
    post_score = math.fsum(w * v for w, v in zip(new_weights, observation.x, strict=True))
    post_score += new_bias
    appended_window = state.adwin_window + (post_score,)

    cut = _try_adwin_cut(
        window=appended_window,
        delta=config.adwin_delta,
        min_window=config.adwin_min_window,
    )

    drift_report: DriftReport | None = None
    proposed_update: LearningUpdate | None = None
    new_window = appended_window
    last_drift_ts_ns = state.last_drift_ts_ns

    if cut is not None:
        split, magnitude, eps = cut
        new_window = appended_window[split:]
        last_drift_ts_ns = observation.ts_ns
        drift_report = DriftReport(
            ts_ns=observation.ts_ns,
            learner_id=state.learner_id,
            change_point=len(new_window),
            magnitude=magnitude,
            n_observations_before=len(appended_window),
            epsilon_cut=eps,
        )
        proposed_update = build_drift_update(drift=drift_report, version=config.version)

    new_state = OnlineLearnerState(
        learner_id=state.learner_id,
        version=state.version,
        weights=new_weights,
        bias=new_bias,
        n_observations=state.n_observations + 1,
        adwin_window=new_window,
        last_drift_ts_ns=last_drift_ts_ns,
    )
    return OnlineLearnerStepOutcome(
        new_state=new_state,
        prediction=pre_pred,
        drift_report=drift_report,
        proposed_update=proposed_update,
    )


# ---------------------------------------------------------------------------
# LearningUpdate projection
# ---------------------------------------------------------------------------


def build_drift_update(*, drift: DriftReport, version: str) -> LearningUpdate:
    """Wrap a :class:`DriftReport` as a governance-bound proposal.

    The lane never applies a parameter mutation directly (INV-12). It
    proposes a *retrain* of the underlying learner; governance
    decides whether to enact it.
    """
    if not isinstance(drift, DriftReport):
        raise TypeError(f"drift must be DriftReport, got {type(drift).__name__}")
    meta: Mapping[str, str] = {
        "change_point": str(drift.change_point),
        "magnitude": f"{drift.magnitude:.12g}",
        "epsilon_cut": f"{drift.epsilon_cut:.12g}",
        "n_observations_before": str(drift.n_observations_before),
        "version": version,
    }
    return LearningUpdate(
        ts_ns=drift.ts_ns,
        strategy_id=drift.learner_id,
        parameter="weights",
        old_value="adwin_drift_pre",
        new_value="adwin_drift_post",
        reason="adwin_drift_detected",
        meta=meta,
    )


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def load_state(payload: Mapping[str, str]) -> OnlineLearnerState:
    """Inverse of :meth:`OnlineLearnerState.to_payload`.

    Rejects unknown / missing keys and version mismatch.
    """
    required = {
        "learner_id",
        "version",
        "weights",
        "bias",
        "n_observations",
        "adwin_window",
        "last_drift_ts_ns",
    }
    missing = required - payload.keys()
    if missing:
        raise OnlineLearnerError(f"load_state payload missing keys: {sorted(missing)}")
    extra = payload.keys() - required
    if extra:
        raise OnlineLearnerError(f"load_state payload has unknown keys: {sorted(extra)}")
    if payload["version"] != ONLINE_LEARNER_VERSION:
        raise OnlineLearnerError(
            f"checkpoint version {payload['version']!r} != current {ONLINE_LEARNER_VERSION!r}"
        )
    raw_weights = payload["weights"]
    weights = tuple(float(w) for w in raw_weights.split(",")) if raw_weights else ()
    raw_window = payload["adwin_window"]
    window = tuple(float(v) for v in raw_window.split(",")) if raw_window else ()
    return OnlineLearnerState(
        learner_id=payload["learner_id"],
        version=payload["version"],
        weights=weights,
        bias=float(payload["bias"]),
        n_observations=int(payload["n_observations"]),
        adwin_window=window,
        last_drift_ts_ns=int(payload["last_drift_ts_ns"]),
    )


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "ONLINE_LEARNER_VERSION",
    "PA_VARIANT_PA",
    "PA_VARIANT_PA_I",
    "PA_VARIANT_PA_II",
    "VALID_LABELS",
    "DriftReport",
    "OnlineLearnerConfig",
    "OnlineLearnerError",
    "OnlineLearnerObservation",
    "OnlineLearnerState",
    "OnlineLearnerStepOutcome",
    "Prediction",
    "build_drift_update",
    "load_state",
    "make_initial_state",
    "predict",
    "step",
]
