# ADAPTED FROM: pola-rs/polars py-polars/polars/lazyframe/frame.py
# (LazyFrame.group_by / agg / sort / collect lazy-API pattern;
#  py-polars/polars/expr/expr.py — pl.col / pl.corr / pl.len / pl.col(...).rank
#  expression patterns; py-polars/polars/dataframe/frame.py — iter_rows(named=True)
#  materialisation.)
"""Polars LazyFrame-backed feature-importance scoring — OFFLINE batch tier.

This module is the **third and final** of the S-10 polars triple
(``pnl_attribution.py`` + ``regime_stats.py`` + ``feature_importance.py``).
It adapts polars' lazy ``group_by("feature_name") -> agg([pl.corr(...)
...]) -> sort -> collect`` pattern into DIX as a deterministic batch
feature-importance scorer over a stream of (feature, target) observations
collected from offline back-tests.

For each named feature the scorer computes:

* ``n_obs``                — number of observations in the bucket
* ``mean_feature``         — sample mean of ``feature_value``
* ``mean_target``          — sample mean of ``target_value``
* ``pearson_corr``         — Pearson correlation between the raw
  ``feature_value`` and ``target_value`` series
  (``E[xy] - E[x]E[y]) / (sigma_x * sigma_y)``)
* ``rank_corr``            — Spearman rank correlation, computed as the
  Pearson correlation of the **ranks** of ``feature_value`` and
  ``target_value`` (``method="min"`` ties to keep replays byte-identical
  even when ties exist)
* ``abs_score``            — primary importance score:
  ``max(|pearson_corr|, |rank_corr|)`` (range ``[0, 1]``)

The output is a frozen :class:`FeatureImportanceReport` with one
:class:`FeatureImportance` row per feature, sorted **descending** by
``abs_score`` with ``feature_name`` (ascending) as a deterministic
tiebreaker so replays are byte-identical (INV-15).

Tier
----
**OFFLINE_ONLY.** ``learning_engine/analytics/`` is the high-throughput
slow-cadence analytics tier. Polars must never be imported from
``execution_engine/``, ``governance_engine/``, ``system_engine/``,
``core/``, or ``intelligence_engine/meta_controller/hot_path.py`` —
that ban is enforced by the S-10.4 follow-up lint rule.

Design constraints
------------------
* **Lazy import.** ``import polars`` lives **inside**
  :func:`compute_feature_importance` so this module imports cleanly in
  environments without ``polars`` installed (mirrors S-10.1 and S-10.2).
* **Pure data.** No wall-clock reads, no IO, no global mutable state,
  no PRNG. Polars is asked for an **eager** ``collect()`` each call —
  there is no shared session, no streaming flag, and the resulting
  :class:`FeatureImportanceReport` is fully serialisable
  (frozen + slotted).
* **Frozen contracts.** :class:`FeatureObservation`,
  :class:`FeatureImportance`, and :class:`FeatureImportanceReport` are
  ``@dataclass(frozen=True, slots=True)`` with eager validation in
  ``__post_init__``.
* **INV-15 byte-identical.** Inputs are sorted by
  ``(feature_name, ts_ns, feature_value, target_value)`` *before* the
  LazyFrame is constructed; the group-by output is post-sorted by
  ``(-abs_score, feature_name)``; per-feature aggregation uses polars'
  associative reductions, ``rank(method="min")`` is deterministic, and
  determinism is guaranteed by the explicit ``sort`` + absence of any
  parallel/streaming flag at ``collect`` time. Pinned by a 3-run
  byte-equality + permutation-invariance test.
* **No new pip deps in module-import time.** :data:`NEW_PIP_DEPENDENCIES`
  declares ``("polars",)`` so the pip-dep audit picks it up, but the
  module body never imports polars at toplevel; calling
  :func:`compute_feature_importance` without polars installed raises a
  clean ``ImportError`` with an actionable hint.

Algorithmic summary
-------------------
For each :class:`FeatureObservation`::

    feature_rank = rank(feature_value, method="min")  # within feature_name
    target_rank  = rank(target_value,  method="min")  # within feature_name

Group-by ``feature_name`` then aggregate::

    n_obs         = pl.len()
    mean_feature  = pl.col("feature_value").mean()
    mean_target   = pl.col("target_value").mean()
    pearson_corr  = pl.corr("feature_value", "target_value")
    rank_corr     = pl.corr("feature_rank",  "target_rank")
    abs_score     = max(|pearson_corr|, |rank_corr|)

Single-observation buckets, zero-variance features, and zero-variance
targets all yield ``corr = NaN``; this module clamps NaN ↦ 0.0 so the
downstream :class:`FeatureImportance` invariant
``0 <= abs_score <= 1`` always holds and replays remain byte-identical.

Empty input returns an empty :class:`FeatureImportanceReport` (no
crashes, no NaN in totals).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

__all__ = (
    "FeatureImportance",
    "FeatureImportanceReport",
    "FeatureObservation",
    "NEW_PIP_DEPENDENCIES",
    "compute_feature_importance",
)

#: Pip dependencies introduced by this module. The pip-dep audit reads
#: this constant to flag new wheels in CI.
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("polars",)


# ---------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeatureObservation:
    """One ``(feature, target)`` observation drawn from an offline run.

    Parameters
    ----------
    ts_ns:
        Caller-supplied event timestamp in nanoseconds (never the wall
        clock). Used only for deterministic input ordering.
    feature_name:
        Identifier of the feature being scored (non-empty).
    feature_value:
        Raw feature value at ``ts_ns``. Must be a finite ``float``.
    target_value:
        Realised target value at the same ``ts_ns`` (e.g. forward
        return). Must be a finite ``float``.
    """

    ts_ns: int
    feature_name: str
    feature_value: float
    target_value: float

    def __post_init__(self) -> None:
        if isinstance(self.ts_ns, bool) or not isinstance(self.ts_ns, int):
            raise TypeError("ts_ns must be int")
        if self.ts_ns < 0:
            raise ValueError("ts_ns must be >= 0")
        if not isinstance(self.feature_name, str):
            raise TypeError("feature_name must be str")
        if not self.feature_name:
            raise ValueError("feature_name must be non-empty")
        for label, value in (
            ("feature_value", self.feature_value),
            ("target_value", self.target_value),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{label} must be float")
            if math.isnan(value):
                raise ValueError(f"{label} must not be NaN")
            if math.isinf(value):
                raise ValueError(f"{label} must be finite")


@dataclass(frozen=True, slots=True)
class FeatureImportance:
    """Per-feature aggregated importance row."""

    feature_name: str
    n_obs: int
    mean_feature: float
    mean_target: float
    pearson_corr: float
    rank_corr: float
    abs_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.feature_name, str) or not self.feature_name:
            raise ValueError("feature_name must be non-empty str")
        if isinstance(self.n_obs, bool) or not isinstance(self.n_obs, int):
            raise TypeError("n_obs must be int")
        if self.n_obs < 0:
            raise ValueError("n_obs must be >= 0")
        for label, value in (
            ("mean_feature", self.mean_feature),
            ("mean_target", self.mean_target),
            ("pearson_corr", self.pearson_corr),
            ("rank_corr", self.rank_corr),
            ("abs_score", self.abs_score),
        ):
            if not isinstance(value, float):
                raise TypeError(f"{label} must be float")
            if math.isnan(value):
                raise ValueError(f"{label} must not be NaN")
        for label, value in (
            ("pearson_corr", self.pearson_corr),
            ("rank_corr", self.rank_corr),
        ):
            if value < -1.0 or value > 1.0:
                raise ValueError(f"{label} must lie in [-1, 1]")
        if self.abs_score < 0.0 or self.abs_score > 1.0:
            raise ValueError("abs_score must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class FeatureImportanceReport:
    """Aggregate output of :func:`compute_feature_importance`.

    ``by_feature`` is sorted **descending** by ``abs_score`` with
    ``feature_name`` (ascending) as a deterministic tiebreaker. The
    aggregate totals (``total_n_obs``, ``mean_abs_score``) are computed
    via :func:`math.fsum` for byte-identical replays.
    """

    by_feature: tuple[FeatureImportance, ...] = field(default_factory=tuple)
    total_n_obs: int = 0
    mean_abs_score: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.by_feature, tuple):
            raise TypeError("by_feature must be tuple")
        names: list[str] = []
        for item in self.by_feature:
            if not isinstance(item, FeatureImportance):
                raise TypeError(
                    f"by_feature entries must be FeatureImportance; got {type(item).__name__}"
                )
            names.append(item.feature_name)
        if len(set(names)) != len(names):
            raise ValueError("by_feature feature_names must be unique")
        for prev, curr in zip(self.by_feature, self.by_feature[1:], strict=False):
            if curr.abs_score > prev.abs_score:
                raise ValueError("by_feature must be sorted descending by abs_score")
            if curr.abs_score == prev.abs_score and curr.feature_name < prev.feature_name:
                raise ValueError("ties in abs_score must break by feature_name ascending")
        if isinstance(self.total_n_obs, bool) or not isinstance(self.total_n_obs, int):
            raise TypeError("total_n_obs must be int")
        if self.total_n_obs < 0:
            raise ValueError("total_n_obs must be >= 0")
        if not isinstance(self.mean_abs_score, float):
            raise TypeError("mean_abs_score must be float")
        if math.isnan(self.mean_abs_score):
            raise ValueError("mean_abs_score must not be NaN")
        if self.mean_abs_score < 0.0 or self.mean_abs_score > 1.0:
            raise ValueError("mean_abs_score must lie in [0, 1]")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def compute_feature_importance(
    observations: Sequence[FeatureObservation],
) -> FeatureImportanceReport:
    """Compute per-feature Pearson + Spearman importance via polars.

    Parameters
    ----------
    observations:
        Sequence of :class:`FeatureObservation`. May be empty.

    Returns
    -------
    FeatureImportanceReport
        Frozen + slotted. ``by_feature`` is sorted by
        ``(-abs_score, feature_name)`` for INV-15 byte-stable replays.

    Raises
    ------
    ImportError:
        if ``polars`` is not installed (lazy import).
    TypeError:
        if ``observations`` contains a non-:class:`FeatureObservation`.
    """
    if not observations:
        return FeatureImportanceReport()

    for obs in observations:
        if not isinstance(obs, FeatureObservation):
            raise TypeError(
                f"observations entries must be FeatureObservation; got {type(obs).__name__}"
            )

    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - exercised in tests
        raise ImportError(
            "polars is required for "
            "learning_engine.analytics.feature_importance; install it with "
            "`pip install polars` (S-10 OFFLINE_ONLY tier)"
        ) from exc

    sorted_obs = sorted(
        observations,
        key=lambda o: (
            o.feature_name,
            o.ts_ns,
            o.feature_value,
            o.target_value,
        ),
    )
    rows = {
        "feature_name": [o.feature_name for o in sorted_obs],
        "feature_value": [float(o.feature_value) for o in sorted_obs],
        "target_value": [float(o.target_value) for o in sorted_obs],
    }

    lf = pl.LazyFrame(rows).with_columns(
        pl.col("feature_value")
        .rank(method="min")
        .over("feature_name")
        .cast(pl.Float64)
        .alias("feature_rank"),
        pl.col("target_value")
        .rank(method="min")
        .over("feature_name")
        .cast(pl.Float64)
        .alias("target_rank"),
    )

    grouped = (
        lf.group_by("feature_name")
        .agg(
            pl.len().alias("n_obs"),
            pl.col("feature_value").mean().alias("mean_feature"),
            pl.col("target_value").mean().alias("mean_target"),
            pl.corr("feature_value", "target_value").alias("pearson_corr"),
            pl.corr("feature_rank", "target_rank").alias("rank_corr"),
        )
        .sort("feature_name")
    )

    df = grouped.collect()

    importances: list[FeatureImportance] = []
    for row in df.iter_rows(named=True):
        pearson = _coerce_corr(row["pearson_corr"])
        rank = _coerce_corr(row["rank_corr"])
        abs_score = max(abs(pearson), abs(rank))
        if abs_score > 1.0:
            abs_score = 1.0
        importances.append(
            FeatureImportance(
                feature_name=str(row["feature_name"]),
                n_obs=int(row["n_obs"]),
                mean_feature=float(row["mean_feature"]),
                mean_target=float(row["mean_target"]),
                pearson_corr=pearson,
                rank_corr=rank,
                abs_score=abs_score,
            )
        )

    importances.sort(key=lambda fi: (-fi.abs_score, fi.feature_name))
    by_feature = tuple(importances)

    total_n_obs = sum(fi.n_obs for fi in by_feature)
    if by_feature:
        mean_abs_score = math.fsum(fi.abs_score for fi in by_feature) / len(by_feature)
        if mean_abs_score < 0.0:
            mean_abs_score = 0.0
        if mean_abs_score > 1.0:
            mean_abs_score = 1.0
    else:
        mean_abs_score = 0.0

    return FeatureImportanceReport(
        by_feature=by_feature,
        total_n_obs=total_n_obs,
        mean_abs_score=mean_abs_score,
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _coerce_corr(value: object) -> float:
    """Coerce a polars correlation cell into ``[-1, 1]`` with NaN/None ↦ 0.0.

    Polars returns ``None`` for empty groups and ``NaN`` for zero-variance
    features/targets. Both must collapse to ``0.0`` so the
    :class:`FeatureImportance` invariants hold and replays remain
    byte-identical.
    """
    if value is None:
        return 0.0
    f = float(value)  # type: ignore[arg-type]
    if math.isnan(f):
        return 0.0
    if f < -1.0:
        return -1.0
    if f > 1.0:
        return 1.0
    return f
