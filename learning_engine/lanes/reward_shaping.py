"""Reward shaping — Tier 1.5 (H5 + J3 per-component breakdown).

This module sits in the **offline** learning engine. It is invoked
once per realised trade outcome and produces a
:class:`RewardBreakdown` record that:

* **Preserves the raw PnL** (INV-47 — shaping must be invertible /
  auditable). The unshaped reward is always recoverable.
* **Attributes the shaped reward to individual components** (J3).
  Each component is named, signed, and contributes additively to
  ``shaped_reward``. The offline calibrator (Phase 6.T1c) reads this
  breakdown to attribute drift to the *cause* — the consensus
  weight, the strength weight, the kelly-cap penalty, etc. — rather
  than to "the reward function" as a whole.
* **Versions the shaping function**. Every breakdown carries the
  config's ``version`` tag so a calibrator looking at a window of
  ledgered breakdowns knows which shaping function was active.

Authority constraints:

* L2 — offline engine, may not import any runtime engine. The
  inputs are deliberately primitives (``confidence_consensus`` etc.)
  rather than ``ConfidenceComponents`` from
  ``intelligence_engine.meta_controller``. The runtime hot path
  ledgers the components onto a ``SystemEvent`` and this module
  reads them from the ledger via ``state.ledger.reader`` (wiring
  lands in Phase 6.T1c).
* INV-47 — raw PnL is retained on the breakdown.
* J3 — shaping is a *sum of named components*; no opaque scalar.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.contracts.events import SystemEvent, SystemEventKind

REWARD_SHAPING_VERSION = "v3.3-J3"


# ---------------------------------------------------------------------------
# Sizing-rationale tokens
# ---------------------------------------------------------------------------
#
# These mirror the strings emitted by
# ``intelligence_engine.meta_controller.allocation.position_sizer``.
# Reproduced here as constants so this module does not import the
# runtime engine (L2). A unit test pins the alignment.

SIZING_RATIONALE_PRIMARY = "primary"
SIZING_RATIONALE_CONFIDENCE_BELOW_FLOOR = "confidence_below_floor"
SIZING_RATIONALE_REGIME_ZERO_MULTIPLIER = "regime_zero_multiplier"
SIZING_RATIONALE_KELLY_CAPPED = "kelly_capped"

KNOWN_SIZING_RATIONALES: frozenset[str] = frozenset(
    {
        SIZING_RATIONALE_PRIMARY,
        SIZING_RATIONALE_CONFIDENCE_BELOW_FLOOR,
        SIZING_RATIONALE_REGIME_ZERO_MULTIPLIER,
        SIZING_RATIONALE_KELLY_CAPPED,
    }
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewardShapingConfig:
    """Coefficients for :func:`compute_reward_breakdown`.

    Each weight scales one named component contribution. Penalties
    are listed as *positive* magnitudes; the function applies the
    sign internally so YAML stays human-readable (no negative values
    in the registry file).
    """

    pnl_weight: float
    slippage_penalty_per_bps: float
    latency_penalty_per_us: float
    confidence_consensus_weight: float
    confidence_strength_weight: float
    confidence_coverage_weight: float
    sizing_kelly_cap_penalty: float
    sizing_floor_penalty: float
    fallback_penalty: float
    version: str = REWARD_SHAPING_VERSION

    def __post_init__(self) -> None:
        for name in (
            "pnl_weight",
            "slippage_penalty_per_bps",
            "latency_penalty_per_us",
            "confidence_consensus_weight",
            "confidence_strength_weight",
            "confidence_coverage_weight",
            "sizing_kelly_cap_penalty",
            "sizing_floor_penalty",
            "fallback_penalty",
        ):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(
                    f"RewardShapingConfig.{name} must be >= 0 "
                    f"(penalties are unsigned magnitudes): {v}"
                )


def load_reward_shaping_config(path: str | Path) -> RewardShapingConfig:
    """Load shaping config from YAML — fail-fast on missing/extra keys."""
    raw: Any = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(
            f"reward shaping config must be a YAML mapping, got "
            f"{type(raw).__name__}"
        )
    required = {
        "pnl_weight",
        "slippage_penalty_per_bps",
        "latency_penalty_per_us",
        "confidence_consensus_weight",
        "confidence_strength_weight",
        "confidence_coverage_weight",
        "sizing_kelly_cap_penalty",
        "sizing_floor_penalty",
        "fallback_penalty",
    }
    missing = required - raw.keys()
    if missing:
        raise ValueError(
            f"reward shaping config missing keys: {sorted(missing)}"
        )
    extra = raw.keys() - (required | {"version"})
    if extra:
        raise ValueError(
            f"reward shaping config has unknown keys: {sorted(extra)}"
        )
    kwargs: dict[str, Any] = {name: float(raw[name]) for name in required}
    if "version" in raw:
        kwargs["version"] = str(raw["version"])
    return RewardShapingConfig(**kwargs)


# ---------------------------------------------------------------------------
# Breakdown record
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Per-trade J3 attribution record.

    Fields:
        ts_ns: Outcome timestamp.
        raw_pnl: The unshaped scalar reward (INV-47 invertibility).
        components: Ordered tuple of ``(name, contribution)`` pairs.
            The shaping function is the *sum* of these contributions.
            Tuple-of-tuples (rather than dict) so the record is
            hashable and replay-deterministic.
        shaped_reward: The downstream training signal — must equal
            ``sum(c for _, c in components)`` exactly.
        shaping_version: Tag of the shaping function used.
    """

    ts_ns: int
    raw_pnl: float
    components: tuple[tuple[str, float], ...]
    shaped_reward: float
    shaping_version: str

    def to_event(
        self,
        source: str = "learning_engine.lanes.reward_shaping",
    ) -> SystemEvent:
        """Project the breakdown into a ledgerable :class:`SystemEvent`."""
        payload: dict[str, str] = {
            "raw_pnl": f"{self.raw_pnl:.6f}",
            "shaped_reward": f"{self.shaped_reward:.6f}",
            "shaping_version": self.shaping_version,
        }
        for name, value in self.components:
            payload[f"c.{name}"] = f"{value:.6f}"
        return SystemEvent(
            ts_ns=self.ts_ns,
            sub_kind=SystemEventKind.REWARD_BREAKDOWN,
            source=source,
            payload=payload,
        )


# ---------------------------------------------------------------------------
# Pure shaping function
# ---------------------------------------------------------------------------


def compute_reward_breakdown(
    *,
    ts_ns: int,
    raw_pnl: float,
    slippage_bps: float,
    latency_ns: int,
    confidence_consensus: float,
    confidence_strength: float,
    confidence_coverage: float,
    sizing_rationale: str,
    fallback: bool,
    config: RewardShapingConfig,
) -> RewardBreakdown:
    """Decompose a realised trade outcome into named reward components.

    The shaped reward is

    ``shaped = pnl_weight·pnl
        + Σ confidence_weights·confidence_components
        − slippage_penalty_per_bps·|slippage_bps|
        − latency_penalty_per_us·max(latency_us, 0)
        − sizing_kelly_cap_penalty·[rationale==KELLY_CAPPED]
        − sizing_floor_penalty·[rationale==CONFIDENCE_BELOW_FLOOR]
        − fallback_penalty·[fallback]``

    Penalties for ``regime_zero_multiplier`` are not applied — that
    rationale represents a *correct* zero-size outcome (the regime
    forbade trading), not a degraded one.

    INV-47: ``raw_pnl`` is preserved unchanged on the record.
    INV-15: deterministic (no clock, no PRNG).

    Raises:
        ValueError: if ``sizing_rationale`` is unknown, ``latency_ns
        < 0``, or any confidence component is outside ``[0, 1]``.
    """
    if sizing_rationale not in KNOWN_SIZING_RATIONALES:
        raise ValueError(
            f"compute_reward_breakdown: unknown sizing_rationale "
            f"{sizing_rationale!r} (expected one of "
            f"{sorted(KNOWN_SIZING_RATIONALES)})"
        )
    if latency_ns < 0:
        raise ValueError(
            f"compute_reward_breakdown: latency_ns must be >= 0: {latency_ns}"
        )
    for cname, cval in (
        ("confidence_consensus", confidence_consensus),
        ("confidence_strength", confidence_strength),
        ("confidence_coverage", confidence_coverage),
    ):
        if not (0.0 <= cval <= 1.0):
            raise ValueError(
                f"compute_reward_breakdown: {cname} must be in [0, 1]: {cval}"
            )

    components: list[tuple[str, float]] = []

    components.append(("pnl", raw_pnl * config.pnl_weight))

    components.append(
        (
            "confidence_consensus",
            confidence_consensus * config.confidence_consensus_weight,
        )
    )
    components.append(
        (
            "confidence_strength",
            confidence_strength * config.confidence_strength_weight,
        )
    )
    components.append(
        (
            "confidence_coverage",
            confidence_coverage * config.confidence_coverage_weight,
        )
    )

    components.append(
        (
            "slippage_penalty",
            -abs(slippage_bps) * config.slippage_penalty_per_bps,
        )
    )

    latency_us = latency_ns / 1000.0
    components.append(
        (
            "latency_penalty",
            -latency_us * config.latency_penalty_per_us,
        )
    )

    if sizing_rationale == SIZING_RATIONALE_KELLY_CAPPED:
        components.append(
            ("sizing_kelly_cap_penalty", -config.sizing_kelly_cap_penalty)
        )
    elif sizing_rationale == SIZING_RATIONALE_CONFIDENCE_BELOW_FLOOR:
        components.append(
            ("sizing_floor_penalty", -config.sizing_floor_penalty)
        )

    if fallback:
        components.append(("fallback_penalty", -config.fallback_penalty))

    shaped = sum(v for _, v in components)

    return RewardBreakdown(
        ts_ns=ts_ns,
        raw_pnl=raw_pnl,
        components=tuple(components),
        shaped_reward=shaped,
        shaping_version=config.version,
    )


def breakdown_components_dict(
    breakdown: RewardBreakdown,
) -> Mapping[str, float]:
    """Return a name→contribution mapping for ergonomic asserts.

    Component names are unique within a single breakdown; this helper
    is purely a convenience for tests / dashboards.
    """
    return dict(breakdown.components)


__all__ = [
    "KNOWN_SIZING_RATIONALES",
    "REWARD_SHAPING_VERSION",
    "RewardBreakdown",
    "RewardShapingConfig",
    "SIZING_RATIONALE_CONFIDENCE_BELOW_FLOOR",
    "SIZING_RATIONALE_KELLY_CAPPED",
    "SIZING_RATIONALE_PRIMARY",
    "SIZING_RATIONALE_REGIME_ZERO_MULTIPLIER",
    "breakdown_components_dict",
    "compute_reward_breakdown",
    "load_reward_shaping_config",
]
