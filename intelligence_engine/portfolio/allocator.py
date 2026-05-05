"""Capital allocator — confidence-weighted with per-symbol caps.

Phase 10 portfolio brain. Consumes a list of
:class:`AllocationCandidate` proposals plus the current
:class:`ExposureSnapshot` and produces one :class:`AllocationDecision`
per candidate. The allocator is a pure function on
``(candidates, exposure, available_capital_usd, config) → decisions``
(INV-15).

Rules (deterministic, audit-tagged):

1. **Confidence floor** — candidates with ``confidence <
   confidence_floor`` get ``share = 0`` and ``rule = "below_floor"``.
2. **Per-symbol cap** — if adding this candidate would push the
   symbol's projected notional past ``max_symbol_notional_usd`` the
   share is clamped to the residual headroom (possibly to 0). Rule:
   ``"symbol_cap_clamped"`` or ``"symbol_cap_rejected"``.
3. **Confidence-weighted normalisation** — surviving candidates split
   the available capital proportionally to ``confidence``. Sum of
   shares is ≤ 1.0; if all fall through, the residual stays unallocated
   (no implicit defaulting).

Authority constraints (manifest §H1):

* Imports only :mod:`core.contracts` and standard library plus PyYAML.
* No engine cross-imports.
* No clock, no PRNG, no IO outside config load.
* Replay-deterministic.

Refs:
- manifest.md §"Portfolio brain"
- governance_engine.control_plane.exposure_store (the *durable*
  compliance-side cap enforcer; this allocator consults a snapshot of
  it but does not mutate it).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from core.contracts.portfolio import (
    AllocationCandidate,
    AllocationDecision,
    ExposureSnapshot,
)


@dataclasses.dataclass(frozen=True, slots=True)
class PortfolioAllocatorConfig:
    """Versioned allocator parameters.

    Loaded from ``registry/portfolio_allocator.yaml`` so the patch
    pipeline is the only mutator (no runtime mutation, INV-08 / INV-15).
    """

    confidence_floor: float
    max_symbol_notional_usd: float
    max_total_share: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence_floor <= 1.0):
            raise ValueError(
                f"confidence_floor must be in [0, 1], "
                f"got {self.confidence_floor!r}"
            )
        if self.max_symbol_notional_usd <= 0.0:
            raise ValueError(
                "max_symbol_notional_usd must be positive, "
                f"got {self.max_symbol_notional_usd!r}"
            )
        if not (0.0 < self.max_total_share <= 1.0):
            raise ValueError(
                f"max_total_share must be in (0, 1], "
                f"got {self.max_total_share!r}"
            )


def _default_config_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "registry"
        / "portfolio_allocator.yaml"
    )


def load_portfolio_allocator_config(
    path: Path | None = None,
) -> PortfolioAllocatorConfig:
    p = path or _default_config_path()
    raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{p}: expected mapping at top level, got {type(raw)!r}"
        )
    try:
        return PortfolioAllocatorConfig(
            confidence_floor=float(raw["confidence_floor"]),
            max_symbol_notional_usd=float(raw["max_symbol_notional_usd"]),
            max_total_share=float(raw["max_total_share"]),
        )
    except KeyError as e:
        raise ValueError(f"{p}: missing required key {e.args[0]!r}") from None


class PortfolioAllocator:
    """Pure confidence-weighted allocator with per-symbol caps."""

    def __init__(self, config: PortfolioAllocatorConfig) -> None:
        self._config = config

    @property
    def config(self) -> PortfolioAllocatorConfig:
        return self._config

    def allocate(
        self,
        candidates: Sequence[AllocationCandidate],
        exposure: ExposureSnapshot,
        available_capital_usd: float,
    ) -> tuple[AllocationDecision, ...]:
        if available_capital_usd < 0.0:
            raise ValueError(
                "available_capital_usd must be non-negative, "
                f"got {available_capital_usd!r}"
            )

        cfg = self._config
        decisions: list[AllocationDecision] = []
        # Sort by candidate_id so the iteration order is stable across
        # hosts / dict orderings (INV-15).
        ordered = sorted(candidates, key=lambda c: c.candidate_id)

        # Pass 1 — confidence floor + per-symbol cap. Track surviving
        # candidates with the requested *raw* weight = confidence.
        # Per-symbol headroom (mutable across the pass: each accepted
        # candidate consumes a chunk).
        headroom: dict[str, float] = {}
        for c in ordered:
            headroom.setdefault(
                c.symbol,
                max(0.0, cfg.max_symbol_notional_usd - abs(exposure.notional(c.symbol))),
            )

        survivors: list[tuple[AllocationCandidate, float, str]] = []
        rejected_indices: dict[str, AllocationDecision] = {}

        for c in ordered:
            if c.confidence < cfg.confidence_floor:
                rejected_indices[c.candidate_id] = AllocationDecision(
                    candidate_id=c.candidate_id,
                    share=0.0,
                    rule_fired="below_floor",
                )
                continue

            room = headroom[c.symbol]
            if room <= 0.0:
                rejected_indices[c.candidate_id] = AllocationDecision(
                    candidate_id=c.candidate_id,
                    share=0.0,
                    rule_fired="symbol_cap_rejected",
                )
                continue

            survivors.append((c, c.confidence, "ok"))

        # Pass 2 — confidence-weighted normalisation, then symbol-cap
        # clamp. Everything sums to at most cfg.max_total_share.
        total_weight = sum(w for _, w, _ in survivors)
        for c, weight, _tag in survivors:
            if available_capital_usd <= 0.0 or total_weight <= 0.0:
                decisions.append(
                    AllocationDecision(
                        candidate_id=c.candidate_id,
                        share=0.0,
                        rule_fired="no_capital",
                    )
                )
                continue

            base_share = (weight / total_weight) * cfg.max_total_share
            base_notional = base_share * available_capital_usd

            room = headroom[c.symbol]
            if base_notional > room:
                clamped_notional = room
                clamped_share = clamped_notional / available_capital_usd
                rule = "symbol_cap_clamped"
            else:
                clamped_notional = base_notional
                clamped_share = base_share
                rule = "ok"

            headroom[c.symbol] = max(0.0, room - clamped_notional)

            if clamped_share <= 0.0:
                # Cap hit during clamping — degenerate to rejection.
                decisions.append(
                    AllocationDecision(
                        candidate_id=c.candidate_id,
                        share=0.0,
                        rule_fired="symbol_cap_rejected",
                    )
                )
            else:
                decisions.append(
                    AllocationDecision(
                        candidate_id=c.candidate_id,
                        share=clamped_share,
                        rule_fired=rule,
                    )
                )

        decisions.extend(rejected_indices.values())

        # Final ordering: by candidate_id so the audit ledger sees
        # the same ordering regardless of the input sequence.
        decisions.sort(key=lambda d: d.candidate_id)
        return tuple(decisions)


__all__ = [
    "PortfolioAllocator",
    "PortfolioAllocatorConfig",
    "load_portfolio_allocator_config",
]
