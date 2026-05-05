"""Portfolio contracts — intelligence-side allocator + exposure manager.

Distinct from :mod:`governance_engine.control_plane.exposure_store`,
which is the durable compliance ledger that authoritatively caps
per-symbol position exposure on the governance side. The contracts
here describe the *decision-making* layer that the portfolio brain
runs *before* candidate intents reach the governance gate:

* :class:`AllocationCandidate` — one (signal, archetype, confidence)
  proposal from the intelligence engine, asking for a slice of free
  capital.
* :class:`AllocationDecision` — the allocator's reply per candidate:
  the proportional share of the available capital that goes to this
  candidate, plus a stable rule tag for audit.
* :class:`ExposureSnapshot` — point-in-time view of live per-symbol
  notional exposure that the allocator reads to enforce caps.

All three are frozen dataclasses with range-checked fields so the
allocator stays pure (INV-15) and the audit ledger has stable types.

Refs:
- manifest.md §"Portfolio brain"
- full_feature_spec.md §"Allocator + ExposureManager"
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping


@dataclasses.dataclass(frozen=True, slots=True)
class AllocationCandidate:
    """One capital-allocation proposal.

    Attributes:
        candidate_id: Stable id for this proposal (typically the
            originating ``SignalEvent.signal_id``). Empty rejected.
        symbol: Asset symbol (e.g. ``"BTC-USD"``). Empty rejected.
        archetype_id: Trader-archetype id (matching
            ``registry/trader_archetypes.yaml``). Empty rejected.
        confidence: Signal confidence in [0, 1]. NaN/out-of-range
            rejected.
        side: Order side. ``"BUY"`` or ``"SELL"``; HOLD is not a
            candidate (no allocation needed).
    """

    candidate_id: str
    symbol: str
    archetype_id: str
    confidence: float
    side: str

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("AllocationCandidate.candidate_id must be non-empty")
        if not self.symbol:
            raise ValueError("AllocationCandidate.symbol must be non-empty")
        if not self.archetype_id:
            raise ValueError("AllocationCandidate.archetype_id must be non-empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                "AllocationCandidate.confidence must be in [0, 1], "
                f"got {self.confidence!r}"
            )
        if self.side not in ("BUY", "SELL"):
            raise ValueError(
                f"AllocationCandidate.side must be 'BUY' or 'SELL', "
                f"got {self.side!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class AllocationDecision:
    """Allocator's reply for one candidate.

    ``share`` is the proportional fraction in [0, 1] of the
    available-capital pool that goes to this candidate, *after*
    cap enforcement. ``share == 0.0`` means "rejected" (cap hit,
    confidence floor missed, etc.). The rule tag is stable so the
    audit trail can attribute the decision unambiguously.
    """

    candidate_id: str
    share: float
    rule_fired: str

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("AllocationDecision.candidate_id must be non-empty")
        if not (0.0 <= self.share <= 1.0):
            raise ValueError(
                f"AllocationDecision.share must be in [0, 1], got {self.share!r}"
            )
        if not self.rule_fired:
            raise ValueError("AllocationDecision.rule_fired must be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class ExposureSnapshot:
    """Read-only per-symbol exposure used by the allocator.

    Values are notional USD exposure (signed: positive = long, negative
    = short). The snapshot is intentionally a point-in-time projection
    so the allocator's decision is reproducible from the same input.
    """

    ts_ns: int
    by_symbol: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("ExposureSnapshot.ts_ns must be positive")
        for sym, notional in self.by_symbol.items():
            if not sym:
                raise ValueError(
                    "ExposureSnapshot.by_symbol contains empty symbol"
                )
            if notional != notional:  # NaN check
                raise ValueError(
                    f"ExposureSnapshot.by_symbol[{sym!r}] is NaN"
                )

    def notional(self, symbol: str) -> float:
        return float(self.by_symbol.get(symbol, 0.0))


__all__ = [
    "AllocationCandidate",
    "AllocationDecision",
    "ExposureSnapshot",
]
