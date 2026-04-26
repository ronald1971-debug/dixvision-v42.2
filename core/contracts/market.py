"""Market data input types (Phase E1).

These types are **not** canonical bus events (see ``core/contracts/events.py``
for the only four cross-engine events). Market ticks are *inputs* to the
``intelligence_engine`` from data feeds; they never cross engine boundaries
on the per-tick canonical bus.

Refs:
- ``manifest.md`` §0.4 (CORE TRUTH — the 4 events)
- ``build_plan.md`` §Phase E1
- INV-08 (only typed events cross domain), INV-15 (replay determinism)

The dataclass is frozen + slotted so it is immutable and hashable, which
keeps the deterministic-replay invariant (TEST-01) intact when ticks are
recorded into the ledger as raw inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MarketTick:
    """One quote/print from a venue.

    Attributes:
        ts_ns: Monotonic timestamp in nanoseconds (TimeAuthority, T0-04).
        symbol: Instrument identifier.
        bid: Best bid.
        ask: Best ask.
        last: Last trade price (used as the deterministic mark for paper
            broker fills in Phase E1).
        volume: Optional volume; defaults to ``0.0``.
        venue: Optional venue tag (e.g. ``"BINANCE"``); empty string if
            irrelevant for the current configuration.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float = 0.0
    venue: str = ""
    meta: Mapping[str, str] = field(default_factory=dict)


__all__ = ["MarketTick"]
