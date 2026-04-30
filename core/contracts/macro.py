"""Macro time-series input types (Wave-04.5).

These types are **not** canonical bus events (see ``core/contracts/events.py``
for the only four cross-engine events). Macro observations are *inputs*
to the intelligence engine from FRED / BLS / similar HTTP endpoints; they
never cross engine boundaries on the per-tick canonical bus.

Refs:
- ``manifest.md`` §0.4 (CORE TRUTH — the 4 events)
- INV-08 (only typed events cross domain), INV-15 (replay determinism)

The dataclass is frozen + slotted so it is immutable and hashable, which
keeps the deterministic-replay invariant (TEST-01) intact when macro
observations are recorded into the ledger as raw inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class MacroObservation:
    """One numeric observation from a macro time-series feed.

    The shape mirrors :class:`core.contracts.news.NewsItem` so the
    intelligence engine can fan both kinds of external input through
    the same per-source SCVS audit trail. The semantic difference is
    that a ``MacroObservation`` carries a *number* (a yield, an index,
    a level, a count) rather than free text, so downstream projection
    layers must treat it as a numeric signal rather than a sentiment
    candidate.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds — when this
            observation was *observed* by the pump (TimeAuthority,
            T0-04). Caller-supplied, never derived from the feed
            payload, so ``parse_*`` projections stay pure (INV-15).
        source: Stable source identifier matching the SCVS registry
            row (e.g. ``"FRED"``, ``"BLS"``). Empty string is rejected.
        series_id: Per-source series identifier (e.g. ``"DGS10"`` for
            the FRED 10-Year Treasury constant-maturity yield, or
            ``"CPIAUCSL"`` for headline CPI). Empty string is rejected
            because dedup + projection logic keys on it.
        observation_date: ISO-8601 date the observation refers to,
            ``"YYYY-MM-DD"``. This is the *as-of* date in the
            publisher's calendar, not the ingestion time. Empty string
            is rejected.
        value: The numeric value at ``observation_date``. ``None`` when
            the source publishes a sentinel (FRED uses ``"."`` for
            holidays / unreleased data); never ``NaN`` — callers must
            map ``"."`` to ``None`` explicitly.
        units: Unit hint from the source (e.g. ``"Percent"``,
            ``"Index 1982-1984=100"``). Sanitized; may be empty.
        title: Optional human-readable series title (e.g.
            ``"10-Year Treasury Constant Maturity Rate"``). Sanitized;
            may be empty.
        observed_ts_ns: Optional ns timestamp derived from
            ``observation_date`` at UTC midnight. ``None`` when the
            date is unparseable. Never ``0`` or negative; callers
            should pass ``None`` explicitly for pre-1970 dates.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    source: str
    series_id: str
    observation_date: str
    value: float | None = None
    units: str = ""
    title: str = ""
    observed_ts_ns: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("MacroObservation.source must be non-empty")
        if not self.series_id:
            raise ValueError("MacroObservation.series_id must be non-empty")
        if not self.observation_date:
            raise ValueError(
                "MacroObservation.observation_date must be non-empty"
            )
        if self.observed_ts_ns is not None and self.observed_ts_ns <= 0:
            raise ValueError(
                "MacroObservation.observed_ts_ns must be positive or None"
            )


__all__ = ["MacroObservation"]
