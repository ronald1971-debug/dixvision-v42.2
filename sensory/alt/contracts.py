"""Value type for prediction-market observations.

Resolves the forward-declared
``sensory.alt.contracts.PredictionMarket`` schema path referenced by
the Polymarket row in :file:`registry/data_source_registry.yaml`.

Frozen + slotted dataclass (INV-15 deterministic-replay safe).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PredictionMarket:
    """One prediction-market quote / snapshot.

    Generic across Polymarket, Kalshi, and similar event-contract
    venues.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds (caller-
            supplied, never derived from the payload — INV-15).
        source: Stable source identifier matching the SCVS registry row
            (e.g. ``"POLYMARKET"``). Empty string is rejected.
        market_id: Provider-stable market identifier. Empty string is
            rejected.
        question: Market question text. Empty string is rejected.
        outcome: Outcome label being quoted (e.g. ``"YES"``, ``"NO"``,
            ``"Trump"``). Empty string is rejected.
        probability: Implied probability of ``outcome`` in ``[0.0, 1.0]``.
        volume_usd: Optional cumulative traded volume in USD. ``None``
            when the venue omits it. Must be ``>= 0`` when present.
        observed_ts_ns: Optional venue timestamp. ``None`` when the
            source omits it. Never ``0``.
        meta: Free-form structural metadata (slug, end_date, etc.).
    """

    ts_ns: int
    source: str
    market_id: str
    question: str
    outcome: str
    probability: float
    volume_usd: float | None = None
    observed_ts_ns: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("PredictionMarket.source must be non-empty")
        if not self.market_id:
            raise ValueError(
                "PredictionMarket.market_id must be non-empty"
            )
        if not self.question:
            raise ValueError(
                "PredictionMarket.question must be non-empty"
            )
        if not self.outcome:
            raise ValueError(
                "PredictionMarket.outcome must be non-empty"
            )
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                "PredictionMarket.probability must be in [0.0, 1.0]"
            )
        if self.volume_usd is not None and self.volume_usd < 0:
            raise ValueError(
                "PredictionMarket.volume_usd must be >= 0 or None"
            )
        if (
            self.observed_ts_ns is not None
            and self.observed_ts_ns <= 0
        ):
            raise ValueError(
                "PredictionMarket.observed_ts_ns must be positive or None"
            )
