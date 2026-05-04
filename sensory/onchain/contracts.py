"""Value type for on-chain intelligence sources.

Resolves the forward-declared ``sensory.onchain.contracts.OnChainMetric``
schema path referenced by Glassnode + Dune rows in
:file:`registry/data_source_registry.yaml`.

Frozen + slotted dataclass (INV-15 deterministic-replay safe).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class OnChainMetric:
    """One on-chain metric observation.

    Generic enough to carry network-level metrics (active addresses,
    realized cap, MVRV) and protocol-level metrics (Uniswap pool TVL,
    holder count, exchange inflows).

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds (caller-
            supplied, never derived from the payload — INV-15).
        source: Stable source identifier matching the SCVS registry row
            (e.g. ``"GLASSNODE"``, ``"DUNE"``). Empty string is rejected.
        metric: Provider-stable metric identifier (e.g. ``"sopr"``,
            ``"active_addresses_24h"``, ``"glassnode/mvrv_z_score"``).
            Empty string is rejected.
        asset: Asset / chain symbol the metric is scoped to
            (e.g. ``"BTC"``, ``"ETH"``). Empty string allowed for
            metrics that are inherently network-wide.
        value: Metric value as a float. Providers that emit integers
            (counts) are coerced upstream — keeping the contract
            single-typed simplifies SCVS validation.
        unit: Free-form unit label carried for display
            (e.g. ``"USD"``, ``"count"``, ``"ratio"``).
        observed_ts_ns: Optional measurement timestamp from the
            provider. ``None`` when the source omits it. Never ``0``.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    source: str
    metric: str
    value: float
    asset: str = ""
    unit: str = ""
    observed_ts_ns: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("OnChainMetric.source must be non-empty")
        if not self.metric:
            raise ValueError("OnChainMetric.metric must be non-empty")
        if (
            self.observed_ts_ns is not None
            and self.observed_ts_ns <= 0
        ):
            raise ValueError(
                "OnChainMetric.observed_ts_ns must be positive or None"
            )
