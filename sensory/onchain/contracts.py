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
        if self.observed_ts_ns is not None and self.observed_ts_ns <= 0:
            raise ValueError("OnChainMetric.observed_ts_ns must be positive or None")


# ---------------------------------------------------------------------------
# A-15 — Solana onchain intelligence value type
# ---------------------------------------------------------------------------
#
# OnchainEvent is the advisory output produced by Solana-chain intelligence
# adapters (Helius enhanced transactions, token holder shifts). It is
# *not* a typed bus event — adapters under ``execution_engine.adapters``
# cannot construct ``SignalEvent`` / ``HazardEvent`` (B27 / B28 / INV-71).
# Downstream intelligence-tier coordinators project ``OnchainEvent`` into
# typed events on the proper side of the authority boundary.


@dataclass(frozen=True, slots=True)
class OnchainEvent:
    """Advisory record describing one parsed onchain observation.

    Frozen + slotted (INV-15 deterministic-replay safe). Eager validation
    on construction; no clock, no IO. Producers must supply ``ts_ns``
    from :class:`system.time_source.TimeAuthority`.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds.
        source: Stable source identifier (e.g. ``"HELIUS"``). Non-empty.
        chain: Chain identifier (e.g. ``"SOLANA"``). Non-empty.
        kind: Event category — one of ``"TRANSFER"``, ``"SWAP"``,
            ``"MINT"``, ``"BURN"``, ``"HOLDER_SHIFT"``, ``"PROGRAM_CALL"``,
            ``"NFT_TRADE"``, ``"UNKNOWN"``. Non-empty.
        asset: Token mint / asset symbol the event scopes to. Empty
            string is allowed for chain-wide observations.
        actor: Wallet address that initiated the event. Empty string
            allowed when the source omits actor attribution.
        signature: Onchain transaction signature (base58). Empty string
            allowed for derived events (e.g. holder shift snapshots).
        rug_score: ``[0.0, 1.0]`` heuristic risk score. ``None`` when
            the source omits it. ``0.0`` is a legitimate "no risk"
            observation.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    source: str
    chain: str
    kind: str
    asset: str = ""
    actor: str = ""
    signature: str = ""
    rug_score: float | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("OnchainEvent.ts_ns must be positive")
        if not self.source:
            raise ValueError("OnchainEvent.source must be non-empty")
        if not self.chain:
            raise ValueError("OnchainEvent.chain must be non-empty")
        if not self.kind:
            raise ValueError("OnchainEvent.kind must be non-empty")
        if self.rug_score is not None and not (0.0 <= float(self.rug_score) <= 1.0):
            raise ValueError("OnchainEvent.rug_score must be in [0.0, 1.0] or None")


@dataclass(frozen=True, slots=True)
class HolderShiftAdvisory:
    """Advisory record describing a shift in a token's top-holder set.

    Emitted by onchain intelligence adapters when a holder snapshot
    diff exceeds a caller-supplied threshold. Frozen + slotted, no
    clock, no IO, eager validation.

    Attributes:
        ts_ns: Monotonic snapshot timestamp in nanoseconds.
        asset: Token mint / asset symbol. Non-empty.
        top_holder_share_before: Combined balance share of the top-N
            holders prior to the diff, in ``[0.0, 1.0]``.
        top_holder_share_after: Combined balance share of the top-N
            holders after the diff, in ``[0.0, 1.0]``.
        holders_changed: Number of top-N wallet addresses that
            entered or left the set. ``>= 0``.
        rug_score: ``[0.0, 1.0]`` heuristic risk score. Required
            (advisory only — never autonomously triggers anything).
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    asset: str
    top_holder_share_before: float
    top_holder_share_after: float
    holders_changed: int
    rug_score: float
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts_ns <= 0:
            raise ValueError("HolderShiftAdvisory.ts_ns must be positive")
        if not self.asset:
            raise ValueError("HolderShiftAdvisory.asset must be non-empty")
        for name in (
            "top_holder_share_before",
            "top_holder_share_after",
            "rug_score",
        ):
            v = float(getattr(self, name))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"HolderShiftAdvisory.{name} must be in [0.0, 1.0]")
        if self.holders_changed < 0:
            raise ValueError("HolderShiftAdvisory.holders_changed must be >= 0")

    @property
    def share_delta(self) -> float:
        """Signed change in top-holder concentration."""
        return self.top_holder_share_after - self.top_holder_share_before


__all__ = [
    "HolderShiftAdvisory",
    "OnChainMetric",
    "OnchainEvent",
]
