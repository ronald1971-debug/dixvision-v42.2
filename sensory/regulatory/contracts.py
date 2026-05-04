"""Value type for regulatory filings.

Resolves the forward-declared ``sensory.regulatory.contracts.Filing``
schema path referenced by the SEC EDGAR row in
:file:`registry/data_source_registry.yaml`.

Frozen + slotted dataclass (INV-15 deterministic-replay safe).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Filing:
    """One regulatory filing snapshot.

    Generic across SEC, FINRA, and equivalent global regulators.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds (caller-
            supplied, never derived from the payload — INV-15).
        source: Stable source identifier matching the SCVS registry row
            (e.g. ``"SEC_EDGAR"``). Empty string is rejected.
        filing_id: Provider-stable identifier (e.g. EDGAR accession
            number ``0001628280-23-001234``). Empty string is rejected.
        form_type: Form classification (e.g. ``"10-K"``, ``"8-K"``,
            ``"13F-HR"``, ``"4"``). Empty string is rejected.
        filer: Filing entity name (e.g. ``"Apple Inc."``). Empty string
            is rejected.
        url: Canonical URL of the filing document.
        filed_ts_ns: Optional filing timestamp from the provider.
            ``None`` when the source omits it. Never ``0``.
        meta: Free-form structural metadata (CIK, ticker, period of
            report, etc.). No PII beyond what the regulator publishes.
    """

    ts_ns: int
    source: str
    filing_id: str
    form_type: str
    filer: str
    url: str = ""
    filed_ts_ns: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Filing.source must be non-empty")
        if not self.filing_id:
            raise ValueError("Filing.filing_id must be non-empty")
        if not self.form_type:
            raise ValueError("Filing.form_type must be non-empty")
        if not self.filer:
            raise ValueError("Filing.filer must be non-empty")
        if self.filed_ts_ns is not None and self.filed_ts_ns <= 0:
            raise ValueError(
                "Filing.filed_ts_ns must be positive or None"
            )
