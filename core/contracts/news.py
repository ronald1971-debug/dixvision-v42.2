"""News input types (Wave-04.5).

These types are **not** canonical bus events (see ``core/contracts/events.py``
for the only four cross-engine events). News items are *inputs* to the
intelligence engine from RSS / HTTP news feeds; they never cross engine
boundaries on the per-tick canonical bus.

Refs:
- ``manifest.md`` §0.4 (CORE TRUTH — the 4 events)
- INV-08 (only typed events cross domain), INV-15 (replay determinism)

The dataclass is frozen + slotted so it is immutable and hashable, which
keeps the deterministic-replay invariant (TEST-01) intact when news
items are recorded into the ledger as raw inputs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class NewsItem:
    """One headline / story snapshot from a news feed.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds — when this
            item was *observed* by the pump (TimeAuthority, T0-04).
            Caller-supplied, never derived from the feed payload, so
            ``parse_*`` projections stay pure (INV-15).
        source: Stable source identifier matching the SCVS registry row
            (e.g. ``"COINDESK"``). Empty string is rejected.
        guid: Per-feed unique identifier from the underlying RSS / HTTP
            payload (e.g. ``<guid>`` or canonical URL). Used downstream
            for deduplication. Empty string is rejected.
        title: Headline text. Already sanitized (no HTML tags, no
            leading/trailing whitespace, no ``\\r\\n`` runs). Empty
            string is rejected.
        url: Canonical URL the headline points at. Empty string is
            allowed — some sources only publish a guid.
        summary: Short body / description. Sanitized; may be empty.
        published_ts_ns: Optional publication timestamp from the feed
            (parsed from ``<pubDate>`` or equivalent). ``None`` when
            the feed omits it or the value is unparseable. Never
            ``0`` — callers should pass ``None`` explicitly.
        meta: Free-form structural metadata (no PII, no secrets).
    """

    ts_ns: int
    source: str
    guid: str
    title: str
    url: str = ""
    summary: str = ""
    published_ts_ns: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("NewsItem.source must be non-empty")
        if not self.guid:
            raise ValueError("NewsItem.guid must be non-empty")
        if not self.title:
            raise ValueError("NewsItem.title must be non-empty")
        if self.published_ts_ns is not None and self.published_ts_ns <= 0:
            raise ValueError(
                "NewsItem.published_ts_ns must be positive or None"
            )


__all__ = ["NewsItem"]
