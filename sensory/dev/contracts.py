"""Value type for repository / developer signals.

Resolves the forward-declared ``sensory.dev.contracts.RepoEvent`` schema
path referenced by the GitHub row in
:file:`registry/data_source_registry.yaml`.

Frozen + slotted dataclass (INV-15 deterministic-replay safe).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RepoEvent:
    """One repository event snapshot.

    Generic across GitHub, GitLab, and similar code-hosting providers.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds (caller-
            supplied, never derived from the payload — INV-15).
        source: Stable source identifier matching the SCVS registry row
            (e.g. ``"GITHUB"``). Empty string is rejected.
        event_id: Provider-stable identifier for the event
            (e.g. GitHub ``X-GitHub-Delivery``). Empty string is
            rejected.
        repo: ``owner/repo`` slug. Empty string is rejected.
        event_type: Event classification (e.g. ``"push"``,
            ``"pull_request"``, ``"release"``, ``"star"``). Empty string
            is rejected.
        actor: Actor handle / username. Empty string is rejected.
        url: Optional canonical URL of the event resource.
        occurred_ts_ns: Optional event timestamp from the provider.
            ``None`` when the source omits it. Never ``0``.
        meta: Free-form structural metadata (ref, sha, action, etc.).
            No PII beyond the public actor handle.
    """

    ts_ns: int
    source: str
    event_id: str
    repo: str
    event_type: str
    actor: str
    url: str = ""
    occurred_ts_ns: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("RepoEvent.source must be non-empty")
        if not self.event_id:
            raise ValueError("RepoEvent.event_id must be non-empty")
        if not self.repo:
            raise ValueError("RepoEvent.repo must be non-empty")
        if not self.event_type:
            raise ValueError("RepoEvent.event_type must be non-empty")
        if not self.actor:
            raise ValueError("RepoEvent.actor must be non-empty")
        if self.occurred_ts_ns is not None and self.occurred_ts_ns <= 0:
            raise ValueError(
                "RepoEvent.occurred_ts_ns must be positive or None"
            )
