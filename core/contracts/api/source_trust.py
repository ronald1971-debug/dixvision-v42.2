"""Pydantic models for the ``/api/operator/source-trust/*`` surface (Paper-S6).

The Paper-S5 governance gate clamps every external SignalEvent's
``confidence`` to the per-source cap loaded from
``registry/external_signal_trust.yaml``. Paper-S6 lets the operator
*promote* a specific source from ``EXTERNAL_LOW`` to ``EXTERNAL_MED``
without redeploying the YAML registry. The promotion is recorded on
the authority ledger so it survives restarts via boot-time replay.

Three routes are typed here:

* ``GET  /api/operator/source-trust`` -- list every source the
  registry knows about and which ones have an active operator
  promotion overlay.
* ``POST /api/operator/source-trust/promote`` -- promote a source
  (currently only ``EXTERNAL_LOW -> EXTERNAL_MED`` is permitted).
* ``POST /api/operator/source-trust/demote`` -- revert a previous
  promotion. Idempotent (demoting an unpromoted source is a no-op
  acknowledged by the response).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SourceTrustPromotionRequest(BaseModel):
    """Operator promotion body for ``POST /api/operator/source-trust/promote``.

    The only supported promotion is ``EXTERNAL_LOW -> EXTERNAL_MED``.
    Sending any other ``target_trust`` returns ``400`` from the route
    handler -- the in-memory store enforces the same invariant so a
    malformed ledger replay also fails closed.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1, max_length=128)
    target_trust: str = Field("EXTERNAL_MED", min_length=1, max_length=32)
    requestor: str = Field("operator", min_length=1, max_length=64)
    reason: str = Field("operator promotion", max_length=512)


class SourceTrustDemotionRequest(BaseModel):
    """Operator demotion body for ``POST /api/operator/source-trust/demote``.

    Idempotent -- demoting a source with no active overlay returns
    ``promoted=false`` with ``previous_target_trust=""``.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(..., min_length=1, max_length=128)
    requestor: str = Field("operator", min_length=1, max_length=64)
    reason: str = Field("operator demotion", max_length=512)


class SourceTrustRow(BaseModel):
    """One row in the ``GET /api/operator/source-trust`` projection.

    ``declared_trust`` is the trust class declared in the YAML
    registry (or ``EXTERNAL_LOW`` if the source is unregistered but
    has an active overlay). ``effective_trust`` reflects the overlay:
    when ``promoted=true`` the effective class is the promoted target
    so the harness gate uses its higher class default.
    """

    model_config = ConfigDict(extra="forbid")

    source_id: str
    declared_trust: str
    effective_trust: str
    declared_cap: float | None = None
    effective_cap: float | None = None
    promoted: bool
    promoted_target_trust: str = ""
    promoted_ts_ns: int = 0
    promoted_requestor: str = ""
    promoted_reason: str = ""


class SourceTrustListResponse(BaseModel):
    """Read projection for ``GET /api/operator/source-trust``."""

    model_config = ConfigDict(extra="forbid")

    rows: list[SourceTrustRow] = Field(default_factory=list)
    promotion_count: int = 0


class SourceTrustPromotionResponse(BaseModel):
    """Write acknowledgement for promote/demote routes.

    The route handler returns the post-mutation row so the UI does
    not need a follow-up ``GET`` to render the new state. The audit
    fields (``ledger_seq``) let the operator confirm the row landed
    on the authority ledger before the response was composed.
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    source_id: str
    declared_trust: str
    effective_trust: str
    declared_cap: float | None = None
    effective_cap: float | None = None
    promoted: bool
    promoted_target_trust: str = ""
    promoted_ts_ns: int = 0
    promoted_requestor: str = ""
    promoted_reason: str = ""
    ledger_seq: int = 0
    ledger_kind: str = ""


__all__ = [
    "SourceTrustDemotionRequest",
    "SourceTrustListResponse",
    "SourceTrustPromotionRequest",
    "SourceTrustPromotionResponse",
    "SourceTrustRow",
]
