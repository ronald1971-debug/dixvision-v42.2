"""Pydantic response models for the ``/api/credentials/*`` HTTP surface.

These models live alongside the rest of ``core/contracts`` so the
TypeScript codegen (`tools/codegen/pydantic_to_ts.py`) can import a
single, stable namespace for the wave-02 dashboard. The route handlers
in ``ui/server.py`` use these as ``response_model=`` so FastAPI's own
validation matches the schema we ship to the client.

Only response shapes live here — request bodies (``CredentialVerifyIn``
etc.) stay in ``ui/server.py`` because they are tightly coupled to the
route's body parser and never round-trip back to the client.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


# Mirrors `system_engine.credentials.status.PresenceState`. Duplicated
# (rather than re-exported) so the API surface does not lock the
# internal enum's identity into the public contract — the TS client
# sees a string-literal union either way.
class PresenceStateApi(StrEnum):
    PRESENT = "present"
    PARTIAL = "partial"
    MISSING = "missing"


class CredentialItem(BaseModel):
    """One row in the credential matrix, rendered per registry source."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_name: str
    category: str
    provider: str
    env_vars: list[str]
    env_vars_present: list[bool]
    missing_env_vars: list[str]
    signup_url: str | None
    free_tier: bool
    notes: str | None
    state: PresenceStateApi


class CredentialsSummary(BaseModel):
    """Aggregate counts so the page header can render a one-line tally."""

    model_config = ConfigDict(extra="forbid")

    total: int
    present: int
    partial: int
    missing: int


class CredentialsStatusResponse(BaseModel):
    """Top-level body of ``GET /api/credentials/status``."""

    model_config = ConfigDict(extra="forbid")

    summary: CredentialsSummary
    writable: bool
    items: list[CredentialItem]


__all__ = [
    "CredentialItem",
    "CredentialsStatusResponse",
    "CredentialsSummary",
    "PresenceStateApi",
]
