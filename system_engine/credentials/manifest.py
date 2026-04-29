"""Static per-provider credential blueprints + registry-driven projection.

Adding a new ``auth: required`` row to ``data_source_registry.yaml``
that maps to a previously-unknown ``provider`` MUST be paired with a
new entry in :data:`CREDENTIAL_BLUEPRINTS`. The strict join in
:func:`requirements_for_registry` raises :class:`KeyError` otherwise,
which is also covered by a unit test that walks every registry row.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from system_engine.scvs.source_registry import (
    SourceDeclaration,
    SourceRegistry,
)


@dataclass(frozen=True, slots=True)
class CredentialBlueprint:
    """Static knowledge about a provider's credential shape.

    Attributes
    ----------
    env_vars
        Names of every environment variable the operator must set for
        this provider. Most providers need exactly one bearer-token
        style key; a couple need a (key, secret) pair.
    signup_url
        Where the operator can request a key. ``None`` when there is
        no public self-signup (paid-only enterprise providers).
    free_tier
        ``True`` when a free / trial tier exists and the operator can
        get a working key without payment. The dashboard renders a
        "free tier available" badge when this is ``True``.
    notes
        Short operator-facing hint shown next to the row on the
        ``/credentials`` page (e.g. "GitHub PAT with ``public_repo``
        scope is sufficient"). Empty string when there is nothing
        useful to add.
    """

    env_vars: tuple[str, ...]
    signup_url: str | None
    free_tier: bool
    notes: str = ""


# --------------------------------------------------------------------
# Blueprint table — single source of truth for env-var conventions.
#
# The keys are the ``provider`` field of a ``SourceDeclaration``. Each
# blueprint covers every registry row that maps to that provider, so
# future registry rows for the same provider (e.g. a second OpenAI
# row tagged with a different ``schema``) inherit the same env vars
# automatically.
# --------------------------------------------------------------------
CREDENTIAL_BLUEPRINTS: Mapping[str, CredentialBlueprint] = MappingProxyType(
    {
        # ----- AI providers -----
        "openai": CredentialBlueprint(
            env_vars=("OPENAI_API_KEY",),
            signup_url="https://platform.openai.com/signup",
            free_tier=True,
            notes="$5 trial credit on signup; key starts with sk-.",
        ),
        "google": CredentialBlueprint(
            # Gemini uses Google AI Studio API keys, NOT GOOGLE_API_KEY
            # (which historically meant Google Cloud Platform).
            env_vars=("GEMINI_API_KEY",),
            signup_url="https://aistudio.google.com/app/apikey",
            free_tier=True,
            notes="Free Google AI Studio key; rate-limited but workable.",
        ),
        "xai": CredentialBlueprint(
            env_vars=("XAI_API_KEY",),
            signup_url="https://console.x.ai/",
            free_tier=False,
            notes="Paid only; key starts with xai-.",
        ),
        "deepseek": CredentialBlueprint(
            env_vars=("DEEPSEEK_API_KEY",),
            signup_url="https://platform.deepseek.com/sign_up",
            free_tier=True,
            notes="Free trial credit on signup.",
        ),
        "cognition": CredentialBlueprint(
            # Devin AI is reached via the Devin MCP integration; the
            # API key is the operator's Devin account token.
            env_vars=("DEVIN_API_KEY",),
            signup_url="https://app.devin.ai/settings/integrations",
            free_tier=False,
            notes=(
                "Routed via Devin MCP. Token from Settings →"
                " Integrations → API keys."
            ),
        ),
        # ----- News -----
        "reuters": CredentialBlueprint(
            env_vars=("REUTERS_API_KEY",),
            # Reuters Connect is enterprise-only — there is no public
            # signup form. Set None so the UI does not show a broken
            # signup button.
            signup_url=None,
            free_tier=False,
            notes="Reuters Connect — enterprise only, no public signup.",
        ),
        # ----- Social -----
        "x": CredentialBlueprint(
            env_vars=("X_BEARER_TOKEN",),
            signup_url="https://developer.x.com/en/portal/dashboard",
            free_tier=True,
            notes=(
                "X (Twitter) v2 API Bearer Token. Free tier is heavily"
                " rate-limited; Basic ($100/mo) for production use."
            ),
        ),
        # ----- Onchain -----
        "glassnode": CredentialBlueprint(
            env_vars=("GLASSNODE_API_KEY",),
            signup_url="https://studio.glassnode.com/settings/api",
            # Glassnode has a free Tier 1 plan with a small set of
            # daily metrics; sufficient for a first-look dashboard.
            free_tier=True,
            notes="Free Tier 1 plan covers a small set of daily metrics.",
        ),
        "dune": CredentialBlueprint(
            env_vars=("DUNE_API_KEY",),
            signup_url="https://dune.com/settings/api",
            free_tier=True,
            notes="Free tier: 2,500 datapoints/mo, 40 executions/mo.",
        ),
        # ----- Macro -----
        "fred": CredentialBlueprint(
            env_vars=("FRED_API_KEY",),
            signup_url="https://fred.stlouisfed.org/docs/api/api_key.html",
            free_tier=True,
            notes="Free; St Louis Fed account is enough.",
        ),
        "bls": CredentialBlueprint(
            env_vars=("BLS_API_KEY",),
            signup_url="https://data.bls.gov/registrationEngine/",
            free_tier=True,
            notes="Free public-API registration; key by email.",
        ),
        # ----- Dev -----
        "github": CredentialBlueprint(
            # Canonical convention is GITHUB_TOKEN (gh CLI, Actions,
            # most SDKs). Fine-grained PAT recommended.
            env_vars=("GITHUB_TOKEN",),
            signup_url="https://github.com/settings/personal-access-tokens",
            free_tier=True,
            notes=(
                "Fine-grained PAT with read-only repo + read-only"
                " issues scope is enough for the current ingestor."
            ),
        ),
    }
)


@dataclass(frozen=True, slots=True)
class CredentialRequirement:
    """Per-row credential requirement projection.

    Joins one ``auth: required`` :class:`SourceDeclaration` against
    the static blueprint for its ``provider``. Pure data — does not
    contain any runtime presence info (see :class:`CredentialStatus`
    in :mod:`system_engine.credentials.status` for that).
    """

    source_id: str
    source_name: str
    category: str
    provider: str
    env_vars: tuple[str, ...]
    signup_url: str | None
    free_tier: bool
    notes: str


def _project(decl: SourceDeclaration, bp: CredentialBlueprint) -> CredentialRequirement:
    return CredentialRequirement(
        source_id=decl.id,
        source_name=decl.name,
        category=decl.category.value,
        provider=decl.provider,
        env_vars=bp.env_vars,
        signup_url=bp.signup_url,
        free_tier=bp.free_tier,
        notes=bp.notes,
    )


def requirements_for_registry(
    registry: SourceRegistry,
) -> tuple[CredentialRequirement, ...]:
    """Return a credential requirement per ``auth: required`` row.

    Order matches registry YAML order so the ``/credentials`` page
    renders rows in a stable, operator-controllable order. Raises
    :class:`KeyError` if a row's ``provider`` lacks a blueprint —
    callers should catch this only at boot and surface it as a
    config error (it means somebody added a registry row without
    adding a credential blueprint, which is a CI-detectable mistake).
    """

    out: list[CredentialRequirement] = []
    for decl in registry.sources:
        if decl.auth != "required":
            continue
        try:
            bp = CREDENTIAL_BLUEPRINTS[decl.provider]
        except KeyError as exc:  # pragma: no cover - guarded by tests
            raise KeyError(
                f"credential blueprint missing for provider"
                f" '{decl.provider}' (source row {decl.id});"
                f" add an entry to CREDENTIAL_BLUEPRINTS in"
                f" system_engine.credentials.manifest"
            ) from exc
        out.append(_project(decl, bp))
    return tuple(out)


__all__ = [
    "CREDENTIAL_BLUEPRINTS",
    "CredentialBlueprint",
    "CredentialRequirement",
    "requirements_for_registry",
]
