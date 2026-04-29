"""Manifest + status projection tests for the credentials module.

These tests are the ones that fire when somebody adds a new
``auth: required`` row to the registry without a matching blueprint:
:func:`test_every_auth_required_row_has_blueprint` walks the live
YAML and asserts every distinct ``provider`` has an entry in
:data:`CREDENTIAL_BLUEPRINTS`. Catches the omission at PR review time
rather than at boot.
"""

from __future__ import annotations

from pathlib import Path

from system_engine.credentials import (
    CREDENTIAL_BLUEPRINTS,
    CredentialBlueprint,
    PresenceState,
    presence_status,
    requirements_for_registry,
)
from system_engine.scvs.source_registry import (
    SourceCategory,
    SourceDeclaration,
    SourceRegistry,
    load_source_registry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LIVE_REGISTRY = REPO_ROOT / "registry" / "data_source_registry.yaml"


def _decl(
    *,
    sid: str,
    provider: str,
    auth: str,
    category: SourceCategory = SourceCategory.AI,
) -> SourceDeclaration:
    return SourceDeclaration(
        id=sid,
        name=f"{provider} test row",
        category=category,
        provider=provider,
        endpoint="https://example.invalid",
        schema="schema.v1",
        auth=auth,
        enabled=False,
        critical=False,
        liveness_threshold_ms=0,
        capabilities=(),
    )


# --------------------------------------------------------------------
# Blueprint table sanity
# --------------------------------------------------------------------


def test_every_auth_required_row_has_blueprint() -> None:
    """Every ``auth: required`` row in the live YAML maps to a blueprint."""

    registry = load_source_registry(LIVE_REGISTRY)
    missing = []
    for s in registry.sources:
        if s.auth == "required" and s.provider not in CREDENTIAL_BLUEPRINTS:
            missing.append((s.id, s.provider))
    assert missing == [], (
        "auth: required rows lacking a CREDENTIAL_BLUEPRINTS entry"
        f" — add to system_engine.credentials.manifest: {missing}"
    )


def test_blueprint_env_vars_are_unique_and_well_formed() -> None:
    """Each blueprint declares at least one well-formed env var name."""

    seen: set[str] = set()
    for provider, bp in CREDENTIAL_BLUEPRINTS.items():
        assert bp.env_vars, f"{provider}: blueprint has no env_vars"
        for name in bp.env_vars:
            assert name == name.upper(), (
                f"{provider}: env var {name!r} should be UPPER_SNAKE_CASE"
            )
            assert " " not in name, (
                f"{provider}: env var {name!r} contains whitespace"
            )
            seen.add(name)
    # Sanity: no two providers should advertise the same env var.
    flat: list[tuple[str, str]] = []
    for provider, bp in CREDENTIAL_BLUEPRINTS.items():
        for name in bp.env_vars:
            flat.append((name, provider))
    by_name: dict[str, list[str]] = {}
    for name, provider in flat:
        by_name.setdefault(name, []).append(provider)
    duplicates = {n: ps for n, ps in by_name.items() if len(ps) > 1}
    assert not duplicates, f"env var collisions across providers: {duplicates}"


def test_paid_only_provider_has_no_signup_url() -> None:
    """Reuters is enterprise-only — sanity check the data we ship."""

    bp = CREDENTIAL_BLUEPRINTS["reuters"]
    assert bp.signup_url is None
    assert bp.free_tier is False


# --------------------------------------------------------------------
# Projection
# --------------------------------------------------------------------


def test_requirements_for_registry_skips_auth_none_rows() -> None:
    registry = SourceRegistry(
        version="v0.1.0",
        sources=(
            _decl(
                sid="SRC-MARKET-X-001",
                provider="binance",
                auth="none",
                category=SourceCategory.MARKET,
            ),
            _decl(
                sid="SRC-AI-OPENAI-001",
                provider="openai",
                auth="required",
            ),
        ),
    )
    reqs = requirements_for_registry(registry)
    assert [r.source_id for r in reqs] == ["SRC-AI-OPENAI-001"]


def test_requirements_for_registry_preserves_yaml_order() -> None:
    registry = SourceRegistry(
        version="v0.1.0",
        sources=(
            _decl(sid="SRC-AI-DEEPSEEK-001", provider="deepseek", auth="required"),
            _decl(sid="SRC-AI-OPENAI-001", provider="openai", auth="required"),
            _decl(sid="SRC-AI-GROK-001", provider="xai", auth="required"),
        ),
    )
    reqs = requirements_for_registry(registry)
    assert [r.source_id for r in reqs] == [
        "SRC-AI-DEEPSEEK-001",
        "SRC-AI-OPENAI-001",
        "SRC-AI-GROK-001",
    ]


def test_unknown_provider_raises_keyerror() -> None:
    registry = SourceRegistry(
        version="v0.1.0",
        sources=(
            _decl(
                sid="SRC-AI-NEW-001",
                provider="some-future-llm",
                auth="required",
            ),
        ),
    )
    try:
        requirements_for_registry(registry)
    except KeyError as exc:
        assert "some-future-llm" in str(exc)
    else:
        raise AssertionError(
            "expected KeyError for unknown provider"
        )


# --------------------------------------------------------------------
# Presence status
# --------------------------------------------------------------------


def _single_blueprint_registry() -> SourceRegistry:
    return SourceRegistry(
        version="v0.1.0",
        sources=(
            _decl(sid="SRC-AI-OPENAI-001", provider="openai", auth="required"),
        ),
    )


def test_presence_present_when_env_set() -> None:
    reqs = requirements_for_registry(_single_blueprint_registry())
    statuses = presence_status(reqs, {"OPENAI_API_KEY": "sk-fake"})
    assert len(statuses) == 1
    assert statuses[0].state is PresenceState.PRESENT
    assert statuses[0].missing_env_vars == ()


def test_presence_missing_when_env_unset() -> None:
    reqs = requirements_for_registry(_single_blueprint_registry())
    statuses = presence_status(reqs, {})
    assert statuses[0].state is PresenceState.MISSING
    assert statuses[0].missing_env_vars == ("OPENAI_API_KEY",)


def test_presence_missing_when_env_empty_string() -> None:
    """Forgotten ``OPENAI_API_KEY=`` line in .env must not count as set."""

    reqs = requirements_for_registry(_single_blueprint_registry())
    statuses = presence_status(reqs, {"OPENAI_API_KEY": ""})
    assert statuses[0].state is PresenceState.MISSING


def test_presence_missing_when_env_only_whitespace() -> None:
    reqs = requirements_for_registry(_single_blueprint_registry())
    statuses = presence_status(reqs, {"OPENAI_API_KEY": "   "})
    assert statuses[0].state is PresenceState.MISSING


def test_presence_partial_for_multi_var_blueprints() -> None:
    """Synthesise a fake two-var blueprint and check tri-state logic."""

    # Build a registry whose sole row maps to a blueprint with two
    # env vars by temporarily wrapping `requirements_for_registry`'s
    # output. We construct CredentialRequirement directly to avoid
    # mutating the public CREDENTIAL_BLUEPRINTS table.
    from system_engine.credentials.manifest import CredentialRequirement
    req = CredentialRequirement(
        source_id="SRC-FAKE-001",
        source_name="fake two-var",
        category="ai",
        provider="fake",
        env_vars=("FAKE_ID", "FAKE_SECRET"),
        signup_url=None,
        free_tier=False,
        notes="",
    )
    statuses = presence_status((req,), {"FAKE_ID": "abc", "FAKE_SECRET": ""})
    assert statuses[0].state is PresenceState.PARTIAL
    assert statuses[0].missing_env_vars == ("FAKE_SECRET",)


def test_presence_status_is_pure() -> None:
    """Same inputs always produce same outputs (replay determinism)."""

    reqs = requirements_for_registry(_single_blueprint_registry())
    a = presence_status(reqs, {"OPENAI_API_KEY": "sk-fake"})
    b = presence_status(reqs, {"OPENAI_API_KEY": "sk-fake"})
    assert a == b


# --------------------------------------------------------------------
# Live registry coverage smoke-tests (PR-A acceptance)
# --------------------------------------------------------------------


def test_live_registry_yields_expected_categories() -> None:
    """At least the 5 AI providers are required; spot-check categories."""

    registry = load_source_registry(LIVE_REGISTRY)
    reqs = requirements_for_registry(registry)
    cats = {r.category for r in reqs}
    # Categories with auth: required as of wave-01.5 (see registry).
    expected_at_minimum = {"ai", "news", "social", "onchain", "macro", "dev"}
    assert expected_at_minimum.issubset(cats), (
        f"live registry no longer covers expected categories;"
        f" got {sorted(cats)}, expected superset of"
        f" {sorted(expected_at_minimum)}"
    )


def test_live_registry_has_no_duplicates() -> None:
    registry = load_source_registry(LIVE_REGISTRY)
    reqs = requirements_for_registry(registry)
    ids = [r.source_id for r in reqs]
    assert len(ids) == len(set(ids))


def test_blueprint_dataclass_is_immutable() -> None:
    """Blueprints must be frozen so the table cannot be mutated at runtime."""

    bp = CredentialBlueprint(env_vars=("X",), signup_url=None, free_tier=False)
    try:
        bp.env_vars = ("Y",)  # type: ignore[misc]
    except (AttributeError, Exception):
        return
    raise AssertionError("CredentialBlueprint should be frozen")
