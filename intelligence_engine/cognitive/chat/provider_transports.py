# Tier-I I-22 — provider transports audit.
#
# Catalogue of the chat provider transports wired into
# ``RegistryDispatchChatTransport`` (see
# ``intelligence_engine/cognitive/chat/http_chat_transport.py``).
#
# This module is intentionally *not* the place where HTTP is performed.
# It is a frozen audit surface so:
#
#   1. AST guardrails (this module's tests) can pin the closed set of
#      wired providers at lint time.
#   2. Operator dashboards / credential matrices can introspect which
#      provider keys are expected by the runtime without importing the
#      transport implementation.
#   3. INV-15 byte-identical replay can checkpoint the audit output
#      across runs.
#
# Forbidden lazy seams (this module declares them but never imports):
#
#   * ``openai`` / ``anthropic`` / ``google-genai`` / ``groq`` — vendor
#     SDKs. Production transports use ``urllib.request`` only.
#
# Authority constraints (pinned by ``tests/test_provider_transports.py``):
#
#   * **RUNTIME_SAFE** — pure audit dataclasses + canonical sort. No
#     clock, no I/O, no PRNG. Three independent calls to
#     :func:`provider_transport_audit` produce byte-identical tuples
#     (INV-15).
#   * **B1** — no execution_engine / governance_engine / system_engine
#     cross-imports.
#   * **B27 / B28 / INV-71** — no typed-event constructors.
#   * **B24** — chat-tier-internal: only allowed imports from
#     ``intelligence_engine.cognitive.chat.*`` and ``core.*`` /
#     stdlib.
#   * No top-level imports of vendor SDKs, ``time``, ``datetime``,
#     ``random``, ``asyncio``, ``requests``.
"""I-22 provider transports — typed audit surface for the chat dispatcher."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "ProviderTransportError",
    "ProviderTransportSpec",
    "TRANSPORT_SPECS",
    "spec_for_provider",
    "provider_transport_audit",
    "wired_provider_keys",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = (
    "openai",
    "anthropic",
    "google-genai",
    "groq",
)


class ProviderTransportError(ValueError):
    """Raised when the audit surface is queried with an unknown provider."""


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderTransportSpec:
    """Audit row for one chat provider wired in ``RegistryDispatchChatTransport``.

    Fields:
        provider: registry key on :class:`core.cognitive_router.AIProvider`.
        family: backend family — one of ``"openai_compat"`` /
            ``"google"`` / ``"cognition"``.
        env_var: environment variable name that holds the API key.
        base_url: HTTPS endpoint base URL.
        auth_scheme: header authentication scheme — one of
            ``"bearer"`` / ``"query_param"`` / ``"header_token"``.
    """

    provider: str
    family: str
    env_var: str
    base_url: str
    auth_scheme: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("provider", self.provider),
            ("family", self.family),
            ("env_var", self.env_var),
            ("base_url", self.base_url),
            ("auth_scheme", self.auth_scheme),
        ):
            if not isinstance(value, str) or not value:
                raise ProviderTransportError(
                    f"ProviderTransportSpec.{field_name} must be a non-empty str, got {value!r}"
                )
        if self.family not in _ALLOWED_FAMILIES:
            raise ProviderTransportError(
                f"ProviderTransportSpec.family must be one of"
                f" {_ALLOWED_FAMILIES!r}, got {self.family!r}"
            )
        if self.auth_scheme not in _ALLOWED_AUTH_SCHEMES:
            raise ProviderTransportError(
                f"ProviderTransportSpec.auth_scheme must be one of"
                f" {_ALLOWED_AUTH_SCHEMES!r}, got {self.auth_scheme!r}"
            )
        if not self.base_url.startswith("https://"):
            raise ProviderTransportError(
                f"ProviderTransportSpec.base_url must be https://, got {self.base_url!r}"
            )


_ALLOWED_FAMILIES: tuple[str, ...] = (
    "openai_compat",
    "google",
    "cognition",
)


_ALLOWED_AUTH_SCHEMES: tuple[str, ...] = (
    "bearer",
    "query_param",
    "header_token",
)


# Closed-set audit table. Order is the canonical replay order.
# Every row mirrors a key in ``build_default_dispatch_transport`` over
# in ``http_chat_transport.py`` — keep them in sync. Tests pin this.
TRANSPORT_SPECS: tuple[ProviderTransportSpec, ...] = (
    ProviderTransportSpec(
        provider="cognition",
        family="cognition",
        env_var="DEVIN_API_KEY",
        base_url="https://api.devin.ai/v1/",
        auth_scheme="bearer",
    ),
    ProviderTransportSpec(
        provider="deepseek",
        family="openai_compat",
        env_var="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com/",
        auth_scheme="bearer",
    ),
    ProviderTransportSpec(
        provider="google",
        family="google",
        env_var="GOOGLE_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/",
        auth_scheme="query_param",
    ),
    ProviderTransportSpec(
        provider="openai",
        family="openai_compat",
        env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1/",
        auth_scheme="bearer",
    ),
    ProviderTransportSpec(
        provider="xai",
        family="openai_compat",
        env_var="XAI_API_KEY",
        base_url="https://api.x.ai/v1/",
        auth_scheme="bearer",
    ),
)


def wired_provider_keys() -> tuple[str, ...]:
    """Return the closed-set provider keys, canonically sorted."""

    return tuple(sorted(s.provider for s in TRANSPORT_SPECS))


def spec_for_provider(provider: str) -> ProviderTransportSpec:
    """Return the :class:`ProviderTransportSpec` for ``provider``.

    Raises :class:`ProviderTransportError` for unknown providers — the
    same fail-loud shape as the runtime dispatcher.
    """

    if not isinstance(provider, str) or not provider:
        raise ProviderTransportError(
            f"spec_for_provider requires a non-empty str, got {provider!r}"
        )
    for spec in TRANSPORT_SPECS:
        if spec.provider == provider:
            return spec
    raise ProviderTransportError(
        f"spec_for_provider: no transport spec wired for provider {provider!r}"
    )


def provider_transport_audit() -> tuple[Mapping[str, str], ...]:
    """Return the audit projection of :data:`TRANSPORT_SPECS`.

    Canonically sorted by provider key. Each row is an immutable
    ``Mapping[str, str]`` with the spec fields plus ``"key_present"``
    set to ``"unknown"`` — the actual env-var lookup is the operator
    credential matrix's job, not this module's (B1 / no os.environ).

    Pure function — INV-15 byte-identical across runs.
    """

    rows: list[Mapping[str, str]] = []
    for spec in sorted(TRANSPORT_SPECS, key=lambda s: s.provider):
        row: dict[str, str] = {
            "provider": spec.provider,
            "family": spec.family,
            "env_var": spec.env_var,
            "base_url": spec.base_url,
            "auth_scheme": spec.auth_scheme,
        }
        rows.append(_FrozenDict(row))
    return tuple(rows)


class _FrozenDict(dict[str, str]):
    """Read-only ``dict[str, str]`` used by the audit projection."""

    __slots__ = ()

    def __setitem__(self, key: str, value: str) -> None:  # pragma: no cover
        raise TypeError("_FrozenDict is immutable")

    def __delitem__(self, key: str) -> None:  # pragma: no cover
        raise TypeError("_FrozenDict is immutable")

    def clear(self) -> None:  # pragma: no cover - signature is `clear()`
        raise TypeError("_FrozenDict is immutable")

    def pop(self, *args: object, **kwargs: object) -> str:  # pragma: no cover
        raise TypeError("_FrozenDict is immutable")

    def popitem(self) -> tuple[str, str]:  # pragma: no cover
        raise TypeError("_FrozenDict is immutable")

    def setdefault(  # pragma: no cover
        self, *args: object, **kwargs: object
    ) -> str:
        raise TypeError("_FrozenDict is immutable")

    def update(  # pragma: no cover
        self, *args: object, **kwargs: object
    ) -> None:
        raise TypeError("_FrozenDict is immutable")
