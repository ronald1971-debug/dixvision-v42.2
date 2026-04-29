"""Registry-driven credential discovery (Dashboard-2026 wave-01.5).

What this package answers
-------------------------
"What API keys does the system actually need before I can flip a row
to ``enabled: true``?"

The package is *projection-only*: it reads the SCVS source registry,
joins each ``auth: required`` row against a static blueprint table
(env-var name + signup URL + free-tier flag), and returns a tuple of
:class:`CredentialRequirement` records. A second pure function then
joins those requirements against an environment mapping to produce
:class:`CredentialStatus` records (present / partial / missing).

Why a static blueprint table instead of declaring env vars in the YAML
---------------------------------------------------------------------
Two reasons:

1. Env-var names are an *operational* contract with the operator's
   shell / launcher / .env file. They are not data-source declarations.
   Mixing them into the SCVS YAML would conflate concerns and force
   every blueprint change through the SCVS schema-migration path.
2. The blueprint table is the *only* place where a provider name is
   spelled out, exactly once. Adding a new provider to the registry
   without adding a blueprint is a CI-detectable error
   (``test_credentials_manifest.py::test_every_auth_required_row_has_blueprint``)
   rather than silent skip.

What this package does NOT do
-----------------------------
- It does NOT call providers. (PR-B adds live verification behind
  ``POST /api/credentials/verify``.)
- It does NOT read or write any secret storage. The presence check
  takes an env-var mapping argument; the caller decides where the
  mapping comes from (``os.environ``, a parsed ``.env`` file, an
  injected dict for tests).
- It does NOT impose a "you must have all keys before booting" gate.
  Operators are explicitly allowed to run with zero AI keys (the
  ``enabled: false`` default state ships exactly like that).
"""

from system_engine.credentials.manifest import (
    CREDENTIAL_BLUEPRINTS,
    CredentialBlueprint,
    CredentialRequirement,
    requirements_for_registry,
)
from system_engine.credentials.status import (
    CredentialStatus,
    PresenceState,
    presence_status,
)
from system_engine.credentials.verifiers import (
    DEFAULT_TIMEOUT_S,
    VERIFIERS,
    VerifierSpec,
    VerifyOutcome,
    VerifyResult,
    verify_provider,
)

__all__ = [
    "CREDENTIAL_BLUEPRINTS",
    "CredentialBlueprint",
    "CredentialRequirement",
    "CredentialStatus",
    "DEFAULT_TIMEOUT_S",
    "PresenceState",
    "VERIFIERS",
    "VerifierSpec",
    "VerifyOutcome",
    "VerifyResult",
    "presence_status",
    "requirements_for_registry",
    "verify_provider",
]
