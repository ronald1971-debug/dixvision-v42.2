"""Pure presence check: requirement × env mapping → status.

Caller decides what env mapping to pass. Production callers pass
``os.environ``; tests pass an injected dict; an ``.env``-aware
launcher would parse ``.env`` first and merge into a dict before
calling :func:`presence_status`.

The check is intentionally minimal: only "is the env var set to a
non-empty string?" Live verification (does the key actually work?)
lands in the follow-up PR-B as ``POST /api/credentials/verify``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from system_engine.credentials.manifest import CredentialRequirement


class PresenceState(StrEnum):
    """Tri-state presence outcome for a multi-env-var requirement."""

    PRESENT = "present"
    PARTIAL = "partial"  # some env vars set, some missing
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CredentialStatus:
    """Runtime presence projection for one :class:`CredentialRequirement`.

    ``env_vars_present`` is parallel to ``requirement.env_vars`` so
    the UI can render exactly which variable name is missing.
    """

    requirement: CredentialRequirement
    env_vars_present: tuple[bool, ...]

    @property
    def state(self) -> PresenceState:
        if all(self.env_vars_present):
            return PresenceState.PRESENT
        if any(self.env_vars_present):
            return PresenceState.PARTIAL
        return PresenceState.MISSING

    @property
    def missing_env_vars(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, present in zip(
                self.requirement.env_vars,
                self.env_vars_present,
                strict=True,
            )
            if not present
        )


def _is_set(value: str | None) -> bool:
    return value is not None and value.strip() != ""


def presence_status(
    requirements: tuple[CredentialRequirement, ...],
    env: Mapping[str, str],
) -> tuple[CredentialStatus, ...]:
    """Project each requirement against ``env``.

    Pure: same inputs always produce same output. Empty-string values
    count as missing (a forgotten ``OPENAI_API_KEY=`` line in ``.env``
    should not silently report present).
    """

    return tuple(
        CredentialStatus(
            requirement=req,
            env_vars_present=tuple(
                _is_set(env.get(name)) for name in req.env_vars
            ),
        )
        for req in requirements
    )


__all__ = [
    "CredentialStatus",
    "PresenceState",
    "presence_status",
]
