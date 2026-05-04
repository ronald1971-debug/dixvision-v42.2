"""AUDIT-P0.3 — boot-time resolution of the authority-ledger path.

The harness historically fell back to an in-memory
:class:`LedgerAuthorityWriter` when ``DIXVISION_LEDGER_PATH`` was not
set. That meant a default operator deployment that forgot to set the
env var would lose **every** governance decision (mode transitions,
approvals, kill-switch hits, hazard reactions) on process restart —
the audit chain advertised by the hardening docs would silently be
inoperative.

The resolver in this module fails closed: in *production mode* the
harness refuses to boot without a SQLite ledger path. *Test mode* is
opt-in via :data:`PERMIT_EPHEMERAL_LEDGER_ENV_VAR` and is set by
``tests/conftest.py`` so the existing unit tree is untouched.

Sentinel values are matched **case-sensitive** against ``"1"`` so a
typo (``"true"``, ``"TRUE"``) does not silently downgrade the
guarantee — the same convention the rest of the harness uses for env
toggles.
"""

from __future__ import annotations

import os

LEDGER_PATH_ENV_VAR = "DIXVISION_LEDGER_PATH"
"""Path to the SQLite database backing the authority ledger."""

PERMIT_EPHEMERAL_LEDGER_ENV_VAR = "DIXVISION_PERMIT_EPHEMERAL_LEDGER"
"""Set to ``"1"`` to allow the harness to boot with an in-memory ledger.

Required for the test tree (``tests/conftest.py`` sets it at session
start) and for short-lived dev sessions where persistence is not
desired. **Never** set this in a production deployment.
"""


class EphemeralLedgerRefused(RuntimeError):
    """Raised when the harness is asked to boot without a persistent
    ledger and the operator has not explicitly opted into the
    ephemeral fallback."""


def resolve_ledger_path(env: os._Environ[str] | None = None) -> str | None:
    """Return the SQLite ledger path or :data:`None` if ephemeral is permitted.

    Behaviour:

    * ``LEDGER_PATH_ENV_VAR`` set (non-empty) → return the value.
    * unset / empty + ``PERMIT_EPHEMERAL_LEDGER_ENV_VAR == "1"`` →
      return :data:`None` (caller should construct an in-memory
      ledger).
    * unset / empty + opt-in absent → raise
      :class:`EphemeralLedgerRefused`.

    Args:
        env: Optional environment mapping for testing. Defaults to
            :data:`os.environ`.

    Raises:
        EphemeralLedgerRefused: production mode without persistence.
    """
    e = env if env is not None else os.environ
    path = (e.get(LEDGER_PATH_ENV_VAR) or "").strip()
    if path:
        return path
    if e.get(PERMIT_EPHEMERAL_LEDGER_ENV_VAR) == "1":
        return None
    raise EphemeralLedgerRefused(
        f"refusing to boot without a persistent authority ledger: set "
        f"{LEDGER_PATH_ENV_VAR}=/path/to/ledger.sqlite to enable "
        f"crash-survivable governance audit, or set "
        f"{PERMIT_EPHEMERAL_LEDGER_ENV_VAR}=1 to explicitly opt into "
        f"the in-memory fallback (NEVER do this in production — every "
        f"governance decision will be lost on restart)."
    )


__all__ = [
    "EphemeralLedgerRefused",
    "LEDGER_PATH_ENV_VAR",
    "PERMIT_EPHEMERAL_LEDGER_ENV_VAR",
    "resolve_ledger_path",
]
