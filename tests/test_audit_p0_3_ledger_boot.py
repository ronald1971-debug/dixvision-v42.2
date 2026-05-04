"""AUDIT-P0.3 — boot-time ledger persistence enforcement."""

from __future__ import annotations

import pytest

from ui._ledger_boot import (
    LEDGER_PATH_ENV_VAR,
    PERMIT_EPHEMERAL_LEDGER_ENV_VAR,
    EphemeralLedgerRefused,
    resolve_ledger_path,
)


def test_returns_path_when_set() -> None:
    env = {LEDGER_PATH_ENV_VAR: "/var/lib/dixvision/ledger.sqlite"}
    assert resolve_ledger_path(env) == "/var/lib/dixvision/ledger.sqlite"  # type: ignore[arg-type]


def test_returns_path_when_set_and_ephemeral_also_permitted() -> None:
    """Explicit path always wins over ephemeral opt-in."""
    env = {
        LEDGER_PATH_ENV_VAR: "/db.sqlite",
        PERMIT_EPHEMERAL_LEDGER_ENV_VAR: "1",
    }
    assert resolve_ledger_path(env) == "/db.sqlite"  # type: ignore[arg-type]


def test_returns_none_when_ephemeral_explicitly_permitted() -> None:
    env = {PERMIT_EPHEMERAL_LEDGER_ENV_VAR: "1"}
    assert resolve_ledger_path(env) is None  # type: ignore[arg-type]


def test_refuses_when_unset() -> None:
    with pytest.raises(EphemeralLedgerRefused, match="refusing to boot"):
        resolve_ledger_path({})  # type: ignore[arg-type]


def test_refuses_when_path_empty_string() -> None:
    with pytest.raises(EphemeralLedgerRefused):
        resolve_ledger_path({LEDGER_PATH_ENV_VAR: ""})  # type: ignore[arg-type]


def test_refuses_when_path_whitespace_only() -> None:
    with pytest.raises(EphemeralLedgerRefused):
        resolve_ledger_path({LEDGER_PATH_ENV_VAR: "   "})  # type: ignore[arg-type]


def test_refuses_with_typo_in_ephemeral_flag() -> None:
    """``PERMIT_EPHEMERAL_LEDGER`` must match ``"1"`` exactly — typos
    like ``"true"`` / ``"TRUE"`` / ``"yes"`` must not silently
    downgrade the persistence guarantee."""
    for v in ("true", "TRUE", "yes", "1 ", "True", "0", " 1"):
        with pytest.raises(EphemeralLedgerRefused):
            resolve_ledger_path(
                {PERMIT_EPHEMERAL_LEDGER_ENV_VAR: v}  # type: ignore[arg-type]
            )


def test_error_message_references_both_env_vars() -> None:
    with pytest.raises(EphemeralLedgerRefused) as exc_info:
        resolve_ledger_path({})  # type: ignore[arg-type]
    msg = str(exc_info.value)
    assert LEDGER_PATH_ENV_VAR in msg
    assert PERMIT_EPHEMERAL_LEDGER_ENV_VAR in msg
    assert ".sqlite" in msg  # operator-actionable hint
