"""GOV-CP-01-PERF — precompiled decision-table tests (Wave 4 / I7 reframe).

Ensures the constant-time ``permit_operator_action`` lookup is
verdict-equivalent to the old branching implementation, that the table
is content-hashed deterministically, and that
``install_policy_table`` / ``verify_policy_table_hash`` enforce SAFE-47.
"""

from __future__ import annotations

import pytest

from core.contracts.governance import (
    OperatorAction,
    OperatorRequest,
    SystemMode,
)
from governance_engine.control_plane import (
    LedgerAuthorityWriter,
    PolicyEngine,
)
from governance_engine.control_plane.policy_engine import (
    POLICY_TABLE_HASH_KEY,
    POLICY_TABLE_INSTALLED_KIND,
    install_policy_table,
    verify_policy_table_hash,
)


def _request(
    action: OperatorAction, *, target_status: str = ""
) -> OperatorRequest:
    payload: dict[str, str] = {}
    if target_status:
        payload["target_status"] = target_status
    return OperatorRequest(
        ts_ns=1, requestor="op", action=action, payload=payload
    )


# ---------------------------------------------------------------------------
# Verdict equivalence — old branching logic vs new table
# ---------------------------------------------------------------------------


def _legacy_verdict(
    action: OperatorAction, mode: SystemMode, target_status: str = ""
) -> tuple[bool, str]:
    """Hand-rolled copy of the pre-Wave-4 branching gate."""

    if mode is SystemMode.LOCKED:
        if action is OperatorAction.REQUEST_UNLOCK:
            return True, ""
        return False, "POLICY_LOCKED"
    if action is OperatorAction.REQUEST_KILL:
        return True, ""
    if action is OperatorAction.REQUEST_PLUGIN_LIFECYCLE:
        if target_status == "ACTIVE" and mode is SystemMode.SAFE:
            return False, "POLICY_LIFECYCLE_REQUIRES_NON_SAFE"
        return True, ""
    if action is OperatorAction.REQUEST_MODE:
        return True, ""
    if action is OperatorAction.REQUEST_INTENT:
        return True, ""
    if action is OperatorAction.REQUEST_UNLOCK:
        return False, "POLICY_NOT_LOCKED"
    return False, "POLICY_UNKNOWN_ACTION"


_LIFECYCLE_PROBES = ("ACTIVE", "SHADOW", "DISABLED", "")


@pytest.mark.parametrize("mode", list(SystemMode))
@pytest.mark.parametrize("action", list(OperatorAction))
def test_table_matches_legacy_verdict_wildcard(
    mode: SystemMode, action: OperatorAction
) -> None:
    """For every (action, mode) the table agrees with the legacy gate."""

    policy = PolicyEngine()
    got = policy.permit_operator_action(_request(action), mode)
    expected = _legacy_verdict(action, mode)
    assert got == expected, (mode, action, got, expected)


@pytest.mark.parametrize("mode", list(SystemMode))
@pytest.mark.parametrize("target_status", _LIFECYCLE_PROBES)
def test_table_matches_legacy_verdict_lifecycle_payload(
    mode: SystemMode, target_status: str
) -> None:
    """REQUEST_PLUGIN_LIFECYCLE branches on ``target_status``."""

    policy = PolicyEngine()
    got = policy.permit_operator_action(
        _request(
            OperatorAction.REQUEST_PLUGIN_LIFECYCLE,
            target_status=target_status,
        ),
        mode,
    )
    expected = _legacy_verdict(
        OperatorAction.REQUEST_PLUGIN_LIFECYCLE, mode, target_status
    )
    assert got == expected, (mode, target_status, got, expected)


# ---------------------------------------------------------------------------
# Determinism + content hash
# ---------------------------------------------------------------------------


def test_decision_table_is_deterministic_across_instances():
    a = PolicyEngine()
    b = PolicyEngine()
    assert dict(a.decision_table) == dict(b.decision_table)
    assert a.table_hash == b.table_hash


def test_decision_table_hash_is_sha256_hex():
    h = PolicyEngine().table_hash
    assert isinstance(h, str)
    assert len(h) == 64
    int(h, 16)  # parses as hex


def test_decision_table_covers_full_action_mode_grid():
    table = PolicyEngine().decision_table
    for mode in SystemMode:
        for action in OperatorAction:
            assert (action, mode, "*") in table, (action, mode)


def test_decision_table_includes_lifecycle_active_overrides():
    table = PolicyEngine().decision_table
    for mode in SystemMode:
        assert (
            OperatorAction.REQUEST_PLUGIN_LIFECYCLE,
            mode,
            "ACTIVE",
        ) in table


# ---------------------------------------------------------------------------
# install_policy_table / verify_policy_table_hash (SAFE-47)
# ---------------------------------------------------------------------------


def test_install_policy_table_writes_one_row_with_hash():
    policy = PolicyEngine()
    ledger = LedgerAuthorityWriter()
    entry = install_policy_table(policy, ledger, ts_ns=42)

    assert entry.kind == POLICY_TABLE_INSTALLED_KIND
    assert entry.payload[POLICY_TABLE_HASH_KEY] == policy.table_hash
    rows = list(ledger.read())
    assert len(rows) == 1
    assert rows[0].seq == entry.seq


def test_verify_policy_table_hash_passes_for_matching_install():
    policy = PolicyEngine()
    ledger = LedgerAuthorityWriter()
    install_policy_table(policy, ledger, ts_ns=1)
    # No exception → success.
    verify_policy_table_hash(policy, ledger)


def test_verify_policy_table_hash_raises_when_no_install_row():
    policy = PolicyEngine()
    ledger = LedgerAuthorityWriter()
    with pytest.raises(LookupError):
        verify_policy_table_hash(policy, ledger)


def test_verify_policy_table_hash_raises_on_mismatch():
    policy = PolicyEngine()
    ledger = LedgerAuthorityWriter()
    ledger.append(
        ts_ns=1,
        kind=POLICY_TABLE_INSTALLED_KIND,
        payload={POLICY_TABLE_HASH_KEY: "deadbeef" * 8},
    )
    with pytest.raises(RuntimeError):
        verify_policy_table_hash(policy, ledger)


def test_verify_uses_most_recent_install_row():
    """If the table is reinstalled (hash differs), the latest row wins."""

    policy = PolicyEngine()
    ledger = LedgerAuthorityWriter()
    # Older row with a stale hash.
    ledger.append(
        ts_ns=1,
        kind=POLICY_TABLE_INSTALLED_KIND,
        payload={POLICY_TABLE_HASH_KEY: "stale" * 12},
    )
    # Current row with the live hash.
    install_policy_table(policy, ledger, ts_ns=2)

    verify_policy_table_hash(policy, ledger)


# ---------------------------------------------------------------------------
# Engine boot writes the install row at seq=1
# ---------------------------------------------------------------------------


def test_governance_engine_boot_writes_policy_table_row_first():
    from governance_engine.engine import GovernanceEngine

    engine = GovernanceEngine(policy_table_installed_at_ns=7)
    rows = list(engine.ledger.read())
    assert rows, "expected boot ledger row"
    first = rows[0]
    assert first.kind == POLICY_TABLE_INSTALLED_KIND
    assert first.payload[POLICY_TABLE_HASH_KEY] == engine.policy.table_hash
    assert first.ts_ns == 7
    # And the post-construct verification helper accepts it.
    verify_policy_table_hash(engine.policy, engine.ledger)
