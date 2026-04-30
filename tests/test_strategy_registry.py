"""Tests for Wave-04.6 PR-D — governance-side StrategyRegistry."""

from __future__ import annotations

import pytest

from core.contracts.strategy_registry import (
    LEGAL_LIFECYCLE_TRANSITIONS,
    StrategyLifecycle,
    StrategyLifecycleError,
    StrategyRecord,
    is_legal_transition,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.strategy_registry import (
    LEDGER_KIND_STRATEGY_LIFECYCLE,
    StrategyRegistry,
)


# ---------------------------------------------------------------------------
# Pure contract tests
# ---------------------------------------------------------------------------


def test_legal_transitions_are_forward_only():
    """No edge in the FSM points 'backwards' to a less-mature state."""
    rank = {
        StrategyLifecycle.DRAFT: 0,
        StrategyLifecycle.VALIDATING: 1,
        StrategyLifecycle.APPROVED: 2,
        StrategyLifecycle.RETIRED: 3,
    }
    for prev, allowed in LEGAL_LIFECYCLE_TRANSITIONS.items():
        for new in allowed:
            assert rank[new] > rank[prev], (
                f"{prev.value} → {new.value} is not forward-only"
            )


def test_retired_is_terminal():
    assert LEGAL_LIFECYCLE_TRANSITIONS[StrategyLifecycle.RETIRED] == frozenset()


def test_is_legal_transition_matches_table():
    for prev, allowed in LEGAL_LIFECYCLE_TRANSITIONS.items():
        for new in StrategyLifecycle:
            expected = new in allowed
            assert is_legal_transition(prev=prev, new=new) is expected


def test_strategy_record_is_frozen():
    record = StrategyRecord(
        strategy_id="s1",
        version=1,
        lifecycle=StrategyLifecycle.DRAFT,
    )
    with pytest.raises(Exception):  # noqa: B017 — frozen dataclass raises FrozenInstanceError or AttributeError
        record.version = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Registry happy paths
# ---------------------------------------------------------------------------


def _make_registry() -> StrategyRegistry:
    return StrategyRegistry(ledger=LedgerAuthorityWriter())


def test_register_draft_creates_record_in_draft():
    reg = _make_registry()
    record = reg.register_draft(
        strategy_id="s1",
        ts_ns=1_000,
        parameters={"alpha": "0.5"},
        composed_from=("c1", "c2"),
        why=("why-1",),
    )
    assert record.strategy_id == "s1"
    assert record.version == 1
    assert record.lifecycle is StrategyLifecycle.DRAFT
    assert record.parameters == {"alpha": "0.5"}
    assert record.composed_from == ("c1", "c2")
    assert record.why == ("why-1",)
    assert record.created_ts_ns == 1_000
    assert record.last_transition_ts_ns == 1_000


def test_register_draft_writes_one_ledger_row():
    ledger = LedgerAuthorityWriter()
    reg = StrategyRegistry(ledger=ledger)
    reg.register_draft(strategy_id="s1", ts_ns=1)
    rows = ledger.read()
    assert len(rows) == 1
    assert rows[0].kind == LEDGER_KIND_STRATEGY_LIFECYCLE
    assert rows[0].payload["strategy_id"] == "s1"
    assert rows[0].payload["lifecycle"] == "DRAFT"
    assert rows[0].payload["version"] == "1"


def test_register_draft_rejects_empty_strategy_id():
    reg = _make_registry()
    with pytest.raises(ValueError):
        reg.register_draft(strategy_id="", ts_ns=1)


def test_register_draft_rejects_duplicate():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    with pytest.raises(ValueError):
        reg.register_draft(strategy_id="s1", ts_ns=2)


def test_full_happy_path_draft_to_approved_to_retired():
    ledger = LedgerAuthorityWriter()
    reg = StrategyRegistry(ledger=ledger)
    reg.register_draft(strategy_id="s1", ts_ns=1)

    r2 = reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=2,
        reason="composer ready",
    )
    assert r2.version == 2
    assert r2.lifecycle is StrategyLifecycle.VALIDATING
    assert r2.last_transition_ts_ns == 2

    r3 = reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.APPROVED,
        ts_ns=3,
        reason="passed shadow",
    )
    assert r3.version == 3
    assert r3.lifecycle is StrategyLifecycle.APPROVED

    r4 = reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.RETIRED,
        ts_ns=4,
        reason="operator retire",
    )
    assert r4.version == 4
    assert r4.lifecycle is StrategyLifecycle.RETIRED

    rows = ledger.read()
    assert [row.payload["lifecycle"] for row in rows] == [
        "DRAFT",
        "VALIDATING",
        "APPROVED",
        "RETIRED",
    ]


def test_validating_can_short_circuit_to_retired():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=2,
        reason="start",
    )
    record = reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.RETIRED,
        ts_ns=3,
        reason="validation failed",
    )
    assert record.lifecycle is StrategyLifecycle.RETIRED


def test_draft_can_short_circuit_to_retired():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    record = reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.RETIRED,
        ts_ns=2,
        reason="composer withdrew",
    )
    assert record.lifecycle is StrategyLifecycle.RETIRED


# ---------------------------------------------------------------------------
# Registry guardrails
# ---------------------------------------------------------------------------


def test_transition_unknown_strategy_raises_keyerror():
    reg = _make_registry()
    with pytest.raises(KeyError):
        reg.transition(
            strategy_id="nope",
            new_lifecycle=StrategyLifecycle.VALIDATING,
            ts_ns=1,
            reason="x",
        )


def test_transition_requires_reason():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    with pytest.raises(ValueError):
        reg.transition(
            strategy_id="s1",
            new_lifecycle=StrategyLifecycle.VALIDATING,
            ts_ns=2,
            reason="",
        )


def test_illegal_skip_draft_to_approved_raises():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    with pytest.raises(StrategyLifecycleError):
        reg.transition(
            strategy_id="s1",
            new_lifecycle=StrategyLifecycle.APPROVED,
            ts_ns=2,
            reason="skip",
        )


def test_illegal_retired_to_validating_raises():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.RETIRED,
        ts_ns=2,
        reason="abort",
    )
    with pytest.raises(StrategyLifecycleError):
        reg.transition(
            strategy_id="s1",
            new_lifecycle=StrategyLifecycle.VALIDATING,
            ts_ns=3,
            reason="resurrect",
        )


def test_illegal_transition_does_not_mutate_state_or_ledger():
    ledger = LedgerAuthorityWriter()
    reg = StrategyRegistry(ledger=ledger)
    reg.register_draft(strategy_id="s1", ts_ns=1)
    rows_before = len(ledger.read())
    record_before = reg.get("s1")
    with pytest.raises(StrategyLifecycleError):
        reg.transition(
            strategy_id="s1",
            new_lifecycle=StrategyLifecycle.APPROVED,
            ts_ns=2,
            reason="skip",
        )
    # neither the record nor the ledger advanced
    assert reg.get("s1") == record_before
    assert len(ledger.read()) == rows_before


# ---------------------------------------------------------------------------
# Replay determinism
# ---------------------------------------------------------------------------


def test_replay_rebuilds_identical_registry():
    """A replay-from-ledger registry must agree with the live one."""
    ledger = LedgerAuthorityWriter()
    live = StrategyRegistry(ledger=ledger)
    live.register_draft(
        strategy_id="s1",
        ts_ns=1,
        parameters={"alpha": "0.5", "beta": "0.25"},
        composed_from=("c1", "c2"),
        why=("why-1", "why-2"),
    )
    live.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=2,
        reason="composer ready",
    )
    live.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.APPROVED,
        ts_ns=3,
        reason="passed shadow",
    )
    live.register_draft(
        strategy_id="s2", ts_ns=4, parameters={"gamma": "1.0"}
    )

    replay = StrategyRegistry(ledger=LedgerAuthorityWriter())
    replay.replay_from_ledger(ledger.read())

    assert replay.get("s1") == live.get("s1")
    assert replay.get("s2") == live.get("s2")
    assert len(replay) == len(live)


def test_replay_ignores_unrelated_ledger_rows():
    """Replay must be composable with other ledger row kinds."""
    ledger = LedgerAuthorityWriter()
    ledger.append(
        ts_ns=1,
        kind="MODE_TRANSITION",
        payload={"prev": "SAFE", "new": "PAPER"},
    )
    reg = StrategyRegistry(ledger=ledger)
    reg.register_draft(strategy_id="s1", ts_ns=2)
    ledger.append(
        ts_ns=3,
        kind="OPERATOR_REJECTED",
        payload={"intent_id": "abc"},
    )
    reg.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=4,
        reason="x",
    )

    replay = StrategyRegistry(ledger=LedgerAuthorityWriter())
    replay.replay_from_ledger(ledger.read())
    assert replay.get("s1") == reg.get("s1")


def test_replay_is_idempotent():
    """Calling replay twice must produce the same registry."""
    ledger = LedgerAuthorityWriter()
    reg_a = StrategyRegistry(ledger=ledger)
    reg_a.register_draft(strategy_id="s1", ts_ns=1)
    reg_a.transition(
        strategy_id="s1",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=2,
        reason="x",
    )

    reg_b = StrategyRegistry(ledger=LedgerAuthorityWriter())
    reg_b.replay_from_ledger(ledger.read())
    snapshot_first = dict(reg_b._records)  # type: ignore[attr-defined]
    reg_b.replay_from_ledger(ledger.read())
    snapshot_second = dict(reg_b._records)  # type: ignore[attr-defined]
    assert snapshot_first == snapshot_second


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_all_in_groups_records_by_lifecycle():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    reg.register_draft(strategy_id="s2", ts_ns=2)
    reg.transition(
        strategy_id="s2",
        new_lifecycle=StrategyLifecycle.VALIDATING,
        ts_ns=3,
        reason="x",
    )
    drafts = reg.all_in(StrategyLifecycle.DRAFT)
    validating = reg.all_in(StrategyLifecycle.VALIDATING)
    assert {r.strategy_id for r in drafts} == {"s1"}
    assert {r.strategy_id for r in validating} == {"s2"}


def test_contains_membership():
    reg = _make_registry()
    reg.register_draft(strategy_id="s1", ts_ns=1)
    assert "s1" in reg
    assert "s2" not in reg
