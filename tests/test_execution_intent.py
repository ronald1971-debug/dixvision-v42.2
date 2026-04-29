"""HARDEN-01 — ``ExecutionIntent`` frozen currency + B25 lint coverage."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from core.contracts.events import Side, SignalEvent
from core.contracts.execution_intent import (
    AUTHORISED_INTENT_ORIGINS,
    UnauthorizedOriginError,
    compute_content_hash,
    compute_intent_id,
    create_execution_intent,
    mark_approved,
    mark_rejected,
)


def _signal(
    *,
    ts_ns: int = 1_000_000_000,
    symbol: str = "BTCUSDT",
    side: Side = Side.BUY,
    confidence: float = 0.9,
) -> SignalEvent:
    return SignalEvent(
        ts_ns=ts_ns,
        symbol=symbol,
        side=side,
        confidence=confidence,
        plugin_chain=("microstructure_v1",),
        meta={"qty": "1.0"},
    )


# ---------------------------------------------------------------------------
# Frozen / immutable currency
# ---------------------------------------------------------------------------


def test_intent_is_frozen_dataclass():
    """The dataclass is frozen — re-bindings raise FrozenInstanceError."""

    intent = create_execution_intent(
        ts_ns=10,
        origin="tests.fixtures",
        signal=_signal(),
    )
    assert dataclasses.is_dataclass(intent)
    with pytest.raises(dataclasses.FrozenInstanceError):
        intent.approved_by_governance = True  # type: ignore[misc]


def test_intent_id_and_hash_are_deterministic():
    """Two intents with the same canonical fields share id + hash."""

    sig = _signal()
    a = create_execution_intent(ts_ns=10, origin="tests.fixtures", signal=sig)
    b = create_execution_intent(ts_ns=10, origin="tests.fixtures", signal=sig)
    assert a.intent_id == b.intent_id
    assert a.content_hash == b.content_hash
    assert a == b
    # And the helpers expose the same value the dataclass carries.
    expected_hash = compute_content_hash(
        ts_ns=10,
        origin="tests.fixtures",
        signal=sig,
        approved_by_governance=False,
        governance_decision_id="",
    )
    assert a.content_hash == expected_hash
    assert a.intent_id == compute_intent_id(expected_hash)


def test_intent_hash_differs_when_any_canonical_field_changes():
    base = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    different_ts = create_execution_intent(
        ts_ns=11, origin="tests.fixtures", signal=_signal()
    )
    different_signal = create_execution_intent(
        ts_ns=10,
        origin="tests.fixtures",
        signal=_signal(symbol="ETHUSDT"),
    )
    assert base.content_hash != different_ts.content_hash
    assert base.content_hash != different_signal.content_hash


def test_intent_hash_no_meta_delimiter_collision():
    """Different meta dicts must not collide via delimiter trickery.

    Regression for Devin Review BUG_0001 on PR #78 — the original
    impl used ``";".join(f"{k}={v}")`` which collided
    ``{"a": "1", "b": "2"}`` with ``{"a": "1;b=2"}``. Canonical JSON
    serialisation closes the gap.
    """

    def _signal_with_meta(meta: dict[str, str]) -> SignalEvent:
        return SignalEvent(
            ts_ns=1_000_000_000,
            symbol="BTCUSDT",
            side=Side.BUY,
            confidence=0.9,
            plugin_chain=("microstructure_v1",),
            meta=meta,
        )

    a = create_execution_intent(
        ts_ns=10,
        origin="tests.fixtures",
        signal=_signal_with_meta({"a": "1", "b": "2"}),
    )
    b = create_execution_intent(
        ts_ns=10,
        origin="tests.fixtures",
        signal=_signal_with_meta({"a": "1;b=2"}),
    )
    assert a.content_hash != b.content_hash


def test_intent_hash_no_plugin_chain_delimiter_collision():
    """Plugin chain entries containing the delimiter must not collide."""

    def _signal_with_chain(chain: tuple[str, ...]) -> SignalEvent:
        return SignalEvent(
            ts_ns=1_000_000_000,
            symbol="BTCUSDT",
            side=Side.BUY,
            confidence=0.9,
            plugin_chain=chain,
            meta={},
        )

    a = create_execution_intent(
        ts_ns=10,
        origin="tests.fixtures",
        signal=_signal_with_chain(("a", "b")),
    )
    b = create_execution_intent(
        ts_ns=10,
        origin="tests.fixtures",
        signal=_signal_with_chain(("a|b",)),
    )
    assert a.content_hash != b.content_hash


def test_intent_verify_content_hash_round_trip():
    intent = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    assert intent.verify_content_hash() is True


def test_intent_verify_content_hash_detects_tampered_field():
    """A hand-edited frozen replacement breaks the hash check."""

    intent = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    tampered = dataclasses.replace(intent, ts_ns=999)
    assert tampered.verify_content_hash() is False


# ---------------------------------------------------------------------------
# Origin allowlist (runtime defence; B25 is the static defence)
# ---------------------------------------------------------------------------


def test_unauthorised_origin_rejected():
    with pytest.raises(UnauthorizedOriginError):
        create_execution_intent(
            ts_ns=10,
            origin="execution_engine.hot_path",  # not allowed
            signal=_signal(),
        )


def test_authorised_origins_are_intelligence_subsystems():
    """Every entry is either tests.fixtures or under intelligence_engine.*."""

    for origin in AUTHORISED_INTENT_ORIGINS:
        assert origin == "tests.fixtures" or origin.startswith(
            "intelligence_engine."
        ), f"unexpected origin in allowlist: {origin}"


# ---------------------------------------------------------------------------
# Approval state transitions are pure (return new instances)
# ---------------------------------------------------------------------------


def test_mark_approved_returns_new_instance_with_decision_id():
    proposed = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    approved = mark_approved(proposed, governance_decision_id="LEDGER-42")
    assert proposed.approved_by_governance is False
    assert approved.approved_by_governance is True
    assert approved.governance_decision_id == "LEDGER-42"
    assert approved is not proposed
    # Hash *changes* on transition — that's the whole point: the
    # downstream chokepoint can detect approval-state tampering.
    assert approved.content_hash != proposed.content_hash


def test_mark_approved_is_idempotent_for_same_decision_id():
    proposed = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    a1 = mark_approved(proposed, governance_decision_id="LEDGER-42")
    a2 = mark_approved(a1, governance_decision_id="LEDGER-42")
    assert a1 == a2


def test_mark_approved_rejects_swap_to_different_decision_id():
    proposed = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    a1 = mark_approved(proposed, governance_decision_id="LEDGER-42")
    with pytest.raises(ValueError):
        mark_approved(a1, governance_decision_id="LEDGER-99")


def test_mark_approved_requires_decision_id():
    proposed = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    with pytest.raises(ValueError):
        mark_approved(proposed, governance_decision_id="")


def test_create_with_approved_flag_requires_decision_id():
    with pytest.raises(ValueError):
        create_execution_intent(
            ts_ns=10,
            origin="tests.fixtures",
            signal=_signal(),
            approved_by_governance=True,
            governance_decision_id="",
        )


def test_mark_rejected_records_decision_id_without_approval():
    proposed = create_execution_intent(
        ts_ns=10, origin="tests.fixtures", signal=_signal()
    )
    rejected = mark_rejected(proposed, governance_decision_id="LEDGER-7")
    assert rejected.approved_by_governance is False
    assert rejected.governance_decision_id == "LEDGER-7"


# ---------------------------------------------------------------------------
# B25 lint rule — only intelligence_engine.* / governance_engine.* may call
# create_execution_intent / mark_approved / mark_rejected.
# ---------------------------------------------------------------------------


def _run_b25(source: str, importer: str) -> list:
    from tools.authority_lint import _check_b25  # type: ignore

    repo_root = Path(__file__).resolve().parent.parent
    file = repo_root / "execution_engine" / "_synthetic_lint_fixture.py"
    tree = ast.parse(source)
    return _check_b25(importer, file, repo_root, tree)


def test_b25_blocks_execution_engine_calling_factory():
    src = (
        "from core.contracts.execution_intent import create_execution_intent\n"
        "create_execution_intent(ts_ns=1, origin='x', signal=None)\n"
    )
    violations = _run_b25(src, "execution_engine.hot_path.fast_execute")
    rules = [v.rule for v in violations]
    assert rules == ["B25"]


def test_b25_blocks_dashboard_calling_factory():
    src = "create_execution_intent(ts_ns=1, origin='x', signal=None)\n"
    violations = _run_b25(src, "ui.dashboard_routes")
    assert any(v.rule == "B25" for v in violations)


def test_b25_allows_intelligence_engine_calling_factory():
    src = "create_execution_intent(ts_ns=1, origin='x', signal=None)\n"
    violations = _run_b25(src, "intelligence_engine.meta_controller")
    assert violations == []


def test_b25_allows_governance_engine_calling_mark_approved():
    src = "mark_approved(intent, governance_decision_id='X')\n"
    violations = _run_b25(
        src, "governance_engine.control_plane.bridge"
    )
    assert violations == []


def test_b25_blocks_mark_rejected_in_execution_engine():
    src = "mark_rejected(intent, governance_decision_id='X')\n"
    violations = _run_b25(src, "execution_engine.lifecycle.fsm")
    assert any(v.rule == "B25" for v in violations)
