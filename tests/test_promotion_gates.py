"""Tests for hash-anchored promotion gates (Reviewer #4 finding 4).

Covers:

* SHA-256 file hash is deterministic across two readers of the same bytes.
* ``bind`` writes exactly one ``PROMOTION_GATES_BOUND`` row and caches
  the hash in memory.
* ``check`` returns ``(True, "")`` for non-gated targets unconditionally.
* ``check`` returns ``(False, "PROMOTION_GATES_NOT_BOUND")`` for a gated
  target if ``bind`` was never called.
* ``check`` returns ``(False, "PROMOTION_GATES_FILE_MISSING")`` if the
  file is deleted after binding.
* ``check`` returns ``(False, "PROMOTION_GATES_HASH_MISMATCH")`` if the
  file is edited after binding.
* ``replay_from_ledger`` recovers the bound hash from the ledger.
* :class:`StateTransitionManager` honours the gate: SHADOW entry binds,
  CANARY entry refuses on mismatch, de-escalation works, re-entry to
  SHADOW resets the bound hash.
* The integration is opt-in: a manager constructed without a
  :class:`PromotionGates` behaves identically to today (no gate check).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.contracts.governance import ModeTransitionRequest, SystemMode
from governance_engine.control_plane import (
    LedgerAuthorityWriter,
    PolicyEngine,
    StateTransitionManager,
)
from governance_engine.control_plane.promotion_gates import (
    LEDGER_KIND_PROMOTION_GATES_BOUND,
    PromotionGates,
    compute_file_hash,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gates_path(tmp_path: Path) -> Path:
    p = tmp_path / "promotion_gates.yaml"
    p.write_bytes(b"schema_version: 1\nshadow_to_canary: {sharpe_min: 1.0}\n")
    return p


# ---------------------------------------------------------------------------
# compute_file_hash
# ---------------------------------------------------------------------------


def test_compute_file_hash_deterministic(gates_path: Path) -> None:
    h1 = compute_file_hash(gates_path)
    h2 = compute_file_hash(gates_path)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_compute_file_hash_changes_on_byte_edit(gates_path: Path) -> None:
    h1 = compute_file_hash(gates_path)
    gates_path.write_bytes(gates_path.read_bytes() + b"# trailing comment\n")
    h2 = compute_file_hash(gates_path)
    assert h1 != h2


# ---------------------------------------------------------------------------
# PromotionGates.bind / check
# ---------------------------------------------------------------------------


def test_bind_writes_one_ledger_row_and_caches_hash(gates_path: Path) -> None:
    ledger = LedgerAuthorityWriter()
    gates = PromotionGates(ledger=ledger, path=gates_path)

    bound = gates.bind(ts_ns=100, requestor="op")

    assert gates.bound_hash() == bound
    rows = ledger.read()
    assert len(rows) == 1
    assert rows[0].kind == LEDGER_KIND_PROMOTION_GATES_BOUND
    assert rows[0].payload["promotion_gates_sha256"] == bound
    assert rows[0].payload["requestor"] == "op"


def test_check_passes_for_non_gated_targets(gates_path: Path) -> None:
    ledger = LedgerAuthorityWriter()
    gates = PromotionGates(ledger=ledger, path=gates_path)

    # Bind not called yet, but non-gated targets pass unconditionally.
    for target in ("SAFE", "PAPER", "SHADOW", "LOCKED"):
        ok, code = gates.check(target)
        assert ok is True
        assert code == ""


@pytest.mark.parametrize("target", ["CANARY", "LIVE", "AUTO"])
def test_check_refuses_gated_target_when_not_bound(
    gates_path: Path, target: str
) -> None:
    ledger = LedgerAuthorityWriter()
    gates = PromotionGates(ledger=ledger, path=gates_path)

    ok, code = gates.check(target)
    assert ok is False
    assert code == "PROMOTION_GATES_NOT_BOUND"


def test_check_refuses_when_file_deleted_after_bind(gates_path: Path) -> None:
    ledger = LedgerAuthorityWriter()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    gates.bind(ts_ns=100, requestor="op")

    gates_path.unlink()

    ok, code = gates.check("CANARY")
    assert ok is False
    assert code == "PROMOTION_GATES_FILE_MISSING"


def test_check_refuses_on_hash_mismatch(gates_path: Path) -> None:
    ledger = LedgerAuthorityWriter()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    gates.bind(ts_ns=100, requestor="op")

    # Mid-window edit -- adding even a comment must trip the gate.
    gates_path.write_bytes(
        gates_path.read_bytes() + b"# operator changed sharpe min\n"
    )

    ok, code = gates.check("CANARY")
    assert ok is False
    assert code == "PROMOTION_GATES_HASH_MISMATCH"


def test_rebind_overwrites_cached_hash(gates_path: Path) -> None:
    ledger = LedgerAuthorityWriter()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    h1 = gates.bind(ts_ns=100, requestor="op")

    gates_path.write_bytes(b"schema_version: 1\nshadow_to_canary: {}\n")
    h2 = gates.bind(ts_ns=200, requestor="op")

    assert h1 != h2
    assert gates.bound_hash() == h2

    ok, code = gates.check("CANARY")
    assert (ok, code) == (True, "")  # live now matches new bound hash


# ---------------------------------------------------------------------------
# replay_from_ledger
# ---------------------------------------------------------------------------


def test_replay_picks_up_most_recent_bound_hash(gates_path: Path) -> None:
    ledger = LedgerAuthorityWriter()
    g1 = PromotionGates(ledger=ledger, path=gates_path)
    g1.bind(ts_ns=100, requestor="op")

    gates_path.write_bytes(b"schema_version: 1\nshadow_to_canary: {x: 1}\n")
    h2 = g1.bind(ts_ns=200, requestor="op")

    # Fresh adapter, same ledger -- should adopt the most recent bound hash.
    g2 = PromotionGates(ledger=ledger, path=gates_path)
    assert g2.bound_hash() is None
    g2.replay_from_ledger()
    assert g2.bound_hash() == h2

    ok, code = g2.check("CANARY")
    assert (ok, code) == (True, "")


# ---------------------------------------------------------------------------
# StateTransitionManager integration
# ---------------------------------------------------------------------------


def _ratchet_to_shadow(state: StateTransitionManager, ts_ns: int = 1) -> None:
    """Drive the FSM SAFE -> PAPER -> SHADOW. Asserts each step approves."""

    for prev_mode, target_mode in (
        (SystemMode.SAFE, SystemMode.PAPER),
        (SystemMode.PAPER, SystemMode.SHADOW),
    ):
        decision = state.propose(
            ModeTransitionRequest(
                ts_ns=ts_ns,
                requestor="op",
                current_mode=prev_mode,
                target_mode=target_mode,
                reason="ratchet",
            )
        )
        assert decision.approved is True
        ts_ns += 1


def test_state_manager_without_gates_behaves_unchanged(gates_path: Path) -> None:
    """Backwards compatibility: no PromotionGates, no gate check."""

    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    state = StateTransitionManager(policy=policy, ledger=ledger)

    _ratchet_to_shadow(state)
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=10,
            requestor="op",
            current_mode=SystemMode.SHADOW,
            target_mode=SystemMode.CANARY,
            reason="promote",
        )
    )
    assert decision.approved is True
    assert state.current_mode() is SystemMode.CANARY


def test_state_manager_binds_hash_on_shadow_entry(gates_path: Path) -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    state = StateTransitionManager(
        policy=policy, ledger=ledger, promotion_gates=gates
    )

    assert gates.bound_hash() is None
    _ratchet_to_shadow(state)
    assert gates.bound_hash() == compute_file_hash(gates_path)

    bound_rows = [
        r for r in ledger.read() if r.kind == LEDGER_KIND_PROMOTION_GATES_BOUND
    ]
    assert len(bound_rows) == 1


def test_state_manager_refuses_canary_on_hash_mismatch(
    gates_path: Path,
) -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    state = StateTransitionManager(
        policy=policy, ledger=ledger, promotion_gates=gates
    )

    _ratchet_to_shadow(state)
    gates_path.write_bytes(
        gates_path.read_bytes() + b"# operator nudged thresholds\n"
    )

    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=99,
            requestor="op",
            current_mode=SystemMode.SHADOW,
            target_mode=SystemMode.CANARY,
            reason="promote",
        )
    )
    assert decision.approved is False
    assert decision.rejection_code == "PROMOTION_GATES_HASH_MISMATCH"
    assert state.current_mode() is SystemMode.SHADOW

    rejected_rows = [
        r for r in ledger.read() if r.kind == "MODE_TRANSITION_REJECTED"
    ]
    assert any(
        r.payload["rejection_code"] == "PROMOTION_GATES_HASH_MISMATCH"
        for r in rejected_rows
    )


def test_state_manager_allows_canary_when_hash_matches(
    gates_path: Path,
) -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    state = StateTransitionManager(
        policy=policy, ledger=ledger, promotion_gates=gates
    )

    _ratchet_to_shadow(state)

    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=99,
            requestor="op",
            current_mode=SystemMode.SHADOW,
            target_mode=SystemMode.CANARY,
            reason="promote",
        )
    )
    assert decision.approved is True
    assert state.current_mode() is SystemMode.CANARY


def test_state_manager_reentry_to_shadow_rebinds_hash(
    gates_path: Path,
) -> None:
    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    state = StateTransitionManager(
        policy=policy, ledger=ledger, promotion_gates=gates
    )

    _ratchet_to_shadow(state)
    h1 = gates.bound_hash()
    assert h1 is not None

    # De-escalate to PAPER (legal backward step).
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=50,
            requestor="op",
            current_mode=SystemMode.SHADOW,
            target_mode=SystemMode.PAPER,
            reason="de-escalate to edit gates",
        )
    )
    assert decision.approved is True

    # Operator edits the gates file...
    gates_path.write_bytes(
        gates_path.read_bytes() + b"# new sharpe floor\n"
    )

    # ...and re-enters SHADOW. The bind must rerun on the new bytes.
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=60,
            requestor="op",
            current_mode=SystemMode.PAPER,
            target_mode=SystemMode.SHADOW,
            reason="restart shadow clock",
        )
    )
    assert decision.approved is True

    h2 = gates.bound_hash()
    assert h2 is not None and h2 != h1
    assert h2 == compute_file_hash(gates_path)

    # CANARY now passes against the fresh bound hash.
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=70,
            requestor="op",
            current_mode=SystemMode.SHADOW,
            target_mode=SystemMode.CANARY,
            reason="promote",
        )
    )
    assert decision.approved is True


def test_state_manager_de_escalation_does_not_require_gate(
    gates_path: Path,
) -> None:
    """Backward edges (e.g. CANARY -> SHADOW) must remain free."""

    ledger = LedgerAuthorityWriter()
    policy = PolicyEngine()
    gates = PromotionGates(ledger=ledger, path=gates_path)
    state = StateTransitionManager(
        policy=policy, ledger=ledger, promotion_gates=gates
    )

    _ratchet_to_shadow(state)
    state.propose(
        ModeTransitionRequest(
            ts_ns=20,
            requestor="op",
            current_mode=SystemMode.SHADOW,
            target_mode=SystemMode.CANARY,
            reason="promote",
        )
    )

    # Now break the file -- de-escalation must still work because backward
    # edges aren't gated. Only forward CANARY/LIVE/AUTO entries are gated.
    gates_path.write_bytes(b"# corrupted\n")
    decision = state.propose(
        ModeTransitionRequest(
            ts_ns=30,
            requestor="op",
            current_mode=SystemMode.CANARY,
            target_mode=SystemMode.SHADOW,
            reason="de-escalate",
        )
    )
    assert decision.approved is True
    assert state.current_mode() is SystemMode.SHADOW


def test_real_repo_promotion_gates_yaml_is_loadable() -> None:
    """The shipped ``docs/promotion_gates.yaml`` must hash and parse."""

    import yaml

    repo_root = Path(__file__).resolve().parent.parent
    real_path = repo_root / "docs" / "promotion_gates.yaml"
    assert real_path.exists(), "shipped promotion_gates.yaml is missing"

    digest = compute_file_hash(real_path)
    assert len(digest) == 64

    # Schema sanity: mandatory top-level sections present.
    with real_path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    for section in ("shadow_to_canary", "canary_to_live", "live_to_auto"):
        assert section in doc, f"missing section {section}"
        assert "performance" in doc[section] or "drift_oracle" in doc[section]
