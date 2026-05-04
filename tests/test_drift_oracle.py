"""GOV-CP-08 / P0-7 — drift composite oracle + auto-downgrade.

The oracle is a pure projection (``observe``) plus a side-effecting
``evaluate_and_downgrade`` that routes through
:class:`StateTransitionManager`. These tests cover both halves and
the boundary cases for the drift-driven downgrade chain
``AUTO -> LIVE -> CANARY -> PAPER``.

SHADOW-DEMOLITION-02 collapsed system-mode SHADOW into PAPER, so the
lowest drift-driven downgrade target is now PAPER.
"""

from __future__ import annotations

from core.contracts.governance import (
    ModeTransitionRequest,
    SystemMode,
)
from core.contracts.operator_consent import (
    OperatorConsent,
    edge_requires_consent,
)
from governance_engine.control_plane.drift_oracle import (
    DEFAULT_DOWNGRADE_THRESHOLD,
    DriftComponentReading,
    DriftCompositeOracle,
)
from governance_engine.control_plane.ledger_authority_writer import (
    LedgerAuthorityWriter,
)
from governance_engine.control_plane.policy_engine import PolicyEngine
from governance_engine.control_plane.state_transition_manager import (
    StateTransitionManager,
)


def _stm(initial: SystemMode = SystemMode.SAFE) -> StateTransitionManager:
    return StateTransitionManager(
        policy=PolicyEngine(),
        ledger=LedgerAuthorityWriter(),
        initial_mode=initial,
    )


def _ratchet_to(
    stm: StateTransitionManager, target: SystemMode, *, operator: bool = True
) -> None:
    """Forward-ratchet the FSM step-by-step (legality requires it)."""

    chain = (
        SystemMode.SAFE,
        SystemMode.PAPER,
        SystemMode.CANARY,
        SystemMode.LIVE,
        SystemMode.AUTO,
    )
    target_idx = chain.index(target)
    for nxt in chain[1 : target_idx + 1]:
        ts_ns = int(nxt.value) * 10
        prev = stm.current_mode()
        consent = None
        if edge_requires_consent(prev, nxt):
            # Hardening-S1 item 8 — SAFE→PAPER and LIVE→AUTO require
            # a typed consent envelope. Bind to the live policy hash
            # so the validator accepts it.
            consent = OperatorConsent(
                ts_ns=ts_ns,
                operator_id="bringup",
                mode_from=prev,
                mode_to=nxt,
                policy_hash=stm._policy.table_hash,
                nonce=f"ratchet-{prev.name}-{nxt.name}-{ts_ns}",
            )
        decision = stm.propose(
            ModeTransitionRequest(
                ts_ns=ts_ns,
                requestor="bringup",
                current_mode=prev,
                target_mode=nxt,
                reason="ratchet for test",
                operator_authorized=operator,
                consent=consent,
            )
        )
        assert decision.approved, (nxt, decision.rejection_code)


def _readings(
    *,
    model: float = 0.0,
    exec_: float = 0.0,
    latency: float = 0.0,
    causal: float = 0.0,
) -> tuple[DriftComponentReading, ...]:
    return (
        DriftComponentReading(component_id="model", deviation=model),
        DriftComponentReading(component_id="exec", deviation=exec_),
        DriftComponentReading(component_id="latency", deviation=latency),
        DriftComponentReading(component_id="causal", deviation=causal),
    )


# ---------------------------------------------------------------------------
# Pure projection
# ---------------------------------------------------------------------------


def test_observe_quiet_window_is_not_breaching() -> None:
    oracle = DriftCompositeOracle()
    reading = oracle.observe(
        ts_ns=1, readings=_readings(model=0.05, exec_=0.10)
    )
    assert reading.composite == 0.10
    assert reading.breached_components == ()
    assert reading.is_breaching is False


def test_observe_max_component_drives_composite() -> None:
    oracle = DriftCompositeOracle()
    reading = oracle.observe(
        ts_ns=1, readings=_readings(model=0.05, exec_=0.50, latency=0.20)
    )
    assert reading.composite == 0.50
    assert reading.breached_components == ("exec",)
    assert reading.is_breaching is True


def test_observe_multiple_breaches_listed_in_input_order() -> None:
    oracle = DriftCompositeOracle()
    reading = oracle.observe(
        ts_ns=1,
        readings=_readings(model=0.40, exec_=0.10, latency=0.30, causal=0.60),
    )
    assert reading.breached_components == ("model", "latency", "causal")


def test_observe_empty_readings_returns_zero_composite() -> None:
    oracle = DriftCompositeOracle()
    reading = oracle.observe(ts_ns=1, readings=())
    assert reading.composite == 0.0
    assert reading.is_breaching is False


def test_default_threshold_matches_dashboard_constant() -> None:
    # Dashboard ``ui/governance_routes.py`` advertises 0.25; a drift
    # higher than this is what the operator panel displays as "red".
    assert DEFAULT_DOWNGRADE_THRESHOLD == 0.25


# ---------------------------------------------------------------------------
# Auto-downgrade — routes through StateTransitionManager (B32)
# ---------------------------------------------------------------------------


def test_auto_breach_downgrades_one_step_to_live() -> None:
    stm = _stm()
    _ratchet_to(stm, SystemMode.AUTO)
    assert stm.current_mode() is SystemMode.AUTO

    oracle = DriftCompositeOracle()
    reading, decision = oracle.evaluate_and_downgrade(
        ts_ns=10_000_000_000,
        readings=_readings(model=0.40),
        stm=stm,
    )
    assert reading.is_breaching is True
    assert decision is not None
    assert decision.approved is True
    assert decision.prev_mode is SystemMode.AUTO
    assert decision.new_mode is SystemMode.LIVE
    assert "drift_composite_breach" in decision.reason
    assert "model" in decision.reason
    assert stm.current_mode() is SystemMode.LIVE


def test_live_breach_downgrades_to_canary() -> None:
    stm = _stm()
    _ratchet_to(stm, SystemMode.LIVE)

    oracle = DriftCompositeOracle()
    _, decision = oracle.evaluate_and_downgrade(
        ts_ns=10_000_000_000,
        readings=_readings(latency=0.90),
        stm=stm,
    )
    assert decision is not None
    assert decision.approved is True
    assert decision.prev_mode is SystemMode.LIVE
    assert decision.new_mode is SystemMode.CANARY
    assert stm.current_mode() is SystemMode.CANARY


def test_canary_breach_downgrades_to_paper() -> None:
    stm = _stm()
    _ratchet_to(stm, SystemMode.CANARY)

    oracle = DriftCompositeOracle()
    _, decision = oracle.evaluate_and_downgrade(
        ts_ns=10_000_000_000,
        readings=_readings(causal=0.40),
        stm=stm,
    )
    assert decision is not None
    assert decision.approved is True
    assert decision.prev_mode is SystemMode.CANARY
    assert decision.new_mode is SystemMode.PAPER
    assert stm.current_mode() is SystemMode.PAPER


def test_paper_breach_is_no_op() -> None:
    """PAPER is the safe floor; nothing to downgrade.

    SHADOW-DEMOLITION-02 collapsed SHADOW into PAPER; PAPER is now
    both ``signals-on, execution-off``-style governed and the lowest
    drift-driven target.
    """

    stm = _stm()
    _ratchet_to(stm, SystemMode.PAPER, operator=False)

    oracle = DriftCompositeOracle()
    reading, decision = oracle.evaluate_and_downgrade(
        ts_ns=10_000_000_000,
        readings=_readings(model=0.99),
        stm=stm,
    )
    assert reading.is_breaching is True
    assert decision is None
    assert stm.current_mode() is SystemMode.PAPER


def test_quiet_window_does_not_propose_transition() -> None:
    stm = _stm()
    _ratchet_to(stm, SystemMode.LIVE)

    oracle = DriftCompositeOracle()
    reading, decision = oracle.evaluate_and_downgrade(
        ts_ns=10_000_000_000,
        readings=_readings(model=0.05, exec_=0.10),
        stm=stm,
    )
    assert reading.is_breaching is False
    assert decision is None
    assert stm.current_mode() is SystemMode.LIVE


def test_breach_logs_mode_transition_in_authority_ledger() -> None:
    """Reviewer #4: every mode transition must hit the ledger."""

    ledger = LedgerAuthorityWriter()
    stm = StateTransitionManager(
        policy=PolicyEngine(),
        ledger=ledger,
        initial_mode=SystemMode.SAFE,
    )
    _ratchet_to(stm, SystemMode.LIVE)
    ledger_rows_before = len(ledger.read())

    oracle = DriftCompositeOracle()
    oracle.evaluate_and_downgrade(
        ts_ns=10_000_000_000,
        readings=_readings(latency=0.90),
        stm=stm,
    )

    rows = ledger.read()
    assert len(rows) == ledger_rows_before + 1
    last = rows[-1]
    assert last.kind == "MODE_TRANSITION"
    assert last.payload["new_mode"] == "CANARY"
    assert last.payload["requestor"] == "drift_oracle"
    assert "drift_composite_breach" in last.payload["reason"]


# ---------------------------------------------------------------------------
# P0-7 followup — Devin Review BUG_0001:
# `_downgrade_threshold` and `_component_ids` were stored but unused.
# These tests pin the public properties + the composite-level breach gate.
# ---------------------------------------------------------------------------


def test_expected_components_property_surfaces_canonical_set() -> None:
    oracle = DriftCompositeOracle()
    assert oracle.expected_components == ("model", "exec", "latency", "causal")

    custom = DriftCompositeOracle(component_ids=("a", "b"))
    assert custom.expected_components == ("a", "b")


def test_downgrade_threshold_property_surfaces_value() -> None:
    assert DriftCompositeOracle().downgrade_threshold == 0.25
    assert DriftCompositeOracle(downgrade_threshold=0.5).downgrade_threshold == 0.5


def test_composite_only_breach_triggers_downgrade() -> None:
    """A high composite with no per-component breach still downgrades.

    Build readings whose individual deviations are *under* their own
    per-component thresholds but whose max is at/above the oracle's
    composite gate. The oracle should still surface ``is_breaching``
    and propose a downgrade.
    """

    oracle = DriftCompositeOracle(downgrade_threshold=0.10)
    stm = _stm()
    _ratchet_to(stm, SystemMode.LIVE)

    # Each per-component threshold is 0.50 so none individually
    # breaches; the max (0.20) is above the composite gate (0.10).
    readings = (
        DriftComponentReading(component_id="model", deviation=0.20, threshold=0.50),
        DriftComponentReading(component_id="exec", deviation=0.05, threshold=0.50),
    )

    reading, decision = oracle.evaluate_and_downgrade(
        ts_ns=1_000, readings=readings, stm=stm
    )

    assert reading.is_breaching is True
    assert reading.breached_components == ()
    assert decision is not None
    assert decision.approved is True
    assert stm.current_mode() is SystemMode.CANARY
    assert "composite=0.2000" in decision.reason
    assert "threshold=0.1000" in decision.reason
    assert " >= " in decision.reason


def test_composite_below_oracle_threshold_no_downgrade() -> None:
    """Composite below the oracle gate AND no per-component breach -> no-op."""

    oracle = DriftCompositeOracle(downgrade_threshold=0.50)
    stm = _stm()
    _ratchet_to(stm, SystemMode.LIVE)

    readings = (
        DriftComponentReading(component_id="model", deviation=0.10, threshold=0.30),
        DriftComponentReading(component_id="exec", deviation=0.20, threshold=0.30),
    )

    reading, decision = oracle.evaluate_and_downgrade(
        ts_ns=1_000, readings=readings, stm=stm
    )

    assert reading.is_breaching is False
    assert decision is None
    assert stm.current_mode() is SystemMode.LIVE


def test_invalid_constructor_args_raise() -> None:
    import pytest

    with pytest.raises(ValueError):
        DriftCompositeOracle(component_ids=())
    with pytest.raises(ValueError):
        DriftCompositeOracle(downgrade_threshold=0.0)
    with pytest.raises(ValueError):
        DriftCompositeOracle(downgrade_threshold=-0.1)
