"""DASH-04 — DecisionTracePanel widget tests."""

from __future__ import annotations

from core.contracts.events import (
    ExecutionEvent,
    ExecutionStatus,
    HazardEvent,
    HazardSeverity,
    Side,
    SignalEvent,
    SystemEvent,
    SystemEventKind,
)
from dashboard.control_plane.decision_trace import DecisionTracePanel
from state.ledger.reader import LedgerReader


def _seeded_ledger():
    ledger = LedgerReader()
    ledger._seed_for_tests(
        [
            SignalEvent(
                ts_ns=10,
                symbol="EURUSD",
                side=Side.BUY,
                confidence=0.7,
                plugin_chain=("microstructure_v1",),
            ),
            ExecutionEvent(
                ts_ns=11,
                symbol="EURUSD",
                side=Side.BUY,
                qty=1.0,
                price=1.10,
                status=ExecutionStatus.FILLED,
            ),
            SignalEvent(
                ts_ns=20,
                symbol="BTCUSDT",
                side=Side.SELL,
                confidence=0.6,
                plugin_chain=("regime_v1", "microstructure_v1"),
            ),
            HazardEvent(
                ts_ns=21,
                code="HAZ-03",
                severity=HazardSeverity.MEDIUM,
                source="system_engine",
            ),
            SystemEvent(
                ts_ns=22,
                sub_kind=SystemEventKind.HEARTBEAT,
                source="system",
            ),
        ]
    )
    return ledger


def test_panel_groups_by_symbol_preserving_order():
    panel = DecisionTracePanel(ledger=_seeded_ledger())
    chains = panel.chains()
    symbols = tuple(c.symbol for c in chains)
    assert symbols == ("EURUSD", "BTCUSDT", "<system>")
    assert chains[0].steps[0].kind == "SIGNAL_EVENT"
    assert chains[0].steps[1].kind == "EXECUTION_EVENT"
    assert "FILLED" in chains[0].steps[1].summary


def test_panel_renders_plugin_chain_for_signal():
    panel = DecisionTracePanel(ledger=_seeded_ledger())
    chains = panel.chains()
    btc = next(c for c in chains if c.symbol == "BTCUSDT")
    summary = btc.steps[0].summary
    assert "regime_v1 -> microstructure_v1" in summary
    assert "0.60" in summary


def test_panel_buckets_system_and_hazard_events():
    panel = DecisionTracePanel(ledger=_seeded_ledger())
    chains = panel.chains()
    sys_chain = next(c for c in chains if c.symbol == "<system>")
    kinds = tuple(s.kind for s in sys_chain.steps)
    assert kinds == ("HAZARD_EVENT", "SYSTEM_EVENT")
