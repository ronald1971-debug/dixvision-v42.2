"""AUDIT-P2.3 — full-stack end-to-end integration test.

The unit-test suite covers each engine and harness handler in
isolation but does not exercise the path

    POST /api/tick    -> intelligence -> execution -> ledger
    POST /api/signal  -> intelligence -> execution -> ledger

with a real ``ui.server.app`` boot. Wiring regressions like the
ones surfaced in the AUDIT-WIRE wave (constructor params built but
never bound in the harness) all manifest as silent absence of an
expected ledger row, which unit tests cannot detect.

This module pins the contract:

* The harness forwards every harness-emitted ``SignalEvent`` and
  ``ExecutionEvent`` to ``GovernanceEngine.process``, which appends
  a ``SIGNAL_AUDIT`` / ``EXECUTION_AUDIT`` row to the authority
  ledger (see ``governance_engine/engine.py``).
* The assertion is on **counts**, not on which event happened first,
  so any reasonable ordering inside the harness is acceptable.

No network IO, no SQLite file: ``_State()`` defaults to the
in-memory governance ledger that the regression suite already
expects to be safe to read.
"""

from __future__ import annotations

import importlib

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client():
    # Reset shared state between tests so order does not matter.
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


def _ledger_kind_count(state: object, kind: str) -> int:
    """Count rows of ``kind`` in the governance authority ledger.

    Reads via the official ``ledger.read()`` API rather than poking
    private attributes, so this assertion still holds when the
    backing store is swapped (e.g. SQLite -> in-memory or back).
    """

    rows = state.governance.ledger.read()  # type: ignore[attr-defined]
    return sum(1 for row in rows if row.kind == kind)


def test_post_signal_writes_signal_audit_row(client):
    """``POST /api/signal`` must surface in the authority ledger."""

    state = ui_server.STATE
    before_signal = _ledger_kind_count(state, "SIGNAL_AUDIT")
    before_execution = _ledger_kind_count(state, "EXECUTION_AUDIT")

    payload = {
        "ts_ns": 1_000_000_000,
        "symbol": "BTC-USD",
        "side": "BUY",
        "confidence": 0.65,
    }
    r = client.post("/api/signal", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["signal"]["symbol"] == "BTC-USD"

    after_signal = _ledger_kind_count(state, "SIGNAL_AUDIT")
    after_execution = _ledger_kind_count(state, "EXECUTION_AUDIT")

    # At minimum the raw operator-emitted SignalEvent (recorded under
    # the ``ui_harness`` source) plus any intelligence-pipeline
    # rewrites must each produce a SIGNAL_AUDIT row. We require >=1
    # to avoid coupling to plugin internals.
    assert after_signal >= before_signal + 1, (
        "POST /api/signal did not produce a SIGNAL_AUDIT ledger row; "
        "STATE.record -> governance.process(SignalEvent) wiring "
        "regressed (see ui/server.py)"
    )

    # Execution downstream rows are emitted only when the intelligence
    # pipeline actually approves the signal. Default plugins may
    # reject low-confidence signals or emit none. Either way, if the
    # response surfaced any executions, every one of them must have a
    # paired EXECUTION_AUDIT row.
    response_executions = body.get("executions", [])
    if response_executions:
        assert (
            after_execution >= before_execution + len(response_executions)
        ), (
            "POST /api/signal returned executions but at least one "
            "EXECUTION_AUDIT row is missing from the authority ledger"
        )


def test_post_tick_writes_audit_rows_for_emitted_events(client):
    """``POST /api/tick`` flow must produce audit rows for every
    SignalEvent and ExecutionEvent it emits.

    We do not assume the intelligence engine produced a signal on
    this single tick (the default plugin set is conservative). What
    we *do* assert is the invariant: ``len(ledger SIGNAL_AUDIT) >=
    len(response signals)`` and the same for executions. If the
    harness forwards no events through Governance the response and
    ledger will diverge.
    """

    state = ui_server.STATE
    before_signal = _ledger_kind_count(state, "SIGNAL_AUDIT")
    before_execution = _ledger_kind_count(state, "EXECUTION_AUDIT")

    payload = {
        "ts_ns": 2_000_000_000,
        "symbol": "BTC-USD",
        "bid": 50_000.0,
        "ask": 50_010.0,
        "last": 50_005.0,
        "volume": 1.5,
        "venue": "paper",
    }
    r = client.post("/api/tick", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    after_signal = _ledger_kind_count(state, "SIGNAL_AUDIT")
    after_execution = _ledger_kind_count(state, "EXECUTION_AUDIT")

    response_signals = body.get("signals", [])
    response_executions = body.get("executions", [])

    assert after_signal - before_signal >= len(response_signals), (
        "POST /api/tick emitted "
        f"{len(response_signals)} signals but the authority ledger "
        f"only gained {after_signal - before_signal} SIGNAL_AUDIT "
        "rows; the intelligence -> governance audit path regressed."
    )
    assert (
        after_execution - before_execution >= len(response_executions)
    ), (
        "POST /api/tick emitted "
        f"{len(response_executions)} executions but the authority "
        "ledger only gained "
        f"{after_execution - before_execution} EXECUTION_AUDIT rows."
    )


def test_full_stack_e2e_signal_then_tick(client):
    """End-to-end smoke: signal then tick. Asserts the ledger stays
    monotonic and the event ring exposes both events to operators.

    This is the most operator-shaped flow: the dashboard cockpit
    drives `/api/signal` (manual override) and the live feed drives
    `/api/tick`. The same ledger has to absorb both.
    """

    state = ui_server.STATE
    before_signal = _ledger_kind_count(state, "SIGNAL_AUDIT")

    r1 = client.post(
        "/api/signal",
        json={
            "ts_ns": 3_000_000_000,
            "symbol": "ETH-USD",
            "side": "SELL",
            "confidence": 0.55,
        },
    )
    assert r1.status_code == 200, r1.text

    r2 = client.post(
        "/api/tick",
        json={
            "ts_ns": 3_000_000_001,
            "symbol": "ETH-USD",
            "bid": 3000.0,
            "ask": 3001.0,
            "last": 3000.5,
            "volume": 0.25,
            "venue": "paper",
        },
    )
    assert r2.status_code == 200, r2.text

    after_signal = _ledger_kind_count(state, "SIGNAL_AUDIT")
    assert after_signal >= before_signal + 1, (
        "End-to-end flow did not produce any SIGNAL_AUDIT rows"
    )

    # Operator-facing event ring must surface both events.
    events = client.get("/api/events", params={"limit": 50})
    assert events.status_code == 200
    body = events.json()
    kinds = [event["kind"] for event in body["events"]]
    assert "MARKET_TICK" in kinds, (
        "POST /api/tick did not append a MARKET_TICK to the operator "
        "event ring"
    )
    assert any(kind == "SIGNAL_EVENT" for kind in kinds), (
        "POST /api/signal did not append a SIGNAL_EVENT to the "
        "operator event ring"
    )
