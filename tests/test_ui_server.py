"""Phase E1 — UI harness server tests.

Exercises the FastAPI surface in ``ui/server.py`` with the in-process
``TestClient``. No network IO; deterministic.
"""

from __future__ import annotations

import importlib

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client():
    # Reset shared state between tests so order doesn't matter.
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    # PR-DEV-A — boot defaults block trading at the Execution Gate. The
    # Phase E1 harness tests in this module pre-date that gate and pin
    # the legacy dispatch contract (REJECTED ↔ trading blocked is a
    # separate regression suite in ``test_pr_dev_a_development_mode``).
    # Flip the gate open for this fixture so the legacy assertions
    # continue to exercise the broker dispatch path.
    from core.contracts.development_mode import DevelopmentModePolicy

    with ui_server.STATE.lock:
        ui_server.STATE.trading_allowed = True
        ui_server.STATE.development_mode_policy = DevelopmentModePolicy(
            development_enabled=ui_server.STATE.development_mode_enabled,
            trading_allowed=True,
            mode=(ui_server.STATE.governance.state_transitions.current_mode()),
        )
        ui_server.STATE.execution.set_development_mode_policy(
            ui_server.STATE.development_mode_policy
        )
    return TestClient(ui_server.app)


def test_root_routes_to_live_dashboard(client):
    """``GET /`` must take operators to the live SPA when the React build
    is present (Wave-Live PR-4) and otherwise fall back to the Phase E1
    stub. The fallback path keeps CI jobs and Node-less environments
    from receiving a hard 404 at the front door.
    """
    r = client.get("/", follow_redirects=False)
    if ui_server._DASH2_AVAILABLE:
        assert r.status_code == 307, "/ must redirect to /dash2/ when the React build is present"
        assert r.headers["location"] == "/dash2/"
    else:
        assert r.status_code == 200
        assert "DIX VISION" in r.text


def test_health_returns_all_six_engines(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    engines = r.json()["engines"]
    assert set(engines) == {
        "intelligence",
        "execution",
        "system",
        "governance",
        "learning",
        "evolution",
    }
    for spec in engines.values():
        assert spec["state"] == "OK"


def test_registry_endpoints(client):
    r1 = client.get("/api/registry/engines")
    r2 = client.get("/api/registry/plugins")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert "engines" in r1.json()
    assert "plugins" in r2.json()


def test_signal_without_tick_returns_failed_execution(client):
    r = client.post(
        "/api/signal",
        json={"symbol": "EURUSD", "side": "BUY", "confidence": 0.7},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["executions"]) == 1
    assert body["executions"][0]["status"] == "FAILED"


def test_tick_then_signal_flows_to_filled(client):
    tick_resp = client.post(
        "/api/tick",
        json={"symbol": "BTCUSDT", "bid": 49990, "ask": 50010, "last": 50000},
    )
    assert tick_resp.status_code == 200

    sig_resp = client.post(
        "/api/signal",
        json={"symbol": "BTCUSDT", "side": "BUY", "confidence": 0.8},
    )
    assert sig_resp.status_code == 200
    body = sig_resp.json()
    exec_evt = body["executions"][0]
    assert exec_evt["status"] == "FILLED"
    assert exec_evt["side"] == "BUY"
    assert exec_evt["price"] == 50000.0


def test_tick_emits_signal_and_fills_via_microstructure_plugin(client):
    # SHADOW-DEMOLITION-01: MicrostructureV1 now defaults to ACTIVE,
    # so a deviating tick produces an untagged signal that flows
    # through the harness execution path without the legacy shadow
    # rejection gate.
    r = client.post(
        "/api/tick",
        json={"symbol": "EURUSD", "bid": 99.99, "ask": 100.01, "last": 100.10},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["signals"]) == 1
    sig = body["signals"][0]
    assert sig["side"] == "BUY"
    assert sig["meta"].get("shadow") != "true"

    assert len(body["executions"]) == 1
    exe = body["executions"][0]
    assert exe["meta"].get("reason") != "shadow signal"


def test_events_endpoint_records_recent(client):
    client.post(
        "/api/tick",
        json={"symbol": "X", "bid": 10, "ask": 11, "last": 10.5},
    )
    client.post(
        "/api/signal",
        json={"symbol": "X", "side": "BUY", "confidence": 0.5},
    )
    r = client.get("/api/events?limit=10")
    assert r.status_code == 200
    events = r.json()["events"]
    kinds = [e.get("kind") for e in events]
    assert "MARKET_TICK" in kinds
    assert "SIGNAL_EVENT" in kinds
    assert "EXECUTION_EVENT" in kinds


def test_tick_emits_meta_controller_ledger(client):
    # P1.1 — ``/api/tick`` now drives ``IntelligenceEngine.run_meta_tick``
    # so the BeliefState / PressureVector / META_AUDIT four-event ledger
    # flows on the bus on every tick. Pinning the contract here so the
    # wiring cannot regress to the dead-code state where the meta
    # controller hot-path was constructed but never invoked.
    r = client.post(
        "/api/tick",
        json={"symbol": "BTCUSDT", "bid": 99.99, "ask": 100.01, "last": 100.10},
    )
    assert r.status_code == 200
    body = r.json()
    meta_ledger = body.get("meta_ledger")
    assert isinstance(meta_ledger, list)
    kinds = [ev.get("sub_kind") for ev in meta_ledger]
    assert "BELIEF_STATE_SNAPSHOT" in kinds
    assert "PRESSURE_VECTOR_SNAPSHOT" in kinds
    assert "META_AUDIT" in kinds

    events = client.get("/api/events?limit=50").json()["events"]
    seen_sub_kinds = [e.get("sub_kind") for e in events]
    assert "META_AUDIT" in seen_sub_kinds
    assert "BELIEF_STATE_SNAPSHOT" in seen_sub_kinds


def test_tick_drives_closed_learning_and_structural_evolution(client):
    # PR-C (P0-3) — ``/api/tick`` now drives both
    # ``ClosedLearningLoop.tick`` and ``StructuralEvolutionLoop.tick`` on
    # the production hot path so the loops actually run, not only via
    # the env-gated ``/api/admin/learning/tick`` debug route. Both loops
    # snapshot the live ``LearningEvolutionFreezePolicy`` so the wiring
    # is observable in the response shape.
    r = client.post(
        "/api/tick",
        json={"symbol": "BTCUSDT", "bid": 99.99, "ask": 100.01, "last": 100.10},
    )
    assert r.status_code == 200
    body = r.json()

    closed = body.get("closed_learning")
    structural = body.get("structural_evolution")
    assert isinstance(closed, dict)
    assert isinstance(structural, dict)

    # Stable projection shape — operators (and smoke tests) read these
    # keys to verify the loops are firing without leaking typed engine
    # objects across the HTTP boundary.
    for key in (
        "ts_ns",
        "frozen",
        "policy_mode_name",
        "operator_override",
    ):
        assert key in closed, key
        assert key in structural, key

    # Under ``v42.2-P0-RELAX`` the freeze gate is
    # ``operator_override is True`` alone (mode no longer consulted).
    # PR #376 flipped the boot seed so ``operator_override`` defaults
    # to ``True``; the loops therefore unfreeze on the very first
    # /api/tick. ``frozen`` must mirror ``operator_override`` —
    # ``frozen == not operator_override``.
    assert closed["operator_override"] is True
    assert structural["operator_override"] is True
    assert closed["frozen"] is False
    assert structural["frozen"] is False


def test_signal_validates_side(client):
    r = client.post(
        "/api/signal",
        json={"symbol": "X", "side": "WRONG", "confidence": 0.5},
    )
    assert r.status_code == 422


def test_tick_validates_positive_prices(client):
    r = client.post(
        "/api/tick",
        json={"symbol": "X", "bid": -1, "ask": 1, "last": 1},
    )
    assert r.status_code == 422
