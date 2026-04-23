"""
tests/test_neuromorphic_triad.py
DIX VISION v42.2 — Phase 0 stubs for the three neuromorphic sensors.

Assertions here are the runtime enforcement of axioms N1..N8
(immutable_core/neuromorphic_axioms.lean):

  N1 — no decision authority (tested via authority_lint + static scan)
  N2 — event-only outputs
  N4 — every output writes a ledger row
  N5 — each detector exposes a dead-man check_self()
  N6 — no import of governance/mind.fast_execute/execution.engine/... (C2)
"""
from __future__ import annotations

from unittest.mock import patch

from execution.monitoring.neuromorphic_detector import (
    ANOMALY_TYPES,
    NeuromorphicDetector,
    SystemAnomalyEvent,
)
from governance.signals.neuromorphic_risk import (
    RISK_SIGNAL_TYPES,
    NeuromorphicRisk,
    RiskSignalEvent,
)
from mind.plugins.neuromorphic_signal import (
    SPIKE_TYPES,
    NeuromorphicSignalPlugin,
    SpikeSignalEvent,
)


# ── N2: event-only outputs ─────────────────────────────────────────────
def test_signal_plugin_emits_spike_event_on_volatility_burst() -> None:
    plugin = NeuromorphicSignalPlugin()
    with patch("mind.plugins.neuromorphic_signal.append_event") as ae:
        out = plugin.evaluate({"venue": "binance.btcusdt", "volatility": 0.15})
    assert out["spike"] is not None
    assert out["spike"]["type"] == "VOLATILITY_SPIKE"
    assert ae.called, "spike emission must write a ledger row (N4)"
    event_type, sub_type, source, payload = ae.call_args.args
    assert event_type == "NEUROMORPHIC"
    assert sub_type == "VOLATILITY_SPIKE"
    assert source == "neuromorphic_signal"


def test_signal_plugin_emits_nothing_on_quiet_market() -> None:
    plugin = NeuromorphicSignalPlugin()
    out = plugin.evaluate({"venue": "binance.btcusdt", "volatility": 0.0,
                           "ofi": 0.0, "momentum": 0.0, "liquidity_delta": 0.0})
    assert out["spike"] is None
    assert out["signal"] == 0.0


def test_detector_emits_latency_drift() -> None:
    d = NeuromorphicDetector()
    with patch("execution.monitoring.neuromorphic_detector.append_event") as ae:
        event = d.on_telemetry({"component": "binance_ws",
                                "latency_ms_p99": 1200.0})
    assert isinstance(event, SystemAnomalyEvent)
    assert event.type == "LATENCY_DRIFT"
    assert ae.called, "anomaly emission must write a ledger row (N4)"


def test_detector_emits_silent_data_failure() -> None:
    d = NeuromorphicDetector()
    with patch("execution.monitoring.neuromorphic_detector.append_event"):
        event = d.on_telemetry({"component": "binance_ws",
                                "event_rate_hz": 0.5,
                                "expected_rate_hz": 100.0})
    assert event is not None and event.type == "SILENT_DATA_FAILURE"


def test_risk_sensor_emits_drawdown_acceleration() -> None:
    r = NeuromorphicRisk()
    with patch("governance.signals.neuromorphic_risk.append_event") as ae:
        event = r.evaluate({"drawdown_velocity": 0.75})
    assert isinstance(event, RiskSignalEvent)
    assert event.type == "RISK_ACCELERATION"
    assert event.context == "drawdown_velocity"
    assert ae.called, "risk-signal emission must write a ledger row (N4)"


def test_risk_sensor_emits_nothing_when_calm() -> None:
    r = NeuromorphicRisk()
    assert r.evaluate({"drawdown_velocity": 0.0, "variance_ratio": 1.0,
                       "strategy_pnl_dispersion": 0.1,
                       "avg_cross_correlation": 0.2}) is None


# ── N5: dead-man check_self() on every sensor ──────────────────────────
def test_every_sensor_exposes_check_self() -> None:
    assert hasattr(NeuromorphicSignalPlugin, "check_self")
    assert hasattr(NeuromorphicDetector, "check_self")
    assert hasattr(NeuromorphicRisk, "check_self")


def test_detector_check_self_fresh_instance_is_alive() -> None:
    # A freshly constructed detector was observed this instant — the
    # dead-man window is 3× heartbeat_interval (default 1.0s), so it
    # must report alive immediately after construction.
    assert NeuromorphicDetector().check_self() is True


# ── N5 regression: calm-market dead-man must NOT falsely trip ──────────
def test_signal_plugin_deadman_survives_calm_market(monkeypatch) -> None:
    """Reg for Devin Review BUG_0001 — calm market evaluate() calls must
    keep the dead-man alive even though no spike emits."""
    plugin = NeuromorphicSignalPlugin()

    t0 = plugin._last_tick_seen
    # Simulate a calm-market evaluate() call 4× the heartbeat interval
    # later (6s > 3×1.0s window). Without the fix, check_self() → False.
    monkeypatch.setattr("mind.plugins.neuromorphic_signal.time.monotonic",
                        lambda: t0 + 4.0 * plugin.heartbeat_interval)
    with patch("mind.plugins.neuromorphic_signal.append_event"):
        plugin.evaluate({"venue": "binance.btcusdt"})    # no threshold
    # Same monotonic clock → check_self still passes.
    assert plugin.check_self() is True, (
        "N5 regression: calm-market evaluate() must keep dead-man alive"
    )


def test_risk_sensor_deadman_survives_calm_conditions(monkeypatch) -> None:
    """Reg for Devin Review BUG_0002 — calm risk-feature evaluate()
    must keep the dead-man alive."""
    import governance.signals.neuromorphic_risk as nr
    r = nr.NeuromorphicRisk()

    t0 = r._last_tick_seen
    monkeypatch.setattr(nr.time, "monotonic",
                        lambda: t0 + 10.0 * r.heartbeat_interval)
    r.evaluate({"drawdown_velocity": 0.0})      # no threshold
    assert r.check_self() is True, (
        "N5 regression: calm risk eval() must keep dead-man alive"
    )


# ── N1 + N6: no forbidden imports (static AST scan) ────────────────────
def test_neuromorphic_modules_do_not_import_decision_surfaces() -> None:
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    targets = [
        "mind/plugins/neuromorphic_signal.py",
        "execution/monitoring/neuromorphic_detector.py",
        "governance/signals/neuromorphic_risk.py",
    ]
    forbidden = {
        "governance.kernel", "governance.policy_engine",
        "governance.constraint_compiler", "governance.mode_manager",
        "governance.patch_pipeline", "mind.fast_execute",
        "execution.engine", "execution.adapter_router",
        "execution.adapters", "security.operator",
        "security.wallet_policy", "security.wallet_connect",
        "core.registry",
    }
    for rel in targets:
        tree = ast.parse((root / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not any(node.module == b or node.module.startswith(b + ".")
                               for b in forbidden), (
                    f"{rel} imports forbidden decision surface: {node.module}"
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not any(alias.name == b or alias.name.startswith(b + ".")
                                   for b in forbidden), (
                        f"{rel} imports forbidden decision surface: {alias.name}"
                    )


def test_spike_types_are_stable() -> None:
    assert set(SPIKE_TYPES) == {
        "VOLATILITY_SPIKE", "OFI_SPIKE",
        "MOMENTUM_IGNITION", "LIQUIDITY_SHOCK"}


def test_anomaly_types_are_stable() -> None:
    assert set(ANOMALY_TYPES) == {
        "LATENCY_DRIFT", "SILENT_DATA_FAILURE",
        "MEMORY_PRESSURE_GRADIENT", "EVENT_RHYTHM_BREAK"}


def test_risk_signal_types_are_stable() -> None:
    assert set(RISK_SIGNAL_TYPES) == {
        "RISK_ACCELERATION", "REGIME_SHIFT",
        "STRATEGY_INSTABILITY", "CORRELATION_BREAKDOWN"}
