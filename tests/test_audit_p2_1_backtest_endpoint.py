"""AUDIT-P2.1 — regression for ``POST /api/testing/backtest``.

Pins:
* the endpoint exists and accepts the form payload the dashboard
  ``Backtester`` widget sends;
* the response carries a stable ``seed`` keyed off the request fields,
  so the same payload returns byte-identical equity / drawdown / trades
  every time;
* the response shape matches the TypeScript ``BacktestRunResponse`` the
  widget consumes;
* malformed payloads produce a 400 (Pydantic validation), not a 500.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv(
        "DIXVISION_LEDGER_PATH", str(tmp_path / "governance.db")
    )
    monkeypatch.setenv(
        "DIXVISION_INTENT_HMAC_KEY", "test-hmac-key-audit-p2-1"
    )
    if "ui.server" in list(os.sys.modules):
        del os.sys.modules["ui.server"]
    from ui.server import app

    return TestClient(app)


def _payload() -> dict[str, object]:
    return {
        "strategy": "ema_cross_20_50",
        "symbol": "BTC/USDT",
        "start_iso": "2025-01-01T00:00:00Z",
        "end_iso": "2025-04-01T00:00:00Z",
        "fill_model": "next_tick",
        "slippage_bps": 8.0,
    }


def test_backtest_endpoint_returns_canonical_shape(client: TestClient) -> None:
    res = client.post("/api/testing/backtest", json=_payload())
    assert res.status_code == 200, res.text
    body = res.json()

    assert isinstance(body["seed"], str) and len(body["seed"]) >= 8
    assert body["request"]["strategy"] == "ema_cross_20_50"
    assert body["request"]["symbol"] == "BTC/USDT"

    assert isinstance(body["equity"], list) and len(body["equity"]) == 241
    assert isinstance(body["drawdown"], list) and len(body["drawdown"]) == 241
    assert isinstance(body["trades"], list) and len(body["trades"]) > 0

    metrics = body["metrics"]
    for key in (
        "final_equity_pct",
        "cagr",
        "sharpe",
        "sortino",
        "max_dd_pct",
        "win_rate",
        "avg_trade_pct",
        "longest_loss_streak",
        "n_trades",
    ):
        assert key in metrics, f"missing metric: {key}"

    assert "deterministic-internal" in body["notes"]
    assert "no-execution-authority" in body["notes"]


def test_backtest_endpoint_is_replay_deterministic(client: TestClient) -> None:
    a = client.post("/api/testing/backtest", json=_payload()).json()
    b = client.post("/api/testing/backtest", json=_payload()).json()
    assert a["seed"] == b["seed"]
    assert a["equity"] == b["equity"]
    assert a["drawdown"] == b["drawdown"]
    assert a["trades"] == b["trades"]
    assert a["metrics"] == b["metrics"]


def test_backtest_endpoint_seed_changes_on_param_change(
    client: TestClient,
) -> None:
    a = client.post("/api/testing/backtest", json=_payload()).json()
    payload = _payload()
    payload["slippage_bps"] = 12.5
    b = client.post("/api/testing/backtest", json=payload).json()
    assert a["seed"] != b["seed"], (
        "Seed must depend on slippage_bps so two operators with different "
        "params get different reports"
    )


def test_backtest_endpoint_rejects_bad_iso_with_400(
    client: TestClient,
) -> None:
    payload = _payload()
    payload["start_iso"] = "not-an-iso-date"
    res = client.post("/api/testing/backtest", json=payload)
    assert res.status_code == 400, res.text


def test_backtest_endpoint_rejects_negative_slippage_with_422(
    client: TestClient,
) -> None:
    payload = _payload()
    payload["slippage_bps"] = -1.0
    res = client.post("/api/testing/backtest", json=payload)
    assert res.status_code == 422, res.text
