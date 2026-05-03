"""Tests for the dashboard plugin manager router (``ui/plugin_routes``).

Covers list / lifecycle-flip / unknown-id / unknown-lifecycle / cognitive
chat binary toggle / ledger audit row shape.
"""

from __future__ import annotations

import dataclasses

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.contracts.engine import PluginLifecycle
from intelligence_engine.plugins import MicrostructureV1
from ui.plugin_routes import (
    PluginRegistry,
    PluginToggleState,
    build_plugin_router,
)


@dataclasses.dataclass(slots=True)
class _FakeLedger:
    rows: list[dict] = dataclasses.field(default_factory=list)

    def append(self, *, ts_ns: int, kind: str, payload) -> None:
        self.rows.append({"ts_ns": ts_ns, "kind": kind, "payload": dict(payload)})


def _build_app(*, env_enabled: bool = True):
    plugin = MicrostructureV1(lifecycle=PluginLifecycle.SHADOW)
    toggle = PluginToggleState()
    registry = PluginRegistry(
        microstructure_plugins=(plugin,),
        toggle_state=toggle,
        cognitive_chat_env_enabled=lambda: env_enabled,
    )
    ledger = _FakeLedger()
    app = FastAPI()
    app.include_router(
        build_plugin_router(
            registry_provider=lambda: registry,
            ledger_provider=lambda: ledger,
            ts_provider=lambda: 12345,
        )
    )
    return TestClient(app), registry, ledger, plugin, toggle


def test_list_plugins_returns_microstructure_and_cognitive_chat() -> None:
    client, _, _, _, _ = _build_app()
    res = client.get("/api/plugins")
    assert res.status_code == 200
    body = res.json()
    ids = [p["id"] for p in body["plugins"]]
    assert "microstructure_v1" in ids
    assert "cognitive_chat" in ids


def test_microstructure_lifecycle_flips_in_place() -> None:
    client, _, ledger, plugin, _ = _build_app()
    res = client.post(
        "/api/plugins/microstructure_v1/lifecycle",
        json={"lifecycle": "ACTIVE", "reason": "promote"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["lifecycle"] == "ACTIVE"
    assert plugin.lifecycle is PluginLifecycle.ACTIVE
    assert ledger.rows[-1]["kind"] == "PLUGIN_LIFECYCLE"
    assert ledger.rows[-1]["payload"]["plugin_path"] == "microstructure_v1"
    assert ledger.rows[-1]["payload"]["target_status"] == "ACTIVE"


def test_cognitive_chat_disable_via_dashboard_overrides_env() -> None:
    client, _, _, _, toggle = _build_app(env_enabled=True)
    res = client.post(
        "/api/plugins/cognitive_chat/lifecycle",
        json={"lifecycle": "DISABLED"},
    )
    assert res.status_code == 200
    assert res.json()["lifecycle"] == "DISABLED"
    assert toggle.cognitive_chat is False
    list_res = client.get("/api/plugins")
    cognitive = next(
        p for p in list_res.json()["plugins"] if p["id"] == "cognitive_chat"
    )
    assert cognitive["lifecycle"] == "DISABLED"


def test_cognitive_chat_rejects_shadow_lifecycle() -> None:
    client, _, _, _, _ = _build_app()
    res = client.post(
        "/api/plugins/cognitive_chat/lifecycle",
        json={"lifecycle": "SHADOW"},
    )
    assert res.status_code == 400
    assert "SHADOW" in res.json()["detail"]


def test_unknown_plugin_returns_404() -> None:
    client, _, _, _, _ = _build_app()
    res = client.post(
        "/api/plugins/does_not_exist/lifecycle",
        json={"lifecycle": "ACTIVE"},
    )
    assert res.status_code == 404


def test_unknown_lifecycle_returns_400() -> None:
    client, _, _, _, _ = _build_app()
    res = client.post(
        "/api/plugins/microstructure_v1/lifecycle",
        json={"lifecycle": "GARBAGE"},
    )
    assert res.status_code == 400


def test_cognitive_chat_default_lifecycle_follows_env_when_no_override() -> None:
    client_on, _, _, _, _ = _build_app(env_enabled=True)
    res = client_on.get("/api/plugins")
    cognitive = next(
        p for p in res.json()["plugins"] if p["id"] == "cognitive_chat"
    )
    assert cognitive["lifecycle"] == "ACTIVE"

    client_off, _, _, _, _ = _build_app(env_enabled=False)
    res2 = client_off.get("/api/plugins")
    cognitive2 = next(
        p for p in res2.json()["plugins"] if p["id"] == "cognitive_chat"
    )
    assert cognitive2["lifecycle"] == "DISABLED"
