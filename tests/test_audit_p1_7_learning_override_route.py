"""AUDIT-P1.7 — operator learning-override HTTP route regression tests.

The audit's P1.7 finding was that there was no operator-visible path to
flip the :class:`LearningEvolutionFreezePolicy` operator-override flag.
The slow learning loop + evolution patch pipeline are gated by that
flag in conjunction with ``mode is SystemMode.LIVE``; without an
endpoint, an operator had to set ``DIXVISION_LEARNING_OVERRIDE=1`` and
restart the harness to ever unfreeze adaptive mutations. The new
``/api/operator/learning-override`` route fixes that by storing the
override on the harness ``_State``, audited to the authority ledger
on every flip.

The tests in this module pin the contract:

* ``GET /api/operator/learning-override`` returns the live flag, the
  current :class:`SystemMode` name, and ``is_freeze_active``.
* ``POST`` flips the flag, returns the freshly-projected response,
  and writes an ``OPERATOR_LEARNING_OVERRIDE_CHANGED`` row to the
  authority ledger with ``previous`` / ``next`` / ``requestor`` /
  ``reason`` / ``mode`` fields.
* The freeze policy is computed from the live mode, so toggling the
  flag in non-LIVE mode (the harness boots in SAFE) flips
  ``enabled`` to ``True`` while ``is_freeze_active`` remains ``True``
  — exactly the documented HARDEN-04 / INV-70 invariant.
* No-op POSTs (toggling to the same value) still write a ledger row
  so the operator-intent trail is preserved.
* The boot-time seed honours ``DIXVISION_LEARNING_OVERRIDE``.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

ui_server = importlib.import_module("ui.server")


@pytest.fixture
def client() -> TestClient:
    # AUDIT-P1.5 / P1.7 — sibling test modules replace ``ui_server.STATE``
    # in their fixtures. Re-bind a fresh ``_State()`` here so we read and
    # write the same instance the route handler dispatches against.
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    return TestClient(ui_server.app)


@pytest.fixture(autouse=True)
def reset_override() -> None:
    """Force the override back to ``False`` between tests.

    The harness ``_State`` is a process-wide singleton; without this
    fixture a test that flips the override would leak into the next.
    """

    with ui_server.STATE.lock:
        ui_server.STATE.learning_override_enabled = False


def _ledger_rows() -> list[dict[str, object]]:
    """Snapshot of the in-memory ledger chain (read-only)."""

    return [
        dict(entry.payload, kind=entry.kind)
        for entry in ui_server.STATE.governance.ledger.read()
    ]


def test_get_returns_default_disabled(client: TestClient) -> None:
    response = client.get("/api/operator/learning-override")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    # Harness boots in SAFE — the freeze must be active without the
    # override regardless of mode (only LIVE+override unfreezes).
    assert body["is_freeze_active"] is True
    assert isinstance(body["mode"], str) and body["mode"]


def test_post_flips_flag_and_audits_ledger(client: TestClient) -> None:
    before = len(_ledger_rows())

    response = client.post(
        "/api/operator/learning-override",
        json={
            "enabled": True,
            "requestor": "ronald",
            "reason": "manual learning unfreeze",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    # The freeze is still active — the harness mode is not LIVE so
    # ``mode is LIVE and operator_override is True`` does not hold.
    assert body["is_freeze_active"] is True

    # Live state mutated.
    with ui_server.STATE.lock:
        assert ui_server.STATE.learning_override_enabled is True

    # Ledger gained exactly one OPERATOR_LEARNING_OVERRIDE_CHANGED row
    # with the typed payload.
    rows = _ledger_rows()
    assert len(rows) == before + 1
    audit = rows[-1]
    assert audit["kind"] == "OPERATOR_LEARNING_OVERRIDE_CHANGED"
    assert audit["requestor"] == "ronald"
    assert audit["reason"] == "manual learning unfreeze"
    assert audit["previous"] == "false"
    assert audit["next"] == "true"
    assert audit["mode"] == body["mode"]


def test_post_noop_still_audits(client: TestClient) -> None:
    """Toggling to the same value writes an audit row.

    The audit trail captures every operator *intent*, not just the
    transitions. A no-op POST is a deliberate operator decision and
    must be visible in the ledger.
    """

    before = len(_ledger_rows())

    response = client.post(
        "/api/operator/learning-override",
        json={"enabled": False},
    )
    assert response.status_code == 200
    rows = _ledger_rows()
    assert len(rows) == before + 1
    assert rows[-1]["previous"] == "false"
    assert rows[-1]["next"] == "false"


def test_get_after_post_returns_persisted_value(client: TestClient) -> None:
    client.post(
        "/api/operator/learning-override",
        json={"enabled": True},
    )
    response = client.get("/api/operator/learning-override")
    assert response.status_code == 200
    assert response.json()["enabled"] is True


def test_post_rejects_missing_enabled(client: TestClient) -> None:
    response = client.post("/api/operator/learning-override", json={})
    assert response.status_code == 422


def test_post_accepts_minimal_body(client: TestClient) -> None:
    """``requestor`` and ``reason`` are optional — operator dashboards
    that only know ``enabled`` must continue to work."""

    response = client.post(
        "/api/operator/learning-override",
        json={"enabled": True},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is True
    rows = _ledger_rows()
    assert rows[-1]["requestor"] == "dashboard"
    assert rows[-1]["reason"] == ""


def test_post_holds_lock_across_mutation_audit_and_response() -> None:
    """Devin Review BUG_0001 + BUG_0002 — the mutation, the audit-row
    write, *and* the response snapshot must happen atomically under
    ``STATE.lock``.

    A concurrent POST that grabs the lock between the mutation and
    either the ledger append or the response snapshot would corrupt
    the audit chain or leak the second caller's value back to the
    first caller. This test pins the invariant by inspecting the
    route handler's source: the body must contain exactly one
    ``with STATE.lock:`` block, and the assignment, the ledger
    append, and the response source-of-truth (``_project_…``) must
    all sit inside it. The pure projection helper that runs after
    the lock is released takes its inputs by argument so it cannot
    re-read shared state.
    """

    import inspect

    from ui.server import (
        _project_learning_override,
        operator_learning_override_post,
    )

    source = inspect.getsource(operator_learning_override_post)
    assert source.count("with STATE.lock:") == 1, source
    lock_idx = source.index("with STATE.lock:")
    assign_idx = source.index("STATE.learning_override_enabled =")
    append_idx = source.index("STATE.governance.ledger.append")
    # The response must be composed from the snapshotted tuple, not
    # by re-reading STATE after the lock is released.
    assert "_project_learning_override(" in source, source
    project_idx = source.index("_project_learning_override(")
    assert lock_idx < assign_idx < append_idx, source
    # The pure projection helper must not touch STATE itself. Strip
    # docstrings/comments before checking so a ``STATE.lock`` mention
    # in prose does not falsely fail the assertion.
    import ast

    tree = ast.parse(inspect.getsource(_project_learning_override))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "STATE", ast.unparse(node)
    # The handler should NOT call the snapshotting projection (which
    # re-acquires the lock) on its return path.
    assert "_learning_override_response()" not in source, source
    # And the projection call must follow the mutation/audit so it
    # uses the local snapshot.
    assert append_idx < project_idx, source


def test_boot_seed_honours_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting ``DIXVISION_LEARNING_OVERRIDE=1`` at boot pre-arms the flag.

    Re-imports ``ui.server`` under the env var so a fresh ``_State`` is
    constructed and the seed path is exercised. The module is restored
    afterwards so the rest of the suite continues to use the existing
    singleton.
    """

    import ui.server as server_module

    monkeypatch.setenv("DIXVISION_LEARNING_OVERRIDE", "1")
    monkeypatch.setenv("DIXVISION_PERMIT_EPHEMERAL_LEDGER", "1")

    reloaded = importlib.reload(server_module)
    try:
        assert reloaded.STATE.learning_override_enabled is True
    finally:
        # Restore the module to its original state so other tests
        # that hold references to ``STATE``/``app`` keep working.
        monkeypatch.delenv("DIXVISION_LEARNING_OVERRIDE", raising=False)
        importlib.reload(server_module)
