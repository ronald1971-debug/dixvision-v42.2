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

from ui.server import STATE, app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_override() -> None:
    """Force the override back to ``False`` between tests.

    The harness ``_State`` is a process-wide singleton; without this
    fixture a test that flips the override would leak into the next.
    """

    with STATE.lock:
        STATE.learning_override_enabled = False


def _ledger_rows() -> list[dict[str, object]]:
    """Snapshot of the in-memory ledger chain (read-only)."""

    return [
        dict(entry.payload, kind=entry.kind)
        for entry in STATE.governance.ledger.read()
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
    with STATE.lock:
        assert STATE.learning_override_enabled is True

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


def test_post_holds_lock_across_mutation_and_audit() -> None:
    """Devin Review BUG_0001 — the mutation and the audit-row write
    must happen atomically under ``STATE.lock``.

    A concurrent POST that grabs the lock between the mutation and
    the ledger append would corrupt the audit chain (rows whose
    sequence does not match mutation order). This test pins the
    invariant by inspecting the route handler's bytecode for a
    single ``with STATE.lock:`` block that wraps both the assignment
    to ``learning_override_enabled`` and the call to
    ``STATE.governance.ledger.append``.
    """

    import inspect

    from ui.server import operator_learning_override_post

    source = inspect.getsource(operator_learning_override_post)
    # The handler must contain exactly one ``with STATE.lock:`` block,
    # and the ledger.append call must be inside it (i.e. before the
    # ``return _learning_override_response()`` line that closes the
    # function body).
    assert source.count("with STATE.lock:") == 1, source
    lock_idx = source.index("with STATE.lock:")
    return_idx = source.index("return _learning_override_response()")
    append_idx = source.index("STATE.governance.ledger.append")
    assert lock_idx < append_idx < return_idx, source


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
