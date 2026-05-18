"""AUDIT-P1.7 — operator learning-override HTTP route regression tests.

The audit's P1.7 finding was that there was no operator-visible path to
flip the :class:`LearningEvolutionFreezePolicy` operator-override flag.
The slow learning loop + evolution patch pipeline are gated by that
flag; without an endpoint, an operator had to set
``DIXVISION_LEARNING_OVERRIDE=1`` and restart the harness to ever
unfreeze adaptive mutations. The new ``/api/operator/learning-override``
route fixes that by storing the override on the harness ``_State``,
audited to the authority ledger on every flip.

The tests in this module pin the contract:

* ``GET /api/operator/learning-override`` returns the live flag, the
  current :class:`SystemMode` name, and ``is_freeze_active``.
* ``POST`` flips the flag, returns the freshly-projected response,
  and writes a **pair** of audit rows on every transition:

  * ``OPERATOR_LEARNING_OVERRIDE_CHANGED`` — the operator-intent row
    carrying ``previous`` / ``next`` / ``requestor`` / ``reason`` /
    ``mode`` (P0 refinement; pre-existing).
  * ``POLICY_STATE`` — the canonical projection emitted by
    :meth:`LearningEvolutionFreezePolicy.to_system_event` carrying
    the resulting policy state (``frozen`` / ``mode`` /
    ``operator_override`` / ``version``). Both rows share the same
    ``flip_ts_ns`` so an offline replay validator can correlate them.

* Under ``v42.2-P0-RELAX`` the freeze gate is
  ``operator_override is True`` alone — the ``mode is LIVE`` half of
  the dual gate was dropped per direct operator directive. Flipping
  ``operator_override=True`` therefore unfreezes the policy in every
  mode (including SAFE, where the harness boots).
* No-op POSTs (toggling to the same value) still write the same pair
  of audit rows so the operator-intent trail is preserved.
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
    #
    # PR-Z1 — ``_State()`` now boots with the override pre-armed (HARDEN-04
    # conditional relaxation). These tests exercise the *route's*
    # reset-from-disabled behavior, which is orthogonal to the boot seed,
    # so we force the flag back to ``False`` after construction. The new
    # boot-seed contract is pinned in
    # ``tests/test_pr_z1_harden04_conditional_relax.py``.
    ui_server.STATE = ui_server._State()  # type: ignore[attr-defined]
    with ui_server.STATE.lock:
        ui_server.STATE.learning_override_enabled = False
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
        dict(entry.payload, kind=entry.kind) for entry in ui_server.STATE.governance.ledger.read()
    ]


def test_get_returns_default_disabled(client: TestClient) -> None:
    response = client.get("/api/operator/learning-override")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    # Under ``v42.2-P0-RELAX`` the freeze gate is ``operator_override``
    # alone. With the override defaulted off in this test (the
    # autouse fixture resets it), the freeze must still be active.
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
    # Under ``v42.2-P0-RELAX`` ``operator_override is True`` is the
    # sole unfreeze predicate — mode is no longer consulted. The
    # harness mode here is non-LIVE but the override flip alone
    # unfreezes the policy.
    assert body["is_freeze_active"] is False

    # Live state mutated.
    with ui_server.STATE.lock:
        assert ui_server.STATE.learning_override_enabled is True

    # Ledger gained the canonical pair of audit rows: the
    # OPERATOR_LEARNING_OVERRIDE_CHANGED operator-intent row plus the
    # POLICY_STATE projection from
    # LearningEvolutionFreezePolicy.to_system_event (P0 refinement).
    rows = _ledger_rows()
    assert len(rows) == before + 2
    audit = rows[-2]
    assert audit["kind"] == "OPERATOR_LEARNING_OVERRIDE_CHANGED"
    assert audit["requestor"] == "ronald"
    assert audit["reason"] == "manual learning unfreeze"
    assert audit["previous"] == "false"
    assert audit["next"] == "true"
    assert audit["mode"] == body["mode"]
    # The POLICY_STATE row mirrors the resulting policy state under
    # the relaxed v42.2-P0-RELAX predicate — override=True → frozen=false.
    policy_row = rows[-1]
    assert policy_row["kind"] == "POLICY_STATE"
    assert policy_row["policy"] == "LearningEvolutionFreezePolicy"
    assert policy_row["operator_override"] == "true"
    assert policy_row["frozen"] == "false"
    assert policy_row["version"] == "v42.2-P0-RELAX"
    assert policy_row["mode"] == body["mode"]


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
    # Same canonical row pair as a real transition — the operator-intent
    # trail records every deliberate decision, no-op or not.
    assert len(rows) == before + 2
    assert rows[-2]["kind"] == "OPERATOR_LEARNING_OVERRIDE_CHANGED"
    assert rows[-2]["previous"] == "false"
    assert rows[-2]["next"] == "false"
    assert rows[-1]["kind"] == "POLICY_STATE"
    assert rows[-1]["operator_override"] == "false"
    assert rows[-1]["frozen"] == "true"


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
    # The most-recent two rows are the operator-intent +
    # POLICY_STATE pair; the minimal-body defaults live on the
    # operator-intent row (rows[-2]).
    assert rows[-1]["kind"] == "POLICY_STATE"
    assert rows[-2]["kind"] == "OPERATOR_LEARNING_OVERRIDE_CHANGED"
    assert rows[-2]["requestor"] == "dashboard"
    assert rows[-2]["reason"] == ""


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
