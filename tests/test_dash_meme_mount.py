"""DIX MEME — mount + harness-independence invariants.

These tests cover the contract for the second dashboard surface mounted
under ``/meme/`` (DEXtools-styled memecoin app):

1. The ``/meme/`` ``StaticFiles`` mount is conditional on a built
   artefact under ``dash_meme/dist`` (same pattern as ``/dash2/``). When
   the build is missing the harness must still boot — it must NOT 500
   or crash at import time.

2. ``_MEME_AVAILABLE`` is a module-load snapshot, mirroring
   ``_DASH2_AVAILABLE``. This prevents the redirect-without-mount race
   documented on PR #123 (BUG_0001).

3. **Harness invariant** — the FastAPI app and its engine state machine
   construct cleanly even when *neither* dashboard build artefact is
   present on disk. Closing both dashboards (or never building them at
   all) must not break the system: sensors, governance, audit ledger,
   and the learning loop are all defined by the harness, not by the
   browser surface.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest


def _reload_server() -> object:
    """Reload ``ui.server`` so the module-load mount snapshots reflect
    whatever the current filesystem state is.

    The mount predicate is captured at import time on purpose (Devin
    Review BUG_0001 on PR #123 — see the ``_DASH2_AVAILABLE`` comment in
    ``ui/server.py``). Tests therefore have to clear the cached module
    before re-importing under a new ``dash_meme/dist`` filesystem state.
    """

    import ui.server as srv

    return importlib.reload(srv)


# --------------------------------------------------------------------------- #
# Module-load snapshots
# --------------------------------------------------------------------------- #


def test_meme_dist_path_under_dash_meme(tmp_path: Path) -> None:
    """The mount predicate must point at ``dash_meme/dist``."""
    del tmp_path  # unused — we only inspect attributes, not files
    server = _reload_server()
    assert server._MEME_DIST.name == "dist"
    assert server._MEME_DIST.parent.name == "dash_meme"


def test_meme_available_is_a_module_load_snapshot() -> None:
    """``_MEME_AVAILABLE`` must be a plain bool (snapshot), never a
    callable that re-checks the filesystem on every request."""
    server = _reload_server()
    assert isinstance(server._MEME_AVAILABLE, bool)


def test_meme_available_matches_index_existence() -> None:
    """The snapshot must agree with the on-disk reality at import time
    (matching the ``_DASH2_AVAILABLE`` discipline so the redirect path
    and the StaticFiles mount stay in lock-step)."""
    server = _reload_server()
    expected = server._MEME_DIST.exists() and server._MEME_INDEX.exists()
    assert server._MEME_AVAILABLE is expected


# --------------------------------------------------------------------------- #
# Mount registration
# --------------------------------------------------------------------------- #


def test_meme_mount_only_when_available() -> None:
    """The Starlette mount under ``/meme`` exists iff ``_MEME_AVAILABLE``."""
    server = _reload_server()
    mount_paths = {
        getattr(r, "path", None) for r in server.app.routes
    }
    if server._MEME_AVAILABLE:
        assert "/meme" in mount_paths
    else:
        assert "/meme" not in mount_paths


# --------------------------------------------------------------------------- #
# Harness invariant — works with no dashboards built
# --------------------------------------------------------------------------- #


def test_harness_app_constructs_without_either_dashboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If neither ``dash_meme/dist`` nor ``dashboard2026/dist`` exist,
    importing ``ui.server`` must still produce a working FastAPI app
    with a live ``/api/health`` endpoint.

    This is the operator-facing guarantee: closing both dashboards (or
    running on a fresh clone where Node is not installed) does NOT stop
    the harness. The learning loop, sensors, governance FSM, and audit
    ledger live inside ``ui.server`` and run independently of any
    browser surface being present.
    """

    # Force both availability flags off via patched ``Path.exists`` —
    # we cannot truly delete the dist dirs in CI, so we patch the
    # snapshot predicate inputs instead.
    real_exists = Path.exists

    def fake_exists(self: Path) -> bool:
        if self.name in ("dist", "index.html"):
            parent = self.parent if self.name == "index.html" else self
            if parent.name in ("dist",):
                parent = parent.parent
            if parent.name in ("dashboard2026", "dash_meme"):
                return False
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", fake_exists)

    server = _reload_server()
    assert server._DASH2_AVAILABLE is False
    assert server._MEME_AVAILABLE is False

    # The harness app must still be a FastAPI app with the canonical
    # operator-control routes registered.
    routes = {getattr(r, "path", None) for r in server.app.routes}
    assert "/api/health" in routes
    assert "/api/dashboard/action/intent" in routes

    # And the engine state singleton must have constructed (sensors,
    # governance, audit ledger all live on this object).
    assert server.STATE is not None
    with server.STATE.lock:
        engines = server.STATE.all_engines()
    assert engines, "harness engines failed to register without dashboards"
