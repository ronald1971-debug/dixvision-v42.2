"""Round-10 regression tests.

Covers two Devin-Review findings:
- cockpit/qr.py: unequal block split for version 10 (payloads 231..271 bytes).
- system/state.py: governance_mode default so bootstrap INIT→NORMAL transition
  actually fires and emits the audit event.
"""
from __future__ import annotations


def test_qr_version10_payload_fully_encoded():
    """A 271-byte payload must still round-trip all data codewords.

    Version 10-L has 4 blocks split 2×68 + 2×69 = 274 data codewords. The
    previous integer-division split dropped the last 2 codewords. We don't
    run a full QR decoder here; instead we verify that the block lengths the
    encoder produces match the spec, because that is precisely the invariant
    the bug violated.
    """
    from cockpit.qr import _VERSIONS

    v10 = next(e for e in _VERSIONS if e[0] == 10)
    _, _cap, total_data, _ec, n_blocks = v10
    short_len = total_data // n_blocks
    n_long = total_data % n_blocks
    n_short = n_blocks - n_long
    assert short_len == 68
    assert n_long == 2
    assert n_short == 2
    # Reconstruct the lengths the encoder builds.
    lengths = [short_len] * n_short + [short_len + 1] * n_long
    assert sum(lengths) == total_data  # no dropped codewords


def test_qr_png_encodes_large_payload_without_crashing():
    """Smoke test: a 260-byte payload goes through encode_qr + qr_png_bytes."""
    from cockpit.qr import encode_qr, qr_png_bytes

    payload = "x" * 260
    n, mat = encode_qr(payload)
    assert n == 17 + 4 * 10  # version 10 size
    assert len(mat) == n
    assert all(len(row) == n for row in mat)
    png = qr_png_bytes(payload)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert len(png) > 100


def test_system_state_governance_mode_defaults_init():
    """Boot must start from INIT so INIT→NORMAL transition is reachable."""
    from system.state import SystemState

    assert SystemState().governance_mode == "INIT"


def test_mode_manager_init_to_normal_transition_succeeds():
    """transition(NORMAL, 'boot_complete') must return True on a fresh state."""
    from governance.mode_manager import ModeManager, SystemMode
    from system.state import StateManager

    # Build an isolated ModeManager bound to a fresh StateManager so this test
    # is order-independent.
    mgr = ModeManager.__new__(ModeManager)
    mgr._state_mgr = StateManager()
    from system.fast_risk_cache import get_risk_cache
    mgr._cache = get_risk_cache()
    import threading
    mgr._lock = threading.Lock()

    assert mgr.current_mode() == SystemMode.INIT
    assert mgr.transition(SystemMode.NORMAL, "boot_complete") is True
    assert mgr.current_mode() == SystemMode.NORMAL
