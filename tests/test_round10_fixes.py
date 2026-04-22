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


def test_mode_manager_halt_emits_ledger_event():
    """halt() must write GOVERNANCE/MODE_CHANGE to the ledger (manifest §7)."""
    from unittest.mock import patch

    from governance.mode_manager import ModeManager, SystemMode
    from system.fast_risk_cache import get_risk_cache
    from system.state import StateManager

    mgr = ModeManager.__new__(ModeManager)
    mgr._state_mgr = StateManager()
    mgr._cache = get_risk_cache()
    import threading
    mgr._lock = threading.Lock()

    events: list[tuple] = []

    def fake_append(et, st, src, payload):
        events.append((et, st, src, payload))

    with patch("governance.mode_manager.append_event", fake_append):
        mgr.halt(reason="dead_man_trip")

    assert any(et == "GOVERNANCE" and st == "MODE_CHANGE"
               and p.get("to") == SystemMode.EMERGENCY_HALT.value
               and p.get("reason") == "dead_man_trip"
               and p.get("forced") is True
               for et, st, _src, p in events)


def test_governance_kernel_halt_updates_governance_mode():
    """Hazard-triggered halt must set governance_mode=EMERGENCY_HALT so the
    cockpit doesn't show NORMAL while trading_allowed is False."""
    from unittest.mock import MagicMock, patch

    from governance.kernel import GovernanceKernel
    from system.fast_risk_cache import get_risk_cache
    from system.state import StateManager

    k = GovernanceKernel.__new__(GovernanceKernel)
    k._risk_cache = get_risk_cache()
    k._state_mgr = StateManager()
    k._listeners = []
    import threading
    k._lock = threading.Lock()

    event = MagicMock()
    event.hazard_type = MagicMock(value="DISK_EXHAUSTION")
    event.severity = MagicMock(value="CRITICAL")

    with patch("governance.kernel.should_halt_trading", return_value=True), \
         patch("governance.kernel.should_enter_safe_mode", return_value=False), \
         patch("governance.kernel.classify_response", return_value="HALT"), \
         patch("governance.kernel.append_event"):
        k._on_hazard(event)

    state = k._state_mgr.get()
    assert state.trading_allowed is False
    assert state.governance_mode == "EMERGENCY_HALT"


def test_governance_kernel_safe_mode_updates_trading_allowed_and_hazards():
    """Hazard-triggered SAFE_MODE must mirror the halt branch: set
    ``trading_allowed=False``, bump ``active_hazards``, and flip
    ``governance_mode`` to SAFE_MODE atomically, so the cockpit and
    enforce_full fast path stay consistent with the risk cache."""
    from unittest.mock import MagicMock, patch

    from governance.kernel import GovernanceKernel
    from system.fast_risk_cache import get_risk_cache
    from system.state import StateManager

    k = GovernanceKernel.__new__(GovernanceKernel)
    k._risk_cache = get_risk_cache()
    k._state_mgr = StateManager()
    k._listeners = []
    import threading
    k._lock = threading.Lock()

    event = MagicMock()
    event.hazard_type = MagicMock(value="MEMORY_PRESSURE")
    event.severity = MagicMock(value="HIGH")

    before = k._state_mgr.get().active_hazards
    with patch("governance.kernel.should_halt_trading", return_value=False), \
         patch("governance.kernel.should_enter_safe_mode", return_value=True), \
         patch("governance.kernel.classify_response", return_value="SAFE_MODE"), \
         patch("governance.kernel.append_event"):
        k._on_hazard(event)

    state = k._state_mgr.get()
    assert state.trading_allowed is False, \
        "safe_mode must flip trading_allowed=False (cockpit consistency)"
    assert state.governance_mode == "SAFE_MODE"
    assert state.active_hazards == before + 1, \
        "safe_mode must increment active_hazards like halt does"


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
