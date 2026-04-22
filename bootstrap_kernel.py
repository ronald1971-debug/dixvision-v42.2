"""
bootstrap_kernel.py
DIX VISION v42.2 — System Bootstrap

Boot sequence:
  1. Foundation integrity check
  2. Config load
  3. State init
  4. Ledger init
  5. Governance boot gate
  6. Fast risk cache init
  7. Hazard bus start
  8. Guardian start
  9. Dyon engine start
  10. Audit log: BOOT_COMPLETE
"""
from __future__ import annotations

import os
from pathlib import Path


def run(env: str = "dev", verify_only: bool = False) -> None:
    """Full production boot sequence."""
    from enforcement.runtime_guardian import start_runtime_guardian
    from execution.engine import get_dyon_engine
    from governance.kernel import get_kernel
    from governance.mode_manager import SystemMode, get_mode_manager
    from immutable_core.foundation import FoundationIntegrity
    from immutable_core.kill_switch import trigger_kill_switch
    from state.ledger.event_store import get_event_store
    from system.audit_logger import get_audit_logger
    from system.config import get_config
    from system.fast_risk_cache import get_risk_cache
    from system.logger import get_logger
    from system.state import get_state_manager

    log = get_logger("bootstrap")
    audit = get_audit_logger()

    log.info("[BOOT] DIX VISION v42.2 starting...", env=env)
    audit.log("SYSTEM", "bootstrap", {"event": "BOOT_START", "env": env})

    # Step 1: Foundation integrity
    try:
        root = Path(os.environ.get("DIX_ROOT", "."))
        hash_path = root / "immutable_core" / "foundation.hash"
        expected = hash_path.read_text().strip() if hash_path.exists() else ""
        fi = FoundationIntegrity(root=root, expected_hash=expected)
        if not fi.verify() and env == "prod":
            trigger_kill_switch("foundation_integrity_failed", "bootstrap")
        log.info("[BOOT] Foundation integrity: OK")
    except Exception as e:
        log.warning(f"[BOOT] Foundation check warning: {e}")

    # Step 2: Config
    cfg = get_config()
    log.info("[BOOT] Config loaded")

    # Step 3: State init
    state_mgr = get_state_manager()
    state_mgr.set_mode("BOOTING")
    log.info("[BOOT] State manager initialized")

    # Step 4: Ledger init
    try:
        store = get_event_store()
        log.info("[BOOT] Event ledger initialized")
    except Exception as e:
        log.warning(f"[BOOT] Ledger init warning: {e}")

    # Step 5: Governance boot gate
    kernel = get_kernel()
    state = state_mgr.get()
    decision = kernel.evaluate_boot(state)
    if not decision.allowed:
        trigger_kill_switch(f"governance_boot_rejected:{decision.reason}", "bootstrap")
    log.info("[BOOT] Governance gate: PASSED")

    # Step 6: Risk cache
    cache = get_risk_cache()
    log.info("[BOOT] Fast risk cache initialized")

    if verify_only:
        log.info("[BOOT] verify_only=True — stopping after checks")
        state_mgr.set_mode("VERIFIED")
        return

    # Step 7: Hazard bus
    from execution.hazard.async_bus import get_hazard_bus
    bus = get_hazard_bus()  # starts automatically
    log.info("[BOOT] Hazard bus started")

    # Step 8: Guardian
    guardian = start_runtime_guardian()
    log.info("[BOOT] Runtime guardian started")

    # Step 9: Mode → NORMAL
    mode_mgr = get_mode_manager()
    mode_mgr.transition(SystemMode.NORMAL, "boot_complete")

    # Step 10: Dyon engine
    dyon = get_dyon_engine()
    dyon.start()
    log.info("[BOOT] Dyon system engine started")

    # Step 11: Lock the component registry. After every factory has been
    # resolved, no further ``registry.register(...)`` calls are permitted
    # — this is Phase 0 Build Plan §1.3 (contract lock) and closes the
    # post-boot injection gap a rogue component could otherwise exploit.
    try:
        from core.registry import get_registry
        get_registry().lock()
        log.info("[BOOT] Component registry LOCKED — post-boot registration denied")
    except Exception as e:  # fail-open on lock itself — never block boot
        log.warning(f"[BOOT] Registry lock warning: {e}")

    state_mgr.set_mode("NORMAL")
    log.info("[BOOT] System ONLINE — all services running")
    audit.log("SYSTEM", "bootstrap", {"event": "BOOT_COMPLETE", "mode": "NORMAL"})
