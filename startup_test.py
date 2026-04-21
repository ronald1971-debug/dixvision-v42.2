"""
startup_test.py — Quick startup verification
Run: python startup_test.py
"""
import sys

sys.path.insert(0, ".")

def main() -> int:
    print("DIX VISION v42.2 — Startup Verification")
    print("=" * 50)
    errors = []
    checks = [
        ("immutable_core.kill_switch", lambda: __import__("immutable_core.kill_switch")),
        ("immutable_core.constants (AXIOMS)", lambda: __import__("immutable_core.constants", fromlist=["AXIOMS"]).AXIOMS.MAX_DRAWDOWN_FLOOR_PCT == 4.0),
        ("system.time_source (now)", lambda: __import__("system.time_source", fromlist=["now"]).now().sequence > 0),
        ("system.logger", lambda: __import__("system.logger", fromlist=["get_logger"]).get_logger("test")),
        ("system.state", lambda: __import__("system.state", fromlist=["get_state"]).get_state().mode == "INIT"),
        ("system.fast_risk_cache", lambda: __import__("system.fast_risk_cache", fromlist=["get_risk_cache"]).get_risk_cache().get().trading_allowed),
        ("execution.hazard.async_bus", lambda: __import__("execution.hazard.async_bus", fromlist=["get_hazard_bus"]).get_hazard_bus()),
        ("execution.hazard.event_emitter", lambda: __import__("execution.hazard.event_emitter", fromlist=["get_hazard_emitter"]).get_hazard_emitter()),
        ("governance.kernel", lambda: __import__("governance.kernel", fromlist=["get_kernel"]).get_kernel()),
        ("enforcement.decorators", lambda: __import__("enforcement.decorators", fromlist=["enforce_full"]).enforce_full),
        ("enforcement.runtime_guardian", lambda: __import__("enforcement.runtime_guardian", fromlist=["get_runtime_guardian"]).get_runtime_guardian()),
        ("state.ledger.event_store", lambda: __import__("state.ledger.event_store", fromlist=["get_event_store"]).get_event_store()),
        ("mind.engine (IndiraEngine)", lambda: __import__("mind.engine", fromlist=["IndiraEngine"]).IndiraEngine()),
        ("trading.place_trade", lambda: __import__("trading", fromlist=["place_trade"]).place_trade("BTCUSDT", 0.5)),
        ("translation.translator", lambda: __import__("translation.translator", fromlist=["get_translator"]).get_translator()),
    ]
    for name, fn in checks:
        try:
            result = fn()
            print(f"  PASS: {name}")
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            errors.append((name, e))
    print()
    if errors:
        print(f"FAILURES: {len(errors)}")
        for name, e in errors:
            print(f"  - {name}: {e}")
        return 1
    print("ALL CHECKS PASS")
    print()
    print("Start: python main.py")
    print("Tests: python tests/test_all.py")
    print("Verify: python dix.py verify")
    return 0

if __name__ == "__main__":
    sys.exit(main())
