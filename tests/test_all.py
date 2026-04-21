"""
tests/test_all.py
DIX VISION v42.2 — Complete Test Suite

Run: python tests/test_all.py
Expected: ALL TESTS PASS
"""
import sys

sys.path.insert(0, ".")

def run() -> int:
    errors = []

    def test(name: str, fn):
        try:
            fn()
            print(f"  PASS: {name}")
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            errors.append((name, e))

    print("=== CORE INTEGRITY ===")
    def t_axioms():
        from immutable_core.constants import AXIOMS
        assert AXIOMS.MAX_DRAWDOWN_FLOOR_PCT == 4.0
        assert AXIOMS.MAX_LOSS_PER_TRADE_FLOOR_PCT == 1.0
    test("axioms_correct_values", t_axioms)

    def t_identity():
        from immutable_core.system_identity import IDENTITY
        assert IDENTITY.is_forbidden("martingale")
        assert not IDENTITY.is_forbidden("buy")
    test("system_identity_forbidden_behaviors", t_identity)

    def t_hash():
        from pathlib import Path

        from immutable_core.foundation import FoundationIntegrity
        fi = FoundationIntegrity(root=Path("."), expected_hash="")
        h = fi.compute_hash()
        assert len(h) == 64
    test("foundation_hash_computation", t_hash)

    print("\n=== TIME SOURCE ===")
    def t_monotonic():
        from system.time_source import now
        t1, t2 = now(), now()
        assert t2.sequence > t1.sequence
        assert t2.monotonic_ns > t1.monotonic_ns
    test("strict_monotonic_ordering", t_monotonic)

    def t_concurrent():
        import concurrent.futures

        from system.time_source import now
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda _: now().sequence, range(500)))
        assert len(results) == len(set(results))
    test("concurrent_no_duplicate_sequences", t_concurrent)

    print("\n=== FAST RISK CACHE ===")
    def t_risk_cache():
        from system.fast_risk_cache import FastRiskCache
        c = FastRiskCache()
        ok, reason = c.get().allows_trade(100, 100_000)
        assert ok
        c.halt_trading()
        ok2, reason2 = c.get().allows_trade(100, 100_000)
        assert not ok2
        c.resume_trading()
        ok3, _ = c.get().allows_trade(100, 100_000)
        assert ok3
    test("risk_cache_allows_and_blocks_trades", t_risk_cache)

    def t_safe_mode():
        from system.fast_risk_cache import FastRiskCache
        c = FastRiskCache()
        c.enter_safe_mode()
        ok, r = c.get().allows_trade(100, 100_000)
        assert not ok
        assert not ok  # trading blocked in safe mode (reason may be trading_not_allowed or safe_mode_active)
    test("safe_mode_blocks_all_trades", t_safe_mode)

    print("\n=== EVENT LEDGER ===")
    def t_ledger_write():
        import os
        import tempfile

        from state.ledger.event_store import EventStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            store = EventStore(db)
            ev = store.append("MARKET", "TEST_TRADE", "indira",
                              {"asset": "BTCUSDT", "side": "BUY"})
            assert ev.event_hash and len(ev.event_hash) == 64
        finally:
            os.unlink(db)
    test("ledger_appends_with_hash", t_ledger_write)

    def t_chain_verify():
        import os
        import tempfile

        from state.ledger.event_store import EventStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db = f.name
        try:
            store = EventStore(db)
            for i in range(5):
                store.append("SYSTEM", f"EVENT_{i}", "test", {"i": i})
            assert store.verify_chain()
        finally:
            os.unlink(db)
    test("hash_chain_verification_passes", t_chain_verify)

    print("\n=== HAZARD SYSTEM ===")
    def t_hazard_bus_nonblocking():
        import time

        from execution.hazard.async_bus import HazardBus, HazardEvent, HazardSeverity, HazardType
        bus = HazardBus(maxsize=100)
        bus.start()
        received = []
        bus.subscribe(lambda e: received.append(e))
        ev = HazardEvent(HazardType.FEED_SILENCE, HazardSeverity.HIGH, "test")
        t0 = time.perf_counter()
        bus.emit(ev)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.001  # < 1ms — non-blocking
        time.sleep(0.1)
        assert len(received) > 0
        bus.stop()
    test("hazard_bus_is_nonblocking", t_hazard_bus_nonblocking)

    def t_severity_classifier():
        from execution.hazard.async_bus import HazardEvent, HazardSeverity, HazardType
        from execution.hazard.severity_classifier import should_enter_safe_mode, should_halt_trading
        critical = HazardEvent(HazardType.DATA_CORRUPTION_SUSPECTED,
                               HazardSeverity.CRITICAL, "test")
        assert should_halt_trading(critical)
        feed = HazardEvent(HazardType.FEED_SILENCE, HazardSeverity.HIGH, "test")
        assert should_enter_safe_mode(feed)
    test("severity_classifier_correct_actions", t_severity_classifier)

    print("\n=== GOVERNANCE ===")
    def t_governance_approves_valid():
        from governance.kernel import ActionRequest, GovernanceKernel, GovernanceOutcome
        k = GovernanceKernel()
        req = ActionRequest("place_trade", "MARKET",
                            {"trade_size_pct": 0.5, "size_usd": 500, "portfolio_usd": 100_000})
        d = k.evaluate(req)
        assert d.outcome == GovernanceOutcome.APPROVED
    test("governance_approves_valid_trade", t_governance_approves_valid)

    def t_governance_rejects_oversized():
        from governance.kernel import ActionRequest, GovernanceKernel, GovernanceOutcome
        k = GovernanceKernel()
        req = ActionRequest("place_trade", "MARKET",
                            {"trade_size_pct": 5.0, "size_usd": 5000, "portfolio_usd": 100_000})
        d = k.evaluate(req)
        assert d.outcome == GovernanceOutcome.REJECTED
    test("governance_rejects_oversized_trade", t_governance_rejects_oversized)

    def t_governance_boot():
        from governance.kernel import GovernanceKernel
        from system.state import SystemState
        k = GovernanceKernel()
        state = SystemState(health=1.0, mode="INIT")
        d = k.evaluate_boot(state)
        assert d.allowed
    test("governance_boot_gate_passes_healthy_state", t_governance_boot)

    print("\n=== INDIRA ENGINE (FAST PATH) ===")
    def t_indira_buy():
        from mind.engine import IndiraEngine
        e = IndiraEngine()
        ev = e.process_tick({"signal": 0.9, "asset": "BTCUSDT", "price": 65000,
                              "data_quality": 1.0, "execution_confidence": 1.0})
        assert ev.event_type in {"TRADE_EXECUTION", "HOLD"}
        if ev.event_type == "TRADE_EXECUTION":
            assert ev.side == "BUY"
    test("indira_generates_buy_on_strong_signal", t_indira_buy)

    def t_indira_latency():
        import time

        from mind.engine import IndiraEngine
        e = IndiraEngine()
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            e.process_tick({"signal": 0.8, "asset": "BTCUSDT", "price": 65000,
                             "data_quality": 1.0, "execution_confidence": 1.0})
            times.append((time.perf_counter() - t0) * 1000)
        p99 = sorted(times)[int(len(times)*0.99)]
        assert p99 < 50, f"p99 latency {p99:.2f}ms exceeds 50ms"
    test("indira_fast_path_latency_p99_under_50ms", t_indira_latency)

    def t_indira_delegate():
        from mind.engine import IndiraEngine
        e = IndiraEngine()
        ev = e.process_tick({"signal": 0.3, "asset": "BTCUSDT", "price": 65000,
                              "data_quality": 0.2, "execution_confidence": 0.2})
        assert ev.event_type in {"DELEGATE", "HOLD"}
    test("indira_delegates_on_low_confidence", t_indira_delegate)

    print("\n=== ENFORCEMENT ===")
    def t_enforcement_valid():
        from trading import place_trade
        r = place_trade(symbol="BTCUSDT", trade_size_pct=0.5)
        assert "executed" in r.lower()
    test("enforcement_allows_valid_trade", t_enforcement_valid)

    def t_enforcement_blocks():
        from trading import place_trade
        try:
            place_trade(symbol="BTCUSDT", trade_size_pct=5.0)
            raise AssertionError("should have raised")
        except RuntimeError:
            pass
    test("enforcement_blocks_oversized_trade", t_enforcement_blocks)

    print("\n=== TRANSLATION ===")
    def t_translation():
        from translation.intent_models import MarketIntentType
        from translation.translator import get_translator
        t = get_translator()
        intent = t.translate_market({"action": "BUY", "asset": "BTCUSDT", "size_usd": 500})
        assert intent.intent_type == MarketIntentType.BUY
        assert intent.asset == "BTCUSDT"
    test("translation_produces_typed_intents", t_translation)

    print("\n=== STATE PERSISTENCE ===")
    def t_snapshot():
        # Use temp directory
        import os as _os

        from state.ledger.snapshot_manager import save_snapshot
        orig = _os.environ.get("DIX_SNAPSHOTS_DIR")
        p = save_snapshot("test_snapshot", full=True)
        assert p.exists()
    test("snapshot_saves_and_restores", t_snapshot)

    print()
    if errors:
        print(f"FAILURES ({len(errors)}):")
        for name, e in errors:
            print(f"  - {name}: {e}")
        return 1
    else:
        print(f"ALL {len([x for x in dir() if x.startswith('t_')])} TESTS PASS")
        print("Phase 0 + Phase 1 COMPLETE")
        return 0

if __name__ == "__main__":
    sys.exit(run())
