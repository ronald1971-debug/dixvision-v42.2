"""
tests/test_new_modules.py
Smoke tests for modules added per manifest §6 / §7 / §8:
 - interrupt
 - system_monitor
 - governance.oracle / mode / policy_engine / emergency_policy
 - mind routers/managers
 - execution (trade_executor, emergency_executor, confirmations)
 - state (writer, stream_router, projectors, snapshots)
 - observability (metrics/traces/alerts/dashboards)
 - security (secrets, encryption, authN/authZ, audit)
 - enforcement (policy_enforcer, kill_switch)
 - core bootstrap/runtime/single_instance
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def test_interrupt_module():
    from interrupt import (
        get_dispatcher,
        get_interrupt_executor,
        get_policy_cache,
        get_resolver,
    )

    assert get_dispatcher() is not None
    cache = get_policy_cache()
    snap = cache.get_snapshot()
    assert isinstance(snap.default_action, str) and snap.default_action
    assert cache.get("FEED_SILENCE") is not None
    assert get_resolver() is not None
    assert get_interrupt_executor() is not None


def test_system_monitor_module():
    from system_monitor import (
        AnomalyWindow,
        get_hazard_detector,
        get_heartbeat_monitor,
        get_system_monitor,
        is_anomalous,
    )

    assert get_system_monitor() is not None
    assert get_hazard_detector() is not None
    assert get_heartbeat_monitor() is not None
    w = AnomalyWindow(maxlen=20, z_threshold=3.0)
    for _ in range(10):
        w.add(1.0)
    assert is_anomalous(w, 100.0) in (True, False)


def test_governance_policy_and_oracle():
    from governance.constraint_compiler import get_constraint_compiler
    from governance.emergency_policy import get_snapshot, publish_canonical_policy
    from governance.oracle.tier_l1_fast import approve_l1_fast
    from governance.oracle.tier_l2_balanced import approve_l2_balanced
    from governance.oracle.tier_l3_deep import approve_l3_deep
    from governance.policy_engine import get_policy_engine

    get_policy_engine()
    compiler = get_constraint_compiler()
    compiled = compiler.compile(portfolio_usd=100_000.0, drawdown_pct=0.02)
    compiler.publish(compiled)

    publish_canonical_policy()
    snap = get_snapshot()
    assert isinstance(snap.default_action, str) and snap.default_action

    ok1, _ = approve_l1_fast({"size_usd": 100, "portfolio_usd": 100_000})
    ok2, _ = approve_l2_balanced({"size_usd": 100, "portfolio_usd": 100_000})
    ok3, _ = approve_l3_deep({
        "size_usd": 100, "portfolio_usd": 100_000,
        "exposure_pct": 10, "correlation_pct": 20,
    })
    assert ok1 and ok2 and ok3


def test_governance_modes():
    from governance.mode.degraded_mode import enter_degraded_mode, exit_degraded_mode
    from governance.mode.halted_mode import enter_halted_mode
    from governance.mode.safe_mode import enter_safe_mode, exit_safe_mode

    assert isinstance(enter_degraded_mode("test"), bool)
    assert isinstance(exit_degraded_mode("test"), bool)
    assert isinstance(enter_safe_mode("test"), bool)
    assert isinstance(exit_safe_mode("test"), bool)
    # don't actually halt a live system in tests; just ensure callable
    assert callable(enter_halted_mode)


def test_mind_routers_and_managers():
    from mind.engine import ExecutionEvent
    from mind.execution_router import AdapterRegistration, ExecutionRouter
    from mind.order_manager import OrderStatus, get_order_manager
    from mind.portfolio_manager import get_portfolio_manager

    router = ExecutionRouter()
    router.register(AdapterRegistration(
        name="noop", priority=0,
        supports=lambda a: True,
        submit=lambda ev: {"ok": True},
    ))

    ev = ExecutionEvent(
        event_type="TRADE_EXECUTION", asset="BTCUSDT", side="BUY",
        order_type="MARKET", size_usd=10.0, price=50_000.0,
        strategy="test", confidence=0.9, latency_ns=0,
        timestamp_utc="", allowed=True,
    )
    res = router.route(ev)
    assert res is not None

    pm = get_portfolio_manager()
    pm.apply_fill(asset="BTCUSDT", side="BUY", size=0.1, price=50_000.0)
    assert pm.snapshot().exposure_usd() >= 0.0

    from mind.order_manager import Order

    om = get_order_manager()
    order = om.submit(Order(client_id="c1", asset="BTCUSDT", side="BUY",
                            size=0.1, price=50_000.0))
    om.update_status(order.order_id, OrderStatus.FILLED, filled=0.1)
    assert order.order_id not in [o.order_id for o in om.open_orders()]


def test_execution_routing():
    from execution.adapter_router import get_adapter_router
    from execution.adapters.binance import BinanceAdapter
    from execution.confirmations import get_fill_tracker, get_reconciliation
    from execution.emergency_executor import get_emergency_executor
    from execution.trade_executor import get_trade_executor

    r = get_adapter_router()
    r.register("binance", BinanceAdapter(), priority=10)
    assert r.route("BTCUSDT") is not None

    te = get_trade_executor()

    class Ev:
        event_type = "TRADE_EXECUTION"
        asset = "BTCUSDT"
        side = "BUY"
        size_usd = 10.0
        order_type = "MARKET"
        allowed = True

    out = te.execute(Ev())
    assert out.ok

    assert get_emergency_executor() is not None
    assert get_fill_tracker() is not None
    assert get_reconciliation() is not None


def test_state_writer_and_projectors():
    from state.ledger.stream_router import get_stream_router
    from state.ledger.writer import get_writer
    from state.projectors import (
        get_governance_projector,
        get_market_projector,
        get_portfolio_projector,
        get_system_projector,
    )

    w = get_writer()
    w.write("MARKET", "TICK", "test", {"asset": "BTCUSDT", "price": 50_000.0})

    router = get_stream_router()
    got = []
    router.subscribe("MARKET", lambda ev: got.append(ev))
    router.publish({"event_type": "MARKET", "payload": {"asset": "X", "price": 1.0}})
    assert got

    mp = get_market_projector()
    mp.apply({"event_type": "MARKET", "payload": {"asset": "X", "price": 1.0}})
    assert mp.snapshot().last_price_by_asset.get("X") == 1.0

    pp = get_portfolio_projector()
    pp.apply({"event_type": "MARKET", "sub_type": "TRADE_EXECUTION",
              "payload": {"asset": "X", "side": "BUY", "size_usd": 100.0}})
    assert pp.snapshot().positions.get("X") == 100.0

    sp = get_system_projector()
    sp.apply({"event_type": "HAZARD", "sub_type": "FEED_SILENCE", "payload": {}})
    assert sp.snapshot().hazard_counts.get("FEED_SILENCE") == 1

    gp = get_governance_projector()
    gp.apply({"event_type": "GOVERNANCE", "sub_type": "DECISION",
              "payload": {"outcome": "APPROVED"}})
    assert gp.snapshot().decision_counts.get("APPROVED") == 1


def test_observability_stack():
    from observability.alerts.alert_engine import AlertRule, get_alert_engine
    from observability.dashboards.cockpit_adapter import build_cockpit_snapshot
    from observability.metrics.metrics_registry import get_metrics_registry
    from observability.metrics.prometheus_exporter import render_prometheus_text
    from observability.traces.trace_manager import get_trace_manager

    reg = get_metrics_registry()
    reg.inc("tests.noop")
    assert isinstance(render_prometheus_text({"x": 1.0}), str)

    tm = get_trace_manager()
    span = tm.start("t")
    tm.end(span)
    assert tm.recent(1)

    ae = get_alert_engine()
    ae.register(AlertRule(
        name="feed_silence", predicate=lambda e: e.get("event_type") == "HAZARD",
        severity="HIGH",
    ))
    fired = ae.evaluate({"event_type": "HAZARD", "payload": {}})
    assert fired

    snap = build_cockpit_snapshot()
    assert set(snap.keys()) >= {"state", "risk", "market", "portfolio", "system", "governance", "metrics"}


def test_security_stack():
    from security import Role, audit, get_authenticator, get_authorizer, get_secrets_manager
    from security.encryption import decrypt_bytes, derive_key, encrypt_bytes

    sm = get_secrets_manager()
    sm.set("API_KEY", "sekret")
    assert sm.get("API_KEY") == "sekret"

    auth = get_authenticator()
    sess = auth.issue("operator")
    assert auth.verify(sess.token) is not None
    auth.revoke(sess.token)
    assert auth.verify(sess.token) is None

    az = get_authorizer()
    az.grant("operator", Role.OPERATOR)
    assert az.authorize("operator", "trade.place")
    assert not az.authorize("operator", "policy.edit")

    k = derive_key(b"pw", b"salt")
    ct = encrypt_bytes(k, b"hello")
    assert decrypt_bytes(k, ct) == b"hello"

    audit("UNIT_TEST", "test", {"ok": True})


def test_enforcement_and_kill_switch():
    from enforcement import arm, disarm, get_policy_enforcer, is_armed

    pe = get_policy_enforcer()
    v = pe.allow({"size_usd": 100.0, "portfolio_usd": 100_000.0})
    assert v.allowed

    disarm()
    assert not is_armed()
    arm()
    assert is_armed()


def test_core_bootstrap_and_runtime():
    from core.bootstrap import DependencyGraph, Lifecycle, load_module
    from core.runtime import (
        get_async_runtime,
        get_runtime_state,
        new_trace_id,
    )

    assert callable(load_module)
    assert Lifecycle is not None
    g = DependencyGraph()
    g.add("a")
    g.add("b", "a")
    order = g.topo_order()
    assert order.index("a") < order.index("b")

    rs = get_runtime_state()
    assert rs is not None
    assert get_async_runtime() is not None
    tid = new_trace_id()
    assert isinstance(tid, str) and tid


def test_dead_man_status_is_pure():
    """Reading status() must never trip the switch or halt trading.

    Regression test for Devin Review round 6: GET /api/safety must
    not become the call that halts trading.  Only the background
    check() may mutate.
    """
    import time

    from system.fast_risk_cache import get_risk_cache
    from system_monitor.dead_man import DeadManSwitch

    rc = get_risk_cache()
    rc.resume_trading()
    dm = DeadManSwitch(timeout_sec=0.001)
    dm.heartbeat(source="test")
    time.sleep(0.05)
    allowed_before = rc.get().trading_allowed
    s = dm.status()
    allowed_after = rc.get().trading_allowed
    assert not s.tripped
    assert allowed_before == allowed_after, "status() mutated risk cache"

    s2 = dm.check()
    assert s2.tripped
    assert not rc.get().trading_allowed, "check() did not halt trading"
    rc.resume_trading()


def test_risk_constraints_enforces_max_order_size_usd():
    """A large absolute notional on a huge portfolio must still fail."""
    from system.fast_risk_cache import RiskConstraints

    rc = RiskConstraints(
        max_order_size_usd=1_000.0,
        circuit_breaker_loss_pct=0.01,
    )
    ok, reason = rc.allows_trade(size_usd=9_999.0, portfolio_usd=10_000_000.0)
    assert not ok and "exceeds_max" in reason
    ok, _ = rc.allows_trade(size_usd=500.0, portfolio_usd=100_000.0)
    assert ok


def test_wallet_connect_expiry_uses_datetime_parse():
    """ISO approval-expiry must be parsed, not string-compared.

    Mixed Z / +00:00 / bare suffixes break lexicographic ordering;
    datetime parse normalises them.
    """
    from security.wallet_connect import _expiry_reached

    assert not _expiry_reached("")
    assert not _expiry_reached("2099-01-01T00:00:00Z")
    assert not _expiry_reached("2099-01-01T00:00:00+00:00")
    assert _expiry_reached("2000-01-01T00:00:00Z")
    assert _expiry_reached("2000-01-01T00:00:00+00:00")
    assert not _expiry_reached("not-a-date")


def test_wallet_policy_check_and_consume_is_atomic():
    """Concurrent callers must not exceed the daily cap.

    Regression test for Devin Review round 6 TOCTOU bug: 20 threads
    each try to spend $20 against a $100 system cap; exactly 5 may
    succeed and the recorded spend must equal the cap.
    """
    import threading
    from datetime import timedelta, timezone

    from security import wallet_policy as wp
    from system.time_source import utc_now

    with wp._lock:
        c = wp._connect()
        row = c.execute(
            "SELECT v FROM policy_meta WHERE k=?", (wp._BIRTH_KEY,)
        ).fetchone()
        original_birth = row["v"] if row else None
        past = utc_now() - timedelta(days=35)
        if past.tzinfo is None:
            past = past.replace(tzinfo=timezone.utc)
        c.execute(
            "INSERT OR REPLACE INTO policy_meta(k,v) VALUES (?,?)",
            (wp._BIRTH_KEY, past.isoformat()),
        )
        c.execute("DELETE FROM policy_spend")
        c.commit()

    try:
        results: list[tuple[bool, str]] = []

        def worker() -> None:
            results.append(
                wp.check_and_consume(
                    "ethereum",
                    "0xABCDEFabcdef0000000000000000000000000000",
                    usd_notional=20.0,
                )
            )

        ts = [threading.Thread(target=worker) for _ in range(20)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        ok_count = sum(1 for ok, _ in results if ok)
        assert ok_count == 5, f"TOCTOU race: {ok_count} succeeded (expected 5)"
        s = wp.snapshot()
        assert s.spent_system_24h_usd <= 100.0
    finally:
        # Restore prior state so the WARMUP test in test_tier_a_b.py
        # still sees a brand-new-policy snapshot.
        with wp._lock:
            c = wp._connect()
            c.execute("DELETE FROM policy_spend")
            if original_birth is not None:
                c.execute(
                    "INSERT OR REPLACE INTO policy_meta(k,v) VALUES (?,?)",
                    (wp._BIRTH_KEY, original_birth),
                )
            else:
                c.execute(
                    "DELETE FROM policy_meta WHERE k=?", (wp._BIRTH_KEY,)
                )
            c.commit()


# --------------------------------------------------------------------
# Executive-summary gap-fill modules: autonomy / operator / custom
# strategies / weekly scout. These verify the three new authority
# axes (autonomy tier, operator-above-all, sandboxed operator
# strategies) behave the way the manifest dictates.
# --------------------------------------------------------------------

def test_autonomy_manager_user_controlled_blocks_auto():
    from system.autonomy import AutonomyMode, get_autonomy
    mgr = get_autonomy()
    mgr.transition(AutonomyMode.USER_CONTROLLED, operator_id="test",
                   reason="unit-test")
    ok, reason = mgr.allows_auto(size_usd=1.0)
    assert not ok
    assert reason == "user_controlled_requires_approval"


def test_autonomy_manager_semi_auto_envelope():
    from system.autonomy import (
        AutonomyBudget,
        AutonomyMode,
        get_autonomy,
    )
    mgr = get_autonomy()
    # Fresh budget so the test is independent of global leak.
    mgr.set_budget(AutonomyMode.SEMI_AUTO,
                   AutonomyBudget(max_size_usd=50.0,
                                  max_trades_per_hour=3,
                                  auto_asset_allowed=False),
                   operator_id="test")
    mgr.transition(AutonomyMode.SEMI_AUTO, operator_id="test",
                   reason="unit-test")
    ok, _ = mgr.allows_auto(size_usd=49.0)
    assert ok
    ok2, r2 = mgr.allows_auto(size_usd=60.0)
    assert not ok2 and r2 == "size_exceeds_autonomy_budget"
    ok3, r3 = mgr.allows_auto(size_usd=10.0, asset_known=False)
    assert not ok3 and r3 == "new_asset_requires_approval"
    # Restore to USER_CONTROLLED so follow-on tests stay deterministic.
    mgr.transition(AutonomyMode.USER_CONTROLLED, operator_id="test",
                   reason="restore")


def test_operator_approval_lifecycle():
    import uuid as _uuid
    from security.operator import (
        ApprovalKind,
        ApprovalState,
        approve,
        deny,
        is_granted,
        pending,
        request_approval,
    )
    subj1 = f"daily_cap_test_{_uuid.uuid4().hex[:8]}"
    subj2 = f"daily_cap_test_{_uuid.uuid4().hex[:8]}"
    req = request_approval(ApprovalKind.DAILY_CAP_CHANGE,
                           subject=subj1,
                           payload={"new_cap_usd": 200.0},
                           ttl_sec=60, requested_by="test")
    assert req.state is ApprovalState.PENDING
    pend_ids = {r.request_id for r in pending(ApprovalKind.DAILY_CAP_CHANGE)}
    assert req.request_id in pend_ids
    granted = approve(req.request_id, operator_id="ronald")
    assert granted.state is ApprovalState.GRANTED
    assert is_granted(ApprovalKind.DAILY_CAP_CHANGE, subj1)
    req2 = request_approval(ApprovalKind.DAILY_CAP_CHANGE,
                            subject=subj2, ttl_sec=60)
    denied = deny(req2.request_id, operator_id="ronald", reason="too_high")
    assert denied.state is ApprovalState.DENIED
    assert not is_granted(ApprovalKind.DAILY_CAP_CHANGE, subj2)


def test_operator_two_person_gate():
    import uuid as _uuid
    from security.operator import (
        ApprovalKind,
        ApprovalState,
        approve,
        is_granted,
        request_approval,
    )
    subject = f"kill_override_test_{_uuid.uuid4().hex[:8]}"
    req = request_approval(ApprovalKind.KILL_SWITCH_OVERRIDE,
                           subject=subject, ttl_sec=60)
    r1 = approve(req.request_id, operator_id="op-1")
    # First approval: still pending, needs second distinct operator.
    assert r1.state is ApprovalState.PENDING
    assert not is_granted(ApprovalKind.KILL_SWITCH_OVERRIDE, subject)
    # Same operator again: no-op; still pending.
    r1b = approve(req.request_id, operator_id="op-1")
    assert r1b.state is ApprovalState.PENDING
    # Distinct second operator promotes.
    r2 = approve(req.request_id, operator_id="op-2")
    assert r2.state is ApprovalState.GRANTED
    assert is_granted(ApprovalKind.KILL_SWITCH_OVERRIDE, subject)


def test_custom_strategy_submission_and_sandbox():
    import uuid as _uuid

    from mind import custom_strategies as cs

    tag = _uuid.uuid4().hex[:8]
    src = (
        "from __future__ import annotations\n"
        f"# unique_tag={tag}\n"
        "SIGNAL = 'HOLD'\n"
        "def decide(tick) -> str:\n"
        "    return SIGNAL\n"
    )
    s = cs.submit(name=f"noop_hold_{tag}", source=src, author="ronald")
    assert s.state is cs.StrategyState.DRAFT
    s2 = cs.run_sandbox(s.strategy_id)
    assert s2.state is cs.StrategyState.SANDBOX_OK
    # A broken strategy must be rejected, never escape to SANDBOX_OK.
    # Top-level import error so sandbox_runner's import step fails.
    bad_src = f"# unique_tag={tag}\nimport nonexistent_mod_xyzzy_{tag}\n"
    bad = cs.submit(name=f"broken_{tag}", source=bad_src, author="ronald")
    bad2 = cs.run_sandbox(bad.strategy_id)
    assert bad2.state is cs.StrategyState.REJECTED


def test_custom_strategy_live_requires_operator_approval():
    import uuid as _uuid

    from mind import custom_strategies as cs

    tag = _uuid.uuid4().hex[:8]
    src = (
        "from __future__ import annotations\n"
        f"# unique_tag={tag}\n"
        "def decide(tick) -> str:\n    return 'HOLD'\n"
    )
    s = cs.submit(name=f"needs_approval_{tag}", source=src, author="ronald")
    cs.run_sandbox(s.strategy_id)
    cs.promote_shadow(s.strategy_id)
    cs.promote_canary(s.strategy_id)
    # Without an OPERATOR/APPROVAL_GRANTED event, go-live must fail
    # closed per manifest §1.
    raised = False
    try:
        cs.promote_live(s.strategy_id)
    except PermissionError:
        raised = True
    assert raised, "promote_live should refuse without operator approval"


def test_weekly_scout_registers_candidates_and_rejects_frozen():
    from system_monitor import weekly_scout as ws

    fired: list[str] = []

    def provider_good() -> list[ws.Candidate]:
        return [ws.Candidate(source="pytest", url="https://example.com/a",
                             head_sha="abc123", title="good",
                             category=ws.CandidateCategory.DEP_BUMP,
                             score=0.9, license_ok=True,
                             target_path="mind/sources/providers/news.py")]

    def provider_frozen() -> list[ws.Candidate]:
        fired.append("x")
        return [ws.Candidate(source="pytest", url="https://example.com/b",
                             head_sha="def456", title="frozen-attempt",
                             category=ws.CandidateCategory.REFACTOR,
                             score=0.95, license_ok=True,
                             target_path="mind/fast_execute.py")]

    ws.register_provider(provider_good)
    ws.register_provider(provider_frozen)
    tick = ws.run_once()
    urls = [c.url for c in tick.candidates]
    assert "https://example.com/a" in urls
    assert "https://example.com/b" not in urls
    frozen_errors = [e for e in tick.errors if e.startswith("refused_frozen_path:")]
    assert any("mind/fast_execute.py" in e for e in frozen_errors)


def test_constraint_compiler_caps_drawdown_at_axiom():
    from governance.constraint_compiler import get_constraint_compiler
    cc = get_constraint_compiler()
    # Caller attempts to raise the drawdown ceiling to 50 % — the
    # compiler MUST clamp down to the 4 % manifest floor.
    compiled = cc.compile(portfolio_usd=100_000.0, drawdown_pct=0.5)
    assert compiled.constraints.circuit_breaker_drawdown <= 0.04 + 1e-9
    # Negative/zero drawdown falls back to the axiom floor.
    compiled0 = cc.compile(portfolio_usd=100_000.0, drawdown_pct=0.0)
    assert compiled0.constraints.circuit_breaker_drawdown == 0.04


if __name__ == "__main__":  # pragma: no cover
    import sys

    mod = sys.modules[__name__]
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    ok = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
            ok += 1
        except Exception as e:  # pragma: no cover
            print(f"FAIL: {fn.__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
