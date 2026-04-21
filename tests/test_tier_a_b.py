"""
tests.test_tier_a_b

Coverage for the Tier-A + Tier-B additions shipped this pass:
  - authority_lint runs clean
  - charter registry covers all four voices
  - locale auto-detect returns something useable
  - AI router exposes >= 8 providers
  - API sniffer builds a candidate without raising
  - chat POST works for each voice + picks DYON when URL present
  - trader KB seeds load
  - episodic memory + strategy arbiter + alpha decay wiring
  - execution algos (VWAP/TWAP/iceberg/POV) plan shapes
  - slippage estimate + MEV guard swap
  - risk VaR/ES + position sizing
  - wallet_policy WARMUP blocks approvals + caps
  - dead-man switch + latency guard snapshots
  - cockpit app wires all new endpoints

All tests are offline, stdlib-only, no network.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_authority_lint_runs_clean():
    res = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "tools" / "authority_lint.py")],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "0 violations" in res.stdout


def test_charters_cover_four_voices():
    # importing chat ensures every voice's charter module self-registers
    from cockpit import chat as _  # noqa: F401
    from core.charter import Voice, all_charters
    cs = all_charters()
    assert set(cs.keys()) == {Voice.INDIRA, Voice.DYON, Voice.GOVERNANCE, Voice.DEVIN}
    for v, c in cs.items():
        assert c.what and c.how and c.why and c.not_do, f"{v} missing fields"


def test_locale_detect():
    from system.locale import current, supported_ui_languages
    info = current()
    assert info.language
    assert info.region
    assert isinstance(supported_ui_languages(), tuple)


def test_ai_router_providers():
    from cockpit.llm import get_router
    rows = get_router().status()
    names = {r.name for r in rows}
    wanted_fragments = ("cognition", "anthropic", "openai", "gemini",
                        "xai", "ollama", "deepseek", "perplexity")
    for frag in wanted_fragments:
        assert any(frag in n for n in names), f"missing provider ~{frag}"
    assert len(names) >= 8


def test_api_sniffer_builds_candidate():
    from mind.sources.providers.api_sniffer import ApiCandidate, propose_candidate
    c = propose_candidate("https://example.com/api/v1")
    assert isinstance(c, ApiCandidate)
    assert c.host == "example.com"


def test_chat_routes_all_voices():
    from cockpit.chat import get_chat
    from core.charter import Voice
    chat = get_chat()
    for v in (Voice.INDIRA, Voice.DYON, Voice.GOVERNANCE, Voice.DEVIN):
        t = chat.send("what is your role?", forced_voice=v)
        assert t.voice is v
        assert t.answer


def test_chat_auto_dyon_on_url():
    from cockpit.chat import get_chat
    from core.charter import Voice
    t = get_chat().send("please look at https://api.binance.com/api/v3/time")
    assert t.voice is Voice.DYON


def test_trader_kb_seed_loads():
    from mind.knowledge.seed_traders import all_seeds, seed_into
    from mind.knowledge.trader_knowledge import get_trader_knowledge
    assert len(all_seeds()) >= 80
    kb = get_trader_knowledge()
    seed_into(kb)
    assert kb.count()["traders"] >= 80


def test_episodic_and_arbiter():
    from mind.strategy_arbiter import get_arbiter
    from state.episodic_memory import get_episodic_memory
    mem = get_episodic_memory()
    mem.record(strategy="trend", symbol="SOL-USD", side="buy",
               context={}, action={}, outcome={}, reward=0.05)
    mem.record(strategy="trend", symbol="SOL-USD", side="buy",
               context={}, action={}, outcome={}, reward=-0.25)
    arb = get_arbiter()
    arb.refresh_decay()
    state = arb.state()
    assert "trend" in state and "mean_reversion" in state and "breakout" in state
    sigs = arb.propose(symbol="SOL-USD",
                       features={"ma_fast": 101.0, "ma_slow": 100.0,
                                 "price_zscore": 0.0,
                                 "high_20d": 100.0, "low_20d": 95.0, "last": 101.0})
    assert sigs
    fused = arb.fuse(sigs)
    assert fused is not None


def test_execution_algos_shapes():
    from execution.algos import plan_iceberg, plan_pov, plan_twap, plan_vwap
    assert len(plan_vwap(symbol="X", side="buy", qty=10.0,
                         window_sec=600).children) == 12
    assert len(plan_twap(symbol="X", side="buy", qty=10.0,
                        window_sec=600, slices=6).children) == 6
    assert len(plan_iceberg(symbol="X", side="buy",
                            qty=10.0, show_size=2.0).children) == 5
    p = plan_pov(symbol="X", side="buy", qty=100.0,
                 observed_volume_per_sec=10.0, participation=0.1,
                 window_sec=120.0)
    assert p.children and p.algo == "POV"


def test_slippage_and_mev_guard():
    from execution.mev_guard import prepare_swap, private_relay_for, validate_and_emit
    from execution.slippage import estimate, min_acceptable_price
    s = estimate(qty=10.0, adv_qty=1000.0, spread_bps=10.0)
    assert s.exp_slippage_bps >= 0
    assert min_acceptable_price(mid=100.0, side="buy",
                                exp_slip_bps=s.exp_slippage_bps) > 100.0
    assert private_relay_for("ethereum").startswith("https://")
    sw = prepare_swap(chain="ethereum", dex="uni_v3",
                     token_in="WETH", token_out="USDC",
                     amount_in=1.0, mid_price=3400.0,
                     adv_qty=10000.0, spread_bps=5.0,
                     max_slippage_bps=50.0, deadline_sec=60)
    assert validate_and_emit(sw) is True


def test_risk_engine():
    import random

    from risk.engine import compute_var_es, position_sizing
    random.seed(1)
    rs = [random.gauss(0.0, 0.01) for _ in range(400)]
    snap = compute_var_es(rs)
    assert snap.n_obs == 400
    assert snap.var_95 >= 0
    assert snap.regime in ("TREND", "MIXED", "RANGE", "UNKNOWN")
    size = position_sizing(equity=50_000, target_vol_annual=0.2,
                           asset_vol_daily=0.02, regime=snap.regime)
    assert size >= 0


def test_wallet_policy_warmup():
    from security import wallet_connect as wc
    from security import wallet_policy as wp
    s = wp.snapshot()
    assert s.phase is wp.Phase.WARMUP
    assert s.warmup_days_remaining > 0
    ok, reason = wp.can_sign("ethereum",
                             "0xDEADBEEF000000000000000000000000000000AA",
                             usd_notional=10.0)
    assert not ok and reason == "warmup_period"
    try:
        wc.approve_live_signing(
            wc.Chain.ETHEREUM,
            "0xDEADBEEF000000000000000000000000000000AA",
            approved_by="test", expires_utc="2099-01-01T00:00:00+00:00")
        assert False, "approve should have raised in WARMUP"
    except PermissionError:
        pass


def test_wallet_connect_watch_only():
    from security import wallet_connect as wc
    wc.connect_wallet(label="t", chain=wc.Chain.ETHEREUM,
                      backend=wc.Backend.WATCH_ONLY,
                      address="0xDEADBEEF000000000000000000000000000000BB")
    assert not wc.can_sign(wc.Chain.ETHEREUM,
                           "0xDEADBEEF000000000000000000000000000000BB",
                           usd_notional=5.0)


def test_safety_switches():
    from system_monitor.dead_man import get_dead_man
    from system_monitor.latency_guard import get_latency_guard
    dm = get_dead_man()
    dm.heartbeat()
    assert not dm.tripped()
    lg = get_latency_guard()
    lg.reset()
    for _ in range(200):
        lg.observe(500.0)
    snap = lg.snapshot()
    assert snap.n == 200 and not snap.tripped


def test_cockpit_endpoints_registered():
    from cockpit.app import create_app
    app = create_app()
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    wanted = {
        "/api/status", "/api/locale", "/api/charters",
        "/api/providers", "/api/ai", "/api/chat", "/api/risk",
        "/api/traders/count", "/api/traders/search",
        "/api/wallets", "/api/wallets/approve", "/api/wallet/policy",
        "/api/strategies", "/api/episodic/count",
        "/api/safety", "/api/safety/heartbeat",
    }
    missing = wanted - paths
    assert not missing, f"missing endpoints: {missing}"


def test_provider_summary_bounds():
    from mind.sources.providers import bootstrap_all_providers, provider_summary
    bootstrap_all_providers()
    summary = provider_summary()
    assert len(summary) >= 50
