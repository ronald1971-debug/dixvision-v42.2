"""A-15 — Helius adapter test suite.

Pins:

* Advisory-only contract — no SignalEvent / HazardEvent construction
  in ``execution_engine/adapters/helius.py`` (B27 / B28 / INV-71).
* No top-level ``helius_sdk`` import (lazy-import discipline).
* No clock / random / asyncio / os imports at module top (INV-15).
* No runtime engine cross-imports (B1).
* Parser determinism and value-object validation.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from execution_engine.adapters._live_base import AdapterState
from execution_engine.adapters.helius import (
    HeliusAdapter,
    InProcessHeliusTransport,
    diff_holder_snapshots,
    parse_enhanced_transaction,
)
from sensory.onchain.contracts import HolderShiftAdvisory, OnchainEvent

# ---------------------------------------------------------------------------
# Value-object validation
# ---------------------------------------------------------------------------


def test_onchain_event_requires_positive_ts_ns() -> None:
    with pytest.raises(ValueError):
        OnchainEvent(ts_ns=0, source="HELIUS", chain="SOLANA", kind="SWAP")


def test_onchain_event_requires_non_empty_source() -> None:
    with pytest.raises(ValueError):
        OnchainEvent(ts_ns=1, source="", chain="SOLANA", kind="SWAP")


def test_onchain_event_requires_non_empty_chain() -> None:
    with pytest.raises(ValueError):
        OnchainEvent(ts_ns=1, source="HELIUS", chain="", kind="SWAP")


def test_onchain_event_requires_non_empty_kind() -> None:
    with pytest.raises(ValueError):
        OnchainEvent(ts_ns=1, source="HELIUS", chain="SOLANA", kind="")


def test_onchain_event_rug_score_range() -> None:
    with pytest.raises(ValueError):
        OnchainEvent(
            ts_ns=1,
            source="HELIUS",
            chain="SOLANA",
            kind="SWAP",
            rug_score=1.5,
        )


def test_onchain_event_accepts_zero_rug_score() -> None:
    event = OnchainEvent(
        ts_ns=1,
        source="HELIUS",
        chain="SOLANA",
        kind="SWAP",
        rug_score=0.0,
    )
    assert event.rug_score == 0.0


def test_holder_shift_advisory_validates_ranges() -> None:
    with pytest.raises(ValueError):
        HolderShiftAdvisory(
            ts_ns=1,
            asset="TOKEN",
            top_holder_share_before=0.5,
            top_holder_share_after=1.2,
            holders_changed=2,
            rug_score=0.5,
        )
    with pytest.raises(ValueError):
        HolderShiftAdvisory(
            ts_ns=1,
            asset="TOKEN",
            top_holder_share_before=0.5,
            top_holder_share_after=0.7,
            holders_changed=-1,
            rug_score=0.5,
        )


def test_holder_shift_advisory_share_delta_signed() -> None:
    rec = HolderShiftAdvisory(
        ts_ns=1,
        asset="TOKEN",
        top_holder_share_before=0.4,
        top_holder_share_after=0.7,
        holders_changed=3,
        rug_score=0.55,
    )
    assert rec.share_delta == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Enhanced-transaction parsing
# ---------------------------------------------------------------------------


def _sample_swap() -> dict[str, object]:
    return {
        "signature": "sig-1",
        "type": "swap",
        "feePayer": "WALLET_A",
        "tokenTransfers": [{"mint": "MINT_TOKEN"}],
        "rugScore": 0.12,
        "meta": {"slot": 100, "fee": 5000, "ok": True},
    }


def test_parse_enhanced_transaction_normalises_kind() -> None:
    event = parse_enhanced_transaction(_sample_swap(), ts_ns=42)
    assert event.kind == "SWAP"
    assert event.signature == "sig-1"
    assert event.actor == "WALLET_A"
    assert event.asset == "MINT_TOKEN"
    assert event.rug_score == 0.12
    assert event.source == "HELIUS"
    assert event.chain == "SOLANA"


def test_parse_enhanced_transaction_unknown_kind_falls_back() -> None:
    event = parse_enhanced_transaction({"type": "weird-unknown-thing"}, ts_ns=1)
    assert event.kind == "UNKNOWN"


def test_parse_enhanced_transaction_rejects_non_mapping() -> None:
    with pytest.raises(TypeError):
        parse_enhanced_transaction([], ts_ns=1)  # type: ignore[arg-type]


def test_parse_enhanced_transaction_requires_positive_ts() -> None:
    with pytest.raises(ValueError):
        parse_enhanced_transaction(_sample_swap(), ts_ns=0)


def test_parse_enhanced_transaction_meta_sorted_keys() -> None:
    event = parse_enhanced_transaction(_sample_swap(), ts_ns=1)
    assert list(event.meta.keys()) == sorted(event.meta.keys())


def test_parse_enhanced_transaction_deterministic_replay() -> None:
    e1 = parse_enhanced_transaction(_sample_swap(), ts_ns=1)
    e2 = parse_enhanced_transaction(_sample_swap(), ts_ns=1)
    e3 = parse_enhanced_transaction(_sample_swap(), ts_ns=1)
    assert e1 == e2 == e3


def test_parse_enhanced_transaction_drops_invalid_rug_score() -> None:
    payload = _sample_swap()
    payload["rugScore"] = 99.0
    event = parse_enhanced_transaction(payload, ts_ns=1)
    assert event.rug_score is None


# ---------------------------------------------------------------------------
# Holder-snapshot diffing
# ---------------------------------------------------------------------------


def _holders(*pairs: tuple[str, float]) -> list[dict[str, object]]:
    return [{"owner": addr, "balance": bal} for addr, bal in pairs]


def test_diff_holder_snapshots_basic_concentration_increase() -> None:
    before = _holders(("A", 10.0), ("B", 10.0), ("C", 80.0))
    after = _holders(("A", 5.0), ("B", 5.0), ("C", 90.0))
    rec = diff_holder_snapshots(
        "TOKEN",
        ts_ns=1,
        before=before,
        after=after,
        top_n=1,
    )
    assert rec.asset == "TOKEN"
    assert rec.top_holder_share_before == pytest.approx(0.8)
    assert rec.top_holder_share_after == pytest.approx(0.9)
    assert rec.holders_changed == 0
    assert 0.0 <= rec.rug_score <= 1.0


def test_diff_holder_snapshots_detects_address_churn() -> None:
    before = _holders(("A", 60.0), ("B", 40.0))
    after = _holders(("C", 60.0), ("D", 40.0))
    rec = diff_holder_snapshots(
        "TOKEN",
        ts_ns=1,
        before=before,
        after=after,
        top_n=2,
    )
    assert rec.holders_changed == 4


def test_diff_holder_snapshots_rejects_empty_asset() -> None:
    with pytest.raises(ValueError):
        diff_holder_snapshots("", ts_ns=1, before=[], after=[], top_n=3)


def test_diff_holder_snapshots_rejects_non_positive_top_n() -> None:
    with pytest.raises(ValueError):
        diff_holder_snapshots("TOKEN", ts_ns=1, before=[], after=[], top_n=0)


def test_diff_holder_snapshots_deterministic_replay() -> None:
    before = _holders(("A", 1.0), ("B", 2.0), ("C", 3.0))
    after = _holders(("A", 2.0), ("B", 1.0), ("D", 4.0))
    r1 = diff_holder_snapshots("TOKEN", ts_ns=99, before=before, after=after, top_n=2)
    r2 = diff_holder_snapshots("TOKEN", ts_ns=99, before=before, after=after, top_n=2)
    r3 = diff_holder_snapshots("TOKEN", ts_ns=99, before=before, after=after, top_n=2)
    assert r1 == r2 == r3


# ---------------------------------------------------------------------------
# Adapter façade
# ---------------------------------------------------------------------------


def test_adapter_scaffold_mode_without_credentials() -> None:
    adapter = HeliusAdapter()
    adapter.connect()
    status = adapter.status()
    assert status.state is AdapterState.DISCONNECTED
    assert "DIX_HELIUS_API_KEY" in status.detail


def test_adapter_scaffold_mode_without_transport() -> None:
    adapter = HeliusAdapter(api_key="k")
    adapter.connect()
    status = adapter.status()
    assert status.state is AdapterState.DISCONNECTED
    assert "transport" in status.detail


def test_adapter_transitions_to_connecting_with_both() -> None:
    transport = InProcessHeliusTransport()
    adapter = HeliusAdapter(api_key="k", transport=transport)
    adapter.connect()
    status = adapter.status()
    assert status.state is AdapterState.CONNECTING


def test_adapter_fetches_enhanced_transactions_returns_events() -> None:
    transport = InProcessHeliusTransport(enhanced_transactions={"sig-1": _sample_swap()})
    adapter = HeliusAdapter(api_key="k", transport=transport)
    events = adapter.fetch_enhanced_transactions(["sig-1"], ts_ns=1)
    assert len(events) == 1
    assert events[0].signature == "sig-1"


def test_adapter_fetches_enhanced_transactions_scaffold_mode() -> None:
    adapter = HeliusAdapter()
    events = adapter.fetch_enhanced_transactions(["sig-1"], ts_ns=1)
    assert events == ()


def test_adapter_diff_token_holders_uses_caller_after() -> None:
    adapter = HeliusAdapter()
    before = _holders(("A", 50.0), ("B", 50.0))
    after = _holders(("A", 90.0), ("B", 10.0))
    rec = adapter.diff_token_holders(
        "TOKEN",
        ts_ns=1,
        before=before,
        after=after,
        top_n=1,
    )
    assert isinstance(rec, HolderShiftAdvisory)


def test_adapter_diff_token_holders_fetches_after_via_transport() -> None:
    transport = InProcessHeliusTransport(
        token_holders={
            "TOKEN": [
                {"owner": "A", "balance": 70.0},
                {"owner": "B", "balance": 30.0},
            ]
        }
    )
    adapter = HeliusAdapter(api_key="k", transport=transport)
    before = _holders(("A", 50.0), ("B", 50.0))
    rec = adapter.diff_token_holders("TOKEN", ts_ns=1, before=before, top_n=2)
    assert rec.top_holder_share_after == pytest.approx(1.0)


def test_adapter_diff_token_holders_requires_transport_when_after_omitted() -> None:
    adapter = HeliusAdapter()
    with pytest.raises(RuntimeError):
        adapter.diff_token_holders("TOKEN", ts_ns=1, before=[], top_n=2)


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_HELIUS_PATH = pathlib.Path("execution_engine/adapters/helius.py").resolve()


def _module_ast() -> ast.Module:
    return ast.parse(_HELIUS_PATH.read_text())


def _toplevel_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_no_toplevel_helius_sdk_import() -> None:
    names = _toplevel_imports(_module_ast())
    assert all(not n.startswith("helius_sdk") for n in names)
    assert all(not n.startswith("helius_py") for n in names)


def test_no_clock_or_random_or_async_imports() -> None:
    forbidden = {"random", "time", "datetime", "asyncio", "os"}
    names = _toplevel_imports(_module_ast())
    assert all(n.split(".", 1)[0] not in forbidden for n in names)


def test_no_runtime_engine_cross_imports() -> None:
    forbidden_prefixes = (
        "governance_engine",
        "system_engine",
        "evolution_engine",
        "intelligence_engine",
        "learning_engine",
    )
    names = _toplevel_imports(_module_ast())
    assert all(not any(n.startswith(pref) for pref in forbidden_prefixes) for n in names)


def test_no_typed_event_construction() -> None:
    forbidden = {"SignalEvent", "HazardEvent", "PatchProposal"}
    tree = _module_ast()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in forbidden:
            raise AssertionError(f"helius.py constructs forbidden typed event: {func.id}")
        if isinstance(func, ast.Attribute) and func.attr in forbidden:
            raise AssertionError(f"helius.py constructs forbidden typed event: {func.attr}")


def test_no_toplevel_random_or_clock_access() -> None:
    """No ``time.monotonic_ns()`` / ``random.random()`` style calls."""

    tree = _module_ast()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        value = node.value
        if isinstance(value, ast.Name) and value.id in {
            "time",
            "random",
            "datetime",
        }:
            raise AssertionError(f"helius.py uses forbidden clock/random: {value.id}.{node.attr}")
