"""Tests for TIER I I-03 — execution_engine.hot_path.fast_structs.

Pinned constraints:

  * INV-15  byte-identical 3-run replay (BLAKE2b-16 digest equality).
  * INV-15  no top-level forbidden imports (msgspec / time / datetime /
            random / asyncio / os / numpy / torch / polars / requests).
  * B1      no runtime-tier cross-imports.
  * B27/B28/INV-71  no typed-event constructors anywhere in module
            source.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from core.contracts.events import (
    EventKind,
    ExecutionStatus,
    Side,
    SignalEvent,
)
from execution_engine.hot_path import fast_structs
from execution_engine.hot_path.fast_structs import (
    FAST_STRUCT_VERSION,
    NEW_PIP_DEPENDENCIES,
    FastExecution,
    FastSignal,
    FastStructBackend,
    canonical_decode_fast_execution,
    canonical_decode_fast_signal,
    canonical_encode_fast_execution,
    canonical_encode_fast_signal,
    enable_msgspec_fast_struct_factory,
    fast_signal_from_event,
    project_fast_signals,
    replay_digest,
    stdlib_fast_struct_factory,
)

_MODULE_PATH = Path(fast_structs.__file__)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------


class TestModuleExports:
    def test_new_pip_dependencies_shape(self) -> None:
        assert isinstance(NEW_PIP_DEPENDENCIES, tuple)
        assert NEW_PIP_DEPENDENCIES == ("msgspec",)

    def test_fast_struct_version_pinned(self) -> None:
        assert FAST_STRUCT_VERSION == "1"

    def test_public_api_complete(self) -> None:
        names = {
            "FAST_STRUCT_VERSION",
            "FastExecution",
            "FastSignal",
            "FastStructBackend",
            "NEW_PIP_DEPENDENCIES",
            "canonical_decode_fast_execution",
            "canonical_decode_fast_signal",
            "canonical_encode_fast_execution",
            "canonical_encode_fast_signal",
            "enable_msgspec_fast_struct_factory",
            "fast_signal_from_event",
            "project_fast_signals",
            "replay_digest",
            "stdlib_fast_struct_factory",
        }
        assert set(fast_structs.__all__) == names


# ---------------------------------------------------------------------------
# FastSignal construction + validation
# ---------------------------------------------------------------------------


class TestFastSignal:
    def test_frozen_slotted(self) -> None:
        s = FastSignal(ts_ns=1, symbol="BTCUSDT", side=Side.BUY, confidence=0.5)
        # frozen
        with pytest.raises((AttributeError, Exception)):
            s.confidence = 0.9  # type: ignore[misc]
        # slotted — no __dict__
        assert not hasattr(s, "__dict__")

    def test_default_plugin_chain_empty(self) -> None:
        s = FastSignal(ts_ns=1, symbol="ETHUSDT", side=Side.SELL, confidence=0.3)
        assert s.plugin_chain == ()

    def test_validates_ts_ns_type(self) -> None:
        with pytest.raises(TypeError):
            FastSignal(ts_ns="0", symbol="BTC", side=Side.BUY, confidence=0.1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            FastSignal(ts_ns=True, symbol="BTC", side=Side.BUY, confidence=0.1)  # type: ignore[arg-type]

    def test_validates_ts_ns_nonnegative(self) -> None:
        with pytest.raises(ValueError):
            FastSignal(ts_ns=-1, symbol="BTC", side=Side.BUY, confidence=0.1)

    def test_validates_symbol_nonempty(self) -> None:
        with pytest.raises(ValueError):
            FastSignal(ts_ns=1, symbol="", side=Side.BUY, confidence=0.1)
        with pytest.raises(ValueError):
            FastSignal(ts_ns=1, symbol=123, side=Side.BUY, confidence=0.1)  # type: ignore[arg-type]

    def test_validates_side_enum(self) -> None:
        with pytest.raises(TypeError):
            FastSignal(ts_ns=1, symbol="BTC", side="BUY", confidence=0.1)  # type: ignore[arg-type]

    def test_validates_confidence_type(self) -> None:
        with pytest.raises(TypeError):
            FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence="0.5")  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=True)  # type: ignore[arg-type]

    def test_validates_confidence_range(self) -> None:
        with pytest.raises(ValueError):
            FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=-0.01)
        with pytest.raises(ValueError):
            FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=1.5)

    def test_accepts_confidence_bounds(self) -> None:
        FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.0)
        FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=1.0)

    def test_validates_plugin_chain_tuple(self) -> None:
        with pytest.raises(TypeError):
            FastSignal(
                ts_ns=1,
                symbol="BTC",
                side=Side.BUY,
                confidence=0.1,
                plugin_chain=["x"],  # type: ignore[arg-type]
            )

    def test_validates_plugin_chain_entries(self) -> None:
        with pytest.raises(ValueError):
            FastSignal(
                ts_ns=1,
                symbol="BTC",
                side=Side.BUY,
                confidence=0.1,
                plugin_chain=("",),
            )
        with pytest.raises(ValueError):
            FastSignal(
                ts_ns=1,
                symbol="BTC",
                side=Side.BUY,
                confidence=0.1,
                plugin_chain=(123,),  # type: ignore[arg-type]
            )

    def test_equality_structural(self) -> None:
        a = FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.4, plugin_chain=("p",))
        b = FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.4, plugin_chain=("p",))
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# FastExecution construction + validation
# ---------------------------------------------------------------------------


class TestFastExecution:
    def test_frozen_slotted(self) -> None:
        x = FastExecution(
            ts_ns=1,
            symbol="BTC",
            side=Side.SELL,
            qty=0.5,
            price=100.0,
            status=ExecutionStatus.FILLED,
        )
        with pytest.raises((AttributeError, Exception)):
            x.qty = 0.9  # type: ignore[misc]
        assert not hasattr(x, "__dict__")

    def test_validates_ts_ns(self) -> None:
        with pytest.raises(TypeError):
            FastExecution(
                ts_ns="0",  # type: ignore[arg-type]
                symbol="BTC",
                side=Side.BUY,
                qty=1.0,
                price=1.0,
                status=ExecutionStatus.FILLED,
            )
        with pytest.raises(ValueError):
            FastExecution(
                ts_ns=-1,
                symbol="BTC",
                side=Side.BUY,
                qty=1.0,
                price=1.0,
                status=ExecutionStatus.FILLED,
            )

    def test_validates_symbol(self) -> None:
        with pytest.raises(ValueError):
            FastExecution(
                ts_ns=1,
                symbol="",
                side=Side.BUY,
                qty=1.0,
                price=1.0,
                status=ExecutionStatus.FILLED,
            )

    def test_validates_side_enum(self) -> None:
        with pytest.raises(TypeError):
            FastExecution(
                ts_ns=1,
                symbol="BTC",
                side="SELL",  # type: ignore[arg-type]
                qty=1.0,
                price=1.0,
                status=ExecutionStatus.FILLED,
            )

    def test_validates_qty_nonnegative(self) -> None:
        with pytest.raises(ValueError):
            FastExecution(
                ts_ns=1,
                symbol="BTC",
                side=Side.BUY,
                qty=-0.01,
                price=1.0,
                status=ExecutionStatus.FILLED,
            )

    def test_accepts_zero_qty(self) -> None:
        # Cancellations have qty=0 — must round-trip.
        FastExecution(
            ts_ns=1,
            symbol="BTC",
            side=Side.HOLD,
            qty=0.0,
            price=0.0,
            status=ExecutionStatus.CANCELLED,
        )

    def test_validates_qty_type(self) -> None:
        with pytest.raises(TypeError):
            FastExecution(
                ts_ns=1,
                symbol="BTC",
                side=Side.BUY,
                qty=True,  # type: ignore[arg-type]
                price=1.0,
                status=ExecutionStatus.FILLED,
            )

    def test_validates_status_enum(self) -> None:
        with pytest.raises(TypeError):
            FastExecution(
                ts_ns=1,
                symbol="BTC",
                side=Side.BUY,
                qty=1.0,
                price=1.0,
                status="FILLED",  # type: ignore[arg-type]
            )

    def test_equality_structural(self) -> None:
        a = FastExecution(
            ts_ns=1,
            symbol="BTC",
            side=Side.BUY,
            qty=0.5,
            price=100.0,
            status=ExecutionStatus.FILLED,
        )
        b = FastExecution(
            ts_ns=1,
            symbol="BTC",
            side=Side.BUY,
            qty=0.5,
            price=100.0,
            status=ExecutionStatus.FILLED,
        )
        assert a == b
        assert hash(a) == hash(b)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


class TestFastSignalFromEvent:
    def test_drops_offline_fields(self) -> None:
        event = SignalEvent(
            ts_ns=42,
            symbol="BTCUSDT",
            side=Side.BUY,
            confidence=0.6,
            plugin_chain=("p1", "p2"),
            meta={"k": "v"},
            produced_by_engine="intelligence",
        )
        fast = fast_signal_from_event(event)
        assert fast == FastSignal(
            ts_ns=42,
            symbol="BTCUSDT",
            side=Side.BUY,
            confidence=0.6,
            plugin_chain=("p1", "p2"),
        )
        # FastSignal has no meta / produced_by_engine fields.
        assert not hasattr(fast, "meta")
        assert not hasattr(fast, "produced_by_engine")

    def test_pure_deterministic(self) -> None:
        event = SignalEvent(ts_ns=10, symbol="ETH", side=Side.SELL, confidence=0.25)
        first = fast_signal_from_event(event)
        second = fast_signal_from_event(event)
        third = fast_signal_from_event(event)
        assert first == second == third

    def test_rejects_non_signal(self) -> None:
        with pytest.raises(TypeError):
            fast_signal_from_event("not-a-signal")  # type: ignore[arg-type]


class TestProjectFastSignals:
    def test_preserves_order(self) -> None:
        events = [
            SignalEvent(ts_ns=2, symbol="A", side=Side.BUY, confidence=0.1),
            SignalEvent(ts_ns=1, symbol="B", side=Side.SELL, confidence=0.2),
            SignalEvent(ts_ns=3, symbol="C", side=Side.HOLD, confidence=0.3),
        ]
        out = project_fast_signals(events)
        assert isinstance(out, tuple)
        assert [s.symbol for s in out] == ["A", "B", "C"]
        assert [s.ts_ns for s in out] == [2, 1, 3]

    def test_empty_stream(self) -> None:
        assert project_fast_signals([]) == ()


# ---------------------------------------------------------------------------
# Canonical encode / decode + round-trip
# ---------------------------------------------------------------------------


class TestCanonicalEncodeSignal:
    def test_bytes_output(self) -> None:
        s = FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.5)
        blob = canonical_encode_fast_signal(s)
        assert isinstance(blob, bytes)
        assert b'"kind":"fast_signal"' in blob

    def test_byte_stable_same_input(self) -> None:
        s = FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.5, plugin_chain=("p",))
        assert (
            canonical_encode_fast_signal(s)
            == canonical_encode_fast_signal(s)
            == canonical_encode_fast_signal(s)
        )

    def test_sort_keys(self) -> None:
        s = FastSignal(ts_ns=99, symbol="ZZZ", side=Side.BUY, confidence=0.5)
        blob = canonical_encode_fast_signal(s)
        # Keys sorted lexicographically — 'confidence' is the smallest.
        idx = blob.index(b'"')
        first_key_end = blob.index(b'"', idx + 1)
        assert blob[idx : first_key_end + 1] == b'"confidence"'

    def test_rejects_non_fast_signal(self) -> None:
        with pytest.raises(TypeError):
            canonical_encode_fast_signal("not-a-signal")  # type: ignore[arg-type]


class TestCanonicalEncodeExecution:
    def test_bytes_output(self) -> None:
        x = FastExecution(
            ts_ns=1,
            symbol="BTC",
            side=Side.BUY,
            qty=1.0,
            price=100.0,
            status=ExecutionStatus.FILLED,
        )
        blob = canonical_encode_fast_execution(x)
        assert isinstance(blob, bytes)
        assert b'"kind":"fast_execution"' in blob

    def test_byte_stable_same_input(self) -> None:
        x = FastExecution(
            ts_ns=1,
            symbol="BTC",
            side=Side.BUY,
            qty=1.0,
            price=100.0,
            status=ExecutionStatus.FILLED,
        )
        assert canonical_encode_fast_execution(x) == canonical_encode_fast_execution(x)

    def test_rejects_non_fast_execution(self) -> None:
        with pytest.raises(TypeError):
            canonical_encode_fast_execution("nope")  # type: ignore[arg-type]


class TestCanonicalRoundTrip:
    def test_signal_round_trip(self) -> None:
        s = FastSignal(
            ts_ns=1,
            symbol="BTC",
            side=Side.BUY,
            confidence=0.5,
            plugin_chain=("a", "b"),
        )
        blob = canonical_encode_fast_signal(s)
        decoded = canonical_decode_fast_signal(blob)
        assert decoded == s
        # encode∘decode is the identity on bytes.
        assert canonical_encode_fast_signal(decoded) == blob

    def test_execution_round_trip(self) -> None:
        x = FastExecution(
            ts_ns=42,
            symbol="ETH",
            side=Side.SELL,
            qty=0.25,
            price=200.0,
            status=ExecutionStatus.PARTIALLY_FILLED,
        )
        blob = canonical_encode_fast_execution(x)
        decoded = canonical_decode_fast_execution(blob)
        assert decoded == x
        assert canonical_encode_fast_execution(decoded) == blob

    def test_decode_rejects_wrong_kind(self) -> None:
        s = FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.1)
        blob = canonical_encode_fast_signal(s)
        with pytest.raises(ValueError):
            canonical_decode_fast_execution(blob)

    def test_decode_rejects_bad_version(self) -> None:
        from system_engine.codec.json_codec import canonical_dumps

        bad = canonical_dumps(
            {
                "fast_struct_version": "999",
                "kind": "fast_signal",
                "ts_ns": 1,
                "symbol": "BTC",
                "side": "BUY",
                "confidence": 0.1,
                "plugin_chain": [],
            }
        )
        with pytest.raises(ValueError):
            canonical_decode_fast_signal(bad)

    def test_decode_rejects_bad_side(self) -> None:
        from system_engine.codec.json_codec import canonical_dumps

        bad = canonical_dumps(
            {
                "fast_struct_version": FAST_STRUCT_VERSION,
                "kind": "fast_signal",
                "ts_ns": 1,
                "symbol": "BTC",
                "side": "DIAGONAL",
                "confidence": 0.1,
                "plugin_chain": [],
            }
        )
        with pytest.raises(ValueError):
            canonical_decode_fast_signal(bad)

    def test_decode_rejects_bad_status(self) -> None:
        from system_engine.codec.json_codec import canonical_dumps

        bad = canonical_dumps(
            {
                "fast_struct_version": FAST_STRUCT_VERSION,
                "kind": "fast_execution",
                "ts_ns": 1,
                "symbol": "BTC",
                "side": "BUY",
                "qty": 1.0,
                "price": 1.0,
                "status": "EXPLODED",
            }
        )
        with pytest.raises(ValueError):
            canonical_decode_fast_execution(bad)

    def test_decode_rejects_non_object(self) -> None:
        from system_engine.codec.json_codec import canonical_dumps

        with pytest.raises(ValueError):
            canonical_decode_fast_signal(canonical_dumps([1, 2, 3]))
        with pytest.raises(ValueError):
            canonical_decode_fast_execution(canonical_dumps([1, 2, 3]))


# ---------------------------------------------------------------------------
# Replay digest — INV-15 anchor
# ---------------------------------------------------------------------------


class TestReplayDigest:
    def test_empty_stream(self) -> None:
        assert replay_digest([], []) == replay_digest([], []) == replay_digest([], [])

    def test_byte_identical_three_run(self) -> None:
        signals = [
            FastSignal(ts_ns=2, symbol="A", side=Side.BUY, confidence=0.1),
            FastSignal(ts_ns=1, symbol="B", side=Side.SELL, confidence=0.2),
        ]
        executions = [
            FastExecution(
                ts_ns=3,
                symbol="A",
                side=Side.BUY,
                qty=1.0,
                price=10.0,
                status=ExecutionStatus.FILLED,
            )
        ]
        d1 = replay_digest(signals, executions)
        d2 = replay_digest(signals, executions)
        d3 = replay_digest(signals, executions)
        assert d1 == d2 == d3

    def test_digest_changes_with_input(self) -> None:
        base = FastSignal(ts_ns=1, symbol="A", side=Side.BUY, confidence=0.5)
        d_a = replay_digest([base], [])
        d_b = replay_digest([FastSignal(ts_ns=1, symbol="A", side=Side.SELL, confidence=0.5)], [])
        assert d_a != d_b


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


class TestFactories:
    def test_stdlib_backend_always_available(self) -> None:
        backend = stdlib_fast_struct_factory()
        assert isinstance(backend, FastStructBackend)
        assert backend.name == "stdlib"
        s = backend.signal_factory(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.5)
        assert isinstance(s, FastSignal)

    def test_msgspec_seam_returns_none_or_backend(self) -> None:
        result = enable_msgspec_fast_struct_factory()
        # When msgspec isn't installed (CI baseline today) the lazy seam
        # returns None.  When it IS installed, the seam returns a
        # FastStructBackend whose factory bytes match the stdlib backend.
        if result is None:
            return
        assert isinstance(result, FastStructBackend)
        assert result.name == "msgspec"
        s_msg = result.signal_factory(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.5)
        s_std = FastSignal(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.5)
        assert canonical_encode_fast_signal(s_msg) == canonical_encode_fast_signal(s_std)

    def test_factory_immutable(self) -> None:
        b = stdlib_fast_struct_factory()
        with pytest.raises((AttributeError, Exception)):
            b.name = "tampered"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _load_module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _toplevel_imports(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestAstGuardrails:
    def test_no_forbidden_toplevel_imports(self) -> None:
        forbidden = {
            "msgspec",
            "time",
            "datetime",
            "random",
            "asyncio",
            "os",
            "numpy",
            "torch",
            "polars",
            "requests",
        }
        names = _toplevel_imports(_load_module_ast())
        assert forbidden.isdisjoint(names), (
            f"Forbidden top-level imports in fast_structs: {forbidden & names}"
        )

    def test_msgspec_only_inside_lazy_seam_body(self) -> None:
        tree = _load_module_ast()
        for node in ast.walk(tree):
            if isinstance(node, ast.Module):
                continue
            if (
                isinstance(node, ast.Import)
                and any(a.name.split(".")[0] == "msgspec" for a in node.names)
            ) or (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == "msgspec"
            ):
                # Walk up the parent chain — must be inside a FunctionDef
                # named enable_msgspec_fast_struct_factory.
                in_seam = False
                for parent in ast.walk(tree):
                    if (
                        isinstance(parent, ast.FunctionDef)
                        and parent.name == "enable_msgspec_fast_struct_factory"
                    ):
                        for child in ast.walk(parent):
                            if child is node:
                                in_seam = True
                                break
                    if in_seam:
                        break
                assert in_seam, (
                    "msgspec imported outside the lazy seam function "
                    "enable_msgspec_fast_struct_factory"
                )

    def test_no_typed_event_constructors(self) -> None:
        source = _MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "PatchProposal(",
            "HazardEvent(",
            "SignalEvent(",
            "ExecutionEvent(",
            "SystemEvent(",
            "LearningUpdate(",
        ):
            assert forbidden not in source, (
                f"fast_structs MUST NOT call {forbidden} — B27/B28/INV-71"
            )

    def test_no_runtime_tier_imports(self) -> None:
        forbidden = {
            "intelligence_engine",
            "governance_engine",
            "evolution_engine",
            "learning_engine",
        }
        names = _toplevel_imports(_load_module_ast())
        assert forbidden.isdisjoint(names), (
            f"fast_structs MUST NOT import runtime tiers: {forbidden & names}"
        )

    def test_no_wallclock_reads(self) -> None:
        source = _MODULE_PATH.read_text(encoding="utf-8")
        for forbidden in (
            "time.time(",
            "time.monotonic(",
            "time.time_ns(",
            "time.monotonic_ns(",
            "datetime.now(",
            "datetime.utcnow(",
        ):
            assert forbidden not in source, f"fast_structs MUST NOT call {forbidden} — INV-15"

    def test_module_imports_clean(self) -> None:
        # Importing the module twice yields the same globals — replay
        # determinism (INV-15).  We deliberately do NOT call
        # ``importlib.reload`` here because that would rebind class
        # identities on the live module and break ``isinstance`` checks
        # in later tests.
        mod1 = importlib.import_module("execution_engine.hot_path.fast_structs")
        mod2 = importlib.import_module("execution_engine.hot_path.fast_structs")
        assert mod1 is mod2
        assert mod2.NEW_PIP_DEPENDENCIES == NEW_PIP_DEPENDENCIES
        assert mod2.FAST_STRUCT_VERSION == FAST_STRUCT_VERSION


# ---------------------------------------------------------------------------
# Integration — kind enum still matches core.contracts.events
# ---------------------------------------------------------------------------


class TestIntegrationWithCoreContracts:
    def test_signal_kind_matches(self) -> None:
        event = SignalEvent(ts_ns=1, symbol="BTC", side=Side.BUY, confidence=0.5)
        assert event.kind == EventKind.SIGNAL
        fast = fast_signal_from_event(event)
        assert fast.ts_ns == event.ts_ns and fast.symbol == event.symbol

    def test_execution_status_round_trips_through_canonical(self) -> None:
        for status in ExecutionStatus:
            x = FastExecution(
                ts_ns=1,
                symbol="BTC",
                side=Side.BUY,
                qty=1.0,
                price=1.0,
                status=status,
            )
            blob = canonical_encode_fast_execution(x)
            assert canonical_decode_fast_execution(blob) == x
