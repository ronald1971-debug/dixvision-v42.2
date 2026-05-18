"""Tests for I-12 aiofiles canonical async ledger writer.

# ADAPTED FROM: https://github.com/Tinche/aiofiles (test idioms only).

Pins:
- module surface (NEW_PIP_DEPENDENCIES, canonical defaults)
- AsyncWriterPolicy + WriteRecord validation
- serialize_record / parse_record round-trip + byte-identical replay
- AsyncLedgerWriter buffer + auto-flush + monotone ts_ns + gap-free seq
- INV-15 byte-identical 3-run replay of a 12-record run
- stdlib factory always available; aiofiles seam runs when installed
- AST guardrails (no top-level forbidden imports, no typed-event ctors,
  no B1 runtime-tier imports, no wall-clock reads)
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
import sys
from collections.abc import Mapping
from hashlib import blake2b
from pathlib import Path

import pytest

from state.ledger import async_writer as aw_mod
from state.ledger.async_writer import (
    DEFAULT_BATCH_SIZE_MAX,
    DEFAULT_FLUSH_INTERVAL_NS,
    DEFAULT_FSYNC_ON_FLUSH,
    NEW_PIP_DEPENDENCIES,
    AsyncLedgerWriter,
    AsyncWriterPolicy,
    FlushResult,
    WriteRecord,
    enable_aiofiles_factory,
    link_record,
    parse_record,
    replay_file,
    serialize_record,
    stdlib_async_writer_factory,
)
from state.ledger.hash_chain import GENESIS_PREV_HASH

# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_module_surface_exports_canonical_names() -> None:
    expected = {
        "NEW_PIP_DEPENDENCIES",
        "DEFAULT_BATCH_SIZE_MAX",
        "DEFAULT_FLUSH_INTERVAL_NS",
        "DEFAULT_FSYNC_ON_FLUSH",
        "AsyncWriterPolicy",
        "WriteRecord",
        "FlushResult",
        "AsyncLedgerWriter",
        "serialize_record",
        "parse_record",
        "stdlib_async_writer_factory",
        "enable_aiofiles_factory",
    }
    assert expected.issubset(set(aw_mod.__all__))


def test_new_pip_dependencies_declares_aiofiles() -> None:
    assert NEW_PIP_DEPENDENCIES == ("aiofiles",)


def test_default_constants_match_canonical() -> None:
    assert DEFAULT_BATCH_SIZE_MAX == 256
    assert DEFAULT_FLUSH_INTERVAL_NS == 1_000_000_000
    assert DEFAULT_FSYNC_ON_FLUSH is True


# ---------------------------------------------------------------------------
# AsyncWriterPolicy
# ---------------------------------------------------------------------------


def test_policy_defaults_match_canonical() -> None:
    p = AsyncWriterPolicy()
    assert p.batch_size_max == DEFAULT_BATCH_SIZE_MAX
    assert p.flush_interval_ns == DEFAULT_FLUSH_INTERVAL_NS
    assert p.fsync_on_flush is DEFAULT_FSYNC_ON_FLUSH


def test_policy_is_frozen_and_slotted() -> None:
    p = AsyncWriterPolicy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.batch_size_max = 99  # type: ignore[misc]
    assert not hasattr(p, "__dict__")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size_max": 0},
        {"batch_size_max": -1},
        {"flush_interval_ns": 0},
        {"flush_interval_ns": -100},
    ],
)
def test_policy_rejects_non_positive_thresholds(kwargs: Mapping[str, int]) -> None:
    with pytest.raises(ValueError):
        AsyncWriterPolicy(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size_max": 1.5},
        {"batch_size_max": True},
        {"flush_interval_ns": 1.0},
        {"flush_interval_ns": True},
        {"fsync_on_flush": 1},
        {"fsync_on_flush": "yes"},
    ],
)
def test_policy_rejects_bad_types(kwargs: Mapping[str, object]) -> None:
    with pytest.raises(TypeError):
        AsyncWriterPolicy(**kwargs)


# ---------------------------------------------------------------------------
# WriteRecord
# ---------------------------------------------------------------------------


def _hex64(seed: int) -> str:
    return blake2b(str(seed).encode(), digest_size=32).hexdigest()


def _record(seq: int, ts_ns: int = 1_000, prev_hash: str | None = None) -> WriteRecord:
    prev = prev_hash if prev_hash is not None else GENESIS_PREV_HASH
    return link_record(
        seq=seq,
        ts_ns=ts_ns,
        kind="MODE_TRANSITION",
        payload={"to": "PAPER", "from": "SAFE"},
        prev_hash=prev,
    )


def test_record_is_frozen_and_slotted() -> None:
    r = _record(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.seq = 99  # type: ignore[misc]
    assert not hasattr(r, "__dict__")


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        ({"seq": -1}, ValueError),
        ({"seq": 1.5}, TypeError),
        ({"ts_ns": -1}, ValueError),
        ({"ts_ns": 1.0}, TypeError),
        ({"kind": ""}, ValueError),
        ({"kind": 123}, TypeError),
        ({"payload": "not-a-mapping"}, TypeError),
        ({"prev_hash": "short"}, ValueError),
        ({"prev_hash": 123}, TypeError),
        ({"hash_chain": "short"}, ValueError),
    ],
)
def test_record_validation(kwargs: Mapping[str, object], exc: type) -> None:
    base = {
        "seq": 0,
        "ts_ns": 1,
        "kind": "K",
        "payload": {"a": "1"},
        "prev_hash": GENESIS_PREV_HASH,
        "hash_chain": _hex64(0),
    }
    base.update(kwargs)
    with pytest.raises(exc):
        WriteRecord(**base)  # type: ignore[arg-type]


def test_link_record_uses_hash_chain_primitives() -> None:
    r = link_record(
        seq=0,
        ts_ns=1,
        kind="K",
        payload={"a": "1"},
        prev_hash=GENESIS_PREV_HASH,
    )
    assert len(r.hash_chain) == 64
    int(r.hash_chain, 16)


# ---------------------------------------------------------------------------
# serialize_record / parse_record
# ---------------------------------------------------------------------------


def test_serialize_record_is_byte_stable() -> None:
    r = _record(0)
    a = serialize_record(r)
    b = serialize_record(r)
    assert a == b
    assert a.endswith(b"\n")
    assert a.count(b"\x1e") == 5


def test_serialize_then_parse_round_trip() -> None:
    r = _record(0)
    bs = serialize_record(r)
    back = parse_record(bs)
    assert back == r


def test_serialize_then_parse_round_trip_empty_payload() -> None:
    r = link_record(
        seq=0,
        ts_ns=1,
        kind="K",
        payload={},
        prev_hash=GENESIS_PREV_HASH,
    )
    bs = serialize_record(r)
    back = parse_record(bs)
    assert back == r


def test_parse_record_rejects_bad_input() -> None:
    with pytest.raises(TypeError):
        parse_record("not bytes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        parse_record(b"only\x1ethree\x1efields")
    bad_utf8 = b"\xff\x1e0\x1eK\x1e\x1e" + b"0" * 64 + b"\x1e" + b"0" * 64
    with pytest.raises(ValueError):
        parse_record(bad_utf8)


# ---------------------------------------------------------------------------
# AsyncLedgerWriter — buffering + flush
# ---------------------------------------------------------------------------


def test_writer_init_rejects_bad_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        AsyncLedgerWriter(path="not a path", policy=AsyncWriterPolicy())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AsyncLedgerWriter(path=tmp_path / "x", policy="not a policy")  # type: ignore[arg-type]


def test_writer_buffers_until_batch_size_max(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(batch_size_max=3))
    r0 = _record(0)
    r1 = link_record(seq=1, ts_ns=2, kind="K", payload={"a": "1"}, prev_hash=r0.hash_chain)
    r2 = link_record(seq=2, ts_ns=3, kind="K", payload={"a": "1"}, prev_hash=r1.hash_chain)
    assert w.append(r0, ts_ns=1) is None
    assert w.append(r1, ts_ns=2) is None
    assert w.buffered_count == 2
    result = w.append(r2, ts_ns=3)
    assert result is not None
    assert result.records_written == 3
    assert result.last_seq == 2
    assert w.buffered_count == 0
    assert w.last_seq == 2
    assert p.exists()
    assert p.stat().st_size == result.bytes_written


def test_writer_buffers_until_flush_interval(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(
        path=p,
        policy=AsyncWriterPolicy(batch_size_max=100, flush_interval_ns=1_000),
    )
    r0 = _record(0)
    assert w.append(r0, ts_ns=10) is None
    r1 = link_record(seq=1, ts_ns=11, kind="K", payload={"a": "1"}, prev_hash=r0.hash_chain)
    result = w.append(r1, ts_ns=10 + 1_000)
    assert result is not None
    assert result.records_written == 2


def test_writer_force_flush_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(
        path=p,
        policy=AsyncWriterPolicy(batch_size_max=100, fsync_on_flush=False),
    )
    empty = w.flush()
    assert empty.records_written == 0
    assert empty.bytes_written == 0
    assert empty.fsync_called is False
    r0 = _record(0)
    w.append(r0, ts_ns=1)
    result = w.flush()
    assert result.records_written == 1
    assert result.last_seq == 0
    again = w.flush()
    assert again.records_written == 0


def test_writer_seq_must_be_gap_free(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(fsync_on_flush=False))
    r0 = _record(0)
    r2 = link_record(seq=2, ts_ns=2, kind="K", payload={"a": "1"}, prev_hash=r0.hash_chain)
    w.append(r0, ts_ns=1)
    with pytest.raises(ValueError):
        w.append(r2, ts_ns=2)


def test_writer_ts_ns_must_be_monotone(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(fsync_on_flush=False))
    r0 = _record(0, ts_ns=100)
    r1 = link_record(seq=1, ts_ns=50, kind="K", payload={"a": "1"}, prev_hash=r0.hash_chain)
    w.append(r0, ts_ns=100)
    with pytest.raises(ValueError):
        w.append(r1, ts_ns=50)


def test_writer_append_rejects_bad_inputs(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(fsync_on_flush=False))
    r0 = _record(0)
    with pytest.raises(TypeError):
        w.append("not a record", ts_ns=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        w.append(r0, ts_ns=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        w.append(r0, ts_ns=-1)


def test_writer_close_flushes_then_rejects_appends(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(fsync_on_flush=False))
    r0 = _record(0)
    w.append(r0, ts_ns=1)
    result = w.close()
    assert result.records_written == 1
    assert w.closed is True
    assert w.close().records_written == 0  # idempotent
    with pytest.raises(RuntimeError):
        w.append(_record(1, prev_hash=r0.hash_chain), ts_ns=2)
    with pytest.raises(RuntimeError):
        w.flush()


def test_writer_creates_parent_directories(tmp_path: Path) -> None:
    p = tmp_path / "deeper" / "subdir" / "ledger.log"
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(fsync_on_flush=False))
    w.append(_record(0), ts_ns=1)
    w.flush()
    assert p.exists()


def test_writer_appends_rather_than_truncates(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    p.write_bytes(b"existing\n")
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(fsync_on_flush=False))
    w.append(_record(0), ts_ns=1)
    w.flush()
    assert p.read_bytes().startswith(b"existing\n")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def test_replay_file_yields_round_trip_records(tmp_path: Path) -> None:
    p = tmp_path / "ledger.log"
    w = AsyncLedgerWriter(path=p, policy=AsyncWriterPolicy(fsync_on_flush=False))
    prev = GENESIS_PREV_HASH
    originals = []
    for i in range(5):
        r = link_record(seq=i, ts_ns=i + 1, kind="K", payload={"i": str(i)}, prev_hash=prev)
        originals.append(r)
        w.append(r, ts_ns=i + 1)
        prev = r.hash_chain
    w.flush()
    replayed = list(replay_file(p))
    assert replayed == originals


# ---------------------------------------------------------------------------
# INV-15: byte-identical 3-run replay
# ---------------------------------------------------------------------------


def _build_12_records() -> list[WriteRecord]:
    out: list[WriteRecord] = []
    prev = GENESIS_PREV_HASH
    for i in range(12):
        r = link_record(
            seq=i,
            ts_ns=10 * (i + 1),
            kind="MODE_TRANSITION" if i % 2 == 0 else "PATCH_DECISION",
            payload={"i": str(i), "bucket": "X" if i % 3 == 0 else "Y"},
            prev_hash=prev,
        )
        out.append(r)
        prev = r.hash_chain
    return out


def test_inv15_byte_identical_three_run_replay(tmp_path: Path) -> None:
    records = _build_12_records()
    files: list[bytes] = []
    for run in range(3):
        p = tmp_path / f"run-{run}.log"
        w = AsyncLedgerWriter(
            path=p,
            policy=AsyncWriterPolicy(batch_size_max=5, fsync_on_flush=False),
        )
        for r in records:
            w.append(r, ts_ns=r.ts_ns)
        w.flush()
        files.append(p.read_bytes())
    assert files[0] == files[1] == files[2]
    digests = {blake2b(f, digest_size=16).hexdigest() for f in files}
    assert len(digests) == 1


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def test_stdlib_factory_returns_writer(tmp_path: Path) -> None:
    p = tmp_path / "x.log"
    w = stdlib_async_writer_factory(path=p)
    assert isinstance(w, AsyncLedgerWriter)
    assert w.path == p
    assert w.policy == AsyncWriterPolicy()


def test_stdlib_factory_accepts_custom_policy(tmp_path: Path) -> None:
    p = tmp_path / "x.log"
    pol = AsyncWriterPolicy(batch_size_max=7, flush_interval_ns=99, fsync_on_flush=False)
    w = stdlib_async_writer_factory(path=p, policy=pol)
    assert w.policy is pol


def test_stdlib_factory_rejects_bad_path() -> None:
    with pytest.raises(TypeError):
        stdlib_async_writer_factory(path="not a path")  # type: ignore[arg-type]


def test_aiofiles_factory_lazy_seam(tmp_path: Path) -> None:
    p = tmp_path / "x.log"
    try:
        importlib.import_module("aiofiles")
    except ImportError:
        pytest.skip("aiofiles not installed; AST guards cover the seam")
    w = enable_aiofiles_factory(path=p)
    assert isinstance(w, AsyncLedgerWriter)


def test_aiofiles_factory_rejects_bad_path() -> None:
    if "aiofiles" not in sys.modules:
        try:
            importlib.import_module("aiofiles")
        except ImportError:
            pytest.skip("aiofiles not installed; AST guards cover the seam")
    with pytest.raises(TypeError):
        enable_aiofiles_factory(path="not a path")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _module_tree() -> ast.Module:
    src = Path(aw_mod.__file__).read_text(encoding="utf-8")
    return ast.parse(src)


def test_ast_no_forbidden_top_level_imports() -> None:
    forbidden = {
        "time",
        "datetime",
        "random",
        "asyncio",
        "numpy",
        "torch",
        "polars",
        "requests",
        "aiofiles",
    }
    tree = _module_tree()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden, f"forbidden top-level import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in forbidden, f"forbidden top-level from-import: {node.module}"


def test_ast_aiofiles_only_in_lazy_seam() -> None:
    tree = _module_tree()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "enable_aiofiles_factory":
            found = any(
                isinstance(child, ast.Import) and any(a.name == "aiofiles" for a in child.names)
                for child in ast.walk(node)
            )
            assert found, "enable_aiofiles_factory must import aiofiles function-locally"
            return
    raise AssertionError("enable_aiofiles_factory not found")


def test_ast_no_typed_event_constructors() -> None:
    forbidden = {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "LearningUpdate",
    }
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in forbidden:
                raise AssertionError(f"forbidden typed-event constructor: {name}")


def test_ast_no_runtime_tier_imports() -> None:
    """B1: state.* must not import any RUNTIME tier."""
    forbidden_prefixes = (
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "learning_engine",
    )
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in forbidden_prefixes, f"B1 violation: import {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in forbidden_prefixes, f"B1 violation: from {node.module} import …"


def test_ast_no_wall_clock_reads() -> None:
    """No ``time.time()`` / ``datetime.now()`` / ``time.monotonic_ns()`` calls."""
    forbidden_attrs = {
        ("time", "time"),
        ("time", "time_ns"),
        ("time", "monotonic"),
        ("time", "monotonic_ns"),
        ("time", "perf_counter"),
        ("time", "perf_counter_ns"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
    }
    tree = _module_tree()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            pair = (node.func.value.id, node.func.attr)
            assert pair not in forbidden_attrs, f"wall-clock read: {pair}"


# ---------------------------------------------------------------------------
# FlushResult typing
# ---------------------------------------------------------------------------


def test_flush_result_is_frozen_and_slotted() -> None:
    fr = FlushResult(bytes_written=0, records_written=0, last_seq=-1, fsync_called=False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        fr.bytes_written = 1  # type: ignore[misc]
    assert not hasattr(fr, "__dict__")
