"""Tests for C-04 :mod:`state.cache.redis_store`.

Covers:

* Module-level constants and version pin.
* :class:`RedisConfig` value-object validation.
* :class:`CacheEntry` validation, ordering, ``is_live_at`` semantics.
* :class:`CommandRecord` / :class:`PipelineResult` validation.
* :func:`serialize_payload` / :func:`deserialize_payload` round trip
  and byte-stability across insertion orders.
* :func:`store_digest` canonical sort + 3-run equality (INV-15).
* :class:`RedisStore` get/set/delete/exists/ttl_remaining_ns
  semantics, TTL expiry, monotone-event-time enforcement.
* ``mget`` / ``mset`` ordering.
* ``expire`` / ``expire_at`` / ``flushdb`` behaviour.
* :class:`RedisPipeline` atomic execute + rollback on error +
  result digest.
* Lazy seams raise :class:`NotImplementedError` + reject bad config.
* AST guardrails: no forbidden top-level imports, no runtime-tier
  imports, no typed-event constructors.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib

import pytest

from state.cache import redis_store as rs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_redis_store_version_is_one() -> None:
    assert rs.REDIS_STORE_VERSION == 1


def test_new_pip_dependencies_pinned() -> None:
    assert rs.NEW_PIP_DEPENDENCIES == ("redis", "hiredis")


# ---------------------------------------------------------------------------
# RedisConfig
# ---------------------------------------------------------------------------


def test_redis_config_defaults() -> None:
    cfg = rs.RedisConfig()
    assert cfg.host == "localhost"
    assert cfg.port == 6379
    assert cfg.db == 0
    assert cfg.socket_timeout_ns == 5_000_000_000
    assert cfg.decode_responses is False


def test_redis_config_frozen_and_slotted() -> None:
    cfg = rs.RedisConfig()
    assert "__slots__" in rs.RedisConfig.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.host = "evil"  # type: ignore[misc]


def test_redis_config_rejects_empty_host() -> None:
    with pytest.raises(ValueError, match="host"):
        rs.RedisConfig(host="")


def test_redis_config_rejects_bad_port() -> None:
    with pytest.raises(ValueError, match="port"):
        rs.RedisConfig(port=0)
    with pytest.raises(ValueError, match="port"):
        rs.RedisConfig(port=70_000)


def test_redis_config_rejects_negative_db() -> None:
    with pytest.raises(ValueError, match="db"):
        rs.RedisConfig(db=-1)


def test_redis_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="socket_timeout_ns"):
        rs.RedisConfig(socket_timeout_ns=0)


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------


def test_cache_entry_basic() -> None:
    e = rs.CacheEntry(
        key="dix:pos:BTCUSDT",
        value=b"{}",
        ts_ns=1_000,
        ttl_ns=5_000,
    )
    assert e.key == "dix:pos:BTCUSDT"
    assert e.value == b"{}"
    assert e.ts_ns == 1_000
    assert e.ttl_ns == 5_000
    assert e.expires_at_ns() == 6_000


def test_cache_entry_frozen_slotted() -> None:
    e = rs.CacheEntry(key="k", value=b"x", ts_ns=1, ttl_ns=1)
    assert "__slots__" in rs.CacheEntry.__dict__
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.key = "evil"  # type: ignore[misc]


def test_cache_entry_rejects_empty_key() -> None:
    with pytest.raises(ValueError, match="key"):
        rs.CacheEntry(key="", value=b"x", ts_ns=0, ttl_ns=1)


def test_cache_entry_rejects_non_bytes_value() -> None:
    with pytest.raises(TypeError, match="bytes"):
        rs.CacheEntry(
            key="k",
            value="not-bytes",
            ts_ns=0,
            ttl_ns=1,  # type: ignore[arg-type]
        )


def test_cache_entry_rejects_negative_ts() -> None:
    with pytest.raises(ValueError, match="ts_ns"):
        rs.CacheEntry(key="k", value=b"x", ts_ns=-1, ttl_ns=1)


def test_cache_entry_rejects_non_positive_ttl() -> None:
    with pytest.raises(ValueError, match="ttl_ns"):
        rs.CacheEntry(key="k", value=b"x", ts_ns=0, ttl_ns=0)


def test_cache_entry_is_live_at() -> None:
    e = rs.CacheEntry(key="k", value=b"x", ts_ns=100, ttl_ns=50)
    assert e.is_live_at(100) is True  # boundary inclusive on start
    assert e.is_live_at(149) is True
    assert e.is_live_at(150) is False  # boundary exclusive on end
    assert e.is_live_at(99) is False  # before ts_ns


def test_cache_entry_ordering_by_key_then_ts() -> None:
    a = rs.CacheEntry(key="a", value=b"x", ts_ns=2, ttl_ns=1)
    b = rs.CacheEntry(key="a", value=b"x", ts_ns=1, ttl_ns=1)
    c = rs.CacheEntry(key="b", value=b"x", ts_ns=0, ttl_ns=1)
    assert sorted([a, b, c]) == [b, a, c]


# ---------------------------------------------------------------------------
# CommandRecord + PipelineResult
# ---------------------------------------------------------------------------


def test_command_record_validation() -> None:
    with pytest.raises(ValueError, match="op"):
        rs.CommandRecord(op="NUKE", key="k", arg=0)
    with pytest.raises(ValueError, match="key"):
        rs.CommandRecord(op="SET", key="", arg=0)
    with pytest.raises(ValueError, match="arg"):
        rs.CommandRecord(op="SET", key="k", arg=-1)


def test_pipeline_result_rejects_bad_digest_len() -> None:
    with pytest.raises(ValueError, match="digest"):
        rs.PipelineResult(commands=(), digest="abc")


# ---------------------------------------------------------------------------
# serialize / deserialize
# ---------------------------------------------------------------------------


def test_serialize_payload_byte_stable() -> None:
    a = {"a": 1, "b": 2, "c": 3}
    b = {"c": 3, "a": 1, "b": 2}
    assert rs.serialize_payload(a) == rs.serialize_payload(b)


def test_serialize_payload_round_trip() -> None:
    payload = {"side": "buy", "qty": 42, "px": 3.14}
    blob = rs.serialize_payload(payload)
    out = rs.deserialize_payload(blob)
    assert out == payload


def test_serialize_payload_rejects_non_mapping() -> None:
    with pytest.raises(TypeError, match="Mapping"):
        rs.serialize_payload([1, 2, 3])  # type: ignore[arg-type]


def test_deserialize_payload_rejects_non_bytes() -> None:
    with pytest.raises(TypeError, match="bytes"):
        rs.deserialize_payload("not bytes")  # type: ignore[arg-type]


def test_deserialize_payload_rejects_non_dict_root() -> None:
    with pytest.raises(TypeError, match="dict"):
        rs.deserialize_payload(b"[1,2,3]")


# ---------------------------------------------------------------------------
# store_digest
# ---------------------------------------------------------------------------


def test_store_digest_is_blake2b_16() -> None:
    e = rs.CacheEntry(key="k", value=b"x", ts_ns=0, ttl_ns=1)
    d = rs.store_digest([e])
    assert len(d) == 32
    int(d, 16)  # hex


def test_store_digest_canonical_sort() -> None:
    e1 = rs.CacheEntry(key="a", value=b"x", ts_ns=1, ttl_ns=1)
    e2 = rs.CacheEntry(key="b", value=b"y", ts_ns=2, ttl_ns=1)
    assert rs.store_digest([e1, e2]) == rs.store_digest([e2, e1])


def test_store_digest_changes_with_value() -> None:
    e1 = rs.CacheEntry(key="k", value=b"x", ts_ns=1, ttl_ns=1)
    e2 = rs.CacheEntry(key="k", value=b"y", ts_ns=1, ttl_ns=1)
    assert rs.store_digest([e1]) != rs.store_digest([e2])


def test_store_digest_empty() -> None:
    d = rs.store_digest([])
    assert len(d) == 32


# ---------------------------------------------------------------------------
# RedisStore core
# ---------------------------------------------------------------------------


def test_store_set_get_round_trip() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=10, ttl_ns=100)
    assert s.get("k", now_ns=10) == b"v"


def test_store_get_miss_returns_none() -> None:
    s = rs.RedisStore()
    assert s.get("nope", now_ns=0) is None


def test_store_get_after_ttl_returns_none() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=10, ttl_ns=5)
    assert s.get("k", now_ns=14) == b"v"  # still live
    assert s.get("k", now_ns=15) is None  # expired exactly at boundary


def test_store_get_before_ts_returns_none() -> None:
    """Clock anomaly — caller MUST fall back to ledger."""
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=100, ttl_ns=50)
    assert s.get("k", now_ns=99) is None


def test_store_get_validates_key() -> None:
    s = rs.RedisStore()
    with pytest.raises(ValueError):
        s.get("", now_ns=0)
    with pytest.raises(TypeError):
        s.get(123, now_ns=0)  # type: ignore[arg-type]


def test_store_get_validates_ts() -> None:
    s = rs.RedisStore()
    with pytest.raises(ValueError):
        s.get("k", now_ns=-1)
    with pytest.raises(TypeError):
        s.get("k", now_ns=1.0)  # type: ignore[arg-type]


def test_store_set_requires_monotone_event_time() -> None:
    s = rs.RedisStore()
    s.set("k", b"v1", ts_ns=10, ttl_ns=100)
    with pytest.raises(ValueError, match="monotone"):
        s.set("k", b"v2", ts_ns=5, ttl_ns=100)


def test_store_set_advances_last_ts() -> None:
    s = rs.RedisStore()
    s.set("a", b"x", ts_ns=10, ttl_ns=100)
    s.set("b", b"y", ts_ns=20, ttl_ns=100)
    # Now setting "a" again at 15 must fail because last_ts_ns is 20.
    with pytest.raises(ValueError, match="monotone"):
        s.set("a", b"z", ts_ns=15, ttl_ns=100)


def test_store_set_overwrite_at_equal_or_higher_ts() -> None:
    s = rs.RedisStore()
    s.set("k", b"v1", ts_ns=10, ttl_ns=100)
    s.set("k", b"v2", ts_ns=10, ttl_ns=100)  # equal allowed
    assert s.get("k", now_ns=10) == b"v2"
    s.set("k", b"v3", ts_ns=20, ttl_ns=100)
    assert s.get("k", now_ns=20) == b"v3"


def test_store_exists_respects_ttl() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=0, ttl_ns=5)
    assert s.exists("k", now_ns=0) is True
    assert s.exists("k", now_ns=4) is True
    assert s.exists("k", now_ns=5) is False


def test_store_ttl_remaining_ns() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=0, ttl_ns=100)
    assert s.ttl_remaining_ns("k", now_ns=0) == 100
    assert s.ttl_remaining_ns("k", now_ns=30) == 70
    assert s.ttl_remaining_ns("k", now_ns=100) is None
    assert s.ttl_remaining_ns("missing", now_ns=0) is None


def test_store_delete() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=0, ttl_ns=10)
    assert s.delete("k") is True
    assert s.delete("k") is False
    assert s.get("k", now_ns=0) is None


def test_store_entry_returns_raw() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=0, ttl_ns=5)
    e = s.entry("k")
    assert e is not None
    assert e.ts_ns == 0 and e.ttl_ns == 5
    # entry() returns the raw row even past TTL
    assert s.entry("k") is e


def test_store_keys_sorted() -> None:
    s = rs.RedisStore()
    s.set("c", b"x", ts_ns=0, ttl_ns=10)
    s.set("a", b"x", ts_ns=0, ttl_ns=10)
    s.set("b", b"x", ts_ns=0, ttl_ns=10)
    assert s.keys() == ("a", "b", "c")


def test_store_live_entries_filters_expired() -> None:
    s = rs.RedisStore()
    s.set("a", b"x", ts_ns=0, ttl_ns=10)
    s.set("b", b"y", ts_ns=0, ttl_ns=100)
    live = s.live_entries(now_ns=50)
    assert tuple(e.key for e in live) == ("b",)


def test_store_mget_mset_ordering() -> None:
    s = rs.RedisStore()
    entries = s.mset({"c": b"3", "a": b"1", "b": b"2"}, ts_ns=0, ttl_ns=100)
    # mset returns canonical-sorted order
    assert tuple(e.key for e in entries) == ("a", "b", "c")
    out = s.mget(["b", "a", "missing"], now_ns=0)
    assert out == (b"2", b"1", None)


def test_store_mset_rejects_non_mapping() -> None:
    s = rs.RedisStore()
    with pytest.raises(TypeError, match="Mapping"):
        s.mset([("a", b"1")], ts_ns=0, ttl_ns=10)  # type: ignore[arg-type]


def test_store_expire_extends_ttl() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=0, ttl_ns=10)
    assert s.expire("k", 100, ts_ns=5) is True
    assert s.get("k", now_ns=50) == b"v"  # would be expired w/o EXPIRE
    assert s.expire("missing", 100, ts_ns=5) is False


def test_store_expire_validates_inputs() -> None:
    s = rs.RedisStore()
    s.set("k", b"v", ts_ns=0, ttl_ns=10)
    with pytest.raises(ValueError, match="ttl_ns"):
        s.expire("k", 0, ts_ns=1)
    with pytest.raises(ValueError, match="monotone"):
        s.set("z", b"v", ts_ns=100, ttl_ns=10)
        s.expire("k", 50, ts_ns=10)


def test_store_expire_at_reaps_expired() -> None:
    s = rs.RedisStore()
    s.set("a", b"x", ts_ns=0, ttl_ns=5)
    s.set("b", b"y", ts_ns=0, ttl_ns=100)
    evicted = s.expire_at(now_ns=50)
    assert evicted == ("a",)
    assert s.keys() == ("b",)


def test_store_flushdb_returns_count() -> None:
    s = rs.RedisStore()
    s.set("a", b"x", ts_ns=0, ttl_ns=10)
    s.set("b", b"y", ts_ns=0, ttl_ns=10)
    assert s.flushdb() == 2
    assert s.keys() == ()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_executes_in_order() -> None:
    s = rs.RedisStore()
    result = (
        s.pipeline()
        .set("a", b"1", ts_ns=0, ttl_ns=10)
        .set("b", b"2", ts_ns=0, ttl_ns=10)
        .delete("a")
        .execute()
    )
    assert isinstance(result, rs.PipelineResult)
    assert tuple(c.op for c in result.commands) == (
        "SET",
        "SET",
        "DEL",
    )
    assert s.keys() == ("b",)


def test_pipeline_rollback_on_validation_error() -> None:
    s = rs.RedisStore()
    s.set("a", b"x", ts_ns=100, ttl_ns=100)
    # Second SET violates monotone clock — pipeline must roll back.
    p = s.pipeline()
    p.set("b", b"y", ts_ns=200, ttl_ns=10)
    p.set("a", b"z", ts_ns=50, ttl_ns=10)
    with pytest.raises(ValueError, match="monotone"):
        p.execute()
    # "a" untouched, "b" never applied
    assert s.get("a", now_ns=100) == b"x"
    assert s.get("b", now_ns=200) is None


def test_pipeline_validates_arguments_at_queue_time() -> None:
    s = rs.RedisStore()
    with pytest.raises(TypeError, match="bytes"):
        s.pipeline().set("k", "not-bytes", ts_ns=0, ttl_ns=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ttl_ns"):
        s.pipeline().expire("k", 0, ts_ns=0)
    with pytest.raises(ValueError, match="ttl_ns"):
        s.pipeline().set("k", b"v", ts_ns=0, ttl_ns=0)


def test_pipeline_requires_redis_store() -> None:
    with pytest.raises(TypeError, match="RedisStore"):
        rs.RedisPipeline(object())  # type: ignore[arg-type]


def test_pipeline_digest_byte_stable_across_runs() -> None:
    def build_and_run() -> str:
        s = rs.RedisStore()
        result = (
            s.pipeline()
            .set("a", b"1", ts_ns=0, ttl_ns=10)
            .expire("a", 100, ts_ns=1)
            .delete("a")
            .execute()
        )
        return result.digest

    a = build_and_run()
    b = build_and_run()
    c = build_and_run()
    assert a == b == c
    assert len(a) == 32


# ---------------------------------------------------------------------------
# INV-15 — 3-run replay equality of full store
# ---------------------------------------------------------------------------


def test_inv15_three_run_replay_equality() -> None:
    def build() -> str:
        s = rs.RedisStore()
        s.set("dix:pos:BTC", b"1", ts_ns=10, ttl_ns=100)
        s.set("dix:pos:ETH", b"2", ts_ns=20, ttl_ns=100)
        s.expire("dix:pos:BTC", 200, ts_ns=30)
        s.set("dix:risk:snap", b"r", ts_ns=40, ttl_ns=50)
        return rs.store_digest(s.live_entries(now_ns=40))

    digests = {build() for _ in range(3)}
    assert len(digests) == 1


# ---------------------------------------------------------------------------
# Lazy seam factories
# ---------------------------------------------------------------------------


def test_redis_client_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="shadow-equivalence"):
        rs.redis_client_factory(rs.RedisConfig())


def test_async_redis_client_factory_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="shadow-equivalence"):
        rs.async_redis_client_factory(rs.RedisConfig())


def test_redis_client_factory_rejects_bad_config() -> None:
    with pytest.raises(TypeError, match="RedisConfig"):
        rs.redis_client_factory(object())  # type: ignore[arg-type]


def test_async_redis_client_factory_rejects_bad_config() -> None:
    with pytest.raises(TypeError, match="RedisConfig"):
        rs.async_redis_client_factory(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AST guardrails — INV-15 / B1 / B27 / B28 / INV-71
# ---------------------------------------------------------------------------


_MODULE_PATH = pathlib.Path(rs.__file__)


def _parsed_module() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def test_no_forbidden_top_level_imports() -> None:
    forbidden = {
        "time",
        "datetime",
        "random",
        "asyncio",
        "os",
        "numpy",
        "torch",
        "polars",
        "redis",
        "hiredis",
        "requests",
        "aiokafka",
        "confluent_kafka",
    }
    tree = _parsed_module()
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in forbidden:
                offenders.append(node.module or "")
    assert offenders == [], f"Forbidden top-level imports in state.cache.redis_store: {offenders}"


def test_no_runtime_tier_imports() -> None:
    forbidden_roots = {
        "intelligence_engine",
        "execution_engine",
        "governance_engine",
        "evolution_engine",
        "learning_engine",
    }
    tree = _parsed_module()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in forbidden_roots:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in forbidden_roots:
                offenders.append(node.module or "")
    assert offenders == [], (
        f"Runtime-tier imports in state.cache.redis_store violate B1: {offenders}"
    )


def test_no_typed_event_constructors_called() -> None:
    """B27 / B28 / INV-71 — transport never builds typed events."""
    forbidden = {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
    }
    tree = _parsed_module()
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                offenders.append(func.id)
            elif isinstance(func, ast.Attribute) and func.attr in forbidden:
                offenders.append(func.attr)
    assert offenders == [], (
        "Typed-event constructor calls in state.cache.redis_store "
        f"violate B27/B28/INV-71: {offenders}"
    )


def test_no_top_level_clock_calls() -> None:
    """INV-15 — module load must not read any wall clock."""
    forbidden = {"time", "monotonic", "perf_counter", "now", "today"}
    tree = _parsed_module()
    offenders: list[str] = []
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                func = sub.func
                if isinstance(func, ast.Attribute) and func.attr in forbidden:
                    offenders.append(func.attr)
                elif isinstance(func, ast.Name) and func.id in forbidden:
                    offenders.append(func.id)
    assert offenders == [], (
        f"Top-level clock reads in state.cache.redis_store violate INV-15: {offenders}"
    )
