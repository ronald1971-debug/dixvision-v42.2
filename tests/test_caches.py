# ADAPTED FROM: https://github.com/tkem/cachetools
# License: MIT
"""I-09 — Comprehensive deterministic-replay test suite for the
TTL + LRU cache mixins (``execution_engine.adapters._cache_mixin``)
and the cognitive response cache
(``intelligence_engine.cognitive._response_cache``).
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

from execution_engine.adapters import _cache_mixin as cache_mod
from execution_engine.adapters._cache_mixin import (
    DEFAULT_TICKER_MAXSIZE,
    DEFAULT_TICKER_TTL_NS,
    LRUCache,
    LRUPolicy,
    TTLCache,
    TTLPolicy,
    enable_cachetools_factory,
    stdlib_cache_factory,
)
from intelligence_engine.cognitive import _response_cache as resp_mod
from intelligence_engine.cognitive._response_cache import (
    DEFAULT_RESPONSE_MAXSIZE,
    ResponseCache,
    ResponseCachePolicy,
    enable_cachetools_response_factory,
    stdlib_response_cache_factory,
)

CACHE_SRC = Path(cache_mod.__file__).read_text(encoding="utf-8")
RESP_SRC = Path(resp_mod.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1) Module surface + canonical defaults
# ---------------------------------------------------------------------------


def test_cache_mixin_declares_new_pip_dependencies() -> None:
    assert cache_mod.NEW_PIP_DEPENDENCIES == ("cachetools",)


def test_response_cache_declares_new_pip_dependencies() -> None:
    assert resp_mod.NEW_PIP_DEPENDENCIES == ("cachetools",)


def test_canonical_ticker_defaults() -> None:
    assert DEFAULT_TICKER_TTL_NS == 1_000_000_000
    assert DEFAULT_TICKER_MAXSIZE == 512


def test_canonical_response_default_maxsize() -> None:
    assert DEFAULT_RESPONSE_MAXSIZE == 100


# ---------------------------------------------------------------------------
# 2) TTLPolicy + LRUPolicy validation
# ---------------------------------------------------------------------------


def test_ttl_policy_defaults() -> None:
    p = TTLPolicy()
    assert p.ttl_ns == DEFAULT_TICKER_TTL_NS
    assert p.maxsize == DEFAULT_TICKER_MAXSIZE


@pytest.mark.parametrize("bad_ttl", [0, -1, -1_000_000])
def test_ttl_policy_rejects_non_positive_ttl(bad_ttl: int) -> None:
    with pytest.raises(ValueError):
        TTLPolicy(ttl_ns=bad_ttl)


@pytest.mark.parametrize("bad_maxsize", [0, -1, -10])
def test_ttl_policy_rejects_non_positive_maxsize(bad_maxsize: int) -> None:
    with pytest.raises(ValueError):
        TTLPolicy(maxsize=bad_maxsize)


@pytest.mark.parametrize("bad_value", [1.5, "100", None, True, False])
def test_ttl_policy_rejects_non_int(bad_value: object) -> None:
    with pytest.raises(TypeError):
        TTLPolicy(ttl_ns=bad_value)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TTLPolicy(maxsize=bad_value)  # type: ignore[arg-type]


def test_ttl_policy_is_frozen() -> None:
    p = TTLPolicy()
    with pytest.raises((AttributeError, TypeError)):
        p.ttl_ns = 999  # type: ignore[misc]


def test_lru_policy_defaults() -> None:
    assert LRUPolicy().maxsize == 100


def test_lru_policy_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        LRUPolicy(maxsize=0)
    with pytest.raises(ValueError):
        LRUPolicy(maxsize=-5)


def test_lru_policy_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        LRUPolicy(maxsize=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LRUPolicy(maxsize=True)  # type: ignore[arg-type]


def test_lru_policy_is_frozen() -> None:
    p = LRUPolicy()
    with pytest.raises((AttributeError, TypeError)):
        p.maxsize = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3) TTLCache semantics
# ---------------------------------------------------------------------------


def test_ttl_cache_set_and_get_within_ttl() -> None:
    c: TTLCache[str, int] = TTLCache(policy=TTLPolicy(ttl_ns=1_000, maxsize=8))
    c.set("a", 1, ts_ns=0)
    assert c.get("a", ts_ns=500) == 1
    assert c.get("a", ts_ns=1_000) == 1  # boundary inclusive


def test_ttl_cache_expires_after_ttl() -> None:
    c: TTLCache[str, int] = TTLCache(policy=TTLPolicy(ttl_ns=1_000, maxsize=8))
    c.set("a", 1, ts_ns=0)
    assert c.get("a", ts_ns=1_001) is None  # strictly past TTL
    assert len(c) == 0  # expired entry is evicted during get


def test_ttl_cache_overflow_evicts_lru() -> None:
    c: TTLCache[str, int] = TTLCache(policy=TTLPolicy(ttl_ns=10_000, maxsize=3))
    c.set("a", 1, ts_ns=0)
    c.set("b", 2, ts_ns=1)
    c.set("c", 3, ts_ns=2)
    # Touch "a" so "b" becomes LRU.
    assert c.get("a", ts_ns=3) == 1
    c.set("d", 4, ts_ns=4)
    assert set(c.keys()) == {"a", "c", "d"}
    assert c.get("b", ts_ns=5) is None


def test_ttl_cache_re_set_refreshes_insertion_ts() -> None:
    c: TTLCache[str, int] = TTLCache(policy=TTLPolicy(ttl_ns=1_000, maxsize=4))
    c.set("a", 1, ts_ns=0)
    c.set("a", 2, ts_ns=900)
    assert c.get("a", ts_ns=1_500) == 2  # refreshed insertion time


def test_ttl_cache_delete_and_clear() -> None:
    c: TTLCache[str, int] = TTLCache(policy=TTLPolicy(ttl_ns=1_000, maxsize=4))
    c.set("a", 1, ts_ns=0)
    c.set("b", 2, ts_ns=1)
    c.delete("a")
    assert c.get("a", ts_ns=2) is None
    assert c.get("b", ts_ns=2) == 2
    c.clear()
    assert len(c) == 0


@pytest.mark.parametrize("bad_ts", [-1, "0", None, 1.5, True])
def test_ttl_cache_rejects_invalid_ts(bad_ts: object) -> None:
    c: TTLCache[str, int] = TTLCache()
    with pytest.raises((TypeError, ValueError)):
        c.set("a", 1, ts_ns=bad_ts)  # type: ignore[arg-type]


def test_ttl_cache_iteration_order_is_lru() -> None:
    c: TTLCache[str, int] = TTLCache(policy=TTLPolicy(ttl_ns=10_000, maxsize=8))
    c.set("a", 1, ts_ns=0)
    c.set("b", 2, ts_ns=1)
    c.set("c", 3, ts_ns=2)
    c.get("a", ts_ns=3)  # touch a → move to MRU
    assert tuple(iter(c)) == ("b", "c", "a")


def test_ttl_cache_replay_byte_identical() -> None:
    """INV-15 — same inputs ⇒ same final state across 3 independent runs."""

    def run() -> tuple[tuple[str, ...], int]:
        c: TTLCache[str, int] = TTLCache(policy=TTLPolicy(ttl_ns=500, maxsize=3))
        c.set("a", 1, ts_ns=0)
        c.set("b", 2, ts_ns=100)
        c.set("c", 3, ts_ns=200)
        c.get("a", ts_ns=300)
        c.set("d", 4, ts_ns=350)
        c.get("c", ts_ns=400)
        return c.keys(), len(c)

    out = [run() for _ in range(3)]
    assert out[0] == out[1] == out[2]


# ---------------------------------------------------------------------------
# 4) LRUCache semantics
# ---------------------------------------------------------------------------


def test_lru_cache_set_and_get() -> None:
    c: LRUCache[str, int] = LRUCache(policy=LRUPolicy(maxsize=3))
    c.set("a", 1)
    assert c.get("a") == 1


def test_lru_cache_overflow_evicts_strict_lru() -> None:
    c: LRUCache[str, int] = LRUCache(policy=LRUPolicy(maxsize=3))
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    # Touch "a" so "b" is now LRU.
    c.get("a")
    c.set("d", 4)
    assert c.get("b") is None
    assert set(c.keys()) == {"a", "c", "d"}


def test_lru_cache_iteration_order_is_lru() -> None:
    c: LRUCache[str, int] = LRUCache(policy=LRUPolicy(maxsize=4))
    c.set("a", 1)
    c.set("b", 2)
    c.set("c", 3)
    c.get("a")
    assert tuple(iter(c)) == ("b", "c", "a")


def test_lru_cache_re_set_moves_to_mru() -> None:
    c: LRUCache[str, int] = LRUCache(policy=LRUPolicy(maxsize=3))
    c.set("a", 1)
    c.set("b", 2)
    c.set("a", 99)
    assert tuple(c.keys()) == ("b", "a")
    assert c.get("a") == 99


def test_lru_cache_delete_and_clear() -> None:
    c: LRUCache[str, int] = LRUCache(policy=LRUPolicy(maxsize=4))
    c.set("a", 1)
    c.set("b", 2)
    c.delete("a")
    assert c.get("a") is None
    c.clear()
    assert len(c) == 0


def test_lru_cache_replay_byte_identical() -> None:
    """INV-15 — same op sequence ⇒ same final state across 3 runs."""

    def run() -> tuple[tuple[str, ...], int]:
        c: LRUCache[str, int] = LRUCache(policy=LRUPolicy(maxsize=3))
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.get("a")
        c.set("d", 4)
        return c.keys(), len(c)

    out = [run() for _ in range(3)]
    assert out[0] == out[1] == out[2]


# ---------------------------------------------------------------------------
# 5) ResponseCache (cognitive LRU half)
# ---------------------------------------------------------------------------


def test_response_cache_defaults() -> None:
    rc: ResponseCache[str, str] = ResponseCache()
    assert rc.maxsize == DEFAULT_RESPONSE_MAXSIZE


def test_response_cache_policy_round_trip() -> None:
    rc: ResponseCache[str, str] = ResponseCache(policy=ResponseCachePolicy(maxsize=7))
    assert rc.maxsize == 7


def test_response_cache_set_get_evict() -> None:
    rc: ResponseCache[str, str] = ResponseCache(maxsize=2)
    rc.set("k1", "v1")
    rc.set("k2", "v2")
    rc.set("k3", "v3")
    assert rc.get("k1") is None
    assert rc.get("k2") == "v2"
    assert rc.get("k3") == "v3"


def test_response_cache_get_touches_lru() -> None:
    rc: ResponseCache[str, str] = ResponseCache(maxsize=3)
    rc.set("a", "1")
    rc.set("b", "2")
    rc.set("c", "3")
    rc.get("a")  # MRU = a
    rc.set("d", "4")  # evicts "b"
    assert rc.get("b") is None
    assert set(rc.keys()) == {"a", "c", "d"}


def test_response_cache_replay_byte_identical() -> None:
    def run() -> tuple[tuple[str, ...], int]:
        rc: ResponseCache[str, str] = ResponseCache(maxsize=3)
        rc.set("a", "1")
        rc.set("b", "2")
        rc.set("c", "3")
        rc.get("a")
        rc.set("d", "4")
        return rc.keys(), len(rc)

    out = [run() for _ in range(3)]
    assert out[0] == out[1] == out[2]


# ---------------------------------------------------------------------------
# 6) Factories
# ---------------------------------------------------------------------------


def test_stdlib_cache_factory_returns_ttlcache() -> None:
    c = stdlib_cache_factory()
    assert isinstance(c, TTLCache)
    assert c.policy.ttl_ns == DEFAULT_TICKER_TTL_NS


def test_stdlib_response_cache_factory_returns_response_cache() -> None:
    rc = stdlib_response_cache_factory()
    assert isinstance(rc, ResponseCache)
    assert rc.maxsize == DEFAULT_RESPONSE_MAXSIZE


def test_enable_cachetools_factory_skips_when_absent() -> None:
    try:
        importlib.import_module("cachetools")
    except ImportError:
        with pytest.raises(ImportError):
            enable_cachetools_factory()
        return
    cache = enable_cachetools_factory()
    cache.set("a", 1, ts_ns=0)
    assert cache.get("a", ts_ns=10) == 1


def test_enable_cachetools_response_factory_skips_when_absent() -> None:
    try:
        importlib.import_module("cachetools")
    except ImportError:
        with pytest.raises(ImportError):
            enable_cachetools_response_factory()
        return
    rc = enable_cachetools_response_factory()
    rc.set("a", "1")
    assert rc.get("a") == "1"


# ---------------------------------------------------------------------------
# 7) AST guards — INV-15 / B1 / B27/B28/INV-71
# ---------------------------------------------------------------------------


FORBIDDEN_TOPLEVEL_MODULES = frozenset(
    {
        "cachetools",
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
)


def _toplevel_imports(src: str) -> set[str]:
    tree = ast.parse(src)
    names: set[str] = set()
    for node in tree.body:  # module-level only
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module.split(".")[0])
    return names


def test_cache_mixin_no_forbidden_toplevel_imports() -> None:
    imports = _toplevel_imports(CACHE_SRC)
    assert not (imports & FORBIDDEN_TOPLEVEL_MODULES), imports


def test_response_cache_no_forbidden_toplevel_imports() -> None:
    imports = _toplevel_imports(RESP_SRC)
    assert not (imports & FORBIDDEN_TOPLEVEL_MODULES), imports


def _local_imports_in_function(src: str, func_name: str) -> set[str]:
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        names.add(alias.name.split(".")[0])
                elif isinstance(sub, ast.ImportFrom) and sub.module is not None:
                    names.add(sub.module.split(".")[0])
    return names


def test_cachetools_imported_only_inside_enable_seam() -> None:
    seam_imports = _local_imports_in_function(CACHE_SRC, "enable_cachetools_factory")
    assert "cachetools" in seam_imports
    # And NOT anywhere else
    for name, obj in inspect.getmembers(cache_mod, inspect.isfunction):
        if name == "enable_cachetools_factory":
            continue
        src = inspect.getsource(obj)
        assert "import cachetools" not in src, name


def test_response_cachetools_imported_only_inside_enable_seam() -> None:
    seam_imports = _local_imports_in_function(RESP_SRC, "enable_cachetools_response_factory")
    assert "cachetools" in seam_imports


def test_no_typed_event_constructors() -> None:
    forbidden = {
        "PatchProposal",
        "HazardEvent",
        "SignalEvent",
        "ExecutionEvent",
        "SystemEvent",
        "LearningUpdate",
        "ExecutionIntent",
    }
    for src in (CACHE_SRC, RESP_SRC):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden, f"forbidden typed-event ctor: {node.func.id}"


def test_no_runtime_engine_cross_imports() -> None:
    """B1 — neither module imports from another runtime engine tier."""
    cache_imports = _toplevel_imports(CACHE_SRC)
    resp_imports = _toplevel_imports(RESP_SRC)
    forbidden_for_cache = {
        "intelligence_engine",
        "governance_engine",
        "system_engine",
        "learning_engine",
        "evolution_engine",
    }
    forbidden_for_resp = {
        "execution_engine",
        "governance_engine",
        "system_engine",
        "learning_engine",
        "evolution_engine",
    }
    assert not (cache_imports & forbidden_for_cache), cache_imports
    assert not (resp_imports & forbidden_for_resp), resp_imports


def test_no_wall_clock_reads_in_source() -> None:
    """No ``time.time()`` / ``datetime.now()`` / ``time.monotonic()`` calls."""
    forbidden_calls = {("time", "time"), ("time", "monotonic"), ("datetime", "now")}
    for src in (CACHE_SRC, RESP_SRC):
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if isinstance(value, ast.Name):
                    pair = (value.id, node.func.attr)
                    assert pair not in forbidden_calls, pair
