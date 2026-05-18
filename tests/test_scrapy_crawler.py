"""A-19.2 — Tests for the Scrapy-backed web-autolearn crawler.

Covers:

* :class:`ScrapyCrawlerConfig` validation (range/type checks, including
  the spec-line-1505 ``1.0`` default rate-limit).
* :class:`ScrapyCrawler` lifecycle (scaffold mode, connect, disconnect,
  idempotency).
* :meth:`ScrapyCrawler.fetch` happy path, unknown seed, runtime
  exception, order preservation, and config pass-through.
* Determinism (INV-15) — 3-run byte-identical replay equality + per-seed
  fail-soft equivalence.
* AST guards — no top-level ``scrapy`` import; no engine / governance
  / system / execution / evolution / intelligence imports; no
  ``random`` / ``time`` / ``datetime`` / ``asyncio`` / ``os`` imports;
  no typed-event construction (B27 / B28 / INV-71 reader authority
  symmetry).

No real Scrapy SDK or Twisted reactor is touched — every test injects
a deterministic fake runtime via ``runtime_factory``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from sensory.web_autolearn.contracts import RawDocument
from sensory.web_autolearn.crawler import Crawler
from sensory.web_autolearn.crawler_scrapy import (
    DEFAULT_CONCURRENT_REQUESTS,
    DEFAULT_DOWNLOAD_DELAY_SEC,
    DEFAULT_DOWNLOAD_TIMEOUT_SEC,
    DEFAULT_USER_AGENT,
    NEW_PIP_DEPENDENCIES,
    CrawlerStatus,
    ScrapyCrawler,
    ScrapyCrawlerConfig,
    ScrapyFetchResult,
    _result_to_raw_document,
    _ScrapyRuntime,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRuntime(_ScrapyRuntime):
    """In-memory runtime — returns pre-baked :class:`ScrapyFetchResult`."""

    def __init__(
        self,
        results: dict[str, ScrapyFetchResult] | None = None,
        *,
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self._results = results or {}
        self._raise_on = raise_on or {}
        self.calls: list[tuple[str, float, str]] = []
        self.closed = False

    def fetch_one(
        self,
        url: str,
        *,
        download_timeout_sec: float,
        user_agent: str,
    ) -> ScrapyFetchResult:
        self.calls.append((url, download_timeout_sec, user_agent))
        if url in self._raise_on:
            raise self._raise_on[url]
        return self._results.get(
            url,
            ScrapyFetchResult(
                ok=True,
                title="",
                body=f"body:{url}",
                status_code=200,
            ),
        )

    def close(self) -> None:
        self.closed = True


def _seeds() -> dict[str, str]:
    return {
        "seed_a": "https://a.example/feed",
        "seed_b": "https://b.example/feed",
    }


# ---------------------------------------------------------------------------
# Sentinels (pip pin + spec-mandated defaults)
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("scrapy",)


def test_spec_mandated_rate_limit_default() -> None:
    """Spec line 1505: ``max 1 req/sec default``."""
    assert DEFAULT_DOWNLOAD_DELAY_SEC == 1.0


def test_default_download_timeout_positive() -> None:
    assert DEFAULT_DOWNLOAD_TIMEOUT_SEC > 0.0


def test_default_concurrent_requests_serial() -> None:
    assert DEFAULT_CONCURRENT_REQUESTS == 1


def test_default_user_agent_non_empty() -> None:
    assert DEFAULT_USER_AGENT


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = ScrapyCrawlerConfig()
    assert cfg.download_delay_sec == DEFAULT_DOWNLOAD_DELAY_SEC
    assert cfg.download_timeout_sec == DEFAULT_DOWNLOAD_TIMEOUT_SEC
    assert cfg.concurrent_requests == DEFAULT_CONCURRENT_REQUESTS
    assert cfg.user_agent == DEFAULT_USER_AGENT
    assert cfg.obey_robots_txt is True


def test_config_accepts_zero_delay() -> None:
    cfg = ScrapyCrawlerConfig(download_delay_sec=0.0)
    assert cfg.download_delay_sec == 0.0


def test_config_rejects_negative_delay() -> None:
    with pytest.raises(ValueError):
        ScrapyCrawlerConfig(download_delay_sec=-0.5)


def test_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        ScrapyCrawlerConfig(download_timeout_sec=0.0)
    with pytest.raises(ValueError):
        ScrapyCrawlerConfig(download_timeout_sec=-1.0)


def test_config_rejects_non_positive_concurrent_requests() -> None:
    with pytest.raises(ValueError):
        ScrapyCrawlerConfig(concurrent_requests=0)
    with pytest.raises(ValueError):
        ScrapyCrawlerConfig(concurrent_requests=-1)


def test_config_rejects_empty_user_agent() -> None:
    with pytest.raises(ValueError):
        ScrapyCrawlerConfig(user_agent="")


def test_config_rejects_non_bool_robots() -> None:
    with pytest.raises(TypeError):
        ScrapyCrawlerConfig(obey_robots_txt="yes")  # type: ignore[arg-type]


def test_config_rejects_non_int_concurrent_requests() -> None:
    with pytest.raises(TypeError):
        ScrapyCrawlerConfig(concurrent_requests=1.5)  # type: ignore[arg-type]


def test_config_is_frozen() -> None:
    cfg = ScrapyCrawlerConfig()
    with pytest.raises(AttributeError):
        cfg.download_delay_sec = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Crawler construction
# ---------------------------------------------------------------------------


def test_crawler_rejects_empty_seed_urls() -> None:
    with pytest.raises(ValueError):
        ScrapyCrawler(seed_urls={})


def test_crawler_rejects_empty_seed_id() -> None:
    with pytest.raises(ValueError):
        ScrapyCrawler(seed_urls={"": "https://x.example"})


def test_crawler_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        ScrapyCrawler(seed_urls={"seed_a": ""})


def test_crawler_starts_disconnected() -> None:
    crawler = ScrapyCrawler(seed_urls=_seeds())
    assert crawler.status is CrawlerStatus.DISCONNECTED
    assert crawler.is_ready is False


def test_crawler_implements_protocol() -> None:
    crawler = ScrapyCrawler(seed_urls=_seeds())
    assert isinstance(crawler, Crawler)


def test_crawler_seed_urls_is_immutable() -> None:
    crawler = ScrapyCrawler(seed_urls=_seeds())
    with pytest.raises(TypeError):
        crawler.seed_urls["seed_a"] = "x"  # type: ignore[index]


def test_crawler_config_attached() -> None:
    cfg = ScrapyCrawlerConfig(download_delay_sec=2.5)
    crawler = ScrapyCrawler(seed_urls=_seeds(), config=cfg)
    assert crawler.config is cfg


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_connect_uses_runtime_factory() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    assert crawler.is_ready
    assert crawler.status is CrawlerStatus.CONNECTED


def test_disconnect_calls_runtime_close() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    crawler.disconnect()
    assert fake.closed is True
    assert crawler.status is CrawlerStatus.DISCONNECTED
    assert crawler.is_ready is False


def test_disconnect_is_idempotent() -> None:
    crawler = ScrapyCrawler(seed_urls=_seeds())
    crawler.disconnect()
    crawler.disconnect()
    assert crawler.status is CrawlerStatus.DISCONNECTED


def test_disconnect_swallows_close_exceptions() -> None:
    class _BadRuntime(_FakeRuntime):
        def close(self) -> None:  # noqa: D401
            raise RuntimeError("boom")

    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: _BadRuntime(),
    )
    crawler.connect()
    crawler.disconnect()
    assert crawler.is_ready is False


def test_fetch_before_connect_raises() -> None:
    crawler = ScrapyCrawler(seed_urls=_seeds())
    with pytest.raises(RuntimeError):
        crawler.fetch(["seed_a"], ts_ns=42)


def test_fetch_rejects_non_positive_ts_ns() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    with pytest.raises(ValueError):
        crawler.fetch(["seed_a"], ts_ns=0)


# ---------------------------------------------------------------------------
# Fetch behaviour
# ---------------------------------------------------------------------------


def test_fetch_happy_path_preserves_order() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_b", "seed_a"], ts_ns=42)
    assert [d.seed_id for d in docs] == ["seed_b", "seed_a"]
    assert all(d.fetched_ok for d in docs)
    assert docs[0].body == "body:https://b.example/feed"
    assert docs[1].body == "body:https://a.example/feed"


def test_fetch_unknown_seed_is_fail_soft() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_missing"], ts_ns=42)
    assert len(docs) == 1
    assert docs[0].fetched_ok is False
    assert docs[0].meta["error"] == "unknown_seed"
    assert docs[0].url.startswith("about:unknown/")


def test_fetch_exception_is_fail_soft() -> None:
    fake = _FakeRuntime(
        raise_on={
            "https://a.example/feed": TimeoutError("timeout"),
        }
    )
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a"], ts_ns=42)
    assert len(docs) == 1
    assert docs[0].fetched_ok is False
    assert docs[0].meta["error"] == "fetch_failed"
    assert docs[0].meta["error_class"] == "TimeoutError"


def test_fetch_passes_config_to_runtime() -> None:
    fake = _FakeRuntime()
    cfg = ScrapyCrawlerConfig(
        download_delay_sec=2.0,
        download_timeout_sec=5.0,
        user_agent="custom-ua",
    )
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        config=cfg,
        runtime_factory=lambda c: fake,
    )
    crawler.connect()
    crawler.fetch(["seed_a"], ts_ns=42)
    assert fake.calls == [
        ("https://a.example/feed", 5.0, "custom-ua"),
    ]


def test_fetch_carries_status_code_into_meta() -> None:
    fake = _FakeRuntime(
        results={
            "https://a.example/feed": ScrapyFetchResult(
                ok=False,
                title="",
                body="",
                status_code=404,
                error="http_404",
            ),
        }
    )
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a"], ts_ns=42)
    assert docs[0].meta["status_code"] == "404"
    assert docs[0].meta["error"] == "http_404"
    assert docs[0].fetched_ok is False


def test_fetch_emits_raw_document_typed() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a"], ts_ns=42)
    assert isinstance(docs[0], RawDocument)


def test_fetch_returns_tuple() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a"], ts_ns=42)
    assert isinstance(docs, tuple)


def test_fetch_empty_seeds_returns_empty() -> None:
    fake = _FakeRuntime()
    crawler = ScrapyCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch([], ts_ns=42)
    assert docs == ()


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_three_run_byte_identical_replay() -> None:
    def _run() -> tuple[RawDocument, ...]:
        fake = _FakeRuntime()
        crawler = ScrapyCrawler(
            seed_urls=_seeds(),
            runtime_factory=lambda cfg: fake,
        )
        crawler.connect()
        return tuple(
            crawler.fetch(
                ["seed_a", "seed_b"],
                ts_ns=1_700_000_000_000_000_000,
            )
        )

    r1 = _run()
    r2 = _run()
    r3 = _run()
    assert r1 == r2 == r3


def test_three_run_byte_identical_with_failures() -> None:
    def _run() -> tuple[RawDocument, ...]:
        fake = _FakeRuntime(
            raise_on={
                "https://a.example/feed": TimeoutError("t"),
            }
        )
        crawler = ScrapyCrawler(
            seed_urls=_seeds(),
            runtime_factory=lambda cfg: fake,
        )
        crawler.connect()
        return tuple(
            crawler.fetch(
                ["seed_a", "seed_b"],
                ts_ns=1_700_000_000_000_000_000,
            )
        )

    assert _run() == _run() == _run()


def test_result_to_raw_document_pure() -> None:
    result = ScrapyFetchResult(ok=True, title="t", body="b", status_code=200)
    a = _result_to_raw_document(result, ts_ns=42, seed_id="seed_a", url="u")
    b = _result_to_raw_document(result, ts_ns=42, seed_id="seed_a", url="u")
    assert a == b


def test_result_to_raw_document_handles_none_status() -> None:
    result = ScrapyFetchResult(ok=False, error="timeout")
    doc = _result_to_raw_document(result, ts_ns=42, seed_id="seed_a", url="u")
    assert "status_code" not in doc.meta
    assert doc.meta["error"] == "timeout"
    assert doc.fetched_ok is False


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / (
    "sensory/web_autolearn/crawler_scrapy.py"
)


def _module_ast() -> ast.Module:
    return ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))


def _toplevel_imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return names


def test_no_toplevel_scrapy_import() -> None:
    """``scrapy`` must be lazy-imported only inside the default factory."""
    imports = _toplevel_imports(_module_ast())
    assert not any(n == "scrapy" or n.startswith("scrapy.") for n in imports), (
        f"toplevel scrapy import found: {imports}"
    )


def test_no_toplevel_twisted_import() -> None:
    """Twisted must never leak to module load time."""
    imports = _toplevel_imports(_module_ast())
    assert not any(n == "twisted" or n.startswith("twisted.") for n in imports), (
        f"toplevel twisted import found: {imports}"
    )


def test_no_engine_imports() -> None:
    """B1 / B27 / B28 — sensory tier must not import engines."""
    imports = _toplevel_imports(_module_ast())
    forbidden_prefixes = (
        "governance_engine",
        "system_engine",
        "execution_engine",
        "evolution_engine",
        "intelligence_engine",
    )
    for imp in imports:
        for prefix in forbidden_prefixes:
            assert not imp.startswith(prefix), f"forbidden engine import: {imp}"


def test_no_clock_or_random_imports() -> None:
    """INV-15 / B-CLOCK — no time / random / datetime / asyncio / os."""
    imports = _toplevel_imports(_module_ast())
    forbidden = {
        "time",
        "random",
        "datetime",
        "asyncio",
        "os",
    }
    for imp in imports:
        root = imp.split(".")[0]
        assert root not in forbidden, f"forbidden import: {imp}"


def test_no_typed_event_construction() -> None:
    """B27 / B28 / INV-71 — sensory tier never constructs typed bus
    events. Only :class:`RawDocument` (the value object from the
    web-autolearn contracts) and :class:`ScrapyFetchResult` /
    :class:`RawDocument` from this module are emitted.
    """
    tree = _module_ast()
    forbidden = {
        "SignalEvent",
        "HazardEvent",
        "PatchProposal",
        "GovernanceDecision",
        "ExecutionEvent",
        "SystemEvent",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name not in forbidden, f"forbidden typed-event construction: {name}"


def test_lazy_scrapy_import_inside_factory() -> None:
    """The only ``scrapy`` import must be inside a function body."""
    tree = _module_ast()
    toplevel_scrapy = False
    func_scrapy = False
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in getattr(node, "names", []) or []:
                if getattr(alias, "name", "").startswith("scrapy"):
                    toplevel_scrapy = True
            module = getattr(node, "module", "") or ""
            if module.startswith("scrapy"):
                toplevel_scrapy = True
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name.startswith("scrapy"):
                            func_scrapy = True
                elif isinstance(sub, ast.ImportFrom):
                    if (sub.module or "").startswith("scrapy"):
                        func_scrapy = True
    assert not toplevel_scrapy
    assert func_scrapy
