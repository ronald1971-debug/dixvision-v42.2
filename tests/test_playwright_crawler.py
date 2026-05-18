"""A-16 — Tests for the Playwright-backed web-autolearn crawler.

Covers:

* :class:`PlaywrightCrawlerConfig` validation (range/type checks).
* :class:`PlaywrightCrawler` lifecycle (scaffold mode, connect,
  disconnect, idempotency).
* :meth:`PlaywrightCrawler.fetch` happy path, unknown seed, runtime
  exception, and order preservation.
* Determinism (INV-15) — 3-run byte-identical replay equality.
* AST guards — no top-level ``playwright`` import; no engine /
  governance / system / execution / evolution import; no
  ``random`` / ``time`` / ``datetime`` import; no audit ledger
  writes.

No real Playwright SDK or Chromium binary is touched — every test
injects a deterministic fake runtime via ``runtime_factory``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from sensory.web_autolearn.contracts import RawDocument
from sensory.web_autolearn.crawler import Crawler
from sensory.web_autolearn.crawler_playwright import (
    DEFAULT_TIMEOUT_MS,
    DEFAULT_USER_AGENT,
    CrawlerStatus,
    PlaywrightCrawler,
    PlaywrightCrawlerConfig,
    _FetchResult,
    _PlaywrightRuntime,
    _result_to_raw_document,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeRuntime(_PlaywrightRuntime):
    """In-memory runtime — returns pre-baked :class:`_FetchResult`."""

    def __init__(
        self,
        results: dict[str, _FetchResult] | None = None,
        *,
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self._results = results or {}
        self._raise_on = raise_on or {}
        self.calls: list[tuple[str, int, str, str]] = []
        self.closed = False

    def fetch_one(
        self,
        url: str,
        *,
        timeout_ms: int,
        user_agent: str,
        locale: str,
    ) -> _FetchResult:
        self.calls.append((url, timeout_ms, user_agent, locale))
        if url in self._raise_on:
            raise self._raise_on[url]
        return self._results.get(
            url,
            _FetchResult(
                ok=True,
                title=f"title:{url}",
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
# Config validation
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = PlaywrightCrawlerConfig()
    assert cfg.timeout_ms == DEFAULT_TIMEOUT_MS
    assert cfg.user_agent == DEFAULT_USER_AGENT
    assert cfg.locale == ""
    assert cfg.headless is True


def test_config_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError):
        PlaywrightCrawlerConfig(timeout_ms=0)
    with pytest.raises(ValueError):
        PlaywrightCrawlerConfig(timeout_ms=-1)


def test_config_rejects_empty_user_agent() -> None:
    with pytest.raises(ValueError):
        PlaywrightCrawlerConfig(user_agent="")


def test_config_rejects_non_str_locale() -> None:
    with pytest.raises(TypeError):
        PlaywrightCrawlerConfig(locale=42)  # type: ignore[arg-type]


def test_config_rejects_non_bool_headless() -> None:
    with pytest.raises(TypeError):
        PlaywrightCrawlerConfig(headless="yes")  # type: ignore[arg-type]


def test_config_is_frozen() -> None:
    cfg = PlaywrightCrawlerConfig()
    with pytest.raises(AttributeError):
        cfg.timeout_ms = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Crawler construction
# ---------------------------------------------------------------------------


def test_crawler_rejects_empty_seed_urls() -> None:
    with pytest.raises(ValueError):
        PlaywrightCrawler(seed_urls={})


def test_crawler_rejects_empty_seed_id() -> None:
    with pytest.raises(ValueError):
        PlaywrightCrawler(seed_urls={"": "https://x.example"})


def test_crawler_rejects_empty_url() -> None:
    with pytest.raises(ValueError):
        PlaywrightCrawler(seed_urls={"seed_a": ""})


def test_crawler_starts_disconnected() -> None:
    crawler = PlaywrightCrawler(seed_urls=_seeds())
    assert crawler.status is CrawlerStatus.DISCONNECTED
    assert crawler.is_ready is False


def test_crawler_implements_protocol() -> None:
    crawler = PlaywrightCrawler(seed_urls=_seeds())
    assert isinstance(crawler, Crawler)


def test_crawler_seed_urls_is_immutable() -> None:
    crawler = PlaywrightCrawler(seed_urls=_seeds())
    with pytest.raises(TypeError):
        crawler.seed_urls["seed_a"] = "x"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_connect_uses_injected_runtime_factory() -> None:
    fake = _FakeRuntime()
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    assert crawler.status is CrawlerStatus.CONNECTED
    assert crawler.is_ready is True


def test_disconnect_is_idempotent() -> None:
    crawler = PlaywrightCrawler(seed_urls=_seeds())
    crawler.disconnect()  # no-op while DISCONNECTED
    crawler.disconnect()  # still no-op
    assert crawler.status is CrawlerStatus.DISCONNECTED


def test_disconnect_closes_runtime() -> None:
    fake = _FakeRuntime()
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    crawler.disconnect()
    assert fake.closed is True
    assert crawler.status is CrawlerStatus.DISCONNECTED


def test_default_factory_raises_without_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate ``playwright`` not being installed by shadowing the
    import. The lazy import inside :func:`_build_default_runtime`
    must surface a structured :class:`RuntimeError`.
    """

    import builtins

    real_import = builtins.__import__

    def _blocked_import(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: object = (),
        level: int = 0,
    ) -> object:
        if name.startswith("playwright"):
            raise ImportError(f"blocked: {name}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    crawler = PlaywrightCrawler(seed_urls=_seeds())
    with pytest.raises(RuntimeError, match="playwright not installed"):
        crawler.connect()


# ---------------------------------------------------------------------------
# Fetch — happy path
# ---------------------------------------------------------------------------


def test_fetch_rejects_non_positive_ts_ns() -> None:
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: _FakeRuntime(),
    )
    crawler.connect()
    with pytest.raises(ValueError):
        crawler.fetch(["seed_a"], ts_ns=0)


def test_fetch_requires_connected() -> None:
    crawler = PlaywrightCrawler(seed_urls=_seeds())
    with pytest.raises(RuntimeError, match="adapter_not_ready"):
        crawler.fetch(["seed_a"], ts_ns=1)


def test_fetch_returns_one_doc_per_seed_in_order() -> None:
    fake = _FakeRuntime(
        results={
            "https://a.example/feed": _FetchResult(
                ok=True,
                title="A",
                body="body-a",
                status_code=200,
            ),
            "https://b.example/feed": _FetchResult(
                ok=True,
                title="B",
                body="body-b",
                status_code=200,
            ),
        }
    )
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a", "seed_b"], ts_ns=42)
    assert isinstance(docs, tuple)
    assert len(docs) == 2
    assert docs[0].seed_id == "seed_a"
    assert docs[0].title == "A"
    assert docs[0].body == "body-a"
    assert docs[0].fetched_ok is True
    assert docs[1].seed_id == "seed_b"
    assert docs[1].title == "B"


def test_fetch_preserves_duplicate_seeds() -> None:
    fake = _FakeRuntime()
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a", "seed_b", "seed_a"], ts_ns=42)
    assert len(docs) == 3
    assert [d.seed_id for d in docs] == ["seed_a", "seed_b", "seed_a"]


def test_fetch_unknown_seed_is_fail_soft() -> None:
    fake = _FakeRuntime()
    crawler = PlaywrightCrawler(
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
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a"], ts_ns=42)
    assert len(docs) == 1
    assert docs[0].fetched_ok is False
    assert docs[0].meta["error"] == "navigation_failed"
    assert docs[0].meta["error_class"] == "TimeoutError"


def test_fetch_passes_config_to_runtime() -> None:
    fake = _FakeRuntime()
    cfg = PlaywrightCrawlerConfig(
        timeout_ms=5_000,
        user_agent="custom-ua",
        locale="en-US",
    )
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        config=cfg,
        runtime_factory=lambda c: fake,
    )
    crawler.connect()
    crawler.fetch(["seed_a"], ts_ns=42)
    assert fake.calls == [
        ("https://a.example/feed", 5_000, "custom-ua", "en-US"),
    ]


def test_fetch_carries_status_code_into_meta() -> None:
    fake = _FakeRuntime(
        results={
            "https://a.example/feed": _FetchResult(
                ok=True,
                title="A",
                body="body-a",
                status_code=404,
            ),
        }
    )
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a"], ts_ns=42)
    assert docs[0].meta["status_code"] == "404"


def test_fetch_emits_raw_document_typed() -> None:
    fake = _FakeRuntime()
    crawler = PlaywrightCrawler(
        seed_urls=_seeds(),
        runtime_factory=lambda cfg: fake,
    )
    crawler.connect()
    docs = crawler.fetch(["seed_a"], ts_ns=42)
    assert isinstance(docs[0], RawDocument)


# ---------------------------------------------------------------------------
# Determinism (INV-15)
# ---------------------------------------------------------------------------


def test_three_run_byte_identical_replay() -> None:
    def _run() -> tuple[RawDocument, ...]:
        fake = _FakeRuntime()
        crawler = PlaywrightCrawler(
            seed_urls=_seeds(),
            runtime_factory=lambda cfg: fake,
        )
        crawler.connect()
        return tuple(crawler.fetch(["seed_a", "seed_b"], ts_ns=1_700_000_000_000_000_000))

    r1 = _run()
    r2 = _run()
    r3 = _run()
    assert r1 == r2 == r3


def test_result_to_raw_document_pure() -> None:
    result = _FetchResult(ok=True, title="t", body="b", status_code=200)
    a = _result_to_raw_document(result, ts_ns=42, seed_id="seed_a", url="u")
    b = _result_to_raw_document(result, ts_ns=42, seed_id="seed_a", url="u")
    assert a == b


def test_result_to_raw_document_handles_none_status() -> None:
    result = _FetchResult(ok=False, error="timeout")
    doc = _result_to_raw_document(result, ts_ns=42, seed_id="seed_a", url="u")
    assert "status_code" not in doc.meta
    assert doc.meta["error"] == "timeout"
    assert doc.fetched_ok is False


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_MODULE_PATH = pathlib.Path(__file__).resolve().parents[1] / (
    "sensory/web_autolearn/crawler_playwright.py"
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


def test_no_toplevel_playwright_import() -> None:
    """``playwright`` must be lazy-imported only inside ``connect``."""
    imports = _toplevel_imports(_module_ast())
    assert not any(n == "playwright" or n.startswith("playwright.") for n in imports), (
        f"toplevel playwright import found: {imports}"
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
    """INV-15 / B-CLOCK — no time / random / datetime / asyncio."""
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
    """B27 / B28 / INV-71 — sensory tier never constructs typed
    bus events. Only :class:`RawDocument` (the value object from the
    web-autolearn contracts) is allowed.
    """
    tree = _module_ast()
    forbidden = {"SignalEvent", "HazardEvent", "PatchProposal"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name not in forbidden, f"forbidden typed-event construction: {name}"
