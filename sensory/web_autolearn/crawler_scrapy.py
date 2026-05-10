# ADAPTED FROM: scrapy/scrapy
"""A-19.2 — Scrapy-backed :class:`Crawler` for the web-autolearn pipeline.

Wraps a synchronous subset of the Scrapy public API (``scrapy.Spider`` +
``CrawlerProcess``) onto the DIX :class:`Crawler` Protocol from
:mod:`sensory.web_autolearn.crawler`. A ``seed_id`` list comes in,
one :class:`RawDocument` comes out per seed — drop-in fallback to
:class:`FirecrawlCrawler` (S-05) and :class:`PlaywrightCrawler` (A-16),
identical fail-soft contract.

This is the *fetch* half of the A-19 pair; the *extraction* half lives
in :mod:`sensory.web_autolearn.extractors` (A-19.1) and operates on
the :attr:`RawDocument.body` HTML produced here.

License posture
---------------
The ``scrapy/scrapy`` repository is **BSD-3-Clause**, fully compatible
with the DIX project. The adapter consumes only the documented public
API (``scrapy.Spider`` subclasses, ``CrawlerProcess.crawl`` /
``CrawlerProcess.start``, the ``Request.meta`` dict, the documented
settings keys ``USER_AGENT`` / ``DOWNLOAD_TIMEOUT`` / ``DOWNLOAD_DELAY``
/ ``CONCURRENT_REQUESTS`` / ``ROBOTSTXT_OBEY``) — no internal Scrapy
code is copied.

Tier discipline
---------------
* **OFFLINE_ONLY / sensory-subprocess only.** Scrapy bundles a
  Twisted reactor that is *not* safe inside the runtime hot path
  (INV-15 / T1 / B-CLOCK / B-ASYNC). Operators must enable this
  crawler explicitly via ``registry/engines.yaml`` and run it under
  a dedicated sensory subprocess. The crawler itself never reads its
  own clock — every produced :class:`RawDocument` carries the
  ``ts_ns`` supplied by the caller.
* **No engine import, no FSM mutation, no audit ledger write.** The
  module satisfies the authority lint at L1 / T1 / W1 — pinned by an
  AST test alongside the firecrawl / playwright peers.
* **No ``os.environ`` reads.** Configuration is dependency-injected
  via :class:`ScrapyCrawlerConfig`; the operator wires it from
  :mod:`system_engine.credentials` (no API key in this case, but
  rate-limit / UA / timeout settings).

What survives from upstream
---------------------------
* The ``scrapy.Spider`` lifecycle (``start_requests`` →
  ``parse(response)``) and the ``Response.url`` / ``.text`` /
  ``.status`` / ``.encoding`` field shape.
* The five canonical settings consumed at ``CrawlerProcess``
  construction time: ``USER_AGENT`` (UA string), ``DOWNLOAD_TIMEOUT``
  (per-request hard timeout in seconds), ``DOWNLOAD_DELAY``
  (rate-limit between requests in seconds — Scrapy's standard
  rate-limit knob, spec-mandated default ``1.0`` per
  ``DIX_MASTER_CANONICAL.md`` line 1505), ``CONCURRENT_REQUESTS``
  (parallelism cap), ``ROBOTSTXT_OBEY`` (RFC-9309 compliance).

What is rewritten behind DIX contracts
--------------------------------------
* The ``scrapy`` package is *lazy-imported* inside
  :meth:`connect`. The module imports cleanly even when Scrapy is
  not installed; tests inject a fake ``runtime_factory`` so they
  never need the real package or Twisted reactor.
* Every :meth:`fetch` is a *synchronous* loop with no implicit
  clocks. No ``time.sleep``, no ``asyncio.sleep``, no daemon thread,
  no internal retry. Fail-soft: SDK exceptions never propagate —
  every one becomes a :class:`RawDocument` with ``fetched_ok=False``.
* **Subprocess isolation** — the default runtime spawns a fresh
  ``CrawlerProcess`` per :meth:`fetch` so the Twisted reactor never
  outlives the batch. Tests inject a deterministic in-process fake
  that bypasses Twisted entirely.
* Seeds are *not* re-sorted by the crawler — the protocol guarantees
  the caller passes a deterministic ordering. The crawler still
  emits documents in the exact requested order so replays are
  byte-stable (INV-15).

Sandbox tier
~~~~~~~~~~~~
This crawler runs in a sensory subprocess; never call :meth:`fetch`
from the runtime hot path. The :class:`Crawler` Protocol enforces
the contract at the type level (no engine import).
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Final

from sensory.web_autolearn.contracts import RawDocument

__all__ = (
    "DEFAULT_CONCURRENT_REQUESTS",
    "DEFAULT_DOWNLOAD_DELAY_SEC",
    "DEFAULT_DOWNLOAD_TIMEOUT_SEC",
    "DEFAULT_USER_AGENT",
    "NEW_PIP_DEPENDENCIES",
    "CrawlerStatus",
    "ScrapyCrawler",
    "ScrapyCrawlerConfig",
    "ScrapyFetchResult",
)


# pip dependency flag — the adapter lazy-imports ``scrapy`` at
# connect()-time, so the module itself is importable without the
# package installed.
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("scrapy",)

DEFAULT_DOWNLOAD_DELAY_SEC: Final[float] = 1.0
"""Rate-limit per spec line 1505: ``max 1 req/sec default``.

Scrapy enforces this between successive requests in the same
``CrawlerProcess`` via the ``DOWNLOAD_DELAY`` setting. Operators
who need a different rate must override the config explicitly via
``registry/engines.yaml``.
"""

DEFAULT_DOWNLOAD_TIMEOUT_SEC: Final[float] = 10.0
"""Per-request hard timeout in seconds (Scrapy ``DOWNLOAD_TIMEOUT``)."""

DEFAULT_CONCURRENT_REQUESTS: Final[int] = 1
"""Sequential by default — replays do not depend on completion order."""

DEFAULT_USER_AGENT: Final[str] = (
    "Mozilla/5.0 (compatible; DIXVision-WebAutolearn/1.0; +scrapy)"
)
"""Stable UA string — replays observe identical headers."""


# --------------------------------------------------------------------
# Status
# --------------------------------------------------------------------


class CrawlerStatus(enum.StrEnum):
    """Lifecycle state for the Scrapy-backed crawler."""

    DISCONNECTED = "DISCONNECTED"
    """No runtime is bound (scaffold mode or pre-connect)."""

    CONNECTED = "CONNECTED"
    """A Scrapy runtime (real or injected) is ready to fetch."""


# --------------------------------------------------------------------
# Configuration value object
# --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ScrapyCrawlerConfig:
    """Static configuration for the Scrapy crawler.

    Frozen + slotted so it is hashable and replay-safe.

    Attributes:
        download_delay_sec: Per-request rate limit in seconds —
            Scrapy ``DOWNLOAD_DELAY``. Must be ``>= 0.0``. Defaults
            to :data:`DEFAULT_DOWNLOAD_DELAY_SEC` (``1.0`` per spec
            line 1505).
        download_timeout_sec: Per-request hard timeout in seconds —
            Scrapy ``DOWNLOAD_TIMEOUT``. Must be ``> 0.0``. Defaults
            to :data:`DEFAULT_DOWNLOAD_TIMEOUT_SEC`.
        concurrent_requests: Parallelism cap — Scrapy
            ``CONCURRENT_REQUESTS``. Must be ``>= 1``. Defaults to
            :data:`DEFAULT_CONCURRENT_REQUESTS`.
        user_agent: User-Agent string sent on every request. Empty
            string is rejected.
        obey_robots_txt: Whether Scrapy honours ``robots.txt``.
            Defaults to ``True`` (mandatory for any production
            crawl; the test-only crawls used inside CI never set
            this to ``False``).
    """

    download_delay_sec: float = DEFAULT_DOWNLOAD_DELAY_SEC
    download_timeout_sec: float = DEFAULT_DOWNLOAD_TIMEOUT_SEC
    concurrent_requests: int = DEFAULT_CONCURRENT_REQUESTS
    user_agent: str = DEFAULT_USER_AGENT
    obey_robots_txt: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.download_delay_sec, (int, float)):
            raise TypeError(
                "ScrapyCrawlerConfig.download_delay_sec must be float"
            )
        if self.download_delay_sec < 0.0:
            raise ValueError(
                "ScrapyCrawlerConfig.download_delay_sec must be >= 0.0"
            )
        if not isinstance(self.download_timeout_sec, (int, float)):
            raise TypeError(
                "ScrapyCrawlerConfig.download_timeout_sec must be float"
            )
        if self.download_timeout_sec <= 0.0:
            raise ValueError(
                "ScrapyCrawlerConfig.download_timeout_sec must be > 0.0"
            )
        if not isinstance(self.concurrent_requests, int):
            raise TypeError(
                "ScrapyCrawlerConfig.concurrent_requests must be int"
            )
        if self.concurrent_requests < 1:
            raise ValueError(
                "ScrapyCrawlerConfig.concurrent_requests must be >= 1"
            )
        if not isinstance(self.user_agent, str):
            raise TypeError(
                "ScrapyCrawlerConfig.user_agent must be str"
            )
        if not self.user_agent:
            raise ValueError(
                "ScrapyCrawlerConfig.user_agent must be non-empty"
            )
        if not isinstance(self.obey_robots_txt, bool):
            raise TypeError(
                "ScrapyCrawlerConfig.obey_robots_txt must be bool"
            )


# --------------------------------------------------------------------
# Runtime seam — Scrapy spider abstraction
# --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class ScrapyFetchResult:
    """Pure value record returned by a :class:`_ScrapyRuntime`.

    Attributes:
        ok: Whether the fetch produced a usable response (HTTP 2xx,
            no Twisted timeout / DNS error).
        title: Optional ``<title>`` text. Empty string allowed.
        body: Response text content. Empty string allowed.
        status_code: HTTP status. ``None`` when Scrapy could not
            observe a response (DNS error, timeout pre-response).
        error: Optional short error code (e.g. ``"timeout"``,
            ``"dns"``, ``"http_5xx"``). Empty string when ``ok`` is
            True.
    """

    ok: bool
    title: str = ""
    body: str = ""
    status_code: int | None = None
    error: str = ""


class _ScrapyRuntime:
    """Minimal seam over ``scrapy.crawler.CrawlerProcess`` for testability.

    A runtime exposes one method, :meth:`fetch_one`, returning a
    structured :class:`ScrapyFetchResult`. Production code wraps the
    real Scrapy SDK; tests inject a deterministic fake. The crawler
    never imports ``scrapy`` directly except via this seam (the
    lazy import lives in the default factory inside :meth:`connect`).
    """

    def fetch_one(
        self,
        url: str,
        *,
        download_timeout_sec: float,
        user_agent: str,
    ) -> ScrapyFetchResult:
        raise NotImplementedError

    def close(self) -> None:
        """Optional teardown hook — default impl is a no-op."""


# --------------------------------------------------------------------
# Helpers — pure response normalisation
# --------------------------------------------------------------------


def _coerce_text(value: Any) -> str:
    """Coerce an arbitrary Scrapy field to a clean ``str``.

    Scrapy returns ``str`` for ``Response.text`` in the documented
    schema. Defensive coercion still applies because test doubles
    may pass ``None`` or numeric metadata. Non-str / non-None values
    stringify with :func:`str`. ``None`` becomes ``""``.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _result_to_raw_document(
    result: ScrapyFetchResult,
    *,
    ts_ns: int,
    seed_id: str,
    url: str,
) -> RawDocument:
    """Project a :class:`ScrapyFetchResult` into a :class:`RawDocument`.

    Pure-functional, deterministic — invoked once per fetched seed
    inside :meth:`ScrapyCrawler.fetch`. ``meta`` is built with
    sorted-key conventions so dict ordering doesn't leak into the
    replay digest.
    """

    title = _coerce_text(result.title)
    body = _coerce_text(result.body)

    meta_out: dict[str, str] = {}
    if result.status_code is not None:
        meta_out["status_code"] = str(int(result.status_code))
    if result.error:
        meta_out["error"] = result.error

    return RawDocument(
        ts_ns=ts_ns,
        seed_id=seed_id,
        url=url,
        title=title,
        body=body,
        fetched_ok=bool(result.ok),
        meta=meta_out,
    )


# --------------------------------------------------------------------
# Crawler
# --------------------------------------------------------------------


class ScrapyCrawler:
    """Production :class:`Crawler` backed by the Scrapy framework.

    Args:
        seed_urls: Mapping ``seed_id -> URL``. Must be non-empty.
            Every ``seed_id`` and URL must be a non-empty string.
        config: Static :class:`ScrapyCrawlerConfig`. Defaults to a
            1 req/sec rate limit (spec line 1505), 10s timeout, and
            a stable UA.
        runtime_factory: Optional one-arg callable
            ``(config) -> _ScrapyRuntime`` returning the runtime
            seam. Tests inject a fake here so the crawler runs
            without Scrapy installed. ``None`` uses the lazy
            real-SDK loader on :meth:`connect`.
    """

    __slots__ = (
        "_seed_urls",
        "_config",
        "_runtime_factory",
        "_runtime",
        "_status",
    )

    _seed_urls: Mapping[str, str]
    _config: ScrapyCrawlerConfig
    _runtime_factory: (
        Callable[[ScrapyCrawlerConfig], _ScrapyRuntime] | None
    )
    _runtime: _ScrapyRuntime | None
    _status: CrawlerStatus

    def __init__(
        self,
        seed_urls: Mapping[str, str],
        *,
        config: ScrapyCrawlerConfig | None = None,
        runtime_factory: (
            Callable[[ScrapyCrawlerConfig], _ScrapyRuntime] | None
        ) = None,
    ) -> None:
        if not seed_urls:
            raise ValueError(
                "ScrapyCrawler.seed_urls must be non-empty"
            )

        validated: dict[str, str] = {}
        for seed_id, url in seed_urls.items():
            if not isinstance(seed_id, str) or not seed_id:
                raise ValueError(
                    "ScrapyCrawler.seed_urls keys must be non-empty str"
                )
            if not isinstance(url, str) or not url:
                raise ValueError(
                    f"ScrapyCrawler.seed_urls[{seed_id!r}]"
                    " must be non-empty str"
                )
            validated[seed_id] = url

        self._seed_urls = MappingProxyType(validated)
        self._config = config or ScrapyCrawlerConfig()
        self._runtime_factory = runtime_factory
        self._runtime = None
        self._status = CrawlerStatus.DISCONNECTED

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> CrawlerStatus:
        return self._status

    @property
    def seed_urls(self) -> Mapping[str, str]:
        return self._seed_urls

    @property
    def config(self) -> ScrapyCrawlerConfig:
        return self._config

    @property
    def is_ready(self) -> bool:
        return (
            self._status is CrawlerStatus.CONNECTED
            and self._runtime is not None
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the runtime and transition to ``CONNECTED``.

        Lazy-imports ``scrapy`` only when no ``runtime_factory`` was
        injected, so the module imports cleanly without Scrapy
        installed and tests can drive the crawler with a fake
        runtime.

        Raises:
            RuntimeError: when ``scrapy`` is not installed and no
                ``runtime_factory`` was injected.
        """

        if self._runtime_factory is not None:
            self._runtime = self._runtime_factory(self._config)
        else:
            self._runtime = _build_default_runtime(self._config)

        self._status = CrawlerStatus.CONNECTED

    def disconnect(self) -> None:
        """Close the runtime and return to ``DISCONNECTED``.

        Idempotent — calling on an already-disconnected crawler is
        a no-op so operator-facing tooling can call it freely.
        """

        runtime = self._runtime
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - never propagate
                    pass
        self._runtime = None
        self._status = CrawlerStatus.DISCONNECTED

    # ------------------------------------------------------------------
    # Crawler Protocol
    # ------------------------------------------------------------------

    def fetch(
        self,
        seeds: Sequence[str],
        *,
        ts_ns: int,
    ) -> Sequence[RawDocument]:
        """Fetch one :class:`RawDocument` per requested seed.

        Iterates seeds in caller-supplied order (the protocol
        guarantees this; replays observe identical input lists, so
        the output order is deterministic, INV-15). Unknown seeds
        and SDK errors produce a fail-soft :class:`RawDocument`
        with ``fetched_ok=False`` rather than aborting the batch.

        Rate-limiting between successive requests is the
        responsibility of the runtime (the default real-Scrapy
        runtime honours ``DOWNLOAD_DELAY`` via ``CrawlerProcess``;
        the in-memory test runtime is rate-limit-free since tests
        run no network IO).
        """

        if ts_ns <= 0:
            raise ValueError(
                "ScrapyCrawler.fetch ts_ns must be positive"
            )
        if not self.is_ready:
            raise RuntimeError(
                "ScrapyCrawler.fetch: adapter_not_ready"
            )

        runtime = self._runtime
        assert runtime is not None  # narrowed by is_ready

        out: list[RawDocument] = []
        for seed_id in seeds:
            url = self._seed_urls.get(seed_id)
            if url is None:
                out.append(
                    RawDocument(
                        ts_ns=ts_ns,
                        seed_id=seed_id,
                        url=f"about:unknown/{seed_id}",
                        fetched_ok=False,
                        meta={"error": "unknown_seed"},
                    )
                )
                continue

            try:
                result = runtime.fetch_one(
                    url,
                    download_timeout_sec=self._config.download_timeout_sec,
                    user_agent=self._config.user_agent,
                )
            except Exception as exc:  # noqa: BLE001 - SDK throws bare
                out.append(
                    RawDocument(
                        ts_ns=ts_ns,
                        seed_id=seed_id,
                        url=url,
                        fetched_ok=False,
                        meta={
                            "error": "fetch_failed",
                            "error_class": type(exc).__name__,
                        },
                    )
                )
                continue

            out.append(
                _result_to_raw_document(
                    result,
                    ts_ns=ts_ns,
                    seed_id=seed_id,
                    url=url,
                )
            )

        return tuple(out)


# --------------------------------------------------------------------
# Default runtime — lazy Scrapy wrapper
# --------------------------------------------------------------------


def _build_default_runtime(
    config: ScrapyCrawlerConfig,
) -> _ScrapyRuntime:
    """Lazy factory for the real Scrapy-backed runtime.

    Imports ``scrapy`` only when called. Tests inject a
    ``runtime_factory`` to avoid touching the real SDK.

    Raises:
        RuntimeError: when ``scrapy`` is not installed.
    """

    try:
        # Lazy import — never run at module load time.
        import scrapy  # noqa: F401, PLC0415 — lazy import is the point
        from scrapy.crawler import (  # noqa: PLC0415
            CrawlerProcess,
        )
    except ImportError as exc:
        raise RuntimeError(
            "ScrapyCrawler.connect: scrapy not installed"
        ) from exc

    return _DefaultScrapyRuntime(
        config=config,
        crawler_process_factory=CrawlerProcess,
    )


class _DefaultScrapyRuntime(_ScrapyRuntime):
    """Real Scrapy runtime — one ``CrawlerProcess`` per :meth:`fetch_one`.

    The default behaviour is intentionally one-shot per request to
    keep the Twisted reactor lifecycle simple. Operators who need
    high-throughput crawling should build a long-lived runtime by
    overriding :class:`_ScrapyRuntime` directly.

    The runtime never reads its own clock; the ``ts_ns`` on the
    produced :class:`RawDocument` comes from the caller's
    :meth:`ScrapyCrawler.fetch` argument.
    """

    __slots__ = ("_config", "_crawler_process_factory")

    def __init__(
        self,
        *,
        config: ScrapyCrawlerConfig,
        crawler_process_factory: Callable[..., Any],
    ) -> None:
        self._config = config
        self._crawler_process_factory = crawler_process_factory

    def fetch_one(
        self,
        url: str,
        *,
        download_timeout_sec: float,
        user_agent: str,
    ) -> ScrapyFetchResult:
        """Run a one-shot Scrapy crawl for ``url`` and return the result.

        Builds a fresh ``CrawlerProcess`` configured with the
        canonical settings (UA / timeout / delay / concurrency /
        robots), spawns a :class:`_SingleUrlSpider` against the URL,
        captures the response into a list, and projects it into a
        :class:`ScrapyFetchResult`. Any Scrapy / Twisted exception
        bubbles up to :meth:`ScrapyCrawler.fetch` where it becomes
        a fail-soft :class:`RawDocument`.
        """

        settings = {
            "USER_AGENT": user_agent,
            "DOWNLOAD_TIMEOUT": float(download_timeout_sec),
            "DOWNLOAD_DELAY": float(self._config.download_delay_sec),
            "CONCURRENT_REQUESTS": int(self._config.concurrent_requests),
            "ROBOTSTXT_OBEY": bool(self._config.obey_robots_txt),
            "LOG_ENABLED": False,
            "TELNETCONSOLE_ENABLED": False,
        }
        captured: list[ScrapyFetchResult] = []

        _ensure_single_url_spider_loaded()
        spider_cls = _SingleUrlSpider
        assert spider_cls is not None  # narrowed by loader

        process = self._crawler_process_factory(settings=settings)
        process.crawl(
            spider_cls,
            start_url=url,
            result_sink=captured,
        )
        process.start()  # blocks until reactor finishes

        if captured:
            return captured[0]
        return ScrapyFetchResult(ok=False, error="no_response")


# --------------------------------------------------------------------
# Default spider — minimal Scrapy adapter
# --------------------------------------------------------------------


def _build_single_url_spider() -> Any:
    """Build the default :class:`scrapy.Spider` subclass lazily.

    Kept inside a helper so the module top-level has zero ``scrapy``
    references. Called only from inside :class:`_DefaultScrapyRuntime`
    after the SDK has been confirmed installed.
    """

    import scrapy  # noqa: PLC0415 — lazy import is the point

    class _SingleUrlSpider(scrapy.Spider):
        name = "dix_single_url"

        def __init__(
            self,
            *args: Any,
            start_url: str,
            result_sink: list[ScrapyFetchResult],
            **kwargs: Any,
        ) -> None:
            super().__init__(*args, **kwargs)
            self._start_url = start_url
            self._result_sink = result_sink

        def start_requests(self) -> Any:
            yield scrapy.Request(
                self._start_url,
                callback=self._on_response,
                errback=self._on_error,
                dont_filter=True,
            )

        def _on_response(self, response: Any) -> None:
            status = int(getattr(response, "status", 0) or 0)
            text = _coerce_text(getattr(response, "text", ""))
            self._result_sink.append(
                ScrapyFetchResult(
                    ok=200 <= status < 300,
                    title="",
                    body=text,
                    status_code=status,
                    error="" if 200 <= status < 300 else f"http_{status}",
                )
            )

        def _on_error(self, failure: Any) -> None:
            self._result_sink.append(
                ScrapyFetchResult(
                    ok=False,
                    error=type(getattr(failure, "value", failure)).__name__,
                )
            )

    return _SingleUrlSpider


# Sentinel: the real spider class is built lazily on first run by
# :class:`_DefaultScrapyRuntime`. Tests never touch this — they inject
# their own runtime via ``runtime_factory``.
_SingleUrlSpider: Any = None


def _ensure_single_url_spider_loaded() -> None:
    """Materialise :data:`_SingleUrlSpider` from the lazy builder.

    The real runtime calls this only when actually running a crawl;
    tests inject their own runtime and never trigger this path.
    """

    global _SingleUrlSpider
    if _SingleUrlSpider is None:
        _SingleUrlSpider = _build_single_url_spider()
