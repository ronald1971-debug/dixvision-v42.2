# ADAPTED FROM: microsoft/playwright-python
"""A-16 — Playwright-backed :class:`Crawler` for the web-autolearn pipeline.

Maps a synchronous subset of the Playwright Python SDK (``playwright``)
onto the DIX :class:`Crawler` Protocol from
:mod:`sensory.web_autolearn.crawler`. A ``seed_id`` list comes in,
one :class:`RawDocument` comes out per seed — drop-in fallback to
:class:`FirecrawlCrawler` from S-05, identical fail-soft contract.

License posture
---------------
The ``microsoft/playwright-python`` repository is **Apache-2.0**, fully
compatible with the DIX project. The adapter consumes only the
documented public API (:func:`playwright.sync_api.sync_playwright`,
``Browser.new_context``, ``BrowserContext.new_page``, ``Page.goto``,
``Page.wait_for_load_state``, ``Page.content``, ``Page.title``) — no
internal Playwright code is copied.

Tier discipline
---------------
* **OFFLINE_ONLY / sensory-subprocess only.** Playwright spawns a
  Chromium subprocess and an asyncio/Greenlet bridge that is *not*
  safe inside the runtime hot path (INV-15 / T1 / B-CLOCK). Operators
  must enable this crawler explicitly via
  ``registry/engines.yaml`` and run it under a sensory subprocess.
  The crawler itself never reads its own clock — every produced
  :class:`RawDocument` carries the ``ts_ns`` supplied by the caller.
* **No engine import, no FSM mutation, no audit ledger write.** The
  module satisfies the authority lint at L1 / T1 / W1 — pinned by an
  AST test alongside the firecrawl peer.
* **No ``os.environ`` reads.** Configuration is dependency-injected via
  :class:`PlaywrightCrawlerConfig`; the operator wires it from
  :mod:`system_engine.credentials` (no API key in this case, but
  optional ``user_agent`` / ``locale`` / ``timeout_ms``).

What survives from upstream
---------------------------
* ``sync_playwright()`` context-manager lifecycle, ``chromium.launch()``,
  ``browser.new_context()``, ``context.new_page()``, ``page.goto(url,
  wait_until="domcontentloaded")``, ``page.wait_for_load_state``,
  ``page.title()``, ``page.content()``. Each fetch follows the
  canonical "new context per crawl → new page → hard timeout →
  close context" pattern from the Playwright docs.
* Per-fetch timeout enforcement via Playwright's own
  ``page.set_default_timeout(timeout_ms)``.

What is rewritten behind DIX contracts
--------------------------------------
* The ``playwright`` package is *lazy-imported* inside
  :meth:`connect`. The module imports cleanly even when Playwright
  is not installed; tests inject a fake ``runtime_factory`` so they
  never need the real package or Chromium binary.
* Every :meth:`fetch` is a *synchronous* loop with no implicit clocks.
  No ``time.sleep``, no ``asyncio.sleep``, no daemon thread, no
  internal retry. Fail-soft: SDK exceptions never propagate — every
  one becomes a :class:`RawDocument` with ``fetched_ok=False``.
* **BrowserContext isolation** — a new ``BrowserContext`` is created
  per ``fetch`` call so cookies/storage never leak between crawls.
  At the end of the batch the context is closed; the browser stays
  open across calls until :meth:`disconnect` is invoked.
* Seeds are *not* re-sorted by the crawler — the protocol guarantees
  the caller passes a deterministic ordering (sorted in the operator
  layer that materialises ``seeds.yaml``). The crawler still emits
  documents in the exact requested order so replays are byte-stable.

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
from typing import Any

from sensory.web_autolearn.contracts import RawDocument

# pip dependency flag — the adapter lazy-imports `playwright` at
# connect()-time, so the module itself is importable without the
# package installed. The package distributes as ``playwright`` on
# PyPI and additionally requires ``playwright install chromium``.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("playwright",)

DEFAULT_TIMEOUT_MS: int = 10_000
"""Hard per-page timeout — Playwright enforces this on goto / wait."""

DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36 "
    "DIXVision-WebAutolearn/1.0"
)
"""Stable UA string — replays observe identical headers."""


# --------------------------------------------------------------------
# Status
# --------------------------------------------------------------------


class CrawlerStatus(enum.StrEnum):
    """Lifecycle state for the Playwright-backed crawler."""

    DISCONNECTED = "DISCONNECTED"
    """No browser runtime is open (scaffold mode or pre-connect)."""

    CONNECTED = "CONNECTED"
    """A Chromium browser (or injected runtime) is ready to fetch."""


# --------------------------------------------------------------------
# Configuration value object
# --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PlaywrightCrawlerConfig:
    """Static configuration for the Playwright crawler.

    Frozen + slotted so it is hashable and replay-safe.

    Attributes:
        timeout_ms: Per-page hard timeout in milliseconds. Must be
            positive. Defaults to 10_000.
        user_agent: User-Agent string sent on every fetch. Empty
            string is rejected (a missing UA reveals scraper traffic).
            Defaults to :data:`DEFAULT_USER_AGENT`.
        locale: BCP-47 locale string passed to ``new_context``.
            Empty string allowed — Playwright falls back to browser
            default.
        headless: Whether to launch Chromium in headless mode.
            Defaults to ``True``. Subprocess-only crawlers should
            never run with a head.
    """

    timeout_ms: int = DEFAULT_TIMEOUT_MS
    user_agent: str = DEFAULT_USER_AGENT
    locale: str = ""
    headless: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_ms, int):
            raise TypeError(
                "PlaywrightCrawlerConfig.timeout_ms must be int"
            )
        if self.timeout_ms <= 0:
            raise ValueError(
                "PlaywrightCrawlerConfig.timeout_ms must be positive"
            )
        if not isinstance(self.user_agent, str):
            raise TypeError(
                "PlaywrightCrawlerConfig.user_agent must be str"
            )
        if not self.user_agent:
            raise ValueError(
                "PlaywrightCrawlerConfig.user_agent must be non-empty"
            )
        if not isinstance(self.locale, str):
            raise TypeError(
                "PlaywrightCrawlerConfig.locale must be str"
            )
        if not isinstance(self.headless, bool):
            raise TypeError(
                "PlaywrightCrawlerConfig.headless must be bool"
            )


# --------------------------------------------------------------------
# Runtime seam — Playwright Page abstraction
# --------------------------------------------------------------------


class _PlaywrightRuntime:
    """Minimal seam over ``playwright.sync_api`` for testability.

    A runtime exposes one method, :meth:`fetch_one`, returning a
    structured :class:`_FetchResult`. Production code wraps the real
    Playwright SDK; tests inject a deterministic fake. The crawler
    never imports ``playwright`` directly except via this seam (the
    lazy import lives in the default factory inside :meth:`connect`).
    """

    def fetch_one(
        self,
        url: str,
        *,
        timeout_ms: int,
        user_agent: str,
        locale: str,
    ) -> _FetchResult:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True, slots=True)
class _FetchResult:
    """Pure value record returned by :meth:`_PlaywrightRuntime.fetch_one`.

    Attributes:
        ok: Whether the fetch reached a usable DOM. Mirrors the
            ``fetched_ok`` field on :class:`RawDocument`.
        title: Document ``<title>`` text. Empty string allowed.
        body: Document text content. Empty string allowed.
        status_code: Optional HTTP status. ``None`` when Playwright
            could not observe a response (timeout, network error).
        error: Optional short error code (e.g. ``"timeout"``,
            ``"navigation_failed"``). Empty string when ``ok`` is
            True.
    """

    ok: bool
    title: str = ""
    body: str = ""
    status_code: int | None = None
    error: str = ""


# --------------------------------------------------------------------
# Helpers — pure response normalisation
# --------------------------------------------------------------------


def _coerce_text(value: Any) -> str:
    """Coerce an arbitrary Playwright field to a clean ``str``.

    Playwright returns ``str`` for ``Page.title()`` and ``Page.content()``
    in the documented schema. Defensive coercion still applies because
    test doubles may pass ``None`` or numeric metadata. Non-str /
    non-None values stringify with :func:`str`. ``None`` becomes ``""``.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _result_to_raw_document(
    result: _FetchResult,
    *,
    ts_ns: int,
    seed_id: str,
    url: str,
) -> RawDocument:
    """Project a :class:`_FetchResult` into a :class:`RawDocument`.

    Pure-functional, deterministic — invoked once per fetched seed
    inside :meth:`PlaywrightCrawler.fetch`. ``meta`` is built with
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


class PlaywrightCrawler:
    """Production :class:`Crawler` backed by the Playwright SDK.

    Args:
        seed_urls: Mapping ``seed_id -> URL``. Must be non-empty (an
            empty crawler can never produce a successful fetch).
            Every ``seed_id`` and URL must be a non-empty string.
        config: Static :class:`PlaywrightCrawlerConfig`. Defaults to
            a 10s timeout + stable UA + headless Chromium.
        runtime_factory: Optional zero-arg callable returning a
            :class:`_PlaywrightRuntime`. Tests inject a fake here so
            the crawler runs without Playwright installed. ``None``
            uses the lazy real-SDK loader on :meth:`connect`.
    """

    __slots__ = (
        "_seed_urls",
        "_config",
        "_runtime_factory",
        "_runtime",
        "_status",
    )

    _seed_urls: Mapping[str, str]
    _config: PlaywrightCrawlerConfig
    _runtime_factory: (
        Callable[[PlaywrightCrawlerConfig], _PlaywrightRuntime] | None
    )
    _runtime: _PlaywrightRuntime | None
    _status: CrawlerStatus

    def __init__(
        self,
        seed_urls: Mapping[str, str],
        *,
        config: PlaywrightCrawlerConfig | None = None,
        runtime_factory: (
            Callable[[PlaywrightCrawlerConfig], _PlaywrightRuntime]
            | None
        ) = None,
    ) -> None:
        if not seed_urls:
            raise ValueError(
                "PlaywrightCrawler.seed_urls must be non-empty"
            )

        validated: dict[str, str] = {}
        for seed_id, url in seed_urls.items():
            if not isinstance(seed_id, str) or not seed_id:
                raise ValueError(
                    "PlaywrightCrawler.seed_urls keys must be"
                    " non-empty str"
                )
            if not isinstance(url, str) or not url:
                raise ValueError(
                    f"PlaywrightCrawler.seed_urls[{seed_id!r}]"
                    " must be non-empty str"
                )
            validated[seed_id] = url

        self._seed_urls = MappingProxyType(validated)
        self._config = config or PlaywrightCrawlerConfig()
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
    def config(self) -> PlaywrightCrawlerConfig:
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

        Lazy-imports ``playwright.sync_api`` only when no
        ``runtime_factory`` was injected, so the module imports cleanly
        without Playwright installed and tests can drive the crawler
        with a fake runtime.

        Raises:
            RuntimeError: when ``playwright`` is not installed and no
                ``runtime_factory`` was injected.
        """

        if self._runtime_factory is not None:
            self._runtime = self._runtime_factory(self._config)
        else:
            self._runtime = _build_default_runtime(self._config)

        self._status = CrawlerStatus.CONNECTED

    def disconnect(self) -> None:
        """Close the runtime and return to ``DISCONNECTED``.

        Idempotent — calling on an already-disconnected crawler is a
        no-op so operator-facing tooling can call it freely.
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
        and SDK errors produce a fail-soft :class:`RawDocument` with
        ``fetched_ok=False`` rather than aborting the batch.
        """

        if ts_ns <= 0:
            raise ValueError(
                "PlaywrightCrawler.fetch ts_ns must be positive"
            )
        if not self.is_ready:
            raise RuntimeError(
                "PlaywrightCrawler.fetch: adapter_not_ready"
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
                    timeout_ms=self._config.timeout_ms,
                    user_agent=self._config.user_agent,
                    locale=self._config.locale,
                )
            except Exception as exc:  # noqa: BLE001 - SDK throws bare
                out.append(
                    RawDocument(
                        ts_ns=ts_ns,
                        seed_id=seed_id,
                        url=url,
                        fetched_ok=False,
                        meta={
                            "error": "navigation_failed",
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
# Default runtime — lazy Playwright wrapper
# --------------------------------------------------------------------


def _build_default_runtime(
    config: PlaywrightCrawlerConfig,
) -> _PlaywrightRuntime:
    """Lazy factory for the real Playwright-backed runtime.

    Imports ``playwright.sync_api`` only when called. Tests inject a
    ``runtime_factory`` to avoid touching the real SDK.

    Raises:
        RuntimeError: when ``playwright`` is not installed.
    """

    try:
        # Lazy import — never run at module load time.
        from playwright.sync_api import (  # type: ignore[import-not-found]
            sync_playwright,
        )
    except ImportError as exc:
        raise RuntimeError(
            "PlaywrightCrawler.connect: playwright not installed"
        ) from exc

    return _DefaultPlaywrightRuntime(
        config=config,
        sync_playwright=sync_playwright,
    )


class _DefaultPlaywrightRuntime(_PlaywrightRuntime):
    """Real Playwright runtime — one Chromium across many crawls.

    Each :meth:`fetch_one` call creates a *new* :class:`BrowserContext`
    so cookie / storage state never leaks between fetches (mandated by
    the A-16 prompt). Hard timeout is set via
    :meth:`Page.set_default_timeout`.
    """

    __slots__ = ("_config", "_sync_playwright", "_pw", "_browser")

    def __init__(
        self,
        *,
        config: PlaywrightCrawlerConfig,
        sync_playwright: Any,
    ) -> None:
        self._config = config
        self._sync_playwright = sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=config.headless)

    def fetch_one(
        self,
        url: str,
        *,
        timeout_ms: int,
        user_agent: str,
        locale: str,
    ) -> _FetchResult:
        context_kwargs: dict[str, Any] = {"user_agent": user_agent}
        if locale:
            context_kwargs["locale"] = locale

        context = self._browser.new_context(**context_kwargs)
        try:
            page = context.new_page()
            page.set_default_timeout(timeout_ms)
            response = page.goto(url, wait_until="domcontentloaded")
            page.wait_for_load_state("domcontentloaded")
            status_code = (
                int(response.status) if response is not None else None
            )
            title = _coerce_text(page.title())
            body = _coerce_text(page.content())
            return _FetchResult(
                ok=True,
                title=title,
                body=body,
                status_code=status_code,
            )
        except Exception:  # noqa: BLE001 - SDK throws bare
            return _FetchResult(
                ok=False,
                error="navigation_failed",
                status_code=None,
                title="",
                body="",
            )
        finally:
            try:
                context.close()
            except Exception:  # noqa: BLE001 - never propagate
                pass

    def close(self) -> None:
        try:
            self._browser.close()
        except Exception:  # noqa: BLE001 - never propagate
            pass
        try:
            self._pw.stop()
        except Exception:  # noqa: BLE001 - never propagate
            pass


__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "DEFAULT_USER_AGENT",
    "NEW_PIP_DEPENDENCIES",
    "CrawlerStatus",
    "PlaywrightCrawler",
    "PlaywrightCrawlerConfig",
]
