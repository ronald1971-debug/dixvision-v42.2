# ADAPTED FROM: firecrawl-py/firecrawl/firecrawl.py
"""S-05 — Firecrawl-backed `Crawler` for the web-autolearn pipeline.

Maps the Firecrawl Python SDK (``firecrawl-py``) onto the DIX
:class:`Crawler` Protocol from
:mod:`sensory.web_autolearn.crawler`. A ``seed_id`` list comes in,
one :class:`RawDocument` comes out per seed — same fail-soft contract
as :class:`DeterministicCrawler`, but the body comes from a real
HTTP scrape via the Firecrawl service.

License posture
---------------
The Firecrawl repository (``mendableai/firecrawl``) is **AGPL-3.0**.
Per the master canonical safety context (PART 1 rule 1) we do **not**
copy any server-side code. The adapter only consumes the published
``FirecrawlApp.scrape_url()`` interface, which is a thin REST client
against the hosted Firecrawl service. This is the standard
"network-only API consumer" mitigation: linking the SDK does not pull
any AGPL server logic into the DIX process.

What survives from upstream
---------------------------
* The :meth:`FirecrawlApp.scrape_url` request shape — single-URL
  fetch with optional ``formats`` / ``params`` arguments — is honoured
  verbatim. The adapter requests ``["markdown"]`` and reads the
  response's ``markdown`` and ``metadata.title`` fields.
* The response field mapping documented in the Firecrawl SDK
  (``data["markdown"]``, ``data["metadata"]["title"]``,
  ``data["metadata"]["description"]``) is preserved.

What is rewritten behind DIX contracts
--------------------------------------
* ``firecrawl`` is *lazy-imported* inside :meth:`connect`. The module
  imports cleanly even when ``firecrawl-py`` is not installed; tests
  never need the real package thanks to ``client_factory`` injection.
* No ``time.sleep``, no ``asyncio.sleep``, no daemon thread, no
  internal retry loop. Every :meth:`fetch` is a synchronous loop with
  no implicit clocks (INV-15 / T1 / B-CLOCK).
* The ``ts_ns`` stamped onto every produced :class:`RawDocument`
  comes from the caller's ``fetch(..., ts_ns=...)`` argument — the
  adapter never reads its own clock.
* Failed scrapes never raise out of :meth:`fetch`. Anything that
  comes out of the SDK as an exception is caught and converted into
  a :class:`RawDocument` with ``fetched_ok=False`` and a structured
  ``meta`` row — the audit ledger never sees a traceback and downstream
  consumers can replay deterministically.
* Until ``credentials`` is supplied **and** :meth:`connect` has run,
  the adapter stays in :attr:`CrawlerStatus.DISCONNECTED` and every
  :meth:`fetch` raises :class:`RuntimeError` rather than silently
  fabricating documents (INV-56 Triad Lock: a sensory module that
  fakes signal is a hard authority breach).

This module never reads ``os.environ``. The Firecrawl API key is
passed in explicitly via :class:`FirecrawlCredentials`, sourced by
the operator from :mod:`system_engine.credentials` so a malformed
env never silently routes traffic against the wrong account.

Sandbox tier
~~~~~~~~~~~~
This crawler is sensory-tier only — it must run in a sensory subprocess
and never touch the runtime hot path. The :class:`Crawler` Protocol
itself enforces this at the type level (no ``EngineCore`` import, no
ledger writes, no FSM mutation). The lint rules in
``tools/authority_lint.py`` (T1 / W1) catch any accidental hot-path
import.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from sensory.web_autolearn.contracts import RawDocument

# pip dependency flag — the adapter lazy-imports `firecrawl` at
# connect()-time, so the module itself is importable without the
# package installed. The package distributes as ``firecrawl-py`` on
# PyPI but exposes the ``firecrawl`` import name.
NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("firecrawl-py",)


# --------------------------------------------------------------------
# Status
# --------------------------------------------------------------------


class CrawlerStatus(enum.StrEnum):
    """Lifecycle state for the live Firecrawl crawler."""

    DISCONNECTED = "DISCONNECTED"
    """No SDK client has been built yet (scaffold mode or pre-connect)."""

    CONNECTED = "CONNECTED"
    """A `firecrawl.FirecrawlApp` (or injected client) is ready to scrape."""


# --------------------------------------------------------------------
# Credentials value object
# --------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class FirecrawlCredentials:
    """Firecrawl API credentials, sourced from
    :mod:`system_engine.credentials`.

    Eagerly validated: an empty key is rejected at construction so the
    failure surface is the credential boundary, not the first scrape.
    """

    api_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.api_key, str):
            raise TypeError("FirecrawlCredentials.api_key must be str")
        if not self.api_key:
            raise ValueError("FirecrawlCredentials.api_key must be non-empty")


# --------------------------------------------------------------------
# Helpers — pure response normalisation
# --------------------------------------------------------------------


def _coerce_text(value: Any) -> str:
    """Coerce an arbitrary SDK field to a clean ``str``.

    Firecrawl returns ``str`` for every text field in the documented
    schema. Defensive coercion still applies because real responses
    sometimes carry ``None`` or numeric metadata. Non-str / non-None
    values stringify with :func:`str`. ``None`` becomes ``""``.
    """

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _extract_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``payload['metadata']`` as a mapping, or empty.

    Firecrawl wraps page metadata under a ``metadata`` key. Older / odd
    responses may omit it; we never let that crash the parser.
    """

    raw = payload.get("metadata")
    if isinstance(raw, Mapping):
        return raw
    return {}


def _parse_payload(
    payload: Any,
    *,
    ts_ns: int,
    seed_id: str,
    url: str,
) -> RawDocument:
    """Build a :class:`RawDocument` from a Firecrawl scrape response.

    Firecrawl's ``scrape_url`` returns a dict with this shape::

        {
            "markdown": "...",
            "html": "...",
            "metadata": {
                "title": "...",
                "description": "...",
                "sourceURL": "...",
                "statusCode": 200,
                ...
            },
        }

    The newer SDK wraps the payload under a top-level ``"data"`` key
    when ``scrape_url`` is called via ``FirecrawlApp.scrape_url``; we
    accept either shape so future SDK upgrades don't break the parser.
    """

    if isinstance(payload, Mapping) and "data" in payload:
        inner = payload.get("data")
        if isinstance(inner, Mapping):
            payload = inner

    if not isinstance(payload, Mapping):
        return RawDocument(
            ts_ns=ts_ns,
            seed_id=seed_id,
            url=url,
            fetched_ok=False,
            meta={"error": "non_mapping_response"},
        )

    metadata = _extract_metadata(payload)
    title = _coerce_text(metadata.get("title"))
    body = _coerce_text(payload.get("markdown"))

    meta_out: dict[str, str] = {}
    status_code = metadata.get("statusCode")
    if isinstance(status_code, int):
        meta_out["status_code"] = str(status_code)
    description = _coerce_text(metadata.get("description"))
    if description:
        meta_out["description"] = description
    source_url = _coerce_text(metadata.get("sourceURL"))
    if source_url:
        meta_out["source_url"] = source_url

    return RawDocument(
        ts_ns=ts_ns,
        seed_id=seed_id,
        url=url,
        title=title,
        body=body,
        fetched_ok=True,
        meta=meta_out,
    )


# --------------------------------------------------------------------
# Crawler
# --------------------------------------------------------------------


class FirecrawlCrawler:
    """Production :class:`Crawler` backed by the Firecrawl SDK.

    Args:
        seed_urls: Mapping ``seed_id -> URL``. Must be non-empty (an
            empty crawler can never produce a successful fetch). Every
            ``seed_id`` and URL must be a non-empty string.
        credentials: Firecrawl API credentials. ``None`` keeps the
            adapter in scaffold mode and every :meth:`fetch` raises
            :class:`RuntimeError` with ``reason="adapter_not_ready"``.
        request_formats: Tuple of Firecrawl scrape formats to request.
            Defaults to ``("markdown",)``. Empty tuple is rejected.
        client_factory: Optional zero-arg callable that returns the
            SDK client (real or stub). Tests inject a fake here so the
            crawler runs without ``firecrawl-py`` installed. ``None``
            uses the lazy real-SDK loader.
    """

    __slots__ = (
        "_seed_urls",
        "_credentials",
        "_request_formats",
        "_client_factory",
        "_client",
        "_status",
    )

    _seed_urls: Mapping[str, str]
    _credentials: FirecrawlCredentials | None
    _request_formats: tuple[str, ...]
    _client_factory: Callable[[FirecrawlCredentials], Any] | None
    _client: Any | None
    _status: CrawlerStatus

    def __init__(
        self,
        seed_urls: Mapping[str, str],
        *,
        credentials: FirecrawlCredentials | None = None,
        request_formats: Sequence[str] = ("markdown",),
        client_factory: (Callable[[FirecrawlCredentials], Any] | None) = None,
    ) -> None:
        if not seed_urls:
            raise ValueError("FirecrawlCrawler.seed_urls must be non-empty")

        validated: dict[str, str] = {}
        for seed_id, url in seed_urls.items():
            if not isinstance(seed_id, str) or not seed_id:
                raise ValueError("FirecrawlCrawler.seed_urls keys must be non-empty str")
            if not isinstance(url, str) or not url:
                raise ValueError(f"FirecrawlCrawler.seed_urls[{seed_id!r}] must be non-empty str")
            validated[seed_id] = url

        formats_tuple = tuple(request_formats)
        if not formats_tuple:
            raise ValueError("FirecrawlCrawler.request_formats must be non-empty")
        for fmt in formats_tuple:
            if not isinstance(fmt, str) or not fmt:
                raise ValueError("FirecrawlCrawler.request_formats entries must be non-empty str")

        self._seed_urls = MappingProxyType(validated)
        self._credentials = credentials
        self._request_formats = formats_tuple
        self._client_factory = client_factory
        self._client = None
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
    def request_formats(self) -> tuple[str, ...]:
        return self._request_formats

    @property
    def is_ready(self) -> bool:
        return self._status is CrawlerStatus.CONNECTED and self._client is not None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Build the SDK client and transition to ``CONNECTED``.

        Raises:
            RuntimeError: when no credentials are wired (scaffold mode).
            RuntimeError: when ``firecrawl-py`` is not installed and no
                ``client_factory`` was injected.
        """

        if self._credentials is None:
            raise RuntimeError("FirecrawlCrawler.connect: scaffold mode (no credentials wired)")

        if self._client_factory is not None:
            self._client = self._client_factory(self._credentials)
        else:
            try:
                from firecrawl import FirecrawlApp  # type: ignore[import-not-found]
            except ImportError as exc:
                raise RuntimeError("FirecrawlCrawler.connect: firecrawl-py not installed") from exc
            self._client = FirecrawlApp(api_key=self._credentials.api_key)

        self._status = CrawlerStatus.CONNECTED

    def disconnect(self) -> None:
        """Drop the SDK client and return to ``DISCONNECTED``.

        Idempotent — calling on an already-disconnected crawler is a
        no-op so operator-facing tooling can call it freely.
        """

        self._client = None
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

        Iterates seeds in caller-supplied order (the protocol guarantees
        this; replays observe identical input lists, so the output
        order is deterministic, INV-15). Unknown seeds and SDK errors
        produce a fail-soft :class:`RawDocument` with
        ``fetched_ok=False`` rather than aborting the batch.
        """

        if ts_ns <= 0:
            raise ValueError("FirecrawlCrawler.fetch ts_ns must be positive")
        if not self.is_ready:
            raise RuntimeError("FirecrawlCrawler.fetch: adapter_not_ready")

        client = self._client
        assert client is not None  # narrowed by is_ready

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
                payload = client.scrape_url(
                    url,
                    formats=list(self._request_formats),
                )
            except Exception as exc:  # noqa: BLE001 - SDK throws bare Exception
                out.append(
                    RawDocument(
                        ts_ns=ts_ns,
                        seed_id=seed_id,
                        url=url,
                        fetched_ok=False,
                        meta={
                            "error": "scrape_failed",
                            "error_class": type(exc).__name__,
                        },
                    )
                )
                continue

            out.append(
                _parse_payload(
                    payload,
                    ts_ns=ts_ns,
                    seed_id=seed_id,
                    url=url,
                )
            )

        return tuple(out)


__all__ = [
    "NEW_PIP_DEPENDENCIES",
    "CrawlerStatus",
    "FirecrawlCredentials",
    "FirecrawlCrawler",
]
