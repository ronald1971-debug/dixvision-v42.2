"""Read-only CoinDesk RSS news adapter (SRC-NEWS-COINDESK-001).

Polls the public CoinDesk RSS endpoint (no API key, no auth) on a
fixed interval and converts each ``<item>`` element into a canonical
:class:`core.contracts.news.NewsItem`.

Layered split (mirrors :mod:`ui.feeds.binance_public_ws`):

* :func:`make_coindesk_rss_url` — pure URL accessor (the endpoint is
  fixed; the function exists so tests can swap it cleanly).
* :func:`parse_rss_feed` — pure ``bytes | str`` → ``tuple[NewsItem, ...]``
  projection. Skips malformed ``<item>`` elements rather than raising,
  matching the Binance pump's tolerance for non-data frames at the
  network boundary. Caller supplies ``ts_ns`` so the parser stays
  INV-15-pure.
* :class:`CoinDeskRSSPump` — thin async I/O wrapper with poll
  interval + reconnect backoff. Takes a ``fetch`` callable and a
  ``clock_ns`` callable so tests inject fakes (no real network, no
  ``time.time``) and the determinism boundary stays explicit.
* :class:`NewsFeedStatus` — frozen telemetry snapshot, exposed by the
  HTTP status endpoint.

INV-15: the parser is pure (caller-supplied ``ts_ns``); the pump is
non-deterministic by design (network), but every item it emits is
funneled into the harness via the same code path the engine ledger
replays deterministically.

CoinDesk RSS reference:
https://www.coindesk.com/arc/outboundfeeds/rss/
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol
from xml.etree import ElementTree as ET  # noqa: S405 - read-only RSS feed

from core.contracts.news import NewsItem

LOG = logging.getLogger(__name__)

#: Public CoinDesk RSS endpoint (no auth required).
COINDESK_RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"

#: Default poll cadence (seconds) — well above CoinDesk's published
#: cache window so we never thunder on the upstream.
DEFAULT_POLL_INTERVAL_S = 60.0

#: Default reconnect backoff floor + ceiling (seconds), used when the
#: HTTP fetch raises.
DEFAULT_RECONNECT_DELAY_S = 5.0
DEFAULT_RECONNECT_DELAY_MAX_S = 300.0

#: Source tag stamped onto every emitted ``NewsItem`` so the ledger
#: distinguishes CoinDesk items from manually injected ones.
SOURCE_TAG = "COINDESK"


def make_coindesk_rss_url() -> str:
    """Return the canonical CoinDesk RSS endpoint.

    Defined as a function (not a constant) so tests can monkeypatch
    a single import without rewriting the constant value.
    """
    return COINDESK_RSS_URL


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_html(text: str) -> str:
    """Best-effort HTML strip — RSS descriptions sometimes embed CDATA
    HTML. Pure regex so we don't pull a parser dep for one field.
    """
    if not text:
        return ""
    cleaned = _TAG_RE.sub(" ", text)
    cleaned = _WS_RE.sub(" ", cleaned).strip()
    return cleaned


def _parse_pub_date(raw: str | None) -> int | None:
    """Parse an RFC-2822 ``<pubDate>`` into a UTC ns timestamp.

    Returns ``None`` for empty / unparseable input — never raises, so
    a single malformed date never poisons the whole feed.
    """
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    try:
        ts_ns = int(dt.astimezone(UTC).timestamp() * 1_000_000_000)
    except (OverflowError, ValueError):
        return None
    # NewsItem.published_ts_ns must be positive or None (per its
    # contract). Return None for the Unix epoch and any pre-1970
    # timestamps so we keep the item's title / guid / url instead of
    # letting NewsItem.__post_init__ raise and the outer except
    # ValueError discard the whole headline.
    return ts_ns if ts_ns > 0 else None


def _local_name(tag: str) -> str:
    """Strip the XML namespace prefix from a tag, if any."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _find_first_text(item: ET.Element, *names: str) -> str:
    """Return the text of the first child whose local-name is in *names*."""
    for child in item:
        if _local_name(child.tag) in names and child.text:
            return child.text
    return ""


def parse_rss_feed(
    payload: bytes | str,
    *,
    ts_ns: int,
    source: str = SOURCE_TAG,
) -> tuple[NewsItem, ...]:
    """Project an RSS document into a tuple of :class:`NewsItem`.

    Skips (never raises on) malformed ``<item>`` elements; an item
    must have a non-empty title and either a non-empty guid or a
    non-empty link to be emitted, otherwise it is dropped silently
    and the caller's pump increments the per-call ``errors`` counter.

    INV-15 (pure projection): ``ts_ns`` is supplied by the caller,
    never derived from ``payload`` or a system clock, so two replays
    with the same input produce byte-identical output.
    """
    if isinstance(payload, str):
        payload_bytes = payload.encode("utf-8")
    else:
        payload_bytes = payload
    if not payload_bytes:
        return ()
    try:
        # Read-only RSS feed; ElementTree is sufficient and avoids
        # pulling defusedxml into base requirements. The RSS contract
        # has no XXE entry points — if upstream becomes hostile we'd
        # have a much larger problem.
        root = ET.fromstring(payload_bytes)  # noqa: S314
    except ET.ParseError:
        return ()

    items: list[NewsItem] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "item":
            continue
        title = _strip_html(_find_first_text(elem, "title")).strip()
        if not title:
            continue
        link = _find_first_text(elem, "link").strip()
        guid = _find_first_text(elem, "guid").strip()
        if not guid:
            guid = link
        if not guid:
            continue
        summary = _strip_html(
            _find_first_text(elem, "description", "summary")
        ).strip()
        published_ts_ns = _parse_pub_date(_find_first_text(elem, "pubDate"))
        try:
            items.append(
                NewsItem(
                    ts_ns=ts_ns,
                    source=source,
                    guid=guid,
                    title=title,
                    url=link,
                    summary=summary,
                    published_ts_ns=published_ts_ns,
                )
            )
        except ValueError:
            # Defensive — NewsItem.__post_init__ already covers the
            # empty-string cases we filter above, but a future field
            # may add stricter validation.
            continue
    return tuple(items)


HTTPFetch = Callable[[str], Awaitable[bytes]]
"""Async URL → bytes callable. Production uses :mod:`urllib.request`."""


async def _default_fetch(url: str) -> bytes:
    """Default fetcher — imported lazily so the parser is usable
    without an HTTP client installed (tests, lint, etc.). Uses the
    stdlib so we don't add a dependency for the canonical news pump.
    """
    import urllib.request  # local import; stdlib

    def _blocking_get() -> bytes:
        req = urllib.request.Request(  # noqa: S310 - fixed https URL
            url,
            headers={"User-Agent": "dixvision-v42.2/coindesk-rss"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            return resp.read()

    return await asyncio.to_thread(_blocking_get)


@dataclass(frozen=True, slots=True)
class NewsFeedStatus:
    """Snapshot of pump health — exposed by ``GET /api/feeds/coindesk/status``."""

    running: bool
    source: str
    url: str
    last_poll_ts_ns: int | None
    last_item_ts_ns: int | None
    items_received: int
    polls: int
    errors: int


class _StopHandle(Protocol):
    """Minimal subset of :class:`asyncio.Event` we use, for typing."""

    def set(self) -> None: ...  # pragma: no cover - protocol
    def is_set(self) -> bool: ...  # pragma: no cover - protocol

    async def wait(self) -> bool: ...  # pragma: no cover - protocol


class CoinDeskRSSPump:
    """Async pump polling the CoinDesk RSS endpoint into a sink.

    The sink callable runs synchronously inside the asyncio loop; for
    cross-thread state mutation use a :class:`ui.feeds.runner.FeedRunner`-
    style wrapper.
    """

    def __init__(
        self,
        sink: Callable[[NewsItem], None],
        *,
        clock_ns: Callable[[], int],
        fetch: HTTPFetch | None = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        reconnect_delay_s: float = DEFAULT_RECONNECT_DELAY_S,
        reconnect_delay_max_s: float = DEFAULT_RECONNECT_DELAY_MAX_S,
        url: str | None = None,
        source: str = SOURCE_TAG,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError(
                "CoinDeskRSSPump: poll_interval_s must be positive"
            )
        if reconnect_delay_s <= 0:
            raise ValueError(
                "CoinDeskRSSPump: reconnect_delay_s must be positive"
            )
        if reconnect_delay_max_s < reconnect_delay_s:
            raise ValueError(
                "CoinDeskRSSPump: reconnect_delay_max_s must be >= "
                "reconnect_delay_s"
            )
        if not source:
            raise ValueError("CoinDeskRSSPump: source must be non-empty")
        self._sink = sink
        self._clock_ns = clock_ns
        self._fetch: HTTPFetch = fetch if fetch is not None else _default_fetch
        self._poll_interval_s = poll_interval_s
        self._reconnect_delay_s = reconnect_delay_s
        self._reconnect_delay_max_s = reconnect_delay_max_s
        self._url = url if url is not None else make_coindesk_rss_url()
        self._source = source
        self._stop_event = asyncio.Event()
        self._items_received = 0
        self._polls = 0
        self._errors = 0
        self._last_poll_ts_ns: int | None = None
        self._last_item_ts_ns: int | None = None
        self._running = False
        self._seen_guids: set[str] = set()

    @property
    def url(self) -> str:
        return self._url

    @property
    def source(self) -> str:
        return self._source

    def status(self) -> NewsFeedStatus:
        return NewsFeedStatus(
            running=self._running,
            source=self._source,
            url=self._url,
            last_poll_ts_ns=self._last_poll_ts_ns,
            last_item_ts_ns=self._last_item_ts_ns,
            items_received=self._items_received,
            polls=self._polls,
            errors=self._errors,
        )

    def stop(self) -> None:
        """Signal the run loop to exit on its next iteration.

        Safe to call from any thread that holds a reference to this
        pump's event loop (use ``loop.call_soon_threadsafe(pump.stop)``
        from outside the loop).
        """
        self._stop_event.set()

    async def run(self) -> None:
        """Poll → parse → emit → wait until ``stop()``.

        Backs off exponentially from ``reconnect_delay_s`` up to
        ``reconnect_delay_max_s`` between failed polls. The first
        successful poll resets the delay floor.
        """
        self._running = True
        delay = self._reconnect_delay_s
        try:
            while not self._stop_event.is_set():
                ts_ns = self._clock_ns()
                self._polls += 1
                self._last_poll_ts_ns = ts_ns
                try:
                    payload = await self._fetch(self._url)
                except Exception:  # noqa: BLE001 - log + reconnect
                    self._errors += 1
                    LOG.exception(
                        "coindesk_rss: fetch failure; retrying in %.1fs",
                        delay,
                    )
                    if await self._sleep_or_stop(delay):
                        break
                    delay = min(delay * 2, self._reconnect_delay_max_s)
                    continue
                delay = self._reconnect_delay_s
                self._dispatch(payload, ts_ns=ts_ns)
                if await self._sleep_or_stop(self._poll_interval_s):
                    break
        finally:
            self._running = False

    def _dispatch(self, payload: bytes | str, *, ts_ns: int) -> None:
        """Parse one fetched payload and forward novel items to the sink."""
        try:
            items = parse_rss_feed(
                payload, ts_ns=ts_ns, source=self._source
            )
        except Exception:  # noqa: BLE001 - parser is best-effort
            self._errors += 1
            LOG.exception("coindesk_rss: parser failure")
            return
        for item in items:
            if item.guid in self._seen_guids:
                continue
            try:
                self._sink(item)
            except Exception:  # noqa: BLE001 - never poison the loop
                self._errors += 1
                LOG.exception("coindesk_rss: sink failure for %s", item.guid)
                continue
            # Mark as seen only after successful delivery so a transient
            # sink failure (e.g. a downstream restart) gets retried on the
            # next poll instead of being silently dropped forever.
            self._seen_guids.add(item.guid)
            self._items_received += 1
            self._last_item_ts_ns = item.ts_ns

    async def _sleep_or_stop(self, delay: float) -> bool:
        """Sleep up to ``delay`` seconds or until ``stop()``.

        Returns ``True`` if ``stop()`` was signalled during the wait
        (caller should break the run loop), ``False`` otherwise.
        """
        if delay <= 0:
            return self._stop_event.is_set()
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
            return True
        except TimeoutError:
            return False


__all__: Sequence[str] = (
    "COINDESK_RSS_URL",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_RECONNECT_DELAY_S",
    "DEFAULT_RECONNECT_DELAY_MAX_S",
    "SOURCE_TAG",
    "CoinDeskRSSPump",
    "HTTPFetch",
    "NewsFeedStatus",
    "make_coindesk_rss_url",
    "parse_rss_feed",
)


def _utc_now_dt() -> datetime:  # pragma: no cover - intentionally unused
    """Reserved hook for callers needing a UTC clock; never called from
    pure helpers above so INV-15 stays intact."""
    from system.time_source import utc_now

    return utc_now()
