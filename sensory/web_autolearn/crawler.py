"""WEBLEARN-01 — Crawler protocol + deterministic test impl.

The production crawler is a Playwright-backed implementation that
fetches URLs from :file:`seeds.yaml`. It is *not* in this module
because Playwright is a heavyweight optional dep — adding it to base
deps would slow down every CI run that doesn't need it.

Instead, this module declares the :class:`Crawler` :class:`Protocol`
that downstream code (AIFilter, Curator, PendingBuffer) depends on,
and ships a deterministic in-memory implementation
:class:`DeterministicCrawler` for tests and replay scenarios.

A concrete Playwright crawler will live alongside this file (e.g.
``sensory/web_autolearn/crawler_playwright.py``) once the data adapter
sprint lands. Both will satisfy the same Protocol so the rest of the
pipeline does not need to know which is in use.

Authority discipline (per :mod:`sensory` docstring): a Crawler does
*not* import any engine, does *not* mutate the SystemMode FSM, and
does *not* write to the audit ledger. It only emits
:class:`RawDocument` instances.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from sensory.web_autolearn.contracts import RawDocument


@runtime_checkable
class Crawler(Protocol):
    """A web autolearn crawler.

    Implementations must be:

      * **stateless across calls** — :meth:`fetch` is called once per
        seed list and must not retain any state that would change
        future outputs given the same inputs;
      * **deterministic for replay** — given identical seeds and an
        identical environment (same ``ts_ns``), :meth:`fetch` must
        return the same :class:`RawDocument` sequence in the same
        order. The Playwright impl achieves this by sorting URLs and
        pinning ``ts_ns`` to the caller's clock;
      * **fail-soft** — a fetch failure for one URL must not abort
        the whole batch. Failed fetches return a :class:`RawDocument`
        with ``fetched_ok=False`` so downstream filters can drop them.
    """

    def fetch(
        self,
        seeds: Sequence[str],
        *,
        ts_ns: int,
    ) -> Sequence[RawDocument]:
        """Fetch one document per seed URL.

        Args:
            seeds: Sequence of seed identifiers (matching ``seeds.yaml``
                rows). The crawler resolves each to a URL internally.
            ts_ns: Caller-supplied ingestion timestamp; every produced
                :class:`RawDocument` carries this exact value
                (deterministic-replay invariant).

        Returns:
            Sequence of :class:`RawDocument` — one per input seed,
            in the same order. Failed fetches still produce a
            :class:`RawDocument` (with ``fetched_ok=False``).
        """
        ...


@dataclass(frozen=True, slots=True)
class _PreparedDocument:
    """One document the deterministic crawler will hand out, by seed_id.

    Decoupled from :class:`RawDocument` so the test fixture can omit
    ``ts_ns`` (which the crawler injects per-call from its caller).
    """

    seed_id: str
    url: str
    title: str = ""
    body: str = ""
    fetched_ok: bool = True
    meta: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeterministicCrawler:
    """In-memory crawler used by tests and replay scenarios.

    Holds a fixed seed-id-keyed table of prepared documents. Each call
    to :meth:`fetch` returns documents in the exact order the caller
    requested seeds (preserving any duplicates). Unknown seeds yield
    a :class:`RawDocument` with ``fetched_ok=False`` so the contract
    matches the Playwright fail-soft behavior.

    Construct with :meth:`from_pairs` for the common case of a static
    fixture::

        crawler = DeterministicCrawler.from_pairs(
            ("seed_a", "https://a.example/feed"),
            ("seed_b", "https://b.example/feed"),
        )
    """

    documents: tuple[_PreparedDocument, ...]

    @classmethod
    def from_pairs(
        cls,
        *pairs: tuple[str, str],
    ) -> DeterministicCrawler:
        """Build a crawler from ``(seed_id, url)`` pairs.

        Each pair becomes a successful 200-equivalent document with
        empty title/body. Use :meth:`from_documents` when you need
        non-default fields.
        """

        prepared = tuple(
            _PreparedDocument(seed_id=s, url=u) for s, u in pairs
        )
        return cls(documents=prepared)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[_PreparedDocument],
    ) -> DeterministicCrawler:
        """Build a crawler from explicit prepared documents."""

        return cls(documents=tuple(documents))

    def _by_seed(self) -> dict[str, _PreparedDocument]:
        # The first prepared doc per seed_id wins; later duplicates
        # are ignored so callers can express override semantics by
        # putting the canonical row first.
        out: dict[str, _PreparedDocument] = {}
        for prep in self.documents:
            out.setdefault(prep.seed_id, prep)
        return out

    def fetch(
        self,
        seeds: Sequence[str],
        *,
        ts_ns: int,
    ) -> Sequence[RawDocument]:
        """Return one :class:`RawDocument` per requested seed."""

        if ts_ns <= 0:
            raise ValueError(
                "DeterministicCrawler.fetch ts_ns must be positive"
            )
        table = self._by_seed()
        out: list[RawDocument] = []
        for seed_id in seeds:
            prep = table.get(seed_id)
            if prep is None:
                # Unknown seed: emit a fail-soft placeholder so the
                # downstream pipeline can drop it cleanly.
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
            out.append(
                RawDocument(
                    ts_ns=ts_ns,
                    seed_id=prep.seed_id,
                    url=prep.url,
                    title=prep.title,
                    body=prep.body,
                    fetched_ok=prep.fetched_ok,
                    meta=dict(prep.meta),
                )
            )
        return tuple(out)


__all__ = [
    "Crawler",
    "DeterministicCrawler",
    "_PreparedDocument",
]
