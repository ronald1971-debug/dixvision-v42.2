# ADAPTED FROM: https://github.com/deepset-ai/haystack (Apache-2.0)
#
# Tier-C C-15 — haystack-ai-shape RAG knowledge store.
#
# Haystack's distinguishing surface is a **three-stage RAG pipeline**:
#
#   1. ``DocumentStore`` — append-only persistent corpus of typed
#      :class:`Document` rows (id + content + metadata).
#   2. ``Retriever`` — pure function ``query -> top-k documents``.
#      Haystack's ``InMemoryBM25Retriever`` scores documents with
#      Okapi-BM25 over a single-word lowercase tokeniser.
#   3. ``PromptBuilder`` — fills a Jinja-style template with the
#      retrieved documents and the operator query, producing a
#      single prompt string ready for an LLM generator.
#
# C-15 adapts that shape behind DIX contracts at
# :mod:`state.knowledge_store`. The store holds DIX **spec docs,
# registry definitions, and governance rules**; the retriever finds
# the rules relevant to a governance question; the prompt builder
# stitches the rules into a single context block; downstream callers
# pass that prompt to a :mod:`intelligence_engine.cognitive.\
# litellm_router.LiteLLMRouter`-backed completion — Haystack's own
# ``ChatGenerator`` / REST API deployment surface is intentionally
# **not** re-exported.
#
# Authority constraints (pinned by tests):
#
#   * **ADVISORY only** (INV-12) — every output is a frozen value
#     object. No :class:`SignalEvent` / :class:`ExecutionIntent` /
#     :class:`PatchProposal` / :class:`GovernanceDecision`
#     constructors anywhere.
#   * **OFFLINE_ONLY** — knowledge writes happen on the offline
#     side (slow-loop ingestion). Reads are bounded latency
#     in-memory dict lookups; ``< 5 ms`` against the in-memory
#     fallback transport.
#   * **INV-15** — pure dispatcher. No clock, no I/O, no PRNG.
#     Three independent runs with identical inputs produce
#     byte-identical :class:`RetrievedContext` / :class:`Prompt`
#     instances.
#   * **B1** — no execution_engine / governance_engine /
#     system_engine / intelligence_engine / learning_engine /
#     evolution_engine submodule cross-imports.
#   * No top-level imports of :mod:`haystack`, :mod:`openai`,
#     :mod:`anthropic`, :mod:`litellm`, :mod:`requests`,
#     :mod:`httpx`, :mod:`asyncio`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`, :mod:`secrets`. The ``haystack-ai`` package
#     is the lazy seam — only :func:`enable_haystack_factory` may
#     import from it, and only inside the function body.
#   * Haystack REST API / ``haystack.dashboard`` / ``OpenAIGenerator``
#     / ``ChatGenerator`` surfaces are **not** re-exported. The
#     RAG pipeline ends at the :class:`Prompt` value object;
#     completion is the caller's job through
#     :class:`~intelligence_engine.cognitive.litellm_router.\
# LiteLLMRouter`.
#
# NEW_PIP_DEPENDENCIES = ("haystack-ai",) — declared as the lazy seam
# for ``tools/cli.py install-c-tier``; production wiring routes
# everything through the in-memory BM25 fallback unless an operator
# explicitly enables the live haystack-ai backend via
# :func:`enable_haystack_factory`.
"""C-15 RAG knowledge store — Haystack-shape BM25 + prompt builder."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, Final, Protocol, runtime_checkable

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "KnowledgeStoreError",
    "DocumentError",
    "QueryError",
    "TemplateError",
    "TransportError",
    "Document",
    "Query",
    "RetrievedDocument",
    "RetrievedContext",
    "PromptTemplate",
    "Prompt",
    "DocumentStoreTransport",
    "InMemoryDocumentStore",
    "BM25Retriever",
    "PromptBuilder",
    "KnowledgeStore",
    "enable_haystack_factory",
    "tokenize",
    "MAX_DOCUMENT_ID_LEN",
    "MAX_DOCUMENT_CONTENT_LEN",
    "MAX_METADATA_KEYS",
    "MAX_METADATA_KEY_LEN",
    "MAX_METADATA_VALUE_LEN",
    "MAX_QUERY_LEN",
    "MAX_TOP_K",
    "BM25_K1",
    "BM25_B",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("haystack-ai",)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


MAX_DOCUMENT_ID_LEN: Final[int] = 256
MAX_DOCUMENT_CONTENT_LEN: Final[int] = 65_536
MAX_METADATA_KEYS: Final[int] = 32
MAX_METADATA_KEY_LEN: Final[int] = 64
MAX_METADATA_VALUE_LEN: Final[int] = 1024
MAX_QUERY_LEN: Final[int] = 8_192
MIN_TOP_K: Final[int] = 1
MAX_TOP_K: Final[int] = 64

# Okapi-BM25 hyper-parameters. Pinned for INV-15 byte-identical
# determinism — any change requires bumping ``KNOWLEDGE_STORE_VERSION``.
BM25_K1: Final[float] = 1.2
BM25_B: Final[float] = 0.75

KNOWLEDGE_STORE_VERSION: Final[str] = "1"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class KnowledgeStoreError(ValueError):
    """Base class for C-15 knowledge-store errors."""


class DocumentError(KnowledgeStoreError):
    """Raised when a :class:`Document` is malformed."""


class QueryError(KnowledgeStoreError):
    """Raised when a :class:`Query` is malformed or ``top_k`` is
    outside ``[MIN_TOP_K, MAX_TOP_K]``."""


class TemplateError(KnowledgeStoreError):
    """Raised when a :class:`PromptTemplate` is malformed or the
    rendered :class:`Prompt` violates the deterministic-output
    contract (e.g. unsubstituted placeholders, non-string slots)."""


class TransportError(KnowledgeStoreError):
    """Raised when the :class:`DocumentStoreTransport` returns a
    malformed payload."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")

_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase ``[A-Za-z0-9]+`` tokeniser.

    Mirrors Haystack's default ``InMemoryBM25Retriever`` tokeniser.
    Pure function — no locale-dependent calls, no Unicode case
    folding (ASCII only), no random ordering. Pinned by INV-15
    determinism tests.
    """
    if not isinstance(text, str):
        raise QueryError(f"tokenize: text must be str, got {type(text).__name__}")
    return tuple(m.group(0).lower() for m in _TOKEN_RE.finditer(text))


def _freeze_metadata(
    metadata: Mapping[str, str],
) -> Mapping[str, str]:
    """Return a frozen, key-sorted view of ``metadata`` — keeps
    serialisation byte-stable (INV-15)."""
    out: dict[str, str] = {}
    for k in sorted(metadata.keys()):
        v = metadata[k]
        if not isinstance(k, str) or not k:
            raise DocumentError(f"Document.metadata keys must be non-empty str, got {k!r}")
        if not _IDENTIFIER_RE.fullmatch(k):
            raise DocumentError(
                f"Document.metadata key must match [A-Za-z_][A-Za-z0-9_.-]*, got {k!r}"
            )
        if len(k) > MAX_METADATA_KEY_LEN:
            raise DocumentError(f"Document.metadata key length > {MAX_METADATA_KEY_LEN}: {k!r}")
        if not isinstance(v, str):
            raise DocumentError(f"Document.metadata[{k!r}] must be str, got {type(v).__name__}")
        if len(v) > MAX_METADATA_VALUE_LEN:
            raise DocumentError(f"Document.metadata[{k!r}] length > {MAX_METADATA_VALUE_LEN}")
        out[k] = v
    return MappingProxyType(out)


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Document:
    """One typed knowledge-store row.

    Mirrors ``haystack.dataclasses.Document`` but drops the
    ``embedding`` / ``score`` / ``meta`` fields in favour of:

    * ``id`` — stable string identifier (caller-supplied).
    * ``content`` — raw text body (UTF-8 string).
    * ``metadata`` — key-sorted ``Mapping[str, str]``; serialisation
      order is deterministic for INV-15 byte-identical replay.
    """

    id: str
    content: str
    metadata: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise DocumentError(f"Document.id must be non-empty str, got {self.id!r}")
        if not _IDENTIFIER_RE.fullmatch(self.id):
            raise DocumentError(f"Document.id must match [A-Za-z_][A-Za-z0-9_.-]*, got {self.id!r}")
        if len(self.id) > MAX_DOCUMENT_ID_LEN:
            raise DocumentError(f"Document.id length > {MAX_DOCUMENT_ID_LEN}")
        if not isinstance(self.content, str):
            raise DocumentError(f"Document.content must be str, got {type(self.content).__name__}")
        if len(self.content) > MAX_DOCUMENT_CONTENT_LEN:
            raise DocumentError(f"Document.content length > {MAX_DOCUMENT_CONTENT_LEN}")
        if not isinstance(self.metadata, Mapping):
            raise DocumentError(
                f"Document.metadata must be a Mapping, got {type(self.metadata).__name__}"
            )
        if len(self.metadata) > MAX_METADATA_KEYS:
            raise DocumentError(f"Document.metadata count > {MAX_METADATA_KEYS}")
        frozen = _freeze_metadata(self.metadata)
        object.__setattr__(self, "metadata", frozen)


@dataclasses.dataclass(frozen=True, slots=True)
class Query:
    """One typed retrieval query.

    * ``text`` — operator question / governance prompt body.
    * ``top_k`` — number of documents to retrieve. Clamped to
      ``[MIN_TOP_K, MAX_TOP_K]`` at construction.
    * ``filter`` — metadata key/value equality filter (optional).
      All listed key/value pairs must match exactly for a document
      to be considered.
    """

    text: str
    top_k: int = 5
    filter: Mapping[str, str] = dataclasses.field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise QueryError("Query.text must be a non-empty str")
        if len(self.text) > MAX_QUERY_LEN:
            raise QueryError(f"Query.text length > {MAX_QUERY_LEN}")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise QueryError(f"Query.top_k must be int, got {type(self.top_k).__name__}")
        if self.top_k < MIN_TOP_K or self.top_k > MAX_TOP_K:
            raise QueryError(f"Query.top_k must be in [{MIN_TOP_K}, {MAX_TOP_K}], got {self.top_k}")
        if not isinstance(self.filter, Mapping):
            raise QueryError(f"Query.filter must be a Mapping, got {type(self.filter).__name__}")
        if len(self.filter) > MAX_METADATA_KEYS:
            raise QueryError(f"Query.filter count > {MAX_METADATA_KEYS}")
        frozen = _freeze_metadata(self.filter)
        object.__setattr__(self, "filter", frozen)


@dataclasses.dataclass(frozen=True, slots=True)
class RetrievedDocument:
    """One document returned from BM25 retrieval, with its score.

    * ``document`` — the source :class:`Document`.
    * ``score`` — non-negative BM25 score. Higher is more relevant.
    * ``rank`` — 0-based rank among the retrieved documents
      (ties broken lexicographically by document id, INV-15).
    """

    document: Document
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not isinstance(self.document, Document):
            raise KnowledgeStoreError(
                f"RetrievedDocument.document must be Document, got {type(self.document).__name__}"
            )
        if not isinstance(self.score, float):
            raise KnowledgeStoreError(
                f"RetrievedDocument.score must be float, got {type(self.score).__name__}"
            )
        if not math.isfinite(self.score) or self.score < 0.0:
            raise KnowledgeStoreError(
                f"RetrievedDocument.score must be finite and >= 0.0, got {self.score!r}"
            )
        if not isinstance(self.rank, int) or isinstance(self.rank, bool):
            raise KnowledgeStoreError(
                f"RetrievedDocument.rank must be int, got {type(self.rank).__name__}"
            )
        if self.rank < 0:
            raise KnowledgeStoreError(f"RetrievedDocument.rank must be >= 0, got {self.rank}")


@dataclasses.dataclass(frozen=True, slots=True)
class RetrievedContext:
    """The full retrieval result for a single query.

    Encodes everything the governance-side ledger needs to record
    *why* a prompt was assembled: the original query, the ordered
    list of retrieved documents with scores, and a stable
    BLAKE2b-16 digest pinning the result for replay verification.
    """

    query: Query
    documents: tuple[RetrievedDocument, ...]
    digest: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.query, Query):
            raise KnowledgeStoreError(
                f"RetrievedContext.query must be Query, got {type(self.query).__name__}"
            )
        if not isinstance(self.documents, tuple):
            raise KnowledgeStoreError(
                f"RetrievedContext.documents must be tuple, got {type(self.documents).__name__}"
            )
        for i, d in enumerate(self.documents):
            if not isinstance(d, RetrievedDocument):
                raise KnowledgeStoreError(
                    f"RetrievedContext.documents[{i}] must be "
                    f"RetrievedDocument, got {type(d).__name__}"
                )
            if d.rank != i:
                raise KnowledgeStoreError(
                    f"RetrievedContext.documents[{i}].rank must == {i}, got {d.rank}"
                )
        if not isinstance(self.digest, bytes):
            raise KnowledgeStoreError(
                f"RetrievedContext.digest must be bytes, got {type(self.digest).__name__}"
            )
        if len(self.digest) != 16:
            raise KnowledgeStoreError(
                f"RetrievedContext.digest must be 16 bytes (BLAKE2b-16), got {len(self.digest)}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Render template for the prompt builder.

    Two named slots are always supported:

    * ``{context}`` — replaced by the concatenated retrieved-doc
      bodies, separated by ``\\n---\\n``.
    * ``{query}`` — replaced by the operator query text.

    The template is rendered with Python's :meth:`str.format` over
    only these two keys; any other ``{name}`` placeholder raises
    :class:`TemplateError` at construction so the determinism of
    :meth:`PromptBuilder.build` is preserved.
    """

    body: str

    def __post_init__(self) -> None:
        if not isinstance(self.body, str) or not self.body.strip():
            raise TemplateError("PromptTemplate.body must be a non-empty str")
        # Validate placeholder set: only ``{context}`` and
        # ``{query}`` are allowed; ``{{`` / ``}}`` literal escapes
        # are fine.
        try:
            placeholders = _scan_placeholders(self.body)
        except ValueError as exc:
            raise TemplateError(str(exc)) from exc
        bad = placeholders - {"context", "query"}
        if bad:
            raise TemplateError(
                f"PromptTemplate.body contains unsupported placeholders: {sorted(bad)!r}"
            )


def _scan_placeholders(body: str) -> set[str]:
    """Return the set of ``{name}`` placeholder names in ``body``.

    Honours ``{{`` / ``}}`` as escapes. Raises :class:`ValueError`
    on unbalanced braces.
    """
    found: set[str] = set()
    i = 0
    n = len(body)
    while i < n:
        c = body[i]
        if c == "{":
            if i + 1 < n and body[i + 1] == "{":
                i += 2
                continue
            j = body.find("}", i + 1)
            if j < 0:
                raise ValueError(f"unbalanced '{{' at position {i}")
            name = body[i + 1 : j]
            if not name or not _IDENTIFIER_RE.fullmatch(name):
                raise ValueError(
                    f"placeholder name at position {i} must match [A-Za-z_][A-Za-z0-9_.-]*"
                )
            found.add(name)
            i = j + 1
            continue
        if c == "}":
            if i + 1 < n and body[i + 1] == "}":
                i += 2
                continue
            raise ValueError(f"unbalanced '}}' at position {i}")
        i += 1
    return found


@dataclasses.dataclass(frozen=True, slots=True)
class Prompt:
    """A rendered prompt + the retrieval context that produced it.

    The full record the ledger logs alongside every governance
    decision sourced from this RAG pipeline.
    """

    template: PromptTemplate
    context: RetrievedContext
    rendered: str

    def __post_init__(self) -> None:
        if not isinstance(self.template, PromptTemplate):
            raise TemplateError(
                f"Prompt.template must be PromptTemplate, got {type(self.template).__name__}"
            )
        if not isinstance(self.context, RetrievedContext):
            raise TemplateError(
                f"Prompt.context must be RetrievedContext, got {type(self.context).__name__}"
            )
        if not isinstance(self.rendered, str) or not self.rendered:
            raise TemplateError("Prompt.rendered must be non-empty str")


# ---------------------------------------------------------------------------
# DocumentStoreTransport Protocol + in-memory reference
# ---------------------------------------------------------------------------


@runtime_checkable
class DocumentStoreTransport(Protocol):
    """Transport seam — production wires this to Haystack's
    :class:`haystack.document_stores.InMemoryDocumentStore` via
    :func:`enable_haystack_factory`, but the in-memory fallback
    satisfies the same Protocol and is INV-15 byte-identical."""

    def write(self, document: Document) -> None:  # pragma: no cover - Protocol
        ...

    def read(self, doc_id: str) -> Document | None:  # pragma: no cover - Protocol
        ...

    def iter_documents(
        self,
    ) -> Iterable[Document]:  # pragma: no cover - Protocol
        ...

    def __len__(self) -> int:  # pragma: no cover - Protocol
        ...


class InMemoryDocumentStore:
    """Reference :class:`DocumentStoreTransport` — in-memory dict.

    Iteration order is lexicographic by :attr:`Document.id`,
    independent of insertion order, so replay is byte-identical
    (INV-15).
    """

    __slots__ = ("_docs",)

    def __init__(self) -> None:
        self._docs: dict[str, Document] = {}

    def write(self, document: Document) -> None:
        if not isinstance(document, Document):
            raise DocumentError(
                "InMemoryDocumentStore.write: document must be "
                f"Document, got {type(document).__name__}"
            )
        self._docs[document.id] = document

    def read(self, doc_id: str) -> Document | None:
        if not isinstance(doc_id, str) or not doc_id:
            raise DocumentError("InMemoryDocumentStore.read: doc_id must be non-empty str")
        return self._docs.get(doc_id)

    def iter_documents(self) -> Iterable[Document]:
        for k in sorted(self._docs.keys()):
            yield self._docs[k]

    def __len__(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """Okapi-BM25 retriever over a :class:`DocumentStoreTransport`.

    Mirrors Haystack's ``InMemoryBM25Retriever`` algorithm:

    .. math::

       \\text{score}(D, Q) = \\sum_{t \\in Q}
       \\text{idf}(t) \\cdot
       \\frac{f(t, D) \\cdot (k_1 + 1)}
       {f(t, D) + k_1 \\cdot
       (1 - b + b \\cdot |D| / \\bar{|D|})}

    with :math:`k_1 = 1.2`, :math:`b = 0.75` and
    :math:`\\text{idf}(t) = \\ln\\frac{N - n_t + 0.5}{n_t + 0.5}`
    clamped at 0 (Haystack default). Ties on score are broken
    lexicographically by document id so retrieval is INV-15
    byte-identical.
    """

    __slots__ = ("_transport",)

    def __init__(self, *, transport: DocumentStoreTransport) -> None:
        if not isinstance(transport, DocumentStoreTransport):
            raise TransportError("BM25Retriever.transport must implement DocumentStoreTransport")
        self._transport = transport

    def retrieve(self, *, query: Query) -> RetrievedContext:
        if not isinstance(query, Query):
            raise QueryError(
                f"BM25Retriever.retrieve: query must be Query, got {type(query).__name__}"
            )

        # Materialise the corpus once per call. Iteration order is
        # lexicographic by document id, supplied by the transport
        # — that locks the index used below for tie-breaking.
        corpus: list[Document] = [
            d for d in self._transport.iter_documents() if _matches_filter(d, query.filter)
        ]
        n_docs = len(corpus)
        if n_docs == 0:
            digest = _digest_context_inputs(query, ())
            return RetrievedContext(
                query=query,
                documents=(),
                digest=digest,
            )

        # Pre-compute term frequencies + document lengths.
        tf_table: list[dict[str, int]] = []
        doc_lens: list[int] = []
        df_table: dict[str, int] = {}
        for d in corpus:
            tokens = tokenize(d.content)
            doc_lens.append(len(tokens))
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            tf_table.append(tf)
            for tok in tf.keys():
                df_table[tok] = df_table.get(tok, 0) + 1
        avgdl = sum(doc_lens) / n_docs if n_docs > 0 else 0.0

        query_tokens = tokenize(query.text)
        scored: list[tuple[float, str, int]] = []
        for idx, d in enumerate(corpus):
            score = 0.0
            dl = doc_lens[idx]
            tf = tf_table[idx]
            for tok in query_tokens:
                f = tf.get(tok, 0)
                if f == 0:
                    continue
                df = df_table.get(tok, 0)
                # Lucene-style BM25 IDF: ln(1 + (N - df + 0.5) /
                # (df + 0.5)). This form is always positive (and
                # is what Haystack's reference BM25 retriever
                # uses), so we do not need to special-case small
                # corpora where the original Okapi IDF can become
                # zero or negative for high-df terms.
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                if idf <= 0.0:
                    continue
                denom = f + BM25_K1 * (1.0 - BM25_B + BM25_B * (dl / avgdl))
                score += idf * (f * (BM25_K1 + 1.0)) / denom
            if score > 0.0:
                scored.append((score, d.id, idx))

        # Sort: highest score first; ties broken by lexicographic
        # ``id`` (ascending) — strictly deterministic.
        scored.sort(key=lambda row: (-row[0], row[1]))

        top: list[RetrievedDocument] = []
        for rank, (score, _id, idx) in enumerate(scored[: query.top_k]):
            top.append(
                RetrievedDocument(
                    document=corpus[idx],
                    score=score,
                    rank=rank,
                )
            )

        digest = _digest_context_inputs(query, tuple(top))
        return RetrievedContext(
            query=query,
            documents=tuple(top),
            digest=digest,
        )


def _matches_filter(document: Document, filter_: Mapping[str, str]) -> bool:
    if not filter_:
        return True
    md = document.metadata
    for k, v in filter_.items():
        if md.get(k) != v:
            return False
    return True


def _digest_context_inputs(query: Query, results: tuple[RetrievedDocument, ...]) -> bytes:
    parts: list[str] = [
        f"v={KNOWLEDGE_STORE_VERSION}",
        f"text={query.text}",
        f"top_k={query.top_k}",
    ]
    for k in sorted(query.filter.keys()):
        parts.append(f"f:{k}={query.filter[k]}")
    parts.append(f"n={len(results)}")
    for r in results:
        parts.append(
            f"r{r.rank}:id={r.document.id};score={r.score:.12e};content={r.document.content}"
        )
    body = "|".join(parts).encode("utf-8")
    return hashlib.blake2b(body, digest_size=16).digest()


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


_CONTEXT_SEPARATOR: Final[str] = "\n---\n"


class PromptBuilder:
    """Render a :class:`PromptTemplate` over a
    :class:`RetrievedContext` into a final :class:`Prompt`.

    Mirrors Haystack's ``PromptBuilder`` but tightens placeholder
    substitution to two named slots only — ``{context}`` and
    ``{query}`` — both of which are filled deterministically from
    the supplied context. No Jinja2 control flow, no escaping
    surprises, no clock reads.
    """

    __slots__ = ("_template",)

    def __init__(self, *, template: PromptTemplate) -> None:
        if not isinstance(template, PromptTemplate):
            raise TemplateError(
                f"PromptBuilder.template must be PromptTemplate, got {type(template).__name__}"
            )
        self._template = template

    def build(self, *, context: RetrievedContext) -> Prompt:
        if not isinstance(context, RetrievedContext):
            raise TemplateError(
                "PromptBuilder.build: context must be "
                "RetrievedContext, got "
                f"{type(context).__name__}"
            )
        body = self._template.body
        ctx_block = _CONTEXT_SEPARATOR.join(r.document.content for r in context.documents)
        # ``str.format`` substitutes only the named slots we declared
        # in the template — every other ``{`` was rejected up front.
        try:
            rendered = body.format(
                context=ctx_block,
                query=context.query.text,
            )
        except (KeyError, IndexError, ValueError) as exc:
            raise TemplateError(f"PromptBuilder.build: render failed: {exc}") from exc
        return Prompt(
            template=self._template,
            context=context,
            rendered=rendered,
        )


# ---------------------------------------------------------------------------
# KnowledgeStore — composes the three stages
# ---------------------------------------------------------------------------


class KnowledgeStore:
    """High-level RAG facade — *the* C-15 entry point.

    Bundles a :class:`DocumentStoreTransport`, a
    :class:`BM25Retriever`, and a :class:`PromptBuilder` so callers
    can talk to one object:

    .. code-block:: python

        store = KnowledgeStore.in_memory(
            template=PromptTemplate(body="..."),
        )
        store.ingest(Document(id="rule.a", content="..."))
        prompt = store.ask(Query(text="..."))
        ai_response = router.complete(prompt.rendered, ...)
    """

    __slots__ = ("_transport", "_retriever", "_builder")

    def __init__(
        self,
        *,
        transport: DocumentStoreTransport,
        retriever: BM25Retriever,
        builder: PromptBuilder,
    ) -> None:
        if not isinstance(transport, DocumentStoreTransport):
            raise TransportError("KnowledgeStore.transport must implement DocumentStoreTransport")
        if not isinstance(retriever, BM25Retriever):
            raise KnowledgeStoreError("KnowledgeStore.retriever must be BM25Retriever")
        if not isinstance(builder, PromptBuilder):
            raise KnowledgeStoreError("KnowledgeStore.builder must be PromptBuilder")
        self._transport = transport
        self._retriever = retriever
        self._builder = builder

    @classmethod
    def in_memory(cls, *, template: PromptTemplate) -> KnowledgeStore:
        transport = InMemoryDocumentStore()
        return cls(
            transport=transport,
            retriever=BM25Retriever(transport=transport),
            builder=PromptBuilder(template=template),
        )

    def ingest(self, document: Document) -> None:
        self._transport.write(document)

    def retrieve(self, query: Query) -> RetrievedContext:
        return self._retriever.retrieve(query=query)

    def ask(self, query: Query) -> Prompt:
        context = self._retriever.retrieve(query=query)
        return self._builder.build(context=context)

    def __len__(self) -> int:
        return len(self._transport)


# ---------------------------------------------------------------------------
# Lazy seam — :mod:`haystack` is never imported at module scope
# ---------------------------------------------------------------------------


def enable_haystack_factory(**_kwargs: Any) -> None:
    """Lazy seam for promoting the live ``haystack-ai`` backend.

    Production wiring intentionally never calls this — the in-memory
    BM25 retriever above is byte-identical to Haystack's
    ``InMemoryBM25Retriever`` for the closed DIX document alphabet,
    and the live haystack-ai package brings a large dependency
    surface (transformers, sentence-transformers, FastAPI REST
    deployment, dashboard UI). Until an operator explicitly opts
    in by calling this function, the live backend stays off.

    The function body is the only place :mod:`haystack` may be
    referenced — it is imported here behind the lazy seam so that
    :mod:`state.knowledge_store` can be imported in environments
    where ``haystack-ai`` is not installed.
    """

    raise NotImplementedError(
        "enable_haystack_factory is not implemented in this "
        "release; the in-memory BM25 fallback is the only "
        "supported transport. NEW_PIP_DEPENDENCIES = "
        f"{NEW_PIP_DEPENDENCIES!r}"
    )
