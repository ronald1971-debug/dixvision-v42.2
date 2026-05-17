# ADAPTED FROM: https://github.com/run-llama/llama_index (MIT)
#
# Tier-C C-16 — llama-index-shape long-term structured memory.
#
# LlamaIndex's distinguishing surface is a **VectorStoreIndex + query
# engine** pipeline:
#
#   1. ``StorageContext`` — owns a typed ``VectorStore`` of
#      ``(node_id, content, embedding, metadata)`` rows.
#   2. ``VectorStoreIndex`` — builds the index over a corpus of
#      ``Document``-shaped nodes via an embedding model.
#   3. ``QueryEngine`` — pure function ``query -> top-k nodes`` via
#      cosine / dot-product similarity over the embedding space, with
#      an optional metadata equality filter.
#   4. The retrieved nodes are stitched into a prompt and sent to an
#      LLM by ``ResponseSynthesizer``.
#
# C-16 adapts that shape behind DIX contracts at
# :mod:`state.knowledge_store_llamaindex`. The store indexes DIX
# **docs/, registry/, and governance decision history** for long-term
# temporal memory; the retriever finds semantically-related rows for a
# governance question; the prompt builder stitches them into a single
# context block; downstream callers route the prompt through
# :mod:`intelligence_engine.cognitive.litellm_router.LiteLLMRouter`,
# **never** a direct LlamaIndex LLM-SDK call.
#
# C-16 is a *drop-in alternative* to C-15. It deliberately re-exports
# the C-15 value-object surface (``Document`` / ``Query`` /
# ``RetrievedDocument`` / ``RetrievedContext`` / ``PromptTemplate`` /
# ``Prompt`` / ``PromptBuilder``) from :mod:`state.knowledge_store` so
# that a caller can swap the retriever implementation by changing one
# import. The only retrieval-surface difference is the scoring family:
#
#   * C-15 (Haystack) — Lucene-form BM25 over a lexical tokeniser.
#   * C-16 (LlamaIndex) — cosine similarity over a deterministic
#     hashing-TF embedding (no vendor deps; lazy seam for the live
#     ``llama-index`` backend).
#
# Authority constraints (pinned by tests):
#
#   * **ADVISORY only** (INV-12) — every output is a frozen value
#     object. No :class:`SignalEvent` / :class:`ExecutionIntent` /
#     :class:`PatchProposal` / :class:`GovernanceDecision`
#     constructors anywhere.
#   * **OFFLINE_ONLY** — writes happen on the offline (slow-loop)
#     side. Reads are bounded latency in-memory cosine lookups; the
#     reference fallback runs ``< 5 ms`` against the in-memory
#     transport for ≤1 000 documents.
#   * **INV-15** — pure dispatcher. No clock, no I/O, no PRNG. Three
#     independent runs with identical inputs produce byte-identical
#     :class:`RetrievedContext` / :class:`Prompt` instances.
#   * **B1** — no execution_engine / governance_engine /
#     system_engine / intelligence_engine / learning_engine /
#     evolution_engine submodule cross-imports.
#   * No top-level imports of :mod:`llama_index`, :mod:`openai`,
#     :mod:`anthropic`, :mod:`litellm`, :mod:`requests`, :mod:`httpx`,
#     :mod:`asyncio`, :mod:`time`, :mod:`datetime`, :mod:`random`,
#     :mod:`secrets`. The ``llama-index`` package is the lazy seam —
#     only :func:`enable_llamaindex_factory` may import from it, and
#     only inside the function body.
#   * LlamaIndex cloud (``llama-cloud``) / ``Settings.llm`` /
#     ``OpenAI`` / ``Anthropic`` surfaces are **not** re-exported.
#     The RAG pipeline ends at the :class:`Prompt` value object;
#     completion is the caller's job through LiteLLMRouter.
#
# NEW_PIP_DEPENDENCIES = ("llama-index",) — declared as the lazy seam
# for ``tools/cli.py install-c-tier``; production wiring routes
# everything through the in-memory cosine fallback unless an operator
# explicitly enables the live backend via
# :func:`enable_llamaindex_factory`.
"""C-16 long-term structured memory — LlamaIndex-shape vector store."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Mapping
from typing import Final, Protocol, runtime_checkable

# Drop-in alternative to C-15: reuse the canonical value-object
# surface so callers can swap the retriever implementation by changing
# a single import. The contracts in :mod:`state.knowledge_store` are
# pure value objects (no engine cross-imports, INV-15-safe), so this
# import is B1-clean.
from state.knowledge_store import (
    MAX_DOCUMENT_CONTENT_LEN,
    MAX_DOCUMENT_ID_LEN,
    MAX_METADATA_KEY_LEN,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LEN,
    MAX_QUERY_LEN,
    MAX_TOP_K,
    Document,
    DocumentError,
    KnowledgeStoreError,
    Prompt,
    PromptBuilder,
    PromptTemplate,
    Query,
    QueryError,
    RetrievedContext,
    RetrievedDocument,
    TemplateError,
    TransportError,
    tokenize,
)

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    # Re-exports for drop-in compatibility with C-15.
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
    "PromptBuilder",
    "tokenize",
    # C-16 originals.
    "VectorIndexTransport",
    "InMemoryVectorIndex",
    "CosineSimilarityRetriever",
    "LlamaIndexKnowledgeStore",
    "enable_llamaindex_factory",
    "embed_text",
    "EMBED_DIM",
    "MAX_DOCUMENT_ID_LEN",
    "MAX_DOCUMENT_CONTENT_LEN",
    "MAX_METADATA_KEYS",
    "MAX_METADATA_KEY_LEN",
    "MAX_METADATA_VALUE_LEN",
    "MAX_QUERY_LEN",
    "MAX_TOP_K",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("llama-index",)


# ---------------------------------------------------------------------------
# Embedding hyper-parameters (pinned for INV-15 byte-identical
# determinism; any change requires a manifest bump).
# ---------------------------------------------------------------------------


EMBED_DIM: Final[int] = 256

LLAMAINDEX_STORE_VERSION: Final[str] = "1"


# ---------------------------------------------------------------------------
# Deterministic embedder
# ---------------------------------------------------------------------------


def embed_text(text: str) -> tuple[float, ...]:
    """Deterministic hashing-trick term-frequency embedding.

    Maps ``text`` to a fixed-dimension non-negative real vector using
    BLAKE2b as the hash function and term-frequency as the per-bucket
    weight. The vector is L2-normalised so cosine similarity reduces
    to dot product.

    Properties:

    * Pure function — no random state, no clock, no I/O.
    * Tokeniser shared with C-15 (:func:`tokenize`) so the two
      adapters disagree only on scoring, not on token boundaries.
    * Empty input returns the zero vector (length ``EMBED_DIM``).
    * Same input always returns the same vector across processes /
      Python versions — pinned by the INV-15 three-run test.

    The hashing trick is the deterministic, vendor-free analogue of
    LlamaIndex's default ``BAAI/bge-small-en`` embedding for the
    purposes of the reference in-memory fallback. The live
    LlamaIndex backend (opted in via
    :func:`enable_llamaindex_factory`) substitutes a true neural
    embedder.
    """
    if not isinstance(text, str):
        raise QueryError(
            "embed_text: text must be str, got "
            f"{type(text).__name__}"
        )
    vec = [0.0] * EMBED_DIM
    for token in tokenize(text):
        # BLAKE2b(8 bytes) is more than enough to project tokens
        # uniformly into EMBED_DIM buckets. We take 4 bytes as a
        # big-endian uint32 and reduce modulo EMBED_DIM.
        digest = hashlib.blake2b(
            token.encode("utf-8"),
            digest_size=4,
        ).digest()
        bucket = int.from_bytes(digest, "big") % EMBED_DIM
        vec[bucket] += 1.0
    # L2-normalise.
    norm_sq = 0.0
    for x in vec:
        norm_sq += x * x
    if norm_sq <= 0.0:
        return tuple(vec)
    norm = math.sqrt(norm_sq)
    return tuple(x / norm for x in vec)


def _cosine_similarity(
    a: tuple[float, ...],
    b: tuple[float, ...],
) -> float:
    """Cosine similarity for L2-normalised vectors == dot product.

    Returns 0.0 if either vector is the zero vector. Inputs must
    share the same length (``EMBED_DIM``).
    """
    if len(a) != len(b):
        raise TransportError(
            "vector dimension mismatch: "
            f"{len(a)} vs {len(b)}"
        )
    total = 0.0
    for x, y in zip(a, b, strict=True):
        total += x * y
    if total < 0.0:
        # Clamp tiny negative drift from float arithmetic to 0.
        return 0.0
    return total


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorIndexTransport(Protocol):
    """Pluggable backend that holds the ``(Document, embedding)``
    rows.

    The reference fallback :class:`InMemoryVectorIndex` is dict-based
    with **lexicographic id ordering** in :meth:`iter_documents` so
    INV-15 three-run replays produce byte-identical retrievals
    regardless of insertion order.
    """

    def write(self, document: Document) -> None:
        ...

    def read(self, doc_id: str) -> Document | None:
        ...

    def iter_documents(self) -> Iterator[Document]:
        ...

    def iter_embeddings(
        self,
    ) -> Iterator[tuple[Document, tuple[float, ...]]]:
        ...

    def __len__(self) -> int:
        ...


class InMemoryVectorIndex:
    """Dict-based reference :class:`VectorIndexTransport`.

    Writes are O(tokens); reads are O(1); :meth:`iter_documents` and
    :meth:`iter_embeddings` yield rows in lexicographic id order so
    that downstream tie-breaking is deterministic.
    """

    __slots__ = ("_rows",)

    def __init__(self) -> None:
        # {id: (document, embedding)}
        self._rows: dict[str, tuple[Document, tuple[float, ...]]] = {}

    def write(self, document: Document) -> None:
        if not isinstance(document, Document):
            raise DocumentError(
                "InMemoryVectorIndex.write: expected Document, got "
                f"{type(document).__name__}"
            )
        embedding = embed_text(document.content)
        self._rows[document.id] = (document, embedding)

    def read(self, doc_id: str) -> Document | None:
        if not isinstance(doc_id, str) or not doc_id:
            raise DocumentError(
                "InMemoryVectorIndex.read: doc_id must be a "
                "non-empty str"
            )
        row = self._rows.get(doc_id)
        if row is None:
            return None
        return row[0]

    def iter_documents(self) -> Iterator[Document]:
        for doc_id in sorted(self._rows.keys()):
            yield self._rows[doc_id][0]

    def iter_embeddings(
        self,
    ) -> Iterator[tuple[Document, tuple[float, ...]]]:
        for doc_id in sorted(self._rows.keys()):
            doc, emb = self._rows[doc_id]
            yield doc, emb

    def __len__(self) -> int:
        return len(self._rows)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class CosineSimilarityRetriever:
    """LlamaIndex-shape vector retriever.

    Scores documents by cosine similarity against the query
    embedding. Filter-by-metadata uses strict equality on every
    declared key (same semantics as C-15 :class:`BM25Retriever`).
    Tie-breaking is lexicographic on doc id — pinned by
    ``test_cosine_tie_breaks_lexicographically_by_id``.
    """

    __slots__ = ("_transport",)

    def __init__(self, *, transport: VectorIndexTransport) -> None:
        if not isinstance(transport, VectorIndexTransport):
            raise TransportError(
                "CosineSimilarityRetriever: transport must "
                "implement VectorIndexTransport, got "
                f"{type(transport).__name__}"
            )
        self._transport = transport

    @property
    def transport(self) -> VectorIndexTransport:
        return self._transport

    def retrieve(self, query: Query) -> RetrievedContext:
        if not isinstance(query, Query):
            raise QueryError(
                "CosineSimilarityRetriever.retrieve: expected "
                "Query, got "
                f"{type(query).__name__}"
            )

        query_vec = embed_text(query.text)

        # Filter corpus by metadata equality, preserving lex order.
        corpus: list[Document] = []
        embeddings: list[tuple[float, ...]] = []
        for doc, emb in self._transport.iter_embeddings():
            if not self._matches_filter(doc, query.filter):
                continue
            corpus.append(doc)
            embeddings.append(emb)

        if not corpus:
            return RetrievedContext(
                query=query,
                documents=(),
                digest=_digest_context(query, ()),
            )

        scored: list[tuple[float, str, int]] = []
        for idx, doc in enumerate(corpus):
            score = _cosine_similarity(query_vec, embeddings[idx])
            if score > 0.0:
                scored.append((score, doc.id, idx))

        # Sort: highest score first; ties broken by lexicographic id.
        scored.sort(key=lambda row: (-row[0], row[1]))

        limit = min(query.top_k, len(scored))
        retrieved: list[RetrievedDocument] = []
        for rank, (score, _doc_id, idx) in enumerate(
            scored[:limit]
        ):
            retrieved.append(
                RetrievedDocument(
                    document=corpus[idx],
                    score=score,
                    rank=rank,
                )
            )

        docs_tuple = tuple(retrieved)
        return RetrievedContext(
            query=query,
            documents=docs_tuple,
            digest=_digest_context(query, docs_tuple),
        )

    @staticmethod
    def _matches_filter(
        doc: Document,
        filt: Mapping[str, str],
    ) -> bool:
        for k, v in filt.items():
            if doc.metadata.get(k) != v:
                return False
        return True


def _digest_context(
    query: Query,
    documents: tuple[RetrievedDocument, ...],
) -> bytes:
    """BLAKE2b-16 digest over a canonical byte projection of the
    retrieved-context tuple.

    The byte projection is intentionally hand-rolled rather than
    using :mod:`json` so that float formatting is fully deterministic
    (Python's ``repr(float)`` is stable across CPython versions on a
    given platform, and we round to a fixed 12-decimal precision to
    avoid platform drift). Pinned by INV-15 three-run test.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"v=")
    h.update(LLAMAINDEX_STORE_VERSION.encode("utf-8"))
    h.update(b"|q=")
    h.update(query.text.encode("utf-8"))
    h.update(b"|tk=")
    h.update(str(query.top_k).encode("utf-8"))
    h.update(b"|f=")
    for k in sorted(query.filter.keys()):
        h.update(k.encode("utf-8"))
        h.update(b"=")
        h.update(query.filter[k].encode("utf-8"))
        h.update(b";")
    h.update(b"|docs=")
    for rd in documents:
        h.update(rd.document.id.encode("utf-8"))
        h.update(b"#")
        h.update(rd.document.content.encode("utf-8"))
        h.update(b"#m=")
        for k in sorted(rd.document.metadata.keys()):
            h.update(k.encode("utf-8"))
            h.update(b"=")
            h.update(rd.document.metadata[k].encode("utf-8"))
            h.update(b";")
        h.update(b"#s=")
        h.update(format(rd.score, ".12f").encode("ascii"))
        h.update(b"#r=")
        h.update(str(rd.rank).encode("ascii"))
        h.update(b"|")
    return h.digest()


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class LlamaIndexKnowledgeStore:
    """C-16 facade — drop-in alternative to :class:`KnowledgeStore`.

    Provides the same three-method surface (``ingest`` / ``retrieve``
    / ``ask``) so a caller can swap the implementation by changing
    one import.

    Construct via :meth:`in_memory`; the live ``llama-index``
    backend is opted in through :func:`enable_llamaindex_factory`.
    """

    __slots__ = ("_transport", "_retriever", "_builder")

    def __init__(
        self,
        *,
        transport: VectorIndexTransport,
        retriever: CosineSimilarityRetriever,
        builder: PromptBuilder,
    ) -> None:
        if not isinstance(transport, VectorIndexTransport):
            raise TransportError(
                "LlamaIndexKnowledgeStore: transport must "
                "implement VectorIndexTransport"
            )
        if not isinstance(retriever, CosineSimilarityRetriever):
            raise KnowledgeStoreError(
                "LlamaIndexKnowledgeStore: retriever must be a "
                "CosineSimilarityRetriever"
            )
        if not isinstance(builder, PromptBuilder):
            raise KnowledgeStoreError(
                "LlamaIndexKnowledgeStore: builder must be a "
                "PromptBuilder"
            )
        self._transport = transport
        self._retriever = retriever
        self._builder = builder

    @classmethod
    def in_memory(
        cls, *, template: PromptTemplate
    ) -> LlamaIndexKnowledgeStore:
        if not isinstance(template, PromptTemplate):
            raise TemplateError(
                "LlamaIndexKnowledgeStore.in_memory: template must "
                "be a PromptTemplate"
            )
        transport = InMemoryVectorIndex()
        return cls(
            transport=transport,
            retriever=CosineSimilarityRetriever(transport=transport),
            builder=PromptBuilder(template=template),
        )

    def ingest(self, document: Document) -> None:
        if not isinstance(document, Document):
            raise DocumentError(
                "LlamaIndexKnowledgeStore.ingest: expected "
                "Document"
            )
        self._transport.write(document)

    def retrieve(self, query: Query) -> RetrievedContext:
        return self._retriever.retrieve(query)

    def ask(self, query: Query) -> Prompt:
        ctx = self.retrieve(query)
        return self._builder.build(context=ctx)

    def __len__(self) -> int:
        return len(self._transport)


# ---------------------------------------------------------------------------
# Lazy seam — live LlamaIndex backend
# ---------------------------------------------------------------------------


def enable_llamaindex_factory() -> None:
    """Opt in to the live ``llama-index`` backend.

    Until activated, the in-memory cosine fallback is the only
    supported transport — this keeps the production path B1-clean
    and INV-15 reproducible without the vendor dependency.

    The live backend wires ``llama_index.core.indices.\
VectorStoreIndex`` + ``llama_index.core.query_engine\
.RetrieverQueryEngine`` into a transport that exposes the same
    :class:`VectorIndexTransport` Protocol. All LLM calls inside the
    live engine MUST be routed through
    :class:`~intelligence_engine.cognitive.litellm_router.\
LiteLLMRouter` — direct LlamaIndex LLM SDK calls are forbidden by
    DIX policy.
    """
    raise NotImplementedError(
        "enable_llamaindex_factory: live llama-index backend not "
        "yet activated — use LlamaIndexKnowledgeStore.in_memory() "
        "for the deterministic in-memory fallback"
    )



