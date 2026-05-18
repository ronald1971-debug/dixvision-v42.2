"""C-16 llama-index test suite — state/knowledge_store_llamaindex.py.

Covers:

* Module invariants (``NEW_PIP_DEPENDENCIES``, ``EMBED_DIM``,
  ``__all__``).
* AST guards: top-level import cleanliness (no llama_index /
  openai / anthropic / litellm / requests / httpx / asyncio /
  time / datetime / random / secrets); no wall-clock / PRNG
  calls; no typed bus event constructors (B27/B28/INV-71); no
  cross-engine imports (only :mod:`state.knowledge_store` is
  allowed since that module is itself B1-clean).
* :func:`embed_text` determinism, dimension, normalisation, empty
  input, type errors.
* :class:`InMemoryVectorIndex` ordering + length + write
  validation.
* :class:`CosineSimilarityRetriever` happy path: ranking by
  semantic similarity, metadata filter, ``top_k`` clamp, empty
  corpus, lex tie-break, transport validation.
* :class:`LlamaIndexKnowledgeStore` composition + drop-in
  compatibility with the C-15 :class:`PromptBuilder` /
  :class:`PromptTemplate`.
* INV-15 byte-identical determinism (BLAKE2b-16 digests over
  three independent runs).
* :class:`VectorIndexTransport` Protocol runtime-checkability.
* :func:`enable_llamaindex_factory` raises ``NotImplementedError``
  (lazy seam not activated by default).
"""

from __future__ import annotations

import ast
import hashlib
import math
import pathlib

import pytest

from state import knowledge_store_llamaindex as li_mod
from state.knowledge_store_llamaindex import (
    EMBED_DIM,
    NEW_PIP_DEPENDENCIES,
    CosineSimilarityRetriever,
    Document,
    DocumentError,
    InMemoryVectorIndex,
    KnowledgeStoreError,
    LlamaIndexKnowledgeStore,
    Prompt,
    PromptBuilder,
    PromptTemplate,
    Query,
    QueryError,
    TemplateError,
    TransportError,
    VectorIndexTransport,
    embed_text,
    enable_llamaindex_factory,
    tokenize,
)

MODULE_PATH = pathlib.Path(li_mod.__file__)


# ---------------------------------------------------------------------------
# Module invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_pinned() -> None:
    assert NEW_PIP_DEPENDENCIES == ("llama-index",)


def test_embed_dim_pinned() -> None:
    assert EMBED_DIM == 256


def test_all_exports_complete() -> None:
    expected = {
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
        "PromptBuilder",
        "tokenize",
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
    }
    assert set(li_mod.__all__) == expected


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_IMPORTS = frozenset(
    (
        "llama_index",
        "openai",
        "anthropic",
        "litellm",
        "requests",
        "httpx",
        "asyncio",
        "time",
        "datetime",
        "random",
        "secrets",
    )
)

# C-16 is a drop-in alternative to C-15 and *is allowed* to import
# value-object contracts from :mod:`state.knowledge_store` (that
# module is itself B1-clean). All other engine submodules remain
# forbidden.
_FORBIDDEN_CROSS_ENGINE_PREFIXES = (
    "execution_engine",
    "governance_engine",
    "system_engine",
    "intelligence_engine",
    "learning_engine",
    "evolution_engine",
    "core.contracts.events",
)


def _module_ast() -> ast.AST:
    return ast.parse(MODULE_PATH.read_text())


def test_no_forbidden_top_level_imports() -> None:
    tree = _module_ast()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                head = alias.name.split(".")[0]
                assert head not in _FORBIDDEN_TOP_IMPORTS, (
                    f"forbidden top-level import: {alias.name}"
                )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            head = mod.split(".")[0]
            assert head not in _FORBIDDEN_TOP_IMPORTS, f"forbidden top-level from-import: {mod}"
            for prefix in _FORBIDDEN_CROSS_ENGINE_PREFIXES:
                assert not (mod == prefix or mod.startswith(prefix + ".")), (
                    f"forbidden cross-engine import: {mod}"
                )


def test_no_wall_clock_or_prng_calls() -> None:
    tree = _module_ast()
    forbidden_attrs = {
        ("time", "time"),
        ("time", "time_ns"),
        ("time", "monotonic"),
        ("time", "monotonic_ns"),
        ("time", "perf_counter"),
        ("time", "perf_counter_ns"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("random", "random"),
        ("random", "randint"),
        ("random", "choice"),
        ("random", "shuffle"),
        ("secrets", "token_bytes"),
        ("secrets", "token_hex"),
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            assert (func.value.id, func.attr) not in forbidden_attrs


def test_no_typed_bus_event_constructors() -> None:
    tree = _module_ast()
    forbidden_names = {
        "SignalEvent",
        "ExecutionIntent",
        "ExecutionResult",
        "PatchProposal",
        "GovernanceDecision",
        "TradeOutcome",
        "HazardEvent",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_names, (
                f"forbidden bus-event constructor: {node.func.id}"
            )


def test_enable_llamaindex_factory_is_lazy_seam() -> None:
    with pytest.raises(NotImplementedError):
        enable_llamaindex_factory()


# ---------------------------------------------------------------------------
# embed_text
# ---------------------------------------------------------------------------


def test_embed_text_returns_fixed_dim_vector() -> None:
    v = embed_text("hello world")
    assert isinstance(v, tuple)
    assert len(v) == EMBED_DIM
    for x in v:
        assert isinstance(x, float)


def test_embed_text_is_deterministic_across_calls() -> None:
    a = embed_text("position size limits")
    b = embed_text("position size limits")
    assert a == b


def test_embed_text_l2_normalised_for_nonzero_input() -> None:
    v = embed_text("alpha bravo charlie")
    norm_sq = sum(x * x for x in v)
    # Float arithmetic — allow tiny drift around 1.0.
    assert math.isclose(norm_sq, 1.0, rel_tol=1e-9, abs_tol=1e-9)


def test_embed_text_empty_string_returns_zero_vector() -> None:
    v = embed_text("")
    assert v == (0.0,) * EMBED_DIM


def test_embed_text_no_tokens_returns_zero_vector() -> None:
    # Punctuation-only input tokenises to ().
    v = embed_text("!!! ... ???")
    assert v == (0.0,) * EMBED_DIM


def test_embed_text_non_str_raises() -> None:
    with pytest.raises(QueryError):
        embed_text(42)  # type: ignore[arg-type]


def test_embed_text_distinct_inputs_distinct_vectors() -> None:
    a = embed_text("alpha bravo")
    b = embed_text("charlie delta")
    assert a != b


def test_embed_text_uses_shared_tokeniser() -> None:
    # Same tokens (different surface formatting) must embed to the
    # same vector — confirms we share :func:`tokenize` with C-15.
    assert tokenize("Alpha, Bravo!") == ("alpha", "bravo")
    a = embed_text("Alpha, Bravo!")
    b = embed_text("alpha   bravo")
    assert a == b


# ---------------------------------------------------------------------------
# InMemoryVectorIndex
# ---------------------------------------------------------------------------


def test_in_memory_index_protocol_runtime_checkable() -> None:
    transport = InMemoryVectorIndex()
    assert isinstance(transport, VectorIndexTransport)


def test_in_memory_index_iter_documents_is_lexicographic() -> None:
    t = InMemoryVectorIndex()
    t.write(Document(id="zeta", content="z"))
    t.write(Document(id="alpha", content="a"))
    t.write(Document(id="mu", content="m"))
    ids = [d.id for d in t.iter_documents()]
    assert ids == ["alpha", "mu", "zeta"]


def test_in_memory_index_iter_embeddings_is_lexicographic() -> None:
    t = InMemoryVectorIndex()
    t.write(Document(id="zeta", content="alpha"))
    t.write(Document(id="alpha", content="bravo"))
    rows = list(t.iter_embeddings())
    ids = [doc.id for doc, _emb in rows]
    assert ids == ["alpha", "zeta"]
    for _doc, emb in rows:
        assert isinstance(emb, tuple)
        assert len(emb) == EMBED_DIM


def test_in_memory_index_overwrites_same_id() -> None:
    t = InMemoryVectorIndex()
    t.write(Document(id="a", content="first"))
    t.write(Document(id="a", content="second"))
    assert len(t) == 1
    doc = t.read("a")
    assert doc is not None
    assert doc.content == "second"


def test_in_memory_index_read_missing_returns_none() -> None:
    t = InMemoryVectorIndex()
    assert t.read("missing") is None


def test_in_memory_index_write_non_document_raises() -> None:
    t = InMemoryVectorIndex()
    with pytest.raises(DocumentError):
        t.write("not a doc")  # type: ignore[arg-type]


def test_in_memory_index_read_empty_id_raises() -> None:
    t = InMemoryVectorIndex()
    with pytest.raises(DocumentError):
        t.read("")


# ---------------------------------------------------------------------------
# CosineSimilarityRetriever
# ---------------------------------------------------------------------------


def _build_store(docs: list[Document]) -> LlamaIndexKnowledgeStore:
    store = LlamaIndexKnowledgeStore.in_memory(
        template=PromptTemplate(body="C={context}\nQ={query}"),
    )
    for d in docs:
        store.ingest(d)
    return store


def test_cosine_known_document_ranks_first() -> None:
    docs = [
        Document(
            id="r.size",
            content=("governance rule about position size limits and exposure caps"),
        ),
        Document(
            id="r.kill",
            content="governance rule for kill switch triggers",
        ),
        Document(
            id="r.spec",
            content="spec text for ledger replay timestamps",
        ),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="position size", top_k=3))
    assert len(ctx.documents) >= 1
    assert ctx.documents[0].document.id == "r.size"


def test_cosine_metadata_filter_excludes_non_matches() -> None:
    docs = [
        Document(
            id="r.a",
            content="rule about kill switch",
            metadata={"tier": "governance"},
        ),
        Document(
            id="r.b",
            content="rule about kill switch",
            metadata={"tier": "execution"},
        ),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(
        Query(
            text="kill switch",
            filter={"tier": "governance"},
        )
    )
    assert {d.document.id for d in ctx.documents} == {"r.a"}


def test_cosine_top_k_caps_result_count() -> None:
    docs = [Document(id=f"r.{i}", content="rule alpha bravo") for i in range(8)]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="rule alpha", top_k=3))
    assert len(ctx.documents) <= 3


def test_cosine_empty_corpus_returns_empty_context() -> None:
    store = LlamaIndexKnowledgeStore.in_memory(
        template=PromptTemplate(body="{context}/{query}"),
    )
    ctx = store.retrieve(Query(text="anything"))
    assert ctx.documents == ()
    assert len(ctx.digest) == 16


def test_cosine_no_token_overlap_returns_empty() -> None:
    docs = [
        Document(id="r.a", content="alpha bravo charlie"),
        Document(id="r.b", content="delta echo foxtrot"),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="zulu yankee"))
    assert ctx.documents == ()


def test_cosine_ranks_are_zero_indexed_and_dense() -> None:
    docs = [
        Document(
            id="r.a",
            content="alpha alpha bravo charlie",
        ),
        Document(id="r.b", content="alpha bravo"),
        Document(id="r.c", content="bravo charlie"),
        Document(id="r.d", content="delta"),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="alpha bravo", top_k=4))
    for i, rd in enumerate(ctx.documents):
        assert rd.rank == i


def test_cosine_tie_breaks_lexicographically_by_id() -> None:
    # Three documents with identical content -> identical cosine
    # scores. Tie-break must be lexicographic on id (ascending).
    docs = [
        Document(id="r.zeta", content="alpha bravo"),
        Document(id="r.alpha", content="alpha bravo"),
        Document(id="r.mu", content="alpha bravo"),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="alpha", top_k=3))
    ids = [d.document.id for d in ctx.documents]
    assert ids == ["r.alpha", "r.mu", "r.zeta"]


def test_cosine_scores_are_non_negative() -> None:
    docs = [
        Document(id="r.a", content="alpha bravo"),
        Document(id="r.b", content="delta echo"),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="alpha"))
    for rd in ctx.documents:
        assert rd.score >= 0.0


def test_cosine_bad_query_type_raises() -> None:
    store = _build_store([Document(id="r.a", content="x")])
    with pytest.raises(QueryError):
        store.retrieve("not a query")  # type: ignore[arg-type]


def test_cosine_bad_transport_raises() -> None:
    with pytest.raises(TransportError):
        CosineSimilarityRetriever(
            transport="not a transport"  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# LlamaIndexKnowledgeStore composition
# ---------------------------------------------------------------------------


def test_store_len_tracks_transport() -> None:
    store = LlamaIndexKnowledgeStore.in_memory(
        template=PromptTemplate(body="{context}\n{query}"),
    )
    assert len(store) == 0
    store.ingest(Document(id="r.a", content="x"))
    store.ingest(Document(id="r.b", content="y"))
    assert len(store) == 2
    store.ingest(Document(id="r.a", content="x'"))
    assert len(store) == 2


def test_store_ask_returns_prompt_with_context_and_query() -> None:
    docs = [
        Document(id="r.a", content="alpha bravo charlie"),
        Document(id="r.b", content="delta echo"),
    ]
    store = _build_store(docs)
    prompt = store.ask(Query(text="alpha", top_k=2))
    assert isinstance(prompt, Prompt)
    assert "alpha bravo charlie" in prompt.rendered
    assert prompt.rendered.endswith("Q=alpha")


def test_store_ask_empty_context_renders_query_only() -> None:
    store = LlamaIndexKnowledgeStore.in_memory(
        template=PromptTemplate(body="C={context}\nQ={query}"),
    )
    prompt = store.ask(Query(text="zulu"))
    assert prompt.rendered == "C=\nQ=zulu"


def test_store_in_memory_bad_template_raises() -> None:
    with pytest.raises(TemplateError):
        LlamaIndexKnowledgeStore.in_memory(
            template="not a template"  # type: ignore[arg-type]
        )


def test_store_bad_transport_in_ctor_raises() -> None:
    with pytest.raises(TransportError):
        LlamaIndexKnowledgeStore(
            transport="not a transport",  # type: ignore[arg-type]
            retriever=CosineSimilarityRetriever(
                transport=InMemoryVectorIndex(),
            ),
            builder=PromptBuilder(
                template=PromptTemplate(body="{context}"),
            ),
        )


def test_store_bad_retriever_in_ctor_raises() -> None:
    transport = InMemoryVectorIndex()
    with pytest.raises(KnowledgeStoreError):
        LlamaIndexKnowledgeStore(
            transport=transport,
            retriever="not a retriever",  # type: ignore[arg-type]
            builder=PromptBuilder(
                template=PromptTemplate(body="{context}"),
            ),
        )


def test_store_bad_builder_in_ctor_raises() -> None:
    transport = InMemoryVectorIndex()
    with pytest.raises(KnowledgeStoreError):
        LlamaIndexKnowledgeStore(
            transport=transport,
            retriever=CosineSimilarityRetriever(
                transport=transport,
            ),
            builder="not a builder",  # type: ignore[arg-type]
        )


def test_store_drop_in_compatible_with_c15_prompt_builder() -> None:
    # The PromptBuilder and PromptTemplate types come from C-15;
    # this test pins that we can mix the two adapters' surfaces.
    from state.knowledge_store import (
        PromptBuilder as Hb,
    )
    from state.knowledge_store import (
        PromptTemplate as Ht,
    )

    assert PromptBuilder is Hb
    assert PromptTemplate is Ht


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def _run_pipeline() -> tuple[bytes, str]:
    docs = [
        Document(
            id="r.alpha",
            content="alpha bravo charlie",
            metadata={"tier": "governance"},
        ),
        Document(
            id="r.bravo",
            content="bravo delta echo",
            metadata={"tier": "execution"},
        ),
        Document(
            id="r.charlie",
            content="charlie foxtrot golf",
            metadata={"tier": "governance"},
        ),
    ]
    store = LlamaIndexKnowledgeStore.in_memory(
        template=PromptTemplate(body="C={context}\nQ={query}"),
    )
    for d in docs:
        store.ingest(d)
    prompt = store.ask(Query(text="alpha bravo", top_k=3))
    rendered_digest = hashlib.blake2b(
        prompt.rendered.encode("utf-8"),
        digest_size=16,
    ).digest()
    return (prompt.context.digest, rendered_digest.hex())


def test_inv15_three_run_byte_identical() -> None:
    a = _run_pipeline()
    b = _run_pipeline()
    c = _run_pipeline()
    assert a == b == c


def test_inv15_different_query_gives_different_digest() -> None:
    docs = [
        Document(id="r.a", content="alpha bravo"),
        Document(id="r.b", content="charlie delta"),
    ]
    store = _build_store(docs)
    ctx_alpha = store.retrieve(Query(text="alpha"))
    ctx_charlie = store.retrieve(Query(text="charlie"))
    assert ctx_alpha.digest != ctx_charlie.digest


def test_inv15_insertion_order_does_not_affect_digest() -> None:
    docs_a = [
        Document(id="r.alpha", content="alpha bravo"),
        Document(id="r.bravo", content="charlie delta"),
        Document(id="r.charlie", content="echo foxtrot"),
    ]
    docs_b = list(reversed(docs_a))
    store_a = _build_store(docs_a)
    store_b = _build_store(docs_b)
    q = Query(text="alpha bravo", top_k=3)
    assert store_a.retrieve(q).digest == store_b.retrieve(q).digest


# ---------------------------------------------------------------------------
# Protocol runtime-checkability
# ---------------------------------------------------------------------------


def test_vector_index_transport_protocol_runtime_checkable() -> None:
    class StubTransport:
        def write(self, document: Document) -> None:
            return None

        def read(self, doc_id: str) -> Document | None:
            return None

        def iter_documents(self):  # type: ignore[no-untyped-def]
            return iter(())

        def iter_embeddings(self):  # type: ignore[no-untyped-def]
            return iter(())

        def __len__(self) -> int:
            return 0

    assert isinstance(StubTransport(), VectorIndexTransport)


def test_protocol_rejects_incomplete_implementation() -> None:
    class Broken:
        # Missing iter_embeddings / __len__.
        def write(self, document: Document) -> None:
            return None

        def read(self, doc_id: str) -> Document | None:
            return None

        def iter_documents(self):  # type: ignore[no-untyped-def]
            return iter(())

    assert not isinstance(Broken(), VectorIndexTransport)
