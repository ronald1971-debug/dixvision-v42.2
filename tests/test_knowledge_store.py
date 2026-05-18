"""C-15 haystack-ai test suite — state/knowledge_store.py.

Covers:

* Module invariants (``NEW_PIP_DEPENDENCIES``, error hierarchy,
  ``__all__``).
* AST guards: top-level import cleanliness (no haystack / openai /
  anthropic / litellm / requests / httpx / asyncio / time /
  datetime / random / secrets); no wall-clock / PRNG calls; no
  typed bus event constructors (B27/B28/INV-71); no cross-engine
  imports.
* :class:`Document`, :class:`Query`, :class:`RetrievedDocument`,
  :class:`RetrievedContext`, :class:`PromptTemplate`,
  :class:`Prompt` validation.
* :class:`tokenize` and :func:`_scan_placeholders` behaviour.
* :class:`InMemoryDocumentStore` ordering + length.
* :class:`BM25Retriever` happy path: known-document ranking,
  metadata filtering, ``top_k`` clamp, empty corpus, tie-breaking
  by lexicographic id, IDF zero/negative branch.
* :class:`PromptBuilder` rendering.
* :class:`KnowledgeStore` composition + INV-15 byte-identical
  determinism (BLAKE2b-16 digests over three independent runs).
* :class:`DocumentStoreTransport` Protocol runtime-checkability.
* :func:`enable_haystack_factory` raises ``NotImplementedError``
  (lazy seam not activated by default).
"""

from __future__ import annotations

import ast
import hashlib
import pathlib

import pytest

from state import knowledge_store as ks_mod
from state.knowledge_store import (
    BM25_B,
    BM25_K1,
    MAX_DOCUMENT_CONTENT_LEN,
    MAX_DOCUMENT_ID_LEN,
    MAX_METADATA_KEY_LEN,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_LEN,
    MAX_QUERY_LEN,
    MAX_TOP_K,
    NEW_PIP_DEPENDENCIES,
    BM25Retriever,
    Document,
    DocumentError,
    DocumentStoreTransport,
    InMemoryDocumentStore,
    KnowledgeStore,
    KnowledgeStoreError,
    PromptBuilder,
    PromptTemplate,
    Query,
    QueryError,
    RetrievedContext,
    RetrievedDocument,
    TemplateError,
    TransportError,
    enable_haystack_factory,
    tokenize,
)

KNOWLEDGE_STORE_PATH = pathlib.Path(ks_mod.__file__)


# ---------------------------------------------------------------------------
# Module invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_pinned() -> None:
    assert NEW_PIP_DEPENDENCIES == ("haystack-ai",)


def test_bm25_hyperparameters_pinned() -> None:
    assert BM25_K1 == 1.2
    assert BM25_B == 0.75


def test_error_hierarchy() -> None:
    assert issubclass(KnowledgeStoreError, ValueError)
    assert issubclass(DocumentError, KnowledgeStoreError)
    assert issubclass(QueryError, KnowledgeStoreError)
    assert issubclass(TemplateError, KnowledgeStoreError)
    assert issubclass(TransportError, KnowledgeStoreError)


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
    }
    assert set(ks_mod.__all__) == expected


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


_FORBIDDEN_TOP_IMPORTS = frozenset(
    (
        "haystack",
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
    return ast.parse(KNOWLEDGE_STORE_PATH.read_text())


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


def test_enable_haystack_factory_is_lazy_seam() -> None:
    # The only module-level reference to ``haystack`` must be a
    # function-local import inside enable_haystack_factory. Until
    # the seam is activated, the function raises NotImplementedError.
    with pytest.raises(NotImplementedError):
        enable_haystack_factory()


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------


def test_tokenize_basic_ascii() -> None:
    assert tokenize("Hello, World 2024!") == (
        "hello",
        "world",
        "2024",
    )


def test_tokenize_empty_returns_empty() -> None:
    assert tokenize("") == ()


def test_tokenize_non_str_raises() -> None:
    with pytest.raises(QueryError):
        tokenize(123)  # type: ignore[arg-type]


def test_tokenize_preserves_order() -> None:
    assert tokenize("a b c a b") == ("a", "b", "c", "a", "b")


# ---------------------------------------------------------------------------
# Document validation
# ---------------------------------------------------------------------------


def test_document_happy_path() -> None:
    d = Document(
        id="rule.a",
        content="hello",
        metadata={"tier": "governance"},
    )
    assert d.id == "rule.a"
    assert d.content == "hello"
    assert dict(d.metadata) == {"tier": "governance"}


def test_document_empty_id_raises() -> None:
    with pytest.raises(DocumentError):
        Document(id="", content="x")


def test_document_id_bad_chars_raises() -> None:
    with pytest.raises(DocumentError):
        Document(id="bad id with space", content="x")


def test_document_id_too_long_raises() -> None:
    with pytest.raises(DocumentError):
        Document(id="a" * (MAX_DOCUMENT_ID_LEN + 1), content="x")


def test_document_non_str_content_raises() -> None:
    with pytest.raises(DocumentError):
        Document(id="ok", content=42)  # type: ignore[arg-type]


def test_document_content_too_long_raises() -> None:
    with pytest.raises(DocumentError):
        Document(
            id="ok",
            content="x" * (MAX_DOCUMENT_CONTENT_LEN + 1),
        )


def test_document_metadata_non_mapping_raises() -> None:
    with pytest.raises(DocumentError):
        Document(
            id="ok",
            content="x",
            metadata=[("k", "v")],  # type: ignore[arg-type]
        )


def test_document_metadata_too_many_keys_raises() -> None:
    with pytest.raises(DocumentError):
        Document(
            id="ok",
            content="x",
            metadata={f"k{i}": "v" for i in range(MAX_METADATA_KEYS + 1)},
        )


def test_document_metadata_empty_key_raises() -> None:
    with pytest.raises(DocumentError):
        Document(id="ok", content="x", metadata={"": "v"})


def test_document_metadata_bad_key_chars_raises() -> None:
    with pytest.raises(DocumentError):
        Document(id="ok", content="x", metadata={"bad key": "v"})


def test_document_metadata_long_key_raises() -> None:
    with pytest.raises(DocumentError):
        Document(
            id="ok",
            content="x",
            metadata={"k" * (MAX_METADATA_KEY_LEN + 1): "v"},
        )


def test_document_metadata_non_str_value_raises() -> None:
    with pytest.raises(DocumentError):
        Document(
            id="ok",
            content="x",
            metadata={"k": 42},  # type: ignore[dict-item]
        )


def test_document_metadata_long_value_raises() -> None:
    with pytest.raises(DocumentError):
        Document(
            id="ok",
            content="x",
            metadata={"k": "v" * (MAX_METADATA_VALUE_LEN + 1)},
        )


def test_document_metadata_keys_are_sorted_for_byte_stable_replay() -> None:
    d = Document(
        id="ok",
        content="x",
        metadata={"z": "1", "a": "2", "m": "3"},
    )
    assert list(d.metadata.keys()) == ["a", "m", "z"]


# ---------------------------------------------------------------------------
# Query validation
# ---------------------------------------------------------------------------


def test_query_happy_path() -> None:
    q = Query(text="position size", top_k=3)
    assert q.text == "position size"
    assert q.top_k == 3
    assert dict(q.filter) == {}


def test_query_empty_text_raises() -> None:
    with pytest.raises(QueryError):
        Query(text="   ")


def test_query_text_too_long_raises() -> None:
    with pytest.raises(QueryError):
        Query(text="x" * (MAX_QUERY_LEN + 1))


def test_query_top_k_lower_bound_raises() -> None:
    with pytest.raises(QueryError):
        Query(text="x", top_k=0)


def test_query_top_k_upper_bound_raises() -> None:
    with pytest.raises(QueryError):
        Query(text="x", top_k=MAX_TOP_K + 1)


def test_query_top_k_non_int_raises() -> None:
    with pytest.raises(QueryError):
        Query(text="x", top_k=True)  # type: ignore[arg-type]


def test_query_filter_non_mapping_raises() -> None:
    with pytest.raises(QueryError):
        Query(
            text="x",
            filter=[("k", "v")],  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# RetrievedDocument validation
# ---------------------------------------------------------------------------


def _doc(id_: str = "d", content: str = "x") -> Document:
    return Document(id=id_, content=content)


def test_retrieved_document_happy_path() -> None:
    rd = RetrievedDocument(document=_doc(), score=1.5, rank=0)
    assert rd.score == 1.5
    assert rd.rank == 0


def test_retrieved_document_bad_document_raises() -> None:
    with pytest.raises(KnowledgeStoreError):
        RetrievedDocument(
            document="not a doc",  # type: ignore[arg-type]
            score=1.0,
            rank=0,
        )


def test_retrieved_document_negative_score_raises() -> None:
    with pytest.raises(KnowledgeStoreError):
        RetrievedDocument(document=_doc(), score=-0.1, rank=0)


def test_retrieved_document_nan_score_raises() -> None:
    with pytest.raises(KnowledgeStoreError):
        RetrievedDocument(
            document=_doc(),
            score=float("nan"),
            rank=0,
        )


def test_retrieved_document_negative_rank_raises() -> None:
    with pytest.raises(KnowledgeStoreError):
        RetrievedDocument(document=_doc(), score=1.0, rank=-1)


# ---------------------------------------------------------------------------
# RetrievedContext validation
# ---------------------------------------------------------------------------


def test_retrieved_context_rank_must_match_index() -> None:
    q = Query(text="x")
    bad = RetrievedDocument(document=_doc(), score=1.0, rank=1)
    with pytest.raises(KnowledgeStoreError):
        RetrievedContext(
            query=q,
            documents=(bad,),
            digest=b"\x00" * 16,
        )


def test_retrieved_context_bad_digest_len_raises() -> None:
    q = Query(text="x")
    with pytest.raises(KnowledgeStoreError):
        RetrievedContext(
            query=q,
            documents=(),
            digest=b"\x00" * 8,
        )


def test_retrieved_context_documents_must_be_tuple() -> None:
    q = Query(text="x")
    with pytest.raises(KnowledgeStoreError):
        RetrievedContext(
            query=q,
            documents=[],  # type: ignore[arg-type]
            digest=b"\x00" * 16,
        )


# ---------------------------------------------------------------------------
# PromptTemplate validation
# ---------------------------------------------------------------------------


def test_prompt_template_happy_path() -> None:
    t = PromptTemplate(body="ctx={context}\nq={query}")
    assert "context" in t.body


def test_prompt_template_empty_body_raises() -> None:
    with pytest.raises(TemplateError):
        PromptTemplate(body="   ")


def test_prompt_template_unsupported_placeholder_raises() -> None:
    with pytest.raises(TemplateError):
        PromptTemplate(body="ctx={context}\nbad={extra}")


def test_prompt_template_unbalanced_brace_raises() -> None:
    with pytest.raises(TemplateError):
        PromptTemplate(body="ctx={context")


def test_prompt_template_escaped_braces_ok() -> None:
    t = PromptTemplate(body="literal {{x}} and {context}")
    assert "{{" in t.body


# ---------------------------------------------------------------------------
# InMemoryDocumentStore
# ---------------------------------------------------------------------------


def test_in_memory_store_protocol_runtime_checkable() -> None:
    store = InMemoryDocumentStore()
    assert isinstance(store, DocumentStoreTransport)


def test_in_memory_store_iteration_is_lexicographic() -> None:
    store = InMemoryDocumentStore()
    store.write(Document(id="zeta", content="z"))
    store.write(Document(id="alpha", content="a"))
    store.write(Document(id="mu", content="m"))
    ids = [d.id for d in store.iter_documents()]
    assert ids == ["alpha", "mu", "zeta"]


def test_in_memory_store_overwrites_same_id() -> None:
    store = InMemoryDocumentStore()
    store.write(Document(id="a", content="first"))
    store.write(Document(id="a", content="second"))
    assert len(store) == 1
    assert store.read("a") is not None
    assert store.read("a").content == "second"  # type: ignore[union-attr]


def test_in_memory_store_read_missing_returns_none() -> None:
    store = InMemoryDocumentStore()
    assert store.read("missing") is None


def test_in_memory_store_write_non_document_raises() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(DocumentError):
        store.write("not a doc")  # type: ignore[arg-type]


def test_in_memory_store_read_empty_id_raises() -> None:
    store = InMemoryDocumentStore()
    with pytest.raises(DocumentError):
        store.read("")


# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


def _build_store(docs: list[Document]) -> KnowledgeStore:
    store = KnowledgeStore.in_memory(
        template=PromptTemplate(body="C={context}\nQ={query}"),
    )
    for d in docs:
        store.ingest(d)
    return store


def test_bm25_known_document_ranks_first() -> None:
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
        Document(
            id="r.misc",
            content="dashboard widget for operator approvals",
        ),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="position size", top_k=3))
    assert len(ctx.documents) >= 1
    assert ctx.documents[0].document.id == "r.size"


def test_bm25_metadata_filter_excludes_non_matches() -> None:
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


def test_bm25_top_k_caps_result_count() -> None:
    docs = [Document(id=f"r.{i}", content="rule") for i in range(8)]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="rule", top_k=3))
    assert len(ctx.documents) <= 3


def test_bm25_empty_corpus_returns_empty_context() -> None:
    store = KnowledgeStore.in_memory(
        template=PromptTemplate(body="{context}/{query}"),
    )
    ctx = store.retrieve(Query(text="anything"))
    assert ctx.documents == ()
    assert len(ctx.digest) == 16


def test_bm25_no_match_returns_empty_context() -> None:
    docs = [
        Document(id="r.a", content="alpha bravo charlie"),
        Document(id="r.b", content="delta echo foxtrot"),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="zulu"))
    assert ctx.documents == ()


def test_bm25_ranks_are_zero_indexed_and_dense() -> None:
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


def test_bm25_tie_breaks_lexicographically_by_id() -> None:
    # Three documents with identical content -> identical BM25
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


def test_bm25_zero_idf_branch_skipped() -> None:
    # When a query term appears in every document, IDF computes to
    # 0 (or negative for very small corpora) and that term should
    # contribute nothing — sanity-check that retrieval still
    # returns documents based on the *other* query tokens.
    docs = [
        Document(id="r.a", content="common alpha"),
        Document(id="r.b", content="common bravo"),
        Document(id="r.c", content="common alpha bravo"),
    ]
    store = _build_store(docs)
    ctx = store.retrieve(Query(text="common alpha", top_k=3))
    ids = {d.document.id for d in ctx.documents}
    assert "r.a" in ids
    assert "r.c" in ids


def test_bm25_bad_query_type_raises() -> None:
    store = _build_store([Document(id="r.a", content="x")])
    with pytest.raises(QueryError):
        store.retrieve("not a query")  # type: ignore[arg-type]


def test_bm25_bad_transport_raises() -> None:
    with pytest.raises(TransportError):
        BM25Retriever(transport="not a transport")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# PromptBuilder + Prompt
# ---------------------------------------------------------------------------


def test_prompt_builder_renders_context_and_query() -> None:
    docs = [
        Document(id="r.a", content="alpha bravo charlie"),
        Document(id="r.b", content="delta echo"),
    ]
    store = _build_store(docs)
    prompt = store.ask(Query(text="alpha", top_k=2))
    assert "alpha bravo charlie" in prompt.rendered
    assert prompt.rendered.endswith("Q=alpha")


def test_prompt_builder_empty_context_renders_query_only() -> None:
    store = KnowledgeStore.in_memory(
        template=PromptTemplate(body="C={context}\nQ={query}"),
    )
    prompt = store.ask(Query(text="zulu"))
    assert prompt.rendered == "C=\nQ=zulu"


def test_prompt_builder_bad_context_raises() -> None:
    builder = PromptBuilder(
        template=PromptTemplate(body="{context}/{query}"),
    )
    with pytest.raises(TemplateError):
        builder.build(context="not a context")  # type: ignore[arg-type]


def test_prompt_builder_bad_template_raises() -> None:
    with pytest.raises(TemplateError):
        PromptBuilder(template="not a template")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# KnowledgeStore composition
# ---------------------------------------------------------------------------


def test_knowledge_store_len_tracks_transport() -> None:
    store = KnowledgeStore.in_memory(
        template=PromptTemplate(body="{context}\n{query}"),
    )
    assert len(store) == 0
    store.ingest(Document(id="r.a", content="x"))
    store.ingest(Document(id="r.b", content="y"))
    assert len(store) == 2
    # Overwrite must not double-count.
    store.ingest(Document(id="r.a", content="x'"))
    assert len(store) == 2


def test_knowledge_store_bad_transport_in_ctor_raises() -> None:
    with pytest.raises(TransportError):
        KnowledgeStore(
            transport="not a transport",  # type: ignore[arg-type]
            retriever=BM25Retriever(transport=InMemoryDocumentStore()),
            builder=PromptBuilder(
                template=PromptTemplate(body="{context}"),
            ),
        )


def test_knowledge_store_bad_retriever_in_ctor_raises() -> None:
    transport = InMemoryDocumentStore()
    with pytest.raises(KnowledgeStoreError):
        KnowledgeStore(
            transport=transport,
            retriever="not a retriever",  # type: ignore[arg-type]
            builder=PromptBuilder(
                template=PromptTemplate(body="{context}"),
            ),
        )


def test_knowledge_store_bad_builder_in_ctor_raises() -> None:
    transport = InMemoryDocumentStore()
    with pytest.raises(KnowledgeStoreError):
        KnowledgeStore(
            transport=transport,
            retriever=BM25Retriever(transport=transport),
            builder="not a builder",  # type: ignore[arg-type]
        )


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
    store = KnowledgeStore.in_memory(
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


# ---------------------------------------------------------------------------
# Protocol runtime-checkability
# ---------------------------------------------------------------------------


def test_document_store_transport_protocol_runtime_checkable() -> None:
    class StubTransport:
        def write(self, document: Document) -> None:
            return None

        def read(self, doc_id: str) -> Document | None:
            return None

        def iter_documents(self):  # type: ignore[no-untyped-def]
            return iter(())

        def __len__(self) -> int:
            return 0

    assert isinstance(StubTransport(), DocumentStoreTransport)


def test_protocol_rejects_incomplete_implementation() -> None:
    class Broken:
        # Missing iter_documents / __len__.
        def write(self, document: Document) -> None:
            return None

        def read(self, doc_id: str) -> Document | None:
            return None

    assert not isinstance(Broken(), DocumentStoreTransport)
