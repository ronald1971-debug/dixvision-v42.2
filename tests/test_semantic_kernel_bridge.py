# ADAPTED FROM: https://github.com/microsoft/semantic-kernel (MIT)
#
# Tests for the C-13 semantic-kernel bridge — plugin registry +
# dispatcher + DIX-side memory contract.
"""C-13 tests: Kernel registry / invocation / memory / lazy seam."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

from intelligence_engine.cognitive import semantic_kernel_bridge as skb
from intelligence_engine.cognitive.semantic_kernel_bridge import (
    NEW_PIP_DEPENDENCIES,
    InMemorySemanticMemory,
    InvocationError,
    Kernel,
    KernelFunction,
    KernelInvocation,
    KernelPlugin,
    KernelResult,
    MemoryEntry,
    MemoryError,
    PluginRegistryError,
    SemanticKernelError,
    SemanticMemoryProtocol,
    enable_semantic_kernel_factory,
)

# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("semantic-kernel",)


def test_error_hierarchy() -> None:
    assert issubclass(SemanticKernelError, ValueError)
    assert issubclass(PluginRegistryError, SemanticKernelError)
    assert issubclass(InvocationError, SemanticKernelError)
    assert issubclass(MemoryError, SemanticKernelError)


def test_public_surface_matches_all() -> None:
    expected = (
        "NEW_PIP_DEPENDENCIES",
        "SemanticKernelError",
        "PluginRegistryError",
        "InvocationError",
        "MemoryError",
        "KernelFunction",
        "KernelPlugin",
        "KernelInvocation",
        "KernelResult",
        "MemoryEntry",
        "SemanticMemoryProtocol",
        "InMemorySemanticMemory",
        "Kernel",
        "enable_semantic_kernel_factory",
    )
    assert skb.__all__ == expected


# ---------------------------------------------------------------------------
# AST guards — vendor + provider + non-determinism
# ---------------------------------------------------------------------------


_MODULE_SRC = Path(skb.__file__).read_text()
_MODULE_TREE = ast.parse(_MODULE_SRC)


def _top_level_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Import):
            for a in node.names:
                names.append(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                names.append(node.module)
    return names


def test_top_level_imports_are_stdlib_only() -> None:
    forbidden = {
        "semantic_kernel",
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
        # B1 forbidden cross-engine imports
        "execution_engine",
        "governance_engine",
        "system_engine",
        "learning_engine",
        "evolution_engine",
        "core.contracts.events",
    }
    found = _top_level_imports(_MODULE_TREE)
    for name in found:
        head = name.split(".")[0]
        assert head not in forbidden, (head, name)
        assert name not in forbidden, name


def test_semantic_kernel_only_imported_inside_factory_body() -> None:
    # The lazy seam: ``semantic_kernel`` may only appear as an
    # ``import semantic_kernel`` statement nested inside the body of
    # ``enable_semantic_kernel_factory``. Any other occurrence —
    # module-level, inside ``Kernel.invoke``, inside the
    # ``InMemorySemanticMemory`` class — fails this guard.
    factory_node: ast.FunctionDef | None = None
    for node in _MODULE_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "enable_semantic_kernel_factory":
            factory_node = node
            break
    assert factory_node is not None, "factory not found"

    factory_offsets: set[int] = set()
    for sub in ast.walk(factory_node):
        if isinstance(sub, ast.Import):
            factory_offsets.add(sub.lineno)

    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "semantic_kernel" or a.name.startswith("semantic_kernel."):
                    assert node.lineno in factory_offsets, (
                        f"semantic_kernel imported at line "
                        f"{node.lineno} outside the lazy factory body"
                    )
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and (
                node.module == "semantic_kernel" or node.module.startswith("semantic_kernel.")
            ):
                pytest.fail(
                    f"semantic_kernel imported via 'from' at line "
                    f"{node.lineno} — must be inside factory body "
                    "and via plain 'import'"
                )


def test_module_imports_cleanly_without_semantic_kernel_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even with semantic_kernel forced unimportable, the bridge module
    # must import + parse cleanly. Production environments without
    # the optional vendor dep still mount this surface.
    monkeypatch.setitem(sys.modules, "semantic_kernel", None)
    spec = importlib.util.spec_from_file_location("_skb_isolated", skb.__file__)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses walks ``sys.modules[cls.__module__]`` to resolve
    # string-form annotations — register the alias first so the
    # ``__post_init__`` checks can construct the value objects.
    monkeypatch.setitem(sys.modules, "_skb_isolated", mod)
    spec.loader.exec_module(mod)
    assert mod.NEW_PIP_DEPENDENCIES == ("semantic-kernel",)


def test_no_wall_clock_or_prng_calls_in_module() -> None:
    forbidden_attrs = {
        ("time", "time"),
        ("time", "time_ns"),
        ("time", "monotonic"),
        ("time", "monotonic_ns"),
        ("datetime", "now"),
        ("datetime", "utcnow"),
        ("random", "random"),
        ("random", "randint"),
    }
    for node in ast.walk(_MODULE_TREE):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            pair = (node.value.id, node.attr)
            assert pair not in forbidden_attrs, pair


# ---------------------------------------------------------------------------
# KernelFunction
# ---------------------------------------------------------------------------


def _echo(invocation: KernelInvocation, memory: SemanticMemoryProtocol) -> str:
    return invocation.arguments["text"]


def test_kernel_function_is_frozen_and_slotted() -> None:
    fn = KernelFunction(
        function_name="echo",
        description="echo text",
        parameter_names=("text",),
        call=_echo,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        fn.function_name = "other"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", "1bad", "bad-name", "with space"])
def test_kernel_function_rejects_bad_identifier(bad: str) -> None:
    with pytest.raises(PluginRegistryError):
        KernelFunction(
            function_name=bad,
            description="",
            parameter_names=(),
            call=_echo,
        )


def test_kernel_function_rejects_duplicate_parameters() -> None:
    with pytest.raises(PluginRegistryError):
        KernelFunction(
            function_name="f",
            description="",
            parameter_names=("a", "a"),
            call=_echo,
        )


def test_kernel_function_rejects_non_tuple_parameters() -> None:
    with pytest.raises(PluginRegistryError):
        KernelFunction(
            function_name="f",
            description="",
            parameter_names=["a"],  # type: ignore[arg-type]
            call=_echo,
        )


def test_kernel_function_rejects_non_callable() -> None:
    with pytest.raises(PluginRegistryError):
        KernelFunction(
            function_name="f",
            description="",
            parameter_names=("a",),
            call="not callable",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# KernelPlugin
# ---------------------------------------------------------------------------


def test_kernel_plugin_rejects_empty_functions() -> None:
    with pytest.raises(PluginRegistryError):
        KernelPlugin(plugin_name="p", functions=())


def test_kernel_plugin_rejects_duplicate_function_names() -> None:
    fn = KernelFunction(
        function_name="f",
        description="",
        parameter_names=("text",),
        call=_echo,
    )
    with pytest.raises(PluginRegistryError):
        KernelPlugin(plugin_name="p", functions=(fn, fn))


def test_kernel_plugin_rejects_non_function_member() -> None:
    fn = KernelFunction(
        function_name="f",
        description="",
        parameter_names=("text",),
        call=_echo,
    )
    with pytest.raises(PluginRegistryError):
        KernelPlugin(
            plugin_name="p",
            functions=(fn, "not a function"),  # type: ignore[arg-type]
        )


def test_kernel_plugin_function_names() -> None:
    a = KernelFunction(
        function_name="a",
        description="",
        parameter_names=("text",),
        call=_echo,
    )
    b = KernelFunction(
        function_name="b",
        description="",
        parameter_names=("text",),
        call=_echo,
    )
    plugin = KernelPlugin(plugin_name="p", functions=(a, b))
    assert plugin.function_names() == ("a", "b")


# ---------------------------------------------------------------------------
# KernelInvocation
# ---------------------------------------------------------------------------


def test_kernel_invocation_rejects_non_mapping_arguments() -> None:
    with pytest.raises(InvocationError):
        KernelInvocation(
            plugin_name="p",
            function_name="f",
            arguments=("text", "hello"),  # type: ignore[arg-type]
        )


def test_kernel_invocation_rejects_non_string_argument_value() -> None:
    with pytest.raises(InvocationError):
        KernelInvocation(
            plugin_name="p",
            function_name="f",
            arguments={"text": 42},  # type: ignore[dict-item]
        )


# ---------------------------------------------------------------------------
# Kernel — happy path + dispatch
# ---------------------------------------------------------------------------


def _build_kernel() -> Kernel:
    plugin = KernelPlugin(
        plugin_name="text",
        functions=(
            KernelFunction(
                function_name="echo",
                description="echo back",
                parameter_names=("text",),
                call=_echo,
            ),
        ),
    )
    kernel = Kernel()
    kernel.register_plugin(plugin)
    return kernel


def test_kernel_invoke_returns_typed_result() -> None:
    kernel = _build_kernel()
    result = kernel.invoke(
        KernelInvocation(
            plugin_name="text",
            function_name="echo",
            arguments={"text": "hello"},
        ),
        audit_id="audit-1",
    )
    assert isinstance(result, KernelResult)
    assert result.plugin_name == "text"
    assert result.function_name == "echo"
    assert result.value == "hello"
    assert result.audit_id == "audit-1"


def test_kernel_invoke_returns_str_only_no_freeform_object() -> None:
    # The dispatcher refuses callables that return non-str — the
    # canonical SK return contract is str. This pins the rule that
    # cognitive callers cannot smuggle untyped objects through the
    # bridge.
    def bad_call(invocation: KernelInvocation, memory: SemanticMemoryProtocol) -> str:
        return 42  # type: ignore[return-value]

    plugin = KernelPlugin(
        plugin_name="text",
        functions=(
            KernelFunction(
                function_name="oops",
                description="",
                parameter_names=("text",),
                call=bad_call,
            ),
        ),
    )
    kernel = Kernel()
    kernel.register_plugin(plugin)
    with pytest.raises(InvocationError):
        kernel.invoke(
            KernelInvocation(
                plugin_name="text",
                function_name="oops",
                arguments={"text": "x"},
            ),
            audit_id="a",
        )


def test_kernel_invoke_rejects_extra_or_missing_arguments() -> None:
    kernel = _build_kernel()
    with pytest.raises(InvocationError):
        kernel.invoke(
            KernelInvocation(
                plugin_name="text",
                function_name="echo",
                arguments={},
            ),
            audit_id="a",
        )
    with pytest.raises(InvocationError):
        kernel.invoke(
            KernelInvocation(
                plugin_name="text",
                function_name="echo",
                arguments={"text": "hello", "extra": "x"},
            ),
            audit_id="a",
        )


def test_kernel_invoke_unknown_plugin_or_function() -> None:
    kernel = _build_kernel()
    with pytest.raises(InvocationError):
        kernel.invoke(
            KernelInvocation(
                plugin_name="missing",
                function_name="echo",
                arguments={"text": "x"},
            ),
            audit_id="a",
        )
    with pytest.raises(InvocationError):
        kernel.invoke(
            KernelInvocation(
                plugin_name="text",
                function_name="missing",
                arguments={"text": "x"},
            ),
            audit_id="a",
        )


def test_kernel_register_plugin_rejects_duplicates() -> None:
    kernel = _build_kernel()
    plugin = KernelPlugin(
        plugin_name="text",
        functions=(
            KernelFunction(
                function_name="echo",
                description="",
                parameter_names=("text",),
                call=_echo,
            ),
        ),
    )
    with pytest.raises(PluginRegistryError):
        kernel.register_plugin(plugin)


def test_kernel_plugin_and_function_names_round_trip() -> None:
    kernel = _build_kernel()
    assert kernel.plugin_names() == ("text",)
    assert kernel.function_names("text") == ("echo",)
    with pytest.raises(InvocationError):
        kernel.function_names("missing")


# ---------------------------------------------------------------------------
# Kernel — INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def _digest_result(r: KernelResult) -> bytes:
    body = (f"{r.plugin_name}|{r.function_name}|{r.value}|{r.audit_id}").encode()
    return hashlib.blake2b(body, digest_size=16).digest()


def test_kernel_invoke_inv15_three_run_byte_identical() -> None:
    runs: list[bytes] = []
    for _ in range(3):
        kernel = _build_kernel()
        result = kernel.invoke(
            KernelInvocation(
                plugin_name="text",
                function_name="echo",
                arguments={"text": "deterministic"},
            ),
            audit_id="audit-7",
        )
        runs.append(_digest_result(result))
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# SemanticMemoryProtocol + InMemorySemanticMemory
# ---------------------------------------------------------------------------


def test_in_memory_semantic_memory_implements_protocol() -> None:
    mem = InMemorySemanticMemory()
    assert isinstance(mem, SemanticMemoryProtocol)


def test_in_memory_semantic_memory_recall_score_ordering() -> None:
    mem = InMemorySemanticMemory()
    mem.store("a", "hello world")
    mem.store("b", "hello")
    mem.store("c", "unrelated")
    hits = mem.recall("hello", top_k=10)
    # Exact match scores 1.0 (b), substring scores 0.5 (a),
    # unrelated row drops out entirely.
    assert tuple(h.entry_id for h in hits) == ("b", "a")
    assert hits[0].score == 1.0
    assert hits[1].score == 0.5


def test_in_memory_semantic_memory_recall_inv15_deterministic() -> None:
    digests: list[bytes] = []
    for _ in range(3):
        mem = InMemorySemanticMemory()
        mem.store("a", "alpha beta")
        mem.store("b", "alpha")
        mem.store("c", "gamma")
        hits = mem.recall("alpha", top_k=2)
        body = "|".join(f"{h.entry_id}:{h.text}:{h.score:.6f}" for h in hits).encode()
        digests.append(hashlib.blake2b(body, digest_size=16).digest())
    assert digests[0] == digests[1] == digests[2]


def test_in_memory_semantic_memory_rejects_bad_top_k() -> None:
    mem = InMemorySemanticMemory()
    with pytest.raises(MemoryError):
        mem.recall("q", top_k=0)
    with pytest.raises(MemoryError):
        mem.recall("q", top_k=-1)
    with pytest.raises(MemoryError):
        mem.recall("q", top_k=True)  # type: ignore[arg-type]


def test_in_memory_semantic_memory_rejects_bad_store_args() -> None:
    mem = InMemorySemanticMemory()
    with pytest.raises(MemoryError):
        mem.store("", "text")
    with pytest.raises(MemoryError):
        mem.store("id", 42)  # type: ignore[arg-type]


def test_memory_entry_score_must_be_in_unit_interval() -> None:
    MemoryEntry(entry_id="a", text="t", score=0.0)
    MemoryEntry(entry_id="a", text="t", score=1.0)
    with pytest.raises(MemoryError):
        MemoryEntry(entry_id="a", text="t", score=-0.1)
    with pytest.raises(MemoryError):
        MemoryEntry(entry_id="a", text="t", score=1.1)


def test_kernel_invoke_receives_bound_memory() -> None:
    # The DIX memory contract is injected at kernel construction —
    # the KernelFunction.call callable receives the same memory
    # instance regardless of how many times it is invoked. This
    # pins that SK plugins read DIX memory, never the SK built-in.
    seen: list[SemanticMemoryProtocol] = []

    def capture(
        invocation: KernelInvocation,
        memory: SemanticMemoryProtocol,
    ) -> str:
        seen.append(memory)
        return "ok"

    mem = InMemorySemanticMemory()
    plugin = KernelPlugin(
        plugin_name="p",
        functions=(
            KernelFunction(
                function_name="f",
                description="",
                parameter_names=("a",),
                call=capture,
            ),
        ),
    )
    kernel = Kernel(memory=mem)
    kernel.register_plugin(plugin)
    for i in range(3):
        kernel.invoke(
            KernelInvocation(
                plugin_name="p",
                function_name="f",
                arguments={"a": str(i)},
            ),
            audit_id=f"a-{i}",
        )
    assert seen[0] is mem
    assert seen[1] is mem
    assert seen[2] is mem
    assert kernel.memory is mem


# ---------------------------------------------------------------------------
# Lazy ``semantic_kernel`` seam — factory contract
# ---------------------------------------------------------------------------


def test_enable_semantic_kernel_factory_signature() -> None:
    sig = inspect.signature(enable_semantic_kernel_factory)
    # Only one keyword-only parameter: completion_callable. The
    # factory is the dispatch seam; everything else is intentionally
    # NOT a parameter so callers cannot pass in provider SDK objects
    # directly.
    assert list(sig.parameters) == ["completion_callable"]
    assert sig.parameters["completion_callable"].kind == inspect.Parameter.KEYWORD_ONLY


def test_enable_semantic_kernel_factory_rejects_non_callable() -> None:
    with pytest.raises(SemanticKernelError):
        enable_semantic_kernel_factory(
            completion_callable="not callable",  # type: ignore[arg-type]
        )


def test_enable_semantic_kernel_factory_routes_through_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The factory must drive every LLM dispatch through the
    # caller-supplied ``completion_callable``. Production wiring
    # passes a LiteLLMRouter-backed function — the bridge never
    # reaches openai/anthropic/etc. directly.
    fake_sk = type(sys)("semantic_kernel")
    monkeypatch.setitem(sys.modules, "semantic_kernel", fake_sk)

    calls: list[str] = []

    def router(prompt: str) -> str:
        calls.append(prompt)
        return f"[litellm] {prompt}"

    invoke = enable_semantic_kernel_factory(completion_callable=router)
    inv = KernelInvocation(
        plugin_name="text",
        function_name="echo",
        arguments={"text": "hello", "lang": "en"},
    )
    result = invoke(inv, "audit-99")
    assert isinstance(result, KernelResult)
    assert result.value.startswith("[litellm] ")
    # arguments rendered in declaration order — Mapping iteration is
    # insertion-ordered under CPython, which DIX relies on for INV-15.
    assert calls == ["text: hello\nlang: en"]
    assert result.audit_id == "audit-99"


def test_enable_semantic_kernel_factory_audit_id_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sk = type(sys)("semantic_kernel")
    monkeypatch.setitem(sys.modules, "semantic_kernel", fake_sk)
    invoke = enable_semantic_kernel_factory(completion_callable=lambda p: "ok")
    with pytest.raises(InvocationError):
        invoke(
            KernelInvocation(
                plugin_name="p",
                function_name="f",
                arguments={"a": "x"},
            ),
            "",
        )


def test_enable_semantic_kernel_factory_completion_must_return_str(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_sk = type(sys)("semantic_kernel")
    monkeypatch.setitem(sys.modules, "semantic_kernel", fake_sk)
    invoke = enable_semantic_kernel_factory(
        completion_callable=lambda p: 42  # type: ignore[arg-type,return-value]
    )
    with pytest.raises(InvocationError):
        invoke(
            KernelInvocation(
                plugin_name="p",
                function_name="f",
                arguments={"a": "x"},
            ),
            "audit",
        )


def test_completion_callable_signature_accepts_only_prompt() -> None:
    # The DIX seam keeps the callable shape narrow: ``(prompt: str) ->
    # str``. This pins that no extra kwargs (api_key, model_name,
    # provider) can leak into the factory boundary — those belong on
    # the LiteLLMRouter side, not here.
    src = Path(skb.__file__).read_text()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "enable_semantic_kernel_factory":
            args = node.args
            assert args.args == [], "factory must take no positional args"
            assert [a.arg for a in args.kwonlyargs] == ["completion_callable"]
            assert args.vararg is None
            assert args.kwarg is None
            return
    pytest.fail("factory function not found")
