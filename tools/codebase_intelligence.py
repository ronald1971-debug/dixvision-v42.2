# ADAPTED FROM: Sourcegraph's ``sg`` CLI patterns
#   sg search --json     — symbol & cross-reference search
#   sg code-nav          — definition / references / implementations
#   sg dependency-graph  — package import graph
#
# License: Apache-2.0 (Sourcegraph), MIT (this adaptation).
"""Codebase intelligence — Sourcegraph ``sg``-shape CLI surface over stdlib AST.

C-27 (OFFLINE_ONLY).

Mirrors the **CLI surface** of Sourcegraph ``sg`` — ``find-refs``,
``find-callers``, ``symbol-search``, ``dependency-graph``,
``authority-violations`` — but runs **entirely on stdlib ``ast``** so
Dyon system-intelligence flows can call into it from a hermetic CI
environment without any Sourcegraph server / login dependency.

Authority constraints:

* OFFLINE_ONLY — never wired to hot path.
* No clock, no PRNG, no network, no Sourcegraph server. Reads source
  files from disk only inside the public entrypoints.
* Pure stdlib — no top-level Sourcegraph imports of any kind.
* ``NEW_PIP_DEPENDENCIES = ()`` — the ``sg`` binary is **optional** and
  only used by :func:`sg_binary_factory`; the in-memory analyzers do
  not depend on it. Adapter is RUNTIME_SAFE for analysis output, but
  the public APIs gate themselves to OFFLINE_ONLY callers.

INV-15 (replay determinism):

* All result tuples are emitted in canonical sort order
  (alphabetical by symbol, then by location).
* Path resolution is relative to the supplied ``root`` so absolute
  filesystem locations never leak into the output. Internal storage
  uses ``str`` paths (no ``pathlib.Path`` objects exposed).

Use case — Dyon system intelligence:

* ``CodebaseIntelligence.find_callers("create_execution_intent")``
  surfaces every site that calls a governance-gated function.
* ``CodebaseIntelligence.authority_violations(tier_map=…)`` flags
  callers in a strict tier calling functions in a permissive tier.
* ``CodebaseIntelligence.dependency_graph()`` returns the module
  import graph for cross-reference analysis.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()
SOURCEGRAPH_ADAPTER_VERSION: str = "1"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
class SymbolKind(StrEnum):
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"


@dataclass(frozen=True, slots=True)
class SymbolRef:
    name: str
    kind: SymbolKind
    module: str
    location: str  # "lineno:col"


@dataclass(frozen=True, slots=True)
class CallSite:
    caller: str
    callee: str
    module: str
    location: str


@dataclass(frozen=True, slots=True)
class ImportEdge:
    from_module: str
    to_module: str
    location: str


@dataclass(frozen=True, slots=True)
class AuthorityViolation:
    caller_module: str
    caller_symbol: str
    callee_module: str
    callee_symbol: str
    caller_tier: str
    callee_tier: str
    location: str

    def to_dict(self) -> dict[str, str]:
        return {
            "caller_module": self.caller_module,
            "caller_symbol": self.caller_symbol,
            "callee_module": self.callee_module,
            "callee_symbol": self.callee_symbol,
            "caller_tier": self.caller_tier,
            "callee_tier": self.callee_tier,
            "location": self.location,
        }


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------
def _path_to_module(rel_path: str) -> str:
    """``execution_engine/hot_path/fast.py`` → ``execution_engine.hot_path.fast``."""
    parts = rel_path.replace("\\", "/").split("/")
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(p for p in parts if p)


def _walk_python_files(root: str) -> Iterator[str]:
    root_path = pathlib.Path(root)
    for p in sorted(root_path.rglob("*.py")):
        rel = p.relative_to(root_path).as_posix()
        yield rel


def _parse_or_skip(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except SyntaxError:
        return None


def _symbol_kind(node: ast.AST, in_class: bool) -> SymbolKind:
    if isinstance(node, ast.AsyncFunctionDef):
        return SymbolKind.ASYNC_FUNCTION if not in_class else SymbolKind.METHOD
    if isinstance(node, ast.FunctionDef):
        return SymbolKind.METHOD if in_class else SymbolKind.FUNCTION
    return SymbolKind.CLASS


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = [node.attr]
        cur: ast.AST = node.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return None


def _short_callee(callee: str) -> str:
    """``a.b.c`` → ``c``; pre-dotted name stays bare."""
    return callee.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# CodebaseIntelligence — Sourcegraph-shape API
# ---------------------------------------------------------------------------
class CodebaseIntelligence:
    """Stdlib AST analyzer with a Sourcegraph ``sg``-shape CLI surface.

    Parameters
    ----------
    root:
        Filesystem root from which all source files are walked. Paths
        in the output are emitted relative to this root.
    include:
        Optional iterable of path prefixes (relative to ``root``) to
        include. Empty means "include all".
    exclude:
        Optional iterable of path prefixes (relative to ``root``) to
        exclude. Common values: ``("tests/", ".venv/")``.
    """

    __slots__ = (
        "_calls",
        "_exclude",
        "_imports",
        "_include",
        "_modules",
        "_root",
        "_symbols",
    )

    def __init__(
        self,
        *,
        root: str,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] | None = None,
    ) -> None:
        if not isinstance(root, str) or not root:
            raise ValueError("root must be a non-empty str")
        root_path = pathlib.Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(f"root is not a directory: {root!r}")
        self._root = root_path.as_posix()
        self._include = tuple(sorted(include or ()))
        self._exclude = tuple(sorted(exclude or ()))
        self._symbols: tuple[SymbolRef, ...] = ()
        self._calls: tuple[CallSite, ...] = ()
        self._imports: tuple[ImportEdge, ...] = ()
        self._modules: tuple[str, ...] = ()
        self._index()

    @property
    def root(self) -> str:
        return self._root

    @property
    def modules(self) -> tuple[str, ...]:
        return self._modules

    def symbols(self) -> tuple[SymbolRef, ...]:
        return self._symbols

    def calls(self) -> tuple[CallSite, ...]:
        return self._calls

    def imports(self) -> tuple[ImportEdge, ...]:
        return self._imports

    # ------------------------------------------------------------------
    # sg CLI mirror — find-refs / find-callers / symbol-search
    # ------------------------------------------------------------------
    def find_refs(self, *, symbol: str) -> tuple[CallSite, ...]:
        """Mirror of ``sg search --refs`` for a single symbol name."""
        if not isinstance(symbol, str):
            raise TypeError("symbol must be a str")
        if not symbol:
            return ()
        out: list[CallSite] = []
        for site in self._calls:
            if site.callee == symbol or _short_callee(site.callee) == symbol:
                out.append(site)
        out.sort(key=lambda s: (s.module, s.caller, s.location))
        return tuple(out)

    def find_callers(self, callee: str) -> tuple[CallSite, ...]:
        """Sourcegraph ``sg code-nav references --callers`` shape."""
        return self.find_refs(symbol=callee)

    def symbol_search(self, *, query: str) -> tuple[SymbolRef, ...]:
        """Sourcegraph ``sg search --symbols`` substring match.

        Pure substring; case-sensitive (matches how ``sg`` defaults).
        """
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        if not query:
            return ()
        out = [s for s in self._symbols if query in s.name]
        out.sort(key=lambda s: (s.name, s.module, s.location))
        return tuple(out)

    def dependency_graph(self) -> tuple[ImportEdge, ...]:
        """Sourcegraph ``sg dependency-graph`` shape — every import edge."""
        return self._imports

    # ------------------------------------------------------------------
    # Authority — Dyon system intelligence use case
    # ------------------------------------------------------------------
    def authority_violations(
        self,
        *,
        tier_map: Mapping[str, str],
    ) -> tuple[AuthorityViolation, ...]:
        """Flag callers whose tier is **stricter** than their callee's.

        Tier rank (smaller is stricter):

        * ``RUNTIME_SAFE`` = 0 (hot path)
        * ``OFFLINE_ONLY`` = 1 (CI / sandbox)
        * ``RESEARCH_SOURCE`` = 2 (notebooks / analysis)

        ``tier_map`` is symbol-keyed (bare name); the caller's tier and
        the callee's tier must both appear in the map for a row to be
        emitted. Otherwise the call is treated as unclassified and
        ignored.
        """
        if not isinstance(tier_map, Mapping):
            raise TypeError("tier_map must be a Mapping")
        rank = {
            "RUNTIME_SAFE": 0,
            "OFFLINE_ONLY": 1,
            "RESEARCH_SOURCE": 2,
        }
        out: list[AuthorityViolation] = []
        for site in self._calls:
            callee_short = _short_callee(site.callee)
            caller_tier = tier_map.get(site.caller)
            callee_tier = tier_map.get(callee_short)
            if caller_tier is None or callee_tier is None:
                continue
            if caller_tier not in rank or callee_tier not in rank:
                continue
            if rank[caller_tier] >= rank[callee_tier]:
                continue
            out.append(
                AuthorityViolation(
                    caller_module=site.module,
                    caller_symbol=site.caller,
                    callee_module=site.module,
                    callee_symbol=callee_short,
                    caller_tier=caller_tier,
                    callee_tier=callee_tier,
                    location=site.location,
                )
            )
        out.sort(
            key=lambda v: (
                v.caller_module,
                v.caller_symbol,
                v.callee_symbol,
                v.location,
            )
        )
        return tuple(out)

    # ------------------------------------------------------------------
    # internal index
    # ------------------------------------------------------------------
    def _included(self, rel: str) -> bool:
        if self._include and not any(rel.startswith(p) for p in self._include):
            return False
        if self._exclude and any(rel.startswith(p) for p in self._exclude):
            return False
        return True

    def _index(self) -> None:
        symbols: list[SymbolRef] = []
        calls: list[CallSite] = []
        imports: list[ImportEdge] = []
        modules: list[str] = []
        for rel in _walk_python_files(self._root):
            if not self._included(rel):
                continue
            module = _path_to_module(rel)
            modules.append(module)
            abs_path = pathlib.Path(self._root) / rel
            try:
                text = abs_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            tree = _parse_or_skip(text)
            if tree is None:
                continue
            self._index_symbols(module, tree, symbols)
            self._index_calls(module, tree, calls)
            self._index_imports(module, tree, imports)
        symbols.sort(key=lambda s: (s.module, s.name, s.location))
        calls.sort(key=lambda c: (c.module, c.caller, c.callee, c.location))
        imports.sort(key=lambda i: (i.from_module, i.to_module, i.location))
        modules.sort()
        self._symbols = tuple(symbols)
        self._calls = tuple(calls)
        self._imports = tuple(imports)
        self._modules = tuple(dict.fromkeys(modules))

    def _index_symbols(
        self,
        module: str,
        tree: ast.Module,
        out: list[SymbolRef],
    ) -> None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = _symbol_kind(node, in_class=False)
                out.append(
                    SymbolRef(
                        name=node.name,
                        kind=kind,
                        module=module,
                        location=f"{node.lineno}:{node.col_offset}",
                    )
                )
            elif isinstance(node, ast.ClassDef):
                out.append(
                    SymbolRef(
                        name=node.name,
                        kind=SymbolKind.CLASS,
                        module=module,
                        location=f"{node.lineno}:{node.col_offset}",
                    )
                )
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(
                            SymbolRef(
                                name=child.name,
                                kind=_symbol_kind(child, in_class=True),
                                module=module,
                                location=(f"{child.lineno}:{child.col_offset}"),
                            )
                        )

    def _index_calls(
        self,
        module: str,
        tree: ast.Module,
        out: list[CallSite],
    ) -> None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._index_calls_in_fn(module, node.name, node, out)
            elif isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        self._index_calls_in_fn(module, child.name, child, out)

    @staticmethod
    def _index_calls_in_fn(
        module: str,
        caller_name: str,
        fn_node: ast.AST,
        out: list[CallSite],
    ) -> None:
        for child in ast.walk(fn_node):
            if isinstance(child, ast.Call):
                callee = _call_name(child.func)
                if callee is None:
                    continue
                out.append(
                    CallSite(
                        caller=caller_name,
                        callee=callee,
                        module=module,
                        location=(
                            f"{getattr(child, 'lineno', 0)}:{getattr(child, 'col_offset', 0)}"
                        ),
                    )
                )

    @staticmethod
    def _index_imports(
        module: str,
        tree: ast.Module,
        out: list[ImportEdge],
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    out.append(
                        ImportEdge(
                            from_module=module,
                            to_module=alias.name,
                            location=(
                                f"{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"
                            ),
                        )
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                out.append(
                    ImportEdge(
                        from_module=module,
                        to_module=node.module,
                        location=(f"{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"),
                    )
                )


# ---------------------------------------------------------------------------
# Optional live ``sg`` binary factory
# ---------------------------------------------------------------------------
def sg_binary_factory(*, binary: str = "sg") -> Any:
    """Lazy-bind the optional ``sg`` CLI binary.

    Returns a thin record describing the resolved binary path. Caller
    is responsible for actually subprocessing it. This factory exists
    so callers can opt into the live ``sg`` index without taking a
    hard dependency on it at module import time.

    Raises
    ------
    RuntimeError
        If the ``sg`` binary cannot be found on ``$PATH``.
    """
    if not isinstance(binary, str) or not binary:
        raise ValueError("binary must be a non-empty str")
    import shutil  # noqa: PLC0415

    resolved = shutil.which(binary)
    if resolved is None:
        raise RuntimeError(
            f"sg binary {binary!r} not found on $PATH; "
            f"install Sourcegraph CLI (see docs/sourcegraph_dyon_usage.md)"
        )
    return {"binary": binary, "path": resolved}


__all__ = [
    "AuthorityViolation",
    "CallSite",
    "CodebaseIntelligence",
    "ImportEdge",
    "NEW_PIP_DEPENDENCIES",
    "SOURCEGRAPH_ADAPTER_VERSION",
    "SymbolKind",
    "SymbolRef",
    "sg_binary_factory",
]
