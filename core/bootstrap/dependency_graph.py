"""
core/bootstrap/dependency_graph.py
Minimal topologically-ordered dependency graph used to sequence boot steps.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass
class DependencyGraph:
    _deps: dict[str, set[str]] = field(default_factory=dict)

    def add(self, node: str, *depends_on: str) -> None:
        self._deps.setdefault(node, set()).update(depends_on)
        for d in depends_on:
            self._deps.setdefault(d, set())

    def topo_order(self) -> list[str]:
        visited: set[str] = set()
        stack: list[str] = []
        temp: set[str] = set()

        def visit(n: str) -> None:
            if n in visited:
                return
            if n in temp:
                raise ValueError(f"cycle at {n}")
            temp.add(n)
            for d in sorted(self._deps.get(n, ())):
                visit(d)
            temp.discard(n)
            visited.add(n)
            stack.append(n)

        for node in sorted(self._deps):
            visit(node)
        return stack

    def nodes(self) -> Iterable[str]:
        return self._deps.keys()
