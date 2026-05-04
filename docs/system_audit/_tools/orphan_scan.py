"""Phase 3 — find orphan Python modules (no other code imports them).

Strategy:
* For each ``*.py`` file under the listed source roots, derive the
  dotted module path it would be imported as.
* Build an import graph by parsing every ``*.py`` with ``ast`` and
  recording every ``import X`` / ``from X import Y`` statement.
* A module is *orphan* if no other module imports it AND it is not a
  recognised entrypoint (FastAPI app, pytest test file, CLI script,
  ``__init__`` re-exporter, plugin loaded by registry).

Output:
* ``orphan_modules.csv`` — ``[file_id, path, in_degree, kind]`` where
  ``kind`` is ``orphan`` / ``entrypoint`` / ``test`` / ``init``.
* ``import_graph.json`` — full edge list for the report.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX = REPO_ROOT / "docs" / "system_audit" / "file_index.csv"
ORPHAN_OUT = REPO_ROOT / "docs" / "system_audit" / "orphan_modules.csv"
GRAPH_OUT = REPO_ROOT / "docs" / "system_audit" / "import_graph.json"

SOURCE_ROOTS = (
    "core",
    "intelligence_engine",
    "execution_engine",
    "governance_engine",
    "system_engine",
    "system",
    "sensory",
    "learning_engine",
    "evolution_engine",
    "ui",
    "dashboard_backend",
    "tools",
    "registry",
    "enforcement",
    "immutable_core",
    "state",
    "contracts",
)

ENTRYPOINT_PATTERNS = (
    re.compile(r"^bootstrap_kernel\.py$"),
    re.compile(r"^ui/server\.py$"),
    re.compile(r"^cockpit/app\.py$"),
    re.compile(r"^scripts/.+\.py$"),
    re.compile(r"^tools/.+_lint\.py$"),
    re.compile(r"^tools/codegen/.+\.py$"),
    re.compile(r"^tools/rust_revival_reminder\.py$"),
    re.compile(r"^docs/system_audit/_tools/.+\.py$"),
)


def path_to_module(path: str) -> str | None:
    if not path.endswith(".py"):
        return None
    if path.startswith("tests/"):
        return None
    parts = path[:-3].split("/")
    if not parts:
        return None
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return None
    return ".".join(parts)


def is_entrypoint(path: str) -> bool:
    return any(p.match(path) for p in ENTRYPOINT_PATTERNS)


def main() -> int:
    with INDEX.open() as fh:
        rows = list(csv.DictReader(fh))
    py_rows = [r for r in rows if r["path"].endswith(".py")]
    path_to_id = {r["path"]: r["file_id"] for r in py_rows}

    module_to_path: dict[str, str] = {}
    for r in py_rows:
        mod = path_to_module(r["path"])
        if mod is not None:
            module_to_path[mod] = r["path"]

    # Build import graph (edges importer -> imported_module).
    edges: list[tuple[str, str]] = []
    parse_failures: list[tuple[str, str]] = []
    for r in py_rows:
        path = r["path"]
        try:
            src = (REPO_ROOT / path).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=path)
        except SyntaxError as exc:
            parse_failures.append((path, str(exc)))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    edges.append((path, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    # relative import — resolve against package
                    pkg_parts = path[:-3].split("/")
                    if pkg_parts[-1] == "__init__":
                        pkg_parts = pkg_parts[:-1]
                    base = pkg_parts[: len(pkg_parts) - node.level + 1]
                    if not base:
                        continue
                    target = ".".join(base)
                    if node.module:
                        target = f"{target}.{node.module}"
                    edges.append((path, target))
                elif node.module:
                    edges.append((path, node.module))
                # Capture 'from pkg import sub' where 'sub' is a
                # submodule of 'pkg' (the alias case the dotted-prefix
                # walker misses).
                if isinstance(node, ast.ImportFrom) and node.module:
                    for alias in node.names:
                        edges.append(
                            (path, f"{node.module}.{alias.name}")
                        )

    # In-degree for each module path.
    in_degree: dict[str, int] = {r["path"]: 0 for r in py_rows}
    for importer, mod in edges:
        # Walk the dotted target up to find the deepest module that
        # exists in the repo (so 'from execution_engine.adapters.foo
        # import x' counts toward execution_engine/adapters/foo.py
        # *and* execution_engine/adapters/__init__.py *and* etc).
        parts = mod.split(".")
        for i in range(len(parts), 0, -1):
            candidate_mod = ".".join(parts[:i])
            tgt_path = module_to_path.get(candidate_mod)
            if tgt_path and tgt_path != importer:
                in_degree[tgt_path] = in_degree.get(tgt_path, 0) + 1

    # Classify
    orphans: list[dict] = []
    for r in py_rows:
        path = r["path"]
        deg = in_degree.get(path, 0)
        kind = "orphan"
        if path.startswith("tests/"):
            kind = "test"
        elif path.endswith("__init__.py"):
            kind = "init"
        elif is_entrypoint(path):
            kind = "entrypoint"
        elif deg > 0:
            kind = "imported"
        orphans.append(
            {
                "file_id": r["file_id"],
                "path": path,
                "in_degree": deg,
                "kind": kind,
            }
        )

    ORPHAN_OUT.parent.mkdir(parents=True, exist_ok=True)
    with ORPHAN_OUT.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_id", "path", "in_degree", "kind"])
        for o in orphans:
            writer.writerow([o["file_id"], o["path"], o["in_degree"], o["kind"]])

    with GRAPH_OUT.open("w") as fh:
        json.dump(
            {
                "n_modules": len(py_rows),
                "n_edges": len(edges),
                "parse_failures": parse_failures,
                "edges": edges[:5000],  # cap to keep file readable
                "edges_truncated": len(edges) > 5000,
            },
            fh,
            indent=2,
        )
    n_orphan = sum(1 for o in orphans if o["kind"] == "orphan")
    print(
        f"wrote {ORPHAN_OUT.relative_to(REPO_ROOT)} ({n_orphan} orphans / "
        f"{len(orphans)} python files)"
    )
    print(f"wrote {GRAPH_OUT.relative_to(REPO_ROOT)} ({len(edges)} edges)")
    if parse_failures:
        print(f"  parse failures: {len(parse_failures)}")
        for p, e in parse_failures[:5]:
            print(f"    {p}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
