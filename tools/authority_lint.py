"""Authority Lint — CORE-31 / CI-05 (Phase E0).

Static AST scan over Python imports. Enforces the architectural seams
declared in ``manifest.md`` §0.8 and ``docs/total_recall_index.md`` §40.

Rule set (ZIP v4):

* **T1**  Fast-path purity: ``hot path`` modules must not import
  ``governance`` (INV-17).
* **C2**  Neuromorphic isolation (NEUR-04 / SAFE-18).
* **C3**  Web-autolearn isolation (SAFE-15 / WEBLEARN-07).
* **W1**  Burner-wallet (memecoin adapters cannot import main wallet).
* **L1**  Learning ↔ Evolution direct imports forbidden in BOTH directions.
* **L2**  Offline engines may not import runtime engines.
* **L3**  Runtime engines may not import ``learning_engine`` or
  ``evolution_engine``.
* **B1**  Cross-runtime-engine direct imports forbidden — generalises T1.

Allow-list applies to every rule:

* ``core`` / ``core.contracts``
* ``state.ledger.reader``
* the standard library and approved third-party packages

Usage::

    python tools/authority_lint.py [--strict] [<repo_root>]

Exits non-zero on any violation. Designed to run in pre-commit and CI
(``.github/workflows/lint.yml``).
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import sys
from collections.abc import Iterable
from pathlib import Path

# ---------------------------------------------------------------------------
# Module-set definitions
# ---------------------------------------------------------------------------

RUNTIME_ENGINE_PACKAGES: tuple[str, ...] = (
    "intelligence_engine",
    "execution_engine",
    "system_engine",
    "governance_engine",
)

OFFLINE_ENGINE_PACKAGES: tuple[str, ...] = (
    "learning_engine",
    "evolution_engine",
)

ALL_ENGINE_PACKAGES: tuple[str, ...] = (
    RUNTIME_ENGINE_PACKAGES + OFFLINE_ENGINE_PACKAGES
)

# Common allow-list — these may be imported from any engine package.
ALLOWED_SHARED_PREFIXES: tuple[str, ...] = (
    "core",
    "core.contracts",
    "core.contracts.events",
    "core.contracts.engine",
    "state.ledger.reader",
)

# Hot-path modules subject to T1.
HOT_PATH_MODULES: tuple[str, ...] = (
    "mind.fast_execute",
    "execution_engine.hot_path",
)

# Neuromorphic files subject to C2.
NEUROMORPHIC_PREFIXES: tuple[str, ...] = (
    "mind.neuromorphic",
)

NEUROMORPHIC_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "execution_engine",
    "governance_engine",
    "intelligence_engine.alpha",
    "mind.fast_execute",
    "mind.execute",
    "execution.adapters",
    "wallet",
)

# Web-autolearn modules subject to C3.
WEB_AUTOLEARN_PREFIXES: tuple[str, ...] = (
    "mind.web_autolearn",
)

WEB_AUTOLEARN_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "execution",
    "execution_engine",
    "governance",
    "governance_engine",
    "mind.fast_execute",
    "wallet",
)

# Memecoin adapters subject to W1.
MEMECOIN_ADAPTER_PREFIXES: tuple[str, ...] = (
    "execution_engine.adapters.memecoin",
    "execution.adapters.memecoin",
)

MAIN_WALLET_FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "wallet.main_wallet",
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    file: Path
    line: int
    importer: str
    imported: str
    detail: str

    def format(self, repo_root: Path) -> str:
        try:
            rel = self.file.relative_to(repo_root)
        except ValueError:
            rel = self.file
        return (
            f"{self.rule} {rel}:{self.line}: "
            f"{self.importer!r} imports {self.imported!r} -- {self.detail}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_name_for(path: Path, repo_root: Path) -> str:
    """Convert a path inside the repo into a dotted module name."""
    rel = path.relative_to(repo_root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _starts_with_any(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == p or name.startswith(p + ".") for p in prefixes)


def _iter_python_files(repo_root: Path) -> Iterable[Path]:
    skip_dirs = {
        ".git",
        ".venv",
        "venv",
        ".tox",
        "node_modules",
        "build",
        "dist",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "rust",
        "target",
    }
    for p in repo_root.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        yield p


def _iter_imports(tree: ast.AST) -> Iterable[tuple[int, str]]:
    """Yield ``(lineno, dotted_module_name)`` for every import target."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                # Relative imports are local — we do not enforce against them
                # because they are by definition in-package.
                continue
            yield node.lineno, mod


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def _check_allow_list(target: str) -> bool:
    """Return True when ``target`` is unconditionally allowed."""
    return _starts_with_any(target, ALLOWED_SHARED_PREFIXES)


def _check_t1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, HOT_PATH_MODULES):
        return None
    if _starts_with_any(target, ("governance", "governance_engine")):
        return Violation(
            "T1",
            file,
            line,
            importer,
            target,
            "hot-path module must not import governance (INV-17)",
        )
    return None


def _check_c2(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, NEUROMORPHIC_PREFIXES):
        return None
    if _starts_with_any(target, NEUROMORPHIC_FORBIDDEN_PREFIXES):
        return Violation(
            "C2",
            file,
            line,
            importer,
            target,
            "neuromorphic isolation (NEUR-04 / SAFE-18)",
        )
    return None


def _check_c3(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, WEB_AUTOLEARN_PREFIXES):
        return None
    if _starts_with_any(target, WEB_AUTOLEARN_FORBIDDEN_PREFIXES):
        return Violation(
            "C3",
            file,
            line,
            importer,
            target,
            "web-autolearn isolation (SAFE-15 / WEBLEARN-07)",
        )
    return None


def _check_w1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, MEMECOIN_ADAPTER_PREFIXES):
        return None
    if _starts_with_any(target, MAIN_WALLET_FORBIDDEN_PREFIXES):
        return Violation(
            "W1",
            file,
            line,
            importer,
            target,
            "memecoin adapters must use burner wallet (INV-20 / SAFE-12)",
        )
    return None


def _check_l1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if _starts_with_any(importer, ("learning_engine",)) and _starts_with_any(
        target, ("evolution_engine",)
    ):
        return Violation(
            "L1",
            file,
            line,
            importer,
            target,
            "Learning ↔ Evolution domain boundary (sharing a process is not "
            "sharing a domain)",
        )
    if _starts_with_any(importer, ("evolution_engine",)) and _starts_with_any(
        target, ("learning_engine",)
    ):
        return Violation(
            "L1",
            file,
            line,
            importer,
            target,
            "Learning ↔ Evolution domain boundary (sharing a process is not "
            "sharing a domain)",
        )
    return None


def _check_l2(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, OFFLINE_ENGINE_PACKAGES):
        return None
    if _starts_with_any(target, RUNTIME_ENGINE_PACKAGES):
        return Violation(
            "L2",
            file,
            line,
            importer,
            target,
            "offline engine must not import runtime engines (INV-15)",
        )
    return None


def _check_l3(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    if not _starts_with_any(importer, RUNTIME_ENGINE_PACKAGES):
        return None
    if _starts_with_any(target, OFFLINE_ENGINE_PACKAGES):
        return Violation(
            "L3",
            file,
            line,
            importer,
            target,
            "runtime engine must not import offline engines (INV-15)",
        )
    return None


def _check_b1(
    importer: str, target: str, file: Path, line: int
) -> Violation | None:
    importer_pkg = next(
        (p for p in RUNTIME_ENGINE_PACKAGES if _starts_with_any(importer, (p,))),
        None,
    )
    if importer_pkg is None:
        return None
    target_pkg = next(
        (p for p in RUNTIME_ENGINE_PACKAGES if _starts_with_any(target, (p,))),
        None,
    )
    if target_pkg is None or target_pkg == importer_pkg:
        return None
    if _check_allow_list(target):
        return None
    return Violation(
        "B1",
        file,
        line,
        importer,
        target,
        f"cross-runtime-engine direct import {importer_pkg} → {target_pkg} "
        "(events bus only; INV-08 / INV-11)",
    )


RULE_CHECKS = (
    _check_t1,
    _check_c2,
    _check_c3,
    _check_w1,
    _check_l1,
    _check_l2,
    _check_l3,
    _check_b1,
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def lint_repo(repo_root: Path) -> list[Violation]:
    repo_root = repo_root.resolve()
    violations: list[Violation] = []
    for path in _iter_python_files(repo_root):
        # Ignore the lint tool itself and its tests' synthetic violation
        # fixtures (those are evaluated through targeted helpers in the test
        # suite, not by linting the repo tree).
        rel = path.relative_to(repo_root)
        if rel.parts[:1] == ("tools",) and rel.name == "authority_lint.py":
            continue
        if rel.parts[:2] == ("tests", "fixtures"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            violations.append(
                Violation(
                    "SYNTAX",
                    path,
                    exc.lineno or 0,
                    _module_name_for(path, repo_root),
                    "",
                    f"unparseable: {exc.msg}",
                )
            )
            continue
        importer = _module_name_for(path, repo_root)
        for line, target in _iter_imports(tree):
            if not target:
                continue
            for check in RULE_CHECKS:
                v = check(importer, target, path, line)
                if v is not None:
                    violations.append(v)
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="DIX VISION authority lint (Phase E0)."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="repo root (default: current directory)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero on any violation (default behaviour anyway)",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.root).resolve()
    violations = lint_repo(repo_root)

    if not violations:
        print(f"authority_lint: 0 violations in {repo_root}")
        return 0

    print(f"authority_lint: {len(violations)} violation(s) in {repo_root}")
    for v in violations:
        print(v.format(repo_root))
    return 1


if __name__ == "__main__":
    sys.exit(main())
