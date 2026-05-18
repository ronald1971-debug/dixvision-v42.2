"""AUDIT-P0.1 guard — base runtime dependencies must cover ``ui.server``.

``ui.server`` is the harness entry point used by the Windows launcher
and every operator deployment. It imports ``fastapi`` / ``uvicorn`` /
``pydantic`` / ``websockets`` at module load. Hiding them under
``[project.optional-dependencies].dev`` meant a bare ``pip install
dixvision`` produced an installable package that crashed on import.

This test fails closed if anyone re-introduces that drift.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REQUIRED_BASE_DEPS = frozenset(
    {
        # cognitive runtime — imported by ``ui.cognitive_chat_runtime``
        "langchain-core",
        "langgraph",
        # harness HTTP / ASGI / validation / WS
        "fastapi",
        "uvicorn",
        "pydantic",
        "websockets",
        # config + registry parsing
        "pyyaml",
    }
)


def _load_pyproject() -> dict:
    repo_root = Path(__file__).resolve().parents[1]
    with (repo_root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _normalize(name: str) -> str:
    return name.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()


def test_ui_server_runtime_deps_are_base_not_dev() -> None:
    pyproject = _load_pyproject()
    base = {_normalize(d) for d in pyproject["project"]["dependencies"]}
    missing = REQUIRED_BASE_DEPS - base
    assert not missing, (
        "AUDIT-P0.1 regression: the following deps are imported by "
        "ui.server at module load and MUST be in [project.dependencies], "
        f"not [project.optional-dependencies].dev: {sorted(missing)}"
    )


def test_no_runtime_dep_lives_only_under_dev_extra() -> None:
    pyproject = _load_pyproject()
    base = {_normalize(d) for d in pyproject["project"]["dependencies"]}
    optional = pyproject["project"].get("optional-dependencies", {})
    dev = {_normalize(d) for d in optional.get("dev", [])}
    leaked = REQUIRED_BASE_DEPS & (dev - base)
    assert not leaked, (
        "AUDIT-P0.1 regression: runtime deps must be in [project.dependencies] "
        f"or both base and dev. These are dev-only: {sorted(leaked)}"
    )


# Phase-6 P0-1 guard — packages.find must cover every top-level Python
# package that ``ui.server`` imports at module load. A non-editable
# ``pip install dixvision`` resolves package contents through this
# include list; missing entries produce ``ModuleNotFoundError`` at
# import time. ``sensory*`` and ``dashboard_backend*`` were silently
# missing across multiple releases.
REQUIRED_INCLUDE_PATTERNS = frozenset(
    {
        "core*",
        "intelligence_engine*",
        "execution_engine*",
        "system_engine*",
        "governance_engine*",
        "learning_engine*",
        "evolution_engine*",
        "state*",
        "tools*",
        "ui*",
        "system*",
        "enforcement*",
        "immutable_core*",
        "opponent_model*",
        "simulation*",
        "sensory*",
        "dashboard_backend*",
    }
)


def test_packages_find_covers_runtime_imports() -> None:
    pyproject = _load_pyproject()
    include = frozenset(pyproject["tool"]["setuptools"]["packages"]["find"]["include"])
    missing = REQUIRED_INCLUDE_PATTERNS - include
    assert not missing, (
        "Phase-6 P0-1 regression: the following top-level packages are "
        "imported by ui.server at module load but are missing from "
        "[tool.setuptools.packages.find].include. A non-editable "
        f"`pip install` will ModuleNotFoundError on them: {sorted(missing)}"
    )
