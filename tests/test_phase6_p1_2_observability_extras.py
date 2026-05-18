"""Phase-6 P1-2 guard — observability extras parity.

``tools/hydra_config.py`` and ``tools/jaeger_tracer.py`` each
declare a ``NEW_PIP_DEPENDENCIES`` tuple naming the optional
dependency they lazy-import inside the factory once an operator
opts into the real backend (Hydra config composition, Jaeger
distributed tracing). The Phase-6 audit flagged that those tuples
were documentary-only — there was no
``[project.optional-dependencies].observability`` row that let an
operator request both via ``pip install dixvision[observability]``.

This test pins the parity:

* every NEW_PIP_DEPENDENCIES entry from each ``tools/`` adapter
  appears in the ``observability`` extra in ``pyproject.toml``;
* the ``observability`` extra is non-empty and namespaced (every
  entry has a version floor so wheel resolution is deterministic).

A future PR adding a new ``tools/<adapter>.py`` with its own
``NEW_PIP_DEPENDENCIES`` tuple is expected to either extend this
extra or add a new one (in which case this guard's MODULES list
must grow).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

_ADAPTERS = (
    "tools/hydra_config.py",
    "tools/jaeger_tracer.py",
)

_TUPLE_RE = re.compile(
    r"^NEW_PIP_DEPENDENCIES:\s*[^=]*=\s*\((?P<body>[^)]*)\)",
    re.MULTILINE,
)


def _normalize(name: str) -> str:
    return name.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower()


def _load_pyproject() -> dict:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def _adapter_declared_deps(adapter_path: str) -> frozenset[str]:
    text = (_REPO_ROOT / adapter_path).read_text()
    match = _TUPLE_RE.search(text)
    assert match is not None, (
        f"{adapter_path} is expected to declare NEW_PIP_DEPENDENCIES as a module-level tuple"
    )
    raw = match.group("body")
    parts = [chunk.strip().strip(",").strip('"').strip("'") for chunk in raw.split(",")]
    return frozenset(_normalize(p) for p in parts if p)


def test_observability_extra_covers_hydra_and_jaeger() -> None:
    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]
    assert "observability" in extras, (
        "Phase-6 P1-2 regression: pyproject.toml is missing the "
        "`observability` extra. tools/hydra_config.py and "
        "tools/jaeger_tracer.py both declare NEW_PIP_DEPENDENCIES "
        "and the operator needs a single `pip install "
        "dixvision[observability]` opt-in path."
    )
    observability = {_normalize(d) for d in extras["observability"]}
    declared: set[str] = set()
    for adapter in _ADAPTERS:
        declared |= _adapter_declared_deps(adapter)
    missing = declared - observability
    assert not missing, (
        "Phase-6 P1-2 regression: the following lazy-adapter deps "
        "are declared via NEW_PIP_DEPENDENCIES but not pinned in "
        f"[project.optional-dependencies].observability: {sorted(missing)}"
    )


def test_observability_extra_entries_carry_version_floor() -> None:
    pyproject = _load_pyproject()
    extras = pyproject["project"]["optional-dependencies"]
    observability = extras["observability"]
    assert observability, "observability extra must not be empty"
    for entry in observability:
        assert any(op in entry for op in (">=", "==", ">", "~=")), (
            "Phase-6 P1-2 regression: observability extras must carry a "
            "version floor for deterministic wheel resolution; got "
            f"`{entry}`"
        )
