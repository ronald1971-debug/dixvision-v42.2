"""Per-module consumption declarations for SCVS Phase 1.

Every module that consumes data declares a ``consumes.yaml`` next to
its ``__init__``. The shape is intentionally tiny:

.. code-block:: yaml

    module: intelligence_engine.signal_pipeline
    inputs:
      - source_id: SRC-MARKET-BINANCE-001
        required: true

This module loads + validates one declaration in isolation, and walks
a repository tree to discover every declaration that exists. The
bidirectional closure (every ``enabled`` source must be consumed; every
declared input must reference a registered source) is enforced by
:mod:`system_engine.scvs.lint`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONSUMES_FILENAME = "consumes.yaml"


@dataclass(frozen=True, slots=True)
class ConsumptionInput:
    """One declared input."""

    source_id: str
    required: bool


@dataclass(frozen=True, slots=True)
class ConsumptionDeclaration:
    """A loaded ``consumes.yaml``."""

    module: str
    inputs: tuple[ConsumptionInput, ...]
    path: Path


def _parse_input(raw: Any, idx: int, ctx: str) -> ConsumptionInput:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{ctx}: inputs[{idx}] is not a mapping")
    if "source_id" not in raw:
        raise ValueError(f"{ctx}: inputs[{idx}] missing 'source_id'")
    sid = str(raw["source_id"])
    if not sid.startswith("SRC-"):
        raise ValueError(
            f"{ctx}: inputs[{idx}] source_id '{sid}' must start with 'SRC-'"
        )
    return ConsumptionInput(
        source_id=sid,
        required=bool(raw.get("required", True)),
    )


def load_consumption_declaration(path: str | Path) -> ConsumptionDeclaration:
    """Load + strictly validate a single ``consumes.yaml``."""

    p = Path(path)
    raw: Any = yaml.safe_load(p.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"consumes file at {p} is not a mapping")

    module = raw.get("module")
    if not isinstance(module, str) or not module:
        raise ValueError(f"{p}: 'module' must be a non-empty string")

    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, list):
        raise ValueError(f"{p}: 'inputs' must be a list")

    ctx = str(p)
    inputs: list[ConsumptionInput] = []
    seen: set[str] = set()
    for idx, row in enumerate(inputs_raw):
        decl = _parse_input(row, idx, ctx)
        if decl.source_id in seen:
            raise ValueError(
                f"{p}: duplicate source_id {decl.source_id!r} in inputs"
            )
        seen.add(decl.source_id)
        inputs.append(decl)

    return ConsumptionDeclaration(
        module=module,
        inputs=tuple(inputs),
        path=p,
    )


def discover_consumption_declarations(
    roots: Iterable[str | Path],
) -> tuple[ConsumptionDeclaration, ...]:
    """Walk ``roots`` and load every ``consumes.yaml`` found."""

    out: list[ConsumptionDeclaration] = []
    seen_modules: set[str] = set()
    for root in roots:
        for p in sorted(Path(root).rglob(CONSUMES_FILENAME)):
            if any(part.startswith(".") for part in p.parts):
                continue
            decl = load_consumption_declaration(p)
            if decl.module in seen_modules:
                raise ValueError(
                    f"duplicate consumes.yaml module {decl.module!r}"
                )
            seen_modules.add(decl.module)
            out.append(decl)
    return tuple(out)
