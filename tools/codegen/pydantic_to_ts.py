"""Pydantic v2 → TypeScript codegen for the operator dashboard.

Renders a deterministic ``.ts`` file from a list of Pydantic models so
the React/TS dashboard (`dashboard2026/`) can consume FastAPI responses
without hand-written types that drift from the Python source.

Design choices
--------------

* **No external Node toolchain.** The script reads
  ``model.model_json_schema()`` and emits TypeScript directly. Adding
  ``pydantic-to-typescript`` / ``json-schema-to-typescript`` would
  drag a Node build step into Python CI; this pipeline is small
  enough to maintain in-tree.
* **Deterministic output.** Models are emitted in the order requested,
  fields in declaration order, no timestamps or random suffixes. The
  generated file is checked into the repo so the React build does not
  depend on Python at install time, and a unit test re-runs the
  generator to detect drift.
* **Minimal type surface today.** Strings, numbers, booleans, lists,
  optional / union, nested models, and string-enum literal unions.
  Anything outside that throws — better to fail loudly than to emit a
  silent ``any``.
* **No defaults / no comments / no validators.** The TS file is a
  read-only mirror of the *shape*; runtime validation belongs at the
  fetch boundary (e.g. via Zod), not in this generator.

Usage
-----

::

    python -m tools.codegen.pydantic_to_ts \\
        ui.server.CredentialsStatusResponse \\
        --out dashboard2026/src/types/generated/api.ts

The CLI accepts one or more dotted import paths to Pydantic v2
``BaseModel`` subclasses. Each is emitted as an ``export interface``
(plus any nested models / enums, also exported, deduplicated).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import sys
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


@dataclasses.dataclass
class _Field:
    name: str
    ts_type: str
    optional: bool


@dataclasses.dataclass
class _Interface:
    name: str
    fields: list[_Field]


@dataclasses.dataclass
class _Enum:
    name: str
    values: list[str]


def _resolve_model(dotted: str) -> Any:
    """Import ``module.attr`` and return the attribute (a Pydantic model)."""

    module_path, _, attr = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"expected dotted path 'pkg.module.Class', got {dotted!r}")
    mod = importlib.import_module(module_path)
    try:
        return getattr(mod, attr)
    except AttributeError as exc:
        raise AttributeError(f"{module_path!r} has no attribute {attr!r}") from exc


def _schema_to_ts(
    schema: dict[str, Any],
    defs: dict[str, Any],
    interfaces: OrderedDict[str, _Interface],
    enums: OrderedDict[str, _Enum],
) -> str:
    """Translate one JSON-schema fragment to a TypeScript type string."""

    if "$ref" in schema:
        ref = schema["$ref"].rsplit("/", 1)[-1]
        target = defs[ref]
        _emit_named(ref, target, defs, interfaces, enums)
        return ref

    # String-enum: pydantic emits these as `enum: [...]` with `type: string`.
    if "enum" in schema:
        values = schema["enum"]
        if not all(isinstance(v, str) for v in values):
            raise NotImplementedError(
                f"only string enums are supported (got {values!r})"
            )
        return " | ".join(repr(v).replace("'", '"') for v in values)

    if "anyOf" in schema:
        # Pydantic encodes Optional[T] as anyOf[T, null].
        parts = [
            _schema_to_ts(s, defs, interfaces, enums) for s in schema["anyOf"]
        ]
        return " | ".join(parts)

    type_ = schema.get("type")
    if type_ == "string":
        return "string"
    if type_ in ("integer", "number"):
        return "number"
    if type_ == "boolean":
        return "boolean"
    if type_ == "null":
        return "null"
    if type_ == "array":
        item = schema.get("items", {})
        return f"{_schema_to_ts(item, defs, interfaces, enums)}[]"
    if type_ == "object":
        # Loose objects (Mapping[str, Any]) → record. The dashboard
        # avoids these in typed responses; nested BaseModels are
        # preferred and surface as $refs above.
        ap = schema.get("additionalProperties")
        if ap is True or ap is None:
            return "Record<string, unknown>"
        return f"Record<string, {_schema_to_ts(ap, defs, interfaces, enums)}>"

    raise NotImplementedError(f"unsupported JSON-schema fragment: {schema!r}")


def _emit_named(
    name: str,
    schema: dict[str, Any],
    defs: dict[str, Any],
    interfaces: OrderedDict[str, _Interface],
    enums: OrderedDict[str, _Enum],
) -> None:
    """Record one named definition (model or enum) in the output buckets."""

    if name in interfaces or name in enums:
        return

    if "enum" in schema:
        values = schema["enum"]
        if not all(isinstance(v, str) for v in values):
            raise NotImplementedError(
                f"only string enums are supported (got {values!r})"
            )
        enums[name] = _Enum(name=name, values=list(values))
        return

    if schema.get("type") != "object":
        raise NotImplementedError(
            f"unsupported named def {name}: {schema!r}"
        )

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: list[_Field] = []
    # Reserve the slot before recursing so cyclic refs don't loop.
    interfaces[name] = _Interface(name=name, fields=fields)

    for field_name, field_schema in properties.items():
        ts_type = _schema_to_ts(field_schema, defs, interfaces, enums)
        is_optional = field_name not in required or _has_null_branch(field_schema)
        fields.append(
            _Field(name=field_name, ts_type=ts_type, optional=is_optional),
        )


def _has_null_branch(schema: dict[str, Any]) -> bool:
    branches = schema.get("anyOf")
    if not branches:
        return False
    return any(b.get("type") == "null" for b in branches)


def _render(
    interfaces: OrderedDict[str, _Interface],
    enums: OrderedDict[str, _Enum],
) -> str:
    """Render the in-memory IR to a TypeScript source file string."""

    lines: list[str] = [
        "// AUTO-GENERATED by tools/codegen/pydantic_to_ts.py",
        "// Run `python -m tools.codegen.pydantic_to_ts` (see Makefile target)",
        "// to regenerate. Editing this file by hand is not allowed; CI",
        "// re-runs the generator and fails on drift.",
        "",
    ]

    for enum in enums.values():
        union = " | ".join(repr(v).replace("'", '"') for v in enum.values)
        lines.append(f"export type {enum.name} = {union};")
        lines.append("")

    for iface in interfaces.values():
        lines.append(f"export interface {iface.name} {{")
        for field in iface.fields:
            opt = "?" if field.optional else ""
            lines.append(f"  {field.name}{opt}: {field.ts_type};")
        lines.append("}")
        lines.append("")

    # Trim trailing blank lines down to exactly one.
    while len(lines) >= 2 and lines[-1] == "" and lines[-2] == "":
        lines.pop()
    if lines and lines[-1] != "":
        lines.append("")
    return "\n".join(lines)


def render_models(dotted_paths: Sequence[str]) -> str:
    """Resolve each dotted path and render a TS source string."""

    interfaces: OrderedDict[str, _Interface] = OrderedDict()
    enums: OrderedDict[str, _Enum] = OrderedDict()

    for dotted in dotted_paths:
        model = _resolve_model(dotted)
        schema = model.model_json_schema(ref_template="#/$defs/{model}")
        defs = schema.get("$defs", {})
        _emit_named(model.__name__, schema, defs, interfaces, enums)

    return _render(interfaces, enums)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="+",
        help="dotted import paths to Pydantic v2 models",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output .ts file (overwritten)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "do not write; exit non-zero if the rendered output differs "
            "from --out's contents"
        ),
    )
    args = parser.parse_args(argv)

    rendered = render_models(args.models)
    if args.check:
        existing = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if existing != rendered:
            print(
                f"codegen drift detected in {args.out}; "
                "regenerate with tools/codegen/pydantic_to_ts.py",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
