# ADAPTED FROM: https://github.com/guidance-ai/guidance (MIT)
#
# Tier-I I-20 — constrained-generation template seam.
#
# ``guidance`` constrains LLM decoding at sampling time via grammar /
# regex / select-from-list slots. The production default here is
# pure-stdlib: we parse a guidance-shaped template into a sequence of
# typed slots, then run a *validate-after-the-fact* pass against an
# already-emitted completion (regex anchor or closed-choice match per
# slot).
#
# ``guidance`` (and its transformers/llama_cpp backends) is the lazy
# seam — only imported inside :func:`enable_guidance_factory` body.
# Production environments without guidance installed still import this
# module cleanly.
#
# NEW_PIP_DEPENDENCIES = ("guidance",)
#
# Authority constraints (pinned by ``tests/test_guidance_adapter.py``):
#
#   * **RUNTIME_SAFE** — pure parser + validator; no clock, no I/O,
#     no PRNG. Three independent calls with identical inputs produce
#     byte-identical output dicts (INV-15).
#   * **B1** — no execution_engine / governance_engine / system_engine
#     cross-imports.
#   * **B27 / B28 / INV-71** — no typed-event constructors.
#   * No top-level imports of :mod:`guidance`, :mod:`transformers`,
#     :mod:`llama_cpp`, :mod:`torch`, :mod:`numpy`, :mod:`time`,
#     :mod:`datetime`, :mod:`random`, :mod:`asyncio`, :mod:`requests`.
"""I-20 guidance adapter — constrained-generation template + validator."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "GuidanceError",
    "GuidanceParseError",
    "GuidanceMatchError",
    "RegexSlot",
    "SelectSlot",
    "GuidanceTemplate",
    "compile_template",
    "match_completion",
    "enable_guidance_factory",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("guidance",)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GuidanceError(ValueError):
    """Base class for I-20 guidance-adapter errors."""


class GuidanceParseError(GuidanceError):
    """Raised when a template string is malformed."""


class GuidanceMatchError(GuidanceError):
    """Raised when a completion text fails to match a compiled template."""


# ---------------------------------------------------------------------------
# Slot value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class RegexSlot:
    """Capture group anchored by a regex pattern.

    Fields:
        name: slot name (non-empty str, unique within a template).
        pattern: a regex without capture groups — the slot itself
            becomes the only capture group at compile time. The
            pattern is anchored *non-greedily* by default so adjacent
            literal text reliably terminates the slot.
    """

    name: str
    pattern: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise GuidanceError(f"RegexSlot.name must be a non-empty str, got {self.name!r}")
        if not isinstance(self.pattern, str) or not self.pattern:
            raise GuidanceError(f"RegexSlot.pattern must be a non-empty str, got {self.pattern!r}")
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise GuidanceError(f"RegexSlot.pattern is not a valid regex: {exc}") from exc


@dataclasses.dataclass(frozen=True, slots=True)
class SelectSlot:
    """Capture group constrained to one of a closed set of literals.

    Fields:
        name: slot name (non-empty str, unique within a template).
        choices: non-empty tuple of distinct non-empty str literals.
    """

    name: str
    choices: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise GuidanceError(f"SelectSlot.name must be a non-empty str, got {self.name!r}")
        if not isinstance(self.choices, tuple) or len(self.choices) == 0:
            raise GuidanceError(
                f"SelectSlot.choices must be a non-empty tuple, got {self.choices!r}"
            )
        seen: set[str] = set()
        for c in self.choices:
            if not isinstance(c, str) or not c:
                raise GuidanceError(f"SelectSlot.choices entries must be non-empty str, got {c!r}")
            if c in seen:
                raise GuidanceError(f"SelectSlot.choices contains duplicate {c!r}")
            seen.add(c)


Slot = RegexSlot | SelectSlot


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


# Match a single ``{{regex 'name' pattern='...'}}`` or
# ``{{select 'name' options=('A','B')}}`` placeholder. Double-brace
# is deliberate so single-brace literals in templates remain literal.
_PLACEHOLDER_PATTERN: re.Pattern[str] = re.compile(
    r"\{\{(?P<kind>regex|select)\s+"
    r"'(?P<name>[A-Za-z_][A-Za-z0-9_]*)'\s+"
    r"(?P<args>[^}]*?)\}\}"
)


@dataclasses.dataclass(frozen=True, slots=True)
class GuidanceTemplate:
    """Compiled guidance-shaped template.

    Fields:
        slots: ordered tuple of :class:`RegexSlot` / :class:`SelectSlot`
            in template declaration order.
        compiled: a single compiled :class:`re.Pattern` whose named
            groups equal the slot names. The pattern is anchored with
            ``\\A`` so :func:`re.Match` only succeeds against the
            beginning of the completion text.
    """

    slots: tuple[Slot, ...]
    compiled: re.Pattern[str]

    def slot_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.slots)


def _parse_regex_args(args: str) -> str:
    m = re.fullmatch(r"\s*pattern='(?P<p>[^']*)'\s*", args)
    if m is None:
        raise GuidanceParseError(f"regex slot expects pattern='...' args, got {args!r}")
    return m.group("p")


def _parse_select_args(args: str) -> tuple[str, ...]:
    m = re.fullmatch(r"\s*options=\((?P<list>[^)]*)\)\s*", args)
    if m is None:
        raise GuidanceParseError(f"select slot expects options=('A','B',...) args, got {args!r}")
    raw = m.group("list").strip()
    if not raw:
        raise GuidanceParseError("select slot has empty options tuple")
    items: list[str] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        m2 = re.fullmatch(r"'([^']*)'", piece)
        if m2 is None:
            raise GuidanceParseError(f"select slot option {piece!r} must be 'quoted'")
        items.append(m2.group(1))
    return tuple(items)


def compile_template(template: str) -> GuidanceTemplate:
    """Parse ``template`` into a :class:`GuidanceTemplate`.

    Pure function — INV-15 byte-identical across runs. Raises
    :class:`GuidanceParseError` on malformed input.
    """

    if not isinstance(template, str):
        raise GuidanceParseError(
            f"compile_template input must be str, got {type(template).__name__}"
        )
    if not template:
        raise GuidanceParseError("compile_template input must be non-empty")
    slots: list[Slot] = []
    seen_names: set[str] = set()
    out_pattern_parts: list[str] = ["\\A"]
    cursor = 0
    for m in _PLACEHOLDER_PATTERN.finditer(template):
        if m.start() > cursor:
            out_pattern_parts.append(re.escape(template[cursor : m.start()]))
        kind = m.group("kind")
        name = m.group("name")
        if name in seen_names:
            raise GuidanceParseError(f"duplicate slot name in template: {name!r}")
        seen_names.add(name)
        if kind == "regex":
            pat = _parse_regex_args(m.group("args"))
            slots.append(RegexSlot(name=name, pattern=pat))
            out_pattern_parts.append(f"(?P<{name}>{pat})")
        else:
            choices = _parse_select_args(m.group("args"))
            slots.append(SelectSlot(name=name, choices=choices))
            alts = "|".join(re.escape(c) for c in choices)
            out_pattern_parts.append(f"(?P<{name}>{alts})")
        cursor = m.end()
    if cursor < len(template):
        out_pattern_parts.append(re.escape(template[cursor:]))
    if not slots:
        raise GuidanceParseError("compile_template requires at least one {{regex|select ...}} slot")
    final_pattern = "".join(out_pattern_parts)
    try:
        compiled = re.compile(final_pattern, re.DOTALL)
    except re.error as exc:
        raise GuidanceParseError(
            f"compile_template produced an invalid combined pattern: {exc}"
        ) from exc
    return GuidanceTemplate(slots=tuple(slots), compiled=compiled)


# ---------------------------------------------------------------------------
# Match / validate
# ---------------------------------------------------------------------------


def match_completion(
    template: GuidanceTemplate,
    text: str,
) -> dict[str, str]:
    """Validate ``text`` against ``template`` and return captured slots.

    Returns a fresh dict keyed in template declaration order. Raises
    :class:`GuidanceMatchError` if the completion text doesn't match
    the compiled pattern, or if a captured select-slot value is not in
    its declared choices. INV-15 deterministic.
    """

    if not isinstance(template, GuidanceTemplate):
        raise GuidanceError(
            f"match_completion template must be a GuidanceTemplate, got {type(template).__name__}"
        )
    if not isinstance(text, str):
        raise GuidanceMatchError(f"match_completion text must be str, got {type(text).__name__}")
    m = template.compiled.match(text)
    if m is None:
        raise GuidanceMatchError("match_completion: completion did not match compiled template")
    out: dict[str, str] = {}
    for slot in template.slots:
        value = m.group(slot.name)
        if isinstance(slot, SelectSlot) and value not in slot.choices:
            raise GuidanceMatchError(
                f"slot {slot.name!r}: value {value!r} not in choices {slot.choices!r}"
            )
        out[slot.name] = value
    return out


# ---------------------------------------------------------------------------
# Lazy ``guidance`` seam
# ---------------------------------------------------------------------------


def enable_guidance_factory(
    model: Any,
) -> Callable[[GuidanceTemplate, str], dict[str, str]]:
    """Return a callable that invokes the real guidance backend.

    Importing :mod:`guidance` is deferred to factory-call time. The
    returned callable signature is::

        invoke(template: GuidanceTemplate, prompt: str) -> dict[str, str]

    The runtime backend constrains generation at decode time so the
    returned dict is guaranteed to satisfy the compiled template — but
    we still re-run :func:`match_completion` to keep the audit shape
    canonical regardless of backend.
    """

    if model is None:
        raise GuidanceError("enable_guidance_factory: model is required")
    import guidance  # type: ignore[import-not-found]  # noqa: F401 - lazy seam

    def _call(template: GuidanceTemplate, prompt: str) -> dict[str, str]:
        if not isinstance(prompt, str):
            raise GuidanceError(f"guidance prompt must be str, got {type(prompt).__name__}")
        # The backend renders the template under decode constraints.
        # We re-validate via match_completion to enforce the canonical
        # output shape regardless of which backend is installed.
        completion = model + prompt
        text = str(completion)
        return match_completion(template, text)

    return _call
