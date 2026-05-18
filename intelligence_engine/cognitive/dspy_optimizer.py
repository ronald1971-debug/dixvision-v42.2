# ADAPTED FROM: stanfordnlp/dspy
# (dspy/signatures/signature.py — typed input/output Signature contract;
#  dspy/predict/chain_of_thought.py — ChainOfThought rationale prefix;
#  dspy/teleprompt/bootstrap_fewshot.py — BootstrapFewShot demo selection;
#  dspy/evaluate/evaluate.py — metric-driven pass/fail filter.)
"""A-06 — DSPy-style programmatic prompt optimization for DIX.

DSPy's ``Signature`` / ``Predict`` / ``ChainOfThought`` /
``BootstrapFewShot`` quartet builds prompts as a typed contract and
then *optimizes* the contract by sampling few-shot demonstrations
that pass an offline metric. We adapt that exact pattern behind DIX
contracts:

1. :class:`Signature` declares the typed input + output field schema
   for one prompt — frozen, no Pydantic-class-as-schema magic.
   Field types are restricted to the four DIX-allowable JSON-shape
   primitives (``str``, ``int``, ``float``, ``bool``) so the
   on-disk artifact is byte-stable across hosts and Python versions.
2. :class:`Predictor` is a small ``Protocol``: ``predict(signature,
   inputs, demonstrations) -> Prediction``. Production wiring binds
   it to :class:`LiteLLMPredictor` which delegates to the existing
   :class:`~intelligence_engine.cognitive.litellm_router.LiteLLMRouter`
   (S-12). No direct API keys; no dspy.LM. Tests inject a fake.
3. :class:`BootstrapFewShot` is an **offline** teleprompter: it runs
   the predictor over a training set, keeps every example that
   passes a caller-supplied :data:`MetricFn`, and emits a frozen
   :class:`OptimizedProgram` carrying that demonstration tuple.
   The optimizer never touches runtime; it produces a value object
   for storage under ``registry/cognitive/`` (operator-owned).
4. :class:`OptimizedProgram` carries the frozen
   :class:`Signature` + selected demonstrations and exposes
   :meth:`predict` for runtime inference. Inference uses the
   optimized prompt only — it cannot re-bootstrap, mutate the
   signature, or change the demonstration tuple. INV-15 byte-stable
   serialization via :func:`serialize_program` /
   :func:`deserialize_program` (sorted-key JSON, monotonic field
   order from the signature).

Tier discipline
---------------

* **OFFLINE_ONLY** for :class:`BootstrapFewShot`: mutation surface,
  never on the hot path. Caller is responsible for shipping the
  resulting :class:`OptimizedProgram` into ``registry/cognitive/``.
* **RUNTIME_SAFE** for :class:`OptimizedProgram.predict`: pure
  string ``Predictor`` call against frozen demos, no further
  mutation, no clock, no PRNG.
* B27 / B28 / INV-71 authority symmetry: this module does **not**
  construct :class:`PatchProposal` / :class:`SignalEvent` /
  :class:`GovernanceDecision`. The optimizer's output is advisory
  text; promotion to typed events happens through the existing
  :class:`~intelligence_engine.cognitive.typed_ai.TypedAIAgent`
  (S-06) → governance approval queue (PR #87, INV-72) outside this
  module.
* B1 cross-engine isolation: the module imports nothing from
  ``governance_engine`` / ``system_engine`` / ``execution_engine``
  / ``evolution_engine``.
* INV-15 determinism: no ``random`` import, no ``time`` /
  ``datetime`` / wall clock. ``Demonstration`` selection order is
  preserved from the trainset (no shuffling). Serialization is
  sorted-key JSON.
* No top-level ``dspy`` import. The module imports cleanly without
  the ``dspy-ai`` package; the lazy adapter
  :func:`dspy_predictor_factory` constructs a dspy-backed
  predictor on demand.

What survives from upstream
---------------------------

* The ``Signature`` typed-field idea: every prompt is a contract,
  not a string. Inputs and outputs are named fields with descriptions.
* The ``ChainOfThought`` prefix-with-rationale shape: an extra
  output field ``rationale`` is rendered first so the LLM "thinks
  out loud" before the structured outputs.
* The ``BootstrapFewShot`` demo-selection loop:
  - run predictor on each example,
  - apply metric,
  - keep up to ``max_bootstrapped_demos`` passing examples,
  - skip examples whose prediction misses required output fields.

What is rewritten behind DIX contracts
--------------------------------------

* DSPy's global ``dspy.settings.lm`` mutable singleton is replaced
  with an injected :class:`Predictor` (no global state — INV-15).
* DSPy's ``dspy.LM`` wraps litellm directly; we route through the
  existing :class:`LiteLLMRouter` so cost auditing, fallback chain,
  and provider resolution are reused.
* DSPy's optimizer mutates the program object in-place; ours
  returns a *new* frozen :class:`OptimizedProgram` so the
  pre-compile and post-compile artifacts can both be stored.

.. _dspy: https://github.com/stanfordnlp/dspy
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("dspy-ai",)
"""A-06 introduces a runtime-optional dependency on dspy-ai.

The package is **only** required if the operator wires
:func:`dspy_predictor_factory` as the production predictor. Test
deployments and any host that exclusively uses an injected fake
predictor (or :class:`LiteLLMPredictor`) do not need dspy-ai
installed; the module imports cleanly without it.
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "ALLOWED_FIELD_TYPES",
    "BootstrapFewShot",
    "DSPY_OPTIMIZER_VERSION",
    "Demonstration",
    "EmptyTrainsetError",
    "Example",
    "Field",
    "FieldKind",
    "MAX_BOOTSTRAPPED_DEMOS",
    "MAX_DEMONSTRATIONS",
    "MAX_DESCRIPTION_LEN",
    "MAX_FIELD_NAME_LEN",
    "MAX_INSTRUCTION_LEN",
    "MAX_INPUT_FIELDS",
    "MAX_OUTPUT_FIELDS",
    "MAX_PROMPT_LEN",
    "MAX_TRAINSET_LEN",
    "MAX_VALUE_LEN",
    "MIN_OUTPUT_FIELDS",
    "MetricFn",
    "NEW_PIP_DEPENDENCIES",
    "NoPassingExamplesError",
    "OptimizedProgram",
    "Prediction",
    "Predictor",
    "Signature",
    "build_chain_of_thought_signature",
    "build_governance_proposal_signature",
    "deserialize_program",
    "dspy_predictor_factory",
    "render_prompt",
    "serialize_program",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DSPY_OPTIMIZER_VERSION: str = "1"
"""Schema version stamped into every serialized
:class:`OptimizedProgram`. Bump on any breaking change to the
on-disk shape so loaders refuse stale artifacts."""

MAX_FIELD_NAME_LEN: int = 64
MAX_DESCRIPTION_LEN: int = 512
MAX_INSTRUCTION_LEN: int = 2048
MAX_VALUE_LEN: int = 8192
MAX_PROMPT_LEN: int = 32768
MAX_INPUT_FIELDS: int = 16
MAX_OUTPUT_FIELDS: int = 16
MIN_OUTPUT_FIELDS: int = 1
MAX_DEMONSTRATIONS: int = 32
MAX_BOOTSTRAPPED_DEMOS: int = 16
MAX_TRAINSET_LEN: int = 1024


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EmptyTrainsetError(ValueError):
    """Raised by :class:`BootstrapFewShot` when ``trainset`` is empty."""


class NoPassingExamplesError(RuntimeError):
    """Raised by :class:`BootstrapFewShot` when no example passes the
    metric.

    The optimizer refuses to emit an empty :class:`OptimizedProgram`
    because that would silently regress runtime to the zero-shot
    baseline.
    """


# ---------------------------------------------------------------------------
# Field kinds + allowed primitive types
# ---------------------------------------------------------------------------


class FieldKind:
    """String-enum-shaped field role.

    Implemented as a class with class attributes (rather than
    :class:`enum.StrEnum`) so the values land in the serialized
    artifact unchanged and tests can compare directly to the string
    constants without unwrapping enums.
    """

    INPUT: str = "input"
    OUTPUT: str = "output"


_VALID_FIELD_KINDS: frozenset[str] = frozenset({FieldKind.INPUT, FieldKind.OUTPUT})

ALLOWED_FIELD_TYPES: tuple[type, ...] = (str, int, float, bool)
"""The four JSON-shape primitives allowed for
:class:`Field.field_type`. Restricting the type space keeps the
serialized artifact byte-stable across hosts and Python versions
(``bool`` is checked *before* ``int`` because ``isinstance(True,
int)`` is ``True`` in Python)."""

_TYPE_NAMES: Mapping[type, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}

_NAME_TO_TYPE: Mapping[str, type] = {v: k for k, v in _TYPE_NAMES.items()}


def _ensure_value_matches_type(name: str, value: Any, field_type: type) -> None:
    """Validate ``value`` against the four allowed primitive types.

    ``bool`` is checked before ``int``/``float`` to avoid the Python
    quirk where ``isinstance(True, int)`` is true.
    """

    if field_type is bool:
        if not isinstance(value, bool):
            raise TypeError(f"field {name!r}: expected bool, got {type(value).__name__}")
        return
    # Reject bool when caller asked for int/float/str — Python treats
    # bool as a subclass of int, but that is rarely what the operator
    # actually wants in a typed signature.
    if isinstance(value, bool):
        raise TypeError(f"field {name!r}: expected {field_type.__name__}, got bool")
    if field_type is int:
        if not isinstance(value, int):
            raise TypeError(f"field {name!r}: expected int, got {type(value).__name__}")
        return
    if field_type is float:
        if not isinstance(value, (int, float)):
            raise TypeError(f"field {name!r}: expected float, got {type(value).__name__}")
        return
    if field_type is str:
        if not isinstance(value, str):
            raise TypeError(f"field {name!r}: expected str, got {type(value).__name__}")
        if len(value) > MAX_VALUE_LEN:
            raise ValueError(
                f"field {name!r}: str length {len(value)} exceeds MAX_VALUE_LEN={MAX_VALUE_LEN}"
            )
        return
    raise TypeError(  # pragma: no cover — defensive
        f"field {name!r}: unsupported field_type {field_type!r}"
    )


# ---------------------------------------------------------------------------
# Field + Signature
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Field:
    """Typed input or output field on a :class:`Signature`.

    Fields:

    * ``name`` — non-empty alpha-numeric+underscore string,
      ``len <= MAX_FIELD_NAME_LEN``. Must not start with a digit.
    * ``description`` — operator-facing docstring; rendered into the
      prompt verbatim. ``len <= MAX_DESCRIPTION_LEN``.
    * ``field_type`` — one of :data:`ALLOWED_FIELD_TYPES`.
    * ``kind`` — :attr:`FieldKind.INPUT` or :attr:`FieldKind.OUTPUT`.
    """

    name: str
    description: str
    field_type: type
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Field.name must be str")
        if not self.name:
            raise ValueError("Field.name must be non-empty")
        if len(self.name) > MAX_FIELD_NAME_LEN:
            raise ValueError(
                f"Field.name length {len(self.name)} exceeds"
                f" MAX_FIELD_NAME_LEN={MAX_FIELD_NAME_LEN}"
            )
        if not (self.name[0].isalpha() or self.name[0] == "_"):
            raise ValueError(
                f"Field.name must start with a letter or underscore; got {self.name!r}"
            )
        for ch in self.name:
            if not (ch.isalnum() or ch == "_"):
                raise ValueError(f"Field.name must be alphanumeric+underscore; got {self.name!r}")
        if not isinstance(self.description, str):
            raise TypeError("Field.description must be str")
        if len(self.description) > MAX_DESCRIPTION_LEN:
            raise ValueError(
                f"Field.description length {len(self.description)} exceeds"
                f" MAX_DESCRIPTION_LEN={MAX_DESCRIPTION_LEN}"
            )
        if self.field_type not in ALLOWED_FIELD_TYPES:
            raise TypeError(
                f"Field.field_type must be one of"
                f" {[t.__name__ for t in ALLOWED_FIELD_TYPES]};"
                f" got {self.field_type!r}"
            )
        if not isinstance(self.kind, str):
            raise TypeError("Field.kind must be str")
        if self.kind not in _VALID_FIELD_KINDS:
            raise ValueError(
                f"Field.kind must be one of {sorted(_VALID_FIELD_KINDS)!r}; got {self.kind!r}"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Signature:
    """Typed input/output schema for one prompt.

    Fields:

    * ``name`` — fully-qualified-ish identifier for the signature
      (e.g. ``"governance_proposal/v1"``). Stored verbatim in the
      serialized artifact.
    * ``instruction`` — operator-facing instruction text rendered at
      the top of the prompt.
    * ``input_fields`` — non-empty tuple of :class:`Field` with
      :attr:`FieldKind.INPUT` kind.
    * ``output_fields`` — non-empty tuple of :class:`Field` with
      :attr:`FieldKind.OUTPUT` kind.

    Field names must be unique across both the input and output
    tuples; rendered prompts use ``"{name}: {value}"`` so collisions
    would silently overwrite.
    """

    name: str
    instruction: str
    input_fields: tuple[Field, ...]
    output_fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("Signature.name must be str")
        if not self.name:
            raise ValueError("Signature.name must be non-empty")
        if len(self.name) > MAX_FIELD_NAME_LEN:
            raise ValueError(
                f"Signature.name length {len(self.name)} exceeds"
                f" MAX_FIELD_NAME_LEN={MAX_FIELD_NAME_LEN}"
            )
        if not isinstance(self.instruction, str):
            raise TypeError("Signature.instruction must be str")
        if len(self.instruction) > MAX_INSTRUCTION_LEN:
            raise ValueError(
                f"Signature.instruction length {len(self.instruction)} exceeds"
                f" MAX_INSTRUCTION_LEN={MAX_INSTRUCTION_LEN}"
            )
        if not isinstance(self.input_fields, tuple):
            raise TypeError("Signature.input_fields must be a tuple")
        if not self.input_fields:
            raise ValueError("Signature.input_fields must be non-empty")
        if len(self.input_fields) > MAX_INPUT_FIELDS:
            raise ValueError(
                f"Signature.input_fields count {len(self.input_fields)} exceeds"
                f" MAX_INPUT_FIELDS={MAX_INPUT_FIELDS}"
            )
        if not isinstance(self.output_fields, tuple):
            raise TypeError("Signature.output_fields must be a tuple")
        if len(self.output_fields) < MIN_OUTPUT_FIELDS:
            raise ValueError(
                f"Signature.output_fields must have at least {MIN_OUTPUT_FIELDS} entry"
            )
        if len(self.output_fields) > MAX_OUTPUT_FIELDS:
            raise ValueError(
                f"Signature.output_fields count {len(self.output_fields)}"
                f" exceeds MAX_OUTPUT_FIELDS={MAX_OUTPUT_FIELDS}"
            )
        seen: set[str] = set()
        for i, f in enumerate(self.input_fields):
            if not isinstance(f, Field):
                raise TypeError(f"Signature.input_fields[{i}] must be a Field")
            if f.kind != FieldKind.INPUT:
                raise ValueError(f"Signature.input_fields[{i}].kind must be INPUT; got {f.kind!r}")
            if f.name in seen:
                raise ValueError(f"Signature: duplicate field name {f.name!r}")
            seen.add(f.name)
        for i, f in enumerate(self.output_fields):
            if not isinstance(f, Field):
                raise TypeError(f"Signature.output_fields[{i}] must be a Field")
            if f.kind != FieldKind.OUTPUT:
                raise ValueError(
                    f"Signature.output_fields[{i}].kind must be OUTPUT; got {f.kind!r}"
                )
            if f.name in seen:
                raise ValueError(f"Signature: duplicate field name {f.name!r}")
            seen.add(f.name)

    def input_names(self) -> tuple[str, ...]:
        """Names of input fields in declaration order."""

        return tuple(f.name for f in self.input_fields)

    def output_names(self) -> tuple[str, ...]:
        """Names of output fields in declaration order."""

        return tuple(f.name for f in self.output_fields)

    def field_for(self, name: str) -> Field:
        """Return the :class:`Field` with the given name (input or
        output). Raises ``KeyError`` if absent.
        """

        for f in self.input_fields:
            if f.name == name:
                return f
        for f in self.output_fields:
            if f.name == name:
                return f
        raise KeyError(name)


# ---------------------------------------------------------------------------
# Example + Prediction + Demonstration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Example:
    """One row of the trainset.

    ``inputs`` carries the values for every input field on the
    signature (order-independent). ``outputs`` carries the
    *ground-truth* values for every output field — these are what
    the metric compares the predictor against. Both maps are stored
    as a sorted-key tuple so the dataclass is itself hashable and
    structurally comparable; see :meth:`as_inputs` /
    :meth:`as_outputs` for ``dict`` views.
    """

    inputs: tuple[tuple[str, Any], ...]
    outputs: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        for fname in ("inputs", "outputs"):
            v = getattr(self, fname)
            if not isinstance(v, tuple):
                raise TypeError(f"Example.{fname} must be a tuple")
            seen: set[str] = set()
            for i, entry in enumerate(v):
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise TypeError(f"Example.{fname}[{i}] must be a (name, value) tuple")
                key, _ = entry
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Example.{fname}[{i}] key must be non-empty str")
                if key in seen:
                    raise ValueError(f"Example.{fname}: duplicate key {key!r}")
                seen.add(key)

    @staticmethod
    def from_dicts(inputs: Mapping[str, Any], outputs: Mapping[str, Any]) -> Example:
        """Build a frozen :class:`Example` from plain dict views.

        Keys are sorted ascending so the resulting tuple is
        order-independent and the dataclass hashes deterministically.
        """

        return Example(
            inputs=tuple(sorted(inputs.items())),
            outputs=tuple(sorted(outputs.items())),
        )

    def as_inputs(self) -> dict[str, Any]:
        """Return the inputs as a fresh ``dict`` (caller may mutate)."""

        return dict(self.inputs)

    def as_outputs(self) -> dict[str, Any]:
        """Return the outputs as a fresh ``dict`` (caller may mutate)."""

        return dict(self.outputs)


@dataclasses.dataclass(frozen=True, slots=True)
class Prediction:
    """One predictor output for a single example.

    Fields:

    * ``outputs`` — sorted-key tuple of (name, value) pairs covering
      every signature output field. Missing or extra fields cause
      :meth:`OptimizedProgram.predict` to raise ``ValueError``.
    * ``provider_id`` — id of the provider that produced the
      prediction (recorded for audit). May be empty for fake
      predictors.
    """

    outputs: tuple[tuple[str, Any], ...]
    provider_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.outputs, tuple):
            raise TypeError("Prediction.outputs must be a tuple")
        seen: set[str] = set()
        for i, entry in enumerate(self.outputs):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(f"Prediction.outputs[{i}] must be a (name, value) tuple")
            key, _ = entry
            if not isinstance(key, str) or not key:
                raise ValueError(f"Prediction.outputs[{i}] key must be non-empty str")
            if key in seen:
                raise ValueError(f"Prediction.outputs: duplicate key {key!r}")
            seen.add(key)
        if not isinstance(self.provider_id, str):
            raise TypeError("Prediction.provider_id must be str")

    def as_dict(self) -> dict[str, Any]:
        """Return the outputs as a fresh ``dict`` (caller may mutate)."""

        return dict(self.outputs)


@dataclasses.dataclass(frozen=True, slots=True)
class Demonstration:
    """Frozen (inputs, outputs) pair selected by
    :class:`BootstrapFewShot`.

    Demonstrations carry the *predicted* outputs (not the ground
    truth) because that is what dspy bootstraps: it shows the LLM
    examples of *its own* successful reasoning chain so the few-shot
    prompt is self-consistent. The ground-truth outputs from
    :class:`Example.outputs` are used only to compute the metric.
    """

    inputs: tuple[tuple[str, Any], ...]
    outputs: tuple[tuple[str, Any], ...]

    def __post_init__(self) -> None:
        for fname in ("inputs", "outputs"):
            v = getattr(self, fname)
            if not isinstance(v, tuple):
                raise TypeError(f"Demonstration.{fname} must be a tuple")
            seen: set[str] = set()
            for i, entry in enumerate(v):
                if not isinstance(entry, tuple) or len(entry) != 2:
                    raise TypeError(f"Demonstration.{fname}[{i}] must be a (name, value) tuple")
                key, _ = entry
                if not isinstance(key, str) or not key:
                    raise ValueError(f"Demonstration.{fname}[{i}] key must be non-empty str")
                if key in seen:
                    raise ValueError(f"Demonstration.{fname}: duplicate key {key!r}")
                seen.add(key)


# ---------------------------------------------------------------------------
# Predictor protocol + LiteLLM-backed concrete impl
# ---------------------------------------------------------------------------


@runtime_checkable
class Predictor(Protocol):
    """Per-call dispatch for one rendered prompt.

    Implementations may shell out to :class:`LiteLLMRouter`,
    a dspy ``Predict`` module, or a local stub. The predictor is
    responsible for:

    * rendering the prompt from the signature + demonstrations +
      inputs (use :func:`render_prompt`);
    * sending it to the underlying LLM;
    * parsing the response back into a :class:`Prediction`.

    The predictor is **not** responsible for selection, ordering,
    fallback, or cost auditing — those live in the
    :class:`LiteLLMRouter` it wraps.
    """

    def predict(
        self,
        signature: Signature,
        inputs: Mapping[str, Any],
        demonstrations: Sequence[Demonstration],
        /,
    ) -> Prediction:
        """Run one prediction and return a :class:`Prediction`."""


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


_RATIONALE_FIELD_NAME: str = "rationale"


def render_prompt(
    signature: Signature,
    inputs: Mapping[str, Any],
    demonstrations: Sequence[Demonstration],
) -> str:
    """Render the dspy-shape prompt for one prediction.

    Layout (mirrors ``dspy/predict/predict.py::Predict.forward``)::

        <instruction>

        Input fields:
        - <name> (<type>): <description>

        Output fields:
        - <name> (<type>): <description>

        ---

        Example 1:
        <name>: <value>
        ...
        ===
        <output_name>: <value>

        ...

        Now:
        <name>: <value>

    Demonstrations are rendered in the order they appear in the
    sequence. Inputs are rendered in :attr:`Signature.input_fields`
    declaration order — never in dict-iteration order — so the
    rendered prompt is INV-15 byte-stable regardless of caller dict
    insertion order.

    Raises:
        ValueError: ``inputs`` is missing a required field, has an
            extra unknown field, or holds a value whose type does not
            match the field's declared type.
    """

    if not isinstance(signature, Signature):
        raise TypeError("render_prompt: signature must be a Signature")
    if not isinstance(inputs, Mapping):
        raise TypeError("render_prompt: inputs must be a Mapping")
    if not isinstance(demonstrations, Sequence) or isinstance(demonstrations, (str, bytes)):
        raise TypeError("render_prompt: demonstrations must be a Sequence")

    expected_inputs = signature.input_names()
    inputs_set = set(inputs.keys())
    missing = set(expected_inputs) - inputs_set
    if missing:
        raise ValueError(f"render_prompt: missing input fields: {sorted(missing)!r}")
    extra = inputs_set - set(expected_inputs)
    if extra:
        raise ValueError(f"render_prompt: unknown input fields: {sorted(extra)!r}")
    for f in signature.input_fields:
        _ensure_value_matches_type(f.name, inputs[f.name], f.field_type)

    parts: list[str] = []
    if signature.instruction:
        parts.append(signature.instruction)
        parts.append("")

    parts.append("Input fields:")
    for f in signature.input_fields:
        parts.append(f"- {f.name} ({_TYPE_NAMES[f.field_type]}): {f.description}")
    parts.append("")
    parts.append("Output fields:")
    for f in signature.output_fields:
        parts.append(f"- {f.name} ({_TYPE_NAMES[f.field_type]}): {f.description}")
    parts.append("")

    for i, demo in enumerate(demonstrations):
        if not isinstance(demo, Demonstration):
            raise TypeError(f"render_prompt: demonstrations[{i}] must be a Demonstration")
        parts.append("---")
        parts.append(f"Example {i + 1}:")
        demo_inputs = dict(demo.inputs)
        demo_outputs = dict(demo.outputs)
        for f in signature.input_fields:
            if f.name not in demo_inputs:
                raise ValueError(
                    f"render_prompt: demonstrations[{i}] missing input field {f.name!r}"
                )
            parts.append(f"{f.name}: {demo_inputs[f.name]}")
        parts.append("===")
        for f in signature.output_fields:
            if f.name not in demo_outputs:
                raise ValueError(
                    f"render_prompt: demonstrations[{i}] missing output field {f.name!r}"
                )
            parts.append(f"{f.name}: {demo_outputs[f.name]}")

    parts.append("---")
    parts.append("Now:")
    for f in signature.input_fields:
        parts.append(f"{f.name}: {inputs[f.name]}")

    rendered = "\n".join(parts)
    if len(rendered) > MAX_PROMPT_LEN:
        raise ValueError(
            f"render_prompt: rendered length {len(rendered)} exceeds"
            f" MAX_PROMPT_LEN={MAX_PROMPT_LEN}"
        )
    return rendered


# ---------------------------------------------------------------------------
# BootstrapFewShot
# ---------------------------------------------------------------------------


MetricFn = Callable[[Example, Prediction], bool]
"""Pass/fail metric for one (example, prediction) pair.

Returning ``True`` keeps the example as a demonstration; returning
``False`` discards it. The optimizer never sees raw scores — that
threshold is the caller's responsibility, embedded in this
callable. INV-15: the metric must be deterministic and free of
clock / PRNG / IO."""


@dataclasses.dataclass(frozen=True, slots=True)
class BootstrapFewShot:
    """Offline teleprompter: select the best demos via a pass/fail
    metric.

    Mirrors ``dspy/teleprompt/bootstrap_fewshot.py::BootstrapFewShot``
    in shape but is a *pure function* with no global state.

    Fields:

    * ``max_bootstrapped_demos`` — upper bound on the number of
      demonstrations selected. Must be in ``[1,
      MAX_BOOTSTRAPPED_DEMOS]``.

    Method:

    * :meth:`compile` — run predictor on each example, keep up to
      ``max_bootstrapped_demos`` passing examples in trainset order,
      return a frozen :class:`OptimizedProgram`.

    The optimizer is OFFLINE_ONLY: it iterates the trainset
    sequentially, never reads the wall clock, never imports
    ``random``, and never mutates global state. The selection order
    is preserved from the trainset (no shuffling).
    """

    max_bootstrapped_demos: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.max_bootstrapped_demos, bool) or not isinstance(
            self.max_bootstrapped_demos, int
        ):
            raise TypeError("BootstrapFewShot.max_bootstrapped_demos must be int")
        if self.max_bootstrapped_demos < 1:
            raise ValueError("BootstrapFewShot.max_bootstrapped_demos must be >= 1")
        if self.max_bootstrapped_demos > MAX_BOOTSTRAPPED_DEMOS:
            raise ValueError(
                "BootstrapFewShot.max_bootstrapped_demos must be <="
                f" {MAX_BOOTSTRAPPED_DEMOS};"
                f" got {self.max_bootstrapped_demos}"
            )

    def compile(
        self,
        signature: Signature,
        trainset: Sequence[Example],
        metric: MetricFn,
        predictor: Predictor,
    ) -> OptimizedProgram:
        """Run the predictor across ``trainset``, keep passing examples,
        and return a frozen :class:`OptimizedProgram`.

        Args:
            signature: The :class:`Signature` to optimize.
            trainset: Non-empty sequence of :class:`Example` rows.
            metric: Callable that receives ``(example, prediction)``
                and returns ``True`` to keep the example as a demo.
            predictor: :class:`Predictor` to run during optimization.

        Returns:
            :class:`OptimizedProgram` carrying the signature + the
            tuple of demonstrations in trainset order.

        Raises:
            EmptyTrainsetError: ``trainset`` is empty.
            NoPassingExamplesError: No example passed the metric.
        """

        if not isinstance(signature, Signature):
            raise TypeError("compile: signature must be a Signature")
        if not isinstance(trainset, Sequence) or isinstance(trainset, (str, bytes)):
            raise TypeError("compile: trainset must be a Sequence")
        if not trainset:
            raise EmptyTrainsetError("BootstrapFewShot.compile: trainset must be non-empty")
        if len(trainset) > MAX_TRAINSET_LEN:
            raise ValueError(
                f"BootstrapFewShot.compile: trainset length {len(trainset)}"
                f" exceeds MAX_TRAINSET_LEN={MAX_TRAINSET_LEN}"
            )
        if not callable(metric):
            raise TypeError("compile: metric must be callable")
        if not isinstance(predictor, Predictor):
            raise TypeError("compile: predictor must implement the Predictor Protocol")

        expected_outputs = set(signature.output_names())
        demos: list[Demonstration] = []
        for i, example in enumerate(trainset):
            if not isinstance(example, Example):
                raise TypeError(f"compile: trainset[{i}] must be an Example")
            inputs_view = example.as_inputs()
            prediction = predictor.predict(signature, inputs_view, ())
            if not isinstance(prediction, Prediction):
                raise TypeError(
                    f"compile: predictor returned {type(prediction).__name__}; expected Prediction"
                )
            pred_outputs = dict(prediction.outputs)
            # Skip examples that miss any required output field.
            if set(pred_outputs.keys()) != expected_outputs:
                continue
            try:
                ok = bool(metric(example, prediction))
            except Exception as exc:  # noqa: BLE001 — propagate as TypeError
                raise TypeError(f"compile: metric raised on trainset[{i}]: {exc!r}") from exc
            if not ok:
                continue
            demos.append(
                Demonstration(
                    inputs=tuple(sorted(inputs_view.items())),
                    outputs=tuple(sorted(pred_outputs.items())),
                )
            )
            if len(demos) >= self.max_bootstrapped_demos:
                break

        if not demos:
            raise NoPassingExamplesError(
                "BootstrapFewShot.compile: no example passed the metric;"
                " refusing to emit an empty OptimizedProgram"
            )

        return OptimizedProgram(
            signature=signature,
            demonstrations=tuple(demos),
        )


# ---------------------------------------------------------------------------
# OptimizedProgram (runtime artifact)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class OptimizedProgram:
    """Frozen runtime artifact emitted by :class:`BootstrapFewShot`.

    Carries the original :class:`Signature` plus the tuple of
    selected :class:`Demonstration` rows. Use :meth:`predict` to run
    the program at runtime — that path uses the optimized prompt
    only, never re-bootstraps, never mutates the demonstration
    tuple.

    Use :func:`serialize_program` and :func:`deserialize_program` to
    persist the artifact under ``registry/cognitive/`` and reload it
    on boot.
    """

    signature: Signature
    demonstrations: tuple[Demonstration, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.signature, Signature):
            raise TypeError("OptimizedProgram.signature must be a Signature")
        if not isinstance(self.demonstrations, tuple):
            raise TypeError("OptimizedProgram.demonstrations must be a tuple")
        if len(self.demonstrations) > MAX_DEMONSTRATIONS:
            raise ValueError(
                "OptimizedProgram.demonstrations count"
                f" {len(self.demonstrations)} exceeds"
                f" MAX_DEMONSTRATIONS={MAX_DEMONSTRATIONS}"
            )
        for i, d in enumerate(self.demonstrations):
            if not isinstance(d, Demonstration):
                raise TypeError(f"OptimizedProgram.demonstrations[{i}] must be a Demonstration")

    def predict(self, inputs: Mapping[str, Any], predictor: Predictor) -> Prediction:
        """Run runtime inference using the optimized prompt.

        The frozen :class:`Signature` + :attr:`demonstrations` are
        passed verbatim to ``predictor.predict``; this method does
        not re-bootstrap, mutate state, or read the wall clock.

        Raises:
            ValueError: ``inputs`` does not match the signature's
                input fields, or ``predictor.predict`` returned a
                :class:`Prediction` that does not match the
                signature's output fields.
        """

        if not isinstance(inputs, Mapping):
            raise TypeError("predict: inputs must be a Mapping")
        if not isinstance(predictor, Predictor):
            raise TypeError("predict: predictor must implement the Predictor Protocol")
        prediction = predictor.predict(self.signature, inputs, self.demonstrations)
        if not isinstance(prediction, Prediction):
            raise TypeError(
                f"predict: predictor returned {type(prediction).__name__}; expected Prediction"
            )
        expected = set(self.signature.output_names())
        actual = {k for k, _ in prediction.outputs}
        if actual != expected:
            raise ValueError(
                "predict: prediction outputs do not match signature;"
                f" expected {sorted(expected)!r}, got {sorted(actual)!r}"
            )
        return prediction

    def fingerprint(self) -> str:
        """Stable BLAKE2b-16 hex digest of the program's serialized
        form.

        Useful for storage paths under ``registry/cognitive/`` —
        callers can assert that the artifact on disk has not drifted
        from the in-memory program.
        """

        payload = serialize_program(self)
        h = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
        return h.hexdigest()


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _field_to_json(f: Field) -> Mapping[str, Any]:
    return {
        "name": f.name,
        "description": f.description,
        "field_type": _TYPE_NAMES[f.field_type],
        "kind": f.kind,
    }


def _field_from_json(blob: Mapping[str, Any]) -> Field:
    if not isinstance(blob, Mapping):
        raise TypeError("field blob must be a Mapping")
    type_name = blob.get("field_type")
    if not isinstance(type_name, str) or type_name not in _NAME_TO_TYPE:
        raise ValueError(f"field blob: unknown field_type {type_name!r}")
    return Field(
        name=str(blob["name"]),
        description=str(blob["description"]),
        field_type=_NAME_TO_TYPE[type_name],
        kind=str(blob["kind"]),
    )


def _items_to_json(
    items: Sequence[tuple[str, Any]],
) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    for key, value in items:
        if isinstance(value, bool):
            out.append({"name": key, "type": "bool", "value": bool(value)})
        elif isinstance(value, int):
            out.append({"name": key, "type": "int", "value": int(value)})
        elif isinstance(value, float):
            out.append({"name": key, "type": "float", "value": float(value)})
        elif isinstance(value, str):
            out.append({"name": key, "type": "str", "value": str(value)})
        else:
            raise TypeError(
                f"serialize_program: unsupported value type for key {key!r}: {type(value).__name__}"
            )
    return out


def _items_from_json(
    blobs: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, Any], ...]:
    out: list[tuple[str, Any]] = []
    for i, b in enumerate(blobs):
        if not isinstance(b, Mapping):
            raise TypeError(f"item blob {i} must be a Mapping")
        name = b.get("name")
        type_name = b.get("type")
        value = b.get("value")
        if not isinstance(name, str) or not name:
            raise ValueError(f"item blob {i}: name must be non-empty str")
        if type_name == "bool":
            if not isinstance(value, bool):
                raise TypeError(f"item blob {i}: bool field with non-bool value")
            out.append((name, bool(value)))
        elif type_name == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"item blob {i}: int field with non-int value")
            out.append((name, int(value)))
        elif type_name == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"item blob {i}: float field with non-numeric value")
            out.append((name, float(value)))
        elif type_name == "str":
            if not isinstance(value, str):
                raise TypeError(f"item blob {i}: str field with non-str value")
            out.append((name, str(value)))
        else:
            raise ValueError(f"item blob {i}: unknown type {type_name!r}")
    return tuple(out)


def serialize_program(program: OptimizedProgram) -> str:
    """Serialize an :class:`OptimizedProgram` to a sorted-key JSON
    string.

    The output is INV-15 byte-stable across Python versions and
    hosts: keys are sorted ascending, list ordering is preserved
    from the program's frozen tuples, and primitive types are
    explicitly tagged so booleans don't decay into ints.
    """

    if not isinstance(program, OptimizedProgram):
        raise TypeError("serialize_program: program must be OptimizedProgram")

    blob = {
        "version": DSPY_OPTIMIZER_VERSION,
        "signature": {
            "name": program.signature.name,
            "instruction": program.signature.instruction,
            "input_fields": [_field_to_json(f) for f in program.signature.input_fields],
            "output_fields": [_field_to_json(f) for f in program.signature.output_fields],
        },
        "demonstrations": [
            {
                "inputs": _items_to_json(d.inputs),
                "outputs": _items_to_json(d.outputs),
            }
            for d in program.demonstrations
        ],
    }
    return json.dumps(blob, sort_keys=True, separators=(",", ":"))


def deserialize_program(payload: str) -> OptimizedProgram:
    """Reconstruct an :class:`OptimizedProgram` from a string emitted
    by :func:`serialize_program`.

    Raises:
        ValueError: ``payload`` is not valid JSON, has an
            unrecognized version, or is structurally malformed.
        TypeError: a primitive value's tagged type does not match its
            actual JSON type.
    """

    if not isinstance(payload, str):
        raise TypeError("deserialize_program: payload must be str")
    try:
        blob = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"deserialize_program: payload is not valid JSON: {exc.msg}") from exc
    if not isinstance(blob, Mapping):
        raise ValueError("deserialize_program: payload must decode to a JSON object")
    version = blob.get("version")
    if version != DSPY_OPTIMIZER_VERSION:
        raise ValueError(
            f"deserialize_program: unsupported version {version!r};"
            f" expected {DSPY_OPTIMIZER_VERSION!r}"
        )
    sig_blob = blob.get("signature")
    if not isinstance(sig_blob, Mapping):
        raise ValueError("deserialize_program: signature must be a JSON object")
    input_fields_blob = sig_blob.get("input_fields") or []
    output_fields_blob = sig_blob.get("output_fields") or []
    if not isinstance(input_fields_blob, list):
        raise ValueError("deserialize_program: input_fields must be a JSON array")
    if not isinstance(output_fields_blob, list):
        raise ValueError("deserialize_program: output_fields must be a JSON array")
    sig = Signature(
        name=str(sig_blob.get("name", "")),
        instruction=str(sig_blob.get("instruction", "")),
        input_fields=tuple(_field_from_json(f) for f in input_fields_blob),
        output_fields=tuple(_field_from_json(f) for f in output_fields_blob),
    )
    demos_blob = blob.get("demonstrations") or []
    if not isinstance(demos_blob, list):
        raise ValueError("deserialize_program: demonstrations must be a JSON array")
    demos: list[Demonstration] = []
    for i, d in enumerate(demos_blob):
        if not isinstance(d, Mapping):
            raise ValueError(f"deserialize_program: demonstrations[{i}] must be an object")
        inputs_blob = d.get("inputs") or []
        outputs_blob = d.get("outputs") or []
        if not isinstance(inputs_blob, list) or not isinstance(outputs_blob, list):
            raise ValueError(
                f"deserialize_program: demonstrations[{i}] inputs/outputs must be JSON arrays"
            )
        demos.append(
            Demonstration(
                inputs=_items_from_json(inputs_blob),
                outputs=_items_from_json(outputs_blob),
            )
        )
    return OptimizedProgram(signature=sig, demonstrations=tuple(demos))


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def build_chain_of_thought_signature(base: Signature) -> Signature:
    """Return a copy of ``base`` with a leading ``rationale`` output
    field — the ChainOfThought prefix.

    Mirrors ``dspy/predict/chain_of_thought.py``: prepend a
    ``rationale`` field of type ``str`` so the LLM is asked to
    explain its reasoning before producing the structured outputs.
    Refuses to add the prefix if ``base`` already declares a
    ``rationale`` output (idempotent failure mode).
    """

    if not isinstance(base, Signature):
        raise TypeError("build_chain_of_thought_signature: base must be a Signature")
    for f in base.output_fields:
        if f.name == _RATIONALE_FIELD_NAME:
            raise ValueError(
                "build_chain_of_thought_signature: base already declares a"
                f" {_RATIONALE_FIELD_NAME!r} output field"
            )
    rationale = Field(
        name=_RATIONALE_FIELD_NAME,
        description=(
            "Step-by-step reasoning. Produce this BEFORE the structured"
            " outputs. Free-text; the operator does not parse it."
        ),
        field_type=str,
        kind=FieldKind.OUTPUT,
    )
    return Signature(
        name=base.name,
        instruction=base.instruction,
        input_fields=base.input_fields,
        output_fields=(rationale,) + base.output_fields,
    )


def build_governance_proposal_signature() -> Signature:
    """Construct the canonical signature for governance-proposal
    optimization.

    The fields *match* the shape of
    :class:`core.contracts.governance.GovernanceDecision` (kind,
    approved, summary, rejection_code) so the optimizer's training
    set can be drawn from historical decisions. The signature itself
    holds nothing from ``governance_engine`` and produces only
    string/bool advisory text — promotion to an actual
    :class:`GovernanceDecision` happens through the existing
    :class:`~intelligence_engine.cognitive.typed_ai.TypedAIAgent`
    surface (S-06) and the operator-approval queue (PR #87).
    """

    return Signature(
        name="governance_proposal/v1",
        instruction=(
            "Given the hazard summary, the operator's intent, and the"
            " current system mode, propose a governance decision."
            " Output the proposed kind, approval bit, summary, and"
            " rejection code (empty when approved is true)."
        ),
        input_fields=(
            Field(
                name="hazard_summary",
                description="One-line summary of active hazards.",
                field_type=str,
                kind=FieldKind.INPUT,
            ),
            Field(
                name="operator_intent",
                description=("Operator's stated intent — e.g. 'promote to CANARY'."),
                field_type=str,
                kind=FieldKind.INPUT,
            ),
            Field(
                name="system_mode",
                description="Current SystemMode, e.g. 'PAPER'.",
                field_type=str,
                kind=FieldKind.INPUT,
            ),
        ),
        output_fields=(
            Field(
                name="kind",
                description=(
                    "Proposed DecisionKind — one of MODE_TRANSITION,"
                    " PLUGIN_LIFECYCLE, KILL, REJECTED, NOOP,"
                    " INTENT_TRANSITION."
                ),
                field_type=str,
                kind=FieldKind.OUTPUT,
            ),
            Field(
                name="approved",
                description=("Whether the proposal recommends approving the operator's intent."),
                field_type=bool,
                kind=FieldKind.OUTPUT,
            ),
            Field(
                name="summary",
                description="Operator-facing one-line summary.",
                field_type=str,
                kind=FieldKind.OUTPUT,
            ),
            Field(
                name="rejection_code",
                description=("Empty when approved is true; otherwise a short code."),
                field_type=str,
                kind=FieldKind.OUTPUT,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Lazy dspy-backed predictor factory (production wiring seam)
# ---------------------------------------------------------------------------


def dspy_predictor_factory(
    *,
    completion: Callable[[str], tuple[str, str]],
    parser: Callable[[Signature, str], Mapping[str, Any]] | None = None,
) -> Predictor:
    """Build a :class:`Predictor` that delegates to a caller-supplied
    completion function.

    Arguments:
        completion: Callable that receives the rendered prompt
            string and returns ``(text, provider_id)``. Production
            wiring binds this to a :class:`LiteLLMRouter`-backed
            closure; tests inject a stub.
        parser: Optional response parser; defaults to
            :func:`_default_response_parser` which expects
            ``"<name>: <value>"`` lines for every signature output
            field.

    The factory itself does **not** import ``dspy`` — the name is a
    nod to upstream. Operators who actually want a dspy ``Predict``
    module can wire the closure to ``dspy.Predict(signature)`` in
    their own bootstrap code; this module remains importable
    without ``dspy-ai``.
    """

    if not callable(completion):
        raise TypeError("dspy_predictor_factory: completion must be callable")
    use_parser = parser if parser is not None else _default_response_parser
    if not callable(use_parser):
        raise TypeError("dspy_predictor_factory: parser must be callable")
    return _ClosurePredictor(completion=completion, parser=use_parser)


def _default_response_parser(signature: Signature, text: str) -> Mapping[str, Any]:
    """Parse ``"<name>: <value>"`` lines into a typed dict."""

    if not isinstance(text, str):
        raise TypeError("response parser: text must be str")
    expected = {f.name: f for f in signature.output_fields}
    out: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        f = expected.get(key)
        if f is None:
            continue
        if f.field_type is bool:
            lower = value.lower()
            if lower in ("true", "yes", "1"):
                out[key] = True
            elif lower in ("false", "no", "0"):
                out[key] = False
            else:
                continue
        elif f.field_type is int:
            try:
                out[key] = int(value)
            except ValueError:
                continue
        elif f.field_type is float:
            try:
                out[key] = float(value)
            except ValueError:
                continue
        else:
            out[key] = value
    return out


@dataclasses.dataclass(frozen=True, slots=True)
class _ClosurePredictor:
    """Internal :class:`Predictor` implementation for
    :func:`dspy_predictor_factory`.

    Frozen so the production wiring is immutable; all state lives in
    the captured closures.
    """

    completion: Callable[[str], tuple[str, str]]
    parser: Callable[[Signature, str], Mapping[str, Any]]

    def predict(
        self,
        signature: Signature,
        inputs: Mapping[str, Any],
        demonstrations: Sequence[Demonstration],
        /,
    ) -> Prediction:
        prompt = render_prompt(signature, inputs, demonstrations)
        text, provider_id = self.completion(prompt)
        if not isinstance(text, str):
            raise TypeError(f"completion: returned text must be str; got {type(text).__name__}")
        if not isinstance(provider_id, str):
            raise TypeError(
                f"completion: returned provider_id must be str; got {type(provider_id).__name__}"
            )
        parsed = self.parser(signature, text)
        if not isinstance(parsed, Mapping):
            raise TypeError(
                f"parser: returned object must be a Mapping; got {type(parsed).__name__}"
            )
        return Prediction(
            outputs=tuple(sorted(parsed.items())),
            provider_id=provider_id,
        )
