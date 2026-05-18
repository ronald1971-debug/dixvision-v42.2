"""A-06 — Tests for ``intelligence_engine/cognitive/dspy_optimizer.py``.

Coverage:

* :class:`Field` / :class:`Signature` validation.
* :class:`Example` / :class:`Prediction` / :class:`Demonstration`
  validation + dict round-trip.
* :func:`render_prompt` — declaration-order rendering, demo
  injection, INV-15 byte stability across dict insertion orders,
  missing/extra/wrong-type input rejection.
* :class:`BootstrapFewShot` — keep up to ``max_bootstrapped_demos``
  passing examples, refuse to emit when no example passes, skip
  predictions that miss required outputs, propagate metric errors.
* :class:`OptimizedProgram` — frozen predict path, output-name
  enforcement, fingerprint stability.
* :func:`serialize_program` / :func:`deserialize_program` —
  sorted-key byte-stable round-trip, version mismatch rejection,
  primitive type tag enforcement.
* :func:`build_chain_of_thought_signature` — prepend rationale,
  idempotent failure.
* :func:`build_governance_proposal_signature` — fields match
  GovernanceDecision shape.
* :func:`dspy_predictor_factory` — closure path, default parser.
* AST guards: no top-level dspy import, no
  governance/system/execution/evolution imports, no
  ``PatchProposal`` / ``SignalEvent`` / ``GovernanceDecision``
  constructor calls (B27 / B28 / INV-71), no top-level
  ``random`` / ``time`` / ``datetime`` / ``os`` imports (INV-15),
  no ``langsmith`` import.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from intelligence_engine.cognitive.dspy_optimizer import (
    ALLOWED_FIELD_TYPES,
    DSPY_OPTIMIZER_VERSION,
    MAX_BOOTSTRAPPED_DEMOS,
    MAX_DEMONSTRATIONS,
    MAX_FIELD_NAME_LEN,
    MAX_INPUT_FIELDS,
    MAX_INSTRUCTION_LEN,
    MAX_OUTPUT_FIELDS,
    MAX_TRAINSET_LEN,
    MAX_VALUE_LEN,
    NEW_PIP_DEPENDENCIES,
    BootstrapFewShot,
    Demonstration,
    EmptyTrainsetError,
    Example,
    Field,
    FieldKind,
    NoPassingExamplesError,
    OptimizedProgram,
    Prediction,
    Predictor,
    Signature,
    build_chain_of_thought_signature,
    build_governance_proposal_signature,
    deserialize_program,
    dspy_predictor_factory,
    render_prompt,
    serialize_program,
)

_MODULE_PATH = Path(__file__).resolve().parents[1] / (
    "intelligence_engine/cognitive/dspy_optimizer.py"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sig_qa() -> Signature:
    """Tiny QA-shaped signature used across many tests."""

    return Signature(
        name="qa/v1",
        instruction="Answer the question concisely.",
        input_fields=(
            Field(
                name="question",
                description="The question.",
                field_type=str,
                kind=FieldKind.INPUT,
            ),
        ),
        output_fields=(
            Field(
                name="answer",
                description="The answer.",
                field_type=str,
                kind=FieldKind.OUTPUT,
            ),
        ),
    )


def _ex(question: str, answer: str) -> Example:
    return Example.from_dicts(inputs={"question": question}, outputs={"answer": answer})


class _StubPredictor:
    """Predictor that returns a caller-controlled answer."""

    def __init__(self, answer_map: dict[str, str]) -> None:
        self._answers = answer_map
        self.calls: list[tuple[str, int]] = []

    def predict(
        self,
        signature: Signature,
        inputs: Mapping[str, Any],
        demonstrations: Sequence[Demonstration],
        /,
    ) -> Prediction:
        q = inputs["question"]
        self.calls.append((q, len(demonstrations)))
        return Prediction(
            outputs=(("answer", self._answers.get(q, "")),),
            provider_id="stub",
        )


# ---------------------------------------------------------------------------
# Module-level smoke + invariants
# ---------------------------------------------------------------------------


def test_module_loads() -> None:
    assert NEW_PIP_DEPENDENCIES == ("dspy-ai",)
    assert DSPY_OPTIMIZER_VERSION == "1"
    assert ALLOWED_FIELD_TYPES == (str, int, float, bool)
    assert MAX_BOOTSTRAPPED_DEMOS <= MAX_DEMONSTRATIONS


def test_predictor_protocol_runtime_checkable() -> None:
    assert isinstance(_StubPredictor({}), Predictor)
    assert not isinstance(object(), Predictor)


def test_module_has_adapted_from_header() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "ADAPTED FROM: stanfordnlp/dspy" in src


def test_module_has_no_top_level_dspy_import() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                assert not n.name.startswith("dspy"), f"top-level import {n.name} forbidden"
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not node.module.startswith("dspy"), (
                f"top-level from-import {node.module} forbidden"
            )


def test_module_has_no_forbidden_runtime_imports() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_modules = {
        "random",
        "time",
        "datetime",
        "os",
        "asyncio",
        "websockets",
        "numpy",
        "torch",
        "polars",
        "langsmith",
    }
    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                top = n.name.split(".")[0]
                assert top not in forbidden_modules, f"forbidden import: {n.name}"
        if isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            assert top not in forbidden_modules, f"forbidden from-import: {node.module}"


def test_module_has_no_engine_cross_imports() -> None:
    src = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    forbidden_prefixes = (
        "governance_engine",
        "system_engine",
        "execution_engine",
        "evolution_engine",
    )
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for p in forbidden_prefixes:
                assert not mod.startswith(p), f"forbidden cross-engine import: {mod}"


def test_module_does_not_construct_typed_bus_events() -> None:
    """B27 / B28 / INV-71 authority symmetry: this module is OFFLINE_ONLY
    advisory. Promotion to typed events happens elsewhere through a
    governance-gated path.
    """

    forbidden = {"PatchProposal", "SignalEvent", "GovernanceDecision"}
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in forbidden:
                raise AssertionError(f"forbidden constructor: {func.id}")
            if isinstance(func, ast.Attribute) and func.attr in forbidden:
                raise AssertionError(f"forbidden constructor: {func.attr}")


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def test_field_basic() -> None:
    f = Field("x", "desc", str, FieldKind.INPUT)
    assert f.name == "x"
    assert f.kind == FieldKind.INPUT


def test_field_rejects_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Field("", "desc", str, FieldKind.INPUT)


def test_field_rejects_oversize_name() -> None:
    with pytest.raises(ValueError, match="MAX_FIELD_NAME_LEN"):
        Field("a" * (MAX_FIELD_NAME_LEN + 1), "desc", str, FieldKind.INPUT)


def test_field_rejects_leading_digit() -> None:
    with pytest.raises(ValueError, match="letter or underscore"):
        Field("1x", "desc", str, FieldKind.INPUT)


def test_field_rejects_punctuation() -> None:
    with pytest.raises(ValueError, match="alphanumeric"):
        Field("x-y", "desc", str, FieldKind.INPUT)


def test_field_rejects_bad_kind() -> None:
    with pytest.raises(ValueError, match="Field.kind"):
        Field("x", "desc", str, "WRONG")


def test_field_rejects_bad_type() -> None:
    with pytest.raises(TypeError, match="field_type"):
        Field("x", "desc", list, FieldKind.INPUT)


def test_field_rejects_non_str_description() -> None:
    with pytest.raises(TypeError, match="description"):
        Field("x", 42, str, FieldKind.INPUT)  # type: ignore[arg-type]


def test_field_allows_underscore_start() -> None:
    Field("_x", "desc", str, FieldKind.INPUT)


# ---------------------------------------------------------------------------
# Signature validation
# ---------------------------------------------------------------------------


def test_signature_basic() -> None:
    sig = _sig_qa()
    assert sig.input_names() == ("question",)
    assert sig.output_names() == ("answer",)


def test_signature_rejects_empty_inputs() -> None:
    out = Field("a", "d", str, FieldKind.OUTPUT)
    with pytest.raises(ValueError, match="input_fields"):
        Signature(name="x", instruction="i", input_fields=(), output_fields=(out,))


def test_signature_rejects_empty_outputs() -> None:
    inp = Field("a", "d", str, FieldKind.INPUT)
    with pytest.raises(ValueError, match="output_fields"):
        Signature(name="x", instruction="i", input_fields=(inp,), output_fields=())


def test_signature_rejects_kind_mismatch_on_inputs() -> None:
    out = Field("a", "d", str, FieldKind.OUTPUT)
    with pytest.raises(ValueError, match="input_fields"):
        Signature(name="x", instruction="i", input_fields=(out,), output_fields=(out,))


def test_signature_rejects_kind_mismatch_on_outputs() -> None:
    inp = Field("a", "d", str, FieldKind.INPUT)
    with pytest.raises(ValueError, match="output_fields"):
        Signature(name="x", instruction="i", input_fields=(inp,), output_fields=(inp,))


def test_signature_rejects_duplicate_names_across_kinds() -> None:
    inp = Field("a", "d", str, FieldKind.INPUT)
    out = Field("a", "d", str, FieldKind.OUTPUT)
    with pytest.raises(ValueError, match="duplicate"):
        Signature(name="x", instruction="i", input_fields=(inp,), output_fields=(out,))


def test_signature_rejects_oversize_instruction() -> None:
    inp = Field("a", "d", str, FieldKind.INPUT)
    out = Field("b", "d", str, FieldKind.OUTPUT)
    with pytest.raises(ValueError, match="instruction"):
        Signature(
            name="x",
            instruction="a" * (MAX_INSTRUCTION_LEN + 1),
            input_fields=(inp,),
            output_fields=(out,),
        )


def test_signature_rejects_too_many_input_fields() -> None:
    inputs = tuple(Field(f"i{i}", "d", str, FieldKind.INPUT) for i in range(MAX_INPUT_FIELDS + 1))
    out = Field("o", "d", str, FieldKind.OUTPUT)
    with pytest.raises(ValueError, match="MAX_INPUT_FIELDS"):
        Signature(name="x", instruction="i", input_fields=inputs, output_fields=(out,))


def test_signature_rejects_too_many_output_fields() -> None:
    inp = Field("i", "d", str, FieldKind.INPUT)
    outputs = tuple(
        Field(f"o{i}", "d", str, FieldKind.OUTPUT) for i in range(MAX_OUTPUT_FIELDS + 1)
    )
    with pytest.raises(ValueError, match="MAX_OUTPUT_FIELDS"):
        Signature(name="x", instruction="i", input_fields=(inp,), output_fields=outputs)


def test_signature_field_for() -> None:
    sig = _sig_qa()
    assert sig.field_for("question").kind == FieldKind.INPUT
    assert sig.field_for("answer").kind == FieldKind.OUTPUT
    with pytest.raises(KeyError):
        sig.field_for("nope")


def test_signature_is_frozen() -> None:
    sig = _sig_qa()
    with pytest.raises(dataclasses_FrozenInstanceError()):
        sig.instruction = "mutated"  # type: ignore[misc]


def dataclasses_FrozenInstanceError() -> type:
    import dataclasses

    return dataclasses.FrozenInstanceError


# ---------------------------------------------------------------------------
# Example / Prediction / Demonstration
# ---------------------------------------------------------------------------


def test_example_from_dicts_sorts_keys() -> None:
    ex = Example.from_dicts(inputs={"b": 1, "a": 2}, outputs={"y": 3, "x": 4})
    assert ex.inputs == (("a", 2), ("b", 1))
    assert ex.outputs == (("x", 4), ("y", 3))


def test_example_round_trip() -> None:
    ex = Example.from_dicts(inputs={"a": 1}, outputs={"b": 2})
    assert ex.as_inputs() == {"a": 1}
    assert ex.as_outputs() == {"b": 2}


def test_example_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Example(inputs=(("a", 1), ("a", 2)), outputs=(("b", 3),))


def test_example_rejects_non_tuple_entry() -> None:
    with pytest.raises(TypeError):
        Example(inputs=("not-a-tuple",), outputs=(("b", 3),))  # type: ignore[arg-type]


def test_prediction_validates_outputs() -> None:
    p = Prediction(outputs=(("a", 1),), provider_id="x")
    assert p.as_dict() == {"a": 1}


def test_prediction_rejects_duplicate() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        Prediction(outputs=(("a", 1), ("a", 2)))


def test_prediction_default_provider_id() -> None:
    p = Prediction(outputs=(("a", 1),))
    assert p.provider_id == ""


def test_demonstration_basic() -> None:
    d = Demonstration(inputs=(("q", "x"),), outputs=(("a", "y"),))
    assert dict(d.inputs) == {"q": "x"}


def test_demonstration_rejects_bad_entry() -> None:
    with pytest.raises(TypeError):
        Demonstration(inputs=(123,), outputs=(("a", "y"),))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


def test_render_prompt_basic() -> None:
    sig = _sig_qa()
    text = render_prompt(sig, {"question": "what is 1+1?"}, ())
    assert "Answer the question concisely." in text
    assert "question (str):" in text
    assert "answer (str):" in text
    assert "question: what is 1+1?" in text
    assert "Now:" in text


def test_render_prompt_includes_demos() -> None:
    sig = _sig_qa()
    demos = (
        Demonstration(inputs=(("question", "q1"),), outputs=(("answer", "a1"),)),
        Demonstration(inputs=(("question", "q2"),), outputs=(("answer", "a2"),)),
    )
    text = render_prompt(sig, {"question": "q3"}, demos)
    assert text.count("Example ") == 2
    assert "q1" in text and "a1" in text
    assert "q2" in text and "a2" in text


def test_render_prompt_is_dict_order_independent() -> None:
    """INV-15 byte-stable rendering: insertion order of inputs dict must
    not affect the rendered string.
    """

    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(
            Field("a", "d", str, FieldKind.INPUT),
            Field("b", "d", str, FieldKind.INPUT),
        ),
        output_fields=(Field("o", "d", str, FieldKind.OUTPUT),),
    )
    text1 = render_prompt(sig, {"a": "1", "b": "2"}, ())
    text2 = render_prompt(sig, {"b": "2", "a": "1"}, ())
    assert text1 == text2


def test_render_prompt_three_run_byte_identical() -> None:
    sig = _sig_qa()
    demos = (Demonstration(inputs=(("question", "q1"),), outputs=(("answer", "a1"),)),)
    runs = [render_prompt(sig, {"question": "what?"}, demos) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]


def test_render_prompt_rejects_missing_input() -> None:
    sig = _sig_qa()
    with pytest.raises(ValueError, match="missing"):
        render_prompt(sig, {}, ())


def test_render_prompt_rejects_extra_input() -> None:
    sig = _sig_qa()
    with pytest.raises(ValueError, match="unknown"):
        render_prompt(sig, {"question": "q", "junk": "x"}, ())


def test_render_prompt_rejects_wrong_input_type() -> None:
    sig = _sig_qa()
    with pytest.raises(TypeError, match="expected str"):
        render_prompt(sig, {"question": 5}, ())  # type: ignore[dict-item]


def test_render_prompt_rejects_bool_as_int() -> None:
    """bool must not satisfy int — DSPy demands typed primitives."""

    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("n", "d", int, FieldKind.INPUT),),
        output_fields=(Field("o", "d", str, FieldKind.OUTPUT),),
    )
    with pytest.raises(TypeError, match="got bool"):
        render_prompt(sig, {"n": True}, ())


def test_render_prompt_rejects_bool_as_float() -> None:
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("n", "d", float, FieldKind.INPUT),),
        output_fields=(Field("o", "d", str, FieldKind.OUTPUT),),
    )
    with pytest.raises(TypeError, match="got bool"):
        render_prompt(sig, {"n": True}, ())


def test_render_prompt_rejects_int_as_bool() -> None:
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("b", "d", bool, FieldKind.INPUT),),
        output_fields=(Field("o", "d", str, FieldKind.OUTPUT),),
    )
    with pytest.raises(TypeError, match="expected bool"):
        render_prompt(sig, {"b": 1}, ())


def test_render_prompt_rejects_demo_missing_input() -> None:
    sig = _sig_qa()
    demos = (Demonstration(inputs=(), outputs=(("answer", "a"),)),)
    with pytest.raises(ValueError, match="missing input"):
        render_prompt(sig, {"question": "q"}, demos)


def test_render_prompt_rejects_demo_missing_output() -> None:
    sig = _sig_qa()
    demos = (Demonstration(inputs=(("question", "q1"),), outputs=()),)
    with pytest.raises(ValueError, match="missing output"):
        render_prompt(sig, {"question": "q"}, demos)


def test_render_prompt_rejects_oversize_value() -> None:
    sig = _sig_qa()
    with pytest.raises(ValueError, match="MAX_VALUE_LEN"):
        render_prompt(sig, {"question": "a" * (MAX_VALUE_LEN + 1)}, ())


def test_render_prompt_rejects_overlong_render() -> None:
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("a", "d" * 200, str, FieldKind.INPUT),),
        output_fields=(Field("o", "d", str, FieldKind.OUTPUT),),
    )
    # Build a demo chain that overflows MAX_PROMPT_LEN.
    big_value = "a" * (MAX_VALUE_LEN - 100)
    demos = tuple(
        Demonstration(inputs=(("a", big_value),), outputs=(("o", big_value),)) for _ in range(20)
    )
    with pytest.raises(ValueError, match="MAX_PROMPT_LEN"):
        render_prompt(sig, {"a": "q"}, demos)


def test_render_prompt_rejects_non_signature() -> None:
    with pytest.raises(TypeError):
        render_prompt("not-a-sig", {"q": "x"}, ())  # type: ignore[arg-type]


def test_render_prompt_rejects_non_demo() -> None:
    sig = _sig_qa()
    with pytest.raises(TypeError, match="Demonstration"):
        render_prompt(sig, {"question": "q"}, ("not-a-demo",))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# BootstrapFewShot
# ---------------------------------------------------------------------------


def _always_pass(_ex: Example, _pr: Prediction) -> bool:
    return True


def _always_fail(_ex: Example, _pr: Prediction) -> bool:
    return False


def test_bootstrap_basic_compile() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1", "q2": "a2"})
    trainset = [_ex("q1", "a1"), _ex("q2", "a2")]
    bs = BootstrapFewShot(max_bootstrapped_demos=2)
    prog = bs.compile(sig, trainset, _always_pass, pred)
    assert isinstance(prog, OptimizedProgram)
    assert len(prog.demonstrations) == 2


def test_bootstrap_preserves_trainset_order() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1", "q2": "a2", "q3": "a3"})
    trainset = [_ex("q3", "a3"), _ex("q1", "a1"), _ex("q2", "a2")]
    bs = BootstrapFewShot(max_bootstrapped_demos=3)
    prog = bs.compile(sig, trainset, _always_pass, pred)
    assert tuple(dict(d.inputs)["question"] for d in prog.demonstrations) == (
        "q3",
        "q1",
        "q2",
    )


def test_bootstrap_caps_at_max_bootstrapped_demos() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1", "q2": "a2", "q3": "a3"})
    trainset = [_ex("q1", "a1"), _ex("q2", "a2"), _ex("q3", "a3")]
    bs = BootstrapFewShot(max_bootstrapped_demos=2)
    prog = bs.compile(sig, trainset, _always_pass, pred)
    assert len(prog.demonstrations) == 2


def test_bootstrap_stops_after_cap_reached() -> None:
    """Predictor should not be called beyond the cap."""

    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1", "q2": "a2", "q3": "a3"})
    trainset = [_ex("q1", "a1"), _ex("q2", "a2"), _ex("q3", "a3")]
    bs = BootstrapFewShot(max_bootstrapped_demos=2)
    bs.compile(sig, trainset, _always_pass, pred)
    assert len(pred.calls) == 2


def test_bootstrap_refuses_empty_trainset() -> None:
    bs = BootstrapFewShot()
    with pytest.raises(EmptyTrainsetError):
        bs.compile(_sig_qa(), [], _always_pass, _StubPredictor({}))


def test_bootstrap_refuses_when_no_example_passes() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1"})
    bs = BootstrapFewShot()
    with pytest.raises(NoPassingExamplesError):
        bs.compile(sig, [_ex("q1", "a1")], _always_fail, pred)


def test_bootstrap_skips_predictions_missing_output() -> None:
    """If predictor returns outputs missing a required field, the
    example is silently skipped (mirrors dspy upstream)."""

    sig = _sig_qa()

    class MissingPredictor:
        def predict(
            self,
            signature: Signature,
            inputs: Mapping[str, Any],
            demonstrations: Sequence[Demonstration],
            /,
        ) -> Prediction:
            q = inputs["question"]
            if q == "q1":
                return Prediction(outputs=(("wrong_field", "x"),))
            return Prediction(outputs=(("answer", "a2"),))

    bs = BootstrapFewShot(max_bootstrapped_demos=2)
    prog = bs.compile(sig, [_ex("q1", "a1"), _ex("q2", "a2")], _always_pass, MissingPredictor())
    assert len(prog.demonstrations) == 1
    assert dict(prog.demonstrations[0].inputs)["question"] == "q2"


def test_bootstrap_metric_filters_examples() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "right", "q2": "wrong"})
    bs = BootstrapFewShot(max_bootstrapped_demos=2)
    prog = bs.compile(
        sig,
        [_ex("q1", "right"), _ex("q2", "right")],
        lambda ex, p: dict(p.outputs).get("answer") == dict(ex.outputs).get("answer"),
        pred,
    )
    assert len(prog.demonstrations) == 1
    assert dict(prog.demonstrations[0].outputs)["answer"] == "right"


def test_bootstrap_metric_exception_is_propagated_as_typeerror() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1"})

    def explode(_ex: Example, _pr: Prediction) -> bool:
        raise RuntimeError("kaboom")

    bs = BootstrapFewShot()
    with pytest.raises(TypeError, match="metric raised"):
        bs.compile(sig, [_ex("q1", "a1")], explode, pred)


def test_bootstrap_rejects_bad_args() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({})
    bs = BootstrapFewShot()
    with pytest.raises(TypeError):
        bs.compile("not-a-sig", [_ex("q", "a")], _always_pass, pred)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bs.compile(sig, "not-a-list", _always_pass, pred)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bs.compile(sig, [_ex("q", "a")], "not-callable", pred)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        bs.compile(sig, [_ex("q", "a")], _always_pass, "not-pred")  # type: ignore[arg-type]


def test_bootstrap_rejects_oversize_trainset() -> None:
    bs = BootstrapFewShot()
    pred = _StubPredictor({})
    trainset = [_ex("q", "a")] * (MAX_TRAINSET_LEN + 1)
    with pytest.raises(ValueError, match="MAX_TRAINSET_LEN"):
        bs.compile(_sig_qa(), trainset, _always_pass, pred)


def test_bootstrap_rejects_bad_max_bootstrapped_demos() -> None:
    with pytest.raises(ValueError):
        BootstrapFewShot(max_bootstrapped_demos=0)
    with pytest.raises(ValueError):
        BootstrapFewShot(max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS + 1)
    with pytest.raises(TypeError):
        BootstrapFewShot(max_bootstrapped_demos=True)  # type: ignore[arg-type]


def test_bootstrap_three_run_replay_equality() -> None:
    """INV-15: three back-to-back compiles with the same inputs and a
    pure-function predictor must produce byte-identical programs.
    """

    sig = _sig_qa()
    trainset = [_ex("q1", "a1"), _ex("q2", "a2"), _ex("q3", "a3")]
    bs = BootstrapFewShot(max_bootstrapped_demos=3)
    progs = [
        bs.compile(
            sig, trainset, _always_pass, _StubPredictor({"q1": "a1", "q2": "a2", "q3": "a3"})
        )
        for _ in range(3)
    ]
    serials = [serialize_program(p) for p in progs]
    assert serials[0] == serials[1] == serials[2]


# ---------------------------------------------------------------------------
# OptimizedProgram
# ---------------------------------------------------------------------------


def test_program_predict_runtime_path() -> None:
    sig = _sig_qa()
    demos = (Demonstration(inputs=(("question", "q1"),), outputs=(("answer", "a1"),)),)
    prog = OptimizedProgram(signature=sig, demonstrations=demos)
    pred = _StubPredictor({"qX": "aX"})
    out = prog.predict({"question": "qX"}, pred)
    assert dict(out.outputs)["answer"] == "aX"
    # The runtime path forwards the frozen demonstrations to the predictor.
    assert pred.calls[-1] == ("qX", 1)


def test_program_predict_rejects_mismatched_outputs() -> None:
    sig = _sig_qa()
    prog = OptimizedProgram(signature=sig, demonstrations=())

    class WrongOutputs:
        def predict(
            self,
            signature: Signature,
            inputs: Mapping[str, Any],
            demonstrations: Sequence[Demonstration],
            /,
        ) -> Prediction:
            return Prediction(outputs=(("nope", "x"),))

    with pytest.raises(ValueError, match="do not match signature"):
        prog.predict({"question": "q"}, WrongOutputs())


def test_program_predict_rejects_non_prediction() -> None:
    sig = _sig_qa()
    prog = OptimizedProgram(signature=sig, demonstrations=())

    class BadPredictor:
        def predict(
            self,
            signature: Signature,
            inputs: Mapping[str, Any],
            demonstrations: Sequence[Demonstration],
            /,
        ) -> Prediction:
            return "not-a-prediction"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="expected Prediction"):
        prog.predict({"question": "q"}, BadPredictor())


def test_program_rejects_too_many_demos() -> None:
    sig = _sig_qa()
    too_many = tuple(
        Demonstration(inputs=(("question", f"q{i}"),), outputs=(("answer", f"a{i}"),))
        for i in range(MAX_DEMONSTRATIONS + 1)
    )
    with pytest.raises(ValueError, match="MAX_DEMONSTRATIONS"):
        OptimizedProgram(signature=sig, demonstrations=too_many)


def test_program_fingerprint_stable() -> None:
    sig = _sig_qa()
    demos = (Demonstration(inputs=(("question", "q1"),), outputs=(("answer", "a1"),)),)
    prog = OptimizedProgram(signature=sig, demonstrations=demos)
    assert prog.fingerprint() == prog.fingerprint()


def test_program_fingerprint_changes_on_demo_change() -> None:
    sig = _sig_qa()
    p1 = OptimizedProgram(
        signature=sig,
        demonstrations=(Demonstration(inputs=(("question", "q1"),), outputs=(("answer", "a1"),)),),
    )
    p2 = OptimizedProgram(
        signature=sig,
        demonstrations=(Demonstration(inputs=(("question", "q2"),), outputs=(("answer", "a2"),)),),
    )
    assert p1.fingerprint() != p2.fingerprint()


def test_program_is_frozen() -> None:
    import dataclasses

    sig = _sig_qa()
    prog = OptimizedProgram(signature=sig, demonstrations=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        prog.signature = sig  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_serialize_roundtrip_basic() -> None:
    sig = _sig_qa()
    demos = (Demonstration(inputs=(("question", "q1"),), outputs=(("answer", "a1"),)),)
    prog = OptimizedProgram(signature=sig, demonstrations=demos)
    payload = serialize_program(prog)
    restored = deserialize_program(payload)
    assert serialize_program(restored) == payload
    assert restored.fingerprint() == prog.fingerprint()


def test_serialize_is_sorted_key_byte_stable() -> None:
    sig = _sig_qa()
    prog = OptimizedProgram(signature=sig, demonstrations=())
    payload = serialize_program(prog)
    # sort_keys=True ⇒ rerunning json.dumps on the parsed blob yields
    # the same string.
    blob = json.loads(payload)
    assert json.dumps(blob, sort_keys=True, separators=(",", ":")) == payload


def test_serialize_preserves_bool_type_tag() -> None:
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("flag", "d", bool, FieldKind.INPUT),),
        output_fields=(Field("ok", "d", bool, FieldKind.OUTPUT),),
    )
    demos = (Demonstration(inputs=(("flag", True),), outputs=(("ok", False),)),)
    prog = OptimizedProgram(signature=sig, demonstrations=demos)
    payload = serialize_program(prog)
    restored = deserialize_program(payload)
    assert dict(restored.demonstrations[0].inputs)["flag"] is True
    assert dict(restored.demonstrations[0].outputs)["ok"] is False


def test_serialize_preserves_numeric_types() -> None:
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(
            Field("i", "d", int, FieldKind.INPUT),
            Field("f", "d", float, FieldKind.INPUT),
        ),
        output_fields=(Field("o", "d", str, FieldKind.OUTPUT),),
    )
    demos = (Demonstration(inputs=(("f", 1.5), ("i", 3)), outputs=(("o", "z"),)),)
    prog = OptimizedProgram(signature=sig, demonstrations=demos)
    restored = deserialize_program(serialize_program(prog))
    d = dict(restored.demonstrations[0].inputs)
    assert isinstance(d["i"], int) and not isinstance(d["i"], bool)
    assert isinstance(d["f"], float)


def test_serialize_rejects_unsupported_value_type() -> None:
    sig = _sig_qa()
    demos = (Demonstration(inputs=(("question", "q"),), outputs=(("answer", "a"),)),)
    prog = OptimizedProgram(signature=sig, demonstrations=demos)
    # Forcefully build a program with an unsupported value (bypass
    # field validation by constructing the Demonstration directly).
    bad = OptimizedProgram(
        signature=sig,
        demonstrations=(Demonstration(inputs=(("question", [1, 2]),), outputs=(("answer", "a"),)),),
    )
    with pytest.raises(TypeError, match="unsupported value type"):
        serialize_program(bad)
    # untouched ok-program still serializes
    serialize_program(prog)


def test_deserialize_rejects_wrong_version() -> None:
    sig = _sig_qa()
    prog = OptimizedProgram(signature=sig, demonstrations=())
    payload = json.dumps(
        {
            **json.loads(serialize_program(prog)),
            "version": "9999",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with pytest.raises(ValueError, match="unsupported version"):
        deserialize_program(payload)


def test_deserialize_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        deserialize_program("not json")


def test_deserialize_rejects_non_object_root() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        deserialize_program("[]")


def test_deserialize_rejects_bool_with_int_tag() -> None:
    """A boolean value tagged as 'int' must be rejected (Python's
    isinstance(True, int) is True, but the on-disk shape is explicit).
    """

    blob = {
        "version": DSPY_OPTIMIZER_VERSION,
        "signature": {
            "name": "x",
            "instruction": "i",
            "input_fields": [
                {"name": "a", "description": "d", "field_type": "int", "kind": "input"}
            ],
            "output_fields": [
                {"name": "o", "description": "d", "field_type": "str", "kind": "output"}
            ],
        },
        "demonstrations": [
            {
                "inputs": [{"name": "a", "type": "int", "value": True}],
                "outputs": [{"name": "o", "type": "str", "value": "x"}],
            }
        ],
    }
    with pytest.raises(TypeError, match="int field"):
        deserialize_program(json.dumps(blob))


def test_deserialize_rejects_unknown_field_type() -> None:
    blob = {
        "version": DSPY_OPTIMIZER_VERSION,
        "signature": {
            "name": "x",
            "instruction": "i",
            "input_fields": [
                {"name": "a", "description": "d", "field_type": "list", "kind": "input"}
            ],
            "output_fields": [
                {"name": "o", "description": "d", "field_type": "str", "kind": "output"}
            ],
        },
        "demonstrations": [],
    }
    with pytest.raises(ValueError, match="unknown field_type"):
        deserialize_program(json.dumps(blob))


def test_deserialize_rejects_unknown_value_type() -> None:
    blob = {
        "version": DSPY_OPTIMIZER_VERSION,
        "signature": {
            "name": "x",
            "instruction": "i",
            "input_fields": [
                {"name": "a", "description": "d", "field_type": "str", "kind": "input"}
            ],
            "output_fields": [
                {"name": "o", "description": "d", "field_type": "str", "kind": "output"}
            ],
        },
        "demonstrations": [
            {
                "inputs": [{"name": "a", "type": "list", "value": [1, 2]}],
                "outputs": [{"name": "o", "type": "str", "value": "x"}],
            }
        ],
    }
    with pytest.raises(ValueError, match="unknown type"):
        deserialize_program(json.dumps(blob))


# ---------------------------------------------------------------------------
# ChainOfThought + governance proposal signature
# ---------------------------------------------------------------------------


def test_chain_of_thought_prepends_rationale() -> None:
    sig = _sig_qa()
    cot = build_chain_of_thought_signature(sig)
    assert cot.output_names() == ("rationale", "answer")
    rationale = cot.field_for("rationale")
    assert rationale.field_type is str
    assert rationale.kind == FieldKind.OUTPUT


def test_chain_of_thought_idempotent_failure() -> None:
    rationale = Field("rationale", "d", str, FieldKind.OUTPUT)
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("a", "d", str, FieldKind.INPUT),),
        output_fields=(rationale,),
    )
    with pytest.raises(ValueError, match="already declares"):
        build_chain_of_thought_signature(sig)


def test_chain_of_thought_renders_rationale_first() -> None:
    sig = build_chain_of_thought_signature(_sig_qa())
    text = render_prompt(sig, {"question": "q?"}, ())
    rationale_idx = text.index("- rationale (str):")
    answer_idx = text.index("- answer (str):")
    assert rationale_idx < answer_idx


def test_governance_proposal_signature_shape() -> None:
    sig = build_governance_proposal_signature()
    assert sig.name == "governance_proposal/v1"
    inputs = sig.input_names()
    outputs = sig.output_names()
    assert set(inputs) == {"hazard_summary", "operator_intent", "system_mode"}
    assert set(outputs) == {"kind", "approved", "summary", "rejection_code"}
    assert sig.field_for("approved").field_type is bool
    assert sig.field_for("kind").field_type is str


def test_governance_proposal_signature_does_not_emit_typed_events() -> None:
    """Sanity: the helper returns a Signature value object, never an
    actual GovernanceDecision (B27 / B28)."""

    sig = build_governance_proposal_signature()
    assert isinstance(sig, Signature)


# ---------------------------------------------------------------------------
# dspy_predictor_factory
# ---------------------------------------------------------------------------


def test_dspy_predictor_factory_round_trip() -> None:
    sig = _sig_qa()

    def completion(prompt: str) -> tuple[str, str]:
        assert "question: hello" in prompt
        return ("answer: world\n", "stub-provider")

    pred = dspy_predictor_factory(completion=completion)
    out = pred.predict(sig, {"question": "hello"}, ())
    assert dict(out.outputs)["answer"] == "world"
    assert out.provider_id == "stub-provider"


def test_dspy_predictor_factory_parses_bool() -> None:
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("q", "d", str, FieldKind.INPUT),),
        output_fields=(Field("ok", "d", bool, FieldKind.OUTPUT),),
    )

    def completion(_p: str) -> tuple[str, str]:
        return ("ok: TRUE\n", "p")

    pred = dspy_predictor_factory(completion=completion)
    out = pred.predict(sig, {"q": "hi"}, ())
    assert dict(out.outputs)["ok"] is True


def test_dspy_predictor_factory_parses_numerics() -> None:
    sig = Signature(
        name="x",
        instruction="i",
        input_fields=(Field("q", "d", str, FieldKind.INPUT),),
        output_fields=(
            Field("i", "d", int, FieldKind.OUTPUT),
            Field("f", "d", float, FieldKind.OUTPUT),
        ),
    )

    def completion(_p: str) -> tuple[str, str]:
        return ("i: 42\nf: 3.5\n", "p")

    pred = dspy_predictor_factory(completion=completion)
    out = pred.predict(sig, {"q": "hi"}, ())
    d = dict(out.outputs)
    assert d["i"] == 42 and isinstance(d["i"], int)
    assert d["f"] == 3.5 and isinstance(d["f"], float)


def test_dspy_predictor_factory_skips_unparseable_lines() -> None:
    """Lines without ':' are ignored; unknown field names are ignored.
    Missing required outputs surface later via
    :meth:`OptimizedProgram.predict`."""

    sig = _sig_qa()

    def completion(_p: str) -> tuple[str, str]:
        return ("garbage line\nanswer: ok\nirrelevant: stuff\n", "p")

    pred = dspy_predictor_factory(completion=completion)
    out = pred.predict(sig, {"question": "q"}, ())
    assert dict(out.outputs) == {"answer": "ok"}


def test_dspy_predictor_factory_custom_parser() -> None:
    sig = _sig_qa()

    def completion(_p: str) -> tuple[str, str]:
        return ("ignored", "p")

    def parser(_sig: Signature, _text: str) -> Mapping[str, Any]:
        return {"answer": "fixed"}

    pred = dspy_predictor_factory(completion=completion, parser=parser)
    out = pred.predict(sig, {"question": "q"}, ())
    assert dict(out.outputs)["answer"] == "fixed"


def test_dspy_predictor_factory_rejects_non_callable() -> None:
    with pytest.raises(TypeError):
        dspy_predictor_factory(completion="nope")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        dspy_predictor_factory(completion=lambda _p: ("a", "b"), parser="nope")  # type: ignore[arg-type]


def test_dspy_predictor_factory_rejects_bad_completion_return() -> None:
    sig = _sig_qa()

    def bad_completion(_p: str) -> tuple[str, str]:
        return (None, "p")  # type: ignore[return-value]

    pred = dspy_predictor_factory(completion=bad_completion)
    with pytest.raises(TypeError, match="text must be str"):
        pred.predict(sig, {"question": "q"}, ())

    def bad_provider(_p: str) -> tuple[str, str]:
        return ("answer: ok", None)  # type: ignore[return-value]

    pred2 = dspy_predictor_factory(completion=bad_provider)
    with pytest.raises(TypeError, match="provider_id must be str"):
        pred2.predict(sig, {"question": "q"}, ())


def test_dspy_predictor_factory_rejects_bad_parser_return() -> None:
    sig = _sig_qa()

    def completion(_p: str) -> tuple[str, str]:
        return ("answer: ok", "p")

    def parser(_sig: Signature, _text: str) -> Mapping[str, Any]:
        return "not-a-mapping"  # type: ignore[return-value]

    pred = dspy_predictor_factory(completion=completion, parser=parser)
    with pytest.raises(TypeError, match="must be a Mapping"):
        pred.predict(sig, {"question": "q"}, ())


# ---------------------------------------------------------------------------
# End-to-end optimize + predict
# ---------------------------------------------------------------------------


def test_e2e_optimize_then_runtime_predict() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1", "q2": "a2", "q3": "a3"})
    trainset = [_ex("q1", "a1"), _ex("q2", "a2"), _ex("q3", "a3")]
    bs = BootstrapFewShot(max_bootstrapped_demos=2)
    prog = bs.compile(sig, trainset, _always_pass, pred)

    # Runtime: a fresh predictor with the same answer space; the demos
    # are passed in by the program, not re-bootstrapped.
    runtime_pred = _StubPredictor({"qX": "AX"})
    out = prog.predict({"question": "qX"}, runtime_pred)
    assert dict(out.outputs)["answer"] == "AX"
    # The runtime call observes the optimized demos length (== 2).
    assert runtime_pred.calls[-1] == ("qX", 2)


def test_e2e_optimize_then_serialize_then_predict() -> None:
    sig = _sig_qa()
    pred = _StubPredictor({"q1": "a1"})
    bs = BootstrapFewShot(max_bootstrapped_demos=1)
    prog = bs.compile(sig, [_ex("q1", "a1")], _always_pass, pred)

    payload = serialize_program(prog)
    restored = deserialize_program(payload)

    runtime_pred = _StubPredictor({"qX": "AX"})
    out = restored.predict({"question": "qX"}, runtime_pred)
    assert dict(out.outputs)["answer"] == "AX"
    assert runtime_pred.calls[-1] == ("qX", 1)
