# ADAPTED FROM: https://github.com/guidance-ai/guidance (MIT)
#
# Tests for the I-20 guidance adapter — constrained-generation template
# + validator.
"""I-20 tests: guidance-style template parse + match."""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from intelligence_engine.cognitive import guidance_adapter as ga
from intelligence_engine.cognitive.guidance_adapter import (
    NEW_PIP_DEPENDENCIES,
    GuidanceError,
    GuidanceMatchError,
    GuidanceParseError,
    RegexSlot,
    SelectSlot,
    compile_template,
    enable_guidance_factory,
    match_completion,
)

# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == ("guidance",)


def test_guidance_parse_error_is_guidance_error() -> None:
    assert issubclass(GuidanceParseError, GuidanceError)


def test_guidance_match_error_is_guidance_error() -> None:
    assert issubclass(GuidanceMatchError, GuidanceError)


def test_guidance_error_is_value_error() -> None:
    assert issubclass(GuidanceError, ValueError)


# ---------------------------------------------------------------------------
# RegexSlot
# ---------------------------------------------------------------------------


def test_regex_slot_happy_path() -> None:
    s = RegexSlot(name="x", pattern=r"\d+")
    assert s.name == "x"
    assert s.pattern == r"\d+"


def test_regex_slot_is_frozen() -> None:
    s = RegexSlot(name="x", pattern=r"\d+")
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.name = "y"  # type: ignore[misc]


@pytest.mark.parametrize("bad", ["", 0, None])
def test_regex_slot_rejects_bad_name(bad: object) -> None:
    with pytest.raises(GuidanceError):
        RegexSlot(name=bad, pattern=r"\d+")  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", ["", 0, None])
def test_regex_slot_rejects_bad_pattern(bad: object) -> None:
    with pytest.raises(GuidanceError):
        RegexSlot(name="x", pattern=bad)  # type: ignore[arg-type]


def test_regex_slot_rejects_invalid_regex() -> None:
    with pytest.raises(GuidanceError):
        RegexSlot(name="x", pattern="[unclosed")


# ---------------------------------------------------------------------------
# SelectSlot
# ---------------------------------------------------------------------------


def test_select_slot_happy_path() -> None:
    s = SelectSlot(name="side", choices=("BUY", "SELL"))
    assert s.choices == ("BUY", "SELL")


def test_select_slot_is_frozen() -> None:
    s = SelectSlot(name="side", choices=("BUY", "SELL"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.choices = ()  # type: ignore[misc]


def test_select_slot_rejects_empty_choices() -> None:
    with pytest.raises(GuidanceError):
        SelectSlot(name="side", choices=())


def test_select_slot_rejects_non_tuple_choices() -> None:
    with pytest.raises(GuidanceError):
        SelectSlot(name="side", choices=["BUY", "SELL"])  # type: ignore[arg-type]


def test_select_slot_rejects_empty_choice_value() -> None:
    with pytest.raises(GuidanceError):
        SelectSlot(name="side", choices=("BUY", ""))


def test_select_slot_rejects_duplicate_choice() -> None:
    with pytest.raises(GuidanceError):
        SelectSlot(name="side", choices=("BUY", "BUY"))


# ---------------------------------------------------------------------------
# compile_template
# ---------------------------------------------------------------------------


def test_compile_template_regex_slot() -> None:
    t = compile_template("Confidence: {{regex 'conf' pattern='\\d\\.\\d+'}}")
    assert t.slot_names() == ("conf",)
    assert isinstance(t.compiled, re.Pattern)


def test_compile_template_select_slot() -> None:
    t = compile_template("Side: {{select 'side' options=('BUY','SELL')}}")
    assert t.slot_names() == ("side",)


def test_compile_template_mixed_slots() -> None:
    t = compile_template(
        "Side: {{select 'side' options=('BUY','SELL')}}; "
        "Conf: {{regex 'conf' pattern='\\d\\.\\d+'}}"
    )
    assert t.slot_names() == ("side", "conf")
    assert isinstance(t.slots[0], SelectSlot)
    assert isinstance(t.slots[1], RegexSlot)


def test_compile_template_preserves_literal_text() -> None:
    t = compile_template("Side: {{select 'side' options=('BUY','SELL')}}!")
    m = t.compiled.match("Side: BUY!")
    assert m is not None
    assert m.group("side") == "BUY"


def test_compile_template_escapes_special_chars_in_literals() -> None:
    t = compile_template("[{{regex 'x' pattern='\\d+'}}]")
    m = t.compiled.match("[42]")
    assert m is not None
    assert m.group("x") == "42"


def test_compile_template_rejects_empty() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template("")


def test_compile_template_rejects_no_slots() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template("plain text with no slots")


def test_compile_template_rejects_duplicate_slot_name() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template("{{regex 'x' pattern='a'}} {{regex 'x' pattern='b'}}")


def test_compile_template_rejects_non_str() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template(123)  # type: ignore[arg-type]


def test_compile_template_rejects_malformed_regex_args() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template("{{regex 'x' broken='a'}}")


def test_compile_template_rejects_malformed_select_args() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template("{{select 'x' broken='a'}}")


def test_compile_template_rejects_select_empty_options_tuple() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template("{{select 'x' options=()}}")


def test_compile_template_rejects_select_unquoted_option() -> None:
    with pytest.raises(GuidanceParseError):
        compile_template("{{select 'x' options=(BUY,SELL)}}")


# ---------------------------------------------------------------------------
# match_completion
# ---------------------------------------------------------------------------


def test_match_completion_happy_path() -> None:
    t = compile_template(
        "Side: {{select 'side' options=('BUY','SELL')}}, "
        "Conf: {{regex 'conf' pattern='\\d\\.\\d+'}}"
    )
    out = match_completion(t, "Side: BUY, Conf: 0.62")
    assert out == {"side": "BUY", "conf": "0.62"}


def test_match_completion_preserves_slot_order() -> None:
    t = compile_template("{{regex 'a' pattern='\\d+'}}-{{regex 'b' pattern='\\d+'}}")
    out = match_completion(t, "1-2")
    assert list(out.keys()) == ["a", "b"]


def test_match_completion_rejects_template_mismatch() -> None:
    t = compile_template("Side: {{select 'side' options=('BUY','SELL')}}")
    with pytest.raises(GuidanceMatchError):
        match_completion(t, "Side: HOLD")


def test_match_completion_rejects_non_str_text() -> None:
    t = compile_template("{{regex 'x' pattern='\\d+'}}")
    with pytest.raises(GuidanceMatchError):
        match_completion(t, 123)  # type: ignore[arg-type]


def test_match_completion_rejects_non_template() -> None:
    with pytest.raises(GuidanceError):
        match_completion("not-a-template", "42")  # type: ignore[arg-type]


def test_match_completion_rejects_partial_match() -> None:
    """Regex is anchored at start; trailing garbage is fine but a
    mismatch in the literal prefix is rejected."""
    t = compile_template("Conf: {{regex 'c' pattern='\\d\\.\\d+'}}")
    with pytest.raises(GuidanceMatchError):
        match_completion(t, "Side: BUY, Conf: 0.62")


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def test_inv15_compile_three_run_byte_identical() -> None:
    template = (
        "Side: {{select 'side' options=('BUY','SELL')}}, "
        "Conf: {{regex 'conf' pattern='\\d\\.\\d+'}}"
    )
    a = compile_template(template)
    b = compile_template(template)
    c = compile_template(template)
    assert a.slot_names() == b.slot_names() == c.slot_names()
    assert a.compiled.pattern == b.compiled.pattern == c.compiled.pattern


def test_inv15_match_three_run_byte_identical() -> None:
    t = compile_template(
        "Side: {{select 'side' options=('BUY','SELL')}}, "
        "Conf: {{regex 'conf' pattern='\\d\\.\\d+'}}"
    )
    text = "Side: BUY, Conf: 0.62"
    a = match_completion(t, text)
    b = match_completion(t, text)
    c = match_completion(t, text)
    assert a == b == c
    assert list(a.items()) == list(b.items()) == list(c.items())


# ---------------------------------------------------------------------------
# Guidance lazy seam
# ---------------------------------------------------------------------------


def test_enable_guidance_factory_rejects_none_model() -> None:
    with pytest.raises(GuidanceError):
        enable_guidance_factory(None)


def test_enable_guidance_factory_lazy_seam_when_missing() -> None:
    """Module imports cleanly without guidance installed; the factory
    must raise :class:`ModuleNotFoundError` only at call time."""
    try:
        import guidance  # noqa: F401
    except ModuleNotFoundError:
        with pytest.raises(ModuleNotFoundError):
            enable_guidance_factory(object())
    else:
        pytest.skip("guidance is installed; lazy-seam path unobservable")


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return inspect.getsource(ga)


def test_no_top_level_guidance_import() -> None:
    for line in _module_source().splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("import guidance", "from guidance")):
            indent = len(line) - len(stripped)
            if indent == 0:
                raise AssertionError(f"guidance must be a lazy seam — no top-level: {line!r}")


def test_no_top_level_transformers_or_llama_cpp_import() -> None:
    for line in _module_source().splitlines():
        if line.startswith(
            (
                "import transformers",
                "from transformers",
                "import llama_cpp",
                "from llama_cpp",
            )
        ):
            raise AssertionError(
                f"transformers/llama_cpp must be lazy seams — no top-level: {line!r}"
            )


def test_no_top_level_torch_or_numpy_import() -> None:
    for line in _module_source().splitlines():
        if line.startswith(("import torch", "from torch", "import numpy", "from numpy")):
            raise AssertionError(f"torch/numpy must be lazy seams — no top-level: {line!r}")


def test_no_forbidden_top_level_imports() -> None:
    forbidden = (
        "import time",
        "from time",
        "import datetime",
        "from datetime",
        "import random",
        "from random",
        "import asyncio",
        "from asyncio",
        "import requests",
        "from requests",
    )
    for line in _module_source().splitlines():
        for bad in forbidden:
            if line.startswith(bad):
                raise AssertionError(f"forbidden top-level import: {line!r}")


def test_no_typed_event_constructors() -> None:
    forbidden_ctors = (
        "SignalEvent(",
        "ExecutionEvent(",
        "ExecutionIntent(",
        "HazardEvent(",
        "LearningUpdate(",
        "PatchProposal(",
    )
    src = _module_source()
    for ctor in forbidden_ctors:
        assert ctor not in src, f"adapter must not construct typed events: {ctor!r}"
