"""Tests for the immutable-core axiom registry (P0-1d)."""

from __future__ import annotations

import pytest

from immutable_core import AXIOM_REGISTRY, AxiomKind, get_axiom, is_axiom
from immutable_core.axioms import _AXIOMS


def test_registry_is_frozen():
    with pytest.raises(TypeError):
        AXIOM_REGISTRY["INV-99"] = None  # type: ignore[index]


def test_no_duplicate_ids_in_source_tuple():
    """Catch a duplicate axiom id in ``_AXIOMS`` -- silently dropped by
    dict construction otherwise."""

    assert len(_AXIOMS) == len(AXIOM_REGISTRY), (
        "_AXIOMS contains duplicate ids -- one was silently dropped by dict construction"
    )


def test_registry_ids_are_well_formed():
    for axiom_id in AXIOM_REGISTRY:
        assert axiom_id.startswith(("INV-", "SAFE-"))
        suffix = axiom_id.split("-", 1)[1]
        assert suffix.isdigit(), f"non-numeric axiom id suffix: {axiom_id}"


def test_invariants_have_invariant_kind():
    for axiom_id, axiom in AXIOM_REGISTRY.items():
        if axiom_id.startswith("INV-"):
            assert axiom.kind is AxiomKind.INVARIANT


def test_safety_axioms_have_safety_kind():
    for axiom_id, axiom in AXIOM_REGISTRY.items():
        if axiom_id.startswith("SAFE-"):
            assert axiom.kind is AxiomKind.SAFETY


def test_get_axiom_returns_registered_axiom():
    axiom = get_axiom("INV-15")
    assert axiom.id == "INV-15"
    assert axiom.kind is AxiomKind.INVARIANT
    assert "determinism" in axiom.label.lower()


def test_get_axiom_unknown_id_raises_keyerror():
    with pytest.raises(KeyError, match="unknown axiom id"):
        get_axiom("INV-9999")


def test_is_axiom_true_for_registered():
    assert is_axiom("SAFE-01")
    assert is_axiom("INV-72")


def test_is_axiom_false_for_unregistered():
    assert not is_axiom("INV-9999")
    assert not is_axiom("FOO-01")
    assert not is_axiom("")


def test_kill_switch_axiom_resolves():
    """SAFE-01 ships with kill_switch.py (P0-1b) -- registry must keep up."""

    axiom = get_axiom("SAFE-01")
    assert "kill" in axiom.label.lower()
    assert "system/kill_switch.py" in axiom.introduced_in


def test_label_is_non_empty():
    for axiom in AXIOM_REGISTRY.values():
        assert axiom.label.strip(), f"empty label on {axiom.id}"


def test_introduced_in_is_non_empty():
    for axiom in AXIOM_REGISTRY.values():
        assert axiom.introduced_in.strip(), f"empty introduced_in on {axiom.id}"
