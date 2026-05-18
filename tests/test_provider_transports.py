"""I-22 tests: provider transports audit surface."""

from __future__ import annotations

import dataclasses
import inspect
import re

import pytest

from intelligence_engine.cognitive.chat import provider_transports as pt
from intelligence_engine.cognitive.chat.provider_transports import (
    NEW_PIP_DEPENDENCIES,
    TRANSPORT_SPECS,
    ProviderTransportError,
    ProviderTransportSpec,
    provider_transport_audit,
    spec_for_provider,
    wired_provider_keys,
)

# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_new_pip_dependencies_declared() -> None:
    assert NEW_PIP_DEPENDENCIES == (
        "openai",
        "anthropic",
        "google-genai",
        "groq",
    )


def test_error_hierarchy() -> None:
    assert issubclass(ProviderTransportError, ValueError)


# ---------------------------------------------------------------------------
# ProviderTransportSpec
# ---------------------------------------------------------------------------


def test_spec_happy_path() -> None:
    s = ProviderTransportSpec(
        provider="openai",
        family="openai_compat",
        env_var="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1/",
        auth_scheme="bearer",
    )
    assert s.provider == "openai"


def test_spec_is_frozen() -> None:
    s = TRANSPORT_SPECS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.provider = "x"  # type: ignore[misc]


def test_spec_is_slotted() -> None:
    s = TRANSPORT_SPECS[0]
    assert "__slots__" in type(s).__dict__


@pytest.mark.parametrize(
    "field",
    ["provider", "family", "env_var", "base_url", "auth_scheme"],
)
def test_spec_rejects_empty_str(field: str) -> None:
    kwargs = {
        "provider": "openai",
        "family": "openai_compat",
        "env_var": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/",
        "auth_scheme": "bearer",
    }
    kwargs[field] = ""
    with pytest.raises(ProviderTransportError):
        ProviderTransportSpec(**kwargs)


@pytest.mark.parametrize(
    "field",
    ["provider", "family", "env_var", "base_url", "auth_scheme"],
)
def test_spec_rejects_non_str(field: str) -> None:
    kwargs = {
        "provider": "openai",
        "family": "openai_compat",
        "env_var": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1/",
        "auth_scheme": "bearer",
    }
    kwargs[field] = 123  # type: ignore[assignment]
    with pytest.raises(ProviderTransportError):
        ProviderTransportSpec(**kwargs)


def test_spec_rejects_unknown_family() -> None:
    with pytest.raises(ProviderTransportError):
        ProviderTransportSpec(
            provider="openai",
            family="wat",
            env_var="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1/",
            auth_scheme="bearer",
        )


def test_spec_rejects_unknown_auth_scheme() -> None:
    with pytest.raises(ProviderTransportError):
        ProviderTransportSpec(
            provider="openai",
            family="openai_compat",
            env_var="OPENAI_API_KEY",
            base_url="https://api.openai.com/v1/",
            auth_scheme="wat",
        )


def test_spec_rejects_non_https_url() -> None:
    with pytest.raises(ProviderTransportError):
        ProviderTransportSpec(
            provider="openai",
            family="openai_compat",
            env_var="OPENAI_API_KEY",
            base_url="http://api.openai.com/v1/",
            auth_scheme="bearer",
        )


# ---------------------------------------------------------------------------
# TRANSPORT_SPECS table
# ---------------------------------------------------------------------------


def test_transport_specs_is_tuple_of_specs() -> None:
    assert isinstance(TRANSPORT_SPECS, tuple)
    assert len(TRANSPORT_SPECS) >= 5
    for s in TRANSPORT_SPECS:
        assert isinstance(s, ProviderTransportSpec)


def test_transport_specs_provider_keys_unique() -> None:
    keys = [s.provider for s in TRANSPORT_SPECS]
    assert len(keys) == len(set(keys))


def test_transport_specs_covers_runtime_dispatcher_keys() -> None:
    """The audit table must match the production dispatcher table.

    Keys come from ``build_default_dispatch_transport`` in
    ``http_chat_transport.py``: openai / xai / deepseek / google /
    cognition.
    """
    expected = {"openai", "xai", "deepseek", "google", "cognition"}
    actual = {s.provider for s in TRANSPORT_SPECS}
    assert actual == expected


def test_transport_specs_env_vars_unique() -> None:
    env_vars = [s.env_var for s in TRANSPORT_SPECS]
    assert len(env_vars) == len(set(env_vars))


# ---------------------------------------------------------------------------
# wired_provider_keys
# ---------------------------------------------------------------------------


def test_wired_provider_keys_returns_canonical_sort() -> None:
    keys = wired_provider_keys()
    assert keys == ("cognition", "deepseek", "google", "openai", "xai")


def test_wired_provider_keys_returns_tuple() -> None:
    assert isinstance(wired_provider_keys(), tuple)


# ---------------------------------------------------------------------------
# spec_for_provider
# ---------------------------------------------------------------------------


def test_spec_for_provider_happy_path() -> None:
    s = spec_for_provider("openai")
    assert s.provider == "openai"
    assert s.family == "openai_compat"
    assert s.env_var == "OPENAI_API_KEY"


def test_spec_for_provider_unknown_raises() -> None:
    with pytest.raises(ProviderTransportError):
        spec_for_provider("does-not-exist")


def test_spec_for_provider_empty_raises() -> None:
    with pytest.raises(ProviderTransportError):
        spec_for_provider("")


def test_spec_for_provider_non_str_raises() -> None:
    with pytest.raises(ProviderTransportError):
        spec_for_provider(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# provider_transport_audit
# ---------------------------------------------------------------------------


def test_provider_transport_audit_returns_tuple_of_mappings() -> None:
    rows = provider_transport_audit()
    assert isinstance(rows, tuple)
    assert len(rows) == len(TRANSPORT_SPECS)
    for row in rows:
        assert isinstance(row, dict)
        assert set(row.keys()) == {
            "provider",
            "family",
            "env_var",
            "base_url",
            "auth_scheme",
        }


def test_provider_transport_audit_canonically_sorted() -> None:
    rows = provider_transport_audit()
    keys = [row["provider"] for row in rows]
    assert keys == sorted(keys)


def test_provider_transport_audit_rows_are_immutable() -> None:
    rows = provider_transport_audit()
    with pytest.raises(TypeError):
        rows[0]["provider"] = "x"  # type: ignore[index]


def test_provider_transport_audit_does_not_leak_secret_values() -> None:
    rows = provider_transport_audit()
    for row in rows:
        for value in row.values():
            # No row should leak anything that looks like an API token —
            # the audit only exposes env-var *names*, not values.
            assert not re.match(r"^(sk-|api_key=|Bearer ).+", value)


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def test_inv15_audit_three_run_byte_identical() -> None:
    a = provider_transport_audit()
    b = provider_transport_audit()
    c = provider_transport_audit()
    assert a == b == c
    assert [list(r.items()) for r in a] == [list(r.items()) for r in b]
    assert [list(r.items()) for r in b] == [list(r.items()) for r in c]


def test_inv15_wired_provider_keys_three_run_byte_identical() -> None:
    assert wired_provider_keys() == wired_provider_keys() == wired_provider_keys()


# ---------------------------------------------------------------------------
# AST guardrails
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return inspect.getsource(pt)


def test_no_top_level_vendor_sdk_imports() -> None:
    forbidden = (
        "import openai",
        "from openai",
        "import anthropic",
        "from anthropic",
        "import google.genai",
        "from google.genai",
        "import google.generativeai",
        "from google.generativeai",
        "import groq",
        "from groq",
    )
    src = _module_source()
    for bad in forbidden:
        # Allow inside docstrings / comments by anchoring at line start
        for line in src.splitlines():
            assert not line.startswith(bad), f"forbidden vendor SDK top-level import: {bad!r}"


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
        "import os",
        "from os",
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


def test_no_cross_engine_imports() -> None:
    forbidden = (
        "from execution_engine",
        "import execution_engine",
        "from governance_engine",
        "import governance_engine",
        "from system_engine",
        "import system_engine",
    )
    for line in _module_source().splitlines():
        for bad in forbidden:
            if line.startswith(bad):
                raise AssertionError(f"B1 violation: {line!r}")


def test_b23_exemption_wired() -> None:
    """The B23 chat-widget vendor-token lint must exempt this module.

    Adding I-22 to ``B23_PYTHON_EXEMPT_MODULES`` is what allows
    ``"openai"`` / ``"google"`` / ``"cognition"`` to appear as
    dispatch keys here without tripping B23 — the same rationale as
    ``http_chat_transport``.
    """

    from tools.authority_lint import B23_PYTHON_EXEMPT_MODULES

    assert "intelligence_engine.cognitive.chat.provider_transports" in B23_PYTHON_EXEMPT_MODULES


def test_no_urllib_import() -> None:
    """HTTP belongs in ``http_chat_transport.py``, not the audit module."""
    forbidden = (
        "import urllib",
        "from urllib",
    )
    for line in _module_source().splitlines():
        for bad in forbidden:
            if line.startswith(bad):
                raise AssertionError(
                    f"urllib must not be imported into provider_transports: {line!r}"
                )
