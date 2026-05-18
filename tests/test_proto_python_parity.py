"""Phase-6 P0-2 guard — ``contracts/events.proto`` ↔ Python parity.

``contracts/events.proto`` declares ``SystemEventKind`` for any future
polyglot consumer (Rust execution / TS dashboard wire format). Its own
header reads "Do not edit one side without the other". Drift between
Python's :class:`core.contracts.events.SystemEventKind` and the proto
enum silently breaks wire compatibility — proto consumers receive
unknown event kinds as field number ``0``.

This test fails closed if any ``SystemEventKind`` value exists in one
side but not the other.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.contracts.events import SystemEventKind

PROTO_PATH = Path(__file__).resolve().parents[1] / "contracts" / "events.proto"


def _parse_proto_enum_names() -> frozenset[str]:
    """Return the set of enum value names declared inside
    ``enum SystemEventKind { ... }`` in ``events.proto``."""

    text = PROTO_PATH.read_text(encoding="utf-8")
    block_match = re.search(
        r"enum\s+SystemEventKind\s*\{(?P<body>.*?)\}",
        text,
        flags=re.DOTALL,
    )
    assert block_match is not None, (
        f"events.proto does not contain `enum SystemEventKind` — looked under {PROTO_PATH}"
    )
    body = block_match.group("body")
    # Strip line comments first so commented-out assignments don't match.
    body_no_comments = re.sub(r"//.*", "", body)
    names = {
        m.group(1)
        for m in re.finditer(
            r"^\s*([A-Z][A-Z0-9_]*)\s*=\s*\d+\s*;",
            body_no_comments,
            flags=re.MULTILINE,
        )
    }
    return frozenset(names)


def test_python_system_event_kinds_have_proto_counterparts() -> None:
    proto_names = _parse_proto_enum_names()
    python_names = {member.value for member in SystemEventKind}
    missing_in_proto = python_names - proto_names
    assert not missing_in_proto, (
        "Phase-6 P0-2 regression: the following Python "
        "SystemEventKind values are missing from contracts/events.proto. "
        "Wire-format consumers will see field number 0 for them: "
        f"{sorted(missing_in_proto)}"
    )


# Proto-3 requires a zero-valued ``_UNSPECIFIED`` sentinel on every
# enum; Python's :class:`StrEnum` has no such concept. Exclude that
# specific value from the symmetric-parity check.
_PROTO_ONLY_SENTINELS: frozenset[str] = frozenset({"SYSTEM_EVENT_KIND_UNSPECIFIED"})


def test_proto_system_event_kinds_have_python_counterparts() -> None:
    proto_names = _parse_proto_enum_names() - _PROTO_ONLY_SENTINELS
    python_names = {member.value for member in SystemEventKind}
    missing_in_python = proto_names - python_names
    assert not missing_in_python, (
        "Phase-6 P0-2 regression: the following proto SystemEventKind "
        "values have no Python counterpart in "
        "core.contracts.events.SystemEventKind. Did proto/python drift "
        f"happen? {sorted(missing_in_python)}"
    )


def test_proto_contains_policy_state_explicitly() -> None:
    """Hard-pin POLICY_STATE specifically — the value the audit caught
    missing. This duplicates the symmetric-parity test above but keeps
    a named test row so future regressions name-and-shame this exact
    drift.
    """

    proto_names = _parse_proto_enum_names()
    assert "POLICY_STATE" in proto_names, (
        "Phase-6 P0-2 regression: POLICY_STATE missing from "
        "contracts/events.proto. Emitted by "
        "LearningEvolutionFreezePolicy.to_system_event since PR #392."
    )
