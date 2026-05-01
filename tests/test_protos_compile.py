"""
tests/test_protos_compile.py

Acceptance test for ``contracts/`` (Tier-0 polyglot manifest §4).

Invariants verified here:

  (1) Every .proto file in ``contracts/`` parses and codegens
      cleanly via ``grpc_tools.protoc``. A syntactically broken
      contract must not be mergeable.
  (2) The generated Python stubs are importable and expose the
      top-level messages declared in each file.
  (3) Field numbers on cross-domain messages never silently shift —
      a concrete set of canonical field numbers is pinned so a
      careless edit is caught by CI.

The test generates stubs into a temp directory; it does not rely on
any checked-in generated files. That matches the ``gen_protos.sh``
contract: generated code is never committed.
"""
from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTRACTS_DIR = REPO_ROOT / "contracts"

# Every .proto file MUST be present and non-empty.
EXPECTED_PROTOS = [
    "market.proto",
    "execution.proto",
    "governance.proto",
    "system.proto",
    "ledger.proto",
]


@pytest.fixture(scope="module")
def generated_dir() -> pathlib.Path:
    """Generate Python stubs for every .proto into a tmp dir."""
    grpc_tools = pytest.importorskip("grpc_tools.protoc")  # noqa: F841
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="dixvision-protos-"))
    (tmp / "__init__.py").write_text("")
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={CONTRACTS_DIR}",
        f"--python_out={tmp}",
        *[str(CONTRACTS_DIR / name) for name in EXPECTED_PROTOS],
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"protoc failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    return tmp


def test_every_expected_proto_exists_and_is_nonempty() -> None:
    for name in EXPECTED_PROTOS:
        p = CONTRACTS_DIR / name
        assert p.exists(), f"missing contract: {name}"
        assert p.stat().st_size > 0, f"empty contract: {name}"
        # Smoke-check: every file must declare a package and syntax.
        text = p.read_text()
        assert 'syntax = "proto3";' in text, f"{name} missing syntax"
        assert "package dixvision.v42_2." in text, f"{name} missing package"


def test_protoc_generates_importable_python_stubs(
    generated_dir: pathlib.Path,
) -> None:
    """Every .proto must yield an importable ``<name>_pb2.py``."""
    sys.path.insert(0, str(generated_dir))
    try:
        for name in EXPECTED_PROTOS:
            stem = name.replace(".proto", "")
            module_name = f"{stem}_pb2"
            stub_path = generated_dir / f"{module_name}.py"
            assert stub_path.exists(), f"missing stub: {stub_path}"
            # Fresh import (module_name is unique per .proto).
            spec = importlib.util.spec_from_file_location(module_name, stub_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(generated_dir))


def test_top_level_messages_are_exposed(generated_dir: pathlib.Path) -> None:
    """Sanity: each .proto must export at least its headline messages."""
    sys.path.insert(0, str(generated_dir))
    try:
        expectations: dict[str, list[str]] = {
            "market_pb2":     ["Tick", "Quote", "MarketFrame"],
            "execution_pb2":  ["OrderIntent", "OrderEvent", "Fill", "AdapterState"],
            "governance_pb2": ["PolicyDecision", "ApprovalRequest", "ApprovalResponse", "ModeTransition", "RiskConstraints"],
            "system_pb2":     ["SystemHazard", "KillSwitchEvent", "ConfigChangeEvent", "FeatureFlagDelta", "HeartBeat"],
            "ledger_pb2":     ["LedgerEvent", "Snapshot", "LedgerHead", "Cursor"],
        }
        for mod_name, messages in expectations.items():
            stub_path = generated_dir / f"{mod_name}.py"
            spec = importlib.util.spec_from_file_location(mod_name, stub_path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for m in messages:
                assert hasattr(module, m), f"{mod_name} is missing message {m}"
    finally:
        sys.path.remove(str(generated_dir))


def test_canonical_field_numbers_are_pinned(generated_dir: pathlib.Path) -> None:
    """Backward-compat gate (Tier-0 Step 15).

    A deliberate, hand-maintained list of (message, field_name, number)
    triples. If any tuple drifts, the contract has silently shifted a
    field number — reject the PR.

    Adding new fields is fine; they just don't appear here until the
    next PR pins them. Changing an existing number here requires an
    explicit contract-diff review.
    """
    sys.path.insert(0, str(generated_dir))
    try:
        import execution_pb2  # type: ignore
        import governance_pb2  # type: ignore
        import ledger_pb2  # type: ignore
        import market_pb2  # type: ignore
        import system_pb2  # type: ignore

        # (module, Message, field_name, expected_number)
        pins = [
            (market_pb2,     "Tick",            "sequence", 1),
            (market_pb2,     "Tick",            "wall_ns", 2),
            (market_pb2,     "Tick",            "venue", 3),
            (execution_pb2,  "OrderIntent",     "intent_id", 1),
            (execution_pb2,  "OrderIntent",     "risk_version_used", 15),
            (execution_pb2,  "OrderEvent",      "event_id", 1),
            (execution_pb2,  "AdapterState",    "health", 4),
            (governance_pb2, "PolicyDecision",  "decision_id", 1),
            (governance_pb2, "PolicyDecision",  "risk_version_used", 7),
            (governance_pb2, "ApprovalResponse","totp_proof", 6),
            (system_pb2,     "SystemHazard",    "hazard_id", 1),
            (system_pb2,     "SystemHazard",    "kind", 4),
            (system_pb2,     "KillSwitchEvent", "scope", 4),
            (ledger_pb2,     "LedgerEvent",     "sequence", 1),
            (ledger_pb2,     "LedgerEvent",     "hash_prev", 7),
            (ledger_pb2,     "LedgerEvent",     "hash_self", 8),
            (ledger_pb2,     "Snapshot",        "cursor", 2),
            (ledger_pb2,     "Snapshot",        "risk_version_id", 6),
        ]

        failures: list[str] = []
        for mod, msg_name, field_name, expected in pins:
            msg = getattr(mod, msg_name)
            descriptor = msg.DESCRIPTOR
            fields_by_name = {f.name: f.number for f in descriptor.fields}
            got = fields_by_name.get(field_name)
            if got != expected:
                failures.append(
                    f"{msg_name}.{field_name}: expected={expected} got={got}"
                )
        assert not failures, "\n".join(failures)
    finally:
        sys.path.remove(str(generated_dir))
