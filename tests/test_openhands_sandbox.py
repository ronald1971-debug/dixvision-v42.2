"""Tier-C C-17 — OpenHands-shape sandboxed-code-execution surface.

Pure backend pytest covering :mod:`evolution_engine.patch_pipeline.sandbox_openhands`.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from core.contracts.patch import PatchStage, StageVerdict
from evolution_engine.patch_pipeline.sandbox_openhands import (
    DEFAULT_FORBIDDEN_COMMANDS,
    DEFAULT_FORBIDDEN_MODULES,
    DEFAULT_SANDBOX_ROOT,
    MAX_ACTION_ID_LEN,
    MAX_ACTIONS_PER_PLAN,
    MAX_ARG_COUNT,
    MAX_ARG_LEN,
    MAX_BASH_COMMAND_LEN,
    MAX_CODE_SOURCE_LEN,
    MAX_FILE_CONTENT_LEN,
    MAX_PATH_LEN,
    NEW_PIP_DEPENDENCIES,
    ActionError,
    ActionKind,
    ActionVerdict,
    BaseAction,
    BashAction,
    BoundaryError,
    CodeAction,
    CommandObservation,
    FileWriteAction,
    FileWriteObservation,
    Observation,
    OpenHandsSandboxStage,
    PlanError,
    SandboxActionResult,
    SandboxBoundary,
    SandboxError,
    SandboxPlan,
    SandboxPlanValidator,
    enable_openhands_factory,
)

MODULE_PATH = Path("evolution_engine/patch_pipeline/sandbox_openhands.py")


# ---------------------------------------------------------------------------
# AST guards
# ---------------------------------------------------------------------------


def _parse_module() -> ast.Module:
    text = MODULE_PATH.read_text(encoding="utf-8")
    return ast.parse(text, filename=str(MODULE_PATH))


def _top_level_imports(tree: ast.Module) -> set[str]:
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module.split(".")[0])
    return out


def test_no_forbidden_top_level_imports() -> None:
    tree = _parse_module()
    top = _top_level_imports(tree)
    forbidden = {
        "openhands",
        "docker",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "ctypes",
        "asyncio",
        "selectors",
        "multiprocessing",
        "threading",
        "signal",
        "psutil",
        "time",
        "datetime",
        "random",
        "secrets",
        "numpy",
        "torch",
    }
    leak = forbidden & top
    assert leak == set(), f"forbidden top-level imports: {leak}"


def test_only_core_contracts_runtime_import() -> None:
    tree = _parse_module()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not mod:
                continue
            if mod.startswith("core.contracts.patch"):
                continue
            if mod in {
                "collections.abc",
                "enum",
                "types",
                "typing",
                "dataclasses",
                "hashlib",
                "re",
                "__future__",
            }:
                continue
            raise AssertionError(
                f"sandbox_openhands.py must not import {mod!r} \u2014 "
                "only core.contracts.patch + stdlib allowed"
            )


def test_no_b1_cross_engine_imports() -> None:
    tree = _parse_module()
    top = _top_level_imports(tree)
    forbidden_engines = {
        "execution_engine",
        "intelligence_engine",
        "learning_engine",
        "governance_engine",
        "system_engine",
        "state",
    }
    leak = forbidden_engines & top
    assert leak == set(), f"B1 violation: sandbox_openhands.py must not cross-import {leak}"


def test_no_wall_clock_calls_anywhere() -> None:
    tree = _parse_module()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func)
            for bad in (
                "time.time",
                "time.time_ns",
                "time.monotonic",
                "time.monotonic_ns",
                "datetime.now",
                "datetime.utcnow",
                "random.",
                "secrets.token",
                "os.urandom",
            ):
                assert bad not in func_str, f"forbidden runtime call: {func_str}"


def test_no_typed_bus_event_constructors() -> None:
    tree = _parse_module()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_str = ast.unparse(node.func)
            for bad in (
                "SignalEvent",
                "ExecutionIntent",
                "ExecutionEvent",
                "HazardEvent",
                "RiskSnapshot",
                "PatchProposal",
                "GovernanceDecision",
            ):
                assert bad not in func_str, (
                    f"C-17 must not construct typed runtime events: {func_str}"
                )


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


def test_module_constants_pinned() -> None:
    assert NEW_PIP_DEPENDENCIES == ("openhands-ai",)
    assert DEFAULT_SANDBOX_ROOT == "/tmp/dix_sandbox/"
    assert MAX_ACTIONS_PER_PLAN == 64
    assert MAX_ACTION_ID_LEN == 64
    assert MAX_BASH_COMMAND_LEN == 256
    assert MAX_CODE_SOURCE_LEN == 65_536
    assert MAX_FILE_CONTENT_LEN == 1_048_576
    assert MAX_PATH_LEN == 512
    assert MAX_ARG_COUNT == 32
    assert MAX_ARG_LEN == 256


def test_default_forbidden_lists_cover_directive_clauses() -> None:
    must_have_modules = {
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "httpx",
        "ctypes",
        "os",
        "sys",
        "shutil",
        "pathlib",
        "asyncio",
        "multiprocessing",
        "threading",
        "psutil",
    }
    assert must_have_modules.issubset(set(DEFAULT_FORBIDDEN_MODULES))
    must_have_commands = {
        "rm",
        "sudo",
        "chmod",
        "chown",
        "curl",
        "wget",
        "ssh",
        "docker",
        "kubectl",
        "kill",
        "reboot",
    }
    assert must_have_commands.issubset(set(DEFAULT_FORBIDDEN_COMMANDS))


# ---------------------------------------------------------------------------
# Action value objects
# ---------------------------------------------------------------------------


def test_action_subclasses_frozen_and_slotted() -> None:
    for cls in (BaseAction, CodeAction, BashAction, FileWriteAction):
        assert cls.__dataclass_params__.frozen is True, cls
        assert hasattr(cls, "__slots__"), cls


def test_code_action_kind_locked() -> None:
    a = CodeAction(id="x", ts_ns=0, source="pass")
    assert a.kind is ActionKind.CODE
    assert a.KIND is ActionKind.CODE
    assert a.language == "python"


def test_bash_action_kind_locked() -> None:
    a = BashAction(id="x", ts_ns=0, command="ls")
    assert a.kind is ActionKind.BASH
    assert a.KIND is ActionKind.BASH


def test_file_write_action_kind_locked() -> None:
    a = FileWriteAction(id="x", ts_ns=0, path="/tmp/dix_sandbox/f", content="")
    assert a.kind is ActionKind.FILE_WRITE
    assert a.KIND is ActionKind.FILE_WRITE


def test_action_id_required() -> None:
    for kwargs in (
        {"id": "", "ts_ns": 0, "source": "pass"},
        {"id": "1bad", "ts_ns": 0, "source": "pass"},
        {"id": "with space", "ts_ns": 0, "source": "pass"},
        {"id": "x" * (MAX_ACTION_ID_LEN + 1), "ts_ns": 0, "source": "pass"},
    ):
        with pytest.raises(ActionError):
            CodeAction(**kwargs)


def test_action_ts_ns_must_be_non_negative_int() -> None:
    with pytest.raises(ActionError):
        CodeAction(id="x", ts_ns=-1, source="pass")
    with pytest.raises(ActionError):
        CodeAction(id="x", ts_ns=True, source="pass")  # type: ignore[arg-type]


def test_code_action_language_locked_to_python() -> None:
    with pytest.raises(ActionError):
        CodeAction(id="x", ts_ns=0, language="javascript", source="")
    with pytest.raises(ActionError):
        CodeAction(id="x", ts_ns=0, language="", source="")


def test_code_action_source_max_len() -> None:
    huge = "a" * (MAX_CODE_SOURCE_LEN + 1)
    with pytest.raises(ActionError):
        CodeAction(id="x", ts_ns=0, source=huge)


def test_bash_action_rejects_shell_metacharacters() -> None:
    for cmd in ("ls;rm", "ls|cat", "ls&cat", "ls`x`", "ls$VAR"):
        with pytest.raises(ActionError):
            BashAction(id="x", ts_ns=0, command=cmd)


def test_bash_action_args_must_be_tuple_of_str() -> None:
    with pytest.raises(ActionError):
        BashAction(id="x", ts_ns=0, command="ls", args=["a"])  # type: ignore[arg-type]
    with pytest.raises(ActionError):
        BashAction(id="x", ts_ns=0, command="ls", args=(1,))  # type: ignore[arg-type]


def test_bash_action_args_max_count() -> None:
    too_many = tuple(str(i) for i in range(MAX_ARG_COUNT + 1))
    with pytest.raises(ActionError):
        BashAction(id="x", ts_ns=0, command="ls", args=too_many)


def test_file_write_action_path_required() -> None:
    with pytest.raises(ActionError):
        FileWriteAction(id="x", ts_ns=0, path="", content="")
    huge_path = "/tmp/dix_sandbox/" + "a" * MAX_PATH_LEN
    with pytest.raises(ActionError):
        FileWriteAction(id="x", ts_ns=0, path=huge_path, content="")


def test_file_write_action_content_max_len() -> None:
    huge = "a" * (MAX_FILE_CONTENT_LEN + 1)
    with pytest.raises(ActionError):
        FileWriteAction(id="x", ts_ns=0, path="/tmp/dix_sandbox/f", content=huge)


# ---------------------------------------------------------------------------
# Observation value objects
# ---------------------------------------------------------------------------


def test_observation_subclasses_frozen_and_slotted() -> None:
    for cls in (Observation, CommandObservation, FileWriteObservation):
        assert cls.__dataclass_params__.frozen is True, cls
        assert hasattr(cls, "__slots__"), cls


def test_command_observation_validates_fields() -> None:
    o = CommandObservation(
        action_id="a",
        ts_ns=1,
        stdout="ok",
        stderr="",
        exit_code=0,
        truncated=False,
    )
    assert o.stdout == "ok"
    with pytest.raises(ActionError):
        CommandObservation(action_id="", ts_ns=0)
    with pytest.raises(ActionError):
        CommandObservation(
            action_id="a",
            ts_ns=0,
            exit_code=True,  # type: ignore[arg-type]
        )


def test_file_write_observation_validates_fields() -> None:
    o = FileWriteObservation(
        action_id="a",
        ts_ns=1,
        path="/tmp/dix_sandbox/x",
        bytes_written=5,
        ok=True,
    )
    assert o.bytes_written == 5
    with pytest.raises(ActionError):
        FileWriteObservation(
            action_id="a",
            ts_ns=0,
            path="/tmp/dix_sandbox/x",
            bytes_written=-1,
            ok=True,
        )


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


def test_boundary_defaults_match_directive() -> None:
    b = SandboxBoundary()
    assert b.root == "/tmp/dix_sandbox/"
    assert b.max_actions == MAX_ACTIONS_PER_PLAN
    assert b.forbidden_commands == DEFAULT_FORBIDDEN_COMMANDS
    assert b.forbidden_modules == DEFAULT_FORBIDDEN_MODULES


def test_boundary_root_must_be_tmp_prefixed() -> None:
    with pytest.raises(BoundaryError):
        SandboxBoundary(root="/var/tmp/sandbox/")
    with pytest.raises(BoundaryError):
        SandboxBoundary(root="/home/operator/")
    with pytest.raises(BoundaryError):
        SandboxBoundary(root="/tmp/sandbox")  # no trailing slash
    with pytest.raises(BoundaryError):
        SandboxBoundary(root="/tmp/../etc/")


def test_boundary_accept_path_inside_root() -> None:
    b = SandboxBoundary()
    assert b.accept_path("/tmp/dix_sandbox/work/x.py") is True
    assert b.accept_path("/tmp/dix_sandbox/") is True


def test_boundary_accept_path_outside_root_rejected() -> None:
    b = SandboxBoundary()
    assert b.accept_path("/etc/passwd") is False
    assert b.accept_path("/tmp/other/") is False
    assert b.accept_path("relative/x") is False


def test_boundary_accept_path_traversal_rejected() -> None:
    b = SandboxBoundary()
    assert b.accept_path("/tmp/dix_sandbox/../etc/passwd") is False
    assert b.accept_path("/tmp/dix_sandbox/~root") is False
    assert b.accept_path("/tmp/dix_sandbox/$HOME/x") is False


def test_boundary_forbidden_command_head() -> None:
    b = SandboxBoundary()
    assert b.accept_command_head("ls") is True
    assert b.accept_command_head("pytest") is True
    assert b.accept_command_head("rm") is False
    assert b.accept_command_head("notarealcommand") is False


def test_boundary_detects_forbidden_imports() -> None:
    b = SandboxBoundary()
    assert b.code_imports_forbidden_module("import subprocess\nprint(1)") is True
    assert b.code_imports_forbidden_module("from urllib.request import urlopen\n") is True
    assert b.code_imports_forbidden_module("  import os.path\nprint(1)\n") is True
    assert b.code_imports_forbidden_module("# import subprocess\nprint(1)\n") is False
    assert b.code_imports_forbidden_module("print(1)") is False


def test_boundary_invalid_max_actions() -> None:
    with pytest.raises(BoundaryError):
        SandboxBoundary(max_actions=0)
    with pytest.raises(BoundaryError):
        SandboxBoundary(max_actions=MAX_ACTIONS_PER_PLAN + 1)
    with pytest.raises(BoundaryError):
        SandboxBoundary(max_actions=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Plan validator
# ---------------------------------------------------------------------------


def _default_validator() -> SandboxPlanValidator:
    return SandboxPlanValidator(boundary=SandboxBoundary())


def test_validator_accepts_clean_code_action() -> None:
    v = _default_validator()
    plan = v.validate([CodeAction(id="a1", ts_ns=0, source="print(1)")])
    assert plan.passed is True
    assert plan.accepted_count == 1
    assert plan.results[0].verdict is ActionVerdict.ACCEPTED


def test_validator_rejects_code_with_forbidden_import() -> None:
    v = _default_validator()
    plan = v.validate([CodeAction(id="a1", ts_ns=0, source="import subprocess")])
    assert plan.results[0].verdict is ActionVerdict.REJECTED
    assert "forbidden module" in plan.results[0].reason


def test_validator_accepts_bash_inside_sandbox_path() -> None:
    v = _default_validator()
    plan = v.validate(
        [
            BashAction(
                id="a1",
                ts_ns=0,
                command="ls",
                args=("/tmp/dix_sandbox/work",),
            )
        ]
    )
    assert plan.results[0].verdict is ActionVerdict.ACCEPTED


def test_validator_reviews_bash_path_outside_sandbox() -> None:
    v = _default_validator()
    plan = v.validate([BashAction(id="a1", ts_ns=0, command="ls", args=("/etc/passwd",))])
    assert plan.results[0].verdict is ActionVerdict.REVIEW_REQUIRED
    assert plan.requires_governance_review is True


def test_validator_rejects_forbidden_bash_head() -> None:
    v = _default_validator()
    plan = v.validate([BashAction(id="a1", ts_ns=0, command="rm", args=())])
    assert plan.results[0].verdict is ActionVerdict.REJECTED


def test_validator_rejects_disallowed_bash_head() -> None:
    v = _default_validator()
    plan = v.validate([BashAction(id="a1", ts_ns=0, command="strangecmd", args=())])
    assert plan.results[0].verdict is ActionVerdict.REJECTED


def test_validator_accepts_file_write_inside_sandbox() -> None:
    v = _default_validator()
    plan = v.validate(
        [
            FileWriteAction(
                id="a1",
                ts_ns=0,
                path="/tmp/dix_sandbox/out.txt",
                content="hi",
            )
        ]
    )
    assert plan.results[0].verdict is ActionVerdict.ACCEPTED


def test_validator_rejects_file_write_outside_sandbox() -> None:
    v = _default_validator()
    plan = v.validate(
        [
            FileWriteAction(
                id="a1",
                ts_ns=0,
                path="/etc/passwd",
                content="boom",
            )
        ]
    )
    assert plan.results[0].verdict is ActionVerdict.REJECTED


def test_validator_rejects_file_write_with_traversal() -> None:
    v = _default_validator()
    plan = v.validate(
        [
            FileWriteAction(
                id="a1",
                ts_ns=0,
                path="/tmp/dix_sandbox/../etc/passwd",
                content="boom",
            )
        ]
    )
    assert plan.results[0].verdict is ActionVerdict.REJECTED


def test_validator_rejects_duplicate_action_ids() -> None:
    v = _default_validator()
    with pytest.raises(PlanError):
        v.validate(
            [
                CodeAction(id="dup", ts_ns=0, source="print(1)"),
                CodeAction(id="dup", ts_ns=1, source="print(2)"),
            ]
        )


def test_validator_rejects_overlong_action_list() -> None:
    boundary = SandboxBoundary(max_actions=2)
    v = SandboxPlanValidator(boundary=boundary)
    with pytest.raises(PlanError):
        v.validate([CodeAction(id=f"a{i}", ts_ns=i, source="pass") for i in range(3)])


def test_validator_rejects_non_action_entries() -> None:
    v = _default_validator()
    with pytest.raises(ActionError):
        v.validate([object()])  # type: ignore[list-item]


def test_validator_counter_invariants() -> None:
    v = _default_validator()
    plan = v.validate(
        [
            CodeAction(id="a1", ts_ns=0, source="print(1)"),
            CodeAction(id="a2", ts_ns=1, source="import subprocess"),
            BashAction(
                id="a3",
                ts_ns=2,
                command="ls",
                args=("/etc/passwd",),
            ),
        ]
    )
    assert (plan.accepted_count + plan.review_count + plan.rejected_count) == 3
    assert plan.accepted_count == 1
    assert plan.review_count == 1
    assert plan.rejected_count == 1
    assert plan.passed is False
    assert plan.requires_governance_review is True


def test_validator_empty_plan_passes_trivially() -> None:
    v = _default_validator()
    plan = v.validate([])
    assert plan.passed is True
    assert plan.accepted_count == 0
    assert plan.requires_governance_review is False


# ---------------------------------------------------------------------------
# INV-15 byte-identical determinism
# ---------------------------------------------------------------------------


def _build_mixed_actions() -> tuple[BaseAction, ...]:
    return (
        CodeAction(id="a1", ts_ns=0, source="print(1)"),
        CodeAction(id="a2", ts_ns=1, source="import subprocess"),
        BashAction(id="a3", ts_ns=2, command="ls", args=("/tmp/dix_sandbox/x",)),
        BashAction(id="a4", ts_ns=3, command="ls", args=("/etc/passwd",)),
        BashAction(id="a5", ts_ns=4, command="rm", args=()),
        FileWriteAction(id="a6", ts_ns=5, path="/tmp/dix_sandbox/out.txt", content="hi"),
        FileWriteAction(id="a7", ts_ns=6, path="/etc/passwd", content="boom"),
    )


def test_inv15_three_run_byte_identical() -> None:
    v = _default_validator()
    actions = _build_mixed_actions()
    digests = {v.validate(actions).digest for _ in range(3)}
    assert len(digests) == 1


def test_inv15_action_id_change_changes_digest() -> None:
    v = _default_validator()
    actions = _build_mixed_actions()
    d1 = v.validate(actions).digest
    altered = (CodeAction(id="a1_renamed", ts_ns=0, source="print(1)"),) + actions[1:]
    d2 = v.validate(altered).digest
    assert d1 != d2


def test_inv15_kind_change_changes_digest() -> None:
    v = _default_validator()
    actions = _build_mixed_actions()
    d1 = v.validate(actions).digest
    altered = (BashAction(id="a1", ts_ns=0, command="echo", args=("hello",)),) + actions[1:]
    d2 = v.validate(altered).digest
    assert d1 != d2


def test_inv15_reasons_included_in_digest() -> None:
    v = _default_validator()
    a = (CodeAction(id="a1", ts_ns=0, source="import subprocess"),)
    b = (CodeAction(id="a1", ts_ns=0, source="import socket"),)
    d1 = v.validate(a).digest
    d2 = v.validate(b).digest
    # both rejected, but for different forbidden imports... actually
    # the reason text is the same. The CodeAction.source contents
    # differ, so the digest still differs.
    assert d1 != d2


# ---------------------------------------------------------------------------
# Stage adapter
# ---------------------------------------------------------------------------


def test_stage_emits_canonical_stage_verdict_clean() -> None:
    v = _default_validator()
    stage = OpenHandsSandboxStage(validator=v)
    plan, sv = stage.evaluate(ts_ns=42, actions=[CodeAction(id="a1", ts_ns=0, source="pass")])
    assert isinstance(sv, StageVerdict)
    assert sv.passed is True
    assert sv.stage is PatchStage.SANDBOX
    assert sv.ts_ns == 42
    assert sv.meta["accepted"] == "1"
    assert sv.meta["review"] == "0"
    assert sv.meta["rejected"] == "0"
    assert sv.meta["digest"] == plan.digest.hex()


def test_stage_emits_failure_when_review_required() -> None:
    v = _default_validator()
    stage = OpenHandsSandboxStage(validator=v)
    _, sv = stage.evaluate(
        ts_ns=0,
        actions=[
            BashAction(
                id="a1",
                ts_ns=0,
                command="cat",
                args=("/etc/passwd",),
            )
        ],
    )
    assert sv.passed is False
    assert sv.meta["review"] == "1"


def test_stage_emits_failure_when_rejected() -> None:
    v = _default_validator()
    stage = OpenHandsSandboxStage(validator=v)
    _, sv = stage.evaluate(
        ts_ns=0,
        actions=[BashAction(id="a1", ts_ns=0, command="rm")],
    )
    assert sv.passed is False
    assert sv.meta["rejected"] == "1"


def test_stage_name_and_spec_id_pinned() -> None:
    v = _default_validator()
    stage = OpenHandsSandboxStage(validator=v)
    assert stage.name == "sandbox_openhands"
    assert stage.spec_id == "C-17"


def test_stage_validator_attribute_round_trip() -> None:
    v = _default_validator()
    stage = OpenHandsSandboxStage(validator=v)
    assert stage.validator is v


def test_stage_rejects_invalid_validator() -> None:
    with pytest.raises(BoundaryError):
        OpenHandsSandboxStage(validator=object())  # type: ignore[arg-type]


def test_stage_rejects_negative_ts_ns() -> None:
    v = _default_validator()
    stage = OpenHandsSandboxStage(validator=v)
    with pytest.raises(ActionError):
        stage.evaluate(ts_ns=-1, actions=[])


# ---------------------------------------------------------------------------
# Plan structural invariants
# ---------------------------------------------------------------------------


def test_plan_construct_rejects_mismatched_lengths() -> None:
    a = (CodeAction(id="a1", ts_ns=0, source="pass"),)
    r = ()
    with pytest.raises(PlanError):
        SandboxPlan(
            actions=a,
            results=r,
            digest=b"\x00" * 16,
            accepted_count=0,
            review_count=0,
            rejected_count=0,
        )


def test_plan_construct_rejects_misaligned_ids() -> None:
    a = (CodeAction(id="a1", ts_ns=0, source="pass"),)
    r = (
        SandboxActionResult(
            action_id="other",
            kind=ActionKind.CODE,
            verdict=ActionVerdict.ACCEPTED,
        ),
    )
    with pytest.raises(PlanError):
        SandboxPlan(
            actions=a,
            results=r,
            digest=b"\x00" * 16,
            accepted_count=1,
            review_count=0,
            rejected_count=0,
        )


def test_plan_construct_rejects_short_digest() -> None:
    with pytest.raises(PlanError):
        SandboxPlan(
            actions=(),
            results=(),
            digest=b"\x00" * 8,
            accepted_count=0,
            review_count=0,
            rejected_count=0,
        )


def test_plan_construct_rejects_counter_sum_mismatch() -> None:
    a = (CodeAction(id="a1", ts_ns=0, source="pass"),)
    r = (
        SandboxActionResult(
            action_id="a1",
            kind=ActionKind.CODE,
            verdict=ActionVerdict.ACCEPTED,
        ),
    )
    with pytest.raises(PlanError):
        SandboxPlan(
            actions=a,
            results=r,
            digest=b"\x00" * 16,
            accepted_count=0,
            review_count=0,
            rejected_count=0,
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_error_hierarchy() -> None:
    assert issubclass(ActionError, SandboxError)
    assert issubclass(BoundaryError, SandboxError)
    assert issubclass(PlanError, SandboxError)
    assert issubclass(SandboxError, ValueError)


# ---------------------------------------------------------------------------
# Lazy seam
# ---------------------------------------------------------------------------


def test_enable_openhands_factory_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        enable_openhands_factory()


def test_module_imports_without_openhands_installed() -> None:
    # The module is already imported at the top of the file; this is
    # a paranoia guard that a fresh import does not pull openhands.
    mod_name = "evolution_engine.patch_pipeline.sandbox_openhands"
    sys.modules.pop(mod_name, None)
    importlib.import_module(mod_name)
    assert "openhands" not in sys.modules
    assert "docker" not in sys.modules
