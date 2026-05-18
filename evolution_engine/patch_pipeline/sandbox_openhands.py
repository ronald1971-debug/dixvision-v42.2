# ADAPTED FROM: https://github.com/All-Hands-AI/OpenHands (MIT)
#
# Tier-C C-17 — OpenHands-shape sandboxed-code-execution surface.
#
# OpenHands' distinguishing surface is an **action / observation
# event loop** running inside a Docker-isolated sandbox runtime:
#
#   1. ``Action`` — typed instructions the agent emits (``CodeAction``
#      for Python, ``BashAction`` for shell, ``FileWriteAction`` for
#      file writes, etc.).
#   2. ``Sandbox`` runtime — executes one action at a time inside a
#      container, emitting a typed ``Observation`` for each
#      (``CommandObservation`` for code / bash, ``FileWriteObservation``
#      for file writes).
#   3. ``AgentController`` — drives the loop and routes observations
#      back to the agent for the next action decision.
#
# C-17 adapts that shape behind DIX contracts at
# :mod:`evolution_engine.patch_pipeline.sandbox_openhands`. The
# sandbox interface wraps **DIX patch validation**: an evolution
# proposal that wants to run code is decomposed into a typed action
# sequence; the validator checks every action against a frozen
# :class:`SandboxBoundary` (sandbox root path, forbidden commands,
# forbidden import prefixes, size caps); actions touching sensitive
# resources (network, ``subprocess``, paths outside the sandbox)
# are flagged ``REVIEW_REQUIRED`` so :class:`PatchApprovalBridge`
# can gate them on a typed :class:`OperatorConsent` envelope before
# any live execution.
#
# Authority constraints (pinned by tests):
#
#   * **ADVISORY / OFFLINE_ONLY** (INV-12) — every output is a frozen
#     value object. No :class:`SignalEvent` / :class:`ExecutionIntent` /
#     :class:`PatchProposal` / :class:`GovernanceDecision`
#     constructors anywhere; the live :class:`PatchApprovalBridge`
#     remains the sole authority for runtime patch transitions
#     (Build Compiler Spec §1.1).
#   * **INV-15** — pure dispatcher. No clock, no I/O, no PRNG, no
#     subprocess. Three independent runs with identical inputs
#     produce byte-identical :class:`SandboxPlan` instances.
#   * **All file writes limited to** ``/tmp/dix_sandbox/`` — paths
#     outside the configured boundary root are rejected outright;
#     symlink / parent-relative paths are rejected on syntactic
#     grounds (no canonicalisation, no ``os.path.realpath`` call).
#   * **gVisor / Firecracker** are the production isolation layer
#     (declared by the directive; bridged by C-73 / C-74). The
#     validator does NOT exec anything itself — it only validates
#     the typed plan.
#   * **OpenHands cloud deployment disabled** — the live ``openhands``
#     SDK is the lazy seam (only :func:`enable_openhands_factory`
#     may import it, and only inside the function body); cloud /
#     remote-runtime surfaces are intentionally not re-exported.
#   * **B1** — no execution_engine / governance_engine /
#     system_engine / intelligence_engine / learning_engine /
#     state submodule cross-imports. Only :mod:`core.contracts.patch`
#     is allowed (frozen value-object contract, B1-clean).
#   * No top-level imports of :mod:`openhands`, :mod:`docker`,
#     :mod:`subprocess`, :mod:`socket`, :mod:`urllib`,
#     :mod:`urllib.request`, :mod:`requests`, :mod:`httpx`,
#     :mod:`asyncio`, :mod:`time`, :mod:`datetime`, :mod:`random`,
#     :mod:`secrets`.
#
# NEW_PIP_DEPENDENCIES = ("openhands-ai",) — declared as the lazy
# seam for ``tools/cli.py install-c-tier``; production wiring routes
# everything through the in-memory plan validator unless an operator
# explicitly enables the live OpenHands Docker backend via
# :func:`enable_openhands_factory`.
"""C-17 sandboxed-code-execution — OpenHands-shape plan validator."""

from __future__ import annotations

import dataclasses
import hashlib
import re
from collections.abc import Iterable, Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, Final, Protocol, runtime_checkable

from core.contracts.patch import PatchStage, StageVerdict

__all__ = (
    "NEW_PIP_DEPENDENCIES",
    "SandboxError",
    "ActionError",
    "BoundaryError",
    "PlanError",
    "ActionKind",
    "ActionVerdict",
    "BaseAction",
    "CodeAction",
    "BashAction",
    "FileWriteAction",
    "Observation",
    "CommandObservation",
    "FileWriteObservation",
    "SandboxBoundary",
    "SandboxActionResult",
    "SandboxPlan",
    "SandboxPlanValidator",
    "OpenHandsSandboxStage",
    "enable_openhands_factory",
    "DEFAULT_SANDBOX_ROOT",
    "DEFAULT_FORBIDDEN_COMMANDS",
    "DEFAULT_FORBIDDEN_MODULES",
    "MAX_ACTION_ID_LEN",
    "MAX_ACTIONS_PER_PLAN",
    "MAX_BASH_COMMAND_LEN",
    "MAX_CODE_SOURCE_LEN",
    "MAX_FILE_CONTENT_LEN",
    "MAX_PATH_LEN",
    "MAX_ARG_COUNT",
    "MAX_ARG_LEN",
)


NEW_PIP_DEPENDENCIES: tuple[str, ...] = ("openhands-ai",)


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


DEFAULT_SANDBOX_ROOT: Final[str] = "/tmp/dix_sandbox/"

DEFAULT_FORBIDDEN_COMMANDS: Final[tuple[str, ...]] = (
    "rm",
    "mv",
    "dd",
    "sudo",
    "su",
    "chmod",
    "chown",
    "mount",
    "umount",
    "ssh",
    "scp",
    "rsync",
    "curl",
    "wget",
    "ftp",
    "nc",
    "ncat",
    "iptables",
    "systemctl",
    "service",
    "kill",
    "pkill",
    "killall",
    "reboot",
    "shutdown",
    "docker",
    "kubectl",
)

DEFAULT_FORBIDDEN_MODULES: Final[tuple[str, ...]] = (
    "subprocess",
    "socket",
    "urllib",
    "urllib.request",
    "requests",
    "httpx",
    "ctypes",
    "os",
    "sys",
    "shutil",
    "pathlib",
    "asyncio",
    "selectors",
    "multiprocessing",
    "threading",
    "signal",
    "psutil",
)

MAX_ACTION_ID_LEN: Final[int] = 64
MAX_ACTIONS_PER_PLAN: Final[int] = 64
MAX_BASH_COMMAND_LEN: Final[int] = 256
MAX_CODE_SOURCE_LEN: Final[int] = 65_536
MAX_FILE_CONTENT_LEN: Final[int] = 1_048_576
MAX_PATH_LEN: Final[int] = 512
MAX_ARG_COUNT: Final[int] = 32
MAX_ARG_LEN: Final[int] = 256
MAX_LANGUAGE_LEN: Final[int] = 16

SANDBOX_VERSION: Final[str] = "1"

_ACTION_ID_RE: Final[re.Pattern[str]] = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")

# Code-language whitelist mirrors OpenHands' supported runtimes.
_ALLOWED_LANGUAGES: Final[frozenset[str]] = frozenset(("python",))

# Bash command head whitelist. The validator is conservative: even
# if a head is on the allow-list, an argument scan still rejects
# anything that looks like a network-touch or filesystem-escape.
_ALLOWED_COMMAND_HEADS: Final[frozenset[str]] = frozenset(
    (
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "wc",
        "echo",
        "pwd",
        "true",
        "false",
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
    )
)

# Tokens that always escape the sandbox boundary, regardless of
# whether the rest of the path is inside the configured root.
_PATH_FORBIDDEN_TOKENS: Final[tuple[str, ...]] = (
    "..",
    "~",
    "$",
)

# Pattern used to spot bash arguments that ``look like a path`` and
# therefore need to be checked against the sandbox boundary. The
# pattern is intentionally syntactic — the validator never resolves
# paths against the live filesystem.
_BASH_PATHISH_ARG_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:[/~]|\.\.[/\\]|\$\{?[A-Za-z_])|[/\\]"
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SandboxError(ValueError):
    """Base class for C-17 sandbox errors."""


class ActionError(SandboxError):
    """Raised when a typed action value object is malformed."""


class BoundaryError(SandboxError):
    """Raised when a :class:`SandboxBoundary` is malformed."""


class PlanError(SandboxError):
    """Raised when a :class:`SandboxPlan` is malformed."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ActionKind(StrEnum):
    """The concrete action kinds the validator understands."""

    CODE = "CODE"
    BASH = "BASH"
    FILE_WRITE = "FILE_WRITE"


class ActionVerdict(StrEnum):
    """Per-action validation outcome.

    * ``ACCEPTED`` — the action is safe to run unattended inside
      the sandbox.
    * ``REVIEW_REQUIRED`` — the action is safe **only after**
      explicit operator approval (the orchestrator surfaces it to
      :class:`PatchApprovalBridge` for a typed consent envelope).
    * ``REJECTED`` — the action escapes the boundary and must
      never run; the orchestrator transitions the patch to
      :data:`PatchStage.REJECTED`.
    """

    ACCEPTED = "ACCEPTED"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_id(name: str, value: str, max_len: int) -> None:
    if not isinstance(value, str) or not value:
        raise ActionError(f"{name} must be a non-empty str")
    if len(value) > max_len:
        raise ActionError(f"{name} length > {max_len}")
    if not _ACTION_ID_RE.fullmatch(value):
        raise ActionError(f"{name} must match [A-Za-z_][A-Za-z0-9_.-]*")


def _validate_ts_ns(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ActionError("ts_ns must be int")
    if value < 0:
        raise ActionError("ts_ns must be non-negative")


def _freeze_meta(
    meta: Mapping[str, str],
) -> Mapping[str, str]:
    if not isinstance(meta, Mapping):
        raise ActionError("meta must be a Mapping[str, str]")
    out: dict[str, str] = {}
    for k in sorted(meta.keys()):
        if not isinstance(k, str) or not k:
            raise ActionError(f"meta key must be a non-empty str, got {k!r}")
        v = meta[k]
        if not isinstance(v, str):
            raise ActionError(f"meta[{k!r}] must be str, got {type(v).__name__}")
        if len(k) > MAX_ARG_LEN or len(v) > MAX_ARG_LEN:
            raise ActionError(f"meta entry too long: {k!r}")
        out[k] = v
    return MappingProxyType(out)


# ---------------------------------------------------------------------------
# Action value objects
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class BaseAction:
    """Common shape for every typed action.

    Mirrors OpenHands' ``Action`` base class but drops the
    ``thought`` / ``agent`` / ``timestamp`` mutable fields in favour
    of:

    * ``id`` — stable string identifier (caller-supplied).
    * ``ts_ns`` — deterministic timestamp from the orchestrator.
    * ``kind`` — :class:`ActionKind` discriminator (subclasses pin
      via the ``KIND`` :class:`ClassVar`; the instance attribute is
      read-only via the ``kind`` property).
    """

    id: str
    ts_ns: int

    KIND: ClassVar[ActionKind] = ActionKind.CODE

    def __post_init__(self) -> None:
        _validate_id("Action.id", self.id, MAX_ACTION_ID_LEN)
        _validate_ts_ns(self.ts_ns)
        if not isinstance(self.KIND, ActionKind):
            raise ActionError(f"Action.KIND must be ActionKind, got {type(self.KIND).__name__}")

    @property
    def kind(self) -> ActionKind:
        return self.KIND


@dataclasses.dataclass(frozen=True, slots=True)
class CodeAction(BaseAction):
    """OpenHands-shape ``CodeAction``: execute source code."""

    language: str = "python"
    source: str = ""

    KIND: ClassVar[ActionKind] = ActionKind.CODE

    def __post_init__(self) -> None:
        BaseAction.__post_init__(self)
        if not isinstance(self.language, str) or not self.language:
            raise ActionError("CodeAction.language must be a non-empty str")
        if len(self.language) > MAX_LANGUAGE_LEN:
            raise ActionError(f"CodeAction.language length > {MAX_LANGUAGE_LEN}")
        if self.language not in _ALLOWED_LANGUAGES:
            raise ActionError(f"CodeAction.language not allowed: {self.language!r}")
        if not isinstance(self.source, str):
            raise ActionError(f"CodeAction.source must be str, got {type(self.source).__name__}")
        if len(self.source) > MAX_CODE_SOURCE_LEN:
            raise ActionError(f"CodeAction.source length > {MAX_CODE_SOURCE_LEN}")


@dataclasses.dataclass(frozen=True, slots=True)
class BashAction(BaseAction):
    """OpenHands-shape ``BashAction``: execute a shell command."""

    command: str = ""
    args: tuple[str, ...] = ()

    KIND: ClassVar[ActionKind] = ActionKind.BASH

    def __post_init__(self) -> None:
        BaseAction.__post_init__(self)
        if not isinstance(self.command, str) or not self.command:
            raise ActionError("BashAction.command must be a non-empty str")
        if len(self.command) > MAX_BASH_COMMAND_LEN:
            raise ActionError(f"BashAction.command length > {MAX_BASH_COMMAND_LEN}")
        # Disallow shell metacharacters in the head; arguments are
        # validated separately by the boundary check.
        if any(c in self.command for c in (";", "|", "&", "`", "$")):
            raise ActionError("BashAction.command must not contain shell metacharacters")
        if not isinstance(self.args, tuple):
            raise ActionError("BashAction.args must be a tuple[str, ...]")
        if len(self.args) > MAX_ARG_COUNT:
            raise ActionError(f"BashAction.args length > {MAX_ARG_COUNT}")
        for a in self.args:
            if not isinstance(a, str):
                raise ActionError(f"BashAction.args must be all str, got {type(a).__name__}")
            if len(a) > MAX_ARG_LEN:
                raise ActionError(f"BashAction.args entry length > {MAX_ARG_LEN}")


@dataclasses.dataclass(frozen=True, slots=True)
class FileWriteAction(BaseAction):
    """OpenHands-shape ``FileWriteAction``: write text content to
    a sandbox-relative path."""

    path: str = ""
    content: str = ""

    KIND: ClassVar[ActionKind] = ActionKind.FILE_WRITE

    def __post_init__(self) -> None:
        BaseAction.__post_init__(self)
        if not isinstance(self.path, str) or not self.path:
            raise ActionError("FileWriteAction.path must be a non-empty str")
        if len(self.path) > MAX_PATH_LEN:
            raise ActionError(f"FileWriteAction.path length > {MAX_PATH_LEN}")
        if not isinstance(self.content, str):
            raise ActionError(
                f"FileWriteAction.content must be str, got {type(self.content).__name__}"
            )
        if len(self.content) > MAX_FILE_CONTENT_LEN:
            raise ActionError(f"FileWriteAction.content length > {MAX_FILE_CONTENT_LEN}")


# ---------------------------------------------------------------------------
# Observation value objects (filled by the live runtime; the
# validator does not produce them)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """Base observation shape.

    Mirrors OpenHands' ``Observation`` base class but is purely a
    data record — no event-loop hooks, no ``cause`` chain.
    """

    action_id: str
    ts_ns: int

    def __post_init__(self) -> None:
        _validate_id("Observation.action_id", self.action_id, MAX_ACTION_ID_LEN)
        _validate_ts_ns(self.ts_ns)


@dataclasses.dataclass(frozen=True, slots=True)
class CommandObservation(Observation):
    """OpenHands-shape command observation: stdout / stderr / exit."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    truncated: bool = False

    def __post_init__(self) -> None:
        Observation.__post_init__(self)
        if not isinstance(self.stdout, str):
            raise ActionError("CommandObservation.stdout must be str")
        if not isinstance(self.stderr, str):
            raise ActionError("CommandObservation.stderr must be str")
        if not isinstance(self.exit_code, int) or isinstance(self.exit_code, bool):
            raise ActionError("CommandObservation.exit_code must be int")
        if not isinstance(self.truncated, bool):
            raise ActionError("CommandObservation.truncated must be bool")


@dataclasses.dataclass(frozen=True, slots=True)
class FileWriteObservation(Observation):
    """OpenHands-shape file-write observation."""

    path: str = ""
    bytes_written: int = 0
    ok: bool = True

    def __post_init__(self) -> None:
        Observation.__post_init__(self)
        if not isinstance(self.path, str) or not self.path:
            raise ActionError("FileWriteObservation.path must be a non-empty str")
        if not isinstance(self.bytes_written, int) or isinstance(self.bytes_written, bool):
            raise ActionError("FileWriteObservation.bytes_written must be int")
        if self.bytes_written < 0:
            raise ActionError("FileWriteObservation.bytes_written must be non-negative")
        if not isinstance(self.ok, bool):
            raise ActionError("FileWriteObservation.ok must be bool")


# ---------------------------------------------------------------------------
# Boundary
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxBoundary:
    """Frozen sandbox policy.

    All defaults are pinned by :data:`DEFAULT_SANDBOX_ROOT` /
    :data:`DEFAULT_FORBIDDEN_COMMANDS` /
    :data:`DEFAULT_FORBIDDEN_MODULES`. Operators can tighten the
    boundary (e.g. shrink the root, expand the forbidden lists) but
    NOT widen it beyond the directive — pinning enforced by tests.
    """

    root: str = DEFAULT_SANDBOX_ROOT
    forbidden_commands: tuple[str, ...] = DEFAULT_FORBIDDEN_COMMANDS
    forbidden_modules: tuple[str, ...] = DEFAULT_FORBIDDEN_MODULES
    max_actions: int = MAX_ACTIONS_PER_PLAN

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root:
            raise BoundaryError("root must be a non-empty str")
        if not self.root.endswith("/"):
            raise BoundaryError(
                f"root must end with '/' to be a directory prefix, got {self.root!r}"
            )
        if ".." in self.root or "~" in self.root:
            raise BoundaryError("root must not contain '..' or '~'")
        # The directive constrains writes to /tmp/dix_sandbox/ —
        # custom roots are allowed for tests but they must still
        # live under /tmp/ to keep the production guarantee.
        if not self.root.startswith("/tmp/"):
            raise BoundaryError("root must live under /tmp/ (directive constraint)")
        if not isinstance(self.forbidden_commands, tuple):
            raise BoundaryError("forbidden_commands must be a tuple")
        for c in self.forbidden_commands:
            if not isinstance(c, str) or not c:
                raise BoundaryError("forbidden_commands entries must be non-empty str")
        if not isinstance(self.forbidden_modules, tuple):
            raise BoundaryError("forbidden_modules must be a tuple")
        for m in self.forbidden_modules:
            if not isinstance(m, str) or not m:
                raise BoundaryError("forbidden_modules entries must be non-empty str")
        if (
            not isinstance(self.max_actions, int)
            or isinstance(self.max_actions, bool)
            or self.max_actions < 1
            or self.max_actions > MAX_ACTIONS_PER_PLAN
        ):
            raise BoundaryError(f"max_actions must be int in [1, {MAX_ACTIONS_PER_PLAN}]")

    def accept_path(self, path: str) -> bool:
        """Return True iff ``path`` is syntactically inside the
        sandbox root.

        The check is intentionally **purely syntactic** — no
        ``os.path.realpath`` call, no filesystem stat — so it is
        INV-15 deterministic and safe to run in the offline
        verifier. Symlink escapes are handled by the live runtime's
        gVisor/Firecracker isolation layer (C-73 / C-74).
        """
        if not isinstance(path, str) or not path:
            return False
        for tok in _PATH_FORBIDDEN_TOKENS:
            if tok in path:
                return False
        return path.startswith(self.root)

    def accept_command_head(self, head: str) -> bool:
        if head in self.forbidden_commands:
            return False
        return head in _ALLOWED_COMMAND_HEADS

    def code_imports_forbidden_module(self, source: str) -> bool:
        """Return True iff a Python source snippet imports any
        module listed in :attr:`forbidden_modules`.

        The detector is intentionally permissive: it matches both
        ``import X`` and ``from X import Y`` at any indentation but
        does NOT recurse into dynamic ``__import__`` calls — those
        are caught upstream by the static-analysis stage
        (:class:`StaticAnalysisStage`).
        """
        if not isinstance(source, str):
            return False
        forbidden = set(self.forbidden_modules)
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for prefix in ("import ", "from "):
                if not stripped.startswith(prefix):
                    continue
                rest = stripped[len(prefix) :].strip()
                head = rest.split()[0] if rest else ""
                head = head.split(".")[0]
                if head in forbidden:
                    return True
        return False


# ---------------------------------------------------------------------------
# Action result + plan
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxActionResult:
    """Per-action validation result."""

    action_id: str
    kind: ActionKind
    verdict: ActionVerdict
    reason: str = ""

    def __post_init__(self) -> None:
        _validate_id(
            "SandboxActionResult.action_id",
            self.action_id,
            MAX_ACTION_ID_LEN,
        )
        if not isinstance(self.kind, ActionKind):
            raise PlanError("SandboxActionResult.kind must be ActionKind")
        if not isinstance(self.verdict, ActionVerdict):
            raise PlanError("SandboxActionResult.verdict must be ActionVerdict")
        if not isinstance(self.reason, str):
            raise PlanError("SandboxActionResult.reason must be str")


@dataclasses.dataclass(frozen=True, slots=True)
class SandboxPlan:
    """Frozen validation report for a typed action sequence.

    * ``actions`` — tuple of validated :class:`BaseAction` rows
      (ordered as the orchestrator submitted them).
    * ``results`` — tuple of :class:`SandboxActionResult`, aligned
      ``results[i].action_id == actions[i].id``.
    * ``digest`` — BLAKE2b-16 over a canonical byte projection of
      ``(actions, results)``. Anchors INV-15 three-run determinism.
    * ``accepted_count`` / ``review_count`` / ``rejected_count`` —
      derived counters surfaced to the orchestrator without
      re-scanning the tuple.
    """

    actions: tuple[BaseAction, ...]
    results: tuple[SandboxActionResult, ...]
    digest: bytes
    accepted_count: int
    review_count: int
    rejected_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.actions, tuple):
            raise PlanError("SandboxPlan.actions must be tuple")
        if not isinstance(self.results, tuple):
            raise PlanError("SandboxPlan.results must be tuple")
        if len(self.actions) != len(self.results):
            raise PlanError(
                "SandboxPlan.actions / results length mismatch: "
                f"{len(self.actions)} vs {len(self.results)}"
            )
        for action, result in zip(self.actions, self.results, strict=True):
            if action.id != result.action_id:
                raise PlanError("SandboxPlan.results[i].action_id must equal actions[i].id")
            if action.kind is not result.kind:
                raise PlanError("SandboxPlan.results[i].kind must equal actions[i].kind")
        if not isinstance(self.digest, bytes) or len(self.digest) != 16:
            raise PlanError("SandboxPlan.digest must be 16 bytes")
        for name in (
            "accepted_count",
            "review_count",
            "rejected_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise PlanError(f"SandboxPlan.{name} must be non-negative int")
        if self.accepted_count + self.review_count + self.rejected_count != len(self.results):
            raise PlanError("SandboxPlan counters must sum to len(results)")

    @property
    def requires_governance_review(self) -> bool:
        return self.review_count > 0

    @property
    def passed(self) -> bool:
        """``True`` iff every action is ``ACCEPTED``."""
        return (
            self.rejected_count == 0
            and self.review_count == 0
            and self.accepted_count == len(self.results)
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SandboxPlanValidator:
    """OpenHands-shape sandbox plan validator.

    Maps a typed action sequence onto a frozen :class:`SandboxPlan`
    with per-action verdicts. No subprocess, no file IO, no clock,
    no PRNG — INV-15 byte-identical across replays.

    The verdict policy:

    * ``REJECTED`` — the boundary is violated unconditionally
      (path escapes the sandbox root, forbidden command, forbidden
      import). The orchestrator transitions the patch to
      :data:`PatchStage.REJECTED`.
    * ``REVIEW_REQUIRED`` — the action is on a sensitive surface
      but inside the boundary (e.g. ``BashAction`` whose head is
      on the allow-list but whose arguments contain a path outside
      the sandbox root). The orchestrator forwards the plan to
      :class:`PatchApprovalBridge` for a typed
      :class:`OperatorConsent` envelope before any live execution.
    * ``ACCEPTED`` — safe to run unattended inside the boundary.
    """

    __slots__ = ("_boundary",)

    def __init__(self, *, boundary: SandboxBoundary) -> None:
        if not isinstance(boundary, SandboxBoundary):
            raise BoundaryError("SandboxPlanValidator: boundary must be a SandboxBoundary")
        self._boundary = boundary

    @property
    def boundary(self) -> SandboxBoundary:
        return self._boundary

    def validate(
        self,
        actions: Iterable[BaseAction],
    ) -> SandboxPlan:
        actions_tuple = tuple(actions)
        if len(actions_tuple) > self._boundary.max_actions:
            raise PlanError(
                "actions length > boundary.max_actions: "
                f"{len(actions_tuple)} vs "
                f"{self._boundary.max_actions}"
            )
        seen_ids: set[str] = set()
        results: list[SandboxActionResult] = []
        accepted = 0
        review = 0
        rejected = 0
        for action in actions_tuple:
            if not isinstance(action, BaseAction):
                raise ActionError(
                    "SandboxPlanValidator.validate: every entry must be a BaseAction subclass"
                )
            if action.id in seen_ids:
                raise PlanError(f"duplicate action id: {action.id!r}")
            seen_ids.add(action.id)
            verdict, reason = self._classify(action)
            if verdict is ActionVerdict.ACCEPTED:
                accepted += 1
            elif verdict is ActionVerdict.REVIEW_REQUIRED:
                review += 1
            else:
                rejected += 1
            results.append(
                SandboxActionResult(
                    action_id=action.id,
                    kind=action.kind,
                    verdict=verdict,
                    reason=reason,
                )
            )
        results_tuple = tuple(results)
        digest = _digest_plan(actions_tuple, results_tuple)
        return SandboxPlan(
            actions=actions_tuple,
            results=results_tuple,
            digest=digest,
            accepted_count=accepted,
            review_count=review,
            rejected_count=rejected,
        )

    def _classify(
        self,
        action: BaseAction,
    ) -> tuple[ActionVerdict, str]:
        if isinstance(action, CodeAction):
            if self._boundary.code_imports_forbidden_module(action.source):
                return (
                    ActionVerdict.REJECTED,
                    "code imports forbidden module",
                )
            return (ActionVerdict.ACCEPTED, "code clean")
        if isinstance(action, BashAction):
            head = action.command
            if not self._boundary.accept_command_head(head):
                return (
                    ActionVerdict.REJECTED,
                    f"command head not allowed: {head!r}",
                )
            # Argument scan: any arg that looks like a path is
            # checked against the sandbox boundary. Paths inside
            # the root are accepted; paths outside trigger a
            # REVIEW_REQUIRED so the orchestrator can route the
            # plan through PatchApprovalBridge.
            for arg in action.args:
                if _BASH_PATHISH_ARG_RE.search(arg):
                    if not self._boundary.accept_path(arg):
                        return (
                            ActionVerdict.REVIEW_REQUIRED,
                            f"argument touches path outside sandbox: {arg!r}",
                        )
            return (ActionVerdict.ACCEPTED, "bash clean")
        if isinstance(action, FileWriteAction):
            if not self._boundary.accept_path(action.path):
                return (
                    ActionVerdict.REJECTED,
                    f"path outside sandbox: {action.path!r}",
                )
            return (ActionVerdict.ACCEPTED, "file write clean")
        return (
            ActionVerdict.REJECTED,
            f"unknown action kind: {type(action).__name__}",
        )


def _digest_plan(
    actions: tuple[BaseAction, ...],
    results: tuple[SandboxActionResult, ...],
) -> bytes:
    """BLAKE2b-16 over a canonical byte projection of the
    ``(actions, results)`` tuple.

    The projection is hand-rolled to avoid Python ``repr`` drift
    and to keep INV-15 byte-identical across CPython versions.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"v=")
    h.update(SANDBOX_VERSION.encode("utf-8"))
    h.update(b"|a=")
    for a in actions:
        h.update(a.id.encode("utf-8"))
        h.update(b"#k=")
        h.update(a.kind.value.encode("ascii"))
        h.update(b"#ts=")
        h.update(str(a.ts_ns).encode("ascii"))
        if isinstance(a, CodeAction):
            h.update(b"#lang=")
            h.update(a.language.encode("utf-8"))
            h.update(b"#src=")
            h.update(a.source.encode("utf-8"))
        elif isinstance(a, BashAction):
            h.update(b"#cmd=")
            h.update(a.command.encode("utf-8"))
            h.update(b"#args=")
            for arg in a.args:
                h.update(arg.encode("utf-8"))
                h.update(b",")
        elif isinstance(a, FileWriteAction):
            h.update(b"#path=")
            h.update(a.path.encode("utf-8"))
            h.update(b"#content=")
            h.update(a.content.encode("utf-8"))
        h.update(b"|")
    h.update(b"|r=")
    for r in results:
        h.update(r.action_id.encode("utf-8"))
        h.update(b"#k=")
        h.update(r.kind.value.encode("ascii"))
        h.update(b"#v=")
        h.update(r.verdict.value.encode("ascii"))
        h.update(b"#why=")
        h.update(r.reason.encode("utf-8"))
        h.update(b"|")
    return h.digest()


# ---------------------------------------------------------------------------
# Pipeline-stage adapter
# ---------------------------------------------------------------------------


@runtime_checkable
class _StageEvaluator(Protocol):
    """Local Protocol mirroring the patch-pipeline stage shape so
    we do not need to import the concrete :class:`SandboxStage`.

    Implementing this Protocol keeps :class:`OpenHandsSandboxStage`
    a drop-in alternative to the existing
    :class:`evolution_engine.patch_pipeline.sandbox.SandboxStage`
    for orchestrator wiring purposes (the orchestrator only depends
    on the protocol shape).
    """

    name: str
    spec_id: str

    def evaluate(
        self,
        *,
        ts_ns: int,
        plan: SandboxPlan,
    ) -> tuple[SandboxPlan, StageVerdict]: ...


class OpenHandsSandboxStage:
    """Patch-pipeline stage wrapper around
    :class:`SandboxPlanValidator`.

    Designed as a structural sibling of
    :class:`evolution_engine.patch_pipeline.sandbox.SandboxStage` but
    consuming the OpenHands action surface instead of a touchpoint
    string list. Returns the same
    :class:`core.contracts.patch.StageVerdict` so the orchestrator
    can mix the two stages on a single patch record.
    """

    name: str = "sandbox_openhands"
    spec_id: str = "C-17"

    __slots__ = ("_validator",)

    def __init__(self, *, validator: SandboxPlanValidator) -> None:
        if not isinstance(validator, SandboxPlanValidator):
            raise BoundaryError("OpenHandsSandboxStage: validator must be a SandboxPlanValidator")
        self._validator = validator

    @property
    def validator(self) -> SandboxPlanValidator:
        return self._validator

    def evaluate(
        self,
        *,
        ts_ns: int,
        actions: Iterable[BaseAction],
    ) -> tuple[SandboxPlan, StageVerdict]:
        _validate_ts_ns(ts_ns)
        plan = self._validator.validate(actions)
        passed = plan.passed
        if plan.rejected_count > 0:
            detail = f"openhands sandbox: {plan.rejected_count} actions rejected"
        elif plan.review_count > 0:
            detail = f"openhands sandbox: {plan.review_count} actions require governance review"
        else:
            detail = "openhands sandbox clean"
        verdict = StageVerdict(
            ts_ns=ts_ns,
            stage=PatchStage.SANDBOX,
            passed=passed,
            detail=detail,
            meta=_freeze_meta(
                {
                    "accepted": str(plan.accepted_count),
                    "review": str(plan.review_count),
                    "rejected": str(plan.rejected_count),
                    "digest": plan.digest.hex(),
                }
            ),
        )
        return plan, verdict


# ---------------------------------------------------------------------------
# Lazy seam — live OpenHands runtime
# ---------------------------------------------------------------------------


def enable_openhands_factory() -> None:
    """Opt in to the live OpenHands sandbox backend.

    Until activated, :class:`SandboxPlanValidator` is the only
    supported transport — this keeps the production path B1-clean
    and INV-15 reproducible without the vendor dependency.

    The live backend will route every ``REVIEW_REQUIRED`` /
    ``ACCEPTED`` plan into a Docker-isolated OpenHands runtime
    wrapped by gVisor/Firecracker (C-73 / C-74). OpenHands cloud
    deployment is forbidden by DIX policy — the live backend MUST
    run against a locally-hosted OpenHands container only.
    """
    raise NotImplementedError(
        "enable_openhands_factory: live OpenHands backend not yet "
        "activated — use SandboxPlanValidator for the deterministic "
        "in-memory plan validation"
    )
