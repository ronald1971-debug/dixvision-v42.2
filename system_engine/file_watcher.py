# ADAPTED FROM: https://github.com/gorakhargosh/watchdog  (Apache-2.0)
#
# Canonical hot-reload file watcher — OFFLINE_ONLY (``system_engine`` tier,
# background-thread mode; never on the RUNTIME hot path).
#
# NEW_PIP_DEPENDENCIES = ("watchdog",)
#
# Authority constraints (pinned by ``tests/test_file_watcher.py``):
#
#   * B1   — never imports from any runtime engine tier (no execution_engine /
#     intelligence_engine / governance_engine imports).
#   * INV-15 — :func:`scan_directory` and :func:`diff_snapshots` are pure
#     functions of their arguments; three independent calls produce
#     byte-identical :class:`DirectorySnapshot` / tuple of
#     :class:`FileChangeEvent` for the same inputs.
#   * B27 / B28 / INV-71 — no typed-event constructors here.
#   * GOVERNANCE_LOCK — the watcher refuses to register any path that
#     resolves under ``governance_engine/`` (or any tier listed in
#     :data:`FORBIDDEN_TIERS`). Governance hot-reload is forbidden by
#     master-doc rule; full restart with governance approval is required.
#   * No top-level imports of :mod:`watchdog`, :mod:`time`, :mod:`datetime`,
#     :mod:`random`, :mod:`asyncio`, :mod:`numpy`, :mod:`torch`,
#     :mod:`polars`, :mod:`requests`.
#
# Backend strategy mirrors I-02..I-09: stdlib (``hashlib`` + ``os.scandir``)
# polling watcher is the production default; ``watchdog.Observer`` is the
# lazy seam exposed via :func:`enable_watchdog_observer_factory` so callers
# can promote to native FS events when watchdog is installed. Both backends
# produce byte-identical :class:`FileChangeEvent` tuples for the same
# directory state.

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

__all__ = [
    "DEFAULT_PATTERNS",
    "FORBIDDEN_TIERS",
    "WATCHER_VERSION",
    "NEW_PIP_DEPENDENCIES",
    "ChangeKind",
    "DirectorySnapshot",
    "FileChangeEvent",
    "FileWatcherError",
    "RegistryWatchSpec",
    "diff_snapshots",
    "enable_watchdog_observer_factory",
    "match_patterns",
    "scan_directory",
]


WATCHER_VERSION: Final[str] = "v1.0-I10"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("watchdog",)

#: File globs the canonical registry hot-reloader cares about. The
#: watcher refuses to register a :class:`RegistryWatchSpec` with an
#: empty pattern tuple — the hard rule is one explicit allow-list per
#: spec, never a "watch everything" wildcard.
DEFAULT_PATTERNS: Final[tuple[str, ...]] = ("*.yaml", "*.yml")

#: Tier path-prefixes that may never be hot-reloaded. Mutating these
#: requires a full restart with governance approval per master-doc rule.
FORBIDDEN_TIERS: Final[tuple[str, ...]] = (
    "governance_engine",
    "execution_engine",
    "intelligence_engine",
)


class FileWatcherError(ValueError):
    """Configuration / state error for the file-watcher surface."""


class ChangeKind(StrEnum):
    """Sub-type discriminator for :class:`FileChangeEvent`."""

    CREATED = "CREATED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


@dataclass(frozen=True, slots=True)
class FileChangeEvent:
    """A single observed change in a watched directory.

    Pure value object — no clocks, no I/O. Constructed by
    :func:`diff_snapshots` from two :class:`DirectorySnapshot` instances.
    """

    kind: ChangeKind
    path: str
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ChangeKind):
            raise FileWatcherError("FileChangeEvent.kind must be ChangeKind")
        if not isinstance(self.path, str) or not self.path:
            raise FileWatcherError("FileChangeEvent.path must be non-empty")
        if not isinstance(self.digest, str):
            raise FileWatcherError("FileChangeEvent.digest must be a string")
        if self.kind is ChangeKind.DELETED and self.digest != "":
            raise FileWatcherError("DELETED events must carry an empty digest")
        if self.kind is not ChangeKind.DELETED and not self.digest:
            raise FileWatcherError(f"{self.kind.value} events require a non-empty digest")


@dataclass(frozen=True, slots=True)
class DirectorySnapshot:
    """Immutable mapping ``path → blake2b-16 digest`` over a directory."""

    root: str
    entries: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root:
            raise FileWatcherError("DirectorySnapshot.root must be non-empty")
        if not isinstance(self.entries, Mapping):
            raise FileWatcherError("DirectorySnapshot.entries must be a mapping")
        for key, val in self.entries.items():
            if not isinstance(key, str) or not key:
                raise FileWatcherError("DirectorySnapshot keys must be non-empty strings")
            if not isinstance(val, str) or not val:
                raise FileWatcherError("DirectorySnapshot values must be non-empty strings")

    def canonical_items(self) -> tuple[tuple[str, str], ...]:
        """Return entries sorted lexicographically for byte-stable replay."""

        return tuple(sorted(self.entries.items()))


@dataclass(frozen=True, slots=True)
class RegistryWatchSpec:
    """Declarative spec for a directory the canonical registry watches.

    The watcher never accepts a spec rooted under :data:`FORBIDDEN_TIERS`.
    """

    root: str
    patterns: tuple[str, ...] = DEFAULT_PATTERNS

    def __post_init__(self) -> None:
        if not isinstance(self.root, str) or not self.root:
            raise FileWatcherError("RegistryWatchSpec.root must be non-empty")
        if not isinstance(self.patterns, tuple) or not self.patterns:
            raise FileWatcherError("RegistryWatchSpec.patterns must be a non-empty tuple")
        for pat in self.patterns:
            if not isinstance(pat, str) or not pat:
                raise FileWatcherError("RegistryWatchSpec.patterns entries must be strings")
        _verify_governance_lock(self.root)


# ---------------------------------------------------------------------------
# Governance lock — refuse to watch paths under forbidden tiers.
# ---------------------------------------------------------------------------


def _verify_governance_lock(root: str) -> None:
    """Raise :class:`FileWatcherError` if ``root`` is under a forbidden tier.

    Pure string check against :data:`FORBIDDEN_TIERS`. The watcher
    refuses to register ANY path that contains a forbidden tier
    segment, regardless of whether the path exists on disk.
    """

    p = Path(root).as_posix()
    parts = [seg for seg in p.split("/") if seg]
    for forbidden in FORBIDDEN_TIERS:
        if forbidden in parts:
            raise FileWatcherError(
                f"governance-lock: refusing to watch {root!r}: "
                f"path resolves under forbidden tier {forbidden!r}"
            )


# ---------------------------------------------------------------------------
# Pattern matching (fnmatch-shape, stdlib-only).
# ---------------------------------------------------------------------------


def match_patterns(name: str, patterns: Sequence[str]) -> bool:
    """Return True iff ``name`` matches any of ``patterns`` (fnmatch shape)."""

    import fnmatch

    if not isinstance(name, str):
        raise FileWatcherError("match_patterns: name must be a string")
    for pat in patterns:
        if not isinstance(pat, str) or not pat:
            raise FileWatcherError("match_patterns: patterns must be non-empty strings")
        if fnmatch.fnmatchcase(name, pat):
            return True
    return False


# ---------------------------------------------------------------------------
# Directory snapshot — pure function of (root, file_loader, patterns).
# ---------------------------------------------------------------------------


def _blake2b16(payload: bytes) -> str:
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def scan_directory(
    root: str,
    *,
    patterns: Sequence[str] = DEFAULT_PATTERNS,
    file_loader: Callable[[str], bytes] | None = None,
    file_lister: Callable[[str], Iterable[str]] | None = None,
) -> DirectorySnapshot:
    """Build a :class:`DirectorySnapshot` of ``root``.

    Pure function of ``(root, patterns, file_loader, file_lister)``.
    Callers wishing to avoid stdlib I/O at the call-site (e.g. tests
    that synthesise file payloads in memory) pass the loaders
    explicitly. When both loaders are ``None`` the default scanner
    walks the real filesystem under ``root`` using stdlib only.
    """

    _verify_governance_lock(root)
    if file_loader is None and file_lister is None:
        return _default_scan(root, patterns)
    loader = file_loader if file_loader is not None else _default_file_loader
    lister = file_lister if file_lister is not None else _default_file_lister

    entries: dict[str, str] = {}
    for name in sorted(lister(root)):
        if not match_patterns(name, patterns):
            continue
        payload = loader(name)
        if not isinstance(payload, (bytes, bytearray)):
            raise FileWatcherError(f"file_loader must return bytes; got {type(payload).__name__}")
        entries[name] = _blake2b16(bytes(payload))
    return DirectorySnapshot(root=root, entries=entries)


def _default_scan(root: str, patterns: Sequence[str]) -> DirectorySnapshot:
    """Real-filesystem scanner used when no loader/lister is injected."""

    entries: dict[str, str] = {}
    if not os.path.isdir(root):
        return DirectorySnapshot(root=root, entries=entries)
    rels: list[tuple[str, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            abs_path = os.path.join(dirpath, fname)
            rel = os.path.relpath(abs_path, root).replace(os.sep, "/")
            rels.append((rel, abs_path))
    for rel, abs_path in sorted(rels):
        if not match_patterns(rel, patterns):
            continue
        try:
            with open(abs_path, "rb") as fh:
                payload = fh.read()
        except OSError:
            continue
        entries[rel] = _blake2b16(payload)
    return DirectorySnapshot(root=root, entries=entries)


def _default_file_lister(root: str) -> Iterable[str]:
    """Default lister — recursively yields file basenames under ``root``."""

    if not os.path.isdir(root):
        return ()
    out: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fname in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fname), root)
            out.append(rel.replace(os.sep, "/"))
    return out


def _default_file_loader(rel_path: str) -> bytes:
    """Default loader — reads ``rel_path`` from disk under cwd."""

    try:
        with open(rel_path, "rb") as fh:
            return fh.read()
    except OSError:
        return b""


# ---------------------------------------------------------------------------
# Snapshot diff — pure function.
# ---------------------------------------------------------------------------


def diff_snapshots(
    previous: DirectorySnapshot,
    current: DirectorySnapshot,
) -> tuple[FileChangeEvent, ...]:
    """Return the canonical event list mapping ``previous → current``.

    Pure function of two snapshots. Events are emitted in sorted-path
    order; three independent calls produce byte-identical output
    (INV-15).
    """

    if previous.root != current.root:
        raise FileWatcherError(
            "diff_snapshots: snapshots cover different roots: "
            f"{previous.root!r} vs {current.root!r}"
        )
    events: list[FileChangeEvent] = []
    prev = previous.entries
    curr = current.entries
    all_paths = sorted(set(prev) | set(curr))
    for path in all_paths:
        if path in curr and path not in prev:
            events.append(FileChangeEvent(kind=ChangeKind.CREATED, path=path, digest=curr[path]))
        elif path in prev and path not in curr:
            events.append(FileChangeEvent(kind=ChangeKind.DELETED, path=path, digest=""))
        elif prev[path] != curr[path]:
            events.append(FileChangeEvent(kind=ChangeKind.MODIFIED, path=path, digest=curr[path]))
    return tuple(events)


# ---------------------------------------------------------------------------
# Lazy seam — watchdog Observer factory.
# ---------------------------------------------------------------------------


def enable_watchdog_observer_factory() -> Callable[..., Any]:
    """Lazy seam — returns a callable that constructs a watchdog Observer.

    Importing :mod:`watchdog` is deferred until this factory is called,
    so production deployments without watchdog installed boot cleanly
    via the polling backend (:func:`scan_directory` + :func:`diff_snapshots`).
    """

    import watchdog.observers  # noqa: F401 - imported here on purpose (lazy seam)

    def _build_observer(*_: Any, **__: Any) -> Any:
        from watchdog.observers import Observer

        return Observer()

    return _build_observer
