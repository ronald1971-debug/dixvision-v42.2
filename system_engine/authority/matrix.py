"""Strict loader + schema for ``registry/authority_matrix.yaml``.

The matrix is the single conflict-resolution table for the v42.2
control plane. This module loads it into immutable dataclasses,
validates internal consistency (every reference resolves; precedence
covers every actor; conflicts and overrides reference declared
actors), and exposes a few small read-only helpers for tests + the CI
lint.

Pure / deterministic. INV-15 — no clock, no PRNG. The loader raises
:class:`ValueError` on the first inconsistency so a malformed matrix
fails the build immediately.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthorityActor:
    """One authority on the control plane (governance / intelligence / …)."""

    id: str
    role: str
    module: str
    invariants: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True, slots=True)
class ConflictRow:
    """One documented decision point where two or more actors collide."""

    id: str
    domain: str
    description: str
    winner: str  # actor id, or the literal string "deferred"
    invariants: tuple[str, ...] = ()
    safety: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class AuthorityOverride:
    """One legal exceptional edge — always routed through Governance."""

    id: str
    name: str
    grants: str  # actor id receiving the override
    overrides: tuple[str, ...]  # actor ids being overridden
    via: str  # always 'governance' but kept explicit
    invariants: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class AuthorityMatrix:
    """The whole matrix as one immutable value."""

    version: str
    actors: tuple[AuthorityActor, ...]
    precedence: tuple[str, ...]
    conflicts: tuple[ConflictRow, ...]
    overrides: tuple[AuthorityOverride, ...]

    @property
    def actor_ids(self) -> frozenset[str]:
        return frozenset(a.id for a in self.actors)

    def actor(self, actor_id: str) -> AuthorityActor:
        for a in self.actors:
            if a.id == actor_id:
                return a
        raise KeyError(f"unknown actor_id: {actor_id!r}")

    def precedence_index(self, actor_id: str) -> int:
        try:
            return self.precedence.index(actor_id)
        except ValueError as exc:  # pragma: no cover — load_* validates
            raise KeyError(f"actor {actor_id!r} not in precedence") from exc

    def resolve(self, a: str, b: str) -> str:
        """Return the higher-precedence actor between two ids."""

        ia = self.precedence_index(a)
        ib = self.precedence_index(b)
        return a if ia <= ib else b


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_REQUIRED_TOP = {"version", "actors", "precedence", "conflicts", "overrides"}


def load_authority_matrix(path: str | Path) -> AuthorityMatrix:
    """Load and validate the authority matrix YAML at ``path``."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"authority matrix not found: {p}")

    with p.open() as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, Mapping):
        raise ValueError(f"{p}: top-level YAML must be a mapping, got {type(raw).__name__}")

    missing = sorted(_REQUIRED_TOP - set(raw))
    if missing:
        raise ValueError(f"{p}: missing required top-level keys: {missing}")

    version = str(raw["version"])
    actors = tuple(_parse_actor(item, idx) for idx, item in enumerate(raw["actors"] or []))
    if not actors:
        raise ValueError(f"{p}: actors[] must be non-empty")

    actor_ids = {a.id for a in actors}
    if len(actor_ids) != len(actors):
        raise ValueError(f"{p}: duplicate actor ids in actors[]")

    precedence = tuple(str(x) for x in (raw["precedence"] or []))
    _validate_precedence(precedence, actor_ids, ctx=str(p))

    conflicts = tuple(
        _parse_conflict(item, idx, actor_ids, ctx=str(p))
        for idx, item in enumerate(raw["conflicts"] or [])
    )
    if {c.id for c in conflicts} != {c.id for c in conflicts}:  # pragma: no cover
        pass
    if len({c.id for c in conflicts}) != len(conflicts):
        raise ValueError(f"{p}: duplicate conflict ids")

    overrides = tuple(
        _parse_override(item, idx, actor_ids, ctx=str(p))
        for idx, item in enumerate(raw["overrides"] or [])
    )
    if len({o.id for o in overrides}) != len(overrides):
        raise ValueError(f"{p}: duplicate override ids")

    return AuthorityMatrix(
        version=version,
        actors=actors,
        precedence=precedence,
        conflicts=conflicts,
        overrides=overrides,
    )


# ---------------------------------------------------------------------------
# Internal parsers
# ---------------------------------------------------------------------------


def _parse_actor(raw: Any, idx: int) -> AuthorityActor:
    ctx = f"actors[{idx}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{ctx}: must be a mapping, got {type(raw).__name__}")
    for k in ("id", "role", "module"):
        if k not in raw:
            raise ValueError(f"{ctx}: missing required field {k!r}")
    return AuthorityActor(
        id=str(raw["id"]),
        role=str(raw["role"]),
        module=str(raw["module"]),
        invariants=tuple(str(x) for x in (raw.get("invariants") or ())),
        notes=str(raw.get("notes", "")).strip(),
    )


def _validate_precedence(
    precedence: tuple[str, ...], actor_ids: set[str], *, ctx: str
) -> None:
    if not precedence:
        raise ValueError(f"{ctx}: precedence[] must be non-empty")
    if len(set(precedence)) != len(precedence):
        raise ValueError(f"{ctx}: duplicate entries in precedence[]")
    unknown = sorted(set(precedence) - actor_ids)
    if unknown:
        raise ValueError(f"{ctx}: precedence references unknown actors: {unknown}")
    missing = sorted(actor_ids - set(precedence))
    if missing:
        raise ValueError(f"{ctx}: actors not covered by precedence: {missing}")


def _parse_conflict(
    raw: Any, idx: int, actor_ids: set[str], *, ctx: str
) -> ConflictRow:
    where = f"{ctx}: conflicts[{idx}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: must be a mapping, got {type(raw).__name__}")
    for k in ("id", "domain", "description", "winner"):
        if k not in raw:
            raise ValueError(f"{where}: missing required field {k!r}")
    winner = str(raw["winner"])
    if winner != "deferred" and winner not in actor_ids:
        raise ValueError(f"{where}: winner {winner!r} is not a declared actor")
    return ConflictRow(
        id=str(raw["id"]),
        domain=str(raw["domain"]),
        description=str(raw["description"]).strip(),
        winner=winner,
        invariants=tuple(str(x) for x in (raw.get("invariants") or ())),
        safety=tuple(str(x) for x in (raw.get("safety") or ())),
        rules=tuple(str(x) for x in (raw.get("rules") or ())),
        notes=str(raw.get("notes", "")).strip(),
    )


def _parse_override(
    raw: Any, idx: int, actor_ids: set[str], *, ctx: str
) -> AuthorityOverride:
    where = f"{ctx}: overrides[{idx}]"
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: must be a mapping, got {type(raw).__name__}")
    for k in ("id", "name", "grants", "overrides", "via"):
        if k not in raw:
            raise ValueError(f"{where}: missing required field {k!r}")

    grants = str(raw["grants"])
    if grants not in actor_ids:
        raise ValueError(f"{where}: grants {grants!r} is not a declared actor")
    overrides = tuple(str(x) for x in (raw["overrides"] or ()))
    if not overrides:
        raise ValueError(f"{where}: overrides[] must be non-empty")
    unknown = sorted(set(overrides) - actor_ids)
    if unknown:
        raise ValueError(f"{where}: overrides references unknown actors: {unknown}")
    via = str(raw["via"])
    if via != "governance":
        raise ValueError(f"{where}: via must be 'governance' (is {via!r})")
    if grants in overrides:
        raise ValueError(f"{where}: actor cannot override itself ({grants!r})")
    return AuthorityOverride(
        id=str(raw["id"]),
        name=str(raw["name"]),
        grants=grants,
        overrides=overrides,
        via=via,
        invariants=tuple(str(x) for x in (raw.get("invariants") or ())),
        notes=str(raw.get("notes", "")).strip(),
    )


__all__ = [
    "AuthorityActor",
    "AuthorityMatrix",
    "AuthorityOverride",
    "ConflictRow",
    "load_authority_matrix",
]
