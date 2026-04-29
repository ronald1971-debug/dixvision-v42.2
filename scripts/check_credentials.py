#!/usr/bin/env python3
"""Operator-facing credential audit (Dashboard-2026 wave-01.5).

Run::

    python -m scripts.check_credentials               # full report
    python -m scripts.check_credentials --missing-only

The script prints a one-line-per-row summary of every
``auth: required`` source in
``registry/data_source_registry.yaml`` together with the env-var
name(s) the system reads at boot and whether each is currently set
in the live process environment. Exit code is non-zero when any
required env var is missing — wire it into the launcher to gate boot
on a clean credential set if you want a strict mode (default boot
remains permissive: missing keys keep the matching registry rows
unusable but never block the harness from starting).

This is a *thin CLI projection* of the same module that powers
``GET /api/credentials/status``; it intentionally has no extra
features so the two surfaces never drift.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Make ``python scripts/check_credentials.py`` work the same as
# ``python -m scripts.check_credentials`` by adding the repo root to
# sys.path before importing the engine packages.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from system_engine.credentials import (  # noqa: E402  (path bootstrap above)
    PresenceState,
    presence_status,
    requirements_for_registry,
)
from system_engine.scvs.source_registry import (  # noqa: E402
    load_source_registry,
)

SOURCE_REGISTRY_PATH = REPO_ROOT / "registry" / "data_source_registry.yaml"


_STATE_GLYPH = {
    PresenceState.PRESENT: "[OK]   ",
    PresenceState.PARTIAL: "[PART] ",
    PresenceState.MISSING: "[MISS] ",
}


def _format_row(status, *, missing_only: bool) -> str | None:
    if missing_only and status.state is PresenceState.PRESENT:
        return None
    req = status.requirement
    env_pairs = []
    for name, present in zip(
        req.env_vars, status.env_vars_present, strict=True
    ):
        env_pairs.append(f"{name}={'set' if present else 'unset'}")
    line = (
        f"{_STATE_GLYPH[status.state]}"
        f"{req.source_id:<28} "
        f"{req.category:<11} "
        f"{req.provider:<12} "
        f"{', '.join(env_pairs)}"
    )
    if status.state is PresenceState.MISSING and req.signup_url:
        line += f"\n           sign up: {req.signup_url}"
    return line


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_credentials",
        description=(
            "Audit the registry's credential requirements against"
            " the live environment."
        ),
    )
    parser.add_argument(
        "--missing-only",
        action="store_true",
        help="Print only rows that are missing or partial.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with code 2 if any required env var is missing"
            " (default behaviour returns 0 even with missing keys)."
        ),
    )
    args = parser.parse_args(argv)

    registry = load_source_registry(SOURCE_REGISTRY_PATH)
    requirements = requirements_for_registry(registry)
    statuses = presence_status(requirements, os.environ)

    counts = {state: 0 for state in PresenceState}
    for st in statuses:
        counts[st.state] += 1

    print(
        f"Credential audit — {len(statuses)} required source(s):"
        f" {counts[PresenceState.PRESENT]} present,"
        f" {counts[PresenceState.PARTIAL]} partial,"
        f" {counts[PresenceState.MISSING]} missing."
    )
    print()
    for st in statuses:
        line = _format_row(st, missing_only=args.missing_only)
        if line is not None:
            print(line)

    if args.strict and counts[PresenceState.MISSING] + counts[
        PresenceState.PARTIAL
    ] > 0:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
