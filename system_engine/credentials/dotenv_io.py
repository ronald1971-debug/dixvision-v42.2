"""Minimal ``.env`` parser and atomic writer (stdlib only).

We deliberately do **not** depend on ``python-dotenv``. The format
we support is the lowest-common-denominator subset every dotenv
implementation agrees on:

    KEY=value           # simple
    KEY="quoted value"  # double-quoted (escapes are NOT processed)
    KEY='quoted value'  # single-quoted
    # comment lines      (full-line comments only)

Inline ``# comment`` after a value is **not** stripped — every
character after ``=`` belongs to the value (after surrounding
quotes are removed). This matches the behaviour of bash's
``set -a; source .env; set +a`` which is the most common operator
flow on the Linux/Mac launcher.

The writer is intentionally line-rewriting rather than full-file
re-templating: when an operator updates a single key, we preserve
every other line (comments, blank lines, ordering) verbatim and
only replace the line(s) matching that key.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

# Match ``KEY=value`` lines. Allows leading whitespace and an
# optional ``export`` prefix (which we silently strip — the value
# is the same either way).
_LINE_RE = re.compile(
    r"""
    ^\s*
    (?:export\s+)?
    (?P<key>[A-Za-z_][A-Za-z0-9_]*)
    \s*=\s*
    (?P<value>.*?)
    \s*$
    """,
    re.VERBOSE,
)

# Valid env-var name (POSIX-ish; same as the regex above for
# the key group).
_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_valid_env_var_name(name: str) -> bool:
    """Return ``True`` iff ``name`` is a syntactically legal env-var name.

    The dotenv format itself does not enforce this, but we do —
    if the operator hand-edits a key into something silly, every
    downstream consumer (FastAPI, the trading engine, the
    verifier) will reject it anyway.
    """

    return bool(_ENV_VAR_NAME_RE.match(name))


def _unquote(raw: str) -> str:
    """Strip a single matching pair of surrounding quotes."""

    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    return raw


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse the contents of a ``.env`` file into a dict.

    Later occurrences of the same key win, matching the way
    ``source .env`` would behave in a shell. Lines that don't
    match the simple ``KEY=value`` shape are silently skipped —
    ``parse_dotenv`` is **lenient on read** because we want to
    cope with whatever the operator typed; ``write_dotenv`` is
    strict on write.
    """

    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        key = m.group("key")
        value = _unquote(m.group("value"))
        out[key] = value
    return out


def load_dotenv_file(path: Path) -> dict[str, str]:
    """Read ``path`` and parse it; missing file → empty dict."""

    if not path.exists():
        return {}
    return parse_dotenv(path.read_text(encoding="utf-8"))


def _format_value(value: str) -> str:
    """Encode a value so it round-trips through bash's ``source``.

    Single-quote and the value carries no special chars → bare.
    Anything else → double-quoted with backslash-escapes for ``\\``,
    ``"`` and ``$``.
    """

    if value == "":
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_\-./:]+", value):
        return value
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
    )
    return f'"{escaped}"'


def update_dotenv_file(
    path: Path, updates: Mapping[str, str]
) -> dict[str, str]:
    """Atomically update ``path`` with ``updates`` and return the
    final ``key → value`` map.

    Behaviour:
    * Lines whose key is in ``updates`` are rewritten in place.
    * Lines whose key is not in ``updates`` are preserved verbatim.
    * Keys in ``updates`` that don't appear in the file are
      appended at the bottom in deterministic (sorted) order.
    * The write is atomic: tempfile in the same dir → fsync →
      ``os.replace``. A crash mid-write never leaves a half-file.

    All keys in ``updates`` must satisfy
    :func:`is_valid_env_var_name`; values must be ``str``.
    Multi-line values are not supported.
    """

    for key, value in updates.items():
        if not is_valid_env_var_name(key):
            raise ValueError(f"invalid env var name: {key!r}")
        if not isinstance(value, str):
            raise TypeError(
                f"value for {key!r} must be str, got {type(value).__name__}"
            )
        if "\n" in value or "\r" in value:
            raise ValueError(
                f"value for {key!r} contains a newline; not supported"
            )

    if path.exists():
        original_lines = path.read_text(encoding="utf-8").splitlines(
            keepends=False
        )
        had_trailing_newline = path.read_text(encoding="utf-8").endswith(
            "\n"
        )
    else:
        original_lines = []
        had_trailing_newline = True

    seen: set[str] = set()
    new_lines: list[str] = []
    for line in original_lines:
        m = _LINE_RE.match(line)
        if m and m.group("key") in updates:
            key = m.group("key")
            if key in seen:
                # Drop duplicate later occurrences entirely; the
                # first occurrence has already been rewritten.
                continue
            new_lines.append(f"{key}={_format_value(updates[key])}")
            seen.add(key)
        else:
            new_lines.append(line)

    for key in sorted(updates.keys() - seen):
        new_lines.append(f"{key}={_format_value(updates[key])}")

    body = "\n".join(new_lines)
    if had_trailing_newline or new_lines:
        body += "\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=".env.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; don't mask the original exception.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    return parse_dotenv(body)
