"""Read shim + write gate for credential storage.

The same dashboard code runs in three contexts:

* **Local launcher** — Linux/Mac/Windows operator box. Persistent
  state lives in a gitignored ``.env`` at the repo root.
* **Devin session** — secrets are injected into ``os.environ`` by
  the platform; we must **not** write to ``.env`` because that
  file is owned by the operator's local checkout, not the Devin
  sandbox.
* **CI** — ``os.environ`` only; no ``.env``, no Devin secrets.

The shim resolves a value with this precedence:

    os.environ  >  .env file  >  None

``os.environ`` wins because (a) Devin secrets are already there
and (b) operators routinely override a key for one shell session
without rewriting ``.env``.

Writes are gated by :func:`is_devin_session`. Inside a Devin
session ``write_credential`` raises :class:`StorageNotWritable`
— the dashboard will translate that into a 409 with an
operator-readable message pointing at the Devin secrets tool.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from system_engine.credentials.dotenv_io import (
    is_valid_env_var_name,
    load_dotenv_file,
    update_dotenv_file,
)


class StorageNotWritable(RuntimeError):
    """Raised when an environment forbids ``.env`` writes."""


def repo_root() -> Path:
    """Best-effort repo root (parent of ``system_engine``)."""

    return Path(__file__).resolve().parents[2]


def default_dotenv_path() -> Path:
    """Path of the ``.env`` file the dashboard writes to."""

    return repo_root() / ".env"


def _has_devin_install_dir() -> bool:
    """Filesystem signal: ``/opt/.devin`` exists.

    Pulled into its own function so tests can monkeypatch it
    without touching ``pathlib.Path.is_dir`` globally (which
    breaks ``Path.mkdir(exist_ok=True)``).
    """

    return Path("/opt/.devin").is_dir()


def is_devin_session(env: Mapping[str, str] | None = None) -> bool:
    """Return ``True`` if we're running inside a Devin sandbox.

    Detection signals (any one is enough):

    * ``/opt/.devin`` exists on disk (the canonical install path
      inside Devin VMs).
    * ``DEVIN_SESSION_ID`` / ``DEVIN_USER_ID`` is set.
    * ``ENVRC`` points into ``/opt/.devin``.

    None of these are present on a normal operator's local box.
    The function is cheap (just stat + env lookup) so we call it
    on every write.
    """

    e = os.environ if env is None else env
    if e.get("DEVIN_SESSION_ID") or e.get("DEVIN_USER_ID"):
        return True
    envrc = e.get("ENVRC", "")
    if envrc.startswith("/opt/.devin"):
        return True
    return _has_devin_install_dir()


def resolve_env(
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
) -> dict[str, str]:
    """Return the merged ``os.environ + .env`` view.

    ``os.environ`` takes precedence. Used by the credential
    presence + verify code paths so a freshly-set ``.env`` value
    is visible without a server restart.
    """

    base = dict(os.environ if env is None else env)
    path = dotenv_path or default_dotenv_path()
    file_values = load_dotenv_file(path)
    for k, v in file_values.items():
        base.setdefault(k, v)
    return base


def write_credential(
    name: str,
    value: str,
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
    refresh_process_env: bool = True,
) -> Path:
    """Persist a single credential.

    On a local box this writes to ``.env`` and (optionally)
    mirrors the value into ``os.environ`` for the running
    process. Inside a Devin session this **always** raises
    :class:`StorageNotWritable` — Devin secrets are operator-set
    via the ``secrets`` tool, never via the dashboard.

    Returns the path that was written.
    """

    if not is_valid_env_var_name(name):
        raise ValueError(f"invalid env var name: {name!r}")
    if not isinstance(value, str):
        raise TypeError("value must be str")
    if value == "":
        raise ValueError("value must be non-empty")
    if "\n" in value or "\r" in value:
        raise ValueError("value must not contain a newline")

    if is_devin_session(env):
        raise StorageNotWritable(
            "credential write refused: running inside a Devin "
            "session. Use the secrets tool to add this key."
        )

    path = dotenv_path or default_dotenv_path()
    update_dotenv_file(path, {name: value})

    if refresh_process_env:
        os.environ[name] = value

    return path
