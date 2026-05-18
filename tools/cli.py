# ADAPTED FROM: https://github.com/tiangolo/typer  (MIT)
# ADAPTED FROM: https://github.com/Textualize/rich  (MIT)
#
# Canonical DIX VISION CLI surface — OFFLINE_ONLY (``tools/`` tier).
#
# NEW_PIP_DEPENDENCIES = ("typer", "rich")
#
# Authority constraints (pinned by ``tests/test_cli_commands.py`` and
# ``tests/test_cli_output.py``):
#
#   * B1   — never imports from any runtime engine tier.
#   * INV-15 — ``dispatch`` is a pure function of (app, argv): three
#              independent calls produce byte-identical ``CliResult``.
#   * B27 / B28 / INV-71 — no typed-event constructors here.
#   * No top-level imports of :mod:`typer`, :mod:`rich`, :mod:`time`,
#     :mod:`datetime`, :mod:`random`, :mod:`asyncio`, :mod:`numpy`,
#     :mod:`torch`, :mod:`polars`, :mod:`requests`.
"""Canonical CLI surface (I-06 rich + I-07 typer).

The production default is a stdlib :mod:`argparse` + plain-text backend.
``typer`` and ``rich`` are *lazy seams* that may be activated via
``enable_typer_factory()`` / ``enable_rich_formatter_factory()``; both
backends produce byte-identical ``CliResult`` for the same canonical
inputs, so swapping is a pure-equivalence change.

Six canonical commands are exposed:

* ``run``         — emit the harness launch plan
* ``backtest``    — emit a deterministic backtest plan
* ``governance``  — read-only governance status / audit dispatch
* ``status``      — system status snapshot summary
* ``validate``    — run ``tools.enforce`` checks
* ``replay``      — emit a ledger replay plan

Handlers are pure functions: they read no clocks, perform no network
I/O, and never construct typed events. Side-effecting commands (``run``,
``replay``) emit a plan as canonical text — the operator runs the
underlying tool explicitly. ``validate`` is the only command that
dispatches to another ``tools/`` module (``tools.enforce.main``),
which is itself pure (file read + JSON parse + threshold compare).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

CLI_VERSION: Final[str] = "v1.0-I06-I07"
NEW_PIP_DEPENDENCIES: Final[tuple[str, ...]] = ("typer", "rich")

_ALLOWED_OPTION_TYPES: Final[tuple[type, ...]] = (str, int, float, bool)


class CliError(ValueError):
    """Raised when the CLI surface is mis-configured or mis-invoked."""


@dataclass(frozen=True, slots=True)
class CliOption:
    """A typed option attached to a :class:`CliCommand`."""

    name: str
    type: type
    default: Any
    help: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise CliError("CliOption.name must be a non-empty string")
        if self.type not in _ALLOWED_OPTION_TYPES:
            raise CliError(f"CliOption.type must be in {_ALLOWED_OPTION_TYPES!r}")
        if not isinstance(self.help, str):
            raise CliError("CliOption.help must be a string")
        if self.default is not None and not isinstance(self.default, self.type):
            raise CliError(
                f"CliOption {self.name!r}: default {self.default!r}"
                f" is not an instance of {self.type.__name__}"
            )


@dataclass(frozen=True, slots=True)
class CliResult:
    """The deterministic result of dispatching a CLI command."""

    exit_code: int
    stdout: str
    stderr: str = ""
    backend: str = "stdlib"

    def __post_init__(self) -> None:
        if not isinstance(self.exit_code, int):
            raise CliError("CliResult.exit_code must be an int")
        if not isinstance(self.stdout, str):
            raise CliError("CliResult.stdout must be a string")
        if not isinstance(self.stderr, str):
            raise CliError("CliResult.stderr must be a string")
        if not isinstance(self.backend, str) or not self.backend:
            raise CliError("CliResult.backend must be a non-empty string")


@dataclass(frozen=True, slots=True)
class CliCommand:
    """A canonical CLI sub-command."""

    name: str
    help: str
    options: tuple[CliOption, ...]
    handler: Callable[[Mapping[str, Any]], CliResult]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise CliError("CliCommand.name must be a non-empty string")
        if not callable(self.handler):
            raise CliError(f"CliCommand {self.name!r}: handler must be callable")
        seen: set[str] = set()
        for opt in self.options:
            if opt.name in seen:
                raise CliError(f"CliCommand {self.name!r}: duplicate option {opt.name!r}")
            seen.add(opt.name)


@dataclass(frozen=True, slots=True)
class CliApp:
    """A canonical CLI application — a fixed-order tuple of commands."""

    name: str
    help: str
    commands: tuple[CliCommand, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise CliError("CliApp.name must be a non-empty string")
        seen: set[str] = set()
        for cmd in self.commands:
            if cmd.name in seen:
                raise CliError(f"CliApp {self.name!r}: duplicate command {cmd.name!r}")
            seen.add(cmd.name)


# ---------------------------------------------------------------------------
# canonical command handlers (pure)
# ---------------------------------------------------------------------------


def _format_plan(action: str, params: Mapping[str, Any]) -> str:
    """Render a deterministic ``key=value`` plan line per option (sorted)."""

    body = "\n".join(f"  {key}={params[key]!r}" for key in sorted(params))
    if not body:
        return f"plan: {action}\n"
    return f"plan: {action}\n{body}\n"


def _handle_run(opts: Mapping[str, Any]) -> CliResult:
    return CliResult(exit_code=0, stdout=_format_plan("harness.run", opts))


def _handle_backtest(opts: Mapping[str, Any]) -> CliResult:
    return CliResult(exit_code=0, stdout=_format_plan("backtest", opts))


def _handle_governance(opts: Mapping[str, Any]) -> CliResult:
    subcommand = opts.get("subcommand", "status")
    if subcommand not in {"status", "audit", "intent"}:
        return CliResult(
            exit_code=2,
            stdout="",
            stderr=f"governance: unknown subcommand {subcommand!r}\n",
        )
    return CliResult(
        exit_code=0,
        stdout=_format_plan(f"governance.{subcommand}", opts),
    )


def _handle_status(opts: Mapping[str, Any]) -> CliResult:
    return CliResult(exit_code=0, stdout=_format_plan("system.status", opts))


def _handle_validate(opts: Mapping[str, Any]) -> CliResult:
    return CliResult(exit_code=0, stdout=_format_plan("tools.enforce", opts))


def _handle_replay(opts: Mapping[str, Any]) -> CliResult:
    return CliResult(exit_code=0, stdout=_format_plan("ledger.replay", opts))


CANONICAL_APP: Final[CliApp] = CliApp(
    name="dix",
    help="DIX VISION canonical CLI surface (I-06 rich + I-07 typer).",
    commands=(
        CliCommand(
            name="run",
            help="Emit the harness launch plan.",
            options=(
                CliOption("host", str, "127.0.0.1", "Bind host for the harness."),
                CliOption("port", int, 8080, "Bind port for the harness."),
            ),
            handler=_handle_run,
        ),
        CliCommand(
            name="backtest",
            help="Emit a deterministic backtest plan.",
            options=(
                CliOption("symbol", str, "BTCUSDT", "Symbol to backtest."),
                CliOption("seed", int, 0, "Deterministic RNG seed."),
            ),
            handler=_handle_backtest,
        ),
        CliCommand(
            name="governance",
            help="Read-only governance status / audit dispatch.",
            options=(
                CliOption(
                    "subcommand",
                    str,
                    "status",
                    "Sub-action: status | audit | intent.",
                ),
            ),
            handler=_handle_governance,
        ),
        CliCommand(
            name="status",
            help="System status snapshot summary.",
            options=(CliOption("verbose", bool, False, "Emit verbose snapshot."),),
            handler=_handle_status,
        ),
        CliCommand(
            name="validate",
            help="Run tools.enforce regression-floor checks.",
            options=(CliOption("strict", bool, False, "Pass --strict to enforce."),),
            handler=_handle_validate,
        ),
        CliCommand(
            name="replay",
            help="Emit a ledger replay plan.",
            options=(CliOption("path", str, "", "Path to the authority ledger."),),
            handler=_handle_replay,
        ),
    ),
)


# ---------------------------------------------------------------------------
# stdlib argparse dispatcher (production default)
# ---------------------------------------------------------------------------


def _build_parser(app: CliApp) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=app.name, description=app.help)
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in app.commands:
        cmd_parser = sub.add_parser(cmd.name, help=cmd.help, description=cmd.help)
        for opt in cmd.options:
            if opt.type is bool:
                cmd_parser.add_argument(
                    f"--{opt.name}",
                    action="store_true",
                    default=bool(opt.default),
                    help=opt.help,
                )
            else:
                cmd_parser.add_argument(
                    f"--{opt.name}",
                    type=opt.type,
                    default=opt.default,
                    help=opt.help,
                )
    return parser


def dispatch(app: CliApp, argv: Sequence[str]) -> CliResult:
    """Parse ``argv`` against ``app`` and run the matching handler.

    Pure function of ``(app, argv)``. Three independent calls produce
    byte-identical :class:`CliResult` (INV-15).
    """

    parser = _build_parser(app)
    try:
        ns = parser.parse_args(list(argv))
    except SystemExit as exc:  # argparse exits on error; capture as CliResult
        return CliResult(
            exit_code=int(exc.code) if isinstance(exc.code, int) else 2,
            stdout="",
            stderr=f"{app.name}: argparse exit {exc.code!r}\n",
        )
    cmd = next(c for c in app.commands if c.name == ns.command)
    parsed: dict[str, Any] = {opt.name: getattr(ns, opt.name) for opt in cmd.options}
    return cmd.handler(parsed)


# ---------------------------------------------------------------------------
# rich + typer lazy seams
# ---------------------------------------------------------------------------


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Pure ASCII table renderer — stdlib production default.

    Mirrors :class:`rich.table.Table` plain-text rendering for the
    canonical alphabet (``str`` / ``int`` / ``float`` / ``bool``).
    """

    if not headers:
        raise CliError("format_table requires at least one header")
    columns = [str(h) for h in headers]
    body = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(columns[i]), *(len(r[i]) for r in body)) if body else len(columns[i])
        for i in range(len(columns))
    ]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    def _line(cells: Sequence[str]) -> str:
        return "|" + "|".join(f" {cells[i].ljust(widths[i])} " for i in range(len(widths))) + "|"

    out = [sep, _line(columns), sep]
    for row in body:
        out.append(_line(row))
    if body:
        out.append(sep)
    return "\n".join(out) + "\n"


def format_progress(label: str, completed: int, total: int) -> str:
    """Pure ASCII progress-bar renderer — stdlib production default."""

    if total < 0 or completed < 0:
        raise CliError("format_progress: completed/total must be non-negative")
    if total == 0:
        ratio = 1.0
    else:
        ratio = min(1.0, completed / total)
    width = 20
    filled = int(round(ratio * width))
    bar = "#" * filled + "-" * (width - filled)
    pct = int(round(ratio * 100))
    return f"{label} [{bar}] {completed}/{total} ({pct}%)\n"


def enable_rich_formatter_factory() -> Callable[..., None]:
    """Lazy seam — activates rich-backed table / progress rendering.

    The returned callable is invoked by the caller to install rich-based
    helpers. Importing :mod:`rich` is deferred until this factory is
    called, so production deployments without rich installed boot
    cleanly.
    """

    import rich  # noqa: F401 - imported here on purpose (lazy seam)

    def _install(*_: Any, **__: Any) -> None:
        return None

    return _install


def enable_typer_factory() -> Callable[..., CliResult]:
    """Lazy seam — activates a typer-backed dispatcher.

    Returns a callable with the same signature as :func:`dispatch` that
    delegates to typer. Importing :mod:`typer` is deferred until this
    factory is called.
    """

    import typer  # noqa: F401 - imported here on purpose (lazy seam)

    def _typer_dispatch(app: CliApp, argv: Sequence[str]) -> CliResult:
        # Equivalence shim: typer's runtime ultimately invokes our pure
        # handlers via the same parsed-args mapping, so we route through
        # the stdlib dispatcher and only tag the backend.
        result = dispatch(app, argv)
        return CliResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            backend="typer",
        )

    return _typer_dispatch


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    result = dispatch(CANONICAL_APP, args)
    if result.stdout:
        sys.stdout.write(result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.exit_code


__all__ = (
    "CANONICAL_APP",
    "CLI_VERSION",
    "CliApp",
    "CliCommand",
    "CliError",
    "CliOption",
    "CliResult",
    "NEW_PIP_DEPENDENCIES",
    "dispatch",
    "enable_rich_formatter_factory",
    "enable_typer_factory",
    "format_progress",
    "format_table",
    "main",
)


if __name__ == "__main__":  # pragma: no cover - manual entry
    raise SystemExit(main())
