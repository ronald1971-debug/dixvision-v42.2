"""Drift guard: the checked-in TS file matches the live codegen output.

Wave-02 PR-1 ships a Pydantic v2 → TypeScript generator
(``tools.codegen.pydantic_to_ts``). The dashboard imports the
generated file as a plain TS source so the React build does not need
Python at install time. To prevent silent drift between the Pydantic
models and the TS mirror, this test invokes the same generator in
``--check`` mode and fails the build if the on-disk file is stale.

If this test fails, regenerate with::

    python -m tools.codegen.pydantic_to_ts \\
        core.contracts.api.credentials.CredentialsStatusResponse \\
        --out dashboard2026/src/types/generated/api.ts
"""

from __future__ import annotations

from pathlib import Path

from tools.codegen.pydantic_to_ts import render_models

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GENERATED_TS = (
    _REPO_ROOT / "dashboard2026" / "src" / "types" / "generated" / "api.ts"
)


# Single source of truth for which Pydantic models the dashboard sees.
# Adding a new API model: register it here AND in the regeneration
# command in dashboard2026/README.md.
_DASHBOARD_API_MODELS: tuple[str, ...] = (
    "core.contracts.api.credentials.CredentialsStatusResponse",
)


def test_generator_runs_without_error() -> None:
    """The generator must produce *some* output for the registered set."""

    rendered = render_models(_DASHBOARD_API_MODELS)
    assert rendered, "renderer produced empty output"
    assert "AUTO-GENERATED" in rendered


def test_checked_in_ts_file_matches_generator() -> None:
    """``api.ts`` on disk equals what the generator emits today."""

    expected = render_models(_DASHBOARD_API_MODELS)
    assert _GENERATED_TS.exists(), (
        f"missing {_GENERATED_TS.relative_to(_REPO_ROOT)} — "
        "run `python -m tools.codegen.pydantic_to_ts ...`"
    )
    actual = _GENERATED_TS.read_text(encoding="utf-8")
    assert actual == expected, (
        "TypeScript codegen drift detected — regenerate with "
        "`python -m tools.codegen.pydantic_to_ts "
        "core.contracts.api.credentials.CredentialsStatusResponse "
        "--out dashboard2026/src/types/generated/api.ts`"
    )


def test_response_model_round_trips_through_endpoint() -> None:
    """Sanity: the FastAPI endpoint returns the Pydantic shape."""

    from fastapi.testclient import TestClient

    from core.contracts.api.credentials import CredentialsStatusResponse
    from ui.server import app

    client = TestClient(app)
    res = client.get("/api/credentials/status")
    assert res.status_code == 200, res.text
    # If the endpoint regresses to a plain dict the schema validator
    # will reject extra keys (`extra="forbid"`) and raise here.
    parsed = CredentialsStatusResponse.model_validate(res.json())
    assert parsed.summary.total == len(parsed.items)
