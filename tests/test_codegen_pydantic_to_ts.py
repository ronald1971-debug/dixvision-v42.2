"""Drift guard + unit tests for the Pydantic→TypeScript generator.

Wave-02 PR-1 ships a Pydantic v2 → TypeScript generator
(``tools.codegen.pydantic_to_ts``). The dashboard imports the
generated file as a plain TS source so the React build does not need
Python at install time. Two things are tested here:

1. The checked-in TS file matches the live generator output (drift
   guard). If this fails, regenerate with::

       python -m tools.codegen.pydantic_to_ts \\
           core.contracts.api.credentials.CredentialsStatusResponse \\
           --out dashboard2026/src/types/generated/api.ts

2. Targeted regression coverage for two semantic edge cases that bit
   the first version of the generator: nullable-but-required fields
   ("optional" in JSON schema only means "key may be absent" — a
   required ``str | None`` is always present), and arrays of unions
   (``[]`` binds tighter than ``|`` in TypeScript).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

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
    "core.contracts.api.operator.OperatorSummaryResponse",
    "core.contracts.api.operator.OperatorActionResponse",
    "core.contracts.api.cognitive_chat.ChatStatusResponse",
    "core.contracts.api.cognitive_chat.ChatTurnRequest",
    "core.contracts.api.cognitive_chat.ChatTurnResponse",
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
    args = " ".join(_DASHBOARD_API_MODELS)
    assert actual == expected, (
        "TypeScript codegen drift detected — regenerate with "
        f"`python -m tools.codegen.pydantic_to_ts {args} "
        "--out dashboard2026/src/types/generated/api.ts`"
    )


class _NullableRequired(BaseModel):
    """``signup_url`` is required-but-nullable (no default).

    Pydantic always serialises this key (as a string or ``null``), so
    the generated TS must NOT mark it ``?`` (which would let TS infer
    ``undefined`` on read).
    """

    signup_url: str | None
    label: str


def test_required_nullable_field_is_not_marked_optional() -> None:
    rendered = render_models((f"{__name__}._NullableRequired",))
    # `field?:` and `field:` both end with the type after a colon, so
    # match the exact tokenisation we emit:
    assert "signup_url: string | null" in rendered, rendered
    assert "signup_url?:" not in rendered, rendered
    # And a plain required field is unaffected.
    assert "label: string" in rendered, rendered


class _OptionalNullable(BaseModel):
    """``hint`` has a default of ``None`` — *both* nullable AND optional.

    JSON schema marks it not-required, so the TS field should keep
    the ``?`` modifier (key really may be absent in serialised JSON
    when the model uses ``exclude_unset``).
    """

    hint: str | None = None


def test_default_none_field_keeps_optional_modifier() -> None:
    rendered = render_models((f"{__name__}._OptionalNullable",))
    assert "hint?: string | null" in rendered, rendered


class _ArrayOfUnion(BaseModel):
    """List of nullable strings — exercises operator-precedence in TS.

    ``string | null[]`` would be parsed as ``string | Array<null>``;
    the generator must parenthesise the union before applying ``[]``.
    """

    tags: list[str | None]


def test_array_of_union_is_parenthesised() -> None:
    rendered = render_models((f"{__name__}._ArrayOfUnion",))
    assert "tags: (string | null)[]" in rendered, rendered
    assert "string | null[]" not in rendered, rendered


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
