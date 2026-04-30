"""Tests for :mod:`intelligence_engine.cognitive.proposal_parser` (Wave-03 PR-5).

The parser is defensive: any malformed reply must return ``None`` so
the chat surface stays backwards-compatible with PR-4. The tests pin
both the happy path (a single ``propose`` fence with valid JSON) and
the every-failure-mode path (zero / multiple fences, malformed JSON,
non-dict body, missing fields, HOLD side, type errors)."""

from __future__ import annotations

import pytest

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalSideApi,
    ProposedSignalApi,
)
from intelligence_engine.cognitive.proposal_parser import extract_proposal


def _wrap(body: str, prefix: str = "Here is what I think:\n\n") -> str:
    return f"{prefix}```propose\n{body}\n```\n"


def test_extract_returns_none_on_empty_input() -> None:
    assert extract_proposal("") is None


def test_extract_returns_none_when_no_fence_present() -> None:
    assert extract_proposal("just chat, nothing structured") is None


def test_extract_returns_none_when_two_fences_present() -> None:
    body = '{"symbol": "EURUSD", "side": "BUY", "confidence": 0.5, "rationale": "x"}'
    text = _wrap(body) + "\n" + _wrap(body, prefix="and another:\n\n")
    assert extract_proposal(text) is None


def test_extract_returns_none_on_malformed_json() -> None:
    assert extract_proposal(_wrap("{not-json")) is None


def test_extract_returns_none_when_body_is_list_not_object() -> None:
    assert extract_proposal(_wrap("[1, 2, 3]")) is None


def test_extract_happy_path_buy() -> None:
    body = (
        '{"symbol": "EURUSD", "side": "BUY", '
        '"confidence": 0.62, "rationale": "macro setup"}'
    )
    proposal = extract_proposal(_wrap(body))
    assert proposal == ProposedSignalApi(
        symbol="EURUSD",
        side=ApprovalSideApi.BUY,
        confidence=0.62,
        rationale="macro setup",
    )


@pytest.mark.parametrize(
    "raw_side,expected",
    [
        ("buy", ApprovalSideApi.BUY),
        ("BUY", ApprovalSideApi.BUY),
        ("long", ApprovalSideApi.BUY),
        ("LONG", ApprovalSideApi.BUY),
        ("sell", ApprovalSideApi.SELL),
        ("short", ApprovalSideApi.SELL),
    ],
)
def test_extract_coerces_llm_side_strings(
    raw_side: str, expected: ApprovalSideApi
) -> None:
    body = (
        f'{{"symbol": "EURUSD", "side": "{raw_side}", '
        '"confidence": 0.5, "rationale": "x"}'
    )
    proposal = extract_proposal(_wrap(body))
    assert proposal is not None
    assert proposal.side is expected


def test_extract_rejects_hold_side() -> None:
    body = (
        '{"symbol": "EURUSD", "side": "HOLD", '
        '"confidence": 0.5, "rationale": "no edge"}'
    )
    assert extract_proposal(_wrap(body)) is None


def test_extract_rejects_unknown_side() -> None:
    body = (
        '{"symbol": "EURUSD", "side": "fly", '
        '"confidence": 0.5, "rationale": "x"}'
    )
    assert extract_proposal(_wrap(body)) is None


def test_extract_rejects_non_string_side() -> None:
    body = (
        '{"symbol": "EURUSD", "side": 1, '
        '"confidence": 0.5, "rationale": "x"}'
    )
    assert extract_proposal(_wrap(body)) is None


def test_extract_rejects_out_of_range_confidence() -> None:
    body = (
        '{"symbol": "EURUSD", "side": "BUY", '
        '"confidence": 1.5, "rationale": "x"}'
    )
    assert extract_proposal(_wrap(body)) is None


def test_extract_rejects_blank_symbol() -> None:
    body = (
        '{"symbol": "", "side": "BUY", '
        '"confidence": 0.5, "rationale": "x"}'
    )
    assert extract_proposal(_wrap(body)) is None


def test_extract_handles_leading_whitespace_on_fence() -> None:
    body = (
        '{"symbol": "EURUSD", "side": "BUY", '
        '"confidence": 0.5, "rationale": "x"}'
    )
    text = "    ```propose\n" + body + "\n    ```\n"
    proposal = extract_proposal(text)
    assert proposal is not None
    assert proposal.symbol == "EURUSD"
