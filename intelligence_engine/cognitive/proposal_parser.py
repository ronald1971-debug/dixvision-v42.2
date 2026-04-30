"""Wave-03 PR-5 — extract a structured ``propose`` block from a chat reply.

The cognitive chat graph (PR-3) replies to the operator in natural
language. PR-5 carves out a *narrow* protocol on top of that prose so
the assistant can additionally propose a strategy action that the
operator can approve or reject from the dashboard.

Wire format (kept deliberately simple):

    The assistant may include a single fenced JSON block tagged with
    ``propose``. Example reply::

        I think there's a setup forming on EURUSD. Here's the proposal:

        ```propose
        {"symbol": "EURUSD", "side": "BUY", "confidence": 0.62,
         "rationale": "..."}
        ```

The parser is *defensive*: any malformed / missing block returns
``None`` and the chat reply is treated as conversational. This keeps
PR-5 backwards-compatible with PR-4 — clients that have not been
updated to render the approval panel keep working without change.

Isolation contract (B1): no governance / system imports. The parser
returns the wire shape (:class:`ProposedSignalApi`); the runtime in
``ui.cognitive_chat_runtime`` is responsible for handing it to the
queue.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from core.contracts.api.cognitive_chat_approvals import (
    ApprovalSideApi,
    ProposedSignalApi,
)

__all__ = [
    "PROPOSE_FENCE_PATTERN",
    "extract_proposal",
]


# Match a single fenced block tagged ``propose``. The fence must be at
# the start of a line (after optional whitespace) so a JSON example
# *inside* the model's prose without a leading newline doesn't get
# mistaken for a real proposal.
PROPOSE_FENCE_PATTERN = re.compile(
    r"^[ \t]*```propose[ \t]*\n(?P<body>.*?)\n[ \t]*```",
    re.DOTALL | re.MULTILINE,
)


def _coerce_side(value: Any) -> ApprovalSideApi | None:
    """Map LLM-style side strings to the strict enum.

    Models often emit ``"buy"`` / ``"long"`` / ``"sell"`` / ``"short"``
    interchangeably. This is the *only* permissive layer in PR-5;
    everything downstream uses the enum.
    """

    if not isinstance(value, str):
        return None
    norm = value.strip().upper()
    if norm in {"BUY", "LONG"}:
        return ApprovalSideApi.BUY
    if norm in {"SELL", "SHORT"}:
        return ApprovalSideApi.SELL
    if norm == "HOLD":
        return ApprovalSideApi.HOLD
    return None


def extract_proposal(reply_text: str) -> ProposedSignalApi | None:
    """Return the parsed proposal or ``None`` if the reply has no block.

    Parsing rules:

    * The reply must contain *exactly one* ``propose`` fence; multiple
      blocks fall back to ``None`` so the operator is never asked to
      approve an ambiguous bundle.
    * The body must be valid JSON with the four required fields.
    * ``side="HOLD"`` is rejected (a non-actionable proposal is not
      worth queuing — the chat reply still reaches the operator).
    * Any other validation error returns ``None`` so the chat reply
      remains conversational.
    """

    if not reply_text:
        return None
    matches = PROPOSE_FENCE_PATTERN.findall(reply_text)
    if len(matches) != 1:
        return None

    try:
        raw = json.loads(matches[0])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None

    side = _coerce_side(raw.get("side"))
    if side is None or side is ApprovalSideApi.HOLD:
        return None

    try:
        return ProposedSignalApi(
            symbol=str(raw.get("symbol", "")),
            side=side,
            confidence=float(raw.get("confidence", 0.0)),
            rationale=str(raw.get("rationale", "")),
        )
    except (TypeError, ValueError, ValidationError):
        return None
