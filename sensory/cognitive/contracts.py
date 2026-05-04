"""Value type for AI-provider responses.

Resolves the forward-declared
``sensory.cognitive.contracts.AIResponse`` schema path referenced by
the OpenAI / Gemini / Grok / DeepSeek / Devin rows in
:file:`registry/data_source_registry.yaml`.

Frozen + slotted dataclass (INV-15 deterministic-replay safe).

Note on authority: an :class:`AIResponse` is *advisory* — it carries
the provider's text and structured metadata into the system, never an
execution decision. The intelligence engine is responsible for any
downstream action; the cognitive sub-package only emits this contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class AIResponse:
    """One response turn from an AI provider.

    Generic across OpenAI, Google Gemini, xAI Grok, DeepSeek, and the
    Devin MCP integration.

    Attributes:
        ts_ns: Monotonic ingestion timestamp in nanoseconds (caller-
            supplied, never derived from the payload — INV-15).
        source: Stable source identifier matching the SCVS registry row
            (e.g. ``"OPENAI"``, ``"GEMINI"``, ``"GROK"``,
            ``"DEEPSEEK"``, ``"DEVIN"``). Empty string is rejected.
        request_id: Provider-stable id for the originating request.
            Empty string is rejected (we audit every AI turn).
        model: Provider-stable model identifier (e.g. ``"gpt-4o"``,
            ``"gemini-2.0-flash"``, ``"grok-2"``). Empty string is
            rejected.
        body: Response text. Empty string allowed for tool-call-only
            turns.
        finish_reason: Provider-stable termination cause (e.g.
            ``"stop"``, ``"length"``, ``"tool_calls"``). Empty string
            allowed.
        prompt_tokens: Optional prompt token count. ``None`` when the
            provider omits it. Must be ``>= 0`` when present.
        completion_tokens: Optional completion token count. ``None``
            when the provider omits it. Must be ``>= 0`` when present.
        latency_ms: Optional end-to-end latency in milliseconds. ``None``
            when not measured. Must be ``>= 0`` when present.
        meta: Free-form structural metadata (provider-specific finish
            reason, safety flags, etc.). No secrets, no API keys.
    """

    ts_ns: int
    source: str
    request_id: str
    model: str
    body: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("AIResponse.source must be non-empty")
        if not self.request_id:
            raise ValueError("AIResponse.request_id must be non-empty")
        if not self.model:
            raise ValueError("AIResponse.model must be non-empty")
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError(
                "AIResponse.prompt_tokens must be >= 0 or None"
            )
        if (
            self.completion_tokens is not None
            and self.completion_tokens < 0
        ):
            raise ValueError(
                "AIResponse.completion_tokens must be >= 0 or None"
            )
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError(
                "AIResponse.latency_ms must be >= 0 or None"
            )
