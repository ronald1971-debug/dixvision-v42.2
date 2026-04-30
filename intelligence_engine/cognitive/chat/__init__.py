"""Chat surfaces (Indira Chat / Dyon Chat) — Dashboard-2026 wave-03.

The wave-03 plan (``docs/dashboard_2026_wave03_cognitive_plan.md``)
puts every chat-domain LangChain / LangGraph touchpoint here, behind
authority-lint rule **B24**. The first piece is the
:class:`RegistryDrivenChatModel` adapter — the *only* way the
cognitive layer is permitted to invoke an LLM. The adapter never
names a vendor; routing is entirely registry-driven via
:func:`core.cognitive_router.select_providers`.
"""

from intelligence_engine.cognitive.chat.registry_driven_chat_model import (
    AllProvidersFailedError,
    ChatTransport,
    FallbackAuditSink,
    NoEligibleProviderError,
    RegistryDrivenChatModel,
    TransientProviderError,
)

__all__ = [
    "AllProvidersFailedError",
    "ChatTransport",
    "FallbackAuditSink",
    "NoEligibleProviderError",
    "RegistryDrivenChatModel",
    "TransientProviderError",
]
