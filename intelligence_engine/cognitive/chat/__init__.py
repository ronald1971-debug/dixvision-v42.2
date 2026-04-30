"""Chat surfaces (Indira Chat / Dyon Chat) — Dashboard-2026 wave-03.

The wave-03 plan (``docs/dashboard_2026_wave03_cognitive_plan.md``)
puts every chat-domain LangChain / LangGraph touchpoint here, behind
authority-lint rule **B24**. The first piece is the
:class:`RegistryDrivenChatModel` adapter — the *only* way the
cognitive layer is permitted to invoke an LLM. The adapter never
names a vendor; routing is entirely registry-driven via
:func:`core.cognitive_router.select_providers`.
"""

from intelligence_engine.cognitive.chat.cognitive_chat_graph import (
    FEATURE_FLAG_ENV_VAR,
    ChatGraphState,
    CognitiveChatBundle,
    CognitiveChatDisabledError,
    CognitiveChatFeatureFlag,
    assemble_cognitive_chat,
    build_cognitive_chat_graph,
)
from intelligence_engine.cognitive.chat.registry_driven_chat_model import (
    AllProvidersFailedError,
    ChatTransport,
    FallbackAuditSink,
    NoEligibleProviderError,
    ProviderResolver,
    RegistryDrivenChatModel,
    TransientProviderError,
)

__all__ = [
    "AllProvidersFailedError",
    "ChatGraphState",
    "ChatTransport",
    "CognitiveChatBundle",
    "CognitiveChatDisabledError",
    "CognitiveChatFeatureFlag",
    "FEATURE_FLAG_ENV_VAR",
    "FallbackAuditSink",
    "NoEligibleProviderError",
    "ProviderResolver",
    "RegistryDrivenChatModel",
    "TransientProviderError",
    "assemble_cognitive_chat",
    "build_cognitive_chat_graph",
]
