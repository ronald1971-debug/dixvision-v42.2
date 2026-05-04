"""Cognitive / AI provider sub-package.

Houses the value-type contracts produced by AI provider adapters
(OpenAI, Gemini, Grok, DeepSeek, Devin). The canonical output shape is
:class:`sensory.cognitive.contracts.AIResponse`.

Authority discipline (see :mod:`sensory`): no engine imports, no FSM
mutation, no ledger writes. AI providers are *suggestion* sources only —
nothing under this sub-package may approve, execute, or mutate state.
"""
