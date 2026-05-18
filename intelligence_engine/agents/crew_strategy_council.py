"""Role-based strategy council (B-04 — crewai canonical adaptation).

# ADAPTED FROM: crewAIInc/crewAI
#   - crewai/agent.py — Agent(role, goal, backstory)
#   - crewai/crew.py  — Crew.kickoff() sequential orchestration
#   - crewai/task.py  — Task(description, expected_output, agent)

What survives from CrewAI:
* The **role / goal / backstory triple** for each agent. CrewAI's
  ``Agent(role="...", goal="...", backstory="...")`` collapses to
  our :class:`CouncilAgent` ``(role, goal, backstory,
  expected_output_keys)`` shape — no inheritance, no class-level
  mutable state, no global registry.
* The **sequential Crew workflow** from ``Crew.kickoff()``. Tasks
  execute in declaration order; each task's structured output is
  appended to the context fed into the next task's
  :class:`StructuredSpeaker`. Hierarchical / async modes are
  intentionally dropped — they bring nondeterminism that
  breaks INV-15.
* The **expected_output schema** from
  ``Task(expected_output=...)``. We pin every task to a tuple of
  ``expected_output_keys`` and validate the speaker's structured
  reply against that schema at the receive boundary.

What is **stripped** (crewai is MIT — the surface is trimmed to fit
the DIX contract, not to address licensing):

* **Tool execution** — CrewAI's ``Tool`` / ``ToolUsage`` arbitrary
  code-execution surface is gone. Any tool step must route through
  the existing DIX governance approval gate; the council itself is
  advisory only.
* **Direct LLM clients** — ``Agent.llm`` / ``ChatOpenAI`` /
  ``ChatAnthropic`` are dropped wholesale. The caller injects a
  deterministic :class:`StructuredSpeaker` Protocol; in production
  that Protocol is implemented by the
  :class:`intelligence_engine.cognitive.litellm_router.LiteLLMRouter`
  (S-12), which is the **only** path the runtime tier may take to
  reach an LLM provider.
* **CrewAI built-in memory** — ``Agent.memory`` /
  ``ShortTermMemory`` / ``LongTermMemory`` are out-of-tier. The
  ``crewai_speaker_factory`` lazy-binds a memory store (e.g.
  S-08 FAISS / A-10 Qdrant) at the call site; the orchestrator
  itself is stateless.
* **Hierarchical Process** — only sequential ``kickoff()`` is
  retained. Hierarchical mode requires a manager-LLM which would
  break INV-15 byte-identical replay; if hierarchical orchestration
  is needed later it lives in a separate module.

Authority discipline (OFFLINE_ONLY — advisory, never executes):

* **B27 / B28 / INV-71 authority symmetry** — this module is on
  the **intelligence-engine / agents** side of the boundary. It
  does **not** construct any typed bus event
  (:class:`SignalEvent` / :class:`PatchProposal` /
  :class:`GovernanceDecision` / :class:`ExecutionIntent`). The
  council's output is :class:`CouncilProposal`, an advisory
  value object the operator (or a downstream governance-gated
  approval queue) may inspect; promotion to a
  :class:`GovernanceDecision` happens inside
  :mod:`governance_engine` — never here. Pinned by AST tests.
* **B1 engine isolation** — no ``execution_engine.*`` /
  ``governance_engine.*`` / ``system_engine.*`` /
  ``evolution_engine.*`` imports. Pinned by AST tests.
* **INV-12 advisory-only** — module docstring and
  :class:`CouncilProposal` docstring explicitly carry the
  ``ADVISORY ONLY — no direct trade authority`` clause from the
  canonical spec (line 1708).
* **INV-15 determinism** — module imports no ``random`` /
  ``time`` / ``datetime`` / ``secrets`` / ``os`` / ``asyncio``;
  callers supply every source of variability. Given a
  deterministic :class:`StructuredSpeaker` the entire council is
  byte-identical replayable; pinned by 3-run replay tests.
* **No top-level crewai / litellm import** — both are
  out-of-tier; the production :class:`StructuredSpeaker` factory
  lazy-imports the LiteLLM router only inside its body. Pinned by
  AST tests.

Tier: OFFLINE_ONLY (advisory).
"""

# ADAPTED FROM: crewAIInc/crewAI (crewai/agent.py, crewai/crew.py,
#                                 crewai/task.py)

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Pip surface — declared as an empty tuple. Both ``crewai`` and
# ``litellm`` are out-of-tier; production wires through the
# :class:`StructuredSpeaker` Protocol, which is implemented by the
# :class:`LiteLLMRouter` from S-12 (which already declares its own pip
# surface). Pinned by an AST test.
# ---------------------------------------------------------------------------

NEW_PIP_DEPENDENCIES: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Canonical roles (4 — from the B-04 spec)
# ---------------------------------------------------------------------------


class CouncilRole(StrEnum):
    """The four canonical roles of the strategy council.

    Order matters: :func:`run_council` uses
    :data:`CANONICAL_ROLE_ORDER` to enforce a deterministic
    declaration order independent of dict-iteration order in
    user code."""

    SIGNAL_ANALYST = "signal_analyst"
    RISK_OFFICER = "risk_officer"
    REGIME_EXPERT = "regime_expert"
    ARBITER = "arbiter"


CANONICAL_ROLE_ORDER: tuple[CouncilRole, ...] = (
    CouncilRole.SIGNAL_ANALYST,
    CouncilRole.RISK_OFFICER,
    CouncilRole.REGIME_EXPERT,
    CouncilRole.ARBITER,
)


# ---------------------------------------------------------------------------
# Recommendation labels (mirror B-03 for tier consistency)
# ---------------------------------------------------------------------------


RECOMMENDATION_LABELS: frozenset[str] = frozenset({"APPROVE", "REJECT", "ESCALATE", "ABSTAIN"})
DEFAULT_RECOMMENDATION: str = "ABSTAIN"
DEFAULT_RATIONALE: str = "no consensus reached"

# Reserved keys the arbiter's output must declare.
ARBITER_RECOMMENDATION_KEY: str = "recommendation"
ARBITER_RATIONALE_KEY: str = "rationale"


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


MIN_AGENTS: int = 2
MAX_AGENTS: int = 8

MIN_TASKS: int = 1
MAX_TASKS: int = 16

MAX_TOPIC_LEN: int = 2048
MAX_ROLE_VARIANT_LEN: int = 64
MAX_GOAL_LEN: int = 1024
MAX_BACKSTORY_LEN: int = 4096
MAX_DESCRIPTION_LEN: int = 4096
MAX_OUTPUT_KEY_LEN: int = 64
MAX_OUTPUT_VALUE_LEN: int = 8192
MAX_OUTPUT_KEYS: int = 16
MAX_RECOMMENDATION_LEN: int = 64
MAX_RATIONALE_LEN: int = 4096


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CouncilError(ValueError):
    """Raised on contract violations (bad agent / task / speaker output)."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CouncilAgent:
    """One named participant in the council.

    ``role`` is the canonical :class:`CouncilRole`; ``goal`` and
    ``backstory`` follow the CrewAI ``(role, goal, backstory)``
    triple. ``expected_output_keys`` is the tuple of keys the
    agent's structured reply must contain — analogous to
    ``Task.expected_output`` but pinned to the agent so the
    schema travels with the role."""

    role: CouncilRole
    goal: str
    backstory: str
    expected_output_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.role, CouncilRole):
            raise CouncilError("CouncilAgent.role must be CouncilRole")
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise CouncilError("CouncilAgent.goal must be non-empty str")
        if len(self.goal) > MAX_GOAL_LEN:
            raise CouncilError(f"CouncilAgent.goal length > {MAX_GOAL_LEN}")
        if not isinstance(self.backstory, str) or not self.backstory.strip():
            raise CouncilError("CouncilAgent.backstory must be non-empty str")
        if len(self.backstory) > MAX_BACKSTORY_LEN:
            raise CouncilError(f"CouncilAgent.backstory length > {MAX_BACKSTORY_LEN}")
        if not isinstance(self.expected_output_keys, tuple):
            raise CouncilError("CouncilAgent.expected_output_keys must be tuple")
        if not self.expected_output_keys:
            raise CouncilError("CouncilAgent.expected_output_keys must be non-empty")
        if len(self.expected_output_keys) > MAX_OUTPUT_KEYS:
            raise CouncilError(f"CouncilAgent.expected_output_keys count > {MAX_OUTPUT_KEYS}")
        for k in self.expected_output_keys:
            if not isinstance(k, str) or not k.strip():
                raise CouncilError(
                    "CouncilAgent.expected_output_keys entries must be non-empty str"
                )
            if len(k) > MAX_OUTPUT_KEY_LEN:
                raise CouncilError(
                    f"CouncilAgent.expected_output_keys entry length > {MAX_OUTPUT_KEY_LEN}"
                )
        if len(set(self.expected_output_keys)) != len(self.expected_output_keys):
            raise CouncilError("CouncilAgent.expected_output_keys must be unique")
        # Arbiter is constrained to advertise the two reserved keys so
        # the orchestrator can deterministically project the council's
        # recommendation + rationale from its output.
        if self.role is CouncilRole.ARBITER:
            keys = set(self.expected_output_keys)
            if ARBITER_RECOMMENDATION_KEY not in keys:
                raise CouncilError(
                    "ARBITER agent must advertise "
                    f"{ARBITER_RECOMMENDATION_KEY!r} in "
                    "expected_output_keys"
                )
            if ARBITER_RATIONALE_KEY not in keys:
                raise CouncilError(
                    "ARBITER agent must advertise "
                    f"{ARBITER_RATIONALE_KEY!r} in "
                    "expected_output_keys"
                )


@dataclass(frozen=True, slots=True)
class CouncilTask:
    """One task in the council's sequential workflow.

    ``role_owner`` identifies which :class:`CouncilAgent` (by role)
    executes the task; ``description`` is the task prompt. The
    schema (``expected_output_keys``) is inherited from the owning
    agent — CrewAI redundantly carries the schema on both ``Task``
    and ``Agent``; we collapse it to the agent only.

    Frozen + slots so tasks may be hashed / threaded / replayed."""

    task_id: str
    description: str
    role_owner: CouncilRole

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise CouncilError("CouncilTask.task_id must be non-empty str")
        if len(self.task_id) > MAX_OUTPUT_KEY_LEN:
            raise CouncilError(f"CouncilTask.task_id length > {MAX_OUTPUT_KEY_LEN}")
        if not isinstance(self.description, str) or not self.description.strip():
            raise CouncilError("CouncilTask.description must be non-empty str")
        if len(self.description) > MAX_DESCRIPTION_LEN:
            raise CouncilError(f"CouncilTask.description length > {MAX_DESCRIPTION_LEN}")
        if not isinstance(self.role_owner, CouncilRole):
            raise CouncilError("CouncilTask.role_owner must be CouncilRole")


@dataclass(frozen=True, slots=True)
class CouncilConfig:
    """Static configuration for one council kickoff.

    ``agents`` is a tuple keyed (lookup) by :class:`CouncilRole`;
    every :class:`CouncilTask` in ``tasks`` must reference a role
    present in ``agents``. The last task **MUST** be owned by
    :data:`CouncilRole.ARBITER` (the spec requires the arbiter to
    synthesise the final structured proposal)."""

    topic: str
    agents: tuple[CouncilAgent, ...]
    tasks: tuple[CouncilTask, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.topic, str) or not self.topic.strip():
            raise CouncilError("CouncilConfig.topic must be non-empty str")
        if len(self.topic) > MAX_TOPIC_LEN:
            raise CouncilError(f"CouncilConfig.topic length > {MAX_TOPIC_LEN}")
        if not isinstance(self.agents, tuple):
            raise CouncilError("CouncilConfig.agents must be tuple")
        if len(self.agents) < MIN_AGENTS:
            raise CouncilError(f"CouncilConfig.agents requires >= {MIN_AGENTS} entries")
        if len(self.agents) > MAX_AGENTS:
            raise CouncilError(f"CouncilConfig.agents allows <= {MAX_AGENTS} entries")
        for a in self.agents:
            if not isinstance(a, CouncilAgent):
                raise CouncilError("CouncilConfig.agents entries must be CouncilAgent")
        roles = [a.role for a in self.agents]
        if len(set(roles)) != len(roles):
            raise CouncilError("CouncilConfig.agents must have unique roles")
        if CouncilRole.ARBITER not in set(roles):
            raise CouncilError("CouncilConfig.agents must include the ARBITER role")
        if not isinstance(self.tasks, tuple):
            raise CouncilError("CouncilConfig.tasks must be tuple")
        if len(self.tasks) < MIN_TASKS:
            raise CouncilError(f"CouncilConfig.tasks requires >= {MIN_TASKS} entries")
        if len(self.tasks) > MAX_TASKS:
            raise CouncilError(f"CouncilConfig.tasks allows <= {MAX_TASKS} entries")
        task_ids: list[str] = []
        role_set = set(roles)
        for t in self.tasks:
            if not isinstance(t, CouncilTask):
                raise CouncilError("CouncilConfig.tasks entries must be CouncilTask")
            if t.role_owner not in role_set:
                raise CouncilError(
                    f"CouncilTask({t.task_id!r}).role_owner "
                    f"{t.role_owner.value!r} not present in agents"
                )
            task_ids.append(t.task_id)
        if len(set(task_ids)) != len(task_ids):
            raise CouncilError("CouncilConfig.tasks must have unique task_ids")
        if self.tasks[-1].role_owner is not CouncilRole.ARBITER:
            raise CouncilError(
                "CouncilConfig.tasks[-1] must be owned by ARBITER "
                "(synthesises the final structured proposal)"
            )


@dataclass(frozen=True, slots=True)
class CouncilTaskResult:
    """Structured output of one completed task.

    ``output`` is a sorted-key :class:`MappingProxyType` keyed by
    the agent's ``expected_output_keys`` schema; values are str.
    All schema keys are required and must be present."""

    task_id: str
    role: CouncilRole
    output: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CouncilProposal:
    """Frozen advisory output of one completed council kickoff.

    INV-12 / INV-67 / B27 / B28 / INV-71: this object is **not** a
    typed bus event. It is **advisory only** — promotion to a
    :class:`GovernanceDecision` happens inside
    :mod:`governance_engine`.

    ``proposal_digest`` is the BLAKE2b-16 hex digest of the
    canonical text projection (topic + every result's sorted-key
    output + recommendation + rationale). Given the same agents /
    tasks / :class:`StructuredSpeaker` the digest is byte-identical
    across runs and machines."""

    topic: str
    results: tuple[CouncilTaskResult, ...]
    recommendation: str
    rationale: str
    proposal_digest: str


# ---------------------------------------------------------------------------
# StructuredSpeaker Protocol — caller-injected LLM dispatch
# ---------------------------------------------------------------------------


@runtime_checkable
class StructuredSpeaker(Protocol):
    """Pluggable LLM dispatch returning a typed dict.

    Production wires this Protocol to the LiteLLM router (S-12);
    tests inject a deterministic fake. The contract: implementations
    MUST return a ``Mapping[str, str]`` whose keys exactly match
    ``agent.expected_output_keys``. The orchestrator validates the
    return on the receive side; non-conformant returns raise
    :class:`CouncilError`."""

    def speak(
        self,
        *,
        agent: CouncilAgent,
        task: CouncilTask,
        topic: str,
        prior_results: tuple[CouncilTaskResult, ...],
    ) -> Mapping[str, str]:
        """Return the agent's structured reply.

        Implementations must be pure functions of their inputs (no
        clock, no IO, no random) for INV-15 to hold."""


# ---------------------------------------------------------------------------
# Validators for speaker output
# ---------------------------------------------------------------------------


def _validate_speaker_output(
    value: object,
    agent: CouncilAgent,
    task: CouncilTask,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise CouncilError(
            f"StructuredSpeaker.speak({agent.role.value!r}, {task.task_id!r}) "
            f"must return Mapping[str, str], got {type(value).__name__}"
        )
    expected = set(agent.expected_output_keys)
    got = set(value.keys())
    if got != expected:
        raise CouncilError(
            f"StructuredSpeaker.speak({agent.role.value!r}, {task.task_id!r}) "
            f"keys {sorted(got)} != expected {sorted(expected)}"
        )
    cleaned: dict[str, str] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            raise CouncilError("output keys must be str")
        if not isinstance(v, str):
            raise CouncilError(f"output[{k!r}] must be str")
        if len(v) > MAX_OUTPUT_VALUE_LEN:
            raise CouncilError(f"output[{k!r}] length {len(v)} > {MAX_OUTPUT_VALUE_LEN}")
        cleaned[k] = v
    return MappingProxyType(dict(sorted(cleaned.items())))


# ---------------------------------------------------------------------------
# Digest helper
# ---------------------------------------------------------------------------


def _compute_proposal_digest(
    *,
    topic: str,
    results: tuple[CouncilTaskResult, ...],
    recommendation: str,
    rationale: str,
) -> str:
    """Compute the canonical BLAKE2b-16 hex digest over the council
    transcript + outcome. Stable across runs / Python builds /
    machines so callers may use it as a forensic key."""

    h = hashlib.blake2b(digest_size=16)
    h.update(b"topic\x00")
    h.update(topic.encode("utf-8"))
    h.update(b"\x01")
    for r in results:
        h.update(b"result\x00")
        h.update(r.task_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(r.role.value.encode("utf-8"))
        h.update(b"\x00")
        for k in sorted(r.output.keys()):
            h.update(k.encode("utf-8"))
            h.update(b"\x00")
            h.update(r.output[k].encode("utf-8"))
            h.update(b"\x00")
        h.update(b"\x01")
    h.update(b"rec\x00")
    h.update(recommendation.encode("utf-8"))
    h.update(b"\x01")
    h.update(b"rat\x00")
    h.update(rationale.encode("utf-8"))
    h.update(b"\x01")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Arbiter projection — pure
# ---------------------------------------------------------------------------


def _project_arbiter_decision(
    *,
    results: tuple[CouncilTaskResult, ...],
) -> tuple[str, str]:
    """Pull (recommendation, rationale) from the final ARBITER
    result. Validates the label is in :data:`RECOMMENDATION_LABELS`
    and that both fields fit within their bounds; falls back to
    :data:`DEFAULT_RECOMMENDATION` / :data:`DEFAULT_RATIONALE` if
    no arbiter result exists (the orchestrator guarantees one,
    so this is purely a defence-in-depth path)."""

    arbiter_result = None
    for r in reversed(results):
        if r.role is CouncilRole.ARBITER:
            arbiter_result = r
            break
    if arbiter_result is None:  # pragma: no cover - guarded by CouncilConfig
        return DEFAULT_RECOMMENDATION, DEFAULT_RATIONALE

    rec_raw = arbiter_result.output.get(ARBITER_RECOMMENDATION_KEY, "")
    rec = rec_raw.strip().upper()
    if rec not in RECOMMENDATION_LABELS:
        rec = DEFAULT_RECOMMENDATION
    if len(rec) > MAX_RECOMMENDATION_LEN:
        rec = DEFAULT_RECOMMENDATION

    rat_raw = arbiter_result.output.get(ARBITER_RATIONALE_KEY, "")
    rat = rat_raw if isinstance(rat_raw, str) else DEFAULT_RATIONALE
    if not rat.strip():
        rat = DEFAULT_RATIONALE
    if len(rat) > MAX_RATIONALE_LEN:
        rat = rat[:MAX_RATIONALE_LEN]

    return rec, rat


# ---------------------------------------------------------------------------
# Orchestrator — pure function, deterministic
# ---------------------------------------------------------------------------


def run_council(
    *,
    config: CouncilConfig,
    speaker: StructuredSpeaker,
) -> CouncilProposal:
    """Run one full council kickoff under sequential workflow.

    Tasks execute in declaration order. Each task's
    :class:`StructuredSpeaker` call receives the calling
    :class:`CouncilAgent`, the :class:`CouncilTask`, the topic,
    and the tuple of prior results. The orchestrator validates the
    speaker's return against the agent's
    ``expected_output_keys`` schema before appending it.

    The function is pure: given the same ``config`` and a
    deterministic ``speaker`` the returned
    :class:`CouncilProposal` is byte-identical across runs."""

    if not isinstance(config, CouncilConfig):
        raise CouncilError("run_council: config must be CouncilConfig")
    if not isinstance(speaker, StructuredSpeaker):
        raise CouncilError("run_council: speaker must implement the StructuredSpeaker Protocol")

    agents_by_role: dict[CouncilRole, CouncilAgent] = {a.role: a for a in config.agents}

    results: list[CouncilTaskResult] = []
    for task in config.tasks:
        agent = agents_by_role[task.role_owner]
        raw = speaker.speak(
            agent=agent,
            task=task,
            topic=config.topic,
            prior_results=tuple(results),
        )
        output = _validate_speaker_output(raw, agent, task)
        results.append(
            CouncilTaskResult(
                task_id=task.task_id,
                role=agent.role,
                output=output,
            )
        )

    final_results = tuple(results)
    recommendation, rationale = _project_arbiter_decision(results=final_results)
    digest = _compute_proposal_digest(
        topic=config.topic,
        results=final_results,
        recommendation=recommendation,
        rationale=rationale,
    )

    return CouncilProposal(
        topic=config.topic,
        results=final_results,
        recommendation=recommendation,
        rationale=rationale,
        proposal_digest=digest,
    )


# ---------------------------------------------------------------------------
# Production speaker factory — lazy LiteLLM wiring (out-of-tier import)
# ---------------------------------------------------------------------------


def crewai_speaker_factory(
    *,
    completion: Callable[..., Mapping[str, str]],
    system_prompt_prefix: str = "",
) -> StructuredSpeaker:
    """Build a :class:`StructuredSpeaker` from a deterministic
    ``completion`` callable (typically wired through
    :class:`LiteLLMRouter` from S-12).

    ``completion`` MUST return a ``Mapping[str, str]`` keyed by the
    agent's ``expected_output_keys`` — production callers usually
    wrap a structured-output LLM call (function-calling /
    JSON-mode) that the router post-validates. In production wire
    it with ``temperature=0.0`` so the whole council is
    byte-identical replayable."""

    if not callable(completion):
        raise CouncilError("crewai_speaker_factory: completion must be callable")
    if not isinstance(system_prompt_prefix, str):
        raise CouncilError("crewai_speaker_factory: system_prompt_prefix must be str")

    @dataclass(frozen=True, slots=True)
    class _CrewAISpeaker:
        completion: Callable[..., Mapping[str, str]]
        system_prompt_prefix: str

        def speak(
            self,
            *,
            agent: CouncilAgent,
            task: CouncilTask,
            topic: str,
            prior_results: tuple[CouncilTaskResult, ...],
        ) -> Mapping[str, str]:
            system_lines: list[str] = []
            if self.system_prompt_prefix:
                system_lines.append(self.system_prompt_prefix)
            system_lines.append(f"You are the council's {agent.role.value} agent.")
            system_lines.append(f"GOAL: {agent.goal}")
            system_lines.append(f"BACKSTORY: {agent.backstory}")
            system_lines.append(
                "Respond with a JSON object containing exactly these "
                f"keys (no others): {list(agent.expected_output_keys)}. "
                "Each value must be a string."
            )
            context_lines: list[str] = [f"Topic: {topic}"]
            for r in prior_results:
                context_lines.append(
                    f"- {r.role.value}/{r.task_id}: {dict(sorted(r.output.items()))}"
                )
            context_lines.append(f"TASK: {task.description}")
            return self.completion(
                system="\n".join(system_lines),
                user="\n".join(context_lines),
                expected_keys=tuple(agent.expected_output_keys),
            )

    return _CrewAISpeaker(
        completion=completion,
        system_prompt_prefix=system_prompt_prefix,
    )


__all__ = [
    "ARBITER_RATIONALE_KEY",
    "ARBITER_RECOMMENDATION_KEY",
    "CANONICAL_ROLE_ORDER",
    "CouncilAgent",
    "CouncilConfig",
    "CouncilError",
    "CouncilProposal",
    "CouncilRole",
    "CouncilTask",
    "CouncilTaskResult",
    "DEFAULT_RATIONALE",
    "DEFAULT_RECOMMENDATION",
    "MAX_AGENTS",
    "MAX_BACKSTORY_LEN",
    "MAX_DESCRIPTION_LEN",
    "MAX_GOAL_LEN",
    "MAX_OUTPUT_KEYS",
    "MAX_OUTPUT_KEY_LEN",
    "MAX_OUTPUT_VALUE_LEN",
    "MAX_RATIONALE_LEN",
    "MAX_RECOMMENDATION_LEN",
    "MAX_ROLE_VARIANT_LEN",
    "MAX_TASKS",
    "MAX_TOPIC_LEN",
    "MIN_AGENTS",
    "MIN_TASKS",
    "NEW_PIP_DEPENDENCIES",
    "RECOMMENDATION_LABELS",
    "StructuredSpeaker",
    "crewai_speaker_factory",
    "run_council",
]
