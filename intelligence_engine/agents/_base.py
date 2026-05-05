"""Abstract base for AGT-XX agents (INV-54, B19).

Provides:

* The bounded ring buffer that every concrete agent uses to back
  :meth:`recent_decisions` (so the implementation is O(1) per call
  and bounded in memory).
* A helper that re-validates a ``state_snapshot()`` return against
  ``registry/agent_state_keys.yaml`` (purely a runtime safety net
  paired with the offline B19 lint rule).

The base does **not** implement :meth:`AgentIntrospection.state_snapshot`
itself — that is the agent's responsibility, since each agent's
state schema differs.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import yaml

from core.contracts.agent import AgentDecisionTrace


class AgentBase:
    """Common machinery for INV-54-conforming agents.

    Subclasses declare:

    * ``agent_id`` (class or instance attribute) — string identifier.
    * ``state_snapshot() -> Mapping[str, str]`` — pure introspection.

    The base owns:

    * ``_decision_buffer`` — a :class:`collections.deque` with
      ``maxlen=ring_capacity`` carrying :class:`AgentDecisionTrace`
      records.
    * ``recent_decisions(n)`` — O(1) read returning the newest ``≤
      n`` traces in oldest-to-newest order.
    """

    DEFAULT_RING_CAPACITY: int = 64

    def __init__(self, agent_id: str, ring_capacity: int | None = None) -> None:
        if not agent_id:
            raise ValueError("agent_id must be non-empty")
        cap = ring_capacity if ring_capacity is not None else self.DEFAULT_RING_CAPACITY
        if cap <= 0:
            raise ValueError("ring_capacity must be > 0")
        self._agent_id = agent_id
        self._decision_buffer: deque[AgentDecisionTrace] = deque(maxlen=cap)

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def ring_capacity(self) -> int:
        cap = self._decision_buffer.maxlen
        # ``deque(maxlen=...)`` always preserves the bound here (set
        # in __init__), so this is non-None by construction. The
        # explicit fallback exists to satisfy strict type checkers
        # without an ignore comment.
        return cap if cap is not None else self.DEFAULT_RING_CAPACITY

    def _record_decision(self, trace: AgentDecisionTrace) -> None:
        self._decision_buffer.append(trace)

    def recent_decisions(self, n: int) -> Sequence[AgentDecisionTrace]:
        if n <= 0:
            return ()
        if n >= len(self._decision_buffer):
            return tuple(self._decision_buffer)
        # Return the last ``n`` items in oldest-to-newest order.
        # ``deque`` slicing requires conversion; the slice copy is
        # O(n) on the slice itself, not on the full buffer, so the
        # call remains O(n) bounded by ``n`` (not by total trades).
        items = tuple(self._decision_buffer)
        return items[-n:]

    # --- INV-54 helper: state-key allowlist enforcement -----------

    @staticmethod
    def _load_allowed_state_keys(
        agent_id: str, registry_path: str | Path | None = None
    ) -> frozenset[str]:
        """Return the allowlisted state-snapshot keys for an agent.

        Reads ``registry/agent_state_keys.yaml`` and looks up the
        per-agent entry. Returns an empty frozenset (interpreted as
        "no validation configured") when the agent is not present.
        """

        path = (
            Path(registry_path)
            if registry_path is not None
            else Path(__file__).resolve().parents[2]
            / "registry"
            / "agent_state_keys.yaml"
        )
        with path.open(encoding="utf-8") as handle:
            doc = yaml.safe_load(handle) or {}
        keys = doc.get("keys", {}).get(agent_id)
        if not keys:
            return frozenset()
        return frozenset(str(k) for k in keys)

    def _assert_state_keys_allowed(
        self, snapshot: Mapping[str, str], allowlist: Iterable[str]
    ) -> None:
        allow = frozenset(allowlist)
        if not allow:
            return
        bad = [k for k in snapshot if k not in allow]
        if bad:
            raise ValueError(
                f"{self._agent_id}: state_snapshot keys {bad!r} are not in "
                "registry/agent_state_keys.yaml"
            )


__all__ = ["AgentBase"]
