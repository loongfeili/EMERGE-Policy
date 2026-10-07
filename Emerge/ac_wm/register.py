"""Assembly helper for registering AC-WM in Emerge's SubagentRegistry."""
from __future__ import annotations

from typing import Any

from Emerge.subagents.registry import SubagentRegistry

from .subagent import AcWmSubagent


def register_ac_wm(
    registry: SubagentRegistry, *, provider: Any, rollout, judge,
    candidate_provider=None, dispatch=None,
) -> AcWmSubagent:
    agent = AcWmSubagent(
        provider=provider, rollout=rollout, judge=judge,
        candidate_provider=candidate_provider, dispatch=dispatch,
    )
    registry.register(agent)
    return agent
