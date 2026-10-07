"""Opt-in construction of the AC-WM rollout and judge from the environment.

``EMERGE_AC_WM=1`` enables mediation. ``EMERGE_AC_WM_ROLLOUT`` selects the
rollout: ``observation`` (default; no world model, the judge sees the current
cameras) or ``cosmos`` (Cosmos3 forward dynamics, configured by ``COSMOS_*``).
``EMERGE_AC_WM_JUDGE`` selects ``provider`` (the agent's own LLM provider) or
``openai`` (direct OpenAI-compatible call using ``EMERGE_AC_WM_JUDGE_PROVIDER``).
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

ROLLOUT_MODES = ("cosmos", "observation")
JUDGE_BACKENDS = ("provider", "openai")


def ac_wm_enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return env.get("EMERGE_AC_WM", "").strip().lower() in {"1", "true", "yes", "on"}


def build_ac_wm_components(config: Any, provider: Any, env: Mapping[str, str] | None = None) -> tuple[Any, Any]:
    """Return ``(rollout, judge)``, or ``(None, None)`` when AC-WM is disabled."""
    env = os.environ if env is None else env
    if not ac_wm_enabled(env):
        return None, None

    mode = env.get("EMERGE_AC_WM_ROLLOUT", "").strip().lower() or "observation"
    if mode == "cosmos":
        from .cosmos_adapter import CosmosFrameworkAdapter, config_from_env

        adapter = CosmosFrameworkAdapter(config_from_env(env))
        default_dir = str(config.workspace_path / "artifacts" / "ac-wm")

        def rollout(request, candidate):
            return adapter.run(request, candidate, request.output_dir or default_dir)
    elif mode == "observation":
        from .observation_rollout import ObservationRollout

        rollout = ObservationRollout()
    else:
        raise ValueError(f"EMERGE_AC_WM_ROLLOUT must be one of {ROLLOUT_MODES}, got {mode!r}")

    backend = env.get("EMERGE_AC_WM_JUDGE", "provider").strip().lower()
    model = env.get("EMERGE_AC_WM_JUDGE_MODEL", "").strip() or None
    if backend == "provider":
        from .vlm_judge import ProviderVlmJudge

        judge = ProviderVlmJudge(
            provider,
            model=model or config.subagents.task_verification.model or config.agents.defaults.model,
        )
    elif backend == "openai":
        from .vlm_judge import OpenAICompatibleVlmJudge

        judge = OpenAICompatibleVlmJudge.from_emerge_config(
            config, provider=env.get("EMERGE_AC_WM_JUDGE_PROVIDER", "custom").strip(), model=model,
        )
    else:
        raise ValueError(f"EMERGE_AC_WM_JUDGE must be one of {JUDGE_BACKENDS}, got {backend!r}")
    return rollout, judge
