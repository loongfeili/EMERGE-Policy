"""AC-WM intermediary between the main agent and executable skills."""
from __future__ import annotations

import asyncio
import inspect
from typing import Any, Awaitable, Callable

from Emerge.subagents.base import BaseSubagent
from Emerge.subagents.models import SubagentResult, SubagentTask

from .bridge import SelectedAction
from .protocol import RolloutRequest
from .selector import AcWmSelector, JudgeFn

CandidateProvider = Callable[[SubagentTask], Awaitable[RolloutRequest]]
SelectedActionDispatcher = Callable[[SubagentTask, SelectedAction], Awaitable[str]]


def _is_async_callable(fn: Any) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.iscoroutinefunction(getattr(fn, "__call__", None))


class AcWmSubagent(BaseSubagent):
    """Coordinate a real skill proposal, world-model selection, and execution.

    The main agent supplies a local subgoal. A low-level skill proposes controls,
    AC-WM evaluates those exact controls, and only the selected proposal is sent
    back to the same execution layer.
    """

    def __init__(
        self, *, provider: Any, rollout, judge,
        candidate_provider: CandidateProvider | None = None,
        dispatch: SelectedActionDispatcher | None = None,
        timeout_s: float = 1800.0,
        judge_timeout_s: float = 300.0,
    ):
        super().__init__(
            name="ac-wm",
            description=(
                "Internal planning layer between the main agent and executable skills; "
                "evaluates skill-proposed actions with a world model and a progress judge, "
                "then dispatches the selected action chunk."
            ),
            system_prompt=(
                "Coordinate an executable skill proposal, evaluate local progress, "
                "and dispatch only the exact selected controls."
            ),
            provider=provider,
            capabilities=(
                "skill-mediation", "action-conditioned-prediction", "candidate-selection",
                "vlm-judge", "selected-action-dispatch",
            ),
            input_modalities=("text", "image"),
            max_iterations=1,
        )
        self._rollout = rollout
        self._judge = judge
        self._candidate_provider = candidate_provider
        self._dispatch = dispatch
        self._timeout_s = timeout_s
        self._judge_timeout_s = judge_timeout_s

    def _selector(self, loop: asyncio.AbstractEventLoop) -> AcWmSelector:
        return AcWmSelector(self._rollout, self._thread_judge(loop))

    def _thread_judge(self, loop: asyncio.AbstractEventLoop) -> JudgeFn:
        """Run an async judge on the agent loop that owns its provider client."""
        if not _is_async_callable(self._judge):
            return self._judge
        judge, timeout_s = self._judge, self._judge_timeout_s

        def call(task_text, candidate, rollout):
            future = asyncio.run_coroutine_threadsafe(judge(task_text, candidate, rollout), loop)
            try:
                return future.result(timeout=timeout_s)
            except BaseException:
                future.cancel()
                raise

        return call

    async def _run(self, task: SubagentTask) -> SubagentResult:
        # Retain the structured selector seam for protocol-only regression tests.
        request = task.input.get("request")
        if not isinstance(request, RolloutRequest):
            if self._candidate_provider is None:
                return SubagentResult.failure(task, self.name, "AC-WM has no executable skill candidate provider")
            try:
                request = await self._candidate_provider(task)
            except Exception as exc:
                return SubagentResult.failure(task, self.name, f"skill proposal failed: {type(exc).__name__}: {exc}")
        if not isinstance(request, RolloutRequest):
            return SubagentResult.failure(task, self.name, "candidate provider must return RolloutRequest")

        try:
            selection = await asyncio.wait_for(
                asyncio.to_thread(self._selector(asyncio.get_running_loop()).select, request),
                timeout=task.timeout or self._timeout_s,
            )
        except asyncio.TimeoutError:
            return SubagentResult.failure(task, self.name, "AC-WM rollout/judge timed out")
        except Exception as exc:
            return SubagentResult.failure(task, self.name, f"AC-WM selection failed: {type(exc).__name__}: {exc}")

        evaluations = [
            {"candidate_id": item.candidate_id, "score": item.score,
             "rationale": item.rationale, "rollout_status": item.rollout.status,
             "video_path": item.rollout.video_path}
            for item in selection.evaluations
        ]
        if selection.selected_candidate_id is None:
            return SubagentResult.failure(
                task, self.name, selection.reason,
                metadata={"evaluations": evaluations, "dispatched": False},
            )

        chosen = next(item for item in request.candidates if item.candidate_id == selection.selected_candidate_id)
        selected = SelectedAction(chosen.candidate_id, chosen.skill_name, chosen.actions, dict(chosen.metadata))
        dispatch_result = None
        if self._dispatch is not None:
            try:
                dispatch_result = await self._dispatch(task, selected)
            except Exception as exc:
                return SubagentResult.failure(
                    task, self.name, f"selected action dispatch failed: {type(exc).__name__}: {exc}",
                    metadata={"evaluations": evaluations, "selected_candidate_id": selected.candidate_id,
                              "dispatched": False},
                )
            if dispatch_result.lstrip().lower().startswith(("error:", "failed:", "interrupted:")):
                return SubagentResult.failure(
                    task, self.name, dispatch_result,
                    metadata={"evaluations": evaluations, "selected_candidate_id": selected.candidate_id,
                              "dispatched": False},
                )

        best = next(item for item in selection.evaluations if item.candidate_id == selected.candidate_id)
        if self._dispatch is None and isinstance(task.input.get("request"), RolloutRequest):
            return SubagentResult.success(
                task, self.name, selection.reason, output=selection,
                metadata={"selected_candidate_id": selection.selected_candidate_id,
                          "evaluations": evaluations},
            )
        output = {"selected_candidate_id": selected.candidate_id, "skill_name": selected.skill_name,
                  "score": best.score, "rationale": best.rationale,
                  "rollout_video": best.rollout.video_path, "dispatch_result": dispatch_result}
        return SubagentResult.success(
            task, self.name,
            f"AC-WM selected {selected.candidate_id} at score {best.score:.3f}"
            + (" and dispatched it" if self._dispatch else ""),
            output=output,
            metadata={"evaluations": evaluations, "selected_candidate_id": selected.candidate_id,
                      "dispatched": self._dispatch is not None},
        )
