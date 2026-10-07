"""Bridge candidate skills to AC-WM selection and Emerge action dispatch."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .protocol import ActionCandidate, RolloutRequest, SelectionResult
from .selector import AcWmSelector


@dataclass(frozen=True, slots=True)
class SelectedAction:
    candidate_id: str
    skill_name: str
    actions: tuple[tuple[float, ...], ...]
    metadata: dict[str, Any]


class AcWmActionBridge:
    """Select one candidate, then hand it to an execution skill callback.

    The callback is invoked only after a positive VLM score. This prevents a
    failed or unscored prediction from reaching ``execute_robot_action``.
    """

    def __init__(self, selector: AcWmSelector):
        self.selector = selector

    def plan(
        self, *, task: str, observation_path: str,
        candidates: Sequence[ActionCandidate], domain_name: str = "libero",
        output_dir: str = "", view_point: str | None = None,
    ) -> tuple[SelectionResult, SelectedAction | None]:
        request = RolloutRequest(task, observation_path, tuple(candidates), domain_name, view_point, output_dir)
        result = self.selector.select(request)
        if result.selected_candidate_id is None:
            return result, None
        chosen = next(c for c in candidates if c.candidate_id == result.selected_candidate_id)
        return result, SelectedAction(chosen.candidate_id, chosen.skill_name, chosen.actions, dict(chosen.metadata))

    def plan_and_dispatch(
        self, *, task: str, observation_path: str,
        candidates: Sequence[ActionCandidate], dispatch: Callable[[SelectedAction], Any],
        domain_name: str = "libero", output_dir: str = "", view_point: str | None = None,
    ) -> SelectionResult:
        result, selected = self.plan(
            task=task, observation_path=observation_path, candidates=candidates,
            domain_name=domain_name, output_dir=output_dir, view_point=view_point,
        )
        if selected is not None:
            dispatch(selected)
        return result
