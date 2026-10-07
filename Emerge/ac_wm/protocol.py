"""Serializable contracts between skills, the world model, and a VLM judge."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    candidate_id: str
    skill_name: str
    actions: tuple[tuple[float, ...], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.skill_name.strip():
            raise ValueError("candidate_id and skill_name are required")
        if not self.actions:
            raise ValueError("actions cannot be empty")
        width = len(self.actions[0])
        if width == 0 or any(len(row) != width for row in self.actions):
            raise ValueError("actions must be a non-empty rectangular matrix")


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    task: str
    observation_path: str
    candidates: tuple[ActionCandidate, ...]
    domain_name: str
    view_point: str | None = None
    output_dir: str = ""

    def __post_init__(self) -> None:
        if not self.task.strip() or not self.observation_path.strip():
            raise ValueError("task and observation_path are required")
        if not self.candidates:
            raise ValueError("at least one candidate is required")


@dataclass(frozen=True, slots=True)
class RolloutResult:
    candidate_id: str
    status: str
    video_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    score: float
    rationale: str
    rollout: RolloutResult

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_candidate_id: str | None
    evaluations: tuple[CandidateEvaluation, ...]
    status: str
    reason: str
