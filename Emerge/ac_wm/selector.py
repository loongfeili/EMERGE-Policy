"""Deterministic selection layer kept independent from model backends."""
from __future__ import annotations

from collections.abc import Callable

from .protocol import (
    ActionCandidate,
    CandidateEvaluation,
    RolloutRequest,
    RolloutResult,
    SelectionResult,
)

RolloutFn = Callable[[RolloutRequest, ActionCandidate], RolloutResult]
JudgeFn = Callable[[str, ActionCandidate, RolloutResult], tuple[float, str]]


class AcWmSelector:
    def __init__(self, rollout: RolloutFn, judge: JudgeFn) -> None:
        self.rollout, self.judge = rollout, judge

    def select(self, request: RolloutRequest) -> SelectionResult:
        evaluations = []
        for candidate in request.candidates:
            try:
                result = self.rollout(request, candidate)
            except Exception as exc:
                result = RolloutResult(candidate.candidate_id, "failed", error=f"rollout error: {exc}")
            if result.status != "success":
                evaluations.append(CandidateEvaluation(
                    candidate.candidate_id, 0.0, result.error or "rollout failed", result,
                ))
                continue
            try:
                score, rationale = self.judge(request.task, candidate, result)
            except Exception as exc:
                score, rationale = 0.0, f"judge error: {type(exc).__name__}: {exc}"
            evaluations.append(CandidateEvaluation(candidate.candidate_id, float(score), rationale, result))
        valid = [e for e in evaluations if e.rollout.status == "success" and e.score > 0]
        if not valid:
            return SelectionResult(
                None, tuple(evaluations), "failed", "no candidate received a positive judge score",
            )
        best = max(valid, key=lambda e: (e.score, e.candidate_id))
        return SelectionResult(best.candidate_id, tuple(evaluations), "success", best.rationale)
