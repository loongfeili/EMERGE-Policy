"""World-model-free rollout: judge a proposal against the current observation.

Used when no action-conditioned world model is available for an embodiment.
The judge then sees the scene the controls would start from instead of a
predicted future, so its score is a plausibility check, not a prediction.
"""
from __future__ import annotations

from pathlib import Path

from .protocol import ActionCandidate, RolloutRequest, RolloutResult


class ObservationRollout:
    def __call__(self, request: RolloutRequest, candidate: ActionCandidate) -> RolloutResult:
        frames = [str(path) for path in candidate.metadata.get("observation_images") or ()
                  if Path(path).is_file()]
        if not frames and not Path(request.observation_path).is_file():
            return RolloutResult(candidate.candidate_id, "failed",
                                 error=f"observation is unavailable: {request.observation_path}")
        return RolloutResult(
            candidate.candidate_id, "success", video_path=request.observation_path,
            metadata={"prediction": "none", "frames": frames},
        )
