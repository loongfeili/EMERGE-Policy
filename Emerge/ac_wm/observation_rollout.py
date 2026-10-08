"""World-model-free rollout: judge a proposal against the current observation.

Used when no action-conditioned world model is available for an embodiment.
The judge then sees the scene the controls would start from, with the
proposal's planned gripper path drawn on it when the embodiment provides one,
instead of a predicted future. Its score is a plausibility check, not a
prediction.
"""
from __future__ import annotations

from pathlib import Path

from .protocol import ActionCandidate, RolloutRequest, RolloutResult


def _existing(paths) -> list[str]:
    return [str(path) for path in paths or () if Path(path).is_file()]


class ObservationRollout:
    def __call__(self, request: RolloutRequest, candidate: ActionCandidate) -> RolloutResult:
        frames = (_existing(candidate.metadata.get("preview_images"))
                  or _existing(candidate.metadata.get("observation_images")))
        if not frames and not Path(request.observation_path).is_file():
            return RolloutResult(candidate.candidate_id, "failed",
                                 error=f"observation is unavailable: {request.observation_path}")
        return RolloutResult(
            candidate.candidate_id, "success", video_path=request.observation_path,
            metadata={"prediction": "none", "frames": frames},
        )
