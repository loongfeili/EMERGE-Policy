"""VLM judges that score AC-WM rollouts."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from .protocol import ActionCandidate, RolloutResult

DEFAULT_CONTROL_DESCRIPTION = "LIBERO action7 rows [xyz translation, xyz rotation, gripper]"
_PROMPT_ROWS = 8

_PREDICTION_RUBRIC = (
    "Evaluate this candidate over the CURRENT planning horizon, not only final task completion. "
    "Use both the predicted key frames and the action controls: when visual displacement is subtle, "
    "judge whether the commanded direction is consistent with the visible gripper-to-target relationship. "
)
_OBSERVATION_RUBRIC = (
    "No world-model prediction is available. The images show the CURRENT scene from the robot cameras "
    "before these controls run. Evaluate whether executing the controls from this scene makes useful "
    "progress over the CURRENT planning horizon: judge the commanded direction and targets against the "
    "visible gripper-to-target relationship and obstacles. "
)
_SCORING_RUBRIC = (
    "A short but clearly goal-directed approach, alignment, grasp preparation, or transport is useful local "
    "progress and should score at least 0.25 even if the endpoint is far from task completion. "
    "Score 0.0 only for static/no useful motion, clearly wrong direction, unsafe motion, or controls "
    "inconsistent with the visible scene. "
    "Score 0.25 for weak but directionally useful local progress; 0.5 for clear progress toward the next subgoal; "
    "0.75 for strong progress such as reliable approach, grasp preparation, grasp, or transport; "
    "1.0 means the visible horizon completes the task. Reward correct direction and safe executable motion even "
    "when the full multi-step task cannot fit in this short rollout. Mention the next subgoal and any risk. "
    "Return JSON only: {\"score\": number 0..1, \"rationale\": string}."
)


def build_judge_prompt(task: str, candidate: ActionCandidate, rollout: RolloutResult) -> str:
    rows = candidate.actions[:min(len(candidate.actions), _PROMPT_ROWS)]
    controls = json.dumps(rows, separators=(",", ":"))
    description = str(candidate.metadata.get("control_description") or DEFAULT_CONTROL_DESCRIPTION)
    predicted = rollout.metadata.get("prediction") != "none"
    return (
        f"Task: {task}\nCandidate: {candidate.candidate_id}\n"
        f"Candidate controls, first up to {_PROMPT_ROWS} of {len(candidate.actions)} {description}: {controls}\n"
        + (_PREDICTION_RUBRIC if predicted else _OBSERVATION_RUBRIC)
        + _SCORING_RUBRIC
    )


def _jpeg_data_url(frame: Any, label: str) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError(f"cannot encode {label}")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode()


def judge_images(rollout: RolloutResult) -> list[tuple[str, str]]:
    """Return ``(label, data_url)`` pairs: observation frames or predicted key frames."""
    try:
        import cv2

        frames = list(rollout.metadata.get("frames") or ())
        if frames:
            images = []
            for path in frames:
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"cannot read observation frame: {path}")
                images.append((Path(path).stem, _jpeg_data_url(frame, str(path))))
            return images
        if not rollout.video_path:
            raise RuntimeError("rollout has neither frames nor a video")
        capture = cv2.VideoCapture(rollout.video_path)
        if not capture.isOpened():
            raise RuntimeError(f"cannot open predicted video: {rollout.video_path}")
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            first_only = rollout.metadata.get("prediction") == "none"
            indices = [0] if count <= 1 or first_only else [0, count - 1]
            images = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"cannot decode predicted frame {index}/{count}")
                label = "first" if index == 0 else "last"
                images.append((label, _jpeg_data_url(frame, f"predicted frame {index}/{count}")))
        finally:
            capture.release()
    except Exception as exc:
        raise RuntimeError(f"failed to extract AC-WM judge frames: {type(exc).__name__}: {exc}") from exc
    if not images:
        raise RuntimeError("predicted rollout has no decodable key frames")
    return images


def judge_content(task: str, candidate: ActionCandidate, rollout: RolloutResult) -> list[dict[str, Any]]:
    images = judge_images(rollout)
    predicted = rollout.metadata.get("prediction") != "none"
    content: list[dict[str, Any]] = [
        {"type": "text", "text": build_judge_prompt(task, candidate, rollout)},
        {"type": "text", "text": "Key frames from the predicted trajectory:" if predicted
         else "Current camera views:"},
    ]
    for label, url in images:
        if not predicted:
            content.append({"type": "text", "text": f"View: {label}"})
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def parse_judge_reply(text: Any) -> tuple[float, str]:
    if not isinstance(text, str):
        text = text[0]["text"]
    start = text.find("{")
    if start < 0:
        raise ValueError(f"judge reply has no JSON object: {text[:200]!r}")
    parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    score = float(parsed["score"])
    if not 0 <= score <= 1:
        raise ValueError("VLM score outside [0,1]")
    return score, str(parsed.get("rationale", ""))


class ProviderVlmJudge:
    """Score rollouts through the agent's own LLM provider.

    Reusing the provider keeps the judge on the same gateway, credentials and
    shared request pacing as the agent and its other sub-agents.
    """

    def __init__(self, provider: Any, model: str | None = None) -> None:
        self.provider = provider
        self.model = model or provider.get_default_model()

    async def __call__(self, task: str, candidate: ActionCandidate, rollout: RolloutResult) -> tuple[float, str]:
        if rollout.status != "success":
            return 0.0, "rollout unavailable"
        content = await asyncio.to_thread(judge_content, task, candidate, rollout)
        response = await self.provider.chat_with_retry(
            messages=[{"role": "user", "content": content}], model=self.model, temperature=0.0,
        )
        if response.finish_reason == "error":
            raise RuntimeError(response.content or "judge model returned an error")
        return parse_judge_reply(response.content or "")


@dataclass(frozen=True, slots=True)
class VlmJudgeConfig:
    base_url: str
    model: str
    api_key_env: str = "EMERGE_VLM_API_KEY"
    timeout_s: float = 120.0
    api_key: str = ""


class OpenAICompatibleVlmJudge:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint directly."""

    def __init__(self, config: VlmJudgeConfig):
        self.config = config

    @classmethod
    def from_emerge_config(cls, config: Any, provider: str = "custom", model: str | None = None):
        settings = getattr(config.providers, provider, None)
        if settings is None:
            raise RuntimeError(f"unknown provider {provider!r}")
        key = settings.api_key
        if match := re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", key.strip()):
            key = os.environ.get(match.group(1), "")
        if not key:
            raise RuntimeError(f"provider {provider!r} has no api_key")
        if not settings.api_base:
            raise RuntimeError(f"provider {provider!r} has no api_base")
        return cls(VlmJudgeConfig(
            base_url=settings.api_base, model=model or config.agents.defaults.model,
            api_key_env="", api_key=key,
        ))

    def __call__(self, task: str, candidate: ActionCandidate, rollout: RolloutResult) -> tuple[float, str]:
        if rollout.status != "success":
            return 0.0, "rollout unavailable"
        key = self.config.api_key_env and os.environ.get(self.config.api_key_env) or self.config.api_key
        if not key:
            raise RuntimeError(f"missing {self.config.api_key_env}")
        body = {"model": self.config.model, "temperature": 0,
                "messages": [{"role": "user", "content": judge_content(task, candidate, rollout)}]}
        req = Request(
            self.config.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "User-Agent": "OpenAI-Python/1.0"},
        )
        with urlopen(req, timeout=self.config.timeout_s) as r:
            data = json.loads(r.read())
        return parse_judge_reply(data["choices"][0]["message"]["content"])
