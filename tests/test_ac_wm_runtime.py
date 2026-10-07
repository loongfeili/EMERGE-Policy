import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from Emerge.ac_wm import (
    AcWmSelector,
    AcWmSubagent,
    ActionCandidate,
    ObservationRollout,
    ProviderVlmJudge,
    RolloutRequest,
    RolloutResult,
)
from Emerge.ac_wm.cosmos_adapter import CosmosFrameworkAdapter, CosmosFrameworkConfig, parse_domain_map
from Emerge.ac_wm.runtime import build_ac_wm_components
from Emerge.subagents.content import TextContent
from Emerge.subagents.models import SubagentTask


class Provider:
    def __init__(self, reply='{"score": 0.6, "rationale": "aligned"}'):
        self.reply = reply
        self.calls = []

    def get_default_model(self):
        return "gpt-6-astra"

    async def chat_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.reply, finish_reason="stop")


def test_selector_prefers_the_highest_scoring_candidate():
    candidates = (ActionCandidate("a", "vla", ((0.0,) * 7,)), ActionCandidate("b", "rule", ((1.0,) * 7,)))
    request = RolloutRequest("place object", "obs.mp4", candidates, "libero")

    def rollout(_request, candidate):
        return RolloutResult(candidate.candidate_id, "success", "vision.mp4")

    def judge(_task, candidate, _rollout):
        return (0.2, "low") if candidate.candidate_id == "a" else (0.8, "high")

    assert AcWmSelector(rollout, judge).select(request).selected_candidate_id == "b"
    agent = AcWmSubagent(provider=Provider(), rollout=rollout, judge=judge)
    result = asyncio.run(agent.run(SubagentTask((TextContent(text="select"),), input={"request": request})))
    assert result.output.selected_candidate_id == "b"


def _config(tmp_path):
    return SimpleNamespace(
        workspace_path=tmp_path,
        subagents=SimpleNamespace(task_verification=SimpleNamespace(model="judge-model")),
        agents=SimpleNamespace(defaults=SimpleNamespace(model="agent-model")),
    )


def test_components_are_disabled_unless_requested(tmp_path):
    assert build_ac_wm_components(_config(tmp_path), Provider(), env={}) == (None, None)


def test_observation_rollout_is_the_default_and_judge_reuses_the_provider(tmp_path):
    provider = Provider()
    rollout, judge = build_ac_wm_components(_config(tmp_path), provider, env={"EMERGE_AC_WM": "1"})
    assert isinstance(rollout, ObservationRollout)
    assert isinstance(judge, ProviderVlmJudge)
    assert judge.provider is provider and judge.model == "judge-model"
    _, judge = build_ac_wm_components(
        _config(tmp_path), provider, env={"EMERGE_AC_WM": "1", "EMERGE_AC_WM_JUDGE_MODEL": "other"})
    assert judge.model == "other"


def test_cosmos_mode_fails_fast_without_its_configuration(tmp_path):
    with pytest.raises(Exception):
        build_ac_wm_components(_config(tmp_path), Provider(),
                               env={"EMERGE_AC_WM": "1", "EMERGE_AC_WM_ROLLOUT": "cosmos"})
    with pytest.raises(ValueError, match="EMERGE_AC_WM_ROLLOUT"):
        build_ac_wm_components(_config(tmp_path), Provider(),
                               env={"EMERGE_AC_WM": "1", "EMERGE_AC_WM_ROLLOUT": "dreams"})


def test_observation_judge_sees_every_camera_and_the_control_description(tmp_path):
    frames = []
    for name in ("cam_head", "cam_left_wrist"):
        path = tmp_path / f"{name}.png"
        cv2.imwrite(str(path), np.full((16, 16, 3), 90, np.uint8))
        frames.append(str(path))
    candidate = ActionCandidate(
        "vla-1", "vla", ((0.1,) * 14,),
        {"observation_images": frames, "control_description": "RoboDojo joint rows"},
    )
    request = RolloutRequest("stack the bowls", str(tmp_path / "missing.mp4"), (candidate,), "robodojo_joint")
    rollout = ObservationRollout()(request, candidate)
    assert rollout.status == "success" and rollout.metadata["prediction"] == "none"

    provider = Provider()
    score, rationale = asyncio.run(ProviderVlmJudge(provider)("stack the bowls", candidate, rollout))
    assert (score, rationale) == (0.6, "aligned")
    content = provider.calls[0]["messages"][0]["content"]
    texts = " ".join(item.get("text", "") for item in content if item["type"] == "text")
    assert "RoboDojo joint rows" in texts and "No world-model prediction" in texts
    assert "View: cam_head" in texts and "View: cam_left_wrist" in texts
    assert sum(item["type"] == "image_url" for item in content) == 2
    assert provider.calls[0]["temperature"] == 0.0


def test_observation_rollout_fails_without_any_observation(tmp_path):
    candidate = ActionCandidate("c", "vla", ((0.0,) * 14,), {"observation_images": [str(tmp_path / "gone.png")]})
    request = RolloutRequest("t", str(tmp_path / "gone.mp4"), (candidate,), "robodojo_joint")
    assert ObservationRollout()(request, candidate).status == "failed"


def test_provider_judge_error_is_not_scored(tmp_path):
    provider = Provider()

    async def failing(**_kwargs):
        return SimpleNamespace(content="HTTP 520", finish_reason="error")

    provider.chat_with_retry = failing
    frame = tmp_path / "cam_head.png"
    cv2.imwrite(str(frame), np.zeros((8, 8, 3), np.uint8))
    rollout = RolloutResult("c", "success", metadata={"prediction": "none", "frames": [str(frame)]})
    with pytest.raises(RuntimeError, match="HTTP 520"):
        asyncio.run(ProviderVlmJudge(provider)("t", ActionCandidate("c", "vla", ((0.0,) * 7,)), rollout))
    assert asyncio.run(ProviderVlmJudge(provider)(
        "t", ActionCandidate("c", "vla", ((0.0,) * 7,)), RolloutResult("c", "failed"))) == (0.0, "rollout unavailable")


@pytest.mark.parametrize("domain,width", [("robodojo_joint", 14), ("robodojo_ee", 16)])
def test_cosmos_input_spec_maps_robodojo_domains(tmp_path, domain, width):
    config = CosmosFrameworkConfig(
        python=str(tmp_path / "python"), checkpoint="ckpt", action_horizon=4,
        domain_names=parse_domain_map(f"{domain}=dual_arm_{width}"),
    )
    candidate = ActionCandidate("c", "vla", ((0.5,) * width, (0.25,) * width))
    request = RolloutRequest("t", str(tmp_path / "obs.mp4"), (candidate,), domain)
    spec_path = Path(CosmosFrameworkAdapter(config).write_input_spec(request, candidate, str(tmp_path / "spec.json")))
    spec = json.loads(spec_path.read_text())
    assert spec["domain_name"] == f"dual_arm_{width}"
    actions = json.loads(spec_path.with_suffix(".actions.json").read_text())
    assert len(actions) == 4 and actions[-1] == [0.25] * width

    wrong = ActionCandidate("w", "vla", ((0.0,) * (width + 1),))
    with pytest.raises(ValueError):
        CosmosFrameworkAdapter(config).write_input_spec(
            RolloutRequest("t", "obs.mp4", (wrong,), domain), wrong, str(tmp_path / "wrong.json"))
