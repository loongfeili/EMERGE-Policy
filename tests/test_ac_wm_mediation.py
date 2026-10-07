import asyncio

from Emerge.ac_wm import ActionCandidate, RolloutRequest, RolloutResult
from Emerge.ac_wm.subagent import AcWmSubagent
from Emerge.subagents.content import TextContent
from Emerge.subagents.models import SubagentTask


class Provider:
    def get_default_model(self):
        return "gpt-6-astra"


def test_ac_wm_mediates_skill_proposal_and_executes_selected_prefix():
    proposed = ((0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
                (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0),
                (0.3, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0))
    dispatched = []

    async def provide(_task):
        return RolloutRequest(
            task="approach the can",
            observation_path="/workspace/current.mp4",
            candidates=(ActionCandidate("live-vla", "vla", proposed,
                                        {"execute_steps": 2}),),
            domain_name="libero",
        )

    async def dispatch(_task, selected):
        dispatched.extend(selected.actions[:selected.metadata["execute_steps"]])
        return "Selected candidate live-vla executed its exact proposed prefix: steps=2."

    agent = AcWmSubagent(
        provider=Provider(),
        rollout=lambda _request, candidate: RolloutResult(candidate.candidate_id, "success", "/workspace/predicted.mp4"),
        judge=lambda _task, _candidate, _rollout: (0.75, "moves toward the target"),
        candidate_provider=provide,
        dispatch=dispatch,
    )
    result = asyncio.run(agent.run(SubagentTask(
        content=(TextContent("approach the can"),),
        input={"instruction": "approach the can", "step": 20},
    )))
    assert result.status.value == "success"
    assert result.output["selected_candidate_id"] == "live-vla"
    assert result.output["score"] == 0.75
    assert dispatched == list(proposed[:2])


def test_ac_wm_zero_score_does_not_dispatch():
    dispatched = []
    async def provide(_task):
        return RolloutRequest("approach", "obs.mp4",
                              (ActionCandidate("candidate", "vla", ((0.1,) * 7,)),),
                              "libero")
    async def dispatch(_task, selected):
        dispatched.append(selected.candidate_id)
        return "unexpected"
    agent = AcWmSubagent(
        provider=Provider(), rollout=lambda _r, c: RolloutResult(c.candidate_id, "success", "future.mp4"),
        judge=lambda _t, _c, _r: (0.0, "no local progress"),
        candidate_provider=provide, dispatch=dispatch,
    )
    result = asyncio.run(agent.run(SubagentTask(content=(TextContent("approach"),), input={"instruction": "approach"})))
    assert result.status.value == "failed"
    assert result.metadata["dispatched"] is False
    assert dispatched == []


def test_async_judge_runs_on_the_agent_loop():
    loops = []

    async def judge(_task, _candidate, _rollout):
        loops.append(asyncio.get_running_loop())
        return 0.5, "clear progress"

    async def provide(_task):
        return RolloutRequest("approach", "obs.mp4",
                              (ActionCandidate("candidate", "vla", ((0.1,) * 7,)),), "libero")

    async def dispatch(_task, _selected):
        return "Selected candidate candidate executed its exact proposed prefix: steps=1, reason=chunk_completed."

    async def main():
        agent = AcWmSubagent(
            provider=Provider(), rollout=lambda _r, c: RolloutResult(c.candidate_id, "success", "future.mp4"),
            judge=judge, candidate_provider=provide, dispatch=dispatch,
        )
        result = await agent.run(SubagentTask(content=(TextContent("approach"),), input={"instruction": "approach"}))
        return result, asyncio.get_running_loop()

    result, loop = asyncio.run(main())
    assert result.status.value == "success"
    assert loops == [loop]


def test_vla_proposal_is_non_executing_and_controller_runs_selected_rows_verbatim():
    import numpy as np
    from robot.mujoco_simulation.mujoco_actions import MujocoActionController
    from robot.vla.mujoco_policy_executor import VLAExecutor

    proposal = np.asarray([[0.2, 0, 0, 0, 0, 0, -1],
                           [0.4, 0, 0, 0, 0, 0, -1]], dtype=np.float32)

    class Env:
        latest_obs = {}
        def __init__(self): self.steps = []
        def step(self, action, **_kwargs):
            self.steps.append(np.asarray(action).copy())
            return {}, 0.0, False, {}
        def check_success(self): return False
        def refresh_cameras(self): pass

    class Client:
        def health_check(self): return True
        def infer(self, _element): return {"actions": proposal}

    env = Env()
    executor = VLAExecutor(env, client=Client())
    executor._build_element = lambda _instruction: {}
    returned = executor.propose("approach target", horizon=2)
    assert np.array_equal(returned, proposal)
    assert env.steps == []

    controller = MujocoActionController(env)
    result = controller.execute("execute_action_chunk", {
        "actions": returned.tolist(), "candidate_id": "selected-vla", "skill_name": "vla"
    })
    assert "exact proposed prefix: steps=2" in result
    assert len(env.steps) == 2
    assert all(np.array_equal(actual, expected) for actual, expected in zip(env.steps, proposal))


def test_vlm_judge_sends_decoded_frames_and_action_context(monkeypatch):
    import cv2
    import json
    import numpy as np
    import Emerge.ac_wm.vlm_judge as module
    from Emerge.ac_wm.protocol import ActionCandidate, RolloutResult
    from Emerge.ac_wm.vlm_judge import OpenAICompatibleVlmJudge, VlmJudgeConfig

    class Capture:
        def __init__(self): self.index = 0; self.reads = []
        def isOpened(self): return True
        def get(self, _property): return 2
        def set(self, _property, value): self.index = int(value); return True
        def read(self):
            self.reads.append(self.index)
            return True, np.full((8, 8, 3), self.index * 80, dtype=np.uint8)
        def release(self): pass

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): pass
        def read(self):
            return json.dumps({"choices": [{"message": {"content": "{\"score\":0.25,\"rationale\":\"directed approach\"}"}}]}).encode()
    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()
    monkeypatch.setattr(module, "urlopen", fake_urlopen)

    candidate = ActionCandidate("approach", "vla", ((0.0, 0.0, -0.7, 0.0, 0.0, 0.0, -1.0),))
    judge = OpenAICompatibleVlmJudge(VlmJudgeConfig(
        base_url="https://example.test/v1", model="gpt-6-astra", api_key_env="", api_key="test-key"
    ))
    score, rationale = judge("approach and grasp the can", candidate,
                             RolloutResult("approach", "success", "/tmp/prediction.mp4"))

    assert score == 0.25
    assert rationale == "directed approach"
    assert capture.reads == [0, 1]
    content = captured["body"]["messages"][0]["content"]
    image_items = [item for item in content if item["type"] == "image_url"]
    assert len(image_items) == 2
    assert all(len(item["image_url"]["url"].split(",", 1)[1]) > 20 for item in image_items)
    prompt = " ".join(item.get("text", "") for item in content if item["type"] == "text")
    assert "action7" in prompt and "-0.7" in prompt


def test_ac_wm_is_internal_to_main_agent_delegate_registry():
    from Emerge.agent.tools.delegate import DelegateSubagentTool
    from Emerge.subagents.registry import SubagentRegistry

    registry = SubagentRegistry()
    agent = AcWmSubagent(
        provider=Provider(),
        rollout=lambda _request, candidate: RolloutResult(candidate.candidate_id, "failed"),
        judge=lambda _task, _candidate, _rollout: (0.0, ""),
    )
    registry.register(agent)
    tool = DelegateSubagentTool(registry, hidden_agents=("ac-wm",))
    visible_names = tool.parameters["properties"]["agent_name"]["enum"]

    assert "ac-wm" in registry.names
    assert registry.get("ac-wm") is agent
    assert "ac-wm" not in visible_names
    assert "ac-wm:" not in tool.description


def test_rule_skill_preview_is_non_executing_and_returns_action7(tmp_path):
    import json
    import numpy as np
    from robot.mujoco_simulation.mujoco_actions import MujocoActionController

    class Env:
        workspace = tmp_path
        cameras = {}
        def __init__(self):
            self.steps = []
        def get_eef_pose(self):
            return np.zeros(3, dtype=np.float32), np.asarray([0, 0, 0, 1], dtype=np.float32)
        def current_gripper_signal(self):
            return -1.0
        def step(self, action, **_kwargs):
            self.steps.append(np.asarray(action).copy())

    env = Env()
    controller = MujocoActionController(env)
    controller._write_ac_wm_observation = lambda: (tmp_path / "observation.mp4", 7)
    result = controller.execute("rule_propose", {
        "skill_action_type": "move_linear",
        "parameters": {"delta_m": [0.12, 0.0, 0.0], "steps": 3, "settle_steps": 0},
    })
    assert result.startswith("RULE_PROPOSAL:")
    proposal = json.loads(result.split(":", 1)[1])
    actions = np.asarray(proposal["actions"])
    assert actions.shape == (3, 7)
    assert np.isfinite(actions).all()
    assert proposal["execute_steps"] == 3
    assert proposal["observation_revision"] == 7
    assert proposal["domain_name"] == "libero"
    assert env.steps == []


def test_all_robot_motion_skills_route_through_internal_ac_wm(tmp_path):
    from types import SimpleNamespace
    from Emerge.agent.tools.embodied import EmbodiedActionTool

    class Status:
        value = "success"

    class FakeAcWm:
        def __init__(self):
            self.tasks = []
        async def run(self, task):
            self.tasks.append(task)
            return SimpleNamespace(status=Status(), summary="selected and dispatched",
                                   output={"selected_candidate_id": "c"}, error=None, metadata={})

    mediated = FakeAcWm()
    tool = EmbodiedActionTool(workspace=tmp_path, ac_wm_subagent=mediated)
    cases = [
        ("vla_execute", {"instruction": "approach target", "step": 20}),
        ("move_to_pose", {"position_m": [0.3, 0.0, 0.4], "orientation_quat": [0, 0, 0, 1]}),
        ("move_linear", {"delta_m": [0.05, 0.0, 0.0], "steps": 10}),
        ("set_gripper", {"command": "open"}),
        ("follow_arc", {"center": [0.3, 0.0, 0.4], "axis": [0, 0, 1], "radius_m": 0.1, "angle_deg": 30}),
    ]
    for action_type, parameters in cases:
        result = asyncio.run(tool.execute(action_type, parameters, "current local subgoal"))
        assert '"agent": "ac-wm"' in result
    assert [task.input["action_type"] for task in mediated.tasks] == [case[0] for case in cases]
    assert all(task.input["parameters"] == case[1] for task, case in zip(mediated.tasks, cases))
    # Mediation must not narrow the free-form parameters object the agent fills.
    schema = tool.parameters["properties"]["parameters"]
    assert schema["type"] == "object" and "properties" not in schema


def test_internal_ac_wm_actions_are_not_agent_callable(tmp_path):
    from Emerge.agent.tools.embodied import EmbodiedActionTool

    tool = EmbodiedActionTool(workspace=tmp_path)
    (tmp_path / "EMBODIED.md").write_text("robot")
    for action_type in ("vla_propose", "rule_propose", "execute_action_chunk", "ac_wm_select"):
        result = asyncio.run(tool.execute(action_type, {"actions": [[0.0] * 7]}, "bypass"))
        assert result.startswith("Error: internal AC-WM"), result
    assert not (tmp_path / "ACTION.md").exists() or (tmp_path / "ACTION.md").read_text() == ""


def test_each_rule_skill_builds_a_non_executing_proposal(tmp_path):
    import json
    import numpy as np
    from robot.mujoco_simulation.mujoco_actions import MujocoActionController

    class Env:
        workspace = tmp_path
        cameras = {}
        def __init__(self): self.steps = []
        def get_eef_pose(self):
            return np.asarray([0.2, 0.0, 0.0]), np.asarray([0, 0, 0, 1])
        def current_gripper_signal(self): return -1.0
        def step(self, action, **_kwargs): self.steps.append(action)

    env = Env()
    controller = MujocoActionController(env)
    controller._write_ac_wm_observation = lambda: (tmp_path / "observation.mp4", 1)
    cases = [
        ("move_to_pose", {"position_m": [0.3, 0.0, 0.0], "orientation_quat": [0, 0, 0, 1], "steps": 4}),
        ("move_linear", {"delta_m": [0.1, 0.0, 0.0], "steps": 4, "settle_steps": 0}),
        ("set_gripper", {"command": "close", "steps": 4}),
        ("follow_arc", {"center": [0, 0, 0], "axis": [0, 0, 1], "radius_m": 0.2, "angle_deg": 45, "steps": 4}),
    ]
    for action_type, parameters in cases:
        result = controller.execute("rule_propose", {
            "skill_action_type": action_type, "parameters": parameters,
        })
        assert result.startswith("RULE_PROPOSAL:"), (action_type, result)
        proposal = json.loads(result.split(":", 1)[1])
        actions = np.asarray(proposal["actions"])
        assert actions.shape == (4, 7), action_type
        assert np.isfinite(actions).all(), action_type
    assert env.steps == []


def test_rule_skill_candidate_flows_through_ac_wm_to_exact_dispatch(tmp_path):
    import json
    from Emerge.agent.tools.embodied import EmbodiedActionTool

    proposal_actions = [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]]
    dispatched = []
    tool = EmbodiedActionTool(workspace=tmp_path)

    async def fake_dispatch(action_type, parameters):
        if action_type == "rule_propose":
            return "RULE_PROPOSAL:" + json.dumps({
                "actions": proposal_actions, "execute_steps": 2,
                "observation_path": str(tmp_path / "obs.mp4"), "observation_revision": 3,
            })
        if action_type == "execute_action_chunk":
            dispatched.append(parameters)
            return "Selected candidate move_linear executed its exact proposed prefix: steps=2."
        raise AssertionError(action_type)

    tool._dispatch_action = fake_dispatch
    agent = AcWmSubagent(
        provider=Provider(),
        rollout=lambda request, candidate: RolloutResult(candidate.candidate_id, "success", "future.mp4"),
        judge=lambda _task, _candidate, _rollout: (0.5, "moves toward the subgoal"),
        candidate_provider=tool.propose_action_candidates,
        dispatch=tool.dispatch_selected_candidate,
    )
    task = SubagentTask(content=(TextContent("move toward the target"),), input={
        "instruction": "move toward the target", "step": 10, "action_type": "move_linear",
        "parameters": {"delta_m": [0.1, 0, 0], "steps": 2}, "output_dir": str(tmp_path),
    })
    result = asyncio.run(agent.run(task))
    assert result.status.value == "success"
    assert result.output["skill_name"] == "rule:move_linear"
    assert result.metadata["dispatched"] is True
    assert dispatched[0]["actions"] == proposal_actions
    assert dispatched[0]["skill_name"] == "rule:move_linear"
