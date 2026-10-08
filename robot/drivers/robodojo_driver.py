"""Emerge driver facade for an already-created RoboDojo EvalEnv.

Isaac Sim must be launched before importing most RoboDojo modules.  Environment
creation therefore belongs to ``scripts/run_robodojo_agent_worker.py``; this
driver only wraps that live object and implements Emerge's uniform HAL
contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from robot.drivers.base_driver import BaseDriver, CancelCheck, SceneCatalog
from robot.robodojo_simulation.action_controller import RoboDojoActionController
from robot.robodojo_simulation.tensors import to_float_array

_NON_STEPPING_ACTIONS = frozenset({"vla_propose", "rule_propose"})


class RoboDojoDriver(BaseDriver):
    """Expose RoboDojo rule skills and Pi0.5 through one action dispatcher."""

    def __init__(
        self,
        environment: Any,
        *,
        motion: Mapping[str, Any] | None = None,
        policy_client: Any | None = None,
        workspace: str | Path | None = None,
        archive_observations: bool = False,
        close_environment: bool = False,
    ) -> None:
        if environment is None:
            raise ValueError(
                "RoboDojoDriver requires a live environment; launch it through "
                "scripts/run_robodojo_agent_worker.py"
            )
        self._environment = environment
        self._policy_client = policy_client
        self._infrastructure_error = None
        self._observations = None
        if workspace is not None:
            from robot.robodojo_simulation.observation_publisher import (
                RoboDojoObservationPublisher,
            )

            self._observations = RoboDojoObservationPublisher(
                environment,
                workspace=workspace,
                archive=archive_observations,
            )
        self._actions = RoboDojoActionController(
            environment,
            motion,
            policy_client=policy_client,
            observation_publisher=self._observations,
            stall_probe=self._stall_probe,
        )
        if self._observations is not None:
            self._observations.camera_pose = self._actions.live_camera_pose
        self._close_environment = bool(close_environment)
        self._connected = True

    def get_profile_path(self) -> Path:
        profiles = Path(__file__).resolve().parents[1] / "profiles"
        return profiles / "robodojo.md"

    def load_environment(self) -> None:
        # The canonical RoboDojo task/layout has already been constructed by
        # the Isaac worker. ROBOT_STATE.md is a view of it, not its source.
        # Give the agent a view of the scene before it plans its first action.
        self._publish_observation()

    def reset_environment(self) -> None:
        raise RuntimeError(
            "RoboDojo layouts are reset by the evaluation worker, which owns the "
            "Isaac environment; the driver cannot reset them"
        )

    def get_scene_catalog(self) -> SceneCatalog:
        # The evaluation worker fixes the task and layout for each episode.
        task = str(getattr(self._environment, "task_name", "")) or None
        return {"root": "RoboDojo", "current": task if self.is_connected() else None, "entries": []}

    def switch_scene(self, scene_id: str) -> None:
        raise ValueError(
            f"RoboDojo scene {scene_id!r} cannot be selected here; tasks and "
            "layouts are chosen by the evaluation worker"
        )

    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        if not self.is_connected():
            return "Failed: RoboDojo environment is not connected"
        try:
            result = self._actions.execute(
                action_type, params, cancel_check=cancel_check
            )
        finally:
            # Rule actions never fetch an observation of their own, so without
            # this the agent would verify the scene as it was before the action.
            # AC-WM proposals do not step the scene, so there is nothing new.
            if action_type not in _NON_STEPPING_ACTIONS:
                self._publish_observation()
        return result

    def _publish_observation(self) -> None:
        if self._observations is None:
            return
        try:
            manifest = self._observations.publish()
            if manifest is None:
                raise RuntimeError("RoboDojo camera observation is unavailable")
        except Exception as exc:
            self._infrastructure_error = f"Camera observation: {exc}"
            raise

    def raise_if_infrastructure_error(self) -> None:
        error = self._infrastructure_error or getattr(self._policy_client, "last_error", None)
        if error:
            raise RuntimeError(f"Infrastructure failure: {error}")

    def get_scene(self) -> dict[str, dict]:
        return {}

    def connect(self) -> bool:
        self._connected = self._environment is not None
        return self._connected

    def is_connected(self) -> bool:
        return self._connected and getattr(self._environment, "sim", None) is not None

    def health_check(self) -> bool:
        return self.is_connected()

    def is_terminal(self) -> bool:
        flags = getattr(self._environment, "end_flag", None)
        return bool(flags and flags[0])

    def request_stop(self) -> None:
        """End a non-terminal agent episode as an official failed attempt."""

        if self.is_terminal():
            return
        self._environment.success[0] = False
        self._environment.end_flag[0] = True
        get_obs_batch = getattr(self._environment, "get_obs_batch", None)
        if callable(get_obs_batch):
            get_obs_batch(env_idx_list=[0], last_frame=True)

    def get_runtime_state(self) -> dict[str, Any]:
        environment = self._environment
        robot_manager = environment.robot_manager
        arms: dict[str, Any] = {}
        for arm in ("left", "right"):
            robot = robot_manager.get_robot_by_arm_name(f"{arm}_arm")
            pose_wxyz = to_float_array(
                robot_manager.get_real_endpose(
                    robot, env_idx_list=[0], is_relative=True
                )[0]
            )
            joints = to_float_array(
                robot_manager.get_joint(robot, env_idx_list=[0])[0]
            )
            gripper = self._actions.current_gripper(arm)
            arms[arm] = {
                "joint_position": joints.tolist(),
                "end_effector_pose": {
                    "frame": "robodojo_env",
                    "position_m": pose_wxyz[:3].tolist(),
                    "orientation_quat_xyzw": [
                        float(pose_wxyz[4]),
                        float(pose_wxyz[5]),
                        float(pose_wxyz[6]),
                        float(pose_wxyz[3]),
                    ],
                },
                "gripper_opening_normalized": float(gripper),
            }

        success_flags = getattr(environment, "success", [False])
        done = self.is_terminal()
        robot_state = {
            "connected": self.is_connected(),
            "task": str(getattr(environment, "task_name", "")),
            "instruction": self._instruction(),
            "layout_id": self._layout_id(),
            "step": int(getattr(environment, "take_action_cnt", [0])[0]),
            "step_limit": int(getattr(environment, "step_lim", 0)),
            "done": done,
            "success": bool(done and success_flags[0]),
            "arms": arms,
            "capabilities": {
                "actions": [
                    "move_to_pose",
                    "move_linear",
                    "set_gripper",
                    "follow_arc",
                    "vla_execute",
                    "recover",
                ],
                "arms": ["left", "right", "both"],
                "pose_frame": "robodojo_env",
                "position_unit": "m",
                "quaternion_order": "xyzw",
            },
        }
        # Object poses are deliberately absent. The benchmark hands a policy
        # camera images, proprioception and the instruction; object ground truth
        # lives in the scene manager, which is also where the grader reads from.
        # Putting it in front of the agent would make the run unsubmittable, so
        # the agent locates objects the same way any real system would -- through
        # the published camera views and the vision sub-agents.
        return {
            "objects": {},
            "robots": {"robodojo": robot_state},
            "scene_graph": {},
        }

    def close(self) -> None:
        self._connected = False
        if self._close_environment and self._environment is not None:
            self._environment.close()

    def _stall_probe(self) -> list[float]:
        """A pose fingerprint of both end-effectors.

        Proprioception only. End-effector pose is part of the observation the
        benchmark hands a policy, so watching it costs the run nothing in
        legitimacy, while object poses and the graded score -- which an earlier
        version of this probe read -- are privileged simulator state.

        The trade is that a policy waving an arm without achieving anything no
        longer counts as stalled. Only a genuinely frozen arm does, which is the
        case worth spending the rest of an episode's budget on.
        """
        fingerprint: list[float] = []
        robot_manager = getattr(self._environment, "robot_manager", None)
        if robot_manager is None:
            return fingerprint
        for arm in ("left", "right"):
            try:
                robot = robot_manager.get_robot_by_arm_name(f"{arm}_arm")
                pose = to_float_array(
                    robot_manager.get_real_endpose(
                        robot, env_idx_list=[0], is_relative=True
                    )[0]
                )
                fingerprint.extend(float(value) for value in pose[:3])
            except Exception:
                continue
        return fingerprint

    def _instruction(self) -> str:
        manager = getattr(self._environment, "obs_manager", None)
        instructions = getattr(manager, "instruction", None)
        if instructions:
            return str(instructions[0])
        return ""

    def _layout_id(self) -> int | None:
        seeds = getattr(self._environment, "env_seeds", None)
        if seeds:
            return int(seeds[0])
        return None
