"""Emerge motion primitives backed by RoboDojo's native controllers.

The controller deliberately depends only on RoboDojo's *runtime protocol*, not
on Isaac Sim modules.  The process that owns the RoboDojo environment passes an
``EvalEnv``-like object exposing ``get_obs()``, ``take_action()`` and
``robot_manager``.  This keeps the module importable and testable in the light
Emerge environment while all Isaac/CuRobo work stays in RoboDojo's process.

RoboDojo represents dual-ARX-X5 end-effector poses as ``xyz + quaternion_wxyz``.
Emerge's public action contract uses ``xyz + quaternion_xyzw``.  Conversion
is kept at this boundary so every skill sees the same quaternion convention.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from robot.mujoco_simulation.pose_utils import PoseUtils
from robot.robodojo_simulation.tensors import to_float_array, to_numpy
from robot.robodojo_simulation.trajectory_preview import (
    ArmPath,
    TrajectoryPreview,
    UrdfChain,
    draw_camera_overlay,
    draw_plan_schematic,
    quaternion_xyzw_matrix,
    remount_camera,
)
from robot.vla.robodojo_policy import ACTION_DIM, encode_observation, unpack_joint_actions

_ARM_NAMES = ("left", "right")


def _pose_matrix(position: Any, quaternion_xyzw: Any) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = quaternion_xyzw_matrix(quaternion_xyzw)
    transform[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
    return transform

# AC-WM proposals are bounded so one world-model rollout covers the whole chunk.
_AC_WM_MAX_ROWS = 64
_JOINT_CONTROL = "robodojo_joint14"
_EE_CONTROL = "robodojo_ee16"
_EE_WIDTH = 16
_CONTROL_DOMAINS = {_JOINT_CONTROL: "robodojo_joint", _EE_CONTROL: "robodojo_ee"}
_CONTROL_DESCRIPTIONS = {
    _JOINT_CONTROL: (
        "RoboDojo dual ARX-X5 absolute joint-position rows (one row per control step, ~0.4 s): "
        "[left 6 joint angles (rad), left gripper opening, right 6 joint angles (rad), "
        "right gripper opening]; gripper opening is normalized, 0 closed and 1 open"
    ),
    _EE_CONTROL: (
        "RoboDojo dual ARX-X5 end-effector waypoint rows in the robodojo_env frame (one row "
        "per control step, ~0.4 s): [left xyz (m), left quaternion xyzw, left gripper opening, "
        "right xyz (m), right quaternion xyzw, right gripper opening]; gripper opening is "
        "normalized, 0 closed and 1 open"
    ),
}


# Declared here rather than reused from ``robot.mujoco_simulation.mujoco_actions``
# so this module stays importable without the MuJoCo runtime.
class _ActionInterrupted(Exception):
    """Raised when the watchdog cancels an action that is still stepping."""


class PolicyClient(Protocol):
    def infer(self, observation: dict[str, Any]) -> dict[str, Any]: ...


class RoboDojoActionController:
    """Execute common Emerge skills in one native RoboDojo environment."""

    def __init__(
        self,
        environment: Any,
        config: Mapping[str, Any] | None = None,
        *,
        policy_client: PolicyClient | None = None,
        observation_publisher: Any | None = None,
        stall_probe: Callable[[], list[float]] | None = None,
    ) -> None:
        self._environment = environment
        self._policy_client = policy_client
        self._observation_publisher = observation_publisher
        self._stall_probe = stall_probe
        self.config = dict(config or {})
        self.position_tolerance = float(
            self.config.get("position_tolerance_m", 0.015)
        )
        self.rotation_tolerance = float(
            self.config.get("rotation_tolerance_rad", 0.15)
        )
        # One take_action commands the arm toward the target for ~0.4 s, which
        # covers a few centimetres. A reach across the workspace therefore needs
        # to hold the target for many of them: measured, a 27 cm move converged
        # in 11. The old default of 3 reported "did not converge" after 0.36 s
        # with the arm barely moved, and the agent read that as unreachable.
        self.max_pose_attempts = max(
            1, int(self.config.get("max_pose_attempts", 60))
        )
        self.settle_attempts = max(
            0, int(self.config.get("settle_attempts", 20))
        )
        self.move_linear_steps = max(
            1, int(self.config.get("move_linear_steps", 10))
        )
        self.follow_arc_steps = max(
            1, int(self.config.get("follow_arc_steps", 12))
        )
        self.gripper_steps = max(1, int(self.config.get("gripper_steps", 1)))
        self.replan_steps = max(1, int(self.config.get("replan_steps", 10)))
        # A rollout that has stopped achieving anything still spends the
        # episode's step budget to the last step: deposit_coin burned all 300 of
        # them without moving a single object by more than 5 mm. Handing control
        # back while budget remains is the only thing that lets the agent try
        # something else. Generous by design -- this must only fire when the
        # cell is provably idle, never on a policy that is merely slow.
        self.vla_stall_steps = max(0, int(self.config.get("vla_stall_steps", 150)))
        self.vla_stall_tolerance_m = float(
            self.config.get("vla_stall_tolerance_m", 0.005)
        )
        # Steps of each judged Pi0.5 chunk that AC-WM executes before replanning.
        self.ac_wm_execute_steps = max(
            1, int(self.config.get("ac_wm_execute_steps", self.replan_steps))
        )
        self._cancel_check: Callable[[], str | None] | None = None
        self._kinematic_chains: dict[str, tuple[UrdfChain, tuple[str, ...]]] = {}
        self._wrist_mounts: dict[str, np.ndarray] = {}

    def execute(
        self,
        action_type: str,
        params: Mapping[str, Any] | None,
        *,
        cancel_check: Callable[[], str | None] | None = None,
    ) -> str:
        handlers = {
            "move_to_pose": self._move_to_pose,
            "move_linear": self._move_linear,
            "set_gripper": self._set_gripper,
            "follow_arc": self._follow_arc,
            "vla_execute": self._vla_execute,
            "recover": self._recover,
            "vla_propose": self._vla_propose,
            "rule_propose": self._rule_propose,
            "execute_action_chunk": self._execute_action_chunk,
        }
        handler = handlers.get(str(action_type).strip())
        if handler is None:
            return f"Unknown action type: {action_type!r}"
        self._cancel_check = cancel_check
        try:
            self._check_interrupted()
            return handler(dict(params or {}))
        except _ActionInterrupted as exc:
            return f"Interrupted: {exc}"
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return f"Failed: {exc}"
        finally:
            self._cancel_check = None

    def _publish_observation(self, observation: Mapping[str, Any]) -> None:
        if self._observation_publisher is None:
            return
        try:
            self._observation_publisher.publish(observation)
        except Exception as exc:
            print(f"[robodojo] observation publish failed: {exc}", flush=True)

    def _check_interrupted(self) -> None:
        if self._cancel_check is None:
            return
        reason = self._cancel_check()
        if reason:
            raise _ActionInterrupted(reason)

    def current_pose_xyzw(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        """Return one arm's current pose in the public Emerge convention."""

        if arm not in _ARM_NAMES:
            raise ValueError("arm must be 'left' or 'right'")
        return self._current_pose_xyzw(arm)

    def current_gripper(self, arm: str) -> float:
        """Return one gripper's normalized opening in ``[0, 1]``."""

        if arm not in _ARM_NAMES:
            raise ValueError("arm must be 'left' or 'right'")
        return self._current_gripper(arm)

    def _move_to_pose(self, params: dict[str, Any]) -> str:
        targets = self._resolve_targets(params, require_orientation=True)
        attempts = max(1, int(params.get("attempts", self.max_pose_attempts)))
        for attempt in range(1, attempts + 1):
            self._check_interrupted()
            self._environment.take_action(self._ee_action(targets))
            if self._terminal():
                return self._terminal_action_result("move_to_pose")
            if self._targets_reached(targets):
                arms = ",".join(targets)
                return f"RoboDojo end-effector pose reached for arm(s) {arms} in {attempt} attempt(s)."
        arms = ",".join(targets)
        return f"Failed: move_to_pose did not converge for arm(s) {arms} after {attempts} attempt(s)"

    def _move_linear(self, params: dict[str, Any]) -> str:
        arm = self._require_single_arm(params)
        start_position, start_orientation = self._current_pose_xyzw(arm)
        if "position_m" in params:
            target_position = PoseUtils.vector(params["position_m"], name="position_m")
        elif "delta_m" in params:
            target_position = start_position + PoseUtils.vector(
                params["delta_m"], name="delta_m"
            )
        else:
            raise ValueError("move_linear requires position_m or delta_m")
        target_orientation = PoseUtils.resolve_orientation(params)
        if target_orientation is None:
            target_orientation = start_orientation
        steps = max(1, int(params.get("steps", self.move_linear_steps)))

        for index in range(steps):
            self._check_interrupted()
            alpha = float(index + 1) / float(steps)
            position = start_position * (1.0 - alpha) + target_position * alpha
            orientation = self._slerp(start_orientation, target_orientation, alpha)
            self._environment.take_action(
                self._ee_action({arm: (position, orientation)})
            )
            if self._terminal():
                break
        # One command per waypoint leaves the arm still catching up to the last
        # one, so hold the final target until it actually arrives. Without this
        # a path that tracked perfectly still reports non-convergence.
        target = {arm: (target_position, target_orientation)}
        for _ in range(self.settle_attempts):
            if self._terminal() or self._targets_reached(target):
                break
            self._check_interrupted()
            self._environment.take_action(self._ee_action(target))
        if self._terminal():
            return self._terminal_action_result("move_linear")
        if not self._terminal() and not self._targets_reached(
            {arm: (target_position, target_orientation)}
        ):
            return f"Failed: move_linear did not converge for arm {arm} after {steps} waypoint(s)"
        return f"RoboDojo arm {arm} moved linearly through {steps} waypoint(s)."

    def _set_gripper(self, params: dict[str, Any]) -> str:
        arms = self._resolve_arms(params, allow_both=True)
        opening = self._resolve_gripper_opening(params)
        steps = max(1, int(params.get("steps", self.gripper_steps)))
        for _ in range(steps):
            self._check_interrupted()
            action = self._joint_hold_action()
            for arm in arms:
                action[f"{arm}_ee_joint_state"] = np.asarray(
                    [opening], dtype=np.float32
                )
            self._environment.take_action(action)
            if self._terminal():
                break
        if self._terminal():
            return self._terminal_action_result("set_gripper")
        return f"RoboDojo gripper command {opening:.3f} applied to arm(s) {','.join(arms)}."

    def _follow_arc(self, params: dict[str, Any]) -> str:
        arm = self._require_single_arm(params)
        center = PoseUtils.vector(params.get("center"), name="center")
        axis = PoseUtils.vector(params.get("axis"), name="axis")
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError("follow_arc.axis must be non-zero")
        axis = axis / norm
        if "radius_m" not in params:
            raise ValueError("follow_arc requires radius_m")
        radius = float(params["radius_m"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("follow_arc.radius_m must be positive")
        angle = self._resolve_angle(params)
        orientation_mode = str(params.get("orientation_mode", "fixed")).strip().lower()
        if orientation_mode not in {"fixed", "co_rotate"}:
            raise ValueError("orientation_mode must be fixed or co_rotate")
        steps = max(1, int(params.get("steps", self.follow_arc_steps)))

        start_position, start_orientation = self._current_pose_xyzw(arm)
        requested_orientation = PoseUtils.resolve_orientation(params)
        if requested_orientation is not None:
            start_orientation = requested_orientation
        start_vector = start_position - center
        if float(np.linalg.norm(start_vector)) <= 1e-12:
            raise ValueError("end-effector cannot start at the arc center")
        start_vector = start_vector / np.linalg.norm(start_vector) * radius

        final_position = start_position
        final_orientation = start_orientation
        for index in range(steps):
            self._check_interrupted()
            step_angle = angle * float(index + 1) / float(steps)
            final_position = center + PoseUtils.rotate_vector(
                start_vector, axis, step_angle
            )
            if orientation_mode == "co_rotate":
                delta = PoseUtils.axis_angle_to_quaternion(axis * step_angle)
                final_orientation = PoseUtils.quaternion_multiply(
                    delta, start_orientation
                )
            else:
                final_orientation = start_orientation
            self._environment.take_action(
                self._ee_action({arm: (final_position, final_orientation)})
            )
            if self._terminal():
                break
        if self._terminal():
            return self._terminal_action_result("follow_arc")
        if not self._terminal() and not self._targets_reached(
            {arm: (final_position, final_orientation)}
        ):
            return f"Failed: follow_arc did not converge for arm {arm}"
        return f"RoboDojo arm {arm} followed an arc through {steps} waypoint(s)."

    def _vla_execute(self, params: dict[str, Any]) -> str:
        if self._policy_client is None:
            raise RuntimeError("RoboDojo Pi0.5 policy client is not configured")
        instruction = str(
            params.get("instruction", params.get("prompt", ""))
        ).strip()
        if not instruction:
            raise ValueError("vla_execute requires instruction or prompt")
        if "step" not in params:
            raise ValueError("vla_execute requires step")
        step_budget = int(params["step"])
        if step_budget <= 0:
            raise ValueError("vla_execute.step must be positive")
        replan_steps = max(1, int(params.get("replan_steps", self.replan_steps)))

        total_steps = 0
        inference_calls = 0
        stall_reference = self._read_stall_probe()
        stalled_since = 0
        while total_steps < step_budget and not self._terminal():
            self._check_interrupted()
            current = self._read_stall_probe()
            if self._is_idle(stall_reference, current):
                if total_steps - stalled_since >= self.vla_stall_steps > 0:
                    return (
                        "RoboDojo VLA execution finished: no progress, "
                        f"steps={total_steps}, inference_calls={inference_calls}. "
                        f"Nothing in the cell moved for {self.vla_stall_steps} steps, "
                        "so the rollout was stopped with budget left. Change the "
                        "situation before handing back to the policy."
                    )
            else:
                stall_reference = current
                stalled_since = total_steps
            observation = dict(self._environment.get_obs())
            observation["instruction"] = instruction
            # Republish from the observation the policy is about to act on, so
            # the visual monitor can interrupt a rollout partway rather than
            # only judging it once the whole step budget is spent. Reusing this
            # one costs no extra render.
            self._publish_observation(observation)
            result = self._policy_client.infer(encode_observation(observation))
            if "actions" not in result:
                raise RuntimeError("Pi0.5 response is missing actions")
            chunk = unpack_joint_actions(result["actions"])
            if not chunk:
                raise RuntimeError("Pi0.5 returned an empty action chunk")
            inference_calls += 1
            execute_count = min(replan_steps, step_budget - total_steps, len(chunk))
            for action in chunk[:execute_count]:
                self._check_interrupted()
                self._environment.take_action(action)
                total_steps += 1
                if self._terminal():
                    break
        if self._terminal() and not self._task_succeeded():
            return (
                "Failed: RoboDojo VLA execution reached a failed terminal state, "
                f"steps={total_steps}, inference_calls={inference_calls}."
            )
        outcome = "task success" if self._terminal() else "step budget reached"
        return (
            f"RoboDojo VLA execution finished: {outcome}, "
            f"steps={total_steps}, inference_calls={inference_calls}."
        )

    def _vla_propose(self, params: dict[str, Any]) -> str:
        """Infer one Pi0.5 chunk for AC-WM to judge, without stepping the scene."""
        if self._policy_client is None:
            raise RuntimeError("RoboDojo Pi0.5 policy client is not configured")
        instruction = str(
            params.get("instruction", params.get("prompt", ""))
        ).strip()
        if not instruction:
            raise ValueError("vla_propose requires instruction or prompt")
        horizon = max(1, min(int(params.get("horizon", _AC_WM_MAX_ROWS)), _AC_WM_MAX_ROWS))
        max_execute = max(1, int(params.get("max_execute_steps", horizon)))
        execute_limit = max(1, int(params.get("replan_steps", self.ac_wm_execute_steps)))
        if self._terminal():
            return self._terminal_action_result("vla_propose")
        observation = dict(self._environment.get_obs())
        observation["instruction"] = instruction
        self._publish_observation(observation)
        views = self._ac_wm_views(observation)
        result = self._policy_client.infer(encode_observation(observation))
        if "actions" not in result:
            raise RuntimeError("Pi0.5 response is missing actions")
        rows = self._action_rows(result["actions"], ACTION_DIM, name="Pi0.5 actions")[:horizon]
        execute_steps = min(execute_limit, max_execute, len(rows))
        return "VLA_PROPOSAL:" + self._proposal(rows, execute_steps, _JOINT_CONTROL, views)

    def _rule_propose(self, params: dict[str, Any]) -> str:
        """Preview a geometric skill as the exact rows it would command."""
        skill = str(params.get("skill_action_type", "")).strip()
        inner = params.get("parameters")
        if not isinstance(inner, Mapping):
            raise ValueError("rule_propose requires parameters")
        builders = {
            "move_to_pose": self._preview_move_to_pose,
            "move_linear": self._preview_move_linear,
            "follow_arc": self._preview_follow_arc,
            "set_gripper": self._preview_set_gripper,
        }
        builder = builders.get(skill)
        if builder is None:
            raise ValueError(f"rule_propose does not support {skill!r}")
        if self._terminal():
            return self._terminal_action_result("rule_propose")
        rows, control_space = builder(dict(inner))
        views = self._ac_wm_views()
        return "RULE_PROPOSAL:" + self._proposal(rows, len(rows), control_space, views)

    def _execute_action_chunk(self, params: dict[str, Any]) -> str:
        """Replay exactly the rows AC-WM selected, stopping only at a terminal or arrival."""
        control_space = str(params.get("control_space") or _JOINT_CONTROL)
        if control_space not in _CONTROL_DOMAINS:
            raise ValueError(f"unsupported control_space {control_space!r}")
        width = ACTION_DIM if control_space == _JOINT_CONTROL else _EE_WIDTH
        rows = self._action_rows(params.get("actions"), width, name="execute_action_chunk.actions")
        if len(rows) > _AC_WM_MAX_ROWS:
            raise ValueError(f"execute_action_chunk accepts at most {_AC_WM_MAX_ROWS} rows")
        candidate_id = str(params.get("candidate_id", "candidate"))
        skill_name = str(params.get("skill_name", "unknown"))
        if control_space == _JOINT_CONTROL:
            actions = unpack_joint_actions(rows)
            final, hold_from = None, len(rows)
        else:
            actions = [self._ee16_action(row) for row in rows]
            final = self._ee16_targets(rows[-1])
            # Rows equal to the final waypoint only hold it; stop once it is reached.
            hold_from = len(rows) - 1
            while hold_from > 0 and np.array_equal(rows[hold_from - 1], rows[-1]):
                hold_from -= 1

        steps, reason = 0, "chunk_completed"
        try:
            for index, action in enumerate(actions):
                self._check_interrupted()
                self._environment.take_action(action)
                steps += 1
                if self._terminal():
                    break
                if final is not None and index >= hold_from and self._targets_reached(final):
                    reason = "target_reached"
                    break
        except _ActionInterrupted as exc:
            return f"Interrupted: {exc}; steps={steps}, reason=interrupted"
        if self._terminal():
            if not self._task_succeeded():
                return (
                    "Failed: RoboDojo selected chunk reached a failed terminal state; "
                    f"steps={steps}, reason=failed_terminal"
                )
            reason = "goal_reached"
        elif final is not None and not self._targets_reached(final):
            return (
                f"Failed: {skill_name} chunk did not converge to its final waypoint; "
                f"steps={steps}, reason=not_converged"
            )
        return (
            f"Selected candidate {candidate_id} from skill {skill_name} executed its "
            f"exact proposed prefix: steps={steps}, reason={reason}."
        )

    def _preview_move_to_pose(self, params: dict[str, Any]) -> tuple[list[list[float]], str]:
        targets = self._resolve_targets(params, require_orientation=True)
        attempts = max(1, int(params.get("attempts", self.max_pose_attempts)))
        return self._ee16_rows([targets] * min(attempts, _AC_WM_MAX_ROWS)), _EE_CONTROL

    def _preview_move_linear(self, params: dict[str, Any]) -> tuple[list[list[float]], str]:
        arm = self._require_single_arm(params)
        start_position, start_orientation = self._current_pose_xyzw(arm)
        if "position_m" in params:
            target_position = PoseUtils.vector(params["position_m"], name="position_m")
        elif "delta_m" in params:
            target_position = start_position + PoseUtils.vector(params["delta_m"], name="delta_m")
        else:
            raise ValueError("move_linear requires position_m or delta_m")
        target_orientation = PoseUtils.resolve_orientation(params)
        if target_orientation is None:
            target_orientation = start_orientation
        steps = min(max(1, int(params.get("steps", self.move_linear_steps))), _AC_WM_MAX_ROWS)
        waypoints = []
        for index in range(steps):
            alpha = float(index + 1) / float(steps)
            waypoints.append({arm: (
                start_position * (1.0 - alpha) + target_position * alpha,
                self._slerp(start_orientation, target_orientation, alpha),
            )})
        waypoints += [{arm: (target_position, target_orientation)}] * min(
            self.settle_attempts, _AC_WM_MAX_ROWS - steps
        )
        return self._ee16_rows(waypoints), _EE_CONTROL

    def _preview_follow_arc(self, params: dict[str, Any]) -> tuple[list[list[float]], str]:
        arm = self._require_single_arm(params)
        center = PoseUtils.vector(params.get("center"), name="center")
        axis = PoseUtils.vector(params.get("axis"), name="axis")
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-12:
            raise ValueError("follow_arc.axis must be non-zero")
        axis = axis / norm
        if "radius_m" not in params:
            raise ValueError("follow_arc requires radius_m")
        radius = float(params["radius_m"])
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("follow_arc.radius_m must be positive")
        angle = self._resolve_angle(params)
        orientation_mode = str(params.get("orientation_mode", "fixed")).strip().lower()
        if orientation_mode not in {"fixed", "co_rotate"}:
            raise ValueError("orientation_mode must be fixed or co_rotate")
        steps = min(max(1, int(params.get("steps", self.follow_arc_steps))), _AC_WM_MAX_ROWS)

        start_position, start_orientation = self._current_pose_xyzw(arm)
        requested_orientation = PoseUtils.resolve_orientation(params)
        if requested_orientation is not None:
            start_orientation = requested_orientation
        start_vector = start_position - center
        if float(np.linalg.norm(start_vector)) <= 1e-12:
            raise ValueError("end-effector cannot start at the arc center")
        start_vector = start_vector / np.linalg.norm(start_vector) * radius
        waypoints = []
        for index in range(steps):
            step_angle = angle * float(index + 1) / float(steps)
            position = center + PoseUtils.rotate_vector(start_vector, axis, step_angle)
            orientation = start_orientation
            if orientation_mode == "co_rotate":
                delta = PoseUtils.axis_angle_to_quaternion(axis * step_angle)
                orientation = PoseUtils.quaternion_multiply(delta, start_orientation)
            waypoints.append({arm: (position, orientation)})
        waypoints += [waypoints[-1]] * min(self.settle_attempts, _AC_WM_MAX_ROWS - steps)
        return self._ee16_rows(waypoints), _EE_CONTROL

    def _preview_set_gripper(self, params: dict[str, Any]) -> tuple[list[list[float]], str]:
        arms = self._resolve_arms(params, allow_both=True)
        opening = self._resolve_gripper_opening(params)
        steps = min(max(1, int(params.get("steps", self.gripper_steps))), _AC_WM_MAX_ROWS)
        manager = self._environment.robot_manager
        row: list[float] = []
        for arm in _ARM_NAMES:
            joints = to_float_array(
                manager.get_joint(self._robot(arm), env_idx_list=[0])[0]
            ).reshape(-1)
            if joints.shape != (6,) or not np.all(np.isfinite(joints)):
                raise RuntimeError(f"RoboDojo returned invalid {arm} joint state")
            row.extend(float(value) for value in joints)
            row.append(opening if arm in arms else self._current_gripper(arm))
        return [list(row) for _ in range(steps)], _JOINT_CONTROL

    def _ee16_rows(
        self, waypoints: list[Mapping[str, tuple[np.ndarray, np.ndarray]]],
    ) -> list[list[float]]:
        """Encode waypoints as ee16 rows; an arm without a target holds its pose."""
        hold = {arm: self._current_pose_xyzw(arm) for arm in _ARM_NAMES}
        grippers = {arm: self._current_gripper(arm) for arm in _ARM_NAMES}
        rows = []
        for targets in waypoints:
            row: list[float] = []
            for arm in _ARM_NAMES:
                position, orientation = targets.get(arm, hold[arm])
                row.extend(float(value) for value in position)
                row.extend(float(value) for value in PoseUtils.normalize_quaternion(orientation))
                row.append(grippers[arm])
            rows.append(row)
        return rows

    @staticmethod
    def _ee16_targets(row: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            arm: (row[index * 8:index * 8 + 3], PoseUtils.normalize_quaternion(row[index * 8 + 3:index * 8 + 7]))
            for index, arm in enumerate(_ARM_NAMES)
        }

    def _ee16_action(self, row: np.ndarray) -> dict[str, np.ndarray]:
        action: dict[str, np.ndarray] = {}
        for index, arm in enumerate(_ARM_NAMES):
            part = row[index * 8:(index + 1) * 8]
            if not -1e-6 <= float(part[7]) <= 1.0 + 1e-6:
                raise ValueError("ee16 gripper opening must be in normalized range [0, 1]")
            action[f"{arm}_ee_pose"] = np.concatenate(
                (part[:3], self._xyzw_to_wxyz(part[3:7]))
            ).astype(np.float32)
            action[f"{arm}_ee_joint_state"] = np.asarray(
                [float(np.clip(part[7], 0.0, 1.0))], dtype=np.float32
            )
        return action

    @staticmethod
    def _action_rows(value: Any, width: int, *, name: str) -> np.ndarray:
        try:
            rows = to_float_array(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a numeric array") from exc
        if rows.ndim == 1:
            rows = rows[None, :]
        if rows.ndim != 2 or rows.shape[1] != width or len(rows) == 0:
            raise ValueError(f"{name} must have shape (rows, {width}), got {rows.shape}")
        if not np.all(np.isfinite(rows)):
            raise ValueError(f"{name} must be finite")
        return rows

    def _ac_wm_views(self, observation: Mapping[str, Any] | None = None) -> list[Any]:
        publisher = self._observation_publisher
        if publisher is None or not hasattr(publisher, "ac_wm_views"):
            raise RuntimeError("AC-WM proposals require the RoboDojo observation publisher")
        try:
            return publisher.ac_wm_views(observation)
        except RuntimeError:
            if observation is not None:
                raise
            # Nothing captured yet this episode: render once rather than fail.
            return publisher.ac_wm_views(dict(self._environment.get_obs()))

    def _proposal(self, rows: Any, execute_steps: int, control_space: str, views: list[Any]) -> str:
        rows = np.asarray(rows, dtype=np.float64)
        payload: dict[str, Any] = {
            "actions": rows.tolist(),
            "execute_steps": int(execute_steps),
            "domain_name": _CONTROL_DOMAINS[control_space],
            "control_space": control_space,
            "control_description": _CONTROL_DESCRIPTIONS[control_space],
        }
        annotated: dict[str, np.ndarray] = {}
        panels: list[tuple[str, np.ndarray]] = []
        try:
            preview = self._trajectory_preview(rows, control_space, execute_steps)
            for view in views:
                arm = self._wrist_arm(view.name)
                transform = self._live_camera_pose(view, arm)
                if view.intrinsics is not None and transform is not None:
                    caption = f"{arm} wrist camera, moves with the {arm} gripper" if arm else None
                    annotated[view.name] = draw_camera_overlay(view.rgb, view.intrinsics, transform, preview, caption)
            panels.append(("plan_schematic", draw_plan_schematic(preview)))
            payload["preview"] = preview.to_json()
        except Exception as exc:
            # The proposal stays executable; the judge falls back to the raw rows.
            annotated, panels = {}, []
            payload["preview_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[robodojo] AC-WM trajectory preview failed: {exc}", flush=True)
        payload.update(self._observation_publisher.write_ac_wm_snapshot(views, annotated=annotated, panels=panels))
        return json.dumps(payload)

    def _trajectory_preview(self, rows: np.ndarray, control_space: str, execute_steps: int) -> TrajectoryPreview:
        """Each arm's fingertip path in robodojo_env, starting from where it is now."""
        manager = self._environment.robot_manager
        arms: dict[str, ArmPath] = {}
        for index, arm in enumerate(_ARM_NAMES):
            robot = self._robot(arm)
            bias = float(getattr(robot, "gripper_bias", 0.0) or 0.0)
            position, orientation = self._current_pose_xyzw(arm)
            now = _pose_matrix(position, orientation)
            if control_space == _EE_CONTROL:
                part = rows[:, index * 8:(index + 1) * 8]
                flanges = [_pose_matrix(row[:3], row[3:7]) for row in part]
                grippers = part[:, 7]
            else:
                part = rows[:, index * 7:(index + 1) * 7]
                chain, names = self._kinematic_chain(arm)
                current = to_float_array(manager.get_joint(robot, env_idx_list=[0])[0]).reshape(-1)
                # Anchored on the measured pose, so any fixed base offset between
                # the URDF and the simulator cancels and FK is exact right now.
                anchor = now @ np.linalg.inv(chain.forward(dict(zip(names, current))))
                flanges = [anchor @ chain.forward(dict(zip(names, row[:6]))) for row in part]
                grippers = part[:, 6]
            tcp = [frame[:3, 3] + frame[:3, 0] * bias for frame in [now, *flanges]]
            arms[arm] = ArmPath(np.asarray(tcp), np.concatenate(([self._current_gripper(arm)], grippers)))
        return TrajectoryPreview(arms, int(execute_steps))

    @staticmethod
    def _wrist_arm(camera_name: str) -> str | None:
        if "wrist" not in camera_name:
            return None
        return next((arm for arm in _ARM_NAMES if f"_{arm}_" in f"_{camera_name}_"), None)

    def _live_camera_pose(self, view: Any, arm: str | None) -> np.ndarray | None:
        """``T_env_camera`` at the moment the frame was rendered; None when unknown."""
        if view.t_env_camera is None or arm is None:
            return view.t_env_camera
        try:
            chain, names = self._kinematic_chain(arm)
            robot = self._robot(arm)
            position, orientation = self._current_pose_xyzw(arm)
            flange = _pose_matrix(position, orientation)
            current = to_float_array(
                self._environment.robot_manager.get_joint(robot, env_idx_list=[0])[0]
            ).reshape(-1)
            base = flange @ np.linalg.inv(chain.forward(dict(zip(names, current))))
            spawn = base @ chain.forward(dict.fromkeys(names, 0.0))
            pose = remount_camera(view.t_env_camera, flange, spawn, self._wrist_mount_translation(arm))
        except (RuntimeError, ValueError) as exc:
            print(f"[robodojo] AC-WM {view.name} calibration unavailable: {exc}", flush=True)
            return None
        if pose is None:
            print(f"[robodojo] AC-WM {view.name} pose does not match its URDF mount; path not drawn", flush=True)
        return pose

    def _wrist_mount_translation(self, arm: str) -> np.ndarray:
        if arm not in self._wrist_mounts:
            robot = self._robot(arm)
            link = next((str(camera.get("link")) for camera in getattr(robot, "camera", None) or ()
                         if str(camera.get("name", "")).endswith("_wrist") and camera.get("link")), "camera")
            mount = UrdfChain.from_urdf(robot.urdf_path, robot.ee_link_name, link).forward({})
            self._wrist_mounts[arm] = mount[:3, 3]
        return self._wrist_mounts[arm]

    def _kinematic_chain(self, arm: str) -> tuple[UrdfChain, tuple[str, ...]]:
        if arm not in self._kinematic_chains:
            robot = self._robot(arm)
            urdf = getattr(robot, "urdf_path", None)
            names = tuple(getattr(robot, "arm_joints_name", ()) or ())
            if not urdf or not Path(urdf).is_file() or len(names) != 6:
                raise RuntimeError(f"RoboDojo {arm} arm has no usable URDF for forward kinematics")
            chain = UrdfChain.from_urdf(urdf, getattr(robot, "base_link", "base_link"), robot.ee_link_name)
            if set(chain.movable_joints) != set(names):
                raise RuntimeError(
                    f"URDF chain joints {chain.movable_joints} do not match the {arm} arm joints {names}"
                )
            self._kinematic_chains[arm] = (chain, names)
        return self._kinematic_chains[arm]

    def _resolve_targets(
        self,
        params: dict[str, Any],
        *,
        require_orientation: bool,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        arm_value = str(params.get("arm", "")).strip().lower()
        if arm_value == "both":
            raw_targets = params.get("targets")
            if not isinstance(raw_targets, Mapping):
                raise ValueError("arm='both' requires targets.left and targets.right")
            targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for arm in _ARM_NAMES:
                raw = raw_targets.get(arm)
                if not isinstance(raw, Mapping):
                    raise ValueError(f"arm='both' requires targets.{arm}")
                targets[arm] = self._parse_pose_target(
                    dict(raw), require_orientation=require_orientation
                )
            return targets
        arm = self._require_single_arm(params)
        return {
            arm: self._parse_pose_target(
                params, require_orientation=require_orientation
            )
        }

    def _parse_pose_target(
        self,
        params: dict[str, Any],
        *,
        require_orientation: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        if "position_m" not in params:
            raise ValueError("pose target requires position_m")
        position = PoseUtils.vector(params["position_m"], name="position_m")
        orientation = PoseUtils.resolve_orientation(params)
        if orientation is None:
            if require_orientation:
                raise ValueError(
                    "pose target requires orientation_quat or orientation_euler"
                )
            orientation = np.array([0.0, 0.0, 0.0, 1.0])
        return position, orientation

    @staticmethod
    def _require_single_arm(params: Mapping[str, Any]) -> str:
        arm = str(params.get("arm", "")).strip().lower()
        if arm not in _ARM_NAMES:
            raise ValueError("RoboDojo dual-arm action requires arm='left' or arm='right'")
        return arm

    @staticmethod
    def _resolve_arms(
        params: Mapping[str, Any],
        *,
        allow_both: bool,
    ) -> tuple[str, ...]:
        arm = str(params.get("arm", "")).strip().lower()
        if arm in _ARM_NAMES:
            return (arm,)
        if allow_both and arm == "both":
            return _ARM_NAMES
        expected = "left, right, or both" if allow_both else "left or right"
        raise ValueError(f"RoboDojo dual-arm action requires arm={expected}")

    @staticmethod
    def _resolve_gripper_opening(params: Mapping[str, Any]) -> float:
        if "opening" in params:
            opening = float(params["opening"])
        elif "opening_m" in params:
            # ARX-X5's benchmark action is a normalized opening, but accept the
            # common metric contract using the physical 0.088 m span.
            opening = float(params["opening_m"]) / 0.088
        elif "width_m" in params:
            opening = float(params["width_m"]) / 0.088
        else:
            command = str(params.get("command", "")).strip().lower()
            if command == "open":
                opening = 1.0
            elif command == "close":
                opening = 0.0
            else:
                raise ValueError(
                    "set_gripper requires opening, opening_m, width_m, or command"
                )
        if not math.isfinite(opening) or not 0.0 <= opening <= 1.0:
            raise ValueError("RoboDojo gripper opening must be in normalized range [0, 1]")
        return opening

    def _ee_action(
        self,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        action: dict[str, np.ndarray] = {}
        for arm in _ARM_NAMES:
            if arm in targets:
                position, orientation_xyzw = targets[arm]
            else:
                position, orientation_xyzw = self._current_pose_xyzw(arm)
            orientation_wxyz = self._xyzw_to_wxyz(orientation_xyzw)
            action[f"{arm}_ee_pose"] = np.concatenate(
                (position, orientation_wxyz)
            ).astype(np.float32)
            action[f"{arm}_ee_joint_state"] = np.asarray(
                [self._current_gripper(arm)], dtype=np.float32
            )
        return action

    def _joint_hold_action(self) -> dict[str, np.ndarray]:
        manager = self._environment.robot_manager
        action: dict[str, np.ndarray] = {}
        for arm in _ARM_NAMES:
            robot = self._robot(arm)
            action[f"{arm}_arm_joint_state"] = np.asarray(
                manager.get_joint(robot, env_idx_list=[0])[0], dtype=np.float32
            )
            action[f"{arm}_ee_joint_state"] = np.asarray(
                [self._current_gripper(arm)], dtype=np.float32
            )
        return action

    def _robot(self, arm: str) -> Any:
        return self._environment.robot_manager.get_robot_by_arm_name(f"{arm}_arm")

    def _current_pose_xyzw(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        manager = self._environment.robot_manager
        pose = to_float_array(
            manager.get_real_endpose(
                self._robot(arm), env_idx_list=[0], is_relative=True
            )[0]
        ).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"RoboDojo returned invalid {arm} end-effector pose")
        return pose[:3], self._wxyz_to_xyzw(pose[3:])

    def _current_gripper(self, arm: str) -> float:
        robot = self._robot(arm)
        manager = self._environment.robot_manager
        raw = float(
            to_numpy(
                manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]
            ).reshape(-1)[0]
        )
        low, high = map(float, robot.gripper_scale)
        if high <= low:
            raise RuntimeError(f"Invalid RoboDojo {arm} gripper scale")
        if robot.gripper_move["sign"] == 1:
            return float(np.clip((raw - low) / (high - low), 0.0, 1.0))
        return float(np.clip((high - raw) / (high - low), 0.0, 1.0))

    def _targets_reached(
        self,
        targets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> bool:
        for arm, (target_position, target_orientation) in targets.items():
            current_position, current_orientation = self._current_pose_xyzw(arm)
            if float(np.linalg.norm(target_position - current_position)) > self.position_tolerance:
                return False
            rotation_error = PoseUtils.quaternion_error(
                current_orientation, target_orientation
            )
            if float(np.linalg.norm(rotation_error)) > self.rotation_tolerance:
                return False
        return True

    _RECOVERIES = ("release_and_retreat", "retreat", "open_gripper", "park")

    def _recover(self, params: dict[str, Any]) -> str:
        """Undo a stuck situation with a fixed, executable sequence.

        Left to compose its own recovery the agent produced absolute poses the
        arm could not reach and gripper commands that changed nothing -- four
        interventions across sixty episodes, none of which worked. These
        sequences use relative targets to simplify recovery planning. Those
        targets can still be unreachable, so every primitive result must be
        checked before continuing or reporting recovery progress.
        """
        mode = str(params.get("mode", "release_and_retreat")).strip().lower()
        if mode not in self._RECOVERIES:
            raise ValueError(
                f"recover.mode must be one of {', '.join(self._RECOVERIES)}, got {mode!r}"
            )
        arm = self._require_single_arm(params)
        height = float(params.get("clearance_m", 0.12))
        if not math.isfinite(height) or height <= 0:
            raise ValueError("recover.clearance_m must be finite and positive")
        steps = max(1, int(params.get("steps", self.move_linear_steps * 2)))
        done: list[str] = []

        def check_result(result: str) -> str | None:
            if result.startswith(("Failed:", "Interrupted:", "Unknown action")):
                return result
            if self._terminal():
                return self._terminal_action_result("recover")
            return None

        if mode in ("release_and_retreat", "open_gripper"):
            result = self._set_gripper({"arm": arm, "command": "open"})
            if stopped := check_result(result):
                return stopped
            done.append("applied the gripper-open command")
        if mode in ("release_and_retreat", "retreat", "park"):
            result = self._move_linear({"arm": arm, "delta_m": [0.0, 0.0, height], "steps": steps})
            if stopped := check_result(result):
                return stopped
            done.append(f"lifted {height * 100:.0f} cm clear")
        if mode == "park":
            # Straight back from the table rather than to a fixed home pose: the
            # workspace differs per task and an absolute target risks the same
            # unreachable-pose failure this exists to avoid.
            result = self._move_linear({"arm": arm, "delta_m": [0.0, -0.15, 0.0], "steps": steps})
            if stopped := check_result(result):
                return stopped
            done.append("moved the arm out of the way")

        return f"RoboDojo recovery '{mode}' on arm {arm}: {', '.join(done)}."

    def _read_stall_probe(self) -> list[float] | None:
        if self._stall_probe is None:
            return None
        try:
            return self._stall_probe()
        except Exception:
            # Losing the stall check must never abort a healthy rollout.
            return None

    def _is_idle(
        self,
        reference: list[float] | None,
        current: list[float] | None,
    ) -> bool:
        """Whether the arms have not moved since the reference reading."""
        if not reference or current is None:
            return False
        if len(reference) != len(current):
            return False
        return all(
            abs(a - b) <= self.vla_stall_tolerance_m
            for a, b in zip(reference, current)
        )

    def _terminal(self) -> bool:
        end_flag = getattr(self._environment, "end_flag", None)
        return bool(end_flag and end_flag[0])

    def _task_succeeded(self) -> bool:
        success = getattr(self._environment, "success", None)
        return bool(success and success[0])

    def _terminal_action_result(self, action_type: str) -> str:
        if self._task_succeeded():
            return f"RoboDojo {action_type} reached official task success."
        return f"Failed: RoboDojo {action_type} reached a failed terminal state"

    @staticmethod
    def _wxyz_to_xyzw(quaternion: Any) -> np.ndarray:
        value = PoseUtils.vector(quaternion, name="quaternion_wxyz", length=4)
        return PoseUtils.normalize_quaternion(
            np.array([value[1], value[2], value[3], value[0]])
        )

    @staticmethod
    def _xyzw_to_wxyz(quaternion: Any) -> np.ndarray:
        value = PoseUtils.normalize_quaternion(quaternion)
        return np.array([value[3], value[0], value[1], value[2]])

    @staticmethod
    def _slerp(start: Any, target: Any, alpha: float) -> np.ndarray:
        left = PoseUtils.normalize_quaternion(start)
        right = PoseUtils.normalize_quaternion(target)
        dot = float(np.dot(left, right))
        if dot < 0.0:
            right = -right
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return PoseUtils.normalize_quaternion(
                left + float(alpha) * (right - left)
            )
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        return PoseUtils.normalize_quaternion(
            math.sin((1.0 - alpha) * theta) / sin_theta * left
            + math.sin(alpha * theta) / sin_theta * right
        )

    @staticmethod
    def _resolve_angle(params: Mapping[str, Any]) -> float:
        if "angle_rad" in params:
            return float(params["angle_rad"])
        if "angle_deg" in params:
            return math.radians(float(params["angle_deg"]))
        if "angle" in params:
            value = float(params["angle"])
            unit = str(params.get("angle_unit", "degrees")).strip().lower()
            return value if unit in {"rad", "radian", "radians"} else math.radians(value)
        raise ValueError("follow_arc requires angle_deg or angle_rad")
