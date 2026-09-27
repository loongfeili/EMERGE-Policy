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

import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol

import numpy as np

from robot.mujoco_simulation.pose_utils import PoseUtils
from robot.robodojo_simulation.tensors import to_float_array, to_numpy
from robot.vla.robodojo_policy import encode_observation, unpack_joint_actions

_ARM_NAMES = ("left", "right")


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
        self._cancel_check: Callable[[], str | None] | None = None

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
