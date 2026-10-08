"""Turn a proposed RoboDojo action chunk into gripper paths a VLM can judge.

Joint angles mean nothing to a vision-language judge, so AC-WM converts every
proposal into the path of each gripper's fingertip centre (TCP) in the
robodojo_env frame, draws it onto every calibrated camera view, and adds a
top/side schematic in metres.  Nothing here touches Isaac: forward kinematics
reads the robot URDF directly.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FRAME_DESCRIPTION = (
    "robodojo_env frame in metres: +x toward the right arm, +y forward away from "
    "the robot bases, +z up"
)
POINT_DESCRIPTION = "gripper fingertip centre (TCP)"
ARM_COLORS_RGB = {"left": (255, 140, 0), "right": (0, 190, 255)}
ARM_COLOR_NAMES = {"left": "orange", "right": "cyan"}
_STILL_M = 0.005
_STILL_GRIPPER = 0.05


def _floats(text: str) -> np.ndarray:
    return np.array([float(value) for value in text.split()], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def quaternion_xyzw_matrix(quaternion: Any) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4) / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass(frozen=True)
class _ChainJoint:
    name: str
    kind: str
    origin: np.ndarray
    axis: np.ndarray


class UrdfChain:
    """Forward kinematics along one URDF chain from a base link to a tip link."""

    def __init__(self, joints: tuple[_ChainJoint, ...]) -> None:
        self._joints = joints

    @classmethod
    def from_urdf(cls, path: str | Path, base_link: str, tip_link: str) -> UrdfChain:
        root = ET.parse(str(path)).getroot()
        by_child = {joint.find("child").get("link"): joint for joint in root.iter("joint")}
        chain: list[_ChainJoint] = []
        link = tip_link
        while link != base_link:
            element = by_child.get(link)
            if element is None:
                raise ValueError(f"{path}: no joint chain from {base_link!r} to {tip_link!r}")
            origin = element.find("origin")
            transform = np.eye(4)
            if origin is not None:
                transform[:3, :3] = _rpy_matrix(_floats(origin.get("rpy", "0 0 0")))
                transform[:3, 3] = _floats(origin.get("xyz", "0 0 0"))
            axis_element = element.find("axis")
            axis = _floats(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0")
            chain.append(_ChainJoint(element.get("name"), element.get("type"), transform, axis / np.linalg.norm(axis)))
            link = element.find("parent").get("link")
        return cls(tuple(reversed(chain)))

    @property
    def movable_joints(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self._joints if joint.kind != "fixed")

    def forward(self, positions: Mapping[str, float]) -> np.ndarray:
        """Return the tip transform in the base-link frame."""
        transform = np.eye(4)
        for joint in self._joints:
            transform = transform @ joint.origin
            if joint.kind == "fixed":
                continue
            if joint.name not in positions:
                raise ValueError(f"missing position for joint {joint.name!r}")
            value = float(positions[joint.name])
            motion = np.eye(4)
            if joint.kind in {"revolute", "continuous"}:
                motion[:3, :3] = _axis_angle_matrix(joint.axis, value)
            elif joint.kind == "prismatic":
                motion[:3, 3] = joint.axis * value
            else:
                raise ValueError(f"unsupported joint type {joint.kind!r} for {joint.name!r}")
            transform = transform @ motion
        return transform


@dataclass(frozen=True)
class ArmPath:
    """One arm's planned TCP path; index 0 is where the gripper is now."""

    tcp: np.ndarray
    gripper: np.ndarray

    @property
    def moves(self) -> bool:
        travel = float(np.max(np.linalg.norm(self.tcp - self.tcp[0], axis=1)))
        return travel > _STILL_M or float(np.ptp(self.gripper)) > _STILL_GRIPPER


@dataclass(frozen=True)
class TrajectoryPreview:
    arms: Mapping[str, ArmPath]
    execute_steps: int

    @property
    def rows(self) -> int:
        return len(next(iter(self.arms.values())).tcp) - 1

    def to_json(self) -> dict[str, Any]:
        return {
            "frame": FRAME_DESCRIPTION,
            "point": POINT_DESCRIPTION,
            "execute_steps": int(self.execute_steps),
            "arms": {
                arm: {"tcp": np.round(path.tcp, 4).tolist(), "gripper": np.round(path.gripper, 3).tolist()}
                for arm, path in self.arms.items()
            },
        }


def tcp_position(flange_position: Any, flange_quaternion_xyzw: Any, bias_m: float) -> np.ndarray:
    """Fingertip centre: the gripper extends along the flange's local +x."""
    rotation = quaternion_xyzw_matrix(flange_quaternion_xyzw)
    return np.asarray(flange_position, dtype=np.float64).reshape(3) + rotation[:, 0] * float(bias_m)


def project(points: np.ndarray, intrinsics: np.ndarray, t_env_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of env-frame points for a ROS-axes camera, plus visibility."""
    t_camera_env = np.linalg.inv(t_env_camera)
    camera = points @ t_camera_env[:3, :3].T + t_camera_env[:3, 3]
    visible = camera[:, 2] > 0.02
    homogeneous = camera @ np.asarray(intrinsics, dtype=np.float64).T
    depth = np.where(visible, homogeneous[:, 2], 1.0)
    return homogeneous[:, :2] / depth[:, None], visible


def _pixel(point: np.ndarray, limit: int) -> tuple[int, int]:
    return tuple(int(round(float(np.clip(value, -limit, limit)))) for value in point)


def _gripper_events(gripper: np.ndarray) -> list[tuple[int, str]]:
    events = []
    for index in range(1, len(gripper)):
        if gripper[index - 1] >= 0.5 > gripper[index]:
            events.append((index, "close"))
        elif gripper[index - 1] < 0.5 <= gripper[index]:
            events.append((index, "open"))
    return events


def _draw_path(image: np.ndarray, points: np.ndarray, visible: np.ndarray, path: ArmPath,
               execute_steps: int, color: tuple[int, int, int], scale: int, label: str) -> None:
    import cv2

    limit = 10 * max(image.shape[:2])
    pixels = [_pixel(point, limit) for point in points]
    last = len(pixels) - 1
    executed = min(execute_steps, last)
    for index in range(last):
        if visible[index] and visible[index + 1]:
            thickness = 3 * scale if index < executed else scale
            cv2.line(image, pixels[index], pixels[index + 1], color, thickness, cv2.LINE_AA)
    if visible[0]:
        cv2.circle(image, pixels[0], 7 * scale, (255, 255, 255), 2 * scale, cv2.LINE_AA)
        cv2.circle(image, pixels[0], 7 * scale, color, scale, cv2.LINE_AA)
        text = f"{label} holds" if not path.moves else label
        cv2.putText(image, text, (pixels[0][0] + 9 * scale, pixels[0][1] - 6 * scale),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, color, scale, cv2.LINE_AA)
    if executed > 0 and visible[executed]:
        cv2.circle(image, pixels[executed], 5 * scale, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, pixels[executed], 4 * scale, color, -1, cv2.LINE_AA)
    if last > executed and visible[last]:
        x, y = pixels[last]
        arm = 5 * scale
        cv2.line(image, (x - arm, y - arm), (x + arm, y + arm), color, 2 * scale, cv2.LINE_AA)
        cv2.line(image, (x - arm, y + arm), (x + arm, y - arm), color, 2 * scale, cv2.LINE_AA)
    for index, event in _gripper_events(path.gripper):
        if visible[index]:
            x, y = pixels[index]
            side = 4 * scale
            cv2.rectangle(image, (x - side, y - side), (x + side, y + side), (255, 255, 255), -1)
            cv2.rectangle(image, (x - side, y - side), (x + side, y + side), color, scale)
            cv2.putText(image, event, (x + 7 * scale, y + 12 * scale), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42 * scale, color, scale, cv2.LINE_AA)


def _draw_legend(image: np.ndarray, preview: TrajectoryPreview, scale: int) -> None:
    import cv2

    lines = [(f"{arm}: {ARM_COLOR_NAMES.get(arm, 'white')} gripper-tip path", ARM_COLORS_RGB.get(arm, (255, 255, 255)))
             for arm in preview.arms]
    lines += [
        (f"thick = next {preview.execute_steps} steps (execute now), thin = rest of plan", (255, 255, 255)),
        ("ring = now, dot = end of executed part, x = plan end", (255, 255, 255)),
    ]
    font_scale = 0.42 * scale
    line_height = int(16 * scale)
    width = max(cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, scale)[0][0] for text, _ in lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (width + 12 * scale, line_height * len(lines) + 8 * scale), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, dst=image)
    for row, (text, color) in enumerate(lines):
        cv2.putText(image, text, (6 * scale, line_height * (row + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, scale, cv2.LINE_AA)


def draw_camera_overlay(rgb: np.ndarray, intrinsics: np.ndarray, t_env_camera: np.ndarray,
                        preview: TrajectoryPreview) -> np.ndarray:
    """Draw every arm's planned TCP path onto one camera view (RGB in, RGB out)."""
    image = np.ascontiguousarray(rgb.copy())
    scale = max(1, int(round(min(image.shape[:2]) / 360)))
    for arm, path in preview.arms.items():
        points, visible = project(path.tcp, intrinsics, t_env_camera)
        _draw_path(image, points, visible, path, preview.execute_steps,
                   ARM_COLORS_RGB.get(arm, (255, 255, 255)), scale, arm[0].upper())
    _draw_legend(image, preview, scale)
    return image


def _schematic(preview: TrajectoryPreview, arms: Mapping[str, ArmPath], axes: tuple[int, int], title: str,
               labels: tuple[str, str], size: int, span: float) -> np.ndarray:
    import cv2

    image = np.full((size, size, 3), 255, np.uint8)
    margin = 34
    points = np.concatenate([path.tcp[:, axes] for path in arms.values()])
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    origin = center - span / 2
    pixels_per_m = (size - 2 * margin) / span

    def to_pixel(value: np.ndarray) -> tuple[int, int]:
        u = margin + (value[0] - origin[0]) * pixels_per_m
        v = size - margin - (value[1] - origin[1]) * pixels_per_m
        return int(round(u)), int(round(v))

    step = next((value for value in (0.01, 0.02, 0.05, 0.1, 0.2) if value * pixels_per_m >= 40), 0.5)
    for axis in (0, 1):
        start = math.ceil(origin[axis] / step) * step
        for tick in np.arange(start, origin[axis] + span, step):
            if axis == 0:
                u = to_pixel(np.array([tick, origin[1]]))[0]
                cv2.line(image, (u, margin), (u, size - margin), (225, 225, 225), 1)
            else:
                v = to_pixel(np.array([origin[0], tick]))[1]
                cv2.line(image, (margin, v), (size - margin, v), (225, 225, 225), 1)
    for arm, path in arms.items():
        _draw_path(image, np.array([to_pixel(point) for point in path.tcp[:, axes]], dtype=float),
                   np.ones(len(path.tcp), bool), path, preview.execute_steps,
                   ARM_COLORS_RGB.get(arm, (0, 0, 0)), 1, arm[0].upper())
    bar = int(round(step * pixels_per_m))
    cv2.line(image, (size - 8 - bar, 28), (size - 8, 28), (0, 0, 0), 2)
    cv2.putText(image, f"{step * 100:g} cm", (size - 8 - bar, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, f"{title} (grid {step * 100:g} cm)", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(image, f"right: {labels[0]}   up: {labels[1]}", (8, size - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (60, 60, 60), 1, cv2.LINE_AA)
    return image


def draw_plan_schematic(preview: TrajectoryPreview, size: int = 360) -> np.ndarray:
    """Top and side views of the planned TCP paths, one zoomed row per moving arm.

    The arms stand ~0.6 m apart, so a shared scale would shrink a few
    centimetres of motion to a dot; a still arm is already stated in text.
    """
    moving = {arm: path for arm, path in preview.arms.items() if path.moves}
    groups = [{arm: path} for arm, path in moving.items()] or [dict(preview.arms)]
    rows = []
    for group in groups:
        name = f"{next(iter(group))} arm " if len(group) == 1 else ""
        points = np.concatenate([path.tcp for path in group.values()])
        # One scale for both views of a group, so lengths compare across them.
        span = max(float(np.max(np.ptp(points, axis=0))) * 1.15, 0.10)
        top = _schematic(preview, group, (0, 1), f"{name}top view", ("+x right arm side", "+y forward"), size, span)
        side = _schematic(preview, group, (1, 2), f"{name}side view", ("+y forward", "+z up"), size, span)
        rows.append(np.concatenate([top, np.full((size, 2, 3), 160, np.uint8), side], axis=1))
    divider = np.full((2, rows[0].shape[1], 3), 160, np.uint8)
    return np.concatenate([part for row in rows for part in (row, divider)][:-1], axis=0)
