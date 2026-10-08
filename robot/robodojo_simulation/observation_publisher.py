"""Publish RoboDojo's cameras as Emerge's calibrated observation.

``workspace/artifacts/observations/observation.json`` is the whole
environment-to-perception interface: the task-verification subagent reads the
images out of it, the object-location subagent additionally needs the
calibration, and the visual monitor keys its dedupe on the manifest revision.
An embodiment that never writes it leaves all three inert, and the agent acts
without ever seeing the scene.

The manifest itself is produced by the shared
:class:`~robot.mujoco_simulation.calibrated_observation.CalibratedObservationWriter`
so the format cannot drift from the LIBERO path; this module only adapts
RoboDojo's camera accessors to the interface that writer expects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _rotation_from_wxyz(quaternion: Any) -> np.ndarray:
    quat = _numpy(quaternion).astype(np.float64).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"camera quaternion must contain 4 values, got {quat.shape}")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("camera quaternion must be finite and non-zero")
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class AcWmView:
    """One full-resolution camera frame with its calibration, if known."""

    name: str
    rgb: np.ndarray
    intrinsics: np.ndarray | None
    t_env_camera: np.ndarray | None


def write_ac_wm_snapshot(
    destination: Path,
    views: Sequence[AcWmView],
    *,
    revision: int,
    annotated: Mapping[str, np.ndarray] | None = None,
    panels: Sequence[tuple[str, np.ndarray]] = (),
    fps: float = 20.0,
) -> dict[str, Any]:
    """Write the frames AC-WM judges a proposal against.

    The reference (first) view becomes a two-frame video, the world model's
    vision input. Raw views are kept for audit; ``annotated`` views (the
    proposal drawn on a camera) and ``panels`` are what the judge sees.
    """
    import cv2

    def save(name: str, rgb: np.ndarray) -> str:
        path = destination / f"{name}.png"
        if not cv2.imwrite(str(path), np.ascontiguousarray(rgb[..., ::-1])):
            raise RuntimeError(f"could not write AC-WM observation frame {path}")
        return str(path)

    if not views:
        raise RuntimeError("AC-WM snapshot requires at least one camera view")
    destination.mkdir(parents=True, exist_ok=True)
    images = [save(view.name, view.rgb) for view in views]
    annotated = annotated or {}
    preview_images = [save(f"{view.name}_plan", annotated[view.name]) for view in views if view.name in annotated]
    preview_images += [save(name, rgb) for name, rgb in panels]
    reference = views[0].rgb
    video = destination / f"{views[0].name}.mp4"
    height, width = reference.shape[:2]
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open AC-WM observation video writer: {video}")
    try:
        bgr = np.ascontiguousarray(reference[..., ::-1])
        writer.write(bgr)
        writer.write(bgr)
    finally:
        writer.release()
    snapshot = {"observation_path": str(video), "observation_images": images, "observation_revision": revision}
    if preview_images:
        snapshot["preview_images"] = preview_images
    return snapshot


@dataclass(frozen=True)
class _ObservationOutput:
    enabled: bool
    reference: bool
    directory: Path


class _CameraView:
    """One RoboDojo camera behind the accessors the shared writer calls.

    Updated in place rather than rebuilt, so a single writer survives the whole
    episode and keeps incrementing its revision.
    """

    def __init__(
        self,
        name: str,
        rgb: np.ndarray,
        intrinsics: np.ndarray,
        transform: np.ndarray,
        output: _ObservationOutput,
    ) -> None:
        self._name = name
        self._rgb = rgb
        self._intrinsics = intrinsics
        self._transform = transform
        self._output = output

    def update(
        self,
        rgb: np.ndarray,
        intrinsics: np.ndarray,
        transform: np.ndarray,
    ) -> None:
        self._rgb = rgb
        self._intrinsics = intrinsics
        self._transform = transform

    def get_name(self) -> str:
        return self._name

    def get_rgb(self) -> np.ndarray:
        return self._rgb

    def get_width(self) -> int:
        return int(self._rgb.shape[1])

    def get_height(self) -> int:
        return int(self._rgb.shape[0])

    def get_intrinsics(self) -> np.ndarray:
        return self._intrinsics

    def get_world_camera_transform(self) -> np.ndarray:
        return self._transform

    def get_observation_output(self) -> _ObservationOutput:
        return self._output


class RoboDojoObservationPublisher:
    """Write RoboDojo's camera observation into the agent workspace."""

    def __init__(
        self,
        environment: Any,
        *,
        workspace: str | Path,
        camera_names: Sequence[str] | None = None,
        reference_view: str | None = None,
        archive: bool = False,
    ) -> None:
        self._environment = environment
        self._workspace = Path(workspace).expanduser().resolve()
        self._requested = tuple(camera_names or _DEFAULT_CAMERAS)
        self._reference = reference_view or self._requested[0]
        if self._reference not in self._requested:
            raise ValueError(
                f"reference view {self._reference!r} is not among {self._requested}"
            )
        self._directory = self._workspace / "artifacts/cameras"
        self._views: dict[str, _CameraView] = {}
        self._writer: Any | None = None
        self._archive = bool(archive)
        self._archive_root = self._workspace / "artifacts/observations/history"
        # Demonstration tasks need chronology, not merely the latest live
        # observation.  These snapshots are deliberately separate from the
        # live manifest: publishing them must not wake the visual action
        # monitor while the controlled arms are correctly holding still.
        self._temporal_root = self._workspace / "artifacts/observations/temporal"
        self._temporal_sequence = 0
        self._revision = 0
        # Full-resolution frames of the latest capture, before the VGGT resize.
        self._latest_frames: dict[str, np.ndarray] = {}

    @property
    def revision(self) -> int:
        """Revision of the latest live manifest written by this publisher."""
        return self._revision

    def reference_camera_origin_m(self) -> np.ndarray | None:
        """Return the latest calibrated reference-camera origin.

        Motion guards occasionally need to check that two visually asserted
        tool endpoints really form a segment through the public gripper.  The
        camera origin plus already localized points is enough for that
        projective check; RGB, depth, simulator objects and reward state remain
        outside the action controller.
        """

        view = self._views.get(self._reference)
        if view is None:
            return None
        transform = np.asarray(
            view.get_world_camera_transform(), dtype=np.float64
        )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            return None
        return transform[:3, 3].copy()

    def publish(self, observation: Mapping[str, Any] | None = None) -> Path | None:
        """Capture every camera and rewrite the manifest.

        Pass ``observation`` when the caller has already fetched one. RoboDojo's
        ``get_obs()`` renders and appends a frame to the per-camera video
        writers, so re-fetching inside a rollout would both cost a render and
        stretch the official episode videos.

        Returns ``None`` when the cameras cannot be read: a stale manifest is
        worse than none, because the visual monitor would keep judging a scene
        the robot has already left.
        """
        from robot.mujoco_simulation.calibrated_observation import (
            CalibratedObservationWriter,
        )

        views = self._capture(observation)
        if not views:
            return None
        if self._writer is None:
            # One writer for the episode: the visual monitor dedupes on the
            # manifest revision, and a fresh writer would restart it at 1 and
            # make every new scene look like the one already judged.
            self._writer = CalibratedObservationWriter(
                self._workspace, views, coordinate_frame="robodojo_env",
                max_localization_distance_m=3.0,
            )
        manifest_path = self._writer.write()
        self._revision += 1
        if self._archive:
            self._archive_revision(manifest_path)
        return manifest_path

    def ac_wm_views(self, observation: Mapping[str, Any] | None = None) -> list[AcWmView]:
        """Return the frames AC-WM evaluates a proposal against, reference first.

        Without ``observation`` the frames of the latest publish are reused, so
        a rule-skill preview costs no render and adds no official video frame.
        Calibration is read live; the scene has not stepped since the capture.
        """
        frames = self._vision_frames(observation) if observation is not None else dict(self._latest_frames)
        if self._reference not in frames:
            raise RuntimeError(f"AC-WM requires the reference camera {self._reference!r}")
        ordered = [self._reference] + [name for name in self._requested if name != self._reference]
        views = []
        for name in ordered:
            if name not in frames:
                continue
            calibration = self._camera_calibration(name)
            intrinsics, transform = calibration if calibration is not None else (None, None)
            views.append(AcWmView(name, frames[name], intrinsics, transform))
        return views

    def write_ac_wm_snapshot(
        self,
        views: Sequence[AcWmView],
        *,
        annotated: Mapping[str, np.ndarray] | None = None,
        panels: Sequence[tuple[str, np.ndarray]] = (),
    ) -> dict[str, Any]:
        import time

        destination = self._workspace / "artifacts/ac-wm/observations" / str(time.time_ns())
        return write_ac_wm_snapshot(destination, views, revision=self._revision,
                                    annotated=annotated, panels=panels)

    def _camera_calibration(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Full-resolution intrinsics and ``T_env_camera`` (ROS axes) of one camera."""
        camera_manager = getattr(self._environment, "camera_manager", None)
        if camera_manager is None:
            return None
        available = list(camera_manager.camera_names[0])
        if name not in available:
            return None
        camera = camera_manager.cameras[0][available.index(name)]
        intrinsics = _numpy(camera.get_intrinsics_matrix(device="cpu")).astype(np.float64)
        position, quaternion_wxyz = camera.get_world_pose(camera_axes="ros")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _rotation_from_wxyz(quaternion_wxyz)
        # Reported in the same robodojo_env frame as every pose the agent
        # commands, so the two are directly comparable.
        transform[:3, 3] = _numpy(position).astype(np.float64).reshape(3) - self._environment_origin()
        return intrinsics, transform

    def _vision_frames(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        vision = observation.get("vision") if isinstance(observation, Mapping) else None
        if not isinstance(vision, Mapping):
            return {}
        frames: dict[str, np.ndarray] = {}
        for name in self._requested:
            payload = vision.get(name)
            if not isinstance(payload, Mapping) or "color" not in payload:
                continue
            rgb = _numpy(payload["color"])
            if rgb.ndim != 3 or rgb.shape[-1] < 3:
                continue
            frames[name] = np.ascontiguousarray(rgb[..., :3].astype(np.uint8, copy=False))
        return frames

    def publish_temporal(
        self,
        observation: Mapping[str, Any] | None = None,
        *,
        control_step: int,
    ) -> Path | None:
        """Archive one calibrated frame without changing the live manifest.

        ``imitate_sorting_sequence`` advances its opponent only while the
        controlled robot steps.  The ordinary driver publishes after an action,
        which collapses a long wait into one final image and destroys the five
        placement transitions the task asks the model to remember.  This path
        stores sparse, ordered camera observations during that wait.  It never
        reads simulator object state or reward state.
        """
        import json
        import os

        import cv2

        views = self._capture(observation)
        if not views:
            return None

        self._temporal_sequence += 1
        destination = self._temporal_root / f"{self._temporal_sequence:04d}"
        destination.mkdir(parents=True, exist_ok=True)
        manifest_views = []
        for view in views:
            image_path = destination / f"{view.get_name()}.png"
            encoded_ok, encoded = cv2.imencode(
                ".png",
                np.ascontiguousarray(view.get_rgb()[..., ::-1]),
            )
            if not encoded_ok:
                raise RuntimeError(
                    f"could not encode temporal camera {view.get_name()}"
                )
            temporary_image = image_path.with_suffix(".png.tmp")
            temporary_image.write_bytes(encoded.tobytes())
            os.replace(temporary_image, image_path)
            manifest_views.append(
                {
                    "name": view.get_name(),
                    "image_path": str(image_path.relative_to(self._workspace)),
                    "width": view.get_width(),
                    "height": view.get_height(),
                    "intrinsics": view.get_intrinsics().tolist(),
                    "T_world_camera": view.get_world_camera_transform().tolist(),
                }
            )

        payload = {
            "coordinate_frame": "robodojo_env",
            "max_localization_distance_m": 3.0,
            "revision": self._temporal_sequence,
            "control_step": int(control_step),
            "reference_view": self._reference,
            "views": manifest_views,
        }
        manifest_path = destination / "observation.json"
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, manifest_path)
        return manifest_path

    def _archive_revision(self, manifest_path: Path) -> None:
        """Keep a copy of what the agent saw.

        The live manifest and its images are overwritten on every publish, so
        without this a finished episode carries no visual history and there is
        no way to ask afterwards what the agent was looking at when it decided.
        """
        import json
        import shutil

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            destination = self._archive_root / f"{int(manifest['revision']):04d}"
            destination.mkdir(parents=True, exist_ok=True)
            for view in manifest["views"]:
                source = self._workspace / view["image_path"]
                if source.exists():
                    shutil.copy2(source, destination / f"{view['name']}.png")
            (destination / "observation.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except (OSError, KeyError, ValueError) as exc:
            print(f"[robodojo] observation archive failed: {exc}", flush=True)

    def _capture(self, observation: Mapping[str, Any] | None) -> list[_CameraView]:
        if observation is None:
            observation = self._environment.get_obs()
        vision = observation.get("vision") if isinstance(observation, Mapping) else None
        if not isinstance(vision, Mapping):
            return []
        camera_manager = getattr(self._environment, "camera_manager", None)
        if camera_manager is None:
            return []
        available = list(camera_manager.camera_names[0])

        views: list[_CameraView] = []
        frames = {
            name: rgb for name, rgb in self._vision_frames(observation).items()
            if name in available
        }
        if frames:
            self._latest_frames = frames
        for name in self._requested:
            if name not in frames:
                continue
            rgb = frames[name]
            intrinsics, transform = self._camera_calibration(name)
            # VGGT requires common H/W multiples of 14. Rescale intrinsics
            # with the exact image transform; all perception services and VLM
            # coordinates use these same saved images. Pi0.5 uses get_obs().
            from PIL import Image
            height, width = rgb.shape[:2]
            new_width, new_height = 518, 392
            scale = np.diag([new_width / width, new_height / height, 1.0])
            intrinsics = scale @ intrinsics
            rgb = np.asarray(Image.fromarray(rgb).resize(
                (new_width, new_height), Image.Resampling.BILINEAR))
            existing = self._views.get(name)
            if existing is None:
                existing = _CameraView(
                    name=name,
                    rgb=rgb,
                    intrinsics=intrinsics,
                    transform=transform,
                    output=_ObservationOutput(
                        enabled=True,
                        reference=(name == self._reference),
                        directory=self._directory / name,
                    ),
                )
                self._views[name] = existing
            else:
                existing.update(rgb, intrinsics, transform)
            views.append(existing)
        return views

    def _environment_origin(self) -> np.ndarray:
        scene_manager = getattr(self._environment, "scene_manager", None)
        origins = getattr(scene_manager, "env_origins", None)
        if origins is None:
            origins = getattr(self._environment, "env_origins", None)
        if origins is None:
            return np.zeros(3, dtype=np.float64)
        origin = _numpy(origins[0]).astype(np.float64).reshape(-1)
        if origin.size != 3:
            raise ValueError(
                f"RoboDojo environment origin must have 3 values, got {origin.shape}"
            )
        return origin
