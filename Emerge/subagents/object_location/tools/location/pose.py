"""Convert server responses into precise object poses."""

from __future__ import annotations

from typing import Any

import numpy as np

from Emerge.subagents.object_location.tools.location.object_pointcloud import (
    merge_masked_pointclouds,
)
from external_model_server.localization_types import (
    TargetMask,
    TargetMaskResult,
    VGGTAlignedCamera,
    VGGTAlignment,
    VGGTResult,
    VGGTViewPrediction,
)


def estimate_object_locations(
    geometry_payload: dict[str, Any],
    masks_payload: dict[str, Any],
    targets: list[dict[str, str]],
    *,
    point_conf_threshold: float,
    min_points: int,
    bbox_padding_pixels: int,
    view_center_tolerance_m: float,
    ray_consensus_tolerance_m: float,
    min_consistent_views: int,
    coordinate_frame: str = "world",
    max_localization_distance_m: float | None = None,
) -> list[dict[str, Any]]:
    """Estimate object poses from standalone model-server results."""
    from Emerge.subagents.object_location.tools.location.object_pose_estimator import (
        ObjectPoseEstimator,
    )
    from Emerge.subagents.object_location.tools.location.object_pose_refiner import (
        refine_object_pose_with_multiview_bboxes,
    )

    geometry = _deserialize_geometry(geometry_payload)
    if not coordinate_frame.strip():
        raise ValueError("coordinate_frame must not be empty")
    if max_localization_distance_m is not None and (
        not np.isfinite(max_localization_distance_m) or max_localization_distance_m <= 0
    ):
        raise ValueError("max_localization_distance_m must be finite and positive")
    masks = _deserialize_masks(masks_payload)
    object_keys = [target["object_key"] for target in targets]
    pointclouds = merge_masked_pointclouds(
        geometry,
        masks,
        target_objects=object_keys,
        point_conf_threshold=point_conf_threshold,
        view_center_tolerance_m=view_center_tolerance_m,
        ray_consensus_tolerance_m=ray_consensus_tolerance_m,
        min_consistent_views=min_consistent_views,
    )
    estimator = ObjectPoseEstimator({"min_points": min_points})

    objects = []
    for target in targets:
        object_key = target["object_key"]
        pointcloud = pointclouds[object_key]
        points_world = np.asarray(
            pointcloud["points_world"],
            dtype=np.float32,
        )
        base = {
            "name": object_key,
            "sam_prompt": target["prompt"],
            **_mask_summary(masks, object_key),
            "view_selection": pointcloud["view_selection"],
            "num_points": int(len(points_world)),
        }
        if len(points_world) < min_points:
            if pointcloud["view_selection"]["candidate_views"]:
                reason = "insufficient_consistent_views"
            else:
                reason = "target_not_segmented"
            objects.append(
                {
                    **base,
                    "found": False,
                    "failure_reason": reason,
                }
            )
            continue

        try:
            pose = estimator.estimate(object_key, points_world)
            selected_views = set(
                pointcloud["view_selection"]["selected_views"]
            )
            selected_masks = {
                view_name: result
                for view_name, result in masks.items()
                if view_name in selected_views
            }
            pose = refine_object_pose_with_multiview_bboxes(
                pose,
                geometry=geometry,
                masks=selected_masks,
                config={"padding_pixels": bbox_padding_pixels},
            )
            _validate_pose_geometry(
                pose, geometry=geometry, selected_views=selected_views,
                max_distance_m=max_localization_distance_m,
            )
        except RuntimeError as error:
            objects.append(
                {
                    **base,
                    "found": False,
                    "failure_reason": str(error),
                }
            )
            continue

        corners = np.asarray(pose.bbox_3d_corners, dtype=np.float64)
        objects.append(
            {
                **base,
                "found": True,
                "frame": coordinate_frame,
                "position_type": "object_bbox_center",
                "position_m": np.asarray(pose.center, dtype=float).tolist(),
                "size_m": np.asarray(pose.extent, dtype=float).tolist(),
                "rotation_matrix": np.asarray(
                    pose.rotation_matrix,
                    dtype=float,
                ).tolist(),
                "rpy_rad": np.asarray(pose.rpy, dtype=float).tolist(),
                "bbox_world_min_m": corners.min(axis=0).astype(float).tolist(),
                "bbox_world_max_m": corners.max(axis=0).astype(float).tolist(),
                "num_inliers": int(pose.num_inliers),
                "pose_inlier_ratio": float(pose.pose_inlier_ratio),
            }
        )
    return objects


def _validate_pose_geometry(pose, *, geometry, selected_views, max_distance_m):
    """Reject invalid or out-of-range estimates; never clamp them into success."""
    arrays = (pose.center, pose.extent, pose.rotation_matrix, pose.bbox_3d_corners)
    if not all(np.isfinite(np.asarray(value)).all() for value in arrays):
        raise RuntimeError("non_finite_pose_geometry")
    if np.any(np.asarray(pose.extent) <= 0):
        raise RuntimeError("invalid_pose_extent")
    if max_distance_m is None:
        return
    cameras = [c for c in geometry.cameras if c.name in selected_views]
    if not cameras:
        raise RuntimeError("missing_selected_camera_geometry")
    for camera in cameras:
        transform = np.asarray(camera.T_world_camera_observed, dtype=float)
        camera_center = transform[:3, 3]
        # Calibrated camera coordinates use +Z forward (ROS optical convention).
        center_camera = transform[:3, :3].T @ (np.asarray(pose.center) - camera_center)
        if not np.isfinite(center_camera).all() or center_camera[2] <= 0:
            raise RuntimeError("pose_behind_selected_camera")
        distance = np.linalg.norm(np.asarray(pose.center) - camera_center)
        if distance > max_distance_m or np.max(pose.extent) > 2 * max_distance_m:
            raise RuntimeError("pose_outside_observation_range")


def _deserialize_geometry(payload: dict[str, Any]) -> VGGTResult:
    alignment = VGGTAlignment(**payload["alignment"])
    views = [
        VGGTViewPrediction(
            name=str(item["name"]),
            rgb=np.asarray(item["rgb"], dtype=np.uint8),
            point_map_world=np.asarray(
                item["point_map_world"],
                dtype=np.float32,
            ),
            depth_m=np.asarray(item["depth_m"], dtype=np.float32),
            depth_conf=_optional_array(item.get("depth_conf")),
            point_conf=_optional_array(item.get("point_conf")),
        )
        for item in payload["views"]
    ]
    cameras = [
        VGGTAlignedCamera(
            name=str(item["name"]),
            intrinsics_pred=np.asarray(
                item["intrinsics_pred"],
                dtype=np.float64,
            ),
            intrinsics_observed=np.asarray(
                item["intrinsics_observed"],
                dtype=np.float64,
            ),
            extrinsics_pred=np.asarray(
                item["extrinsics_pred"],
                dtype=np.float64,
            ),
            T_world_camera_observed=np.asarray(
                item["T_world_camera_observed"],
                dtype=np.float64,
            ),
        )
        for item in payload["cameras"]
    ]
    reference_view = str(payload["reference_view"])
    reference = next(view for view in views if view.name == reference_view)
    return VGGTResult(
        reference_view=reference_view,
        point_map_world=reference.point_map_world,
        depth_m=reference.depth_m,
        depth_conf=reference.depth_conf,
        point_conf=reference.point_conf,
        cameras=cameras,
        alignment=alignment,
        views=views,
    )


def _deserialize_masks(
    payload: dict[str, Any],
) -> dict[str, TargetMaskResult]:
    results = {}
    for view in payload["views"]:
        result = TargetMaskResult(
            view_name=str(view["view_name"]),
            image_shape=tuple(int(value) for value in view["image_shape"]),
            targets=[
                TargetMask(
                    object_key=str(item["object_key"]),
                    prompt=str(item["prompt"]),
                    found=bool(item["found"]),
                    confidence=float(item["confidence"]),
                    mask=(
                        np.asarray(item["mask"], dtype=bool)
                        if item["mask"] is not None
                        else None
                    ),
                    bbox_2d=(
                        [int(value) for value in item["bbox_2d"]]
                        if item["bbox_2d"] is not None
                        else None
                    ),
                    area_pixels=int(item["area_pixels"]),
                    instance_index=int(item.get("instance_index", 1)),
                )
                for item in view["targets"]
            ],
        )
        results[result.view_name] = result
    return results


def _mask_summary(
    masks: dict[str, TargetMaskResult],
    object_key: str,
) -> dict[str, Any]:
    visible = []
    confidences = []
    for view_name, result in masks.items():
        target = next(
            item
            for item in result.targets
            if item.object_key == object_key
        )
        if target.found and target.mask is not None:
            visible.append(view_name)
            confidences.append(float(target.confidence))
    return {
        "sam3_confidence": max(confidences, default=0.0),
        "visible_views": visible,
    }


def _optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value, dtype=np.float32)
