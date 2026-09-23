"""RoboDojo-specific OpenPI policy configuration and wire-format helpers.

The working RoboDojo integration historically loaded Pi0.5 through XPolicy.
This module contains the small amount of policy-specific logic that is
actually required by OpenPI so Emerge can load the checkpoint directly.
It deliberately has no import-time dependency on OpenPI or Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(value):
    """Detach and move to host before converting.

    Some RoboDojo tasks return observations as CUDA tensors, which a bare
    ``np.asarray`` rejects part-way through an episode.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


ROBODOJO_TRAIN_CONFIG_NAME = "pi05_base_aloha_full_sim_arx-x5_seed_0"
ROBODOJO_ASSET_ID = "arx_x5_sim"
ROBODOJO_REPO_ID = "RoboDojo_sim_arx-x5_v30"

ARM_DIMS = (6, 6)
GRIPPER_DIMS = (1, 1)
ACTION_DIM = sum(ARM_DIMS) + sum(GRIPPER_DIMS)

_CAMERA_ALIASES = {
    "cam_high": ("cam_high", "cam_head", "head_camera", "top_camera"),
    "cam_left_wrist": (
        "cam_left_wrist",
        "left_camera",
        "left_wrist",
        "wrist_left",
    ),
    "cam_right_wrist": (
        "cam_right_wrist",
        "right_camera",
        "right_wrist",
        "wrist_right",
    ),
}

_STATE_KEYS = (
    ("left_arm_joint_state", ARM_DIMS[0]),
    ("left_ee_joint_state", GRIPPER_DIMS[0]),
    ("right_arm_joint_state", ARM_DIMS[1]),
    ("right_ee_joint_state", GRIPPER_DIMS[1]),
)


def is_robodojo_config(config_name: str) -> bool:
    """Return whether ``config_name`` selects the direct RoboDojo adapter."""

    return str(config_name).strip() == ROBODOJO_TRAIN_CONFIG_NAME


def resolve_checkpoint_dir(checkpoint_dir: str | Path) -> Path:
    """Resolve a checkpoint root or numbered parent to an inference checkpoint."""

    root = Path(checkpoint_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RoboDojo checkpoint path does not exist: {root}")

    def is_checkpoint(path: Path) -> bool:
        return (path / "params").is_dir() and (
            path / "assets" / ROBODOJO_ASSET_ID / "norm_stats.json"
        ).is_file()

    if is_checkpoint(root):
        selected = root
    else:
        candidates = [path for path in root.iterdir() if path.is_dir() and is_checkpoint(path)]
        if not candidates:
            raise FileNotFoundError(
                "RoboDojo checkpoint must contain params/ and "
                f"assets/{ROBODOJO_ASSET_ID}/norm_stats.json: {root}"
            )

        def checkpoint_order(path: Path) -> tuple[int, str]:
            digits = "".join(character for character in path.name if character.isdigit())
            return (int(digits) if digits else -1, path.name)

        selected = max(candidates, key=checkpoint_order)

    _validate_norm_stats(selected)
    return selected


def _validate_norm_stats(checkpoint_dir: Path) -> None:
    path = checkpoint_dir / "assets" / ROBODOJO_ASSET_ID / "norm_stats.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = payload["norm_stats"]
        state = stats["state"]
        actions = stats["actions"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid RoboDojo normalization stats: {path}") from exc

    for name, values in (("state", state), ("actions", actions)):
        mean = values.get("mean") if isinstance(values, Mapping) else None
        if not isinstance(mean, list) or len(mean) != ACTION_DIM:
            size = len(mean) if isinstance(mean, list) else "missing"
            raise ValueError(
                f"RoboDojo {name} norm stats must have {ACTION_DIM} dimensions, got {size}: {path}"
            )


def build_train_config(checkpoint_dir: str | Path) -> Any:
    """Build the exact OpenPI TrainConfig used by the RoboDojo Pi0.5 run.

    Imports stay lazy so the codec and its tests work in Emerge's light
    runtime. The policy-server environment supplies OpenPI/JAX.
    """

    resolved = resolve_checkpoint_dir(checkpoint_dir)

    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training import config

    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.cam_high",
                        "cam_left_wrist": "observation.images.cam_left_wrist",
                        "cam_right_wrist": "observation.images.cam_right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                }
            )
        ]
    )
    return config.TrainConfig(
        name=ROBODOJO_TRAIN_CONFIG_NAME,
        model=pi0_config.Pi0Config(pi05=True),
        data=config.LeRobotAlohaDataConfig(
            repo_id=ROBODOJO_REPO_ID,
            assets=config.AssetsConfig(
                assets_dir=str(resolved / "assets"),
                asset_id=ROBODOJO_ASSET_ID,
            ),
            # RoboDojo's ARX-X5 tensors are already in their native joint and
            # gripper convention. Applying the standard ALOHA conversion here
            # silently changes both observations and actions.
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            repack_transforms=repack,
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        seed=0,
    )


def create_robodojo_policy(checkpoint_dir: str | Path) -> Any:
    """Load the RoboDojo Pi0.5 checkpoint as a native OpenPI policy."""

    resolved = resolve_checkpoint_dir(checkpoint_dir)
    train_config = build_train_config(resolved)
    from openpi.policies import policy_config

    return policy_config.create_trained_policy(train_config, str(resolved))


def encode_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one raw RoboDojo observation to the OpenPI inference contract."""

    images_source = observation.get("images")
    if not isinstance(images_source, Mapping):
        images_source = observation.get("vision")
    if not isinstance(images_source, Mapping):
        raise KeyError("RoboDojo observation must contain images or vision")

    images = {
        target: _ensure_chw_uint8(_find_image(images_source, aliases))
        for target, aliases in _CAMERA_ALIASES.items()
    }

    raw_state = observation.get("state")
    if isinstance(raw_state, Mapping):
        state_parts = []
        for key, dimension in _STATE_KEYS:
            if key not in raw_state:
                raise KeyError(f"RoboDojo observation state is missing {key!r}")
            value = _to_numpy(raw_state[key]).astype(np.float32).reshape(-1)
            if value.shape != (dimension,):
                raise ValueError(
                    f"RoboDojo state {key!r} must have shape ({dimension},), got {value.shape}"
                )
            state_parts.append(value)
        state = np.concatenate(state_parts).astype(np.float32, copy=False)
    else:
        state = _to_numpy(raw_state).astype(np.float32).reshape(-1)
        if state.shape != (ACTION_DIM,):
            raise ValueError(
                f"RoboDojo packed state must have shape ({ACTION_DIM},), got {state.shape}"
            )

    prompt = str(observation.get("instruction", observation.get("prompt", ""))).strip()
    if not prompt:
        raise ValueError("RoboDojo observation must contain instruction or prompt")
    return {"images": images, "state": state, "prompt": prompt}


def unpack_joint_actions(actions: Any) -> list[dict[str, np.ndarray]]:
    """Unpack an OpenPI action chunk into RoboDojo dual-arm joint actions."""

    array = _to_numpy(actions).astype(np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != ACTION_DIM:
        raise ValueError(
            f"RoboDojo actions must have shape (chunk, {ACTION_DIM}), got {array.shape}"
        )

    unpacked: list[dict[str, np.ndarray]] = []
    for row in array:
        offset = 0
        action: dict[str, np.ndarray] = {}
        for key, dimension in _STATE_KEYS:
            action[key] = row[offset : offset + dimension].copy()
            offset += dimension
        unpacked.append(action)
    return unpacked


def _find_image(source: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias not in source:
            continue
        image = source[alias]
        if isinstance(image, Mapping):
            for key in ("color", "rgb"):
                if key in image:
                    return image[key]
        else:
            return image
    raise KeyError(f"RoboDojo observation is missing camera aliases {aliases}")


def _ensure_chw_uint8(image: Any) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim != 3:
        raise ValueError(f"RoboDojo image must be 3D, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = array.astype(np.uint8)

    if array.shape[-1] in (1, 3):
        chw = np.transpose(array, (2, 0, 1))
    elif array.shape[0] in (1, 3):
        chw = array
    else:
        raise ValueError(f"Unsupported RoboDojo image shape: {array.shape}")
    return np.ascontiguousarray(chw)
