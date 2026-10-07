"""Cosmos3 framework adapter with auditable subprocess execution."""
from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .protocol import ActionCandidate, RolloutRequest, RolloutResult

# Action row widths each Emerge control domain may hand to Cosmos.
ACTION_WIDTHS: dict[str, tuple[int, ...]] = {
    "libero": (7, 10),
    # RoboDojo dual ARX-X5 joint targets: [left joints(6), left gripper, right joints(6), right gripper].
    "robodojo_joint": (14,),
    # RoboDojo end-effector waypoints: per arm [xyz, quaternion xyzw, gripper], left then right.
    "robodojo_ee": (16,),
}


@dataclass(frozen=True, slots=True)
class CosmosFrameworkConfig:
    python: str
    checkpoint: str = "Cosmos3-Edge"
    config_file: str | None = None
    inference_module: str = "cosmos_framework.scripts.inference"
    parallelism_preset: str = "latency"
    seed: int = 0
    timeout_s: float = 1800.0
    no_guardrails: bool = True
    action_horizon: int = 32
    fps: int = 20
    # Emerge domain -> domain name registered in the Cosmos checkpoint.
    domain_names: Mapping[str, str] = field(default_factory=dict)
    view_point: str | None = None
    cuda_visible_devices: str | None = None


def parse_domain_map(raw: str) -> dict[str, str]:
    """Parse ``emerge_domain=cosmos_domain`` pairs separated by commas."""
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        source, sep, target = item.partition("=")
        if not sep or not source.strip() or not target.strip():
            raise ValueError(f"COSMOS_DOMAIN_MAP entries must look like source=target, got {item!r}")
        mapping[source.strip()] = target.strip()
    return mapping


def config_from_env(env: Mapping[str, str]) -> CosmosFrameworkConfig:
    python = env.get("COSMOS_PYTHON", "").strip()
    checkpoint = env.get("COSMOS_CHECKPOINT", "").strip()
    if not python or not checkpoint:
        raise RuntimeError("AC-WM cosmos rollout requires COSMOS_PYTHON and COSMOS_CHECKPOINT")
    if not Path(python).is_file():
        raise FileNotFoundError(f"COSMOS_PYTHON does not exist: {python}")
    return CosmosFrameworkConfig(
        python=python,
        checkpoint=checkpoint,
        config_file=env.get("COSMOS_CONFIG", "").strip() or None,
        inference_module=env.get("COSMOS_INFERENCE_MODULE", "").strip()
        or CosmosFrameworkConfig.inference_module,
        parallelism_preset=env.get("COSMOS_PARALLELISM_PRESET", "").strip()
        or CosmosFrameworkConfig.parallelism_preset,
        seed=int(env.get("COSMOS_SEED", "0")),
        timeout_s=float(env.get("COSMOS_TIMEOUT_S", "1800")),
        action_horizon=int(env.get("COSMOS_ACTION_HORIZON", "32")),
        fps=int(env.get("COSMOS_FPS", "20")),
        domain_names=parse_domain_map(env.get("COSMOS_DOMAIN_MAP", "")),
        view_point=env.get("COSMOS_VIEW_POINT", "").strip() or None,
        cuda_visible_devices=env.get("COSMOS_CUDA_VISIBLE_DEVICES", "").strip() or None,
    )


class CosmosFrameworkAdapter:
    def __init__(self, config: CosmosFrameworkConfig) -> None:
        self.config = config

    def write_input_spec(self, request: RolloutRequest, candidate: ActionCandidate, path: str) -> str:
        widths = ACTION_WIDTHS.get(request.domain_name)
        if widths is None:
            raise ValueError(f"unsupported action domain: {request.domain_name}")
        width = len(candidate.actions[0])
        if width not in widths:
            raise ValueError(f"{request.domain_name} action width must be one of {widths}, got {width}")
        out = Path(path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if self.config.action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        raw_actions = tuple(tuple(float(v) for v in row) for row in candidate.actions)
        # Cosmos needs a temporal action window. Preserve the proposed prefix
        # and hold its final control for the remaining horizon instead of
        # silently producing an almost-static one-step prediction.
        horizon = max(len(raw_actions), self.config.action_horizon)
        actions = raw_actions + (raw_actions[-1],) * (horizon - len(raw_actions))
        action_path = out.with_suffix(".actions.json")
        action_path.write_text(json.dumps([list(row) for row in actions], indent=2) + "\n")
        payload = {
            "model_mode": "forward_dynamics",
            "name": candidate.candidate_id,
            "domain_name": self.config.domain_names.get(request.domain_name, request.domain_name),
            "vision_path": str(Path(request.observation_path).resolve()),
            "action_path": str(action_path),
            "action_chunk_size": len(actions),
            "prompt": request.task,
            "fps": self.config.fps,
        }
        view_point = request.view_point or self.config.view_point
        if view_point is not None:
            payload["view_point"] = view_point
        out.write_text(json.dumps(payload, indent=2) + "\n")
        return str(out)

    def command(self, input_spec: str, output_dir: str) -> tuple[str, ...]:
        args = [self.config.python, "-m", self.config.inference_module]
        if self.config.no_guardrails:
            args.append("--no-guardrails")
        if self.config.config_file:
            args += ["--config-file", self.config.config_file]
        args += ["--parallelism-preset", self.config.parallelism_preset, "-i", input_spec,
                 "-o", output_dir, "--checkpoint-path", self.config.checkpoint,
                 "--seed", str(self.config.seed)]
        return tuple(args)

    def environment(self) -> dict[str, str]:
        env = os.environ.copy()
        if self.config.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.config.cuda_visible_devices
        # The official Cosmos launch environment needs bundled PyAV/OpenCV and
        # NVIDIA wheel libraries visible to torchcodec and CUDA extensions.
        # Keep venv symlinks unresolved so the prefix is the environment itself.
        prefix = Path(self.config.python).absolute().parent.parent
        lib_dirs = [prefix / "lib"]
        for site_packages in sorted(prefix.glob("lib/python3*/site-packages")):
            lib_dirs += [site_packages / "av.libs", site_packages / "opencv_python.libs",
                         site_packages / "nvidia/npp/lib"]
        lib_dirs.append(Path("/usr/local/cuda/targets/x86_64-linux/lib"))
        existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
        env["LD_LIBRARY_PATH"] = ":".join([str(p) for p in lib_dirs if p.exists()] + existing)
        return env

    def run(self, request: RolloutRequest, candidate: ActionCandidate, work_dir: str) -> RolloutResult:
        root = Path(work_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        spec = self.write_input_spec(request, candidate, str(root / f"{candidate.candidate_id}.json"))
        output = root / candidate.candidate_id
        output.mkdir(exist_ok=True)
        try:
            p = subprocess.run(
                self.command(spec, str(output)), cwd=str(root), env=self.environment(),
                text=True, capture_output=True, timeout=self.config.timeout_s,
            )
        except subprocess.TimeoutExpired as e:
            (root / f"{candidate.candidate_id}.stderr").write_text(e.stderr if isinstance(e.stderr, str) else "")
            return RolloutResult(candidate.candidate_id, "timeout", metadata={"output_dir": str(output)},
                                 error=f"timeout after {self.config.timeout_s}s")
        (root / f"{candidate.candidate_id}.stdout").write_text(p.stdout)
        (root / f"{candidate.candidate_id}.stderr").write_text(p.stderr)
        if p.returncode != 0:
            return RolloutResult(candidate.candidate_id, "failed",
                                 metadata={"returncode": p.returncode, "output_dir": str(output)},
                                 error=p.stderr[-4000:])
        videos = list(output.rglob("vision.mp4"))
        return RolloutResult(
            candidate.candidate_id, "success" if videos else "failed", str(videos[0]) if videos else None,
            {"returncode": 0, "output_dir": str(output), "raw_action_chunk_size": len(candidate.actions),
             "action_horizon": max(len(candidate.actions), self.config.action_horizon),
             "prediction": "cosmos"},
            None if videos else "missing vision.mp4",
        )

    def failed(self, candidate: ActionCandidate, error: str) -> RolloutResult:
        return RolloutResult(candidate.candidate_id, "failed", error=error)
