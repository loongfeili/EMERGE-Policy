#!/usr/bin/env python3
"""Run Emerge-controlled episodes in RoboDojo's native EvalEnv.

The process owns Isaac Sim, the benchmark scene, official reward logic and
video writers. Emerge communicates with it through ACTION.md and
ROBOT_STATE.md; Pi0.5 calls go straight to Emerge's OpenPI server.  In
persistent mode one AppLauncher is reused for several layouts of the same task;
RoboDojo still closes and rebuilds the simulation scene between layouts.
"""

# ruff: noqa: E402 -- Isaac-dependent imports must follow AppLauncher creation.

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Import OpenCV before Kit to avoid loading Kit's older libstdc++ first.
import cv2  # noqa: F401
import numpy as np
from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robodojo-root",
        default=os.environ.get(
            "ROBODOJO_ROOT",
            str(Path(__file__).absolute().parents[2] / "RoboDojo"),
        ),
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--layout-id", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--policy-server-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help=(
            "Start the exact episode AppLauncher, import the selected task, "
            "then close without creating a scene or result."
        ),
    )
    parser.add_argument("--workspace", default=None)
    parser.add_argument(
        "--episode-manifest",
        default=None,
        help=(
            "JSON list of same-task episodes to run in one AppLauncher. Each "
            "entry needs layout_id and workspace."
        ),
    )
    parser.add_argument("--agent-python", default=None)
    parser.add_argument("--agent-config", default=None)
    parser.add_argument("--agent-message", default=None)
    parser.add_argument(
        "--episode-timeout-s",
        type=float,
        default=3600.0,
        help="Per-layout timeout used by persistent mode.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--motion-config", default=None)
    parser.add_argument("--agent-done-file", default=None)
    parser.add_argument(
        "--trajectory-path",
        default=None,
        help="Per-decision JSONL (default: WORKSPACE/trajectory.jsonl).",
    )
    parser.add_argument(
        "--archive-observations",
        action="store_true",
        help="Keep every published camera view, so a finished episode can be reviewed.",
    )
    parser.add_argument(
        "--status-path",
        default=None,
        help="Machine-readable episode status (default: WORKSPACE/episode_status.json).",
    )
    parser.add_argument(
        "--smoke-action-json",
        default=None,
        help="Execute one HAL action, print state, and exit without recording an eval result.",
    )
    parser.add_argument(
        "--record-every-step",
        action="store_true",
        help=(
            "Film every control step instead of one frame per policy replan. "
            "Costs a render per step; intended for diagnostic runs."
        ),
    )
    parser.add_argument(
        "--policy-baseline",
        action="store_true",
        help=(
            "Run the episode as one uninterrupted VLA rollout on the task's own "
            "instruction and full step budget, with no agent. This is the "
            "reference the agent is worth measuring against."
        ),
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if os.environ.get("ROBODOJO_UNSET_CUDA_VISIBLE_DEVICES") == "1":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        args.device = f"cuda:{args.device_id}"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
        args.device = "cuda:0"
    return args


ARGS = _parse_args()
sys.path.insert(0, str(Path(ARGS.robodojo_root).expanduser().resolve()))
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
from task.RoboDojo import task_registry as startup_task_registry
try:
    import yaml
    cfg_path = startup_task_registry.task_config_path(
        str(Path(ARGS.robodojo_root) / "task/RoboDojo/config"), ARGS.task)
    with open(cfg_path) as config_stream:
        PHYSX_MONITOR_ENABLED = bool((yaml.safe_load(config_stream) or {}).get("Articulation"))
except Exception:
    PHYSX_MONITOR_ENABLED = True
if PHYSX_MONITOR_ENABLED:
    from src.eval_client.physx_warning_monitor import get_monitor
    get_monitor().start(enabled=True)
add_zero_delay_kit_args(ARGS)
APP_LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = APP_LAUNCHER.app

# Isaac-dependent RoboDojo imports must happen after AppLauncher.
ROBODOJO_ROOT = Path(ARGS.robodojo_root).expanduser().resolve()
EMERGE_ROOT = Path(__file__).resolve().parents[1]
for import_root in (
    ROBODOJO_ROOT,
    ROBODOJO_ROOT / "XPolicyLab",
    EMERGE_ROOT,
):
    value = str(import_root)
    if value not in sys.path:
        sys.path.insert(0, value)

from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
from omegaconf import OmegaConf
from robot.robodojo_simulation.eval_bridge import create_agent_eval_env as create_eval_env
from task.RoboDojo import task_registry
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization

from robot.drivers.robodojo_driver import RoboDojoDriver
from robot.robodojo_simulation.agent_workspace import install_agent_skills
from robot.robodojo_simulation.evaluation_health import pause_on_provider_error, worker_stop_file
from robot.robodojo_simulation.controller_loop import watch_driver_loop
from robot.mujoco_simulation.scene_io import (
    default_robot_state_doc,
    save_robot_state_doc,
)
from robot.robodojo_simulation.isaacsim_compat import apply_isaacsim_compat, select_planner_device
from robot.vla.openpi_bridge import Pi05Client


def _build_config(layout_ids: list[int] | None = None) -> Any:
    layout_ids = list(layout_ids or [ARGS.layout_id])
    if not layout_ids:
        raise ValueError("at least one layout id is required")
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, f"{ARGS.env_cfg}.yml"))
    eval_cfg.update(
        {
            "task_name": ARGS.task,
            "num_envs": 1,
            "device_id": ARGS.device_id,
            "seed": ARGS.policy_seed,
            "layout_ids": layout_ids,
            "eval_num": len(layout_ids),
            "eval_batch": False,
            "policy_name": "Emerge",
            "additional_info": (
                f"agentic_layout_{layout_ids[0]}"
                if len(layout_ids) == 1
                else f"agentic_batch_{layout_ids[0]}_{layout_ids[-1]}_{os.getpid()}"
            ),
            "physx_monitor_enabled": PHYSX_MONITOR_ENABLED,
            "config_name": eval_cfg.get("config_name", ARGS.env_cfg),
        }
    )
    deploy_cfg = {
        "policy_name": "Emerge",
        "direct_agentic": True,
        "port": 1,
    }
    benchmark_path = os.path.join(ROOT_DIR, "task", BENCHMARK)
    config = OmegaConf.create(
        {
            "sim": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "sim",
                    eval_cfg["config"]["sim"] + ".yml",
                )
            ),
            "scene": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "scene",
                    eval_cfg["config"]["scene"] + ".yml",
                )
            ),
            "camera": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "camera",
                    eval_cfg["config"]["camera"] + ".yml",
                )
            ),
            "robot": load_yaml(
                os.path.join(
                    ENV_CONFIG_PATH,
                    "robot",
                    eval_cfg["config"]["robot"] + ".yml",
                )
            ),
            "task_env": load_yaml(
                task_registry.task_config_path(
                    os.path.join(benchmark_path, "config"), ARGS.task
                )
            ),
            "eval_cfg": eval_cfg,
            "deploy_cfg": deploy_cfg,
        }
    )
    OmegaConf.update(config, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(config, "eval_cfg.num_envs", 1, force_add=True)
    OmegaConf.update(config, "eval_cfg.eval_num", len(layout_ids), force_add=True)
    OmegaConf.update(config, "sim.seed", [layout_ids[0]], force_add=True)
    OmegaConf.update(
        config,
        "camera.default_frequency",
        eval_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )
    config = process_randomization(config)
    config, _ = process_config(config, task_name=ARGS.task)
    OmegaConf.update(config, "eval_cfg.eval_num", len(layout_ids), force_add=True)
    return config


def _load_motion_config() -> dict[str, Any]:
    if ARGS.motion_config is None:
        return {}
    # The launcher runs this from the RoboDojo checkout so that Kit finds its
    # own assets, so a relative path -- which is how every documented invocation
    # writes it -- resolves against the wrong root. Falling back to the
    # Emerge tree costs nothing and saves a six-minute Isaac start that
    # ends in FileNotFoundError after the scene has finished loading.
    path = Path(ARGS.motion_config).expanduser()
    if not path.is_absolute() and not path.exists():
        path = EMERGE_ROOT / path
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"motion config must contain a JSON object: {path}")
    if ARGS.record_every_step:
        payload["record_every_step"] = True
    return payload


def _workspace() -> Path:
    if ARGS.workspace:
        return Path(ARGS.workspace).expanduser().resolve()
    return (
        EMERGE_ROOT
        / "artifacts"
        / "robodojo_agent"
        / f"{ARGS.task}-layout-{ARGS.layout_id}"
        / "workspace"
    )


def _prepare_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    template_root = EMERGE_ROOT / "Emerge" / "templates"
    for source in template_root.glob("*.md"):
        destination = path / source.name
        if not destination.exists():
            shutil.copy2(source, destination)
    memory_dir = path / "memory"
    memory_dir.mkdir(exist_ok=True)
    memory_template = template_root / "memory" / "MEMORY.md"
    if memory_template.exists() and not (memory_dir / "MEMORY.md").exists():
        shutil.copy2(memory_template, memory_dir / "MEMORY.md")
    install_agent_skills(path, EMERGE_ROOT / "Emerge" / "skills")
    environment = default_robot_state_doc()
    environment["task"] = {
        "benchmark": "RoboDojo",
        "name": ARGS.task,
        "layout_id": ARGS.layout_id,
        "policy_seed": ARGS.policy_seed,
    }
    save_robot_state_doc(path / "ROBOT_STATE.md", environment)
    (path / "ACTION.md").write_text("", encoding="utf-8")


def _output_path(raw: str | None, workspace: Path, default_name: str) -> Path:
    return Path(raw).expanduser().resolve() if raw else workspace / default_name


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    output = {
        "schema_version": "Emerge.robodojo_episode_status.v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "task": ARGS.task,
        "layout_id": ARGS.layout_id,
        "policy_seed": ARGS.policy_seed,
        **payload,
    }
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _wait_for_finalized_episode_videos(
    save_dir: Path,
    episode_number: int,
    *,
    minimum_count: int = 3,
    timeout_s: float = 15.0,
) -> list[Path]:
    """Wait for RoboDojo's ffmpeg writers to publish their final MP4 names.

    ``EvalEnv.run_eval`` can return just before the three camera encoders rename
    their ``.tmp.mp4`` streams.  Publishing the episode status before that
    rename lets the outer evaluator sync a complete verdict but no videos.
    Final MP4 names are only visible after ffmpeg closes them, so waiting for
    the expected three non-empty files also avoids copying an active stream.
    """

    pattern = f"episode_{episode_number:07d}_*.mp4"
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    videos: list[Path] = []
    while True:
        videos = sorted(
            path
            for path in save_dir.glob(pattern)
            if path.is_file() and path.stat().st_size > 0
        )
        if len(videos) >= minimum_count or time.monotonic() >= deadline:
            return videos
        time.sleep(0.1)


def _write_smoke_result(path: Path, actions: list[dict[str, Any]]) -> None:
    """Persist diagnostic outcomes that would otherwise vanish in Kit stdout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "schema_version": "Emerge.robodojo_smoke_result.v1",
                "task": ARGS.task,
                "layout_id": ARGS.layout_id,
                "policy_seed": ARGS.policy_seed,
                "actions": actions,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _parse_smoke_actions(raw: str) -> list[tuple[str, dict[str, Any]]]:
    """One action, or a sequence of them.

    A sequence is what makes a primitive reproducible in isolation: a descent
    that fails after the arm has lifted and traversed cannot be told apart from
    one issued at the reset pose unless the same approach runs first.
    """
    payload = json.loads(raw)
    items = payload if isinstance(payload, list) else [payload]
    actions: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("--smoke-action-json must be an object or a list of objects")
        action_type = str(item.get("action_type", "")).strip()
        parameters = item.get("parameters", {})
        if not action_type or not isinstance(parameters, dict):
            raise ValueError("smoke action requires action_type and object parameters")
        actions.append((action_type, parameters))
    if not actions:
        raise ValueError("--smoke-action-json needs at least one action")
    return actions


def _gripper_link_positions(environment: Any) -> dict[str, list[float]]:
    """Where the fingers actually are, for the smoke path only.

    The commanded pose is ``link6``, the wrist flange, and the fingers hang off
    it. Reasoning about that offset from the URDF and a quaternion is how three
    grasp attempts closed on empty table: read the links instead.
    """
    manager = environment.robot_manager
    positions: dict[str, list[float]] = {}
    for arm in ("left", "right"):
        robot = manager.get_robot_by_arm_name(f"{arm}_arm")
        for link in ("link6", "link7", "link8"):
            try:
                pose = manager.get_link_pose(
                    robot, link_name=link, env_idx_list=[0], is_relative=True
                )[0]
            except Exception:
                continue
            positions[f"{arm}_{link}"] = [
                round(float(value), 4) for value in np.asarray(pose)[:3]
            ]
    return positions


def _load_episode_manifest() -> list[dict[str, Any]]:
    path = Path(str(ARGS.episode_manifest)).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"episode manifest must be a non-empty JSON list: {path}")
    episodes: list[dict[str, Any]] = []
    seen_layouts: set[int] = set()
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise ValueError(f"episode manifest entry {index} is not an object")
        task = str(raw.get("task", ARGS.task))
        if task != ARGS.task:
            raise ValueError(
                "one persistent worker can only serve one task: "
                f"expected {ARGS.task!r}, entry {index} has {task!r}"
            )
        layout_id = int(raw["layout_id"])
        if layout_id in seen_layouts:
            raise ValueError(f"duplicate layout {layout_id} in episode manifest")
        seen_layouts.add(layout_id)
        workspace = Path(str(raw["workspace"])).expanduser().resolve()
        episodes.append(
            {
                **raw,
                "task": task,
                "layout_id": layout_id,
                "workspace": str(workspace),
            }
        )
    return episodes


@contextlib.contextmanager
def _episode_process_log(path: Path):
    """Send Python and Kit output for one reset/run to its own worker log."""

    path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(sys.stdout.fileno())
    saved_stderr = os.dup(sys.stderr.fileno())
    with path.open("a", encoding="utf-8") as stream:
        try:
            os.dup2(stream.fileno(), sys.stdout.fileno())
            os.dup2(stream.fileno(), sys.stderr.fileno())
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, sys.stdout.fileno())
            os.dup2(saved_stderr, sys.stderr.fileno())
            os.close(saved_stdout)
            os.close(saved_stderr)


def _default_agent_message() -> str:
    return (
        "Solve the RoboDojo task described in ROBOT_STATE.md. Use the available "
        "embodied skills, re-read ROBOT_STATE.md after every action, and stop as "
        "soon as robots.robodojo.done is true."
    )


def _start_agent_supervisor(
    *,
    workspace: Path,
    session_id: str,
    done_file: Path,
    verdict_ready: threading.Event,
) -> tuple[threading.Thread, dict[str, Any]]:
    """Launch a fresh EMERGE session once the watchdog publishes a live view."""

    state: dict[str, Any] = {
        "process": None,
        "returncode": None,
        "timed_out": False,
        "error": None,
        "stopped_after_verdict": False,
    }

    def supervise() -> None:
        environment_path = workspace / "ROBOT_STATE.md"
        ready_deadline = time.monotonic() + 600.0
        while time.monotonic() < ready_deadline:
            if verdict_ready.is_set():
                return
            try:
                if '"connected": true' in environment_path.read_text(
                    encoding="utf-8"
                ):
                    break
            except FileNotFoundError:
                pass
            time.sleep(0.1)
        else:
            state["error"] = "agent timed out waiting for simulator readiness"
            state["timed_out"] = True
            done_file.touch()
            return

        agent_python = Path(
            ARGS.agent_python or (EMERGE_ROOT / ".venv" / "bin" / "python")
        ).expanduser().absolute()
        command = [
            str(agent_python),
            "-m", "Emerge.cli.headless",
            "--workspace",
            str(workspace),
            "--session",
            session_id,
            ARGS.agent_message or _default_agent_message(),
            "--restrict-to-workspace",
            "--output-dir", str(workspace / "agent_run"),
        ]
        if ARGS.agent_config:
            command.extend(
                ["--config", str(Path(ARGS.agent_config).expanduser().resolve())]
            )
        try:
            with (workspace / "agent.log").open("w", encoding="utf-8") as agent_log:
                process = subprocess.Popen(
                    command,
                    cwd=EMERGE_ROOT,
                    env={**os.environ, "EMERGE_POLICY_BACKEND": "vla"},
                    stdout=agent_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                state["process"] = process
                deadline = time.monotonic() + ARGS.episode_timeout_s
                verdict_seen_at: float | None = None
                while process.poll() is None:
                    now = time.monotonic()
                    if verdict_ready.is_set():
                        verdict_seen_at = verdict_seen_at or now
                        if now - verdict_seen_at >= 15.0:
                            state["stopped_after_verdict"] = True
                            process.terminate()
                            break
                    elif now >= deadline:
                        state["timed_out"] = True
                        state["error"] = (
                            f"episode exceeded {ARGS.episode_timeout_s:.0f}s"
                        )
                        process.terminate()
                        break
                    time.sleep(0.25)
                try:
                    returncode = process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait()
                # Terminating EMERGE after RoboDojo already produced a verdict is
                # normal cleanup, not an agent infrastructure failure.
                state["returncode"] = (
                    0 if state["stopped_after_verdict"] else int(returncode)
                )
        except BaseException as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["returncode"] = 1
        finally:
            done_file.touch()

    thread = threading.Thread(
        target=supervise,
        name=f"emerge-layout-{ARGS.layout_id}",
        daemon=True,
    )
    thread.start()
    return thread, state


def _policy_baseline_episode(driver: RoboDojoDriver) -> None:
    state = driver.get_runtime_state()["robots"]["robodojo"]
    budget = max(1, int(state.get("step_limit") or 0) - int(state.get("step") or 0))
    result = driver.execute_action(
        "vla_execute",
        {"instruction": state.get("instruction") or "", "step": budget},
    )
    print(f"[robodojo-baseline] {result}", flush=True)
    if not driver.is_terminal():
        print(
            "[robodojo-baseline] non-terminal rollout; forcing official failure",
            flush=True,
        )
        driver.request_stop()


def _run_persistent_batch() -> None:
    """Run several same-task layouts while paying AppLauncher startup once."""

    episodes = _load_episode_manifest()
    layout_ids = [int(episode["layout_id"]) for episode in episodes]
    policy_client = Pi05Client(
        ARGS.policy_server_url,
        timeout=ARGS.policy_timeout,
    )
    environment = None
    try:
        apply_isaacsim_compat()
        select_planner_device(ARGS.device_id if os.environ.get("ROBODOJO_UNSET_CUDA_VISIBLE_DEVICES") == "1" else 0)
        environment = create_eval_env(
            _build_config(layout_ids),
            SIMULATION_APP,
        )
        for episode_index, episode in enumerate(episodes):
            stop_file = worker_stop_file()
            if stop_file is not None and stop_file.exists():
                break
            ARGS.layout_id = int(episode["layout_id"])
            ARGS.workspace = str(episode["workspace"])
            policy_client.begin_episode()
            workspace = _workspace()
            _prepare_workspace(workspace)
            status_path = workspace / "episode_status.json"
            trajectory_path = workspace / "trajectory.jsonl"
            trajectory_path.unlink(missing_ok=True)
            done_file = workspace / ".agent_done"
            done_file.unlink(missing_ok=True)
            started_at = datetime.now(timezone.utc).isoformat()
            started_monotonic = time.monotonic()
            _write_status(
                status_path,
                {
                    "ready": False,
                    "finished": False,
                    "success": False,
                    "workspace": str(workspace),
                    "started_at": started_at,
                    "persistent_worker": True,
                    "batch_episode_index": episode_index,
                },
            )

            with _episode_process_log(workspace / "robodojo_worker.log"):
                driver = None
                supervisor_thread = None
                supervisor_state: dict[str, Any] = {}
                verdict_ready = threading.Event()
                try:
                    # EvalEnv.close() releases the previous simulation context;
                    # reset() recreates it inside the same long-lived Kit app.
                    if episode_index:
                        environment.close()
                    environment.reset(seed=[ARGS.layout_id])
                    driver = RoboDojoDriver(
                        environment,
                        motion=_load_motion_config(),
                        policy_client=policy_client,
                        workspace=workspace,
                        archive_observations=ARGS.archive_observations,
                    )
                    _write_status(
                        status_path,
                        {
                            "ready": True,
                            "finished": False,
                            "success": False,
                            "workspace": str(workspace),
                            "started_at": started_at,
                            "persistent_worker": True,
                            "batch_episode_index": episode_index,
                        },
                    )

                    if ARGS.policy_baseline:
                        environment.eval_one_episode = lambda: _policy_baseline_episode(
                            driver
                        )
                    else:
                        session_id = str(
                            episode.get("session_id")
                            or f"robodojo:{ARGS.task}:layout-{ARGS.layout_id}:seed-{ARGS.policy_seed}"
                        )
                        supervisor_thread, supervisor_state = _start_agent_supervisor(
                            workspace=workspace,
                            session_id=session_id,
                            done_file=done_file,
                            verdict_ready=verdict_ready,
                        )

                        def agent_episode() -> None:
                            watch_driver_loop(
                                driver,
                                workspace,
                                driver_name="robodojo",
                                poll_interval=ARGS.poll_interval,
                                stop_when_terminal=True,
                                raise_on_interrupt=True,
                                stop_file=done_file,
                                trajectory_file=trajectory_path,
                            )

                        environment.eval_one_episode = agent_episode

                    print(
                        f"[robodojo-agent] ready task={ARGS.task} "
                        f"layout={ARGS.layout_id} workspace={workspace}",
                        flush=True,
                    )
                    episode_number = int(environment.success_nums) + int(
                        environment.fail_nums
                    )
                    environment.run_eval()
                    driver.raise_if_infrastructure_error()
                    verdict_ready.set()
                    if supervisor_thread is not None:
                        supervisor_thread.join(timeout=25.0)
                        process = supervisor_state.get("process")
                        if supervisor_thread.is_alive() and process is not None:
                            process.kill()
                            supervisor_thread.join(timeout=5.0)

                    save_dir_path = (ROBODOJO_ROOT / environment.save_dir).resolve()
                    save_dir = str(save_dir_path)
                    # A persistent batch shares RoboDojo's aggregate save_dir.
                    # Copy only this layout's finalized videos into its private
                    # workspace so the evaluator can sync incrementally without
                    # repeatedly copying a growing directory (or its live tmp).
                    video_dir = workspace / "artifacts" / "robodojo_video"
                    video_dir.mkdir(parents=True, exist_ok=True)
                    video_paths: list[str] = []
                    finalized_videos = _wait_for_finalized_episode_videos(
                        save_dir_path,
                        episode_number,
                        timeout_s=float(
                            os.environ.get(
                                "ROBODOJO_VIDEO_FINALIZE_TIMEOUT_S", "15"
                            )
                        ),
                    )
                    if len(finalized_videos) < 3:
                        print(
                            "[robodojo-agent] WARNING expected three finalized "
                            f"episode videos, found {len(finalized_videos)} in "
                            f"{save_dir_path}",
                            flush=True,
                        )
                    for source in finalized_videos:
                        destination = video_dir / source.name
                        shutil.copy2(source, destination)
                        video_paths.append(str(destination))
                    aggregate_result = save_dir_path / "_result.json"
                    if aggregate_result.exists():
                        shutil.copy2(aggregate_result, video_dir / "_result.json")
                    success_flags = getattr(environment, "success", [False])
                    success = bool(success_flags and success_flags[0])
                    _write_status(
                        status_path,
                        {
                            "ready": True,
                            "finished": True,
                            "success": success,
                            "workspace": str(workspace),
                            "save_dir": save_dir,
                            "video_paths": video_paths,
                            "result": environment.eval_result,
                            "official_excluded": 0 in getattr(environment, "unstable_envs", set()),
                            "started_at": started_at,
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "duration_s": round(
                                time.monotonic() - started_monotonic, 3
                            ),
                            "agent_returncode": supervisor_state.get("returncode", 0),
                            "timed_out": bool(supervisor_state.get("timed_out")),
                            "agent_error": supervisor_state.get("error"),
                            "persistent_worker": True,
                            "batch_episode_index": episode_index,
                        },
                    )
                    if stop_file is not None:
                        pause_on_provider_error(workspace, stop_file)
                except BaseException as exc:
                    verdict_ready.set()
                    process = supervisor_state.get("process")
                    if process is not None and process.poll() is None:
                        process.terminate()
                    if supervisor_thread is not None:
                        supervisor_thread.join(timeout=5.0)
                    _write_status(
                        status_path,
                        {
                            "ready": driver is not None,
                            "finished": True,
                            "success": False,
                            "workspace": str(workspace),
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                            "started_at": started_at,
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "duration_s": round(
                                time.monotonic() - started_monotonic, 3
                            ),
                            "persistent_worker": True,
                            "batch_episode_index": episode_index,
                        },
                    )
                    # A simulator exception can poison the PhysX/CUDA context.
                    # Stop this batch; the outer evaluator records completed
                    # layouts and --resume starts a clean worker for the rest.
                    raise
    finally:
        policy_client.close()
        if environment is not None:
            environment.close()


def main() -> None:
    if ARGS.warmup_only:
        try:
            task_registry.load_task_class(ARGS.task)
            print(
                f"[robodojo-agent] warmup ready task={ARGS.task} "
                f"device={ARGS.device}",
                flush=True,
            )
        finally:
            SIMULATION_APP.close()
        return

    if ARGS.episode_manifest:
        try:
            _run_persistent_batch()
        finally:
            SIMULATION_APP.close()
        return

    workspace = _workspace()
    _prepare_workspace(workspace)
    status_path = _output_path(ARGS.status_path, workspace, "episode_status.json")
    trajectory_path = _output_path(ARGS.trajectory_path, workspace, "trajectory.jsonl")
    trajectory_path.unlink(missing_ok=True)
    _write_status(
        status_path,
        {
            "ready": False,
            "finished": False,
            "success": False,
            "workspace": str(workspace),
        },
    )
    done_file = (
        Path(ARGS.agent_done_file).expanduser().resolve()
        if ARGS.agent_done_file
        else workspace / ".agent_done"
    )
    if done_file.exists():
        done_file.unlink()

    environment = None
    policy_client = Pi05Client(
        ARGS.policy_server_url,
        timeout=ARGS.policy_timeout,
    )
    try:
        config = _build_config()
        # Must precede reset(): the articulations are read during scene reload.
        apply_isaacsim_compat()
        select_planner_device(ARGS.device_id if os.environ.get("ROBODOJO_UNSET_CUDA_VISIBLE_DEVICES") == "1" else 0)
        environment = create_eval_env(config, SIMULATION_APP)
        environment.reset(seed=[ARGS.layout_id])
        driver = RoboDojoDriver(
            environment,
            motion=_load_motion_config(),
            policy_client=policy_client,
            workspace=workspace,
            archive_observations=ARGS.archive_observations,
        )

        if ARGS.smoke_action_json:
            environment.run_reward()
            from external_model_server.protocol import pack_message
            from robot.vla.robodojo_policy import encode_observation
            (workspace / "policy_observation.msgpack").write_bytes(
                pack_message(encode_observation(environment.get_obs())))
            outcomes = []
            for index, (action_type, parameters) in enumerate(
                _parse_smoke_actions(ARGS.smoke_action_json)
            ):
                result = driver.execute_action(action_type, parameters)
                pose = {
                    arm: value["end_effector_pose"]["position_m"]
                    for arm, value in driver.get_runtime_state()["robots"]["robodojo"][
                        "arms"
                    ].items()
                }
                outcomes.append(
                    {
                        "index": index,
                        "action": action_type,
                        "parameters": parameters,
                        "result": result,
                        "end_effector_position_m": pose,
                        "gripper_links_m": _gripper_link_positions(environment),
                    }
                )
                print(f"[smoke] {index} {action_type} -> {result}", flush=True)
            _write_smoke_result(workspace / "smoke_result.json", outcomes)
            print(json.dumps({"actions": outcomes}, indent=2))
            return

        _write_status(
            status_path,
            {
                "ready": True,
                "finished": False,
                "success": False,
                "workspace": str(workspace),
            },
        )

        def _agent_episode() -> None:
            watch_driver_loop(
                driver,
                workspace,
                driver_name="robodojo",
                poll_interval=ARGS.poll_interval,
                stop_when_terminal=True,
                raise_on_interrupt=True,
                stop_file=done_file,
                trajectory_file=trajectory_path,
            )

        def _policy_baseline_episode() -> None:
            """One uninterrupted rollout, the reference the agent must beat.

            Deliberately runs inside the same run_eval() path as the agent so
            the two arms are scored identically.
            """
            state = driver.get_runtime_state()["robots"]["robodojo"]
            budget = max(1, int(state.get("step_limit") or 0) - int(state.get("step") or 0))
            result = driver.execute_action(
                "vla_execute",
                {"instruction": state.get("instruction") or "", "step": budget},
            )
            print(f"[robodojo-baseline] {result}", flush=True)
            # EvalEnv initializes success optimistically and normally replaces
            # it when an episode reaches a terminal state. A rejected policy
            # call (or a rollout stopped by the idle guard) can return without
            # taking an action, leaving that initial True behind. Such a run
            # is an official failure, never a baseline success.
            if not driver.is_terminal():
                print(
                    "[robodojo-baseline] non-terminal rollout; forcing "
                    "official failure",
                    flush=True,
                )
                driver.request_stop()

        environment.eval_one_episode = (
            _policy_baseline_episode if ARGS.policy_baseline else _agent_episode
        )
        print(
            f"[robodojo-agent] ready task={ARGS.task} layout={ARGS.layout_id} "
            f"workspace={workspace}",
            flush=True,
        )
        environment.run_eval()
        driver.raise_if_infrastructure_error()
        save_dir = str((ROBODOJO_ROOT / environment.save_dir).resolve())
        success_flags = getattr(environment, "success", [False])
        success = bool(success_flags and success_flags[0])
        _write_status(
            status_path,
            {
                "ready": True,
                "finished": True,
                "success": success,
                "workspace": str(workspace),
                "save_dir": save_dir,
                "result": environment.eval_result,
                            "official_excluded": 0 in getattr(environment, "unstable_envs", set()),
            },
        )
        print(
            json.dumps(
                {
                    "save_dir": save_dir,
                    "result": environment.eval_result,
                            "official_excluded": 0 in getattr(environment, "unstable_envs", set()),
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException as exc:
        _write_status(
            status_path,
            {
                "ready": environment is not None,
                "finished": True,
                "success": False,
                "workspace": str(workspace),
                "error": f"{type(exc).__name__}: {exc}",
                # Kit writes tens of thousands of lines around ours and can take
                # the process down before stderr is flushed, so the one thing
                # that says where the failure came from is recorded here.
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        policy_client.close()
        try:
            if environment is not None:
                environment.close()
        finally:
            SIMULATION_APP.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[robodojo-agent] interrupted", file=sys.stderr, flush=True)
        os._exit(130)
    except BaseException:
        # SimulationApp shutdown can otherwise mask Python failures with rc=0.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
