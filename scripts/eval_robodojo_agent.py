#!/usr/bin/env python3
"""Evaluate Emerge + Pi0.5 on RoboDojo with persistent simulator workers.

Each GPU slot owns one AppLauncher per task and resets it across assigned
layouts, avoiding a full Isaac/Kit restart for every episode. Pi0.5 is served
separately and shared by every worker. The legacy one-process-per-layout path
remains available for bisecting simulator lifecycle issues.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import random
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Keep the source path stable even when the repository is reached through a
# symlink; callers can choose separate local scratch with ROBODOJO_EVAL_SCRATCH.
REPO_ROOT = Path(__file__).absolute().parents[1]
DEFAULT_TASKS_FILE = REPO_ROOT / "configs/robodojo_tasks_arx_x5_seed0.txt"
DEFAULT_AGENT_CONFIG = Path.home() / ".config/emerge-robodojo/agent.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts/robodojo_agent_eval"
# Episode workspaces are polled several times a second, which the shared mount
# cannot serve (~50 file opens/s), so they stay on local disk and only finished
# attempts are mirrored to --run-dir.
LOCAL_SCRATCH_ROOT = Path(
    os.environ.get(
        "ROBODOJO_EVAL_SCRATCH",
        str(Path(tempfile.gettempdir()) / "agenticvla-robodojo-eval"),
    )
)
LLM_ERROR_MARKERS = (
    # The agent loop emits this terminal marker after all provider retries and
    # model fallback are exhausted. The nested payload is not guaranteed to
    # repeat an exception class, so match the terminal prefix itself;
    # transient warning lines are handled below.
    "LLM returned error:",
    "Error calling Codex:",
    "Error calling LLM:",
    "AuthenticationError",
    "RateLimitError",
    "insufficient_user_quota",
    "用户额度不足",
    "余额不足",
)
RECOVERED_LLM_ERROR_LINE_MARKERS = (
    "LLM transient error (attempt",
    "Main LLM transient failure; switching model",
    "Main LLM recoverable route failure; switching model",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _policy_url_for_device(urls: str, device_id: int) -> str:
    """Distribute workers across explicit, comma-separated policy replicas."""
    endpoints = [url.strip() for url in urls.split(",") if url.strip()]
    if not endpoints:
        raise ValueError("At least one policy server URL is required")
    return endpoints[device_id % len(endpoints)]


def _repo_path(path: Path) -> Path:
    """Resolve CLI paths consistently, independent of the caller's cwd."""
    path = path.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _parse_int_set(text: str) -> list[int]:
    values: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = token.split("-", 1)
            start, end = int(first), int(last)
            if start > end:
                raise ValueError(f"descending range is not allowed: {token}")
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    if not values:
        raise ValueError("at least one integer is required")
    return sorted(values)


def _tasks(path: Path, selected: str | None) -> list[str]:
    available = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not selected or selected.strip().lower() == "all":
        return available
    requested = [item.strip() for item in selected.split(",") if item.strip()]
    unknown = [item for item in requested if item not in available]
    if unknown:
        raise ValueError(f"unknown task(s): {', '.join(unknown)}")
    return list(dict.fromkeys(requested))


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "episode"


def _resolve_devices(selection: str) -> list[int]:
    """Simulation GPUs for this node.

    ``auto`` uses every local GPU; policy inference runs on remote A100s.
    """
    if selection.strip().lower() != "auto":
        return _parse_int_set(selection)
    total = int(os.environ.get("ROBODOJO_GPU_COUNT", "0") or 0)
    if total < 1:
        raise ValueError(
            "--devices auto needs ROBODOJO_GPU_COUNT >= 1; "
            f"got {total}. Pass --devices explicitly instead."
        )
    return list(range(total))


def _available_layouts(robodojo_root: Path, env_cfg: str, seed: int, task: str) -> list[int]:
    """Layout ids that actually exist on disk for one task."""
    layout_dir = robodojo_root / "Assets/Eval_Layout/RoboDojo" / env_cfg / str(seed)
    ids: list[int] = []
    for path in layout_dir.glob(f"{task}_*.json"):
        suffix = path.stem[len(task) + 1 :]
        if suffix.isdigit():
            ids.append(int(suffix))
    return sorted(ids)


def _native_episode_counts(robodojo_root: Path, tasks: list[str]) -> dict[str, int]:
    """Read RoboDojo's own per-task episode counts from its task config.

    The benchmark defines ``eval_nums`` per task -- 50 by default, 25 for two
    dozen of them -- and that per-task count, not a uniform layout range, is
    what the official protocol evaluates.
    """
    import yaml

    config_path = robodojo_root / "task/RoboDojo/config/_task.yml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    default = int((document.get("common") or {}).get("eval_nums", 50))
    overrides = document.get("tasks") or {}
    counts = {}
    for task in tasks:
        entry = overrides.get(task) or {}
        counts[task] = int(entry.get("eval_nums", default))
    return counts


def _layouts_per_task(
    tasks: list[str],
    selection: str,
    *,
    robodojo_root: Path,
    env_cfg: str,
    policy_seed: int,
) -> dict[str, list[int]]:
    """Map each task to the layout ids it will be evaluated on.

    ``native`` follows the benchmark's own per-task ``eval_nums``; anything else
    is an explicit range applied uniformly. Every requested layout must exist. Missing assets must never silently
    turn a full standard evaluation into a smaller subset.
    """
    if selection.strip().lower() == "native":
        counts = _native_episode_counts(robodojo_root, tasks)
        requested = {task: list(range(counts[task])) for task in tasks}
    else:
        explicit = _parse_int_set(selection)
        requested = {task: list(explicit) for task in tasks}

    resolved: dict[str, list[int]] = {}
    for task in tasks:
        available = set(_available_layouts(robodojo_root, env_cfg, policy_seed, task))
        missing = sorted(set(requested[task]) - available)
        if missing:
            raise ValueError(f"Missing required layouts for {task}: {missing}")
        chosen = list(requested[task])
        if not chosen:
            raise ValueError(
                f"no layout files for task {task!r} under "
                f"{robodojo_root}/Assets/Eval_Layout/RoboDojo/{env_cfg}/{policy_seed}"
            )
        resolved[task] = chosen
    return resolved


def _sample_layouts_per_task(
    layouts_per_task: dict[str, list[int]],
    *,
    count: int,
    seed: int,
) -> dict[str, list[int]]:
    """Choose a reproducible, task-independent layout sample.

    A stable task-specific seed keeps one task's sample unchanged when another
    task is added to or removed from the command.  Sorting the result makes the
    exact sample easy to review in ``run_config.json`` and run metadata.
    """
    if count <= 0:
        raise ValueError("--sample-layout-count must be positive")

    sampled: dict[str, list[int]] = {}
    for task, layouts in layouts_per_task.items():
        if count > len(layouts):
            raise ValueError(
                f"cannot sample {count} layouts for {task!r}; only "
                f"{len(layouts)} candidate layouts remain"
            )
        digest = hashlib.sha256(f"{seed}:{task}".encode()).digest()
        task_seed = int.from_bytes(digest[:8], byteorder="big")
        sampled[task] = sorted(random.Random(task_seed).sample(layouts, count))
    return sampled


def _runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    no_proxy = env.get("no_proxy", "")
    for host in ("localhost", "127.0.0.1", "::1"):
        if host not in no_proxy.split(","):
            no_proxy = f"{no_proxy},{host}" if no_proxy else host
    env["no_proxy"] = no_proxy
    env["NO_PROXY"] = no_proxy
    env["ROBODOJO_UNSET_CUDA_VISIBLE_DEVICES"] = "1"
    env["NO_COLOR"] = "1"
    env["TERM"] = "dumb"
    return env


def _model_metadata(config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    defaults = ((config.get("agents") or {}).get("defaults") or {})
    return {
        "llm_model": defaults.get("model"),
        "llm_provider": defaults.get("provider"),
        "llm_reasoning_effort": defaults.get("reasoningEffort"),
    }


def _missing_provider_credentials(
    config_path: Path, env: dict[str, str] | None = None
) -> list[str]:
    """Return unresolved provider ``${ENV_VAR}`` credentials.

    Persistent workers launch EMERGE directly and therefore bypass the legacy
    episode shell's preflight. Without this check an unset variable is sent
    literally as the bearer token, spending minutes on Isaac startup before a
    deterministic 401. The caller remains responsible for sourcing its chosen
    secrets file; this function never reads or logs secret values.
    """

    source_env = os.environ if env is None else env
    providers = (_load_json(config_path).get("providers") or {})
    missing: list[str] = []
    for provider, settings in providers.items():
        if not isinstance(settings, dict):
            continue
        value = str(settings.get("apiKey", "")).strip()
        match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
        if match and not str(source_env.get(match.group(1), "")).strip():
            missing.append(f"{provider}:{match.group(1)}")
    return missing


def _seed_results_from_durable(results_path: Path, sync_dir: Path | None) -> None:
    """Restore this shard's history from shared storage before resuming.

    Local scratch does not survive a container restart and the platform hands a
    restarted job an empty one, so without this a resume would find no history
    and re-run episodes that are already recorded on shared storage.
    """
    if sync_dir is None or results_path.exists():
        return
    durable = sync_dir / "results.jsonl"
    if not durable.exists():
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(durable, results_path)
    recovered = len(_read_previous_results(results_path))
    print(
        f"[robodojo-eval] recovered {recovered} episode(s) from {durable}",
        flush=True,
    )


def _read_previous_results(path: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return results
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("episode_key"):
            results[str(item["episode_key"])] = item
    return results


def _llm_failed(log_path: Path) -> bool:
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    # Retry/fallback warnings quote the transient error that triggered them.
    # Searching the complete file for ``Error calling LLM`` therefore labels a
    # successfully recovered episode as infrastructure failure.  A genuinely
    # exhausted call is logged later as ``LLM returned error`` and remains a
    # match, as do authentication/quota errors that cannot be recovered.
    return any(
        any(marker in line for marker in LLM_ERROR_MARKERS)
        and not any(
            recovered_marker in line
            for recovered_marker in RECOVERED_LLM_ERROR_LINE_MARKERS
        )
        for line in content.splitlines()
    )


def _termination_reason(
    status: dict[str, Any],
    *,
    timed_out: bool,
    llm_failed: bool,
    returncode: int,
) -> str:
    """Classify one episode from the verdict the worker left on disk."""
    # A timeout that fires after the worker recorded its verdict means the
    # simulator wedged while shutting down, not that the episode failed to run.
    # Filing that as an infrastructure error drops a real result from the
    # denominator and makes --resume pay for the whole episode a second time.
    if timed_out and not status.get("finished"):
        return "infrastructure_error:episode_timeout"
    if llm_failed:
        return "infrastructure_error:llm"
    if status.get("error"):
        return "infrastructure_error:robodojo_worker"
    # The worker marks itself finished once the supervisor signals that the
    # agent exited, whether or not the agent did any work, so a crashed or
    # unconfigured agent otherwise lands in the results as a task failure.
    if status.get("finished") and status.get("agent_finish_reason") in {"max_iterations", "length"}:
        return "success" if status.get("success") else "official_failure:agent_budget_exhausted"
    if not status.get("finished") or (not timed_out and returncode != 0):
        return f"infrastructure_error:runner_exit_{returncode}"
    if status.get("official_excluded"):
        return "official_excluded:unstable_scene"
    return "success" if status.get("success") else "official_failure"


def _episode_score(status: dict[str, Any], layout_id: int) -> float | None:
    for detail in status.get("result", {}).get("details", {}).values():
        if int(detail.get("layout_id", -1)) == layout_id:
            return float(detail["score"])
    return None


def _run_episode(
    spec: dict[str, Any],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    device_slots: queue.Queue[int],
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    device_id = device_slots.get()
    started_at = _utc_now()
    start = time.monotonic()
    episode_root = (
        output_dir
        / "episodes"
        / _safe_name(spec["task"])
        / f"layout_{spec['layout_id']:02d}_seed_{spec['policy_seed']}"
    )
    episode_dir = episode_root / datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
    workspace = episode_dir / "workspace"
    episode_dir.mkdir(parents=True, exist_ok=True)
    runner_log_path = episode_dir / "runner.log"
    command = [
        "bash",
        str(REPO_ROOT / "scripts/run_robodojo_agent_episode.sh"),
        "--task",
        spec["task"],
        "--layout-id",
        str(spec["layout_id"]),
        "--device-id",
        str(device_id),
        "--policy-seed",
        str(spec["policy_seed"]),
        "--policy-server-url",
        _policy_url_for_device(args.policy_server_url, device_id),
        "--workspace",
        str(workspace),
        "--agent-config",
        str(args.agent_config),
        "--session-id",
        f"robodojo-eval:{spec['task']}:layout-{spec['layout_id']}:seed-{spec['policy_seed']}",
    ]
    if args.agent_message:
        command.extend(["--agent-message", args.agent_message])
    if args.agent_python is not None:
        command.extend(["--agent-python", str(args.agent_python)])
    if args.archive_observations:
        command.append("--archive-observations")
    if args.policy_baseline:
        command.append("--policy-baseline")
    if args.record_every_step:
        command.append("--record-every-step")
    returncode = -1
    timed_out = False
    error: str | None = None
    try:
        with runner_log_path.open("w", encoding="utf-8") as runner_log:
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=_runtime_env(),
                    stdout=runner_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=args.episode_timeout_s,
                )
                returncode = int(completed.returncode)
            except subprocess.TimeoutExpired:
                timed_out = True
                error = f"episode exceeded {args.episode_timeout_s:.0f}s"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        device_slots.put(device_id)

    status = _load_json(workspace / "episode_status.json")
    status["agent_finish_reason"] = _load_json(workspace / "agent_run" / "result.json").get("finish_reason")
    termination_reason = _termination_reason(
        status,
        timed_out=timed_out,
        llm_failed=_llm_failed(workspace / "agent.log"),
        returncode=returncode,
    )

    return {
        "schema_version": "Emerge.robodojo_episode_result.v1",
        "episode_key": spec["episode_key"],
        "task": spec["task"],
        "layout_id": spec["layout_id"],
        "policy_seed": spec["policy_seed"],
        "device_id": device_id,
        "success": bool(status.get("success")),
        "official_score": _episode_score(status, spec["layout_id"]),
        "official_excluded": bool(status.get("official_excluded")),
        "termination_reason": termination_reason,
        "returncode": returncode,
        "agent_finish_reason": status.get("agent_finish_reason"),
        "error": error,
        "duration_s": round(time.monotonic() - start, 3),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "episode_dir": str(episode_dir),
        "workspace": str(workspace),
        "robodojo_save_dir": status.get("save_dir"),
        **model_metadata,
    }


def _episode_attempt(
    spec: dict[str, Any], output_dir: Path
) -> tuple[Path, Path]:
    episode_root = (
        output_dir
        / "episodes"
        / _safe_name(spec["task"])
        / f"layout_{spec['layout_id']:02d}_seed_{spec['policy_seed']}"
    )
    episode_dir = episode_root / datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
    workspace = episode_dir / "workspace"
    episode_dir.mkdir(parents=True, exist_ok=True)
    return episode_dir, workspace


def _persistent_result(
    *,
    spec: dict[str, Any],
    status: dict[str, Any],
    episode_dir: Path,
    workspace: Path,
    device_id: int,
    batch_returncode: int,
    batch_started_at: str,
    batch_start_monotonic: float,
    first_in_batch: bool,
    batch_log_path: Path,
    model_metadata: dict[str, Any],
) -> dict[str, Any]:
    agent_result = _load_json(workspace / "agent_run" / "result.json")
    status = {**status, "agent_finish_reason": agent_result.get("finish_reason")}
    finished = bool(status.get("finished"))
    timed_out = bool(status.get("timed_out"))
    returncode = int(
        status.get("agent_returncode")
        if status.get("agent_returncode") is not None
        else (0 if finished else batch_returncode)
    )
    if timed_out:
        termination_reason = "infrastructure_error:episode_timeout"
    else:
        termination_reason = _termination_reason(
            status,
            timed_out=False,
            llm_failed=_llm_failed(workspace / "agent.log"),
            returncode=returncode,
        )

    duration_s = status.get("duration_s")
    started_at = str(status.get("started_at") or batch_started_at)
    if first_in_batch and duration_s is not None and status.get("started_at"):
        try:
            app_start = datetime.fromisoformat(batch_started_at)
            episode_start = datetime.fromisoformat(str(status["started_at"]))
            duration_s = float(duration_s) + max(
                0.0, (episode_start - app_start).total_seconds()
            )
            started_at = batch_started_at
        except ValueError:
            pass
    if duration_s is None:
        duration_s = time.monotonic() - batch_start_monotonic

    error = status.get("error") or status.get("agent_error")
    if not finished and error is None:
        error = f"persistent worker exited with rc={batch_returncode} before verdict"
    return {
        "schema_version": "Emerge.robodojo_episode_result.v1",
        "episode_key": spec["episode_key"],
        "task": spec["task"],
        "layout_id": spec["layout_id"],
        "policy_seed": spec["policy_seed"],
        "device_id": device_id,
        "success": bool(status.get("success")),
        "official_score": _episode_score(status, spec["layout_id"]),
        "official_excluded": bool(status.get("official_excluded")),
        "termination_reason": termination_reason,
        "returncode": returncode,
        "agent_finish_reason": status.get("agent_finish_reason"),
        "error": error,
        "duration_s": round(float(duration_s), 3),
        "started_at": started_at,
        "finished_at": str(status.get("finished_at") or _utc_now()),
        "episode_dir": str(episode_dir),
        "workspace": str(workspace),
        "robodojo_save_dir": status.get("save_dir"),
        "persistent_worker": True,
        "batch_log": str(batch_log_path),
        **model_metadata,
    }


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_persistent_task_batch(
    specs: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    device_id: int,
    slot_index: int,
    model_metadata: dict[str, Any],
    on_result: Callable[[dict[str, Any]], None],
) -> None:
    """Run a same-task slice in one AppLauncher and stream completed results."""

    if not specs:
        return
    task = str(specs[0]["task"])
    if any(str(spec["task"]) != task for spec in specs):
        raise ValueError("persistent task batch contains multiple tasks")

    batch_dir = (
        output_dir
        / "batches"
        / _safe_name(task)
        / f"device_{device_id}_slot_{slot_index}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_log_path = batch_dir / "runner.log"
    entries: list[dict[str, Any]] = []
    for spec in specs:
        episode_dir, workspace = _episode_attempt(spec, output_dir)
        (episode_dir / "runner.log").write_text(
            f"Persistent worker log: {batch_log_path}\n"
            f"Per-layout simulator log: {workspace / 'robodojo_worker.log'}\n",
            encoding="utf-8",
        )
        entries.append(
            {
                "spec": spec,
                "episode_dir": episode_dir,
                "workspace": workspace,
                "manifest": {
                    "task": task,
                    "layout_id": spec["layout_id"],
                    "workspace": str(workspace),
                    "session_id": (
                        f"robodojo-eval:{task}:layout-{spec['layout_id']}:"
                        f"seed-{spec['policy_seed']}"
                    ),
                },
            }
        )
    manifest_path = batch_dir / "episodes.json"
    temporary_manifest = manifest_path.with_name(
        f".{manifest_path.name}.{os.getpid()}.tmp"
    )
    temporary_manifest.write_text(
        json.dumps(
            [entry["manifest"] for entry in entries],
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)

    command = [
        "bash",
        str(REPO_ROOT / "scripts/run_robodojo_agent_worker.sh"),
        "--task",
        task,
        "--device-id",
        str(device_id),
        "--policy-seed",
        str(specs[0]["policy_seed"]),
        "--policy-server-url",
        _policy_url_for_device(args.policy_server_url, device_id),
        "--motion-config",
        str(REPO_ROOT / "configs/robodojo_motion.json"),
        "--episode-manifest",
        str(manifest_path),
        "--agent-config",
        str(args.agent_config),
        "--episode-timeout-s",
        str(args.episode_timeout_s),
    ]
    agent_python = args.agent_python or (REPO_ROOT / ".venv" / "bin" / "python")
    command.extend(["--agent-python", str(agent_python)])
    if args.agent_message:
        command.extend(["--agent-message", args.agent_message])
    if args.archive_observations:
        command.append("--archive-observations")
    if args.policy_baseline:
        command.append("--policy-baseline")
    if args.record_every_step:
        command.append("--record-every-step")

    batch_started_at = _utc_now()
    batch_start_monotonic = time.monotonic()
    reported: set[str] = set()
    returncode = -1
    all_reported_at: float | None = None
    shutdown_grace_s = float(os.environ.get("WORKER_SHUTDOWN_GRACE_S", "180"))
    with batch_log_path.open("w", encoding="utf-8") as batch_log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=_runtime_env(),
            stdout=batch_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                for index, entry in enumerate(entries):
                    key = str(entry["spec"]["episode_key"])
                    if key in reported:
                        continue
                    status = _load_json(entry["workspace"] / "episode_status.json")
                    if status.get("finished"):
                        on_result(
                            _persistent_result(
                                spec=entry["spec"],
                                status=status,
                                episode_dir=entry["episode_dir"],
                                workspace=entry["workspace"],
                                device_id=device_id,
                                batch_returncode=0,
                                batch_started_at=batch_started_at,
                                batch_start_monotonic=batch_start_monotonic,
                                first_in_batch=index == 0,
                                batch_log_path=batch_log_path,
                                model_metadata=model_metadata,
                            )
                        )
                        reported.add(key)
                if len(reported) == len(entries):
                    all_reported_at = all_reported_at or time.monotonic()
                    if time.monotonic() - all_reported_at >= shutdown_grace_s:
                        batch_log.write(
                            "[robodojo-eval] persistent simulator did not shut "
                            f"down within {shutdown_grace_s:.0f}s after the final "
                            "verdict; terminating process group\n"
                        )
                        batch_log.flush()
                        _terminate_process_group(process)
                        break
                time.sleep(0.5)
            returncode = int(process.wait())
        except BaseException:
            _terminate_process_group(process)
            returncode = int(process.returncode or -1)
            raise
        finally:
            _terminate_process_group(process)

    for index, entry in enumerate(entries):
        key = str(entry["spec"]["episode_key"])
        if key in reported:
            continue
        status = _load_json(entry["workspace"] / "episode_status.json")
        on_result(
            _persistent_result(
                spec=entry["spec"],
                status=status,
                episode_dir=entry["episode_dir"],
                workspace=entry["workspace"],
                device_id=device_id,
                batch_returncode=returncode,
                batch_started_at=batch_started_at,
                batch_start_monotonic=batch_start_monotonic,
                first_in_batch=index == 0,
                batch_log_path=batch_log_path,
                model_metadata=model_metadata,
            )
        )


def _persistent_slot_assignments(
    specs: list[dict[str, Any]], slot_count: int
) -> list[list[dict[str, Any]]]:
    """Split each task across slots while keeping same-task layouts together."""

    assignments: list[list[dict[str, Any]]] = [[] for _ in range(slot_count)]
    task_offsets: dict[str, int] = {}
    for spec in specs:
        task = str(spec["task"])
        offset = task_offsets.get(task, 0)
        assignments[offset % slot_count].append(spec)
        task_offsets[task] = offset + 1
    return assignments


def _run_persistent_slot(
    specs: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    output_dir: Path,
    device_id: int,
    slot_index: int,
    model_metadata: dict[str, Any],
    on_result: Callable[[dict[str, Any]], None],
) -> None:
    cursor = 0
    while cursor < len(specs):
        task = str(specs[cursor]["task"])
        end = cursor + 1
        while end < len(specs) and str(specs[end]["task"]) == task:
            end += 1
        _run_persistent_task_batch(
            specs[cursor:end],
            args=args,
            output_dir=output_dir,
            device_id=device_id,
            slot_index=slot_index,
            model_metadata=model_metadata,
            on_result=on_result,
        )
        cursor = end


def _write_summary(
    path: Path,
    results: dict[str, dict[str, Any]],
    *,
    scheduled_episodes: int,
    model_metadata: dict[str, Any],
) -> None:
    items = list(results.values())
    evaluated = [
        item
        for item in items
        if not str(item.get("termination_reason", "")).startswith("infrastructure_error:")
        and not item.get("official_excluded")
    ]
    excluded = sum(bool(item.get("official_excluded")) for item in items)
    successes = sum(int(bool(item.get("success"))) for item in evaluated)
    per_task: dict[str, dict[str, Any]] = {}
    for item in evaluated:
        task = str(item["task"])
        task_result = per_task.setdefault(task, {"episodes": 0, "successes": 0})
        task_result["episodes"] += 1
        task_result["successes"] += int(bool(item.get("success")))
    for task_result in per_task.values():
        task_result["success_rate"] = (
            task_result["successes"] / task_result["episodes"]
            if task_result["episodes"]
            else 0.0
        )
    payload = {
        "schema_version": "Emerge.robodojo_evaluation_summary.v1",
        "updated_at": _utc_now(),
        "scheduled_episodes": scheduled_episodes,
        "finished_episodes": len(items),
        "benchmark_episodes": len(evaluated),
        "infrastructure_errors": len(items) - len(evaluated) - excluded,
        "official_excluded_episodes": excluded,
        "successes": successes,
        "success_rate": successes / len(evaluated) if evaluated else 0.0,
        "per_task": per_task,
        **model_metadata,
    }
    _atomic_write_json(path, payload)


def _sync_completed_result(
    *,
    output_dir: Path,
    sync_dir: Path | None,
    result: dict[str, Any],
) -> None:
    """Persist one completed attempt and current aggregate metadata to HDFS/FUSE."""
    if sync_dir is None:
        return
    try:
        sync_dir.mkdir(parents=True, exist_ok=True)
        for name in ("run_config.json", "results.jsonl", "summary.json"):
            source = output_dir / name
            if source.exists():
                destination = sync_dir / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

        episode_dir = Path(str(result["episode_dir"])).resolve()
        relative_episode = episode_dir.relative_to(output_dir.resolve())
        shutil.copytree(
            episode_dir,
            sync_dir / relative_episode,
            dirs_exist_ok=True,
        )

        # Persistent workers already copy each layout's finalized videos into
        # its private workspace. Their native save_dir is shared by the whole
        # live batch; mirroring it here would repeatedly copy a growing tree and
        # could race the next layout's temporary video stream.
        save_dir_raw = (
            None if result.get("persistent_worker") else result.get("robodojo_save_dir")
        )
        if save_dir_raw:
            save_dir = Path(str(save_dir_raw)).resolve()
            if save_dir.exists():
                destination = (
                    sync_dir
                    / "robodojo_eval_result"
                    / _safe_name(str(result["task"]))
                    / f"layout_{int(result['layout_id']):02d}_seed_{int(result['policy_seed'])}"
                    / episode_dir.name
                )
                shutil.copytree(save_dir, destination, dirs_exist_ok=True)
    except (OSError, ValueError) as exc:
        print(f"[robodojo-eval] WARNING sync failed: {exc}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--tasks", default="all", help="all or comma-separated task names")
    parser.add_argument(
        "--layouts",
        default="native",
        help=(
            "native to follow RoboDojo's per-task eval_nums (the official "
            "protocol), or an explicit range such as 0-9 or 0,3"
        ),
    )
    parser.add_argument(
        "--exclude-layouts",
        default=None,
        help=(
            "Optional comma-separated layout ids/ranges to remove before "
            "sampling, for example layouts already used by earlier iterations."
        ),
    )
    parser.add_argument(
        "--sample-layout-count",
        type=int,
        default=None,
        help=(
            "Randomly choose this many remaining layouts per task. The exact "
            "sample is deterministic under --sample-seed and recorded in "
            "run_config.json."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Deterministic seed used by --sample-layout-count.",
    )
    parser.add_argument(
        "--robodojo-root",
        type=Path,
        default=Path(
            os.environ.get("ROBODOJO_ROOT", str(REPO_ROOT.parent / "RoboDojo"))
        ),
    )
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument(
        "--devices",
        default="auto",
        help=(
            "Simulation GPU ids, or auto for every local GPU with remote inference."
        ),
    )
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Reuse one Isaac AppLauncher across same-task layouts (default). "
            "Use --no-persistent-workers only to diagnose lifecycle failures."
        ),
    )
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--policy-server-url", default="ws://127.0.0.1:8000")
    parser.add_argument("--agent-config", type=Path, default=DEFAULT_AGENT_CONFIG)
    parser.add_argument(
        "--agent-python",
        type=Path,
        default=None,
        help="Override the agent Python executable used by each episode runner.",
    )
    message_group = parser.add_mutually_exclusive_group()
    message_group.add_argument(
        "--agent-message",
        default=None,
        help="Override the user message passed to every episode agent.",
    )
    message_group.add_argument(
        "--agent-message-file",
        type=Path,
        default=None,
        help="Read the per-episode agent message from a UTF-8 text file.",
    )
    parser.add_argument(
        "--record-every-step",
        action="store_true",
        help=(
            "Film every control step rather than one frame per policy replan, so "
            "the footage can be read alongside the agent's decisions. Costs a "
            "render per step."
        ),
    )
    parser.add_argument(
        "--policy-baseline",
        action="store_true",
        help=(
            "Run each episode as one uninterrupted VLA rollout with no agent. "
            "This is the reference the agent's own numbers mean nothing without."
        ),
    )
    parser.add_argument(
        "--archive-observations",
        action="store_true",
        help=(
            "Keep every published camera view per episode. Intended for "
            "diagnostic batches; a full run would write a lot of images."
        ),
    )
    parser.add_argument("--episode-timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Durable run directory on shared storage -- the only path a run "
            "normally needs. This node writes RUN_DIR/node-NN/, and its local "
            "scratch is derived automatically."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the local scratch directory. Derived from --run-dir otherwise.",
    )
    parser.add_argument(
        "--sync-dir",
        type=Path,
        default=None,
        help="Override this node's durable directory. Derived from --run-dir otherwise.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=int(os.environ.get("ROBODOJO_SHARD_INDEX", "0")),
        help=(
            "This node's position in a multi-node run. Defaults to "
            "$ROBODOJO_SHARD_INDEX or 0."
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=int(os.environ.get("ROBODOJO_SHARD_COUNT", "1")),
        help=(
            "How many nodes share the run, each claiming a disjoint slice. "
            "Defaults to $ROBODOJO_SHARD_COUNT or 1."
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _shard(specs: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    """Take this node's disjoint slice of the episode list.

    Strided rather than contiguous: the list is task-major and per-task cost
    varies by an order of magnitude (step_lim runs from 200 to 1900), so
    contiguous blocks would hand one node every long task. Striding interleaves
    them, which keeps nodes finishing at roughly the same time without any
    cross-node coordination.
    """
    if count < 1:
        raise ValueError("--shard-count must be at least 1")
    if not 0 <= index < count:
        raise ValueError(f"--shard-index must be in [0, {count}), got {index}")
    return specs[index::count]


def main() -> None:
    args = _parse_args()
    tasks_file = _repo_path(args.tasks_file)
    agent_config = _repo_path(args.agent_config)
    args.agent_config = agent_config
    if not args.policy_baseline:
        missing_credentials = _missing_provider_credentials(agent_config)
        if missing_credentials:
            raise ValueError(
                "missing provider credential(s): "
                + ", ".join(missing_credentials)
                + "; export them before starting the evaluation"
            )
    if args.agent_message_file is not None:
        message_path = _repo_path(args.agent_message_file)
        args.agent_message = message_path.read_text(encoding="utf-8").strip()
        if not args.agent_message:
            raise ValueError(f"agent message file is empty: {message_path}")
    if args.agent_python is not None:
        args.agent_python = args.agent_python.expanduser().absolute()
        if not args.agent_python.is_file():
            raise ValueError(f"agent Python executable does not exist: {args.agent_python}")
    tasks = _tasks(tasks_file, args.tasks)
    robodojo_root = args.robodojo_root.expanduser()
    layouts_per_task = _layouts_per_task(
        tasks,
        args.layouts,
        robodojo_root=robodojo_root,
        env_cfg=args.env_cfg,
        policy_seed=args.policy_seed,
    )
    if args.exclude_layouts:
        excluded = set(_parse_int_set(args.exclude_layouts))
        layouts_per_task = {
            task: [layout for layout in layouts if layout not in excluded]
            for task, layouts in layouts_per_task.items()
        }
        empty_tasks = [task for task, layouts in layouts_per_task.items() if not layouts]
        if empty_tasks:
            raise ValueError(
                "--exclude-layouts removed every candidate for task(s): "
                + ", ".join(empty_tasks)
            )
    if args.sample_layout_count is not None:
        layouts_per_task = _sample_layouts_per_task(
            layouts_per_task,
            count=args.sample_layout_count,
            seed=args.sample_seed,
        )
    devices = _resolve_devices(args.devices)
    if args.workers_per_device <= 0:
        raise ValueError("--workers-per-device must be positive")
    shard_name = f"node-{args.shard_index:02d}"
    if args.run_dir is not None:
        run_dir = args.run_dir.expanduser()
        sync_dir = (args.sync_dir or (run_dir / shard_name)).expanduser()
        output_dir = (
            args.output_dir or (LOCAL_SCRATCH_ROOT / run_dir.name / shard_name)
        ).expanduser().resolve()
    else:
        sync_dir = args.sync_dir.expanduser() if args.sync_dir else None
        if args.output_dir is None:
            run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = DEFAULT_OUTPUT_ROOT / run_name
        else:
            output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if sync_dir is not None:
        sync_dir.mkdir(parents=True, exist_ok=True)

    model_metadata = _model_metadata(agent_config)
    specs = [
        {
            "episode_key": f"{task}:layout-{layout}:seed-{args.policy_seed}",
            "task": task,
            "layout_id": layout,
            "policy_seed": args.policy_seed,
        }
        for task in tasks
        for layout in layouts_per_task[task]
    ]
    total_episodes = len(specs)
    specs = _shard(specs, args.shard_index, args.shard_count)
    results_path = output_dir / "results.jsonl"
    if args.resume:
        _seed_results_from_durable(results_path, sync_dir)
    previous = _read_previous_results(results_path) if args.resume else {}
    pending = [
        spec
        for spec in specs
        if spec["episode_key"] not in previous
        or str(previous[spec["episode_key"]].get("termination_reason", "")).startswith(
            "infrastructure_error:"
        )
    ]

    run_config = {
        "schema_version": "Emerge.robodojo_evaluation_config.v1",
        "created_at": _utc_now(),
        "tasks_file": str(tasks_file),
        "tasks": tasks,
        "layouts_selection": args.layouts,
        "excluded_layouts": args.exclude_layouts,
        "sample_layout_count": args.sample_layout_count,
        "sample_seed": args.sample_seed,
        "layouts_per_task": layouts_per_task,
        "devices": devices,
        "workers_per_device": args.workers_per_device,
        "persistent_workers": args.persistent_workers,
        "policy_seed": args.policy_seed,
        "policy_server_url": args.policy_server_url,
        "agent_config": str(agent_config),
        "agent_python": str(args.agent_python) if args.agent_python is not None else None,
        "agent_message": args.agent_message,
        "record_every_step": args.record_every_step,
        "archive_observations": args.archive_observations,
        "policy_baseline": args.policy_baseline,
        "episode_timeout_s": args.episode_timeout_s,
        "output_dir": str(output_dir),
        "sync_dir": str(sync_dir) if sync_dir else None,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "shard_episodes": len(specs),
        "total_episodes": total_episodes,
        **model_metadata,
    }
    _atomic_write_json(output_dir / "run_config.json", run_config)
    _write_summary(
        output_dir / "summary.json",
        previous,
        scheduled_episodes=len(specs),
        model_metadata=model_metadata,
    )
    print(
        f"[robodojo-eval] output={output_dir} "
        f"shard={args.shard_index}/{args.shard_count} "
        f"scheduled={len(specs)} of {total_episodes} "
        f"pending={len(pending)} workers={len(devices) * args.workers_per_device} "
        f"persistent={args.persistent_workers} "
        f"model={model_metadata['llm_model']}",
        flush=True,
    )
    if args.record_every_step:
        print(
            "[robodojo-eval] WARNING --record-every-step adds one three-camera "
            "render per control action; reserve it for diagnostic reruns.",
            flush=True,
        )
    if not pending:
        return

    slots = [
        (device, per_device_index)
        for device in devices
        for per_device_index in range(args.workers_per_device)
    ]
    workers = len(slots)
    all_results = dict(previous)
    result_lock = threading.Lock()
    with results_path.open("a", encoding="utf-8") as result_stream:
        def record_result(result: dict[str, Any]) -> None:
            with result_lock:
                all_results[result["episode_key"]] = result
                result_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                result_stream.flush()
                os.fsync(result_stream.fileno())
                _write_summary(
                    output_dir / "summary.json",
                    all_results,
                    scheduled_episodes=len(specs),
                    model_metadata=model_metadata,
                )
                _sync_completed_result(
                    output_dir=output_dir,
                    sync_dir=sync_dir,
                    result=result,
                )
                print(
                    f"[robodojo-eval] {len(all_results)}/{len(specs)} "
                    f"{result['episode_key']} success={result['success']} "
                    f"reason={result['termination_reason']} "
                    f"duration={result['duration_s']:.0f}s",
                    flush=True,
                )

        if args.persistent_workers:
            assignments = _persistent_slot_assignments(pending, workers)
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="robodojo-persistent-slot",
            ) as executor:
                futures = [
                    executor.submit(
                        _run_persistent_slot,
                        assignment,
                        args=args,
                        output_dir=output_dir,
                        device_id=device_id,
                        slot_index=slot_index,
                        model_metadata=model_metadata,
                        on_result=record_result,
                    )
                    for assignment, (device_id, slot_index) in zip(
                        assignments, slots, strict=True
                    )
                    if assignment
                ]
                for future in concurrent.futures.as_completed(futures):
                    future.result()
            return

        device_slots: queue.Queue[int] = queue.Queue()
        for device, _ in slots:
            device_slots.put(device)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="robodojo-episode",
        ) as executor:
            futures = {
                executor.submit(
                    _run_episode,
                    spec,
                    args=args,
                    output_dir=output_dir,
                    device_slots=device_slots,
                    model_metadata=model_metadata,
                ): spec
                for spec in pending
            }
            for future in concurrent.futures.as_completed(futures):
                record_result(future.result())


if __name__ == "__main__":
    main()
