#!/usr/bin/env python3
"""Merge per-node RoboDojo shards into one benchmark result.

Every node writes its own ``results.jsonl``; shards are disjoint by
construction, so merging is a union keyed on ``episode_key`` rather than a
reconciliation. Re-running is safe and is the intended way to watch a
distributed run: it reports coverage against the expected episode count so an
unfinished or dead shard is visible rather than silently shrinking the
denominator.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).absolute().parents[1]

# RoboDojo's official seed-level table reports 42 task cells.  The 12
# generalization cells merge 25 base layouts with 25 ``_random`` layouts; the
# other cells contain 50 layouts.  Dimension and overall scores are macro
# averages, matching RoboDojo/scripts/internal/summarize_result.py.
ROBODOJO_DIMENSIONS = {
    "Generalization": [
        "stack_bowls",
        "push_T",
        "pack_objects_into_box",
        "fold_clothes",
        "hang_mugs",
        "sweep_blocks",
        "pour_liquid_into_cup",
        "make_toast",
        "arrange_largest_number",
        "sort_nesting_dolls_by_size",
        "store_laptop_and_headphones",
        "stack_blocks",
    ],
    "Precision": [
        "fasten_screws",
        "plug_in_charger",
        "insert_tubes",
        "pour_balls_into_vase",
        "play_Xylophone",
        "deposit_coin",
        "insert_key",
        "build_tower",
    ],
    "Long-Horizon": [
        "put_bottles_into_dustbin",
        "fill_pen_holder",
        "classify_objects",
        "play_tic_tac_toe",
        "fill_egg_holder",
        "organize_table",
        "make_kong",
        "play_stacking_toy",
    ],
    "Memory": [
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_blocks",
        "swap_T",
        "press_by_number",
        "imitate_sorting_sequence",
    ],
    "Open": [
        "align_blocks",
        "general_pickup",
        "stack_blocks_by_language",
        "solve_equation",
        "classify_objects_by_language",
        "pick_from_conveyor_by_image",
        "store_tools_in_toolbox",
        "pour_by_language",
    ],
}
ROBODOJO_GENERALIZATION_TASKS = set(ROBODOJO_DIMENSIONS["Generalization"])
ROBODOJO_OFFICIAL_EPISODES = 2100


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
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


def _discover_shards(run_root: Path) -> list[Path]:
    """Find every shard's results.jsonl under a run root."""
    direct = run_root / "results.jsonl"
    if direct.exists():
        return [direct]
    return sorted(run_root.glob("*/results.jsonl"))


def _official_episode_score(item: dict[str, Any]) -> float | None:
    """Use the official partial credit, independently of binary success."""
    value = item.get("official_score")
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid official score: {value}")
    return value * 100.0


def _official_protocol_summary(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    """Build RoboDojo's official 42-task, five-dimension seed-0 table."""

    by_task: dict[str, dict[int, dict[str, Any]]] = {}
    for item in evaluated:
        try:
            layout_id = int(item["layout_id"])
        except (KeyError, TypeError, ValueError):
            continue
        by_task.setdefault(str(item.get("task")), {})[layout_id] = item

    task_scores: dict[str, dict[str, Any]] = {}
    for dimension_tasks in ROBODOJO_DIMENSIONS.values():
        for task in dimension_tasks:
            if task in ROBODOJO_GENERALIZATION_TASKS:
                split_names = (task, f"{task}_random")
                expected_per_split = 25
            else:
                split_names = (task,)
                expected_per_split = 50

            selected: list[dict[str, Any]] = []
            split_coverage: dict[str, dict[str, int]] = {}
            complete = True
            for split_name in split_names:
                layouts = by_task.get(split_name, {})
                split_items = [
                    layouts[index]
                    for index in range(expected_per_split)
                    if index in layouts
                ]
                selected.extend(split_items)
                split_coverage[split_name] = {
                    "finished_episodes": len(split_items),
                    "expected_episodes": expected_per_split,
                }
                complete &= len(split_items) == expected_per_split

            official_scores = [
                score
                for item in selected
                if (score := _official_episode_score(item)) is not None
            ]
            valid = [item for item in selected if not item.get("official_excluded")]
            score_complete = len(official_scores) == len(valid) and bool(valid)
            successes = sum(int(bool(item.get("success"))) for item in valid)
            task_scores[task] = {
                "complete": complete and score_complete,
                "finished_episodes": len(selected),
                "official_excluded_episodes": len(selected) - len(valid),
                "expected_episodes": sum(
                    split["expected_episodes"] for split in split_coverage.values()
                ),
                "successes": successes,
                "success_rate_percent": (
                    successes / len(valid) * 100.0 if valid else None
                ),
                "score_percent": (
                    statistics.mean(official_scores) if official_scores else None
                ),
                "splits": split_coverage,
            }

    dimension_scores: dict[str, dict[str, Any]] = {}
    for dimension, tasks in ROBODOJO_DIMENSIONS.items():
        completed = [task_scores[task] for task in tasks if task_scores[task]["complete"]]
        dimension_scores[dimension] = {
            "complete": len(completed) == len(tasks),
            "completed_tasks": len(completed),
            "expected_tasks": len(tasks),
            "success_rate_percent": (
                statistics.mean(row["success_rate_percent"] for row in completed)
                if completed
                else None
            ),
            "score_percent": (
                statistics.mean(row["score_percent"] for row in completed)
                if completed
                else None
            ),
        }

    completed_dimensions = [
        row for row in dimension_scores.values() if row["score_percent"] is not None
    ]
    complete_tasks = sum(int(row["complete"]) for row in task_scores.values())
    return {
        "protocol": "RoboDojo arx_x5 seed0 official",
        "complete": (
            len(evaluated) == ROBODOJO_OFFICIAL_EPISODES
            and complete_tasks == len(task_scores)
        ),
        "finished_episodes": len(evaluated),
        "expected_episodes": ROBODOJO_OFFICIAL_EPISODES,
        "completed_tasks": complete_tasks,
        "expected_tasks": len(task_scores),
        "task_scores": task_scores,
        "dimension_scores": dimension_scores,
        "overall": {
            "success_rate_percent": (
                statistics.mean(row["success_rate_percent"] for row in completed_dimensions)
                if completed_dimensions
                else None
            ),
            "score_percent": (
                statistics.mean(row["score_percent"] for row in completed_dimensions)
                if completed_dimensions
                else None
            ),
        },
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def summarize(
    results: dict[str, dict[str, Any]],
    *,
    expected_episodes: int | None,
    shards: dict[str, int],
) -> dict[str, Any]:
    items = list(results.values())
    evaluated = [
        item
        for item in items
        if not str(item.get("termination_reason", "")).startswith("infrastructure_error:")
    ]
    processed = evaluated
    evaluated = [item for item in processed if not item.get("official_excluded")]
    excluded = len(processed) - len(evaluated)
    successes = sum(int(bool(item.get("success"))) for item in evaluated)
    official_scores = [
        score
        for item in evaluated
        if (score := _official_episode_score(item)) is not None
    ]

    per_task: dict[str, dict[str, Any]] = {}
    for item in evaluated:
        task = str(item["task"])
        entry = per_task.setdefault(
            task,
            {"episodes": 0, "successes": 0, "official_scores": []},
        )
        entry["episodes"] += 1
        entry["successes"] += int(bool(item.get("success")))
        score = _official_episode_score(item)
        if score is not None:
            entry["official_scores"].append(score)
    for entry in per_task.values():
        entry["success_rate"] = (
            entry["successes"] / entry["episodes"] if entry["episodes"] else 0.0
        )
        scores = entry.pop("official_scores")
        entry["official_score_count"] = len(scores)
        entry["official_score_sum"] = sum(scores)
        entry["official_score_mean"] = statistics.mean(scores) if scores else None

    models = {str(item.get("llm_model")) for item in items if item.get("llm_model")}
    return {
        "schema_version": "Emerge.robodojo_evaluation_summary.v1",
        "expected_episodes": expected_episodes,
        "finished_episodes": len(items),
        "benchmark_episodes": len(evaluated),
        "infrastructure_errors": len(items) - len(evaluated) - excluded,
        "official_excluded_episodes": excluded,
        "infrastructure_error_kinds": dict(
            Counter(
                str(item.get("termination_reason"))
                for item in items
                if str(item.get("termination_reason", "")).startswith("infrastructure_error:")
            )
        ),
        "successes": successes,
        "success_rate": successes / len(evaluated) if evaluated else 0.0,
        "official_score_count": len(official_scores),
        "official_score_sum": sum(official_scores),
        "official_score_mean": (
            statistics.mean(official_scores) if official_scores else None
        ),
        "official_score_median": (
            statistics.median(official_scores) if official_scores else None
        ),
        "official_score_distribution": dict(
            sorted(Counter(official_scores).items())
        ),
        "per_task": dict(sorted(per_task.items())),
        "official_protocol_seed0": _official_protocol_summary(processed),
        "episodes_per_shard": shards,
        "llm_models": sorted(models),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root",
        type=Path,
        help="Directory holding one subdirectory per shard, or a single shard.",
    )
    parser.add_argument(
        "--expected-episodes",
        type=int,
        default=None,
        help="Episode count of the full protocol, for coverage reporting.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root.expanduser()
    shard_paths = _discover_shards(run_root)
    if not shard_paths:
        raise SystemExit(f"no results.jsonl found under {run_root}")

    merged: dict[str, dict[str, Any]] = {}
    shards: dict[str, int] = {}
    for path in shard_paths:
        shard_results = _load_results(path)
        name = path.parent.name if path.parent != run_root else "."
        shards[name] = len(shard_results)
        overlap = merged.keys() & shard_results.keys()
        if overlap:
            print(
                f"[summarize] WARNING {len(overlap)} episode(s) reported by more "
                f"than one shard, keeping the newer result: {sorted(overlap)[:3]}...",
                flush=True,
            )
        merged.update(shard_results)

    summary = summarize(
        merged,
        expected_episodes=args.expected_episodes,
        shards=shards,
    )
    output = args.output or (run_root / "summary_merged.json")
    _atomic_write_json(output, summary)

    covered = summary["finished_episodes"]
    expected = summary["expected_episodes"]
    coverage = f"{covered}/{expected}" if expected else str(covered)
    print(
        f"[summarize] shards={len(shard_paths)} episodes={coverage} "
        f"benchmark={summary['benchmark_episodes']} "
        f"infra_errors={summary['infrastructure_errors']} "
        f"success_rate={summary['success_rate']:.4f}",
        flush=True,
    )
    print(f"[summarize] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
