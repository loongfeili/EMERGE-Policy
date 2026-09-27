"""Audit terminal multi-node results, official verdicts and readable trajectories/videos."""
import argparse
import json
from pathlib import Path
import subprocess
from result_snapshots import read_node_results


def verify(run, release, config_name):
    cfg = json.loads((release / config_name).read_text())
    lock = json.loads((release / "source-lock.json").read_text())
    progress = json.loads((run / "final-progress.json").read_text())
    assert progress["coverage_complete"] and progress["all_shards_exited"] and not progress["pending_reads"], progress
    assert not (run / "STOP.json").exists(), "Run stopped on infrastructure failure"
    seen = set()
    evidence = []
    for rank in range(cfg["nodes"]):
        node = run / f"node-{rank:02d}"
        deployment = json.loads((node / "deployment.json").read_text())
        assert deployment["source_lock"] == lock
        assert deployment["config"] == cfg
        inventory = json.loads((node / "standard-inventory.json").read_text())
        assert inventory["task_entries"] == 54 and inventory["episodes"] == 2100
        status = json.loads((node / "runner-exit.json").read_text())
        assert status["state"] == "completed" and status["returncode"] == 0, status
        api = json.loads((node / "api-probe.json").read_text())
        assert api["passed"] == 1
        policy = json.loads((node / "policy-probe.json").read_text())
        assert policy["status"] == "PASS" and policy["action_shape"] == [50, 14]
        perception = json.loads((node / "perception-probe.json").read_text())
        assert perception["reference"]["status"].startswith("PASS")
        assert "vggt" in perception["reference"] and "sam3" in perception["reference"]
        runtime = json.loads((node / "run_config.json").read_text())
        scratch = Path(runtime["output_dir"])
        latest = {row["episode_key"]: row for row in read_node_results(node)}
        for key, row in latest.items():
            assert key not in seen, "Episode evaluated in multiple shards: " + key
            seen.add(key)
            assert not row["termination_reason"].startswith("infrastructure_error:"), row
            workspace = node / Path(row["workspace"]).relative_to(scratch)
            episode_status = json.loads((workspace / "episode_status.json").read_text())
            assert episode_status["finished"]
            verdict = next(value for value in episode_status["result"]["details"].values() if value["layout_id"] == row["layout_id"])
            assert bool(verdict["success"]) == bool(row["success"])
            assert abs(float(verdict["score"]) - float(row["official_score"])) < 1e-9
            assert (workspace / "ACTION.md").stat().st_size > 0
            sessions = list(workspace.rglob("*.jsonl"))
            assert sessions, "Missing session trace"
            videos = sorted((workspace / "artifacts/robodojo_video").glob("*.mp4"))
            assert len(videos) == 3, "Missing one or more camera videos"
            inspected = []
            for video in videos:
                response = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name,width,height,nb_frames", "-of", "json", str(video)], capture_output=True, text=True, timeout=60)
                assert response.returncode == 0, str(video)
                metadata = json.loads(response.stdout)
                assert float(metadata["format"]["duration"]) > 0 and metadata["streams"]
                inspected.append({"file": str(video), "bytes": video.stat().st_size, "metadata": metadata})
            evidence.append({"episode_key": key, "workspace": str(workspace), "official_verdict_matches": True, "success": row["success"], "score": row["official_score"], "videos": inspected})
    assert len(seen) == cfg["expected_episodes"], (len(seen), cfg["expected_episodes"])
    return {"passed": True, "episodes": len(seen), "nodes": cfg["nodes"], "source_lock": lock, "evidence": evidence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--config", default="verify.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run_dir, args.release, args.config)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("passed", "episodes", "nodes")}))
