"""Publish an immutable launcher/config release; application source comes from GitHub."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def resources(kind, nodes, gpus):
    infer = kind == "infer"
    return {"resource_backend": "arnold", "arnold_resource_config": {
        "cluster_ids": [44 if infer else 20], "cluster_select_mode": "single", "group_id": 1894 if infer else 402,
        "roles": [{"name": "worker", "num": nodes, "gpu": gpus,
                   "gpu_type": "A100_SXM_80GB" if infer else "NVIDIA_L20",
                   "cpu": 14 * gpus if infer else 22 * gpus,
                   "memory_mb": 231424 * gpus if infer else 117760 * gpus,
                   "ports": max(10, gpus + 2),
                   "queue_name": "a100-sxm-80gb.hpccluster-ydfgrrp7ac9tiffwmqs7.ai" if infer else "compute-85-hl-cloudnative-ai-ailab.nlp-guarantee"}]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--cache", type=Path, default=Path("/mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925"))
    parser.add_argument("--robodojo-commit", default="b08b49c081953bb3302d079a383c9e059f952f0d")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.release_id):
        raise ValueError("Invalid release name")
    scripts = Path(__file__).resolve().parent
    repository = scripts.parents[1]
    commit = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repository), "diff", "HEAD", "--ignore-submodules=all"], text=True):
        raise ValueError("Commit tracked changes before publishing")
    remote = subprocess.check_output(["git", "ls-remote", "git@github.com:loongfeili/EMERGE-Policy.git", "refs/heads/robodojo"], text=True).split()[0]
    if remote != commit:
        raise ValueError("Push the current robodojo commit before publishing")
    destination = args.cache / "releases" / args.release_id
    if destination.exists():
        raise ValueError("Release already exists; use a new ID")
    destination.mkdir()
    for path in scripts.iterdir():
        if path.suffix in (".py", ".sh", ".md"):
            shutil.copyfile(path, destination / path.name)
    legacy = args.cache / "releases/cluster-20260926-r11-8"
    legacy_manifest = json.loads((legacy / "release.json").read_text())
    dependencies = ["isaaclab-full.tar.gz", "cl100k_base.tiktoken", "simulator-offline.txt", "perception-reference.tar.gz"]
    dependencies += [p.name for p in legacy.glob("*.whl")]
    for name in dependencies:
        assert sha(legacy / name) == legacy_manifest["files"][name], name
        shutil.copyfile(legacy / name, destination / name)
    cache_roles = {
        'assets/assets.tar.zst': ['eval'], 'assets/assets.sha256': ['eval'],
        'runtime/robodojo-venv.tar.zst': ['eval'], 'runtime/robodojo-venv.sha256': ['eval'],
        'runtime/host-compat.tar.zst': ['eval'], 'runtime/isaac-extension-cache.tar.zst': ['eval'],
        'runtime/agent-venv-cluster-20260926-r3.tar.zst': ['eval'],
        'runtime/agent-venv-cluster-20260926-r3.tar.json': ['eval'],
        'runtime/zstd': ['eval','infer'], 'runtime/uv': ['eval','infer'],
        'runtime/glibc235.tar.gz': ['eval','infer'], 'runtime/python311.tar.gz': ['infer'],
        'runtime/lerobot-git-cache.tar.gz': ['infer'], 'runtime/paligemma_tokenizer.model': ['infer'],
        'source/robodojo-submodules.tar.gz': ['eval'], 'source/patch-python-runtime.sh': ['eval'],
        'source/activate.sh': ['eval'], 'models/manifest.json': ['infer'],
    }
    for dependency in ['openpi','sam3','vggt']:
        cache_roles['runtime/'+dependency+'-source.tar']=['eval','infer']
    for path in (args.cache/'runtime').glob('nvidia-driver-*'):
        if path.is_file():cache_roles['runtime/'+path.name]=['eval']
    if (args.cache/'runtime/inference-v1/manifest.json').exists():
        cache_roles['runtime/inference-v1/manifest.json']=['infer']
    # Large archives already have recorded SHA values; each worker checks their bytes.
    known={x['file']:x['sha256'] for x in json.loads((args.cache/'archive-recheck-20260926.json').read_text())}
    cache_lock={name:{'sha256': known[name] if name in known else sha(args.cache/name), 'roles':roles} for name,roles in cache_roles.items()}
    (destination/'cache-lock.json').write_text(json.dumps(cache_lock,indent=2))
    lock = {"repositories": {
        "emerge": {"url": "https://github.com/loongfeili/EMERGE-Policy.git", "branch": "robodojo", "commit": commit, "destination": "/home/tiger/EMERGE-Policy"},
        "robodojo": {"url": "https://github.com/loongfeili/RoboDojo.git", "branch": "main", "commit": args.robodojo_commit, "destination": "/home/tiger/RoboDojo"},
    }, "dependency_snapshot": "cluster-20260926-r11-8", "dependency_files": {n: sha(destination / n) for n in dependencies}}
    (destination / "source-lock.json").write_text(json.dumps(lock, indent=2))
    attachments = [{"kind": "hdfs_volume", "hdfs_volume_attachment": {"hdfs_volumes": [
        {"access_mode": "RO", "extra": "", "mnt": "/mnt/hdfs/emerge_cache", "path": "hdfs://harunawl/home/byte_data_seed_wl/user/loongfei.li03/emerge_robodojo_20260925"},
        {"access_mode": "RW", "extra": "", "mnt": "/mnt/hdfs/emerge_output", "path": "hdfs://haruna/home/byte_data_seed/ssd_hldy/user/lilongfei.xjgm"},
    ]}}]
    output = "/mnt/hdfs/emerge_output/emerge_merlin/" + args.release_id
    shared = {"release": args.release_id, "release_mount": "/mnt/hdfs/emerge_cache/releases/" + args.release_id,
              "result_root": output + "/runs", "services_manifest": output + "/services/pool.json",
              "attachments": attachments, "policy_seed": 0, "image_vid": "d8h852v1enldjpkjjr7g"}
    for name, nodes, gpus, workers, tasks, layouts, count in [
        ("verify", 2, 1, 1, "stack_bowls,build_tower", "0,1", 4),
        ("full8", 1, 8, 2, "all", "native", 2100),
        ("full16", 2, 8, 2, "all", "native", 2100),
        ("full32", 4, 8, 2, "all", "native", 2100),
        ("full64", 8, 8, 2, "all", "native", 2100),
    ]:
        cfg = {**shared, "kind": "eval", "baseline_job": "433b239fc535fb2b", "run_id": args.release_id + "-" + name + "-seed0",
               "nodes": nodes, "gpus_per_node": gpus, "workers_per_device": workers, "tasks": tasks, "layouts": layouts,
               "expected_episodes": count, "resource_config": resources("eval", nodes, gpus)}
        (destination / (name + ".json")).write_text(json.dumps(cfg, indent=2))
    for name, nodes, gpus, build in [("infer-build", 1, 4, True), ("infer", 1, 4, False), ("infer8", 1, 8, False), ("infer-scale8", 2, 4, False)]:
        cfg = {**shared, "kind": "infer", "baseline_job": "14c201bfa03ae16d", "run_id": args.release_id + "-" + name,
               "nodes": nodes, "gpus_per_node": gpus, "build_inference_runtime": build,
               "inference_runtime": "/mnt/hdfs/emerge_cache/runtime/inference-v1",
               "runtime_build_output": output + "/runtime-build", "resource_config": resources("infer", nodes, gpus)}
        (destination / (name + ".json")).write_text(json.dumps(cfg, indent=2))
    manifest = {"release": args.release_id, "files": {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()}}
    (destination / "release.json").write_text(json.dumps(manifest, indent=2))
    for name, digest in manifest["files"].items():
        assert sha(destination / name) == digest
    print(json.dumps({"release": str(destination), "commit": commit, "manifest_sha256": sha(destination / "release.json"), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
