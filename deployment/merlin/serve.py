"""Supervise independent 4-GPU model replicas and publish a versioned service pool."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request


def atomic(path, value):
    temporary = path.with_name(path.name + "." + str(os.getpid()) + ".partial")
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    rank = int(os.environ["ARNOLD_ID"])
    assert 0 <= rank < cfg["nodes"] and cfg["gpus_per_node"] % 4 == 0
    root = Path("/home/tiger/EMERGE-Policy")
    setup = Path("/home/tiger/emerge-setup")
    registry = Path(cfg["services_manifest"]).parent
    registry.mkdir(parents=True, exist_ok=True)
    lock = json.loads((Path(args.config).parent / "source-lock.json").read_text())
    commit = lock["repositories"]["emerge"]["commit"]
    addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
    host = next(ip for ip in addresses if ":" in ip and not ip.startswith("fdbd:fdbd:"))
    pools = {name: [] for name in ("policy_urls", "vggt_urls", "sam3_urls")}
    processes = []
    base = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
            "PYTHONPATH": str(root) + ":" + str(root / "third_party/openpi/src") + ":" + str(root / "third_party/openpi/packages/openpi-client/src")}
    common = ["--host", "::", "--max-batch-size", "1", "--batch-wait-ms", "0"]
    specs = [
        ("pi05", "policy_urls", "third_party/openpi/.venv/bin/python", "external_model_server.openpi_batch_server", ["--config-name", "pi05_base_aloha_full_sim_arx-x5_seed_0", "--checkpoint-dir", "/home/tiger/emerge-models/pi05"]),
        ("vggt", "vggt_urls", ".venv/bin/python", "external_model_server.vggt_server", ["--model-path", "/home/tiger/emerge-models/vggt.pt"]),
        ("sam3", "sam3_urls", ".venv/bin/python", "external_model_server.sam3_server", ["--model-path", "/home/tiger/emerge-models/sam3.pt"]),
        ("pi05", "policy_urls", "third_party/openpi/.venv/bin/python", "external_model_server.openpi_batch_server", ["--config-name", "pi05_base_aloha_full_sim_arx-x5_seed_0", "--checkpoint-dir", "/home/tiger/emerge-models/pi05"]),
    ]
    for gpu in range(cfg["gpus_per_node"]):
        name, pool, python, module, arguments = specs[gpu % 4]
        port = 8000 + gpu
        log = (setup / f"{name}-{gpu}.log").open("w")
        processes.append(subprocess.Popen([str(root / python), "-u", "-m", module, *arguments, "--port", str(port), *common], cwd=root, env={**base, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=log, stderr=subprocess.STDOUT))
        log.close()
        pools[pool].append(f"ws://[{host}]:{port}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 1800
    own_manifest = registry / f"node-{rank:02d}.json"
    def terminate(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A model service exited; see service logs")
            try:
                for urls in pools.values():
                    for url in urls:
                        with opener.open(url.replace("ws://", "http://") + "/healthz", timeout=5) as response:
                            assert response.status == 200
                break
            except (OSError, AssertionError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(5)
        while True:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A model service exited")
            atomic(own_manifest, {**pools, "rank": rank, "emerge_commit": commit, "updated_at": time.time(), "state": "healthy"})
            if rank == 0:
                try:
                    nodes = [json.loads((registry / f"node-{i:02d}.json").read_text()) for i in range(cfg["nodes"])]
                    assert all(n["state"] == "healthy" and n["emerge_commit"] == commit and time.time() - n["updated_at"] < 90 for n in nodes)
                    merged = {name: [url for n in nodes for url in n[name]] for name in pools}
                    atomic(Path(cfg["services_manifest"]), {**merged, "emerge_commit": commit, "updated_at": time.time(), "nodes": cfg["nodes"]})
                except (OSError, ValueError, KeyError, AssertionError):
                    Path(cfg["services_manifest"]).unlink(missing_ok=True)
            time.sleep(10)
    finally:
        atomic(own_manifest, {"rank": rank, "state": "stopped", "updated_at": time.time()})
        if rank == 0:
            Path(cfg["services_manifest"]).unlink(missing_ok=True)
        for p in processes:
            p.terminate()
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
        import shutil
        for path in setup.glob("*.log"):
            shutil.copyfile(path, registry / f"node-{rank:02d}-{path.name}")


if __name__ == "__main__":
    main()
