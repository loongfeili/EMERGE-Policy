"""Seal a built model runtime once; ordinary inference trials restore verified bytes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["restore", "seal"])
    args = parser.parse_args()
    cfg = json.loads(Path(os.environ["EMERGE_CONFIG"]).read_text())
    root = Path("/home/tiger/EMERGE-Policy")
    if args.operation == "restore":
        directory = Path(cfg["inference_runtime"])
        manifest = json.loads((directory / "manifest.json").read_text())
        for entry in manifest["archives"]:
            archive = directory / entry["file"]
            assert archive.stat().st_size == entry["bytes"] and digest(archive) == entry["sha256"], archive.name
            subprocess.run(["tar", "-I", "zstd", "-xf", str(archive), "-C", str(root)], check=True)
    else:
        if not cfg.get("build_inference_runtime"):
            raise ValueError("Only the explicit runtime-builder config may seal a runtime")
        directory = Path(cfg["runtime_build_output"])
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "manifest.json").exists():
            raise ValueError("Refusing to overwrite a sealed runtime")
        entries = []
        for name, member in [("perception.tar.zst", ".venv"), ("policy.tar.zst", "third_party/openpi/.venv")]:
            local = Path("/tmp") / name
            subprocess.run(["tar", "-I", "zstd -T4 -3", "-cf", str(local), "-C", str(root), member], check=True)
            import shutil
            target = directory / name
            shutil.copyfile(local, target)
            sha = digest(local)
            assert digest(target) == sha, "HDFS runtime readback differs"
            entries.append({"file": name, "bytes": local.stat().st_size, "sha256": sha})
            local.unlink()
        packages = {}
        for key, python in [("perception", root / ".venv/bin/python"), ("policy", root / "third_party/openpi/.venv/bin/python")]:
            packages[key] = subprocess.check_output(["uv", "pip", "freeze", "--python", str(python)], text=True)
        lock = json.loads((Path(os.environ["EMERGE_RELEASE"]) / "source-lock.json").read_text())
        (directory / "manifest.json").write_text(json.dumps({"archives": entries, "packages": packages, "build_source_lock": lock}, indent=2))


if __name__ == "__main__":
    main()
