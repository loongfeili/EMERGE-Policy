"""Fetch the declared revision from a verified bundle or the declared fork."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def git(root, *args):
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_", "GIT_TRACE")) and key != "GIT_CONFIG_COUNT"}
    environment.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null",
                       GIT_TERMINAL_PROMPT="0", GIT_LFS_SKIP_SMUDGE="1")
    return subprocess.check_output(["git", "-c", "http.version=HTTP/1.1", "-C", str(root), *args],
                                   text=True, env=environment, timeout=240).strip()


def fetch(destination, branch, remote="origin"):
    for attempt in range(5):
        try:
            git(destination, "fetch", "--no-recurse-submodules", remote, branch)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if attempt == 4:
                raise
            print(f"Git fetch failed; retry {attempt + 2}/5 for {destination.name}", flush=True)
            time.sleep(3 * 2 ** attempt)


def checkout(spec, destination):
    revision = spec["commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("An exact 40-character Git commit is required")
    destination = Path(destination)
    if not (destination / ".git").exists():
        if destination.exists() and any(destination.iterdir()):
            raise ValueError(f"Refusing to replace a nonempty source directory: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        git(destination, "init", "-q")
        git(destination, "remote", "add", "origin", spec["url"])
    if git(destination, "remote", "get-url", "origin") != spec["url"]:
        raise ValueError("Existing checkout belongs to a different repository")
    if git(destination, "diff", "--ignore-submodules=all") or git(destination, "diff", "--cached", "--ignore-submodules=all"):
        raise ValueError("Refusing to discard local source changes")
    if spec.get("bundle"):
        bundle = Path(spec["bundle"])
        expected = spec.get("bundle_sha256", "")
        if not bundle.is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Source bundle requires an absolute path and SHA256")
        digest = hashlib.sha256()
        with bundle.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("Source bundle integrity failure")
        fetch(destination, spec["branch"], str(bundle))
    else:
        fetch(destination, spec["branch"])
    git(destination, "cat-file", "-e", revision + "^{commit}")
    git(destination, "merge-base", "--is-ancestor", revision, "FETCH_HEAD")
    git(destination, "checkout", "--detach", revision)
    verify(spec, destination)


def verify(spec, destination):
    if git(destination, "rev-parse", "HEAD") != spec["commit"]:
        raise ValueError("Deployed commit differs from release")
    if git(destination, "diff", "HEAD", "--ignore-submodules=all"):
        raise ValueError("Tracked deployment source was modified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lock")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--only", choices=["emerge", "robodojo"])
    args = parser.parse_args()
    for name, spec in json.loads(Path(args.lock).read_text())["repositories"].items():
        if args.only and name != args.only:
            continue
        (verify if args.verify else checkout)(spec, spec["destination"])
        print(json.dumps({"repository": name, "commit": spec["commit"], "verified": True}))
