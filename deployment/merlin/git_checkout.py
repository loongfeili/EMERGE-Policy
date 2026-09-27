"""Fetch the declared fork revision; never overlay a source archive or dirty tree."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


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
    git(destination, "fetch", "--no-recurse-submodules", "origin", spec["branch"])
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
