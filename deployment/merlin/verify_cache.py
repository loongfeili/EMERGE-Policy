"""Check immutable cache inputs needed by a worker before extracting or executing them."""
import argparse
import hashlib
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lock")
    parser.add_argument("kind", choices=["eval", "infer"])
    parser.add_argument("--cache", default="/mnt/hdfs/emerge_cache")
    args = parser.parse_args()
    entries = json.loads(Path(args.lock).read_text())
    for name, entry in entries.items():
        if args.kind not in entry["roles"]:
            continue
        source = Path(args.cache) / name
        h = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(16 * 1024 * 1024):
                h.update(chunk)
        if h.hexdigest() != entry["sha256"]:
            raise ValueError("Cache integrity failure: " + name)
        print("Verified cache", name, flush=True)
