"""Immutable final shard data avoids stale FUSE size/content cache combinations."""
import hashlib
import json
from pathlib import Path
import uuid


def publish_final_results(node: Path, source: Path):
    raw = source.read_bytes() if source.exists() else b''
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    name = 'results-final-' + uuid.uuid4().hex + '.jsonl'
    temporary = node / ('.' + name)
    temporary.write_bytes(raw)
    temporary.rename(node / name)
    return {'file': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'rows': len(rows)}


def read_node_results(node: Path):
    exit_file = node / 'runner-exit.json'
    exit_status = json.loads(exit_file.read_text()) if exit_file.exists() else {}
    snapshot = exit_status.get('results_snapshot')
    if snapshot:
        name = snapshot['file']
        if Path(name).name != name:
            raise ValueError('Snapshot path must remain inside its shard')
        raw = (node / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != snapshot['sha256']:
            raise ValueError('Final results snapshot checksum mismatch')
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(rows) != snapshot['rows']:
            raise ValueError('Final results snapshot count mismatch')
        return rows
    source = node / 'results.jsonl'
    return [json.loads(line) for line in source.read_text().splitlines() if line.strip()] if source.exists() else []
