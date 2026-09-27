import importlib.util
import json
from pathlib import Path


def evaluator():
    path = Path(__file__).parents[1] / 'scripts/eval_robodojo_agent.py'
    spec = importlib.util.spec_from_file_location('eval_storage', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def episode(root):
    directory = root / 'episodes/stack_bowls/layout_00_seed_0/attempt_1'
    directory.mkdir(parents=True)
    (directory / 'trace.json').write_text('{"called": "vla"}')
    result = {'episode_key': 'stack_bowls:layout-0:seed-0', 'episode_dir': str(directory), 'success': False, 'persistent_worker': True}
    (root / 'results.jsonl').write_text(json.dumps(result) + '\n')
    (root / 'summary.json').write_text('{"finished_episodes": 1}')
    return result


def test_storage_failure_stops_dispatch_and_preserves_local_score(tmp_path, monkeypatch):
    module = evaluator()
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    result = episode(local)
    def fail(*args, **kwargs):
        raise PermissionError('storage write denied')
    monkeypatch.setattr(module.shutil, 'copytree', fail)
    module._sync_completed_result(output_dir=local, sync_dir=remote, result=result)
    assert json.loads((local / 'API_STOP.json').read_text())['reason'] == 'result_sync_failed'
    assert json.loads((local / 'STORAGE_ERROR.json').read_text())['episode_key'] == result['episode_key']
    assert not (remote / 'results.jsonl').exists()
    assert json.loads((local / 'results.jsonl').read_text())['success'] is False


def test_results_are_published_after_artifacts_and_as_complete_files(tmp_path, monkeypatch):
    module = evaluator()
    local, remote = tmp_path / 'local', tmp_path / 'remote'
    result = episode(local)
    replace = module.os.replace
    def inspect(source, target):
        if target.name == 'results.jsonl':
            assert (remote / Path(result['episode_dir']).relative_to(local) / 'trace.json').exists()
            assert json.loads(Path(source).read_text())['episode_key'] == result['episode_key']
        return replace(source, target)
    monkeypatch.setattr(module.os, 'replace', inspect)
    module._sync_completed_result(output_dir=local, sync_dir=remote, result=result)
    assert (remote / 'results.jsonl').read_bytes() == (local / 'results.jsonl').read_bytes()
    assert not (local / 'API_STOP.json').exists()


def test_eight_nodes_cover_all_2100_episodes_exactly_once():
    module = evaluator()
    cases = [{'episode_key': str(i)} for i in range(2100)]
    shards = [module._shard(cases, i, 8) for i in range(8)]
    keys = [item['episode_key'] for shard in shards for item in shard]
    assert len(keys) == len(set(keys)) == 2100
    assert set(keys) == {item['episode_key'] for item in cases}
    assert max(map(len, shards)) - min(map(len, shards)) == 1
