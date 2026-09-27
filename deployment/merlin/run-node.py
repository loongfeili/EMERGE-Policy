"""Run one disjoint evaluator shard with shared HDFS outputs and stop propagation."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,time,urllib.request,shutil
from result_snapshots import publish_final_results
from git_checkout import verify


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)); os.replace(temporary, path)


def rank_for(config, env):
    rank = int(env.get('ROBODOJO_SHARD_INDEX', env.get('ARNOLD_ID', '-1')))
    count = config['nodes']
    if not 0 <= rank < count:
        raise ValueError('Explicit ROBODOJO_SHARD_INDEX or ARNOLD_ID required and must fit node count')
    if env.get('ARNOLD_WORKER_NUM') and int(env['ARNOLD_WORKER_NUM']) != count:
        raise ValueError('Configured shard count differs from actual worker count')
    return rank


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config', required=True);args=parser.parse_args()
    config=json.loads(Path(args.config).read_text());rank=rank_for(config,os.environ)
    run=Path(config['result_root'])/config['run_id'];node=run/f'node-{rank:02d}'
    scratch=Path('/tmp/agenticvla-robodojo-eval')/config['run_id']/f'node-{rank:02d}'
    root=Path('/home/tiger/EMERGE-Policy');node.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    # Bound one process per shard, including across accidental duplicate launches.
    import fcntl
    lock=(scratch/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source_lock=json.loads((Path(args.config).parent/'source-lock.json').read_text())
    for spec in source_lock['repositories'].values():verify(spec,spec['destination'])
    probe={'rank':rank,'time':time.time(),'release':config['release']}
    atomic(node/'storage-check.json',probe);assert json.loads((node/'storage-check.json').read_text())==probe
    services=json.loads(Path(os.environ['EMERGE_LOCAL_SERVICES_MANIFEST']).read_text())
    assert services['emerge_commit']==source_lock['repositories']['emerge']['commit'], 'Inference source version differs'
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for url in services['policy_urls']+services['vggt_urls']+services['sam3_urls']:
        with opener.open(url.replace('ws://','http://')+'/healthz',timeout=5) as response:assert response.status==200
    private=Path('/home/tiger/.config/emerge-robodojo/agent.json')
    agent=json.loads(private.read_text())
    agent.setdefault('subagents',{}).setdefault('objectLocation',{}).update(vggtUrl=services['vggt_urls'][rank%len(services['vggt_urls'])],sam3Url=services['sam3_urls'][rank%len(services['sam3_urls'])])
    private.write_text(json.dumps(agent));private.chmod(0o600)
    urls=services['policy_urls'];offset=(rank*config['gpus_per_node'])%len(urls);urls=urls[offset:]+urls[:offset]
    command=[str(root/'.venv/bin/python'),'scripts/eval_robodojo_agent.py','--robodojo-root','/home/tiger/RoboDojo','--devices',','.join(map(str,range(config['gpus_per_node']))),'--workers-per-device',str(config['workers_per_device']),'--shard-index',str(rank),'--shard-count',str(config['nodes']),'--policy-server-url',','.join(urls),'--agent-config',str(private),'--agent-python',str(root/'.venv/bin/python'),'--policy-seed',str(config['policy_seed']),'--run-dir',str(run),'--tasks',config['tasks'],'--layouts',config['layouts'],'--resume']
    for filename in ['api-probe.json','policy-probe.json','perception-probe.json','standard-inventory.json','agent-packages.txt','simulator-packages.txt','graphics-env.sh']:
        source=Path('/home/tiger/robodojo-setup')/filename
        if source.exists():shutil.copyfile(source,node/filename)
    atomic(node/'deployment.json',{'config':config,'rank':rank,'source_lock':source_lock,'services':services,'command':command})
    if (run/'STOP.json').exists():raise RuntimeError('Shared STOP.json exists; inspect failure before a deliberate resume')
    for name in ['API_STOP.json','STORAGE_ERROR.json']:(scratch/name).unlink(missing_ok=True)
    environment={**os.environ,'PYTHONPATH':str(root),'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','ROBODOJO_RUN_ID':config['run_id']+f'-node-{rank:02d}'}
    started=time.time()
    local_log=scratch/'eval.log'
    with local_log.open('a') as log:
        process=subprocess.Popen(command,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                rows={}
                results=scratch/'results.jsonl'
                if results.exists():
                    for line in results.read_text().splitlines():
                        try:
                            row=json.loads(line);rows[row['episode_key']]=row
                        except (ValueError,KeyError):pass
                ordered=list(rows.values())
                if len(ordered)>=4 and all(str(r.get('termination_reason','')).startswith('infrastructure_error:') for r in ordered[-4:]):
                    atomic(scratch/'API_STOP.json',{'reason':'four_consecutive_infrastructure_errors'})
                local_stop=scratch/'API_STOP.json';global_stop=run/'STOP.json'
                if local_stop.exists() and not global_stop.exists():
                    atomic(global_stop,{'source_rank':rank,'detail':json.loads(local_stop.read_text()),'at':time.time()})
                if global_stop.exists() and not local_stop.exists():atomic(local_stop,json.loads(global_stop.read_text()))
                with local_log.open('rb') as stream:
                    stream.seek(max(0,local_log.stat().st_size-32768));tail=stream.read()
                temporary=node/'eval-tail.log.partial';temporary.write_bytes(tail);os.replace(temporary,node/'eval-tail.log')
                atomic(node/'status.json',{'state':'running','rank':rank,'completed':len(rows),'successes':sum(bool(r.get('success')) for r in rows.values()),'infrastructure_errors':sum(str(r.get('termination_reason','')).startswith('infrastructure_error:') for r in rows.values()),'updated_at':time.time(),'elapsed_s':time.time()-started})
            except OSError:
                atomic(scratch/'API_STOP.json',{'reason':'shared_storage_unavailable'})
                raise
            time.sleep(5)
    temporary=node/'eval.log.partial';shutil.copyfile(local_log,temporary);os.replace(temporary,node/'eval.log')
    result={'state':'completed' if process.returncode==0 and not (run/'STOP.json').exists() else 'stopped','returncode':process.returncode,'rank':rank,'updated_at':time.time(),'elapsed_s':time.time()-started}
    result['results_snapshot']=publish_final_results(node,scratch/'results.jsonl')
    atomic(node/'status.json',result)
    atomic(node/'runner-exit.json',result)
    if rank==0:
        subprocess.run([str(root/'.venv/bin/python'),str(Path(args.config).parent/'aggregate.py'),'--config',args.config],check=True)
    if result['state']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
