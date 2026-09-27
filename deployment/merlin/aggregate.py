"""Summarize validated final snapshots; dynamic FUSE reads may need another poll."""
from pathlib import Path
import argparse,json,subprocess,time,os,tempfile
from result_snapshots import read_node_results
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--once',action='store_true');a=p.parse_args();cfg=json.loads(Path(a.config).read_text());run=Path(cfg['result_root'])/cfg['run_id'];root=Path('/home/tiger/EMERGE-Policy')
def publish(name,raw):
 target=run/name;tmp=run/('.'+name+'.'+str(os.getpid()));tmp.write_bytes(raw);os.replace(tmp,target)
with tempfile.TemporaryDirectory(prefix='robodojo-summary-') as temporary:
 local=Path(temporary)
 while True:
  seen={};duplicates=[];pending=[]
  for node in sorted(run.glob('node-*')):
   try:rows=read_node_results(node)
   except (ValueError,OSError):pending.append(node.name);continue
   latest={r['episode_key']:r for r in rows}
   for key,row in latest.items():
    if key in seen:duplicates.append(key)
    seen[key]=row
  if duplicates:raise ValueError('Duplicate episodes across nodes: '+str(duplicates))
  summary=None
  if seen:
   (local/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in seen.values()))
   subprocess.run([str(root/'.venv/bin/python'),'scripts/summarize_robodojo_eval.py',str(local),'--expected-episodes',str(cfg['expected_episodes']),'--output',str(local/'summary.json')],cwd=root,check=True)
   summary=(local/'summary.json').read_bytes();publish('official-summary.json',summary)
  exits=list(run.glob('node-*/runner-exit.json'));failed=[]
  for status in run.glob('node-*/workflow-status.json'):
   try:
    if json.loads(status.read_text()).get('phase')=='failed':failed.append(status.parent.name)
   except (ValueError,OSError):pending.append(status.parent.name)
  done=len(exits)==cfg['nodes'];payload={'completed':len(seen),'expected':cfg['expected_episodes'],'failed_shards':failed,'pending_reads':pending,'shards_finished':len(exits),'shards_expected':cfg['nodes'],'coverage_complete':len(seen)==cfg['expected_episodes'],'all_shards_exited':done,'paused':(run/'STOP.json').exists(),'updated_at':time.time()}
  raw=json.dumps(payload,indent=2).encode();publish('progress.json',raw)
  if done and not pending:
   # These paths are written only at terminal state; readers need not follow a mutable file.
   publish('final-progress.json',raw)
   if summary is not None:publish('final-official-summary.json',summary)
   break
  if a.once or failed or (run/'STOP.json').exists():break
  time.sleep(15)
