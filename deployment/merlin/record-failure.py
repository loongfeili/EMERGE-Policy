"""Propagate pre-evaluation and runner startup failures to all cluster shards."""
from pathlib import Path
import argparse,json,os,time

def record_failure(config,rank,phase,code):
    if not 0 <= rank < config['nodes']:
        raise ValueError('rank outside configured node count')
    run=Path(config['result_root'])/config['run_id'];node=run/f'node-{rank:02d}';node.mkdir(parents=True,exist_ok=True)
    data={'reason':'node_failure','source_rank':rank,'phase':phase,'exit_code':code,'at':time.time()}
    def write(path,value):
        tmp=path.with_name('.'+path.name+'.'+str(os.getpid())+'.tmp');tmp.write_text(json.dumps(value,indent=2));os.replace(tmp,path)
    write(node/'startup-failure.json',data)
    write(node/'workflow-status.json',{'phase':'failed','rank':rank,'at':data['at'],'error':data})
    if not (run/'STOP.json').exists():write(run/'STOP.json',data)
    return data

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--phase',required=True);parser.add_argument('--exit-code',type=int,required=True);args=parser.parse_args()
    record_failure(json.loads(args.config.read_text()),int(os.environ['ARNOLD_ID']),args.phase,args.exit_code)
