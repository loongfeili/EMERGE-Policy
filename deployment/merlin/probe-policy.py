import os,json
from pathlib import Path
_services=json.loads(Path(os.environ['EMERGE_SERVICES_MANIFEST']).read_text())
import time, urllib.request
health_opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
health_deadline=time.monotonic()+240
while True:
    try:
        with health_opener.open(_services['policy_urls'][0].replace('ws://','http://')+'/healthz',timeout=5) as response:
            assert response.status==200
        break
    except Exception:
        if time.monotonic()>health_deadline:raise
        time.sleep(3)
import json,time,concurrent.futures
from pathlib import Path
import numpy as np
from robot.vla.robodojo_client import Pi05Client
from external_model_server.protocol import unpack_message
obs=unpack_message(Path('/home/tiger/emerge-smoke-bulk25/policy_observation.msgpack').read_bytes())
url=_services['policy_urls'][0]
def call(_):
 c=Pi05Client(url,timeout=120);t=time.monotonic()
 try:
  result=c.infer(obs);a=np.asarray(result['actions']);assert a.shape==(50,14) and np.isfinite(a).all(),a.shape
  return time.monotonic()-t
 finally:c.close()
report={}
for concurrency in [1, 4]:
 t=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:elapsed=list(pool.map(call,range(concurrency)))
 report[str(concurrency)]={'wall_seconds':time.monotonic()-t,'latencies':elapsed}
 print(concurrency,report[str(concurrency)],flush=True)
Path('/home/tiger/robodojo-setup/policy-probe.json').write_text(json.dumps({'status':'PASS','action_shape':[50,14],**report},indent=2))
