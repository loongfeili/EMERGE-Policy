from pathlib import Path
import json,hashlib,os
src=Path('/mnt/hdfs/emerge_cache/models');dst=Path('/home/tiger/emerge-models');dst.mkdir(parents=True,exist_ok=True)
for item in json.loads((src/'manifest.json').read_text()):
 p=dst/item['path'];p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial');h=hashlib.sha256()
 with (src/item['path']).open('rb') as r,tmp.open('wb') as w:
  while b:=r.read(16*1024*1024):w.write(b);h.update(b)
 assert tmp.stat().st_size==item['size'] and h.hexdigest()==item['sha256'],item['path']
 os.replace(tmp,p)
 print('verified',item['path'],flush=True)
(dst/'download.exit').write_text('0\n')
