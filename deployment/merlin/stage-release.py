"""Copy one sealed HDFS release to local disk and verify every byte before execution."""
from pathlib import Path
import os,json,hashlib,shutil,sys
source=Path(sys.argv[1]);expected=sys.argv[2]
raw=(source/'release.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()==expected,'release manifest checksum mismatch'
manifest=json.loads(raw);destination=Path('/home/tiger/emerge-releases')/source.name
staging=destination.with_name(destination.name+'.partial-'+str(os.getpid()));staging.mkdir(parents=True,exist_ok=False)
try:
 for name,digest in manifest['files'].items():
  assert Path(name).name==name,name
  target=staging/name;shutil.copyfile(source/name,target)
  assert hashlib.sha256(target.read_bytes()).hexdigest()==digest,'checksum mismatch: '+name
 (staging/'release.json').write_bytes(raw)
 if destination.exists():
  for name,digest in manifest['files'].items():
   assert hashlib.sha256((destination/name).read_bytes()).hexdigest()==digest,name
  shutil.rmtree(staging)
 else:os.rename(staging,destination)
 print(destination)
except BaseException:
 shutil.rmtree(staging,ignore_errors=True);raise
