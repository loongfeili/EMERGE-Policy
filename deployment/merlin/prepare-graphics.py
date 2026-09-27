"""Use graphics userspace matching this node's kernel driver, without installing a driver."""
from pathlib import Path
import json,os,re,shutil,subprocess,tarfile,time,hashlib
version=subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).splitlines()[0].strip()
assert re.fullmatch(r'\d+\.\d+\.\d+',version),version
import fcntl
lock=Path('/tmp/emerge-graphics.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
base=Path('/home/tiger/.local/opt');overlay=base/('nvidia-graphics-'+version)
if version=='535.261.03':
 source=base/('nvidia-'+version)
else:
 archive=Path('/mnt/hdfs/emerge_cache/runtime')/('nvidia-driver-'+version+'.tar.xz')
 for attempt in range(180):
  if archive.exists() and archive.with_suffix('.json').exists():break
  time.sleep(5)
 else:raise RuntimeError('Missing matching driver archive: '+str(archive))
 expected=json.loads(archive.with_suffix('.json').read_text())['sha256'];h=hashlib.sha256()
 with archive.open('rb') as f:
  while b:=f.read(16*1024*1024):h.update(b)
 assert h.hexdigest()==expected,'driver archive checksum mismatch'
 extracted=base/('nvidia-redist-'+version);extracted.mkdir(parents=True,exist_ok=True)
 if not (extracted/'EXTRACTED').exists():
  with tarfile.open(archive) as t:t.extractall(extracted)
  (extracted/'EXTRACTED').touch()
 candidates=[p for p in extracted.rglob('libGLX_nvidia.so.'+version) if 'lib32' not in p.parts]
 assert len(candidates)==1,candidates
 source=candidates[0].parent

overlay.mkdir(parents=True,exist_ok=True)
for p in source.glob('*.so*'):
 if not p.name.startswith(('libGL','libEGL','libnvidia-egl','libnvidia-gl','libnvidia-tls','libnvidia-rtcore','libnvidia-allocator')):continue
 target=overlay/p.name
 if not target.exists():target.symlink_to(p.resolve())
for p in list(overlay.iterdir()):
 out=subprocess.run(['readelf','-d',str(p)],capture_output=True,text=True).stdout
 soname=re.search(r'\(SONAME\).*\[(.*?)\]',out)
 if soname and not (overlay/soname[1]).exists():(overlay/soname[1]).symlink_to(p.name)
setup=Path('/home/tiger/robodojo-setup');setup.mkdir(parents=True,exist_ok=True)
icd=setup/'nvidia-icd.json';egl=setup/'nvidia-egl.json'
icd.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(overlay/('libGLX_nvidia.so.'+version)),'api_version':'1.3.242'}}))
egl.write_text(json.dumps({'file_format_version':'1.0.0','ICD':{'library_path':str(overlay/('libEGL_nvidia.so.'+version))}}))
assert (overlay/('libGLX_nvidia.so.'+version)).exists()
(setup/'graphics-env.sh').write_text('export LD_LIBRARY_PATH="'+str(overlay)+':${LD_LIBRARY_PATH:-}"\nexport VK_ICD_FILENAMES="'+str(icd)+'"\nexport VK_DRIVER_FILES="$VK_ICD_FILENAMES"\nexport __EGL_VENDOR_LIBRARY_FILENAMES="'+str(egl)+'"\n')
activate=setup/'activate.sh';line='\nsource /home/tiger/robodojo-setup/graphics-env.sh\n'
if line not in activate.read_text():activate.write_text(activate.read_text()+line)
print(json.dumps({'driver':version,'graphics_overlay':str(overlay)}))
