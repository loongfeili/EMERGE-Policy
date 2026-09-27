#!/bin/bash
set -euo pipefail
setup=/home/tiger/robodojo-setup
cache=/mnt/hdfs/emerge_cache
release=${EMERGE_RELEASE:?}
python3 "$release/verify_cache.py" "$release/cache-lock.json" eval
mkdir -p "$setup" /home/tiger/.local/bin /home/tiger/RoboDojo /home/tiger/EMERGE-Policy /home/tiger/robodojo-data
exec > >(tee -a "$setup/bulk25-bootstrap.log") 2>&1
trap 'echo $? > /home/tiger/robodojo-setup/bulk25-bootstrap.exit' EXIT
export PATH=/home/tiger/.local/bin:$PATH
cp "$cache/runtime/zstd" /home/tiger/.local/bin/zstd
cp "$cache/runtime/uv" /home/tiger/.local/bin/uv
chmod 755 /home/tiger/.local/bin/zstd /home/tiger/.local/bin/uv
sed "s|/mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925|$cache|g" "$cache/runtime/robodojo-venv.sha256" | sha256sum -c -
tar -I zstd -xf "$cache/runtime/host-compat.tar.zst" -C /home/tiger
tar -I zstd -xf "$cache/runtime/isaac-extension-cache.tar.zst" -C /home/tiger
tar -xf "$cache/runtime/glibc235.tar.gz" -C /home/tiger/.local/opt
tar -xf "$cache/source/robodojo-submodules.tar.gz" -C /home/tiger/RoboDojo
tar -xf "$release/isaaclab-full.tar.gz" -C /home/tiger/RoboDojo
tar -I zstd -xf "$cache/runtime/robodojo-venv.tar.zst" -C /home/tiger/RoboDojo
for sub in openpi sam3 vggt; do
 mkdir -p "/home/tiger/EMERGE-Policy/third_party/$sub"
 tar -xf "$cache/runtime/$sub-source.tar" -C "/home/tiger/EMERGE-Policy/third_party/$sub"
done
if ! command -v patchelf >/dev/null; then sudo -n apt-get update -qq && sudo -n apt-get install -y patchelf; fi
bash "$cache/source/patch-python-runtime.sh"
cp "$cache/source/activate.sh" "$setup/activate.sh"
mkdir -p /home/tiger/.cache/tiktoken
token_cache_key=$(python3 -c 'import hashlib; print(hashlib.sha1(b"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken").hexdigest())')
cp "$release/cl100k_base.tiktoken" "/home/tiger/.cache/tiktoken/$token_cache_key"
echo 'export TIKTOKEN_CACHE_DIR=/home/tiger/.cache/tiktoken' >> "$setup/activate.sh"
source "$setup/activate.sh"
cp "$cache/runtime/uv" /home/tiger/.local/bin/uv
chmod 755 /home/tiger/.local/bin/uv
python3 - <<'PY'
from pathlib import Path
# Editable imports in the restored simulation environment must reference the deployed sources.
s=Path('/home/tiger/RoboDojo/.venv/lib/python3.11/site-packages')
for p in s.glob('__editable*finder.py'):
 text=p.read_text().replace('/home/tiger/AgenticVLA/third_party','/home/tiger/EMERGE-Policy/third_party')
 p.write_text(text)
PY
# The simulation process imports the shared action-queue package, whose helpers require tiktoken.
uv pip install --offline --no-index --find-links "$release" --python /home/tiger/RoboDojo/.venv/bin/python -r "$release/simulator-offline.txt"
cd /home/tiger/EMERGE-Policy
PYTHONPATH="$PWD" /home/tiger/RoboDojo/.venv/bin/python -c "import robot.controller, tiktoken; assert tiktoken.get_encoding('cl100k_base').encode('offline verification')"
agent_archive=$cache/runtime/agent-venv-cluster-20260926-r3.tar.zst
if test -f "${agent_archive%.zst}.json"; then
 python3 - "$agent_archive" <<'PYCACHE'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);expected=json.loads(p.with_suffix('.json').read_text());h=hashlib.sha256()
with p.open('rb') as f:
 while b:=f.read(16*1024*1024):h.update(b)
assert p.stat().st_size==expected['bytes'] and h.hexdigest()==expected['sha256'],'agent runtime checksum mismatch'
PYCACHE
 tar -I zstd -xf "$agent_archive" -C /home/tiger/EMERGE-Policy
else
 echo "Missing verified HDFS Agent runtime metadata: ${agent_archive%.zst}.json" >&2
 exit 1
fi
# Extract to local disk; rendering must not load thousands of small USD files through HDFS.
sed "s|/mnt/hdfs/__MERLIN_USER_DIR__/emerge_robodojo_20260925|$cache|g" "$cache/assets/assets.sha256" | sha256sum -c -
tar -I zstd -xf "$cache/assets/assets.tar.zst" -C /home/tiger/robodojo-data
ln -sfn /home/tiger/robodojo-data/Assets /home/tiger/RoboDojo/Assets
python3 - <<'PY'
from pathlib import Path
p=Path('/home/tiger/robodojo-data/Assets/Robots/x5');(p/'curobo.yml').write_text((p/'curobo_tmp.yml').read_text().replace('${ASSETS_PATH}','/home/tiger/RoboDojo'))
PY
# The system ffmpeg should use system libraries, not the simulator's glibc overlay.
if ! test -x /usr/bin/ffmpeg; then sudo -n apt-get update -qq && sudo -n apt-get install -y ffmpeg; fi
printf '#!/bin/sh\nunset LD_LIBRARY_PATH\nexec /usr/bin/ffmpeg "$@"\n' > /home/tiger/.local/bin/ffmpeg
chmod 755 /home/tiger/.local/bin/ffmpeg
cd /home/tiger/RoboDojo
bash scripts/robodojo.sh doctor --skip-policy --summary "$setup/bulk25-doctor.json"
echo READY_FOR_PROBES
