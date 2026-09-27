#!/bin/bash
set -euo pipefail
cache=/mnt/hdfs/emerge_cache
release=${EMERGE_RELEASE:?}
python3 "$release/verify_cache.py" "$release/cache-lock.json" infer
setup=/home/tiger/emerge-setup
mkdir -p "$setup" /home/tiger/.local/bin /home/tiger/.local/share/uv/python /home/tiger/EMERGE-Policy /home/tiger/emerge-models
exec > >(tee -a "$setup/bootstrap.log") 2>&1
trap 'echo $? > /home/tiger/emerge-setup/bootstrap.exit' EXIT
export PATH=/home/tiger/.local/bin:$PATH
export GIT_LFS_SKIP_SMUDGE=1 UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=8 CC=gcc CXX=g++ GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
cp "$cache/runtime/zstd" /home/tiger/.local/bin/zstd
chmod 755 /home/tiger/.local/bin/zstd
cp "$cache/runtime/uv" /home/tiger/.local/bin/uv
chmod 755 /home/tiger/.local/bin/uv
mkdir -p /home/tiger/.local/opt
tar -xf "$cache/runtime/glibc235.tar.gz" -C /home/tiger/.local/opt
tar -xf "$cache/runtime/python311.tar.gz" -C /home/tiger/.local/share/uv/python
for sub in openpi sam3 vggt; do
 mkdir -p "/home/tiger/EMERGE-Policy/third_party/$sub"
 tar -xf "$cache/runtime/$sub-source.tar" -C "/home/tiger/EMERGE-Policy/third_party/$sub"
done
mkdir -p /home/tiger/.cache/uv/git-v0/db/b2400a7a62d6a7cf
tar -xf "$cache/runtime/lerobot-git-cache.tar.gz" -C /home/tiger/.cache/uv/git-v0/db/b2400a7a62d6a7cf
python3 - <<'PY'
from pathlib import Path
import subprocess
cache=Path('/home/tiger/.cache/uv/git-v0');repo=cache/'db/b2400a7a62d6a7cf';checkout=cache/'checkouts/b2400a7a62d6a7cf/0cf8648';commit='0cf864870cf29f4738d3ade893e6fd13fbd7cdb5'
subprocess.run(['git','--git-dir',str(repo),'update-ref','refs/heads/cache',commit],check=True)
subprocess.run(['git','--git-dir',str(repo),'symbolic-ref','HEAD','refs/heads/cache'],check=True)
if not (checkout/'.git').exists():
 checkout.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['git','clone',str(repo),str(checkout)],check=True)
subprocess.run(['git','-C',str(checkout),'fetch',str(repo),commit],check=True)
subprocess.run(['git','-C',str(checkout),'reset','--hard',commit],check=True)
PY
mkdir -p /home/tiger/.cache/openpi/big_vision
cp "$cache/runtime/paligemma_tokenizer.model" /home/tiger/.cache/openpi/big_vision/
python3 "$release/copy-models.py" &
models_pid=$!
cd /home/tiger/EMERGE-Policy
python_exe=/home/tiger/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11
build_runtime=$(python3 -c 'import json,os;print(int(json.load(open(os.environ["EMERGE_CONFIG"])).get("build_inference_runtime",False)))')
if [[ "$build_runtime" == 1 ]]; then
uv venv --python "$python_exe" .venv
uv pip install --python .venv/bin/python -e . -e third_party/sam3 -e third_party/vggt 'opencv-python<4.12' einops scipy hydra-core timm ftfy iopath 'huggingface-hub<1' decord
uv sync --project third_party/openpi --python "$python_exe" --frozen --no-dev
uv pip install --python third_party/openpi/.venv/bin/python 'websockets>=16,<17' 'msgpack>=1.1,<2'
 python3 "$release/inference_runtime.py" seal
else
 python3 "$release/inference_runtime.py" restore
fi
wait "$models_pid"
echo READY_FOR_SERVICES
