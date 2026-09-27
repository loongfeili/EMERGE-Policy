#!/bin/bash
set -euo pipefail
setup=/home/tiger/robodojo-setup
cache=${EMERGE_RELEASE:?}
exec > >(tee -a "$setup/bulk25-preflight.log") 2>&1
rm -f "$setup/bulk25-preflight.exit"
trap 'echo $? > /home/tiger/robodojo-setup/bulk25-preflight.exit' EXIT
python3 "$cache/prepare-graphics.py"
source "$setup/activate.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd /home/tiger/EMERGE-Policy
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python "$cache/git_checkout.py" "$cache/source-lock.json" --verify
uv pip freeze --python .venv/bin/python > "$setup/agent-packages.txt"
uv pip freeze --python /home/tiger/RoboDojo/.venv/bin/python > "$setup/simulator-packages.txt"
.venv/bin/python "$cache/probe-api.py" > "$setup/api-probe.json"
.venv/bin/python -c 'import json; d=json.load(open("/home/tiger/robodojo-setup/api-probe.json")); assert d["passed"] == 1, d'
POLICY_URL=$(python3 -c "import json;print(json.load(open(__import__('os').environ['EMERGE_SERVICES_MANIFEST']))['policy_urls'][0])")
timeout --signal=TERM --kill-after=15s 900 bash scripts/run_robodojo_agent_worker.sh --robodojo-root /home/tiger/RoboDojo --task stack_bowls --layout-id 0 --device-id 0 --policy-seed 0 --workspace /home/tiger/emerge-smoke-bulk25 --policy-server-url "$POLICY_URL" --motion-config configs/robodojo_motion.json --smoke-action-json '{"action_type":"vla_execute","parameters":{"instruction":"Stack the bowls.","step":10}}'
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('/home/tiger/emerge-smoke-bulk25/smoke_result.json');d=json.loads(p.read_text());assert d['actions'],d
for a in d['actions']:assert not str(a['result']).startswith(('Failed:','Interrupted:')),a
PY
.venv/bin/python "$cache/probe-policy.py"
mkdir -p "$setup/perception-reference"
tar -xzf "$cache/perception-reference.tar.gz" -C "$setup/perception-reference"
.venv/bin/python "$cache/probe-perception.py"
echo READY_FOR_FULL_EVALUATION
