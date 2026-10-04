#!/usr/bin/env bash
set -euo pipefail
umask 077
export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
: "${EMERGE_RELEASE:?}" "${EMERGE_CONFIG:?}" "${EMERGE_STAGE_SHA256:?}" "${EMERGE_MANIFEST_SHA256:?}"
if [[ -n "${EMERGE_ASSET_PROXY:-}" ]]; then
  export http_proxy="$EMERGE_ASSET_PROXY" https_proxy="$EMERGE_ASSET_PROXY"
  export HTTP_PROXY="$EMERGE_ASSET_PROXY" HTTPS_PROXY="$EMERGE_ASSET_PROXY"
  export no_proxy="localhost,127.0.0.1,::1,.byted.org${no_proxy:+,$no_proxy}" NO_PROXY="localhost,127.0.0.1,::1,.byted.org${NO_PROXY:+,$NO_PROXY}"
fi
config_name=$(basename "$EMERGE_CONFIG")
cp "$EMERGE_RELEASE/stage-release.py" /tmp/emerge-stage.py
printf '%s  %s\n' "$EMERGE_STAGE_SHA256" /tmp/emerge-stage.py | sha256sum -c -
EMERGE_RELEASE=$(python3 /tmp/emerge-stage.py "$EMERGE_RELEASE" "$EMERGE_MANIFEST_SHA256")
export EMERGE_RELEASE EMERGE_CONFIG="$EMERGE_RELEASE/$config_name"
export ROBODOJO_SHARD_INDEX="${ARNOLD_ID:?}"
export EMERGE_SERVICES_MANIFEST
EMERGE_SERVICES_MANIFEST=$(python3 -c 'import json,os;print(json.load(open(os.environ["EMERGE_CONFIG"]))["services_manifest"])')
EMERGE_PHASE=source_checkout
on_exit() {
  code=$?
  if (( code != 0 )); then
    python3 "$EMERGE_RELEASE/record-failure.py" --config "$EMERGE_CONFIG" --phase "$EMERGE_PHASE" --exit-code "$code" || true
  fi
}
trap on_exit EXIT
python3 "$EMERGE_RELEASE/git_checkout.py" "$EMERGE_RELEASE/source-lock.json"
EMERGE_PHASE=bootstrap
bash "$EMERGE_RELEASE/bootstrap-l20.sh"
EMERGE_PHASE=services
export EMERGE_LOCAL_SERVICES_MANIFEST=/home/tiger/robodojo-setup/services.json
python3 "$EMERGE_RELEASE/wait-services.py"
export EMERGE_SERVICES_MANIFEST="$EMERGE_LOCAL_SERVICES_MANIFEST"
EMERGE_PHASE=credentials
python3 "$EMERGE_RELEASE/configure_agent.py"
unset EMERGE_API_KEY AZURE_OPENAI_API_KEY
EMERGE_PHASE=preflight
bash "$EMERGE_RELEASE/preflight.sh"
EMERGE_PHASE=evaluation
source /home/tiger/robodojo-setup/activate.sh
cd /home/tiger/EMERGE-Policy
.venv/bin/python "$EMERGE_RELEASE/run-node.py" --config "$EMERGE_CONFIG"
