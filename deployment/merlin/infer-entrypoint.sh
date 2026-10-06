#!/usr/bin/env bash
set -euo pipefail
umask 077
export GIT_LFS_SKIP_SMUDGE=1 GIT_TERMINAL_PROMPT=0
: "${EMERGE_RELEASE:?}" "${EMERGE_CONFIG:?}"
if [[ -n "${EMERGE_ASSET_PROXY:-}" ]]; then
  export http_proxy="$EMERGE_ASSET_PROXY" https_proxy="$EMERGE_ASSET_PROXY"
  export HTTP_PROXY="$EMERGE_ASSET_PROXY" HTTPS_PROXY="$EMERGE_ASSET_PROXY"
  export no_proxy="localhost,127.0.0.1,::1,.byted.org" NO_PROXY="localhost,127.0.0.1,::1,.byted.org"
fi
config_name=$(basename "$EMERGE_CONFIG")
cp "$EMERGE_RELEASE/stage-release.py" /tmp/emerge-stage.py
printf '%s  %s\n' "$EMERGE_STAGE_SHA256" /tmp/emerge-stage.py | sha256sum -c -
EMERGE_RELEASE=$(python3 /tmp/emerge-stage.py "$EMERGE_RELEASE" "$EMERGE_MANIFEST_SHA256")
export EMERGE_RELEASE EMERGE_CONFIG="$EMERGE_RELEASE/$config_name"
python3 "$EMERGE_RELEASE/git_checkout.py" "$EMERGE_RELEASE/source-lock.json" --only emerge
bash "$EMERGE_RELEASE/bootstrap-infer.sh"
python3 "$EMERGE_RELEASE/git_checkout.py" "$EMERGE_RELEASE/source-lock.json" --verify --only emerge
cd /home/tiger/EMERGE-Policy
exec .venv/bin/python "$EMERGE_RELEASE/serve.py" --config "$EMERGE_CONFIG"
