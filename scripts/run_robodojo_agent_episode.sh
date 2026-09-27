#!/usr/bin/env bash
set -euo pipefail

EMERGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK=""
ENV_CFG="arx_x5"
LAYOUT_ID="0"
DEVICE_ID="0"
POLICY_SEED="0"
POLICY_SERVER_URL="ws://127.0.0.1:8000"
MOTION_CONFIG="${EMERGE_ROOT}/configs/robodojo_motion.json"
ARCHIVE_OBSERVATIONS=""
POLICY_BASELINE=""
RECORD_EVERY_STEP=""
WORKSPACE=""
AGENT_CONFIG=""
AGENT_MESSAGE=""
SESSION_ID=""
AGENT_PYTHON="${EMERGE_ROOT}/.venv/bin/python"

export AGENTICVLA_LLM_TIMEOUT_S="${AGENTICVLA_LLM_TIMEOUT_S:-60}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2 ;;
    --env-cfg) ENV_CFG="$2"; shift 2 ;;
    --layout-id) LAYOUT_ID="$2"; shift 2 ;;
    --device-id) DEVICE_ID="$2"; shift 2 ;;
    --policy-seed) POLICY_SEED="$2"; shift 2 ;;
    --policy-server-url) POLICY_SERVER_URL="$2"; shift 2 ;;
    --motion-config) MOTION_CONFIG="$2"; shift 2 ;;
    --archive-observations) ARCHIVE_OBSERVATIONS=1; shift ;;
    --policy-baseline) POLICY_BASELINE=1; shift ;;
    --record-every-step) RECORD_EVERY_STEP=1; shift ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --agent-config) AGENT_CONFIG="$2"; shift 2 ;;
    --agent-message) AGENT_MESSAGE="$2"; shift 2 ;;
    --session-id) SESSION_ID="$2"; shift 2 ;;
    --agent-python) AGENT_PYTHON="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "${TASK}" ]]; then
  echo "usage: $0 --task TASK [--layout-id N] [--device-id N] [--agent-config PATH]" >&2
  exit 2
fi
if [[ -z "${WORKSPACE}" ]]; then
  WORKSPACE="${EMERGE_ROOT}/artifacts/robodojo_agent/${TASK}-layout-${LAYOUT_ID}/workspace"
fi
if [[ -z "${AGENT_MESSAGE}" ]]; then
  AGENT_MESSAGE="Solve the RoboDojo task described in ROBOT_STATE.md. Use the available embodied skills, re-read ROBOT_STATE.md after every action, and stop as soon as robots.robodojo.done is true."
fi
if [[ -z "${SESSION_ID}" ]]; then
  SESSION_ID="robodojo:${TASK}:layout-${LAYOUT_ID}:seed-${POLICY_SEED}"
fi
if [[ ! -x "${AGENT_PYTHON}" ]]; then
  echo "[robodojo-agent] agent Python executable not found: ${AGENT_PYTHON}" >&2
  echo "Run 'uv sync --extra dev' in ${EMERGE_ROOT}, or pass --agent-python." >&2
  exit 2
fi

# Fail before spending several minutes starting Isaac when an explicitly named
# provider credential is still an unresolved `${ENV_VAR}` reference.  Without
# this guard the agent exits on its first LLM call, the worker converts that
# infrastructure error into a two-frame official failure, and an unevaluated
# layout looks like a policy score of zero.
if [[ -n "${AGENT_CONFIG}" ]]; then
  if [[ ! -f "${AGENT_CONFIG}" ]]; then
    echo "[robodojo-agent] agent config not found: ${AGENT_CONFIG}" >&2
    exit 2
  fi
  missing_credentials="$({ python3 - "${AGENT_CONFIG}" <<'PY'
import json
import os
import re
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    providers = json.load(stream).get("providers", {})
missing = []
for provider, settings in providers.items():
    value = str(settings.get("apiKey", "")).strip()
    match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", value)
    if match and not os.environ.get(match.group(1), "").strip():
        missing.append(f"{provider}:{match.group(1)}")
print(",".join(missing))
PY
  } 2>/dev/null)"
  if [[ -n "${missing_credentials}" ]]; then
    echo "[robodojo-agent] missing provider credential(s): ${missing_credentials}" >&2
    echo "Export them before starting Isaac; no episode was launched." >&2
    exit 2
  fi
fi

mkdir -p "${WORKSPACE}"
DONE_FILE="${WORKSPACE}/.agent_done"
WORKER_LOG="${WORKSPACE}/robodojo_worker.log"

worker_args=(
  --task "${TASK}"
  --env-cfg "${ENV_CFG}"
  --layout-id "${LAYOUT_ID}"
  --device-id "${DEVICE_ID}"
  --policy-seed "${POLICY_SEED}"
  --policy-server-url "${POLICY_SERVER_URL}"
  --motion-config "${MOTION_CONFIG}"
  --workspace "${WORKSPACE}"
  --agent-done-file "${DONE_FILE}"
)
if [[ -n "${ARCHIVE_OBSERVATIONS}" ]]; then
  worker_args+=(--archive-observations)
fi
if [[ -n "${POLICY_BASELINE}" ]]; then
  worker_args+=(--policy-baseline)
fi
if [[ -n "${RECORD_EVERY_STEP}" ]]; then
  worker_args+=(--record-every-step)
fi

bash "${EMERGE_ROOT}/scripts/run_robodojo_agent_worker.sh" \
  "${worker_args[@]}" \
  >"${WORKER_LOG}" 2>&1 &
WORKER_PID=$!

# The worker is a shell wrapper around the Isaac Sim process, so signalling the
# pid we started leaves the simulator holding a CUDA context on every GPU of the
# node -- which the next episode scheduled there then fails to allocate.
kill_tree() {
  local pid="$1" signal="${2:-TERM}" child
  for child in $(pgrep -P "${pid}" 2>/dev/null); do
    kill_tree "${child}" "${signal}"
  done
  kill "-${signal}" "${pid}" 2>/dev/null || true
}

cleanup() {
  if [[ -n "${AGENT_PID:-}" ]] && kill -0 "${AGENT_PID}" 2>/dev/null; then
    kill "${AGENT_PID}" 2>/dev/null || true
  fi
  if kill -0 "${WORKER_PID}" 2>/dev/null; then
    kill_tree "${WORKER_PID}"
  fi
}
trap cleanup EXIT INT TERM

echo "[robodojo-agent] worker pid=${WORKER_PID} log=${WORKER_LOG}"

# The baseline arm has no agent: the worker drives the whole episode itself, so
# there is no ROBOT_STATE.md handshake to wait for and nothing to supervise.
if [[ -n "${POLICY_BASELINE}" ]]; then
  set +e
  wait "${WORKER_PID}"
  worker_rc=$?
  set -e
  trap - EXIT INT TERM
  exit "${worker_rc}"
fi

ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "${WORKER_PID}" 2>/dev/null; then
    echo "[robodojo-agent] worker exited before becoming ready" >&2
    tail -n 100 "${WORKER_LOG}" >&2 || true
    wait "${WORKER_PID}"
    exit $?
  fi
  if [[ -f "${WORKSPACE}/ROBOT_STATE.md" ]] &&
    grep -q '"connected": true' "${WORKSPACE}/ROBOT_STATE.md"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "[robodojo-agent] timed out waiting for the simulator" >&2
  exit 1
fi

agent_args=(-m Emerge.cli.headless --workspace "${WORKSPACE}" --session "${SESSION_ID}" --restrict-to-workspace --output-dir "${WORKSPACE}/agent_run" "${AGENT_MESSAGE}")
if [[ -n "${AGENT_CONFIG}" ]]; then
  agent_args+=(--config "${AGENT_CONFIG}")
fi

set +e
EMERGE_POLICY_BACKEND=vla "${AGENT_PYTHON}" "${agent_args[@]}" >"${WORKSPACE}/agent.log" 2>&1 &
AGENT_PID=$!
while kill -0 "${AGENT_PID}" 2>/dev/null && kill -0 "${WORKER_PID}" 2>/dev/null; do
  sleep 1
done

if ! kill -0 "${WORKER_PID}" 2>/dev/null && kill -0 "${AGENT_PID}" 2>/dev/null; then
  wait "${WORKER_PID}"
  worker_rc=$?
  kill "${AGENT_PID}" 2>/dev/null || true
  wait "${AGENT_PID}" 2>/dev/null
  echo "[robodojo-agent] worker exited while agent was running (rc=${worker_rc})" >&2
  trap - EXIT INT TERM
  exit "${worker_rc}"
fi

wait "${AGENT_PID}"
agent_rc=$?
touch "${DONE_FILE}"

# By this point the worker has already written its verdict to
# episode_status.json, and all that is left is tearing down the simulator. Isaac
# Sim can wedge there indefinitely -- deleting a prim invalidates the physics
# tensor view and the shutdown never completes -- and an unbounded wait here
# turned a nine-minute episode into a one-hour timeout that was then filed as an
# infrastructure error and re-run. Give teardown a bounded grace period, then
# take the verdict that is already on disk.
worker_rc=0
for _ in $(seq 1 "${WORKER_SHUTDOWN_GRACE_S:-180}"); do
  kill -0 "${WORKER_PID}" 2>/dev/null || break
  sleep 1
done
if kill -0 "${WORKER_PID}" 2>/dev/null; then
  echo "[robodojo-agent] simulator did not shut down within ${WORKER_SHUTDOWN_GRACE_S:-180}s; terminating" >&2
  kill_tree "${WORKER_PID}" TERM
  sleep 5
  kill_tree "${WORKER_PID}" KILL
  wait "${WORKER_PID}" 2>/dev/null || true
else
  wait "${WORKER_PID}"
  worker_rc=$?
fi
set -e
trap - EXIT INT TERM

if [[ "${agent_rc}" -ne 0 ]]; then
  echo "[robodojo-agent] agent exited with rc=${agent_rc}" >&2
  exit "${agent_rc}"
fi
exit "${worker_rc}"
