#!/usr/bin/env bash
set -euo pipefail

EMERGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBODOJO_ROOT="${ROBODOJO_ROOT:-$(cd "${EMERGE_ROOT}/.." && pwd)/RoboDojo}"
DEVICE_ID="0"

args=("$@")
for ((index = 0; index < ${#args[@]}; index++)); do
  case "${args[$index]}" in
    --robodojo-root)
      ROBODOJO_ROOT="${args[$((index + 1))]}"
      ;;
    --device-id)
      DEVICE_ID="${args[$((index + 1))]}"
      ;;
  esac
done

if [[ ! -d "${ROBODOJO_ROOT}" ]]; then
  echo "[robodojo-agent] RoboDojo root not found: ${ROBODOJO_ROOT}" >&2
  exit 2
fi

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-Y}"
export OMNI_KIT_ALLOW_ROOT="${OMNI_KIT_ALLOW_ROOT:-1}"
export PYTHONPATH="${EMERGE_ROOT}:${EMERGE_ROOT}/third_party/openpi/packages/openpi-client/src:${ROBODOJO_ROOT}:${ROBODOJO_ROOT}/XPolicyLab:${PYTHONPATH:-}"
export NO_PROXY="localhost,127.0.0.1,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,${no_proxy}}"

PYTHON=(uv run --project "${ROBODOJO_ROOT}" --no-sync python)
CXX_RUNTIME_OVERLAY=""
if [[ -f "${ROBODOJO_ROOT}/scripts/internal/prepare_libstdcxx_compat.sh" ]]; then
  CXX_RUNTIME_OVERLAY="$(bash "${ROBODOJO_ROOT}/scripts/internal/prepare_libstdcxx_compat.sh")"
fi
if [[ -n "${CXX_RUNTIME_OVERLAY}" ]]; then
  export LD_LIBRARY_PATH="${CXX_RUNTIME_OVERLAY}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
# shellcheck source=/dev/null
if [[ -f "${ROBODOJO_ROOT}/scripts/internal/configure_nvidia_graphics_compat.sh" ]]; then
  source "${ROBODOJO_ROOT}/scripts/internal/configure_nvidia_graphics_compat.sh"
fi
DRIVER_OVERLAY="$(bash "${ROBODOJO_ROOT}/scripts/internal/prepare_nvidia_driver_compat.sh")"
if [[ -n "${DRIVER_OVERLAY}" ]]; then
  export LD_LIBRARY_PATH="${DRIVER_OVERLAY}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ "${ROBODOJO_UNSET_CUDA_VISIBLE_DEVICES:-0}" == "1" ]]; then
  unset CUDA_VISIBLE_DEVICES
else
  export CUDA_VISIBLE_DEVICES="${DEVICE_ID}"
fi

ISAAC_EXPERIENCE="$(
  "${PYTHON[@]}" "${ROBODOJO_ROOT}/scripts/internal/prepare_isaac_experience.py"
)"
KIT_ARGS="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
KIT_ARGS+=" --/renderer/activeGpu=${DEVICE_ID}"
KIT_ARGS+=" --/renderer/multiGpu/enabled=false"
KIT_ARGS+=" --/renderer/multiGpu/autoEnable=false"
KIT_ARGS+=" --/renderer/multiGpu/maxGpuCount=1"
if [[ "${ROBODOJO_SKIP_DRIVER_CHECK:-auto}" == "1" ]] || {
  [[ "${ROBODOJO_SKIP_DRIVER_CHECK:-auto}" == "auto" ]] &&
    bash "${ROBODOJO_ROOT}/scripts/internal/prepare_nvidia_driver_compat.sh" --needs-version-bypass
}; then
  KIT_ARGS+=" --/rtx/verifyDriverVersion/enabled=false"
fi
if [[ -n "${ROBODOJO_KIT_ARGS:-}" ]]; then
  KIT_ARGS+=" ${ROBODOJO_KIT_ARGS}"
fi

cd "${ROBODOJO_ROOT}"
exec "${PYTHON[@]}" -u "${EMERGE_ROOT}/scripts/run_robodojo_agent_worker.py" \
  --robodojo-root "${ROBODOJO_ROOT}" \
  --enable_cameras \
  --experience "${ISAAC_EXPERIENCE}" \
  --kit_args "${KIT_ARGS}" \
  --headless \
  "$@"
