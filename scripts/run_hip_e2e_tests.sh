#!/bin/bash
# Run HIP end-to-end regression on Frontier (serial + distributed + interface).
#
# Usage (inside a GPU allocation):
#   source scripts/frontier_hip_env.sh
#   cmake --build build-frontier-hip
#   srun -n 4 --gpus-per-task=1 bash scripts/run_hip_e2e_tests.sh
#
# Options (env vars):
#   CHASE_BUILD_DIR                 build tree (default: build-frontier-hip)
#   CHASE_NPROCS                    MPI ranks for distributed tests (default: 4)
#   CHASE_GPUS_TASK                 GPUs per task (default: 1)
#   CHASE_SKIP_SERIAL_UNITTESTS=1   skip the long serial kernel sweep (default: run)
#   CHASE_SKIP_DIST_UNITTESTS=1     skip distributed kernel unit tests (default: run)
#   CHASE_INTEGRATION_COLLECTIVES   mpi|nccl|both for distributed solves (default: both)
#
# This script always runs:
#   - serial solver integration (ChaseSerialSolve*)
#   - distributed solver integration (ChaseDistributedSolve* GPU filters)
#   - C/Fortran interface smoke tests when the targets are built
#
# Exit code: 0 if all selected stages pass, 1 otherwise.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${CHASE_BUILD_DIR:-${ROOT}/build-frontier-hip}"
NPROCS="${CHASE_NPROCS:-4}"
GPUS_PER_TASK="${CHASE_GPUS_TASK:-1}"

if [[ -f "${ROOT}/scripts/frontier_hip_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/frontier_hip_env.sh"
fi

echo "ChASE HIP end-to-end regression"
echo "  build: ${BUILD}"
echo "  distributed ranks: ${NPROCS}"

STAGE_FAIL=0

run_stage() {
  local label="$1"
  shift
  echo ""
  echo "################################################################"
  echo "# ${label}"
  echo "################################################################"
  if "$@"; then
    echo "STAGE PASSED: ${label}"
  else
    echo "STAGE FAILED: ${label}"
    STAGE_FAIL=1
  fi
}

if [[ "${CHASE_SKIP_SERIAL_UNITTESTS:-0}" != "1" ]]; then
  run_stage "serial HIP kernel unit tests" \
    env CHASE_BUILD_DIR="${BUILD}" CHASE_GPUS_TASK="${GPUS_PER_TASK}" \
      bash "${ROOT}/scripts/run_hip_serial_unittests.sh"
fi

run_stage "serial solver + interface smoke" \
  env CHASE_BUILD_DIR="${BUILD}" CHASE_GPUS_TASK="${GPUS_PER_TASK}" \
    CHASE_INTEGRATION_ONLY=1 CHASE_RUN_INTEGRATION=1 CHASE_RUN_INTERFACE=1 \
    bash "${ROOT}/scripts/run_hip_serial_unittests.sh"

if [[ "${CHASE_SKIP_DIST_UNITTESTS:-0}" != "1" ]]; then
  run_stage "distributed HIP kernel unit tests" \
    env CHASE_BUILD_DIR="${BUILD}" CHASE_NPROCS="${NPROCS}" \
      CHASE_GPUS_TASK="${GPUS_PER_TASK}" CHASE_RUN_INTEGRATION=0 \
      bash "${ROOT}/scripts/run_hip_dist_unittests.sh"
fi

run_stage "distributed solver integration" \
  env CHASE_BUILD_DIR="${BUILD}" CHASE_NPROCS="${NPROCS}" \
    CHASE_GPUS_TASK="${GPUS_PER_TASK}" CHASE_RUN_INTEGRATION=1 \
    CHASE_INTEGRATION_COLLECTIVES="${CHASE_INTEGRATION_COLLECTIVES:-both}" \
    bash "${ROOT}/scripts/run_hip_dist_unittests.sh"

run_stage "distributed C/Fortran interface smoke" \
  env CHASE_BUILD_DIR="${BUILD}" CHASE_NPROCS="${NPROCS}" \
    CHASE_GPUS_TASK="${GPUS_PER_TASK}" \
    bash "${ROOT}/scripts/run_hip_dist_interface_smoke.sh"

echo ""
if [[ ${STAGE_FAIL} -ne 0 ]]; then
  echo "E2E regression finished with failures."
  exit 1
fi
echo "E2E regression finished successfully."
