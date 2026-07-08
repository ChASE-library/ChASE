#!/bin/bash
# Run serial HIP unit tests on Frontier (inside a GPU allocation).
#
# Usage:
#   source scripts/frontier_hip_env.sh
#   cmake --build build-frontier-hip
#   srun -n 1 --gpus-per-task=1 bash scripts/run_hip_serial_unittests.sh
#
# Options (env vars):
#   CHASE_BUILD_DIR          build tree (default: build-frontier-hip)
#   CHASE_GPUS_TASK          GPUs per task when launching with srun (default: 1)
#   CHASE_LAUNCHER           override launcher, e.g. "srun -n 1 --gpus-per-task=1"
#   CHASE_RUN_INTEGRATION=1  also run ChaseSerialSolve* integration tests
#   CHASE_RUN_INTERFACE=1    also run C/Fortran interface smoke tests (when built)
#   CHASE_INTEGRATION_ONLY=1 run only integration/interface stages (skip kernel sweep)
#
# Exit code: 0 if all selected tests pass, 1 otherwise.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${CHASE_BUILD_DIR:-${ROOT}/build-frontier-hip}"
GPUS_PER_TASK="${CHASE_GPUS_TASK:-1}"
GPU_SERIAL_TESTS="${BUILD}/tests/linalg/internal/gpu/serial/hip"
GPU_DEVICE_TESTS="${BUILD}/tests/linalg/internal/gpu/kernels/hip"
TOP_TESTS="${BUILD}/tests"
EXAMPLES="${BUILD}/examples/4_interface"
export CHASE_LINALG_INTERNAL_TEST_DATA_ROOT="${BUILD}/tests/linalg/internal"

if [[ -f "${ROOT}/scripts/frontier_hip_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/frontier_hip_env.sh"
fi

if [[ -z "${MPICH_GPU_SUPPORT_ENABLED:-}" ]]; then
  echo "WARNING: MPICH_GPU_SUPPORT_ENABLED is unset. Run: source scripts/frontier_hip_env.sh"
fi

run_one() {
  local name="$1"
  local exe="$2"
  local wd="${3:-}"
  shift 3 || true
  echo ""
  echo "========== ${name} =========="
  if [[ ! -x "${exe}" ]]; then
    echo "SKIP (not built): ${exe}"
    FAIL=$((FAIL + 1))
    FAILED+=("${name} (missing)")
    return
  fi
  if [[ -n "${wd}" ]]; then
    (cd "${wd}" && ${LAUNCHER} "${exe}" "$@")
  else
    ${LAUNCHER} "${exe}" "$@"
  fi
  echo "PASSED: ${name}"
  PASS=$((PASS + 1))
}

if [[ -n "${SLURM_NTASKS:-}" || -n "${PMI_RANK:-}" || -n "${OMPI_COMM_WORLD_SIZE:-}" ]]; then
  LAUNCHER="${CHASE_LAUNCHER:-}"
else
  LAUNCHER="${CHASE_LAUNCHER:-srun -n 1 --gpus-per-task=${GPUS_PER_TASK}}"
fi

PASS=0
FAIL=0
FAILED=()

echo "ChASE HIP serial unit tests"
echo "  build: ${BUILD}"
echo "  launcher: ${LAUNCHER:-<current MPI rank set>}"

RUN_KERNELS=1
if [[ "${CHASE_INTEGRATION_ONLY:-0}" == "1" ]]; then
  RUN_KERNELS=0
fi

if [[ "${RUN_KERNELS}" -eq 1 ]]; then
# gpu/serial/hip — shared impl paths (CholQR, RR, residuals, Lanczos)
GPU_SERIAL_PRIORITY=(
  hipKernelsSmokeHIPTest
  cholqrHIPTest
  rayleighRitzHIPTest
  residualsMatrixHIPTest
  lanczosHIPTest
  PseudoHermitianLanczosHIPTest
  PseudoHermitianRayleighRitzHIPTest
)

for t in "${GPU_SERIAL_PRIORITY[@]}"; do
  run_one "${t}" "${GPU_SERIAL_TESTS}/${t}"
done

# gpu/kernels/hip — device kernels used by serial + distributed stacks
GPU_DEVICE_PRIORITY=(
  shiftDiagonalHIPTest
  flipLowerHalfSignHIPTest
  lacpyHIPTest
  absTraceHIPTest
  conjugateHIPTest
  forceHermitianHIPTest
  precisionConversionHIPTest
  randomNormalHIPTest
  residualsHIPTest
  pseudoHermitianLanczosDiagHIPTest
  householderScalarKernelHIPTest
  lanczosKernelsHIPTest
)

for t in "${GPU_DEVICE_PRIORITY[@]}"; do
  run_one "${t}" "${GPU_DEVICE_TESTS}/${t}"
done
fi

if [[ "${CHASE_RUN_INTEGRATION:-0}" == "1" || "${CHASE_INTEGRATION_ONLY:-0}" == "1" ]]; then
  echo ""
  echo "========== serial solver integration =========="
  run_one "ChaseSerialSolveTest (HIP)" \
    "${TOP_TESTS}/ChaseSerialSolveTest" "" \
    --gtest_filter='*ChaseHIP*'
  run_one "ChaseSerialSolvePseudoBSETest (GPU)" \
    "${TOP_TESTS}/ChaseSerialSolvePseudoBSETest" "" \
    --gtest_filter='*CPU_and_GPU/1/*'
fi

if [[ "${CHASE_RUN_INTERFACE:-0}" == "1" || "${CHASE_INTEGRATION_ONLY:-0}" == "1" ]]; then
  echo ""
  echo "========== C/Fortran interface smoke =========="
  run_one "ChaseFortranSerialSolveTest" \
    "${TOP_TESTS}/ChaseFortranSerialSolveTest"
  run_one "4_c_serial_chase.exe" \
    "${EXAMPLES}/4_c_serial_chase.exe"
  run_one "4_f_serial_chase.exe" \
    "${EXAMPLES}/4_f_serial_chase.exe"
fi

echo ""
echo "Summary: ${PASS} passed, ${FAIL} failed/skipped"
if [[ ${FAIL} -gt 0 ]]; then
  echo "Failed:"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
