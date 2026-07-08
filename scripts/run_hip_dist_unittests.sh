#!/bin/bash
# Run distributed HIP unit tests on Frontier (inside a GPU allocation).
#
# Usage:
#   source scripts/frontier_hip_env.sh
#   cmake --build build-frontier-hip
#   srun -N 1 -n 4 --gpus-per-task=1 bash scripts/run_hip_dist_unittests.sh
#
# Options (env vars):
#   CHASE_BUILD_DIR   build tree (default: build-frontier-hip)
#   CHASE_NPROCS      MPI ranks (default: 4)
#   CHASE_GPUS_TASK   GPUs per task for srun when launched without srun (default: 1)
#   CHASE_RUN_INTEGRATION=1  also run ChaseDistributedSolve* GPU tests
#   CHASE_INTEGRATION_COLLECTIVES=mpi|nccl|both  which integration binaries (HIP; default both if *GPUNCCL* built)
#   CHASE_LAUNCHER    override launcher, e.g. "srun -n 4 --gpus-per-task=1"
#
# Exit code: 0 if all selected tests pass, 1 otherwise.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${CHASE_BUILD_DIR:-${ROOT}/build-frontier-hip}"
NPROCS="${CHASE_NPROCS:-4}"
GPUS_PER_TASK="${CHASE_GPUS_TASK:-1}"
HIP_DIST_TESTS="${BUILD}/tests/linalg/internal/gpu/dist_mpi/hip"
RCCL_DIST_TESTS="${BUILD}/tests/linalg/internal/gpu/dist_xccl/rccl"
TOP_TESTS="${BUILD}/tests"
export CHASE_LINALG_INTERNAL_TEST_DATA_ROOT="${BUILD}/tests/linalg/internal"

if [[ -f "${ROOT}/scripts/frontier_hip_env.sh" ]]; then
  # Safe if already sourced.
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
  echo ""
  echo "========== ${name} =========="
  if [[ ! -x "${exe}" ]]; then
    echo "SKIP (not built): ${exe}"
    FAIL=$((FAIL + 1))
    FAILED+=("${name} (missing)")
    return
  fi
  if [[ -n "${wd}" ]]; then
    (cd "${wd}" && ${LAUNCHER} "${exe}")
  else
    ${LAUNCHER} "${exe}"
  fi
  echo "PASSED: ${name}"
  PASS=$((PASS + 1))
}

# If already inside srun/mpirun, run tests directly; otherwise wrap with srun.
if [[ -n "${SLURM_NTASKS:-}" || -n "${PMI_RANK:-}" || -n "${OMPI_COMM_WORLD_SIZE:-}" ]]; then
  LAUNCHER="${CHASE_LAUNCHER:-}"
else
  LAUNCHER="${CHASE_LAUNCHER:-srun -n ${NPROCS} --gpus-per-task=${GPUS_PER_TASK}}"
fi

PASS=0
FAIL=0
FAILED=()

echo "ChASE HIP distributed unit tests"
echo "  build: ${BUILD}"
echo "  launcher: ${LAUNCHER:-<current MPI rank set>}"
echo "  MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-unset}"

# Optional configure probe
if [[ -x "${BUILD}/hip_mpi_probe" ]]; then
  run_one "hip_mpi_probe" "${BUILD}/hip_mpi_probe"
fi

# Kernels touched by gpu/dist_mpi sharing (run first)
PRIORITY=(
  HEMMHIPDistTest
  PseudoHEMMHIPDistTest
  shiftDiagonalHIPDistTest
  ResidualsHIPDistTest
  RayleighRitzHIPDistTest
  PseudoRayleighRitzHIPDistTest
)

for t in "${PRIORITY[@]}"; do
  run_one "${t}" "${HIP_DIST_TESTS}/${t}"
done

# Remaining *HIPDistTest binaries (alphabetical, skip already run)
shopt -s nullglob
for exe in "${HIP_DIST_TESTS}"/*HIPDistTest; do
  base="$(basename "${exe}")"
  skip=0
  for p in "${PRIORITY[@]}"; do
    [[ "${base}" == "${p}" ]] && skip=1 && break
  done
  [[ "${skip}" -eq 1 ]] && continue
  # CholQR reads QR matrices relative to this directory
  if [[ "${base}" == "CholQRHIPDistTest" ]]; then
    run_one "${base}" "${exe}" "${HIP_DIST_TESTS}"
  else
    run_one "${base}" "${exe}"
  fi
done

# Step-level CholQR tests (different target name)
if [[ -x "${HIP_DIST_TESTS}/CholQRHIPDistStepsTest" ]]; then
  run_one "CholQRHIPDistStepsTest" "${HIP_DIST_TESTS}/CholQRHIPDistStepsTest" \
    "${HIP_DIST_TESTS}"
fi

# RCCL collective path (Frontier GPU collectives)
if [[ -d "${RCCL_DIST_TESTS}" ]]; then
  echo ""
  echo "========== RCCL distributed tests =========="
  RCCL_PRIORITY=(
    RayleighRitzGPURCCLDistTest
    PseudoHermitianRayleighRitzGPURCCLDistTest
    CholQRGPURCCLDistTest
    LanczosGPURCCLDistTest
    PseudoHermitianLanczosGPURCCLDistTest
    HEMMGPURCCLDistTest
    PseudoHEMMGPURCCLDistTest
    shiftDiagonalGPURCCLDistTest
    flipSignGPURCCLDistTest
    ResidualsGPURCCLDistTest
    symOrHermGPURCCLDistTest
  )
  for t in "${RCCL_PRIORITY[@]}"; do
    if [[ "${t}" == "CholQRGPURCCLDistTest" ]]; then
      run_one "${t}" "${RCCL_DIST_TESTS}/${t}" "${RCCL_DIST_TESTS}"
    else
      run_one "${t}" "${RCCL_DIST_TESTS}/${t}"
    fi
  done
fi

if [[ "${CHASE_RUN_INTEGRATION:-0}" == "1" ]]; then
  echo ""
  echo "========== integration (GPU) =========="
  INTEGRATION_COLLECTIVES="${CHASE_INTEGRATION_COLLECTIVES:-both}"
  if [[ "${INTEGRATION_COLLECTIVES}" != "mpi" && "${INTEGRATION_COLLECTIVES}" != "nccl" && "${INTEGRATION_COLLECTIVES}" != "both" ]]; then
    echo "ERROR: CHASE_INTEGRATION_COLLECTIVES must be mpi, nccl, or both (got '${INTEGRATION_COLLECTIVES}')"
    exit 1
  fi

  run_integration_solve() {
    local label="$1"
    local exe="$2"
    if [[ ! -x "${exe}" ]]; then
      echo "SKIP (not built): ${exe}"
      FAIL=$((FAIL + 1))
      FAILED+=("${label} (missing)")
      return
    fi
    echo ""
    echo "========== ${label} GPU =========="
    (cd "${TOP_TESTS}" && ${LAUNCHER} "./$(basename "${exe}")" --gtest_filter='*GPU*')
    echo "PASSED: ${label} GPU"
    PASS=$((PASS + 1))
  }

  if [[ "${INTEGRATION_COLLECTIVES}" == "mpi" || "${INTEGRATION_COLLECTIVES}" == "both" ]]; then
    run_integration_solve "ChaseDistributedSolveTest (HIP-MPI)" \
      "${TOP_TESTS}/ChaseDistributedSolveTest"
    run_integration_solve "ChaseDistributedSolvePseudoBSETest (HIP-MPI)" \
      "${TOP_TESTS}/ChaseDistributedSolvePseudoBSETest"
  fi

  if [[ "${INTEGRATION_COLLECTIVES}" == "nccl" || "${INTEGRATION_COLLECTIVES}" == "both" ]]; then
    if [[ -x "${TOP_TESTS}/ChaseDistributedSolveGPUNCCLTest" ]]; then
      run_integration_solve "ChaseDistributedSolveGPUNCCLTest (RCCL)" \
        "${TOP_TESTS}/ChaseDistributedSolveGPUNCCLTest"
    elif [[ "${INTEGRATION_COLLECTIVES}" == "nccl" ]]; then
      echo "SKIP (not built): ChaseDistributedSolveGPUNCCLTest"
      echo "  Hint: default build uses RCCL in ChaseDistributedSolveTest when RCCL is linked."
      echo "  Or reconfigure with -DCHASE_HIP_DISTRIBUTED_COLLECTIVES=MPI and use *GPUNCCLTest for RCCL."
      FAIL=$((FAIL + 1))
      FAILED+=("ChaseDistributedSolveGPUNCCLTest (missing)")
    fi
    if [[ -x "${TOP_TESTS}/ChaseDistributedSolvePseudoBSEGPUNCCLTest" ]]; then
      run_integration_solve "ChaseDistributedSolvePseudoBSEGPUNCCLTest (RCCL)" \
        "${TOP_TESTS}/ChaseDistributedSolvePseudoBSEGPUNCCLTest"
    elif [[ "${INTEGRATION_COLLECTIVES}" == "nccl" ]]; then
      echo "SKIP (not built): ChaseDistributedSolvePseudoBSEGPUNCCLTest"
      FAIL=$((FAIL + 1))
      FAILED+=("ChaseDistributedSolvePseudoBSEGPUNCCLTest (missing)")
    fi
  fi
fi

echo ""
echo "Summary: ${PASS} passed, ${FAIL} failed/skipped"
if [[ ${FAIL} -gt 0 ]]; then
  echo "Failed:"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
