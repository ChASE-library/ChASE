#!/bin/bash
# Distributed C/Fortran interface smoke tests on Frontier (inside a GPU allocation).
#
# Usage:
#   srun -n 4 --gpus-per-task=1 bash scripts/run_hip_dist_interface_smoke.sh
#
# Options (env vars):
#   CHASE_BUILD_DIR   build tree (default: build-frontier-hip)
#   CHASE_NPROCS      MPI ranks (default: 4)
#   CHASE_GPUS_TASK   GPUs per task (default: 1)
#   CHASE_LAUNCHER    override launcher

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${CHASE_BUILD_DIR:-${ROOT}/build-frontier-hip}"
NPROCS="${CHASE_NPROCS:-4}"
GPUS_PER_TASK="${CHASE_GPUS_TASK:-1}"
TOP_TESTS="${BUILD}/tests"
EXAMPLES="${BUILD}/examples/4_interface"

if [[ -f "${ROOT}/scripts/frontier_hip_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/frontier_hip_env.sh"
fi

if [[ -n "${SLURM_NTASKS:-}" || -n "${PMI_RANK:-}" || -n "${OMPI_COMM_WORLD_SIZE:-}" ]]; then
  LAUNCHER="${CHASE_LAUNCHER:-}"
else
  LAUNCHER="${CHASE_LAUNCHER:-srun -n ${NPROCS} --gpus-per-task=${GPUS_PER_TASK}}"
fi

PASS=0
FAIL=0
FAILED=()

run_one() {
  local name="$1"
  local exe="$2"
  echo ""
  echo "========== ${name} =========="
  if [[ ! -x "${exe}" ]]; then
    echo "SKIP (not built): ${exe}"
    FAIL=$((FAIL + 1))
    FAILED+=("${name} (missing)")
    return
  fi
  ${LAUNCHER} "${exe}"
  echo "PASSED: ${name}"
  PASS=$((PASS + 1))
}

echo "ChASE distributed C/Fortran interface smoke"
echo "  build: ${BUILD}"
echo "  launcher: ${LAUNCHER:-<current MPI rank set>}"

run_one "ChaseFortranDistributedSolveTest" \
  "${TOP_TESTS}/ChaseFortranDistributedSolveTest"
run_one "4_c_dist_chase.exe" \
  "${EXAMPLES}/4_c_dist_chase.exe"
run_one "4_f_dist_chase.exe" \
  "${EXAMPLES}/4_f_dist_chase.exe"

echo ""
echo "Summary: ${PASS} passed, ${FAIL} failed/skipped"
if [[ ${FAIL} -gt 0 ]]; then
  echo "Failed:"
  printf '  - %s\n' "${FAILED[@]}"
  exit 1
fi
