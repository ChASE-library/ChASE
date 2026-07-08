#!/bin/bash
# Frontier AMD GPU/HIP environment for ChASE.
#
# Usage on Frontier login or inside a Slurm allocation:
#   source scripts/frontier_hip_env.sh
#
# Serial GPU (Phase 3) and distributed GPU-aware MPI (Phase 4+) both use this
# module stack. Load a matching cpe/cray-mpich/rocm trio per OLCF release notes.

module reset
module load PrgEnv-amd
module load amd
module load rocm
module load craype-accel-amd-gfx90a
module load cray-mpich
module load cmake
module load netlib-scalapack

# Required for Cray MPICH to pass GPU (HIP) pointers in MPI calls.
export MPICH_GPU_SUPPORT_ENABLED=1

# Help CMake FindRCCL (grid/mpi_grid_rccl) on Frontier.
if [[ -n "${ROCM_PATH:-}" ]]; then
  export RCCL_ROOT="${RCCL_ROOT:-${ROCM_PATH}}"
fi

# When using non-default cray-mpich/rocm modules, prepend Cray PE library paths.
if [[ -n "${CRAY_LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"
fi

# Optional: confirm the MPICH build at runtime (uncomment when debugging modules).
# export MPICH_VERSION_DISPLAY=1

module list
which cc
which CC
which ftn
which hipcc
cmake --version

echo ""
echo "HIP env ready. GPU-aware MPI: MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED}"
if [[ -n "${ROCM_PATH:-}" ]]; then
  echo "ROCM_PATH=${ROCM_PATH} (used by CMake FindRCCL)"
fi
echo "Configure ChASE with:  cmake -S . -B build-frontier-hip -DCHASE_ENABLE_HIP=ON -DCHASE_ENABLE_TESTS=ON ..."
echo "Serial HIP unit tests (inside GPU allocation):"
echo "  srun -n 1 --gpus-per-task=1 bash scripts/run_hip_serial_unittests.sh"
echo "Full HIP E2E regression (serial + distributed + interface):"
echo "  srun -n 4 --gpus-per-task=1 bash scripts/run_hip_e2e_tests.sh"
echo "RCCL grid test (inside GPU allocation, 4 ranks):"
echo "  srun -n 4 --gpus-per-task=1 build-frontier-hip/tests/MpiGrid2DTest"
echo "Runtime probe (inside GPU allocation):"
echo "  srun -n 2 build-frontier-hip/hip_mpi_probe"
