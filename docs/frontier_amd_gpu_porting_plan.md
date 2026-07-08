# Frontier AMD GPU Porting Plan

## Purpose

This document is the working plan for porting ChASE to AMD GPUs on Frontier/OLCF.
It also defines the collaboration boundary for local agent work: the agent edits
code, prepares patches, writes Slurm scripts, and analyzes logs; Frontier
performs the real configure, build, test, and benchmark steps through Slurm.

Important rule: do not treat Frontier compute nodes as persistent SSH targets.
They are reached only through Slurm allocations using `sbatch`, `salloc`, and
`srun`.

## Current Status Snapshot

Last updated: 2026-07-02.

The current Frontier/HIP direction is no longer just a plan. The single-GPU HIP
path and the distributed HIP-aware MPI/pChASE path have both reached a passing
correctness baseline on Frontier, including end-to-end distributed solver
integration tests. Keep this section updated after each meaningful porting step
so the next agent iteration starts from the real state, not from the original
early-stage assumptions below.

Final convergence is recorded in `docs/amd_port_convergence_plan.md`. The
current architecture — with **mermaid dependency and routing diagrams** — is
documented in
[`docs/amd_gpu_port_architecture.md`](amd_gpu_port_architecture.md#architecture-diagrams).

### Done

- Added `scripts/frontier_hip_env.sh` with the known Frontier module stack:
  `PrgEnv-amd`, `amd`, `rocm`, `craype-accel-amd-gfx90a`, `cray-mpich`, and
  `cmake`.
- Added HIP/ROCm build support guarded by `CHASE_ENABLE_HIP`, `HAS_HIP`, and
  `HAS_HIP_AWARE_MPI`, keeping the CUDA/NVIDIA path separate.
- Added the unified serial GPU implementation under `Impl/chase_gpu` (vendor
  conditional: CUDA or HIP).
- Added HIP kernel implementations under `linalg/internal/gpu/kernels/hip/` and
  serial algorithms under `linalg/internal/gpu/serial/hip/` with shared
  orchestration in `linalg/internal/gpu/serial/impl/`; tests under
  `tests/linalg/internal/gpu/kernels/hip` and `tests/linalg/internal/gpu/serial/hip`.
- Completed the serial `gpu_serial` shared-impl refactor (PR1–PR5): CholQR,
  Rayleigh–Ritz (Hermitian and pseudo-Hermitian), residuals, and Lanczos now
  share `linalg/internal/gpu/serial/impl/` with thin `cuda/` and `hip/`
  wrappers. GPU-resident Lanczos algorithm bodies remain in
  `lanczos_{cuda,hip}_detail.hpp` without behavioral changes.
- Removed the old CUDA-name compatibility bridge headers. Shared GPU code now
  uses neutral `gpu_backend` runtime/BLAS/SOLVER/serial-kernel entry points.
- Wired `interface/chase_c_interface.cpp` to `ChASEGPU` / `pChASEGPU` when
  `HAS_HIP` is active (serial and distributed C/Fortran paths).
- Added `scripts/run_hip_serial_unittests.sh` and `scripts/run_hip_e2e_tests.sh`
  for Frontier serial and full-stack HIP regression.
- Verified the serial HIP tests elsewhere on Frontier; do not spend time
  rerunning all serial HIP tests unless a related serial file changes.
- Added HIP-aware MPI kernels under `linalg/internal/gpu/dist_mpi/hip/` and tests
  under `tests/linalg/internal/gpu/dist_mpi/hip/`.
- Verified the following distributed HIP-aware MPI kernel tests on Frontier:
  `HEMMHIPDistTest`, `ResidualsHIPDistTest`, `shiftDiagonalHIPDistTest`,
  `SymOrHermHIPDistTest`, `CholQRHIPDistTest`, `LanczosHIPDistTest`, and
  `RayleighRitzHIPDistTest`.
- Fixed HIP-aware MPI `cholQR` runtime issues and removed temporary debug
  prints after Frontier validation.
- Added a GPU solver backend dispatch layer under
  `linalg/internal/gpu/backend`, with separate CUDA and HIP backend tags.
- Migrated pChASE HIP workspace-size queries for QR, POTRF, and HEEVD away from
  direct cuSOLVER-style assumptions.
- Implemented rocSOLVER workspace queries through the rocBLAS device-memory
  query model and explicit `rocblas_set_workspace` binding.
- Fixed HIP pChASE HEEVD/POTRF execution to use the backend dispatch instead of
  relying on cuSOLVER-compatible names at the main call sites.
- Verified the distributed solver BlockBlock GPU path on Frontier:

```bash
srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolveTest \
  --gtest_filter='*GPU*BlockBlock*'
```

- Added HIP pseudo-Hermitian Rayleigh-Ritz v2 wiring for the HIP-aware MPI
  path, including native rocBLAS enum/API names.
- Enabled the HIP pseudo-Hermitian Rayleigh-Ritz v1 entry point through the
  host LAPACK GEEV fallback, matching the CUDA-aware MPI no-XGEEV fallback
  behavior. ROCm-native GEEV remains unavailable.
- Wired pChASE HIP pseudo-Hermitian Rayleigh-Ritz to use the v2 POTRF/HEEVD
  path, matching the single practical solver path we want to maintain on ROCm.
  The v1 host LAPACK GEEV fallback remains available only as a low-level
  fallback/unit-test target. CUDA/NVIDIA keeps its existing `XGEEV_EXISTS`
  policy and v2 fallback behavior.
- Enabled the distributed pseudo-BSE solve test for HIP builds and made its GPU
  parameter use block-block layout with the MPI backend, so `--gtest_filter=*GPU*`
  runs a real HIP pChASE pseudo solve instead of matching zero tests.
- Fixed RCCL pseudo-Hermitian Rayleigh-Ritz v2 to use the ROCm backend solver
  dispatch for POTRF/HEEVD workspace handling, matching the pChASE HIP solver
  path. Verified both distributed pseudo-BSE GPU integration targets on
  Frontier:

```bash
srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolvePseudoBSEGPUNCCLTest \
  --gtest_filter='*GPU*'

srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolvePseudoBSETest \
  --gtest_filter='*GPU*'
```

- Added a formal GPU BLAS backend dispatch layer under
  `linalg/internal/gpu/backend/blas.hpp`. New or migrated shared code should
  call `gpu_backend::blas<backend>::...` instead of CUDA- or ROCm-named wrapper
  namespaces directly.
- Added shared CUDA/HIP-aware MPI implementation helpers under
  `linalg/internal/gpu/dist_mpi`:
  `residuals.hpp`, `rayleighRitz_impl.hpp`, and `cholqr_backend.hpp`.
- Migrated distributed CUDA/HIP-aware MPI `residuals` to the shared helper while
  preserving HIP's extra final residual broadcast and host-to-device update.
- Migrated Hermitian distributed CUDA/HIP-aware MPI `rayleighRitz` to the shared
  helper while keeping backend-specific HEEVD, triangular unpack/extract, and
  synchronization hooks.
- Migrated all distributed CUDA/HIP-aware MPI CholQR variants to backend BLAS
  and POTRF dispatch: `cholQR1`, `cholQR2`, `shiftedcholQR2`, and
  `modifiedGramSchmidtCholQR`.
- Migrated distributed CUDA/HIP-aware MPI HEMM and Hermitian Lanczos BLAS call
  sites to `gpu_backend::blas` dispatch. HIP Lanczos float dot/nrm2 now uses a
  GPU-resident double-accumulation reduction for Frontier stability.
- Migrated distributed CUDA/HIP-aware MPI pseudo-Hermitian Lanczos BLAS call
  sites to `gpu_backend::blas` dispatch while leaving the existing
  pseudo-Hermitian algorithm and MPI behavior unchanged.
- Migrated distributed CUDA/HIP-aware MPI pseudo-Hermitian Rayleigh-Ritz BLAS
  call sites to `gpu_backend::blas` dispatch. The v1 path can use host LAPACK
  GEEV on HIP when native GEEV is unavailable, but full pChASE HIP solves use
  the v2 POTRF/HEEVD path.
- Added `PseudoHermitianLanczosHIPDistTest` to cover HIP-aware MPI
  pseudo-Hermitian Lanczos full and simplified overloads, matching the existing
  CPU/NVIDIA pseudo-Hermitian Lanczos test structure.
- Enabled distributed pseudo-Hermitian GPU matrix type definitions for HIP as
  well as CUDA.
- Added `PseudoHEMMHIPDistTest` as HIP-aware MPI coverage for the
  pseudo-Hermitian HEMM relation, using the CUDA-aware MPI/NVIDIA GPU test as
  the reference behavior.
- Migrated all remaining distributed CUDA/HIP-aware MPI kernels into shared
  `linalg/internal/gpu/dist_mpi` helpers: `hemm_impl.hpp`, `flip_sign_impl.hpp`,
  `shift_diagonal_impl.hpp`, `sym_or_herm_impl.hpp`, `lanczos_impl.hpp`,
  `cholqr_impl.hpp`, `pseudo_hermitian_lanczos_impl.hpp`, and
  `pseudo_hermitian_rayleighRitz_impl.hpp` (+ backend hooks).
- Verified the full distributed HIP kernel unit-test suite on Frontier via
  `scripts/run_hip_dist_unittests.sh` (all `*HIPDistTest` targets plus
  `CholQRHIPDistStepsTest`).
- Verified end-to-end distributed HIP solver integration tests on Frontier:
  `ChaseDistributedSolveTest` (`*GPU*BlockBlock*`) and
  `ChaseDistributedSolvePseudoBSETest` (`*GPU*`), via
  `CHASE_RUN_INTEGRATION=1` in `run_hip_dist_unittests.sh`.
- Migrated the NCCL/RCCL distributed GPU collective path to shared
  implementations for CholQR, Hermitian Lanczos, pseudo-Hermitian Lanczos,
  Hermitian Rayleigh-Ritz, pseudo-Hermitian Rayleigh-Ritz, and
  symmetry/Hermitianization wrappers. The `nccl/` and `rccl/` namespaces remain
  thin compatibility facades, while the algorithm bodies are shared.
- Keep NCCL/RCCL Hermitian and pseudo-Hermitian Lanczos on the GPU-resident
  batched-kernel algorithm used by the NVIDIA NCCL implementation. Do not
  replace this path with the CUDA/HIP-aware MPI host-scalar Lanczos structure;
  GPU-aware MPI and NCCL/RCCL are intentionally different communication and
  execution paths here.
- Added a shared NCCL/RCCL pseudo-Hermitian Rayleigh-Ritz implementation with
  backend dispatch for CUDA/HIP BLAS, POTRF, HEEVD, diagonal shifts, and vector
  normalization. RCCL uses the same conceptual path as NCCL, with ROCm-specific
  solver behavior isolated behind backend dispatch.
- Verified that the integration tests pass after the shared NCCL/RCCL refactor:
  `ChaseDistributedSolveTest` and `ChaseDistributedSolvePseudoBSETest`.
- Verified all tests under `tests/linalg/internal/gpu/dist_xccl/rccl/` on Frontier after the
  shared NCCL/RCCL refactor.
- Verified that the code compiles on Frontier after the Scheme A distributed
  layer rename (`gpu_collective` → `gpu_xccl`, `gpu_aware_mpi` → `gpu_mpi`,
  `cuda_aware_mpi` → `cuda_mpi`, `hip_aware_mpi` → `hip_mpi`).
- Verified direct end-to-end distributed GPU integration commands after the
  distributed layer rename. Prefer these direct executable checks over the
  helper unit-test script for the current workflow:

```bash
srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolveTest \
  --gtest_filter='*GPU*'

srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolvePseudoBSETest \
  --gtest_filter='*GPU*'

srun -n 4 --gpus-per-task=1 \
  ./tests/ChaseDistributedSolvePseudoBSEGPUNCCLTest --gtest_filter='*GPU*'
```

- Documented current status in `linalg/internal/gpu/dist_mpi/hip/README.md`.
- GPU Householder QR remains intentionally low priority. Do not spend current
  AMD porting time on it unless a user explicitly changes the priority. The
  shared collective migration excludes the standalone
  `linalg/internal/gpu/dist_xccl/nccl/householder_qr.hpp` implementation.
- NVIDIA/NCCL distributed tests should still be run before final merge/release,
  but they may be deferred while NVIDIA queue time is high. Conceptually the
  NCCL path now mirrors the validated RCCL path through shared implementations.
- Added `linalg/internal/gpu/backend/context.hpp` as the first
  `GpuExecutionContext` implementation. The single-GPU `ChASEGPU` and
  distributed `pChASEGPU` paths now use it as the owner for BLAS/SOLVER handles,
  `devInfo`, and the main reusable solver workspace while preserving existing
  algorithm call sites. `ChASEGPU` keeps its owning stream behavior;
  `pChASEGPU` uses the context in default-stream mode to preserve the previous
  distributed execution ordering.
- Added `linalg/internal/gpu/backend/runtime.hpp` as the shared CUDA/HIP runtime
  portability layer. Matrix GPU allocation/copy helpers, distributed matrix GPU
  portability helpers, and `GpuExecutionContext` now route device allocation,
  frees, streams, synchronization, and column-major matrix copies through this
  backend-tagged layer. This keeps NVIDIA-only development independent from
  ROCm headers while still giving HIP the same public ChASE-side API.
- Migrated `ChASEGPU` and `pChASEGPU` direct runtime primitives
  (`cudaMalloc`, `cudaFree`, `cudaMemcpy`, device synchronization, events, and
  random-state types) to `gpu_backend::runtime<backend>`. Shared Impl code no
  longer keeps CUDA-era compatibility aliases.
- Completed the final convergence cleanup: stale `fixednev` was removed from
  `QR` / `Resd` APIs, and Orthog launcher dispatch now routes through
  `gpu_device::detail::orthog_kernels<Backend>` for shared serial,
  GPU-aware MPI, and xCCL paths.
- User reported the final Frontier test set passing after the final cleanup
  commits. AMD GPU v1 is ready to converge; distributed HIP GPU Householder QR
  remains intentionally deferred behind the ScaLAPACK fallback policy.

### Current Test Baseline

Use the Frontier environment script first:

```bash
source scripts/frontier_hip_env.sh
```

Recommended build check after pulling a new porting commit:

```bash
cmake --build build-frontier-hip -j16
```

Serial HIP kernel regression (after `gpu_serial` or `gpu_device` changes):

```bash
srun -n 1 --gpus-per-task=1 bash scripts/run_hip_serial_unittests.sh
```

Serial solver + C/Fortran interface smoke:

```bash
CHASE_RUN_INTEGRATION=1 CHASE_RUN_INTERFACE=1 \
  srun -n 1 --gpus-per-task=1 bash scripts/run_hip_serial_unittests.sh
```

Full HIP end-to-end (serial kernels + serial solve + distributed kernels +
distributed solve + interface smoke):

```bash
srun -n 4 --gpus-per-task=1 bash scripts/run_hip_e2e_tests.sh
```

Skip the long serial kernel sweep when only solver paths changed:

```bash
CHASE_SKIP_SERIAL_UNITTESTS=1 CHASE_SKIP_DIST_UNITTESTS=1 \
  srun -n 4 --gpus-per-task=1 bash scripts/run_hip_e2e_tests.sh
```

Recommended distributed integration regression after pulling a new porting
commit:

```bash
srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolveTest \
  --gtest_filter='*GPU*'

srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolvePseudoBSETest \
  --gtest_filter='*GPU*'

srun -n 4 --gpus-per-task=1 \
  ./tests/ChaseDistributedSolvePseudoBSEGPUNCCLTest --gtest_filter='*GPU*'
```

Optional broader RCCL/unit regression when a low-level kernel changes:

```bash
ctest --test-dir build-frontier-hip -R 'RCCL|GPUNCCL|DistributedSolve' \
  --output-on-failure
```

Deferred NVIDIA/NCCL regression before final merge/release:

```bash
srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolveTest \
  --gtest_filter='*GPU*'

srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolvePseudoBSETest \
  --gtest_filter='*GPU*'

ctest -R 'NCCL|DistributedSolve' --output-on-failure
```

Individual solver checks (if needed):

```bash
srun -n 4 --gpus-per-task=1 ./build-frontier-hip/tests/ChaseDistributedSolveTest \
  --gtest_filter='*GPU*BlockBlock*'

srun -n 4 --gpus-per-task=1 ./build-frontier-hip/tests/ChaseDistributedSolvePseudoBSETest \
  --gtest_filter='*GPU*'

srun -n 4 --gpus-per-task=1 ./build-frontier-hip/tests/ChaseDistributedSolvePseudoBSEGPUNCCLTest \
  --gtest_filter='*GPU*'
```

After pseudo-Hermitian Lanczos backend-dispatch changes, run the focused
HIP-aware MPI pseudo-Hermitian Lanczos test plus the broad distributed solver
smoke test:

```bash
cmake --build build-frontier-hip -j16

srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/PseudoHermitianLanczosHIPDistTest

srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolveTest \
  --gtest_filter='*GPU*BlockBlock*'
```

If the build tree is currently rooted inside `build-frontier-hip`, drop the
`build-frontier-hip/` prefix and run the binaries from that directory.

After CholQR or Rayleigh-Ritz changes, run the focused HIP-aware MPI tests first:

```bash
srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/CholQRHIPDistTest

srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/CholQRHIPDistStepsTest

srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/RayleighRitzHIPDistTest

srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/PseudoRayleighRitzHIPDistTest
```

After HEMM or Lanczos backend-dispatch changes, run:

```bash
srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/HEMMHIPDistTest

srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/PseudoHEMMHIPDistTest

srun -n 4 --gpus-per-task=1 \
  ./tests/linalg/internal/gpu/dist_mpi/hip/LanczosHIPDistTest

srun -n 4 --gpus-per-task=1 ./tests/ChaseDistributedSolveTest \
  --gtest_filter='*GPU*BlockBlock*'
```

### Next To Do (convergence phase)

**Active plan:** `docs/amd_port_convergence_plan.md` — Definition of Done for
closing the AMD port (v1: SolverWorkspacePlan, MPI Lanczos audit, C/F + examples;
Later: orthog merge; Example 6 CUDA-only; QR ScaLAPACK policy on AMD).

Priority 1: execute convergence v1 checklist in that document.

Priority 2: CUDA/NVIDIA regression before final merge/release (deferred while
queue time is high).

Priority 3: performance and larger cases after v1 sign-off.

Historical items below remain as background only.

### Known Rough Edges

- HIP pseudo-Hermitian Rayleigh-Ritz v1 uses a host LAPACK GEEV fallback because
  ROCm-native GEEV is still not available. Full pChASE HIP pseudo-Hermitian
  solves should use v2 POTRF/HEEVD instead.
- Some variables in shared pChASE code still use CUDA-era names such as
  `cublas_handle` or `cusolver_handle` even when the active backend is HIP.
- rocSOLVER workspace handling is not cuSOLVER-compatible. The HIP path uses
  rocBLAS device-memory-size queries and explicit workspace binding.
- The MPI synchronization/collective pattern is correct enough for current
  tests but should be audited later for cleanliness and performance.

### Shared Backend Design Notes

- `linalg/internal/gpu/backend/runtime.hpp` owns backend-specific runtime
  dispatch for allocation, free, memset, pinned host allocation, streams, device
  synchronization, and column-major matrix copies. CUDA and HIP are implemented
  as independent `runtime<backend>` specializations; NVIDIA-only builds should
  not require ROCm headers.
- `linalg/internal/gpu/backend/solver.hpp` owns backend-specific SOLVER
  dispatch. CUDA calls `cusolverpp`; HIP calls native rocSOLVER wrappers and
  uses the rocBLAS workspace model where required.
- `linalg/internal/gpu/backend/blas.hpp` owns backend-specific BLAS dispatch.
  CUDA calls `cublaspp`; HIP calls `rocblaspp`.
- `linalg/internal/gpu/backend/serial_ops.hpp` selects the active CUDA or
  HIP serial-kernel namespace for shared `ChASEGPU` and `pChASEGPU` call sites.
  It replaces the old CUDA-name bridge; HIP builds no longer alias HIP kernels
  into `internal::cuda`.
- `linalg/internal/gpu/serial/impl/` shares serial GPU algorithms between CUDA
  and HIP: CholQR, Rayleigh–Ritz (standard and pseudo-Hermitian), residuals, and
  Lanczos dispatch. Backend-specific Lanczos bodies live in
  `lanczos_{cuda,hip}_detail.hpp`; thin `cuda/` and `hip/` headers re-export
  the public API.
- `linalg/internal/gpu/dist_mpi/impl/residuals_impl.hpp` is the first complete
  shared distributed GPU MPI helper. It keeps the HIP-only residual broadcast as
  an explicit parameter.
- `linalg/internal/gpu/dist_mpi/impl/rayleighRitz_impl.hpp` shares the Hermitian
  Rayleigh-Ritz skeleton and delegates BLAS enums, triangular extraction,
  synchronization, and HEEVD execution to backend hooks.
- `linalg/internal/gpu/dist_mpi/impl/cholqr_backend.hpp` currently provides backend
  hooks used by all distributed CholQR variants: BLAS enum mapping, BLAS status
  checks, transpose selection, POTRF workspace sizing, and POTRF execution.
- Distributed CUDA/HIP-aware MPI HEMM, Lanczos, CholQR, residuals,
  shiftDiagonal, symOrHerm, flipSign, and pseudo-Hermitian Lanczos/Rayleigh-Ritz
  now share implementation in `linalg/internal/gpu/dist_mpi/impl/`. Backend-specific
  files are thin wrappers; BLAS/solver execution goes through `gpu_backend`.
- HIP pseudo-Hermitian Rayleigh-Ritz v1 still uses host LAPACK GEEV when native
  GEEV is unavailable; full pChASE HIP solves use the v2 POTRF/HEEVD path.
- GPU `householder_qr` is the main remaining CUDA-only distributed kernel
  (`gpu/dist_mpi/cuda/householder_qr.hpp`).

### Current GPU Architecture Boundaries

The current porting architecture should stay layered around execution model,
not around vendor names alone. This is the boundary to preserve while doing the
next cleanup passes.

- `Impl/chase_gpu` and `Impl/pchase_gpu` own solver orchestration, GPU handles,
  streams, high-level workspaces, and calls into internal GPU kernels. They
  should not accumulate new CUDA- or ROCm-specific algorithm branches when a
  backend dispatch helper can express the difference.
- `linalg/internal/gpu/backend` owns narrow BLAS/SOLVER dispatch. It translates
  backend tags into CUDA or ROCm handles, enums, workspace queries, workspace
  binding, and status checks. It should not own ChASE algorithm policy.
- `linalg/internal/gpu/backend/runtime.hpp` owns CUDA/HIP runtime primitives.
  Higher-level files such as `matrix_gpu_runtime.hpp` and
  `dist_gpu_portability.hpp` may keep their public-facing names, but their
  allocation, copy, stream, and synchronization operations should forward to
  this backend layer.
- `linalg/internal/gpu/kernels` owns backend-neutral device operations and the
  corresponding CUDA/HIP kernels. This is the right layer for small fused
  kernels, triangular packing/unpacking, shifts, normalization, and simple
  elementwise transformations.
- `linalg/internal/gpu/serial` owns single-GPU algorithms. The `impl/` folder is
  the shared algorithm layer; the `cuda/` and `hip/` folders should stay thin
  public wrappers or backend-specific detail hooks.
- `linalg/internal/gpu/dist_mpi` owns the GPU-aware MPI path. This path is allowed to
  keep the conservative MPI/host-scalar structure inherited from the NVIDIA
  CUDA-aware MPI implementation. It should not be rewritten to look like the
  NCCL/RCCL path just for symmetry.
- `linalg/internal/gpu/dist_xccl` owns the NCCL/RCCL collective path. This path must
  preserve the NVIDIA NCCL execution model as the reference: GPU-resident data,
  batched operations where the master branch uses them, fused kernels where the
  master branch uses them, and device collectives for the distributed reductions.

Boundary rules for future changes:

- Do not mix `gpu_mpi` and `gpu_xccl` implementations. They solve different
  communication problems and intentionally have different execution structures.
- New shared algorithm code should go into the relevant `impl/` folder, while
  vendor folders should mostly translate types, handles, enums, and
  communicator details.
- CUDA/NVIDIA remains the behavioral reference. HIP should diverge only when
  ROCm APIs, missing ROCm functionality, or Frontier MPI behavior require it.
- Keep long-lived solver and BLAS workspace allocated by `ChASEGPU` /
  `pChASEGPU` constructors wherever possible. Temporary allocation inside hot
  kernels is an optimization target unless it is also present in the NVIDIA
  reference path or only used by low-priority/deferred code.
- GPU Householder QR remains deferred and should not block the AMD porting
  cleanup unless the priority changes explicitly.

### GpuExecutionContext Design Sketch

The next structural cleanup target is to reduce scattered GPU resource
ownership. Recent HIP bugs came from this exact problem: BLAS and SOLVER handles
were easy to alias accidentally, compatibility wrappers could silently ignore
the intended solver handle, and matrix-copy compatibility helpers hid
backend-specific runtime details in a high-level file.

`GpuExecutionContext` is a small owning object for backend execution resources,
not a new algorithm layer. It makes resource lifetime and workspace ownership
explicit while leaving Lanczos, Rayleigh-Ritz, CholQR, and solver policy in
their current algorithm layers.

One possible shape:

```cpp
template <typename Backend>
class GpuExecutionContext {
public:
    using blas_handle_type = typename BackendTraits<Backend>::blas_handle_type;
    using solver_handle_type = typename BackendTraits<Backend>::solver_handle_type;
    using stream_type = typename BackendTraits<Backend>::stream_type;

    explicit GpuExecutionContext(bool create_stream = true);
    ~GpuExecutionContext();

    GpuExecutionContext(const GpuExecutionContext&) = delete;
    GpuExecutionContext& operator=(const GpuExecutionContext&) = delete;

    blas_handle_type blas() const;
    solver_handle_type solver() const;
    stream_type stream() const;

    void set_stream(stream_type stream, bool take_ownership = false);
    int* dev_info();

    template <typename T>
    T* work();

    int lwork() const;
    void ensure_workspace_bytes(std::size_t bytes);
};
```

The backend traits are the only place that should know the raw vendor resource
types:

```cpp
template <typename Backend>
struct BackendTraits;

template <>
struct BackendTraits<gpu_backend::cuda_backend> {
    using blas_handle_type = cublasHandle_t;
    using solver_handle_type = cusolverDnHandle_t;
    using stream_type = cudaStream_t;
};

template <>
struct BackendTraits<gpu_backend::hip_backend> {
    using blas_handle_type = rocblas_handle;
    using solver_handle_type = rocblas_handle;
    using stream_type = hipStream_t;
};
```

For HIP, `blas_handle_type` and `solver_handle_type` are both `rocblas_handle`,
but they must still be created and destroyed as separate handles. The type alone
does not express ownership, and this is exactly why an execution context is
useful.

The first version should keep workspace planning simple:

```cpp
struct SolverWorkspacePlan {
    int geqrf_lwork = 0;
    int gqr_lwork = 0;
    int heevd_lwork = 0;
    int potrf_lwork = 0;
    int packed_triangular_lwork = 0;

    int main_lwork() const;
};
```

`ChASEGPU` would build this plan from its serial dimensions in the constructor,
allocate one reusable main workspace, and pass `context.blas()`,
`context.solver()`, `context.work<T>()`, `context.lwork()`, and
`context.dev_info()` into the existing call sites. `pChASEGPU` would do the
same using local distributed dimensions and the existing HIP exclusions for
deferred Householder QR workspace queries.

Migration stages:

1. Introduce `GpuExecutionContext` as a thin wrapper around the current handle,
   stream, `devInfo`, and workspace allocation logic. Do not move algorithms.
   This is now implemented for both `ChASEGPU` and `pChASEGPU`.
2. Validate the serial and distributed CUDA/HIP solver paths, because this layer
   recently exposed handle/copy bugs.
3. Add `SolverWorkspacePlan` only after the handle wrapper is stable. This keeps
   lifetime cleanup separate from workspace-sizing cleanup.
4. Later, consider reusable scratch buffers for selected hot internal kernels
   such as pseudo-Hermitian normalization scratch or double-precision temporary
   Rayleigh-Ritz buffers, but only after confirming the NVIDIA reference path and
   current Frontier timings.

Non-goals for `GpuExecutionContext`:

- It should not merge `gpu_mpi` and `gpu_xccl`.
- It should not hide whether an algorithm is using MPI collectives or NCCL/RCCL
  collectives.
- It should not decide between pseudo-Hermitian v1 GEEV and v2 POTRF/HEEVD.
- It should not force Householder QR into the active porting path.
- It should not become a broad global singleton. ChASE objects should still own
  their execution resources clearly.

## Access Model

The expected connection path is:

```text
local machine
  -> ssh home.ccs.ornl.gov
  -> ssh frontier.olcf.ornl.gov
  -> sbatch / salloc / srun
  -> Frontier compute nodes
```

Recommended local SSH configuration:

```sshconfig
Host olcf-home
    HostName home.ccs.ornl.gov
    User xinzhewu

Host frontier
    HostName frontier.olcf.ornl.gov
    User xinzhewu
    ProxyJump olcf-home
```

Smoke test:

```bash
ssh frontier
```

## Agent Boundary

The local agent may:

- inspect the local source tree;
- edit C++, CUDA/HIP, CMake, documentation, and scripts;
- generate minimal patches;
- generate Slurm build/test scripts;
- analyze compiler, linker, runtime, and test logs returned from Frontier;
- suggest module, compiler wrapper, ROCm, MPI, and runtime fixes.

The local agent must not:

- SSH to Frontier compute nodes directly;
- run heavy builds or benchmarks on Frontier login nodes;
- assume local macOS builds are equivalent to Frontier builds;
- run Codex or other AI tools on Frontier unless explicitly allowed;
- store API credentials on Frontier;
- rewrite unrelated parts of the build system when a focused patch is enough.

## Development Loop

Use this loop for every porting iteration:

1. Inspect and edit the local checkout.
2. Generate a patch locally.
3. Copy the patch to Frontier.
4. Apply the patch on Frontier.
5. Configure, build, and test through Slurm.
6. Return logs to the local agent.
7. Diagnose the logs and generate the next minimal patch.

Local patch commands:

```bash
git diff > frontier.patch
scp frontier.patch frontier:~/ChASE/
```

Frontier patch commands:

```bash
ssh frontier
cd ~/ChASE
git apply frontier.patch
git diff
```

## Historical Plan Notes

The sections below capture the original porting assessment and staged plan.
They are useful as background, but the `Current Status Snapshot` above is the
source of truth for the active development state. In particular, the RCCL path
is now implemented and validated for the current integration tests, so early
notes that say to avoid NCCL/RCCL-equivalent work describe the original
milestone ordering rather than the current capability.

## Difficulty Assessment

Overall difficulty: **medium-high to high**.

This is not a simple CMake option change. The current GPU implementation is
deeply CUDA-oriented in four layers:

- **Build system:** GPU enablement is keyed on `CMAKE_CUDA_COMPILER`,
  `CUDAToolkit`, `CUDA::cublas`, `CUDA::cusolver`, `CUDA::curand`, and
  `HAS_CUDA`.
- **Runtime and memory:** matrix classes and backends use CUDA runtime types and
  APIs such as `cudaStream_t`, `cudaMalloc`, `cudaMemcpy`, `cudaFree`, and CUDA
  error handling.
- **BLAS/SOLVER wrappers:** CUDA and HIP expose different library names and
  some different workspace semantics. Shared code should use the neutral
  `gpu_backend` BLAS/SOLVER dispatch instead of CUDA-name compatibility.
- **Custom kernels and communication:** `linalg/internal/gpu/kernels/cuda` contains many
  `.cu` / `.cuh` kernels, and the distributed GPU path has CUDA-aware MPI and
  NCCL-specific code.

The good news is that the high-level solver design is already backend-abstract:
`Algorithm<T>` calls `ChaseBase<T>`, and concrete backends implement the
operations. That means the AMD port can be staged below the algorithm layer
without rewriting the solver itself.

Risk by subsystem:

| Area | Difficulty | Main risk |
| --- | --- | --- |
| CMake feature detection | Medium | Avoid breaking existing NVIDIA builds while adding HIP/ROCm targets. |
| CUDA runtime API mapping | Medium | Mechanical but widespread; stream, memory, error, and event semantics must stay consistent. |
| cuBLAS to rocBLAS | Medium | Most BLAS calls should map, but complex type and pointer-mode details need care. |
| cuSOLVER to rocSOLVER | High | Eigen/SVD/QR/LAPACK-style routines may not be one-to-one. This is likely the hardest library layer. |
| Custom `.cu` kernels | Medium-high | Many kernels may compile with `hipcc` after type/header conversion, but fused Lanczos/Householder kernels need validation. |
| NCCL/RCCL collective path | High, now partially retired risk | Early plan said to avoid this first. Current status: RCCL integration tests pass through the shared `gpu_xccl` layer; NVIDIA/NCCL final regression remains deferred until queue time is acceptable. |
| Distributed GPU correctness | High | GPU buffers plus MPI collectives are where correctness bugs can hide. |
| Performance tuning | High | Correctness can arrive first; matching NVIDIA performance will require later profiling. |

Recommended strategy: **do not fork the project into AMD-only code**. Keep
NVIDIA as the primary development and regression environment, and add AMD as a
parallel backend path guarded by explicit build options/macros.

## Dual-Track Development Strategy

The main development environment remains NVIDIA. Frontier AMD support should be
developed as an additional target, not as a replacement for the current CUDA
target.

Track A: NVIDIA primary path

- Preserve existing `chase_gpu`, `pchase_gpu`, CUDA, cuBLAS, cuSOLVER, and NCCL
  behavior.
- Run NVIDIA builds/tests after every nontrivial GPU abstraction change.
- Treat NVIDIA test success as the first regression gate before asking Frontier
  to spend queue time.
- Keep file and class names stable where practical, even if internal aliases
  become more generic.

Track B: AMD Frontier path

- Add HIP/ROCm detection and AMD targets incrementally.
- Start with single-GPU `chase_gpu`; defer ROCm-aware MPI and `pchase_gpu`
  until the serial GPU path is correct.
- Historical first-milestone note: do not attempt an NCCL-equivalent path before
  the serial/HIP-aware MPI baselines are correct. This has since advanced: RCCL
  now works through the shared `gpu_xccl` layer for the validated
  integration tests.
- Use Frontier logs as the source of truth for AMD compile/runtime behavior.
- Keep AMD-specific fixes small and return logs after each Slurm run.

Practical rule: every shared abstraction must be tested against NVIDIA first,
then Frontier. Every AMD-only workaround must be isolated behind `HAS_HIP`,
`CHASE_ENABLE_HIP`, or similarly explicit feature guards.

## Initial Scope: Single-GPU `chase_gpu`

Start with the serial GPU backend only. The first AMD target should be a
single-GPU equivalent of the current NVIDIA `chase_gpu` path.

In scope for the first port:

- `Impl/chase_hip/*` as the experimental AMD single-GPU implementation area;
- `Impl/chase_gpu/chase_gpu.hpp`;
- `Impl/chase_gpu/CMakeLists.txt`;
- `Impl/chase_gpu/cuda_utils.hpp` or a new runtime abstraction beside it;
- `linalg/matrix/matrix.hpp` GPU matrix allocation and copy support;
- `linalg/internal/gpu/kernels/cuda/launchers/*` kernels needed by serial `ChASEGPU`;
- `external/nvidia_gpu/cublaspp` and `external/nvidia_gpu/cusolverpp`
  equivalents or compatibility wrappers (AMD: `external/amd_gpu/rocblaspp`,
  `external/amd_gpu/rocsolverpp`);
- serial GPU tests such as `ChaseSerialSolveTest`.

Out of scope for the first port:

- `Impl/pchase_gpu`;
- `linalg/internal/gpu/dist_xccl/nccl`;
- `grid` NCCL support;
- distributed GPU multi-vector communication;
- ROCm-aware MPI correctness;
- multi-GPU performance tuning;
- pseudo-Hermitian distributed GPU behavior.

This keeps the first milestone small enough to debug from Frontier logs. It also
protects the main NVIDIA development path because `chase_gpu` can be tested on
NVIDIA before each Frontier iteration.

First single-GPU success criterion:

- configure detects HIP/ROCm on Frontier;
- a HIP single-GPU target compiles for `gfx90a`;
- the first standalone HIP kernel test (`shiftDiagonalHIPTest`) passes;
- a small serial GPU Hermitian solve runs through `chase::Solve`;
- the same shared changes still compile and pass the NVIDIA `chase_gpu` baseline.

First experimental code slice:

- add `Impl/chase_hip` as an independent HIP target;
- add `linalg/internal/gpu/kernels/hip/launchers/shift_diagonal.hpp` and
  `linalg/internal/gpu/kernels/hip/kernels/shift_diagonal.hip` as the first
  standalone HIP kernel;
- add `tests/linalg/internal/gpu/kernels/hip/shiftDiagonal.cpp` as the first HIP
  unit test;
- keep the target behind `-DCHASE_ENABLE_HIP=ON`.

## Detailed Porting Plan

### Phase 0: Protect the NVIDIA baseline

Goal: establish that later AMD work does not damage the current main GPU path.

Actions:

- Record the known-good NVIDIA configure command.
- Build the existing CUDA GPU targets before GPU abstraction edits.
- Run a small serial GPU solve test and, if available, a small distributed NCCL
  test.
- Keep this as the local/main regression command set.

Suggested NVIDIA baseline commands:

```bash
cmake -S . -B build-nvidia \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHASE_ENABLE_TESTS=ON \
  -DCHASE_BUILD_WITH_EXAMPLES=ON

cmake --build build-nvidia -j16
ctest --test-dir build-nvidia --output-on-failure
```

If the local NVIDIA machine cannot run all MPI/GPU tests, at least build all GPU
targets and run the smallest available GPU test.

### Phase 1: Frontier single-GPU baseline

Goal: establish a reproducible Frontier build baseline for the non-distributed
GPU backend before any multi-GPU work.

Actions:

- confirm the current ChASE checkout builds with Frontier compiler wrappers;
- capture the exact loaded modules and CMake configure output;
- identify all CUDA-only assumptions in CMake and source files;
- keep the first build scripts small and reproducible;
- first target should be CPU configure and build, then HIP configure-only;
- do not enable `pchase_gpu`, NCCL, or distributed GPU tests in this phase.

### Phase 2: Build-system abstraction

Goal: make CMake distinguish NVIDIA CUDA and AMD ROCm/HIP cleanly for
`chase_gpu`.

Likely files:

- `CMakeLists.txt`;
- `linalg/CMakeLists.txt`;
- `Impl/CMakeLists.txt`;
- `Impl/chase_gpu/CMakeLists.txt`;
- `cmake/modules/*`;
- `chase_config.h.in`.

Expected changes:

- add ROCm/HIP feature detection without breaking CUDA builds;
- introduce config macros for HIP/ROCm separately from CUDA;
- use Frontier compiler wrappers `cc`, `CC`, and `ftn`;
- pass `gfx90a` as the Frontier GPU architecture;
- avoid assuming CUDA-aware MPI and NCCL are available in the AMD path;
- create only the serial GPU AMD target first.

Design target:

- keep `HAS_CUDA` for NVIDIA CUDA only;
- add a separate macro such as `HAS_HIP` or `CHASE_HAS_HIP`;
- add a generic internal macro only if it simplifies shared code, for example
  `CHASE_HAS_GPU`;
- do not redefine CUDA names to HIP names globally in public headers.

### Phase 3: Single-GPU backend portability

Goal: isolate CUDA-specific serial GPU code and decide what can be reused for
HIP in `ChASEGPU`.

Likely files:

- `Impl/chase_gpu/*`;
- `linalg/internal/gpu/kernels/cuda/launchers/*`;
- `external/nvidia_gpu/cublaspp/*`;
- `external/nvidia_gpu/cusolverpp/*`.

Expected changes:

- replace or wrap CUDA runtime calls with HIP equivalents where appropriate;
- map cuBLAS/cuSOLVER usage to rocBLAS/rocSOLVER through the neutral
  `gpu_backend` dispatch headers;
- keep the existing CUDA path intact while adding the AMD path;
- keep backend changes narrow and test after each slice.

Recommended implementation order:

1. Add neutral GPU runtime dispatch for stream, memory, memcpy, and error
   checks.
2. Add rocBLAS wrappers matching the existing `cublaspp` call surface.
3. Audit cuSOLVER calls and map only the routines actually used by ChASE.
4. Port the smallest utility kernels first: `shiftDiagonal`, `flipSign`,
   `lacpy`, `residuals`.
5. Port larger kernels later: Lanczos, Householder QR, pseudo-Hermitian
   Lanczos.

The early compatibility bridge has now been retired. Future shared GPU work
should add explicit CUDA/HIP wrappers or backend-neutral dispatch under
`linalg/internal/gpu/backend` rather than reintroducing CUDA-name aliases.

### Phase 4: Single-GPU validation

Goal: prove that the serial AMD GPU backend is correct before touching
distributed GPU code.

Likely files:

- `tests/chase_serial_solve.cpp`;
- `tests/linalg/internal/gpu/kernels/cuda/launchers/*` or HIP-equivalent test targets;
- `examples/1_hello_world/1_hello_world.cpp`;
- `tests/noinput.cpp`.

Expected changes:

- add or adapt a small serial GPU test that can run under `srun -n 1`;
- compare a few eigenvalues/residuals against CPU or NVIDIA reference output;
- keep the test matrix small enough for fast Frontier queue iterations;
- do not use MPI communication beyond launching one process.

First serial AMD test command should look like:

```bash
srun -n 1 ./build-frontier/tests/shiftDiagonalHIPTest
srun -n 1 ./build-frontier/tests/ChaseSerialSolveTest
```

If the GTest binary contains both CPU and GPU parameterizations, use the GTest
filter needed to run only the serial GPU case once that filter name is known
from Frontier build output.

### Phase 5: Distributed GPU communication, later

Goal: after `chase_gpu` works, make the distributed GPU path work with
Frontier's ROCm-aware MPI environment.

Likely files:

- `Impl/pchase_gpu/pchase_gpu.hpp`;
- `Impl/pchase_gpu/CMakeLists.txt`;
- `grid/*`;
- `linalg/internal/gpu/dist_mpi/cuda/*` or a new HIP-aware equivalent.

First distributed AMD milestone, later:

- build `pchase_gpu` without NCCL;
- use ROCm-aware MPI collectives and point-to-point communication;
- run one rank per GPU;
- validate a small Hermitian solve before pseudo-Hermitian/BSE cases.

### Phase 6: Validation and performance

Goal: verify correctness first, then benchmark.

Validation order:

1. configure-only build;
2. compile core libraries;
3. compile serial tests;
4. run a small CPU serial test;
5. run a small serial GPU test;
6. only then compile distributed tests;
7. run a distributed GPU test;
8. only then run larger examples or benchmarks.

Acceptance criteria for first AMD milestone:

- CMake configures on Frontier with ROCm/HIP enabled.
- Core GPU kernels compile for `gfx90a`.
- A small serial GPU Hermitian solve runs correctly.
- NVIDIA CUDA build still configures, compiles, and passes the baseline tests.

Acceptance criteria for second AMD milestone:

- A small distributed GPU Hermitian solve runs correctly with ROCm-aware MPI.
- Pseudo-Hermitian GPU path compiles on Frontier.
- Small BSE-style test agrees with CPU or NVIDIA reference within tolerance.
- No AMD-specific workaround changes CUDA/NCCL behavior.

## Frontier Build Script Template

Use the saved environment template when running interactively:

```bash
source scripts/frontier_hip_env.sh
```

The module block in that script is:

```bash
module reset
module load PrgEnv-amd
module load amd
module load rocm
module load craype-accel-amd-gfx90a
module load cray-mpich
module load cmake

module list
which cc
which CC
which ftn
which hipcc
cmake --version
```

Create `build_chase_frontier.slurm` on Frontier:

```bash
#!/bin/bash
#SBATCH -A <PROJECT_ID>
   #SBATCH -J chase-gpu-build
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p batch
   #SBATCH -o chase-gpu-build-%j.out
   #SBATCH -e chase-gpu-build-%j.err

set -euo pipefail

module reset
module load PrgEnv-amd
module load amd
module load rocm
module load craype-accel-amd-gfx90a
module load cray-mpich
module load cmake

cd ~/ChASE

   cmake -S . -B build-frontier \
     -DCMAKE_BUILD_TYPE=Release \
     -DCMAKE_C_COMPILER=cc \
     -DCMAKE_CXX_COMPILER=CC \
     -DCMAKE_Fortran_COMPILER=ftn \
     -DCMAKE_HIP_ARCHITECTURES=gfx90a \
     -DCHASE_ENABLE_TESTS=ON \
     -DCHASE_BUILD_WITH_EXAMPLES=ON \
     -DCHASE_BUILD_WITH_DOCS=OFF

   cmake --build build-frontier -j16
```

Submit and inspect:

```bash
sbatch build_chase_frontier.slurm
squeue -u xinzhewu
   cat chase-gpu-build-<jobid>.out
   cat chase-gpu-build-<jobid>.err
```

## Frontier Test Script Template

Create `test_chase_frontier.slurm` on Frontier:

```bash
#!/bin/bash
#SBATCH -A <PROJECT_ID>
   #SBATCH -J chase-gpu-test
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -p batch
   #SBATCH -o chase-gpu-test-%j.out
   #SBATCH -e chase-gpu-test-%j.err

set -euo pipefail

module reset
module load PrgEnv-amd
module load amd
module load rocm
module load craype-accel-amd-gfx90a
module load cray-mpich
module load cmake

cd ~/ChASE

   srun -n 1 ./build-frontier/tests/ChaseSerialSolveTest
```

Submit and inspect:

```bash
sbatch test_chase_frontier.slurm
squeue -u xinzhewu
   cat chase-gpu-test-<jobid>.out
   cat chase-gpu-test-<jobid>.err
```

## Interactive Debug Allocation

Use this only for human-controlled debugging:

```bash
salloc -A <PROJECT_ID> -N 1 -t 00:30:00 -p batch
module reset
module load PrgEnv-amd amd rocm craype-accel-amd-gfx90a cray-mpich cmake
cd ~/ChASE
cmake --build build-frontier -j16
srun -n 1 ./build-frontier/tests/ChaseSerialSolveTest
exit
```

The local agent should treat this as a temporary debug session controlled by the
user, not as a persistent remote backend.

## Failure Triage Checklist

When a build or test fails on Frontier, collect:

```bash
module list
which cc
which CC
which ftn
which hipcc
cmake --version
rocminfo | head
cd ~/ChASE/build-frontier
grep -E "CMAKE_|HIP|ROCM|MPI" CMakeCache.txt
```

For build errors:

```bash
cat chase-build-<jobid>.out
cat chase-build-<jobid>.err
```

For test errors:

```bash
cat chase-test-<jobid>.out
cat chase-test-<jobid>.err
```

Return this information to the local agent:

- the Slurm script;
- `module list` output;
- compiler wrapper paths;
- CMake configure output;
- the exact compiler command, if visible;
- stdout/stderr logs;
- relevant source files or patch context.

## Operating Checklist For Each Agent Change

Before editing:

- state whether the change targets CMake detection, HIP runtime portability,
  BLAS/SOLVER portability, MPI communication, or tests;
- identify the smallest file set to touch;
- preserve existing CUDA behavior unless the user explicitly asks otherwise.

After editing:

- show the local diff summary;
- provide the patch generation command;
- provide the Frontier build/test command to run next;
- ask the user to return logs rather than trying to access Frontier directly.

## One-Line Rule

```text
Codex stays local. Frontier login stores and submits. Frontier compute nodes build and run only through Slurm. Logs come back for the next patch.
```
