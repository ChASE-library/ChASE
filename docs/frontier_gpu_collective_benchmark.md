# Frontier Distributed GPU Collective Benchmark Notes

This note records the current observation from the distributed GPU matrix
operations benchmark on Frontier. It is meant to guide future choices between
GPU-aware MPI and RCCL/NCCL collectives in pChASE.

## Benchmark Scope

The benchmark is implemented in:

```text
examples/7_matrix_operations_benchmark
```

It measures the communication-heavy matrix operations used by distributed
GPU pChASE:

- HEMM from column-distributed multivectors to row-distributed multivectors
- HEMM from row-distributed multivectors to column-distributed multivectors
- standalone device-buffer allreduce on row and column communicators
- `DistMultiVector` redistribution from column to row and back to column
- block and block-cyclic multivector layouts

The benchmark supports:

```text
--comm mpi|xccl|both
```

On Frontier/HIP, `mpi` means GPU-aware MPI and `xccl` means RCCL through the
NCCL-compatible API.

## Frontier Test Setup

Representative runs used:

```text
ranks=32
grid=8x4
n=80000
nvec=80
mb=256,512,1024
iters=20
warmup=3
comm=both
```

Both Cartesian rank orderings were tested:

```text
grid_major=col
grid_major=row
```

ChASE currently requires `grid_rows >= grid_cols`, so `8x4` is a legal shape,
while the reversed `4x8` shape is not.

## Main Observation

RCCL performance is extremely sensitive to which logical communicator maps to
the topology-friendly GPU/rank path.

For `grid=8x4, grid_major=col`:

```text
RCCL column communicator: fast
RCCL row communicator: very slow
```

Representative HEMM-payload allreduce timings:

```text
xccl-hemm-block-payload-col-to-row-allreduce  ~0.25 ms
xccl-hemm-block-payload-row-to-col-allreduce  ~21 ms
```

The full HEMM timing follows the same pattern:

```text
xccl-hemm-block-col-to-row  ~2.4 ms
xccl-hemm-block-row-to-col  ~21-23 ms
```

For `grid=8x4, grid_major=row`, the slow direction flips:

```text
RCCL row communicator: fast
RCCL column communicator: very slow
```

Representative HEMM-payload allreduce timings:

```text
xccl-hemm-block-payload-row-to-col-allreduce  ~0.42 ms
xccl-hemm-block-payload-col-to-row-allreduce  ~30 ms
```

The full HEMM timing again follows the same pattern:

```text
xccl-hemm-block-row-to-col  ~2.6 ms
xccl-hemm-block-col-to-row  ~30-32 ms
```

This indicates that the bottleneck is not GEMM, block vs block-cyclic layout,
or HEMM logic. The bottleneck is the RCCL allreduce on the topology-unfriendly
communicator.

## MPI Comparison

GPU-aware MPI is much more stable across the same runs.

For `grid_major=col`, representative HEMM-payload allreduce timings were:

```text
mpi-hemm-block-payload-col-to-row-allreduce  ~0.87 ms
mpi-hemm-block-payload-row-to-col-allreduce  ~0.98 ms
```

For `grid_major=row`, representative timings were:

```text
mpi-hemm-block-payload-col-to-row-allreduce  ~1.42 ms
mpi-hemm-block-payload-row-to-col-allreduce  ~0.35 ms
```

MPI is not always the fastest for the topology-friendly direction, but it avoids
the 20-30 ms RCCL outlier on the bad direction.

## Current Recommendation

For Frontier multi-node distributed pChASE, RCCL is the default collective path
when RCCL is linked at configure time (`DefaultDistributedGpu` → NCCL/RCCL).
GPU-aware MPI remains available as an opt-out:

```text
cmake ... -DCHASE_HIP_DISTRIBUTED_COLLECTIVES=MPI
```

RCCL performance is still extremely sensitive to which logical communicator maps
to the topology-friendly GPU/rank path. The data suggests a future hybrid
strategy may be beneficial:

```text
use RCCL for topology-friendly communicators
use GPU-aware MPI for topology-unfriendly communicators
```

However, that policy should be based on explicit benchmark evidence and should
not be hard-coded blindly.

## Implementation Status

Based on these observations, the default distributed GPU collective backend
depends on the platform:

- NVIDIA (CUDA + NCCL): defaults to NCCL, unchanged.
- AMD (HIP + RCCL): defaults to RCCL. GPU-aware MPI is opt-out.

This is centralized in `grid/default_distributed_backend.hpp`, which defines
`chase::grid::backend::DefaultDistributedGpu`. It is used as the default
`CommBackend` template argument of `pChASEGPU`, by the C/Fortran interface, by
the examples, and by the integration tests.

To opt out of RCCL on HIP, configure with
`-DCHASE_HIP_DISTRIBUTED_COLLECTIVES=MPI`, which propagates the
`CHASE_DISTRIBUTED_GPU_BACKEND_MPI` compile definition to the relevant targets.

The individual HEMM and redistribution kernels themselves are unchanged; only
the default choice of which collective backend the distributed solver uses has
been updated. A future topology-aware hybrid policy (as sketched above) remains
possible but is not yet implemented.

