# `linalg/internal/gpu`

This directory contains GPU-specific internal code. CPU-only code remains in
`linalg/internal/cpu/`; CPU+MPI distributed code remains in
`linalg/internal/mpi/`.

The directory names describe architectural layers. Historical C++ namespaces
are intentionally kept for now (`gpu_backend`, `gpu_device`, `gpu_serial`,
`gpu_mpi`, `gpu_xccl`) so this move is a path cleanup, not an algorithm or ABI
rewrite.

```text
linalg/internal/
├── cpu/          # serial CPU internals
├── mpi/          # CPU + MPI distributed internals
└── gpu/
    ├── backend/  # CUDA/HIP runtime, BLAS, SOLVER, context, workspace plan
    ├── kernels/  # ChASE-owned CUDA/HIP device kernels and launchers
    ├── serial/   # single-GPU algorithms and CUDA/HIP wrappers
    ├── dist_mpi/ # GPU-aware MPI distributed algorithms
    └── dist_xccl/# NCCL/RCCL distributed algorithms
```

## Where New Code Goes

| Task | Directory |
|------|-----------|
| Runtime allocation, streams, events, memcpy | `gpu/backend/` |
| BLAS/SOLVER dispatch or handle utilities | `gpu/backend/` |
| ChASE CUDA/HIP kernels or launchers | `gpu/kernels/` |
| Single-GPU algorithm implementation | `gpu/serial/impl/` |
| CUDA/HIP single-GPU wrapper | `gpu/serial/cuda/` or `gpu/serial/hip/` |
| GPU-aware MPI algorithm implementation | `gpu/dist_mpi/impl/` |
| CUDA/HIP GPU-aware MPI wrapper | `gpu/dist_mpi/cuda/` or `gpu/dist_mpi/hip/` |
| NCCL/RCCL algorithm implementation | `gpu/dist_xccl/impl/` |
| NCCL/RCCL wrapper | `gpu/dist_xccl/nccl/` or `gpu/dist_xccl/rccl/` |

Do not mix `dist_mpi` and `dist_xccl`: the MPI path and NCCL/RCCL path use
different collective behavior and must stay independently testable.
Common runtime, BLAS, SOLVER, synchronization, and workspace helpers belong in
`gpu/backend/`, not in either distributed collective layer.
