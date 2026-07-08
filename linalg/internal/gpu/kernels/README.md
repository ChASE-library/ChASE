# `gpu/kernels` — ChASE-owned GPU kernels

This directory contains ChASE kernel launchers and kernel translation units.
It does **not** own runtime allocation, BLAS, SOLVER, handles, or workspaces;
those belong to `linalg/internal/gpu/backend/`.

The C++ namespace is still `chase::linalg::internal::gpu_device`; the directory
name describes the architectural layer.

## Structure

```text
gpu/kernels/
├── shared/
│   ├── device_kernels.hpp   # public backend-neutral kernel dispatch
│   ├── orthog_kernels.hpp   # shared Orthog/Lanczos launcher facade
│   └── impl/                # implementation headers included by .cu/.hip files
├── cuda/
│   ├── launchers/           # CUDA launcher declarations
│   └── kernels/             # CUDA translation units
└── hip/
    ├── launchers/           # HIP launcher declarations
    └── kernels/             # HIP translation units
```

## Include Rules

Use this from algorithm code:

```cpp
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
```

Then dispatch through:

```cpp
chase::linalg::internal::gpu_device::kernels<Backend>
```

Do not include `shared/impl/*` from algorithms. Those headers are shared kernel
bodies used by CUDA/HIP translation units.

Do not use this layer for runtime or library glue:

- allocation, copies, streams, events: `gpu/backend/runtime.hpp`
- BLAS/SOLVER enums, status checks, handle streams: `gpu/backend/handle_utils.hpp`
- owned handles, `devInfo`, reusable workspace: `gpu/backend/context.hpp`

## Adding a Kernel

1. Add backend launcher declarations under `cuda/launchers/` and/or
   `hip/launchers/`.
2. Add translation units under `cuda/kernels/` and/or `hip/kernels/`.
3. Put shared implementation detail in `shared/impl/` only when CUDA and HIP can
   share the kernel body.
4. Expose a backend-neutral method on `shared/device_kernels.hpp`.
5. Register new translation units in `linalg/CMakeLists.txt`.
