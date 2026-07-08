// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * @page gpu_backend_module chase::linalg::internal::gpu_backend
 * @brief Umbrella entry point for serial GPU driver code.
 *
 * Include this header from `ChASEGPU`, `pChASEGPU`, and similar drivers that
 * need BLAS, solver, runtime, serial kernels, workspace planning, device
 * kernels, and the `active_stack` type bundle. Finer-grained headers remain
 * available for layers that only need a subset (for example `blas.hpp` alone).
 */

#include "linalg/internal/gpu/backend/serial_ops.hpp"
#include "linalg/internal/gpu/backend/workspace_plan.hpp"
#include "linalg/internal/gpu/backend/active_stack.hpp"
