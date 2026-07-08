// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/context.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"

/**
 * @page gpu_backend_active_stack_module chase::linalg::internal::gpu_backend::active_stack
 * @brief Bundled type aliases for the active GPU backend.
 *
 * Driver code (`ChASEGPU`, `pChASEGPU`) uses `active_stack` to obtain BLAS,
 * solver, runtime, kernel dispatch, handle types, and execution context without
 * repeating `#if` backend-selection logic at every call site.
 */

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_backend
{

#if defined(HAS_CUDA) || defined(HAS_HIP)
/**
 * @brief Type bundle for one GPU backend tag.
 *
 * @tparam Backend `cuda_backend` or `hip_backend`.
 */
template <typename Backend>
struct backend_stack
{
    using backend = Backend;
    using solver = gpu_backend::solver<Backend>;
    using blas = gpu_backend::blas<Backend>;
    using execution_context = GpuExecutionContext<Backend>;
    using runtime = gpu_backend::runtime<Backend>;
    using handle_utils = gpu_backend::handle_utils<Backend>;
    using kernels = gpu_device::kernels<Backend>;
    using stream_type = typename runtime::stream_type;
    using event_type = typename runtime::event_type;
    using rand_state_type = typename runtime::rand_state_type;
    using blas_handle = typename traits<Backend>::blas_handle;
    using solver_handle = typename traits<Backend>::solver_handle;
    using solver_params = typename traits<Backend>::params_type;
    using device_deleter = runtime_device_deleter<Backend>;
};

/** @brief Stack for the backend selected at configure time. */
using active_stack = backend_stack<active_backend>;
#endif

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase
