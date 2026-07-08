// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * @page gpu_backend_types_module chase::linalg::internal::gpu_backend types
 * @brief Compile-time GPU backend tags for CUDA/HIP dispatch.
 *
 * Shared GPU code selects a vendor implementation by instantiating templates on
 * `cuda_backend` or `hip_backend`. Handle types, BLAS/SOLVER entry points, and
 * runtime helpers are provided by specializations in `blas.hpp`, `solver.hpp`,
 * and `runtime.hpp`.
 *
 * At most one active vendor is selected per build via `HAS_CUDA` or `HAS_HIP`.
 */

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_backend
{

/**
 * @brief Tag type for the NVIDIA CUDA backend (cuBLAS, cuSOLVER, CUDA runtime).
 */
struct cuda_backend
{
};

/**
 * @brief Tag type for the AMD HIP backend (rocBLAS, rocSOLVER, HIP runtime).
 */
struct hip_backend
{
};

/**
 * @brief Backend-specific type aliases and constants.
 *
 * Specializations live in `solver.hpp` (`traits<cuda_backend>`,
 * `traits<hip_backend>`) and supply handle types, status codes, and related
 * vendor typedefs used by `blas`, `solver`, and `GpuExecutionContext`.
 *
 * @tparam Backend `cuda_backend` or `hip_backend`.
 */
template <typename Backend>
struct traits;

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase
