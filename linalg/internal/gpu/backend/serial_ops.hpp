// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/types.hpp"

#if defined(HAS_CUDA)
#include "linalg/internal/gpu/kernels/cuda/launchers/orthog_kernels.cuh"
#include "linalg/internal/gpu/serial/cuda/serial_kernels.hpp"
#elif defined(HAS_HIP)
#include "linalg/internal/gpu/kernels/hip/launchers/orthog_launchers.hpp"
#include "linalg/internal/gpu/serial/hip/serial_kernels.hpp"
#endif

#ifndef SCOPED_NVTX_RANGE
#define SCOPED_NVTX_RANGE() ((void)0)
#endif

/**
 * @page gpu_backend_serial_ops_module serial_ops
 * @brief Build-time alias to the active serial GPU kernel namespace.
 *
 * GPU driver code (`ChASEGPU`, `pChASEGPU`) calls serial algorithm stages such
 * as Lanczos, Rayleigh-Ritz, Cholesky QR, and Householder QR without
 * `#if defined(HAS_CUDA)` at every call site.
 *
 * This header pulls in:
 * - `gpu/serial/{cuda,hip}/serial_kernels.hpp` — Lanczos / ChASE serial stages.
 * - `gpu/kernels/.../orthog_kernels*` — orthogonalization device launchers used
 *   by the serial stages.
 *
 * and defines `chase::linalg::internal::active_gpu_serial` as either
 * `chase::linalg::internal::cuda` or `chase::linalg::internal::hip`
 * depending on the compiled backend.
 *
 * @section serial_ops_usage Typical usage
 *
 * @code
 * #include "linalg/internal/gpu/backend/serial_ops.hpp"
 * chase::linalg::internal::active_gpu_serial::lanczos(...);
 * @endcode
 *
 * For simple device launchers such as `lacpy`, random-vector initialization,
 * residual kernels, and diagonal shifts, prefer
 * `gpu_device::kernels<gpu_backend>::...`; that shared dispatch owns the
 * CUDA/HIP stream and raw-launcher differences.
 */

namespace chase
{
namespace linalg
{
namespace internal
{

#if defined(HAS_CUDA)
/** @brief Serial GPU kernels for the CUDA build (`internal::cuda`). */
namespace active_gpu_serial = cuda;
#elif defined(HAS_HIP)
/** @brief Serial GPU kernels for the HIP build (`internal::hip`). */
namespace active_gpu_serial = hip;
#endif

} // namespace internal
} // namespace linalg
} // namespace chase
