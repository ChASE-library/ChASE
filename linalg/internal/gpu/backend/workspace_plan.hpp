// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/solver.hpp"

#include <algorithm>
#include <cstddef>

/**
 * @page gpu_backend_workspace_plan_module SolverWorkspacePlan
 * @brief Constructor-time aggregation of GPU solver workspace sizes.
 *
 * `ChASEGPU` and `pChASEGPU` call the `plan_query_*` helpers during
 * initialization, store the per-routine element counts in a
 * `SolverWorkspacePlan`, then allocate **one** reusable device buffer via
 * `GpuExecutionContext::ensure_solver_plan`.
 *
 * @section swp_lifecycle Typical lifecycle
 *
 * @code
 * SolverWorkspacePlan plan;
 * plan_query_qr_workspace<solver<Backend>, T>(plan, solver, ...);
 * plan_query_heevd_workspace<solver<Backend>, T>(plan, solver, ...);
 * plan_query_potrf_workspace<solver<Backend>, T>(plan, solver, ...);
 * plan_include_packed_triangular(plan, block_size);
 * gpu_context.ensure_solver_plan<T>(plan);  // bytes = plan.device_element_count() * sizeof(T)
 * T* d_work = gpu_context.work<T>();
 * @endcode
 *
 * Individual kernels (QR, Rayleigh–Ritz, CholQR, …) reuse `d_work` by passing
 * it into `solver<Backend>::geqrf`, `heevd`, `potrf`, etc. with the stored
 * `lwork` element count. Only one solver routine that needs device workspace
 * runs at a time, so taking the **maximum** size is safe.
 *
 * @section swp_solver_dispatch Template parameter `SolverDispatch`
 *
 * Each `plan_query_*` function is parameterized on `SolverDispatch`, normally
 * `gpu_backend::solver<cuda_backend>` or `gpu_backend::solver<hip_backend>`.
 * That keeps this header vendor-neutral: CUDA uses cuSOLVER `*bufferSize`
 * APIs; HIP uses rocBLAS workspace queries inside `solver<hip_backend>` (see
 * `solver.hpp`).
 *
 * @section swp_heevd_hip Note on `heevd_lwork` (HIP)
 *
 * On HIP, `solver<hip_backend>::heevd_buffer_size` returns an element count
 * that already includes:
 * - the rocSOLVER `E` array prefix, and
 * - the rocBLAS workspace suffix bound by `rocblas_set_workspace`.
 *
 * Callers pass the unified `d_work` buffer into `heevd(..., workspace, lwork,
 * devInfo)`; `solver.hpp` splits the prefix for `E` internally. No extra
 * planning field is required in `SolverWorkspacePlan`.
 *
 * @section swp_pchase_hip pChASEGPU on pure HIP builds
 *
 * When `HAS_HIP && !HAS_CUDA`, distributed Householder QR falls back to
 * ScaLAPACK and `pChASEGPU` **skips** `plan_query_qr_workspace`. Serial
 * `ChASEGPU` still queries QR sizes for the GPU CholQR / Householder path.
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
 * @brief Per-routine workspace sizes gathered at solver construction time.
 *
 * All `*_lwork` fields count elements of type `T` (the solver precision),
 * not bytes. `device_element_count()` returns the single allocation size
 * (in `T` elements) passed to `GpuExecutionContext::ensure_solver_plan`.
 */
struct SolverWorkspacePlan
{
    int geqrf_lwork = 0; ///< QR factorization (`geqrf`).
    int gqr_lwork = 0;   ///< Explicit Q from QR (`gqr` / `orgqr`).
    int heevd_lwork = 0; ///< Hermitian eigen (`heevd` / `syevd`); on HIP includes `E` prefix.
    int geev_device_lwork = 0; ///< Device workspace for `geev` when `XGEEV_EXISTS` (CUDA path).
    int potrf_lwork = 0; ///< Cholesky (`potrf`) for CholQR.
    int packed_triangular_elements = 0; ///< Packed triangular buffer used with CholQR.
    int host_geev_lwork = 0; ///< Host workspace for `geev` when enabled (not device allocation).

    /**
     * @brief Maximum device workspace size (in `T` elements) to allocate once.
     *
     * Computes `max(geqrf, gqr, heevd, geev_device, potrf)` and also accounts
     * for CholQR, which may need `potrf_lwork + packed_triangular_elements`
     * concurrently within one logical workspace reuse window.
     */
    int device_element_count() const noexcept
    {
        const int solver_lwork = std::max(
            {geqrf_lwork, gqr_lwork, heevd_lwork, geev_device_lwork, potrf_lwork});
        const int cholqr_lwork = potrf_lwork + packed_triangular_elements;
        return std::max(solver_lwork, cholqr_lwork);
    }
};

/**
 * @brief Record packed triangular storage required by CholQR for @p block_size.
 *
 * Updates `plan.packed_triangular_elements` with
 * `block_size * (block_size + 1) / 2` (taking the maximum seen so far).
 * Used together with `potrf_lwork` inside `device_element_count()`.
 *
 * @param plan Plan updated in place.
 * @param block_size Local subspace / CholQR block dimension.
 */
inline void plan_include_packed_triangular(SolverWorkspacePlan& plan,
                                           std::size_t block_size)
{
    const int packed =
        static_cast<int>(block_size * (block_size + 1) / 2);
    plan.packed_triangular_elements =
        std::max(plan.packed_triangular_elements, packed);
}

/**
 * @brief Query and accumulate QR workspace (`geqrf` + `gqr`).
 *
 * Uses representative dimensions from the live solver buffers:
 * @p n rows of the distributed matrix, @p block_size columns in the local
 * subspace, @p v_data / @p ldv for the basis matrix, and @p tau_buffer for
 * Householder scalars.
 *
 * @tparam SolverDispatch `solver<cuda_backend>` or `solver<hip_backend>`.
 * @tparam T Solver scalar type.
 * @param plan Plan updated in place (`geqrf_lwork`, `gqr_lwork`).
 * @param solver Backend solver handle (`cusolverDn` or `rocblas`).
 */
template <typename SolverDispatch, typename T, typename SolverHandle>
void plan_query_qr_workspace(SolverWorkspacePlan& plan, SolverHandle solver,
                             std::size_t n, std::size_t block_size, T* v_data,
                             std::size_t ldv, T* tau_buffer)
{
    int lwork_geqrf = 0;
    int lwork_gqr = 0;

    SolverDispatch::check_status(SolverDispatch::geqrf_buffer_size(
        solver, n, block_size, v_data, ldv, &lwork_geqrf));
    SolverDispatch::check_status(SolverDispatch::gqr_buffer_size(
        solver, n, block_size, block_size, v_data, ldv, tau_buffer,
        &lwork_gqr));

    plan.geqrf_lwork = std::max(plan.geqrf_lwork, lwork_geqrf);
    plan.gqr_lwork = std::max(plan.gqr_lwork, lwork_gqr);
}

/**
 * @brief Query and accumulate Hermitian eigen workspace (`heevd`).
 *
 * Uses `eig_mode_vector` and lower triangular storage, matching Rayleigh–Ritz
 * and serial CholQR eigen stages.
 *
 * On HIP, `heevd_buffer_size` already folds in the rocSOLVER `E` prefix; the
 * allocated unified buffer must be passed to the `heevd(..., workspace, lwork,
 * devInfo)` overload in `solver.hpp`.
 *
 * @param plan Plan updated in place (`heevd_lwork`).
 * @param block_size Leading dimension / matrix order for the Rayleigh–Ritz block.
 * @param a_data Device matrix buffer used for the query (same layout as solve).
 * @param lda Leading dimension of @p a_data.
 * @param ritzv_data Device buffer for eigenvalues (query argument to cuSOLVER/rocSOLVER).
 */
template <typename SolverDispatch, typename T, typename SolverHandle>
void plan_query_heevd_workspace(SolverWorkspacePlan& plan, SolverHandle solver,
                                std::size_t block_size, T* a_data,
                                std::size_t lda, chase::Base<T>* ritzv_data)
{
    int lwork_heevd = 0;
    SolverDispatch::check_status(SolverDispatch::heevd_buffer_size(
        solver, SolverDispatch::eig_mode_vector,
        SolverDispatch::fill_lower, block_size, a_data, lda, ritzv_data,
        &lwork_heevd));
    plan.heevd_lwork = std::max(plan.heevd_lwork, lwork_heevd);
}

/**
 * @brief Query and accumulate generalized eigen workspace (`geev`).
 *
 * Only used when `XGEEV_EXISTS` is defined (CUDA ≥ 12.6 pseudo-Hermitian path).
 * HIP production builds use `heevd` instead and typically do not call this.
 *
 * Stores both device (`geev_device_lwork`) and host (`host_geev_lwork`) sizes.
 * Host workspace is allocated separately by `ChASEGPU` / `pChASEGPU` (`h_work_`).
 *
 * @param params Solver params object (`cusolverDnParams_t`; unused placeholder on HIP).
 */
template <typename SolverDispatch, typename T, typename SolverHandle,
          typename ParamsHandle>
void plan_query_geev_workspace(SolverWorkspacePlan& plan, SolverHandle solver,
                               ParamsHandle params, std::size_t block_size,
                               T* a_data, std::size_t lda, T* vr_data,
                               std::size_t ldvr)
{
    std::size_t temp_ldwork = 0;
    std::size_t temp_lhwork = 0;

    SolverDispatch::check_status(SolverDispatch::geev_buffer_size(
        solver, params, SolverDispatch::eig_mode_no_vector,
        SolverDispatch::eig_mode_vector, block_size, a_data, lda, vr_data,
        nullptr, 1, vr_data, ldvr, &temp_ldwork, &temp_lhwork));

    plan.geev_device_lwork =
        std::max(plan.geev_device_lwork, static_cast<int>(temp_ldwork));
    plan.host_geev_lwork =
        std::max(plan.host_geev_lwork, static_cast<int>(temp_lhwork));
}

/**
 * @brief Query and accumulate Cholesky workspace (`potrf`) for CholQR.
 *
 * Uses upper triangular storage (`fill_upper`), consistent with CholQR stages
 * in `gpu/serial/impl` and `gpu/dist_mpi/impl`.
 *
 * @param plan Plan updated in place (`potrf_lwork`).
 * @param block_size Matrix order for the CholQR diagonal block.
 */
template <typename SolverDispatch, typename T, typename SolverHandle>
void plan_query_potrf_workspace(SolverWorkspacePlan& plan, SolverHandle solver,
                                std::size_t block_size, T* a_data,
                                std::size_t lda)
{
    int lwork_potrf = 0;
    SolverDispatch::check_status(SolverDispatch::potrf_buffer_size(
        solver, SolverDispatch::fill_upper, block_size, a_data, lda,
        &lwork_potrf));
    plan.potrf_lwork = std::max(plan.potrf_lwork, lwork_potrf);
}

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase
