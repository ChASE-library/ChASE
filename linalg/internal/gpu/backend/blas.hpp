// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/types.hpp"

#include <utility>

#if defined(HAS_CUDA)
#include "external/nvidia_gpu/cublaspp/cublaspp.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#if defined(HAS_HIP)
#include "external/amd_gpu/rocblaspp/rocblaspp.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

/**
 * @page gpu_backend_blas_module chase::linalg::internal::gpu_backend::blas
 * @brief Vendor-neutral GPU BLAS dispatch (cuBLAS or rocBLAS).
 *
 * Shared GPU algorithms should call `gpu_backend::blas<Backend>::gemm` (and
 * related wrappers) rather than `cublas*` / `rocblas_*` directly. Typed
 * routines are forwarded to `cublaspp` or `rocblaspp`, which select the
 * correct precision at compile time.
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
 * @brief GPU BLAS facade specialized per backend.
 *
 * @tparam Backend `cuda_backend` or `hip_backend`.
 */
template <typename Backend>
struct blas;

#if defined(HAS_CUDA)
/**
 * @brief cuBLAS specialization (`cublas_v2.h` + `cublaspp`).
 */
template <>
struct blas<cuda_backend>
{
    using handle_type = cublasHandle_t;   ///< cuBLAS library handle.
    using operation = cublasOperation_t;  ///< Transpose/conjugate operation.
    using fill_mode = cublasFillMode_t;   ///< Upper/lower triangular storage.
    using side_mode = cublasSideMode_t;   ///< Left/right multiply side.
    using diag_mode = cublasDiagType_t;   ///< Unit/non-unit triangular diagonal.
    using status_type = cublasStatus_t;   ///< cuBLAS status code.
    using stream_type = cudaStream_t;     ///< CUDA stream bound to the handle.

    /// @name BLAS enum constants
    /// @{

    static constexpr operation op_n = CUBLAS_OP_N;
    static constexpr operation op_t = CUBLAS_OP_T;
    static constexpr operation op_c = CUBLAS_OP_C;
    static constexpr fill_mode fill_lower = CUBLAS_FILL_MODE_LOWER;
    static constexpr fill_mode fill_upper = CUBLAS_FILL_MODE_UPPER;
    static constexpr side_mode side_left = CUBLAS_SIDE_LEFT;
    static constexpr side_mode side_right = CUBLAS_SIDE_RIGHT;
    static constexpr diag_mode diag_non_unit = CUBLAS_DIAG_NON_UNIT;

    /// @}

    /// @name Handle utilities
    /// @{

    /** @brief Terminate on non-success cuBLAS status (via `CHECK_CUBLAS_ERROR`). */
    static void check_status(status_type status)
    {
        CHECK_CUBLAS_ERROR(status);
    }

    /** @brief Query the CUDA stream associated with a cuBLAS handle. */
    static status_type get_stream(handle_type handle, stream_type* stream)
    {
        return cublasGetStream(handle, stream);
    }

    /** @brief Associate a CUDA stream with a cuBLAS handle. */
    static status_type set_stream(handle_type handle, stream_type stream)
    {
        return cublasSetStream(handle, stream);
    }

    /// @}

    /// @name Typed BLAS routines (forwarded to cublaspp)
    /// @{

    /** @brief General matrix multiply (`cublasTgemm`). */
    template <typename... Args>
    static auto gemm(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTgemm(
            std::forward<Args>(args)...);
    }

    /** @brief Triangular solve (`cublasTtrsm`). */
    template <typename... Args>
    static auto trsm(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTtrsm(
            std::forward<Args>(args)...);
    }

    /** @brief Vector scale (`cublasTscal`). */
    template <typename... Args>
    static auto scal(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTscal(
            std::forward<Args>(args)...);
    }

    /** @brief Euclidean norm (`cublasTnrm2`). */
    template <typename... Args>
    static auto nrm2(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTnrm2(
            std::forward<Args>(args)...);
    }

    /** @brief Inner product (`cublasTdot`). */
    template <typename... Args>
    static auto dot(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTdot(
            std::forward<Args>(args)...);
    }

    /** @brief AXPY (`cublasTaxpy`). */
    template <typename... Args>
    static auto axpy(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTaxpy(
            std::forward<Args>(args)...);
    }

    /** @brief Symmetric/Hermitian rank-k update (`cublasTsyherk`). */
    template <typename... Args>
    static auto syherk(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTsyherk(
            std::forward<Args>(args)...);
    }

    /** @brief Swap two vectors (`cublasTswap`). */
    template <typename... Args>
    static auto swap(Args&&... args)
    {
        return chase::linalg::cublaspp::cublasTswap(
            std::forward<Args>(args)...);
    }

    /// @}
};
#endif

#if defined(HAS_HIP)
/**
 * @brief rocBLAS specialization (`rocblas.h` + `rocblaspp`).
 *
 * Exposes the same static API and enum constants as `blas<cuda_backend>`;
 * routines forward to `rocblaspp` instead of `cublaspp`.
 */
template <>
struct blas<hip_backend>
{
    using handle_type = rocblas_handle;
    using operation = rocblas_operation;
    using fill_mode = rocblas_fill;
    using side_mode = rocblas_side;
    using diag_mode = rocblas_diagonal;
    using status_type = rocblas_status;
    using stream_type = hipStream_t;

    static constexpr operation op_n = rocblas_operation_none;
    static constexpr operation op_t = rocblas_operation_transpose;
    static constexpr operation op_c = rocblas_operation_conjugate_transpose;
    static constexpr fill_mode fill_lower = rocblas_fill_lower;
    static constexpr fill_mode fill_upper = rocblas_fill_upper;
    static constexpr side_mode side_left = rocblas_side_left;
    static constexpr side_mode side_right = rocblas_side_right;
    static constexpr diag_mode diag_non_unit = rocblas_diagonal_non_unit;

    static void check_status(status_type status)
    {
        CHECK_ROCBLAS_ERROR(status);
    }

    static status_type get_stream(handle_type handle, stream_type* stream)
    {
        return rocblas_get_stream(handle, stream);
    }

    static status_type set_stream(handle_type handle, stream_type stream)
    {
        return rocblas_set_stream(handle, stream);
    }

    template <typename... Args>
    static auto gemm(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTgemm(
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto trsm(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTtrsm(
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto scal(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTscal(
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto nrm2(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTnrm2(
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto dot(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTdot(
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto axpy(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTaxpy(
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto syherk(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTsyherk(
            std::forward<Args>(args)...);
    }

    template <typename... Args>
    static auto swap(Args&&... args)
    {
        return chase::linalg::rocblaspp::rocblasTswap(
            std::forward<Args>(args)...);
    }
};
#endif

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase
