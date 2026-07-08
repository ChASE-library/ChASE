// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "linalg/internal/gpu/backend/types.hpp"

#include <cstddef>
#include <limits>
#include <utility>

#if defined(HAS_CUDA)
#include "external/nvidia_gpu/cusolverpp/cusolverpp.hpp"
#include <cublas_v2.h>
#include <cusolverDn.h>
#endif

#if defined(HAS_HIP)
#include "external/amd_gpu/rocsolverpp/rocsolverpp.hpp"
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>
#endif

/**
 * @page gpu_backend_solver_module chase::linalg::internal::gpu_backend::solver
 * @brief Vendor-neutral GPU dense-solver dispatch (cuSOLVER or rocSOLVER).
 *
 * Unlike `blas.hpp`, which wraps Level-2/3 BLAS (GEMM, dot, axpy, …), this
 * header wraps **matrix factorizations and eigen-solvers** used in ChASE:
 * QR (`geqrf` / `gqr`), Cholesky (`potrf`), and Hermitian eigen-decomposition
 * (`heevd`), plus optional generalized eigen (`geev`) on CUDA when
 * `XGEEV_EXISTS` is defined.
 *
 * @section solver_vs_blas Solver vs BLAS
 *
 * | | `blas.hpp` | `solver.hpp` |
 * |---|------------|--------------|
 * | Library | cuBLAS / rocBLAS | cuSOLVER / rocSOLVER |
 * | Typical ops | `gemm`, `dot`, `syherk` | `geqrf`, `potrf`, `heevd` |
 * | CUDA/HIP symmetry | Mostly thin `*pp` forwards | **Asymmetric on HIP** (see below) |
 * | Workspace | Not applicable | Central to solver design |
 *
 * @section solver_cuda_vs_hip CUDA vs HIP workspace model (important)
 *
 * **CUDA** follows the cuSOLVER model:
 * - Separate `cusolverDnHandle_t` (see `traits<cuda_backend>::solver_handle`).
 * - Each routine has a real `*_buffer_size` API (`geqrf_buffer_size`, …).
 * - Calls pass caller-owned `workspace` + `lwork` + `devInfo`.
 * - Implementation is mostly a forward to `cusolverpp` (cuSOLVER `*bufferSize`
 *   wrappers).
 *
 * **HIP** does **not** expose cuSOLVER-style external workspace APIs:
 * - `traits<hip_backend>::solver_handle` is `rocblas_handle` (rocSOLVER runs on
 *   the rocBLAS handle model).
 * - `rocsolverpp` provides **typed routine dispatch only** (`rocsolverTgeqrf`,
 *   …); it does not fake `*bufferSize` stubs.
 * - Workspace sizing uses `rocblas_start_device_memory_size_query` /
 *   `rocblas_stop_device_memory_size_query` inside `query_workspace_bytes`.
 * - Execution binds memory with `rocblas_set_workspace` via `set_workspace`.
 * - Public `*_buffer_size` names match CUDA so `SolverWorkspacePlan` and
 *   `GpuExecutionContext` stay backend-neutral.
 * - `heevd` additionally splits the caller buffer: leading bytes hold the
 *   real eigenvalue workspace `E` required by rocSOLVER; the remainder is
 *   passed to `rocblas_set_workspace`.
 * - Workspace queries use a **scratch** rocBLAS handle so query side effects
 *   do not corrupt the live compute handle used for BLAS.
 *
 * @section solver_traits Role of `traits<Backend>`
 *
 * `traits` is declared in `types.hpp` and **defined here** because solver
 * headers pull in vendor types. It is the single place that maps a backend tag
 * to the handle and enum types shared by:
 *
 * - `solver<Backend>` (this file),
 * - `GpuExecutionContext<Backend>` (`context.hpp`),
 * - `handle_utils<Backend>` (`handle_utils.hpp`).
 *
 * Shared algorithm code should depend on `traits<Backend>::solver_handle`,
 * `eig_mode`, `fill_mode`, and `info_type` rather than spelling
 * `cusolverDnHandle_t` or `rocblas_handle` directly. On HIP, note that
 * `blas_handle` and `solver_handle` are the same C++ type but ChASE still
 * creates **two** handles in `GpuExecutionContext` to avoid stream/workspace
 * ordering bugs between BLAS and solver traffic.
 */

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_backend
{

#if defined(HAS_CUDA)
/**
 * @brief CUDA backend type aliases for BLAS, solver, and status codes.
 *
 * Consumed by `solver<cuda_backend>`, `blas<cuda_backend>` (via matching enums),
 * `GpuExecutionContext`, and `handle_utils`.
 */
template <>
struct traits<cuda_backend>
{
    using blas_handle = cublasHandle_t;       ///< cuBLAS handle.
    using solver_handle = cusolverDnHandle_t; ///< cuSOLVER dense handle (distinct from BLAS).
    using params_type = cusolverDnParams_t;   ///< cuSOLVER params (e.g. `geev`).
    using eig_mode = cusolverEigMode_t;       ///< Eigenvector job mode for `heevd` / `geev`.
    using fill_mode = cublasFillMode_t;       ///< Upper/lower storage for `potrf` / `heevd`.
    using status_type = cusolverStatus_t;     ///< cuSOLVER status.
    using info_type = int;                    ///< Device `devInfo` integer type.
};
#endif

#if defined(HAS_HIP)
/**
 * @brief HIP backend type aliases for BLAS/solver and status codes.
 *
 * rocSOLVER reuses the rocBLAS handle type. `params_type` is `nullptr_t`
 * because ROCm has no cuSOLVER-style params object; `create_params` is a no-op.
 */
template <>
struct traits<hip_backend>
{
    using blas_handle = rocblas_handle;   ///< rocBLAS handle.
    using solver_handle = rocblas_handle; ///< rocSOLVER uses rocBLAS handles.
    using params_type = std::nullptr_t;    ///< Placeholder; no ROCm params object.
    using eig_mode = rocblas_evect;        ///< Eigenvector job mode (`rocblas_evect_*`).
    using fill_mode = rocblas_fill;        ///< Upper/lower storage.
    using status_type = rocblas_status;  ///< rocBLAS/rocSOLVER status.
    using info_type = rocblas_int;       ///< Device `devInfo` integer type.
};
#endif

/**
 * @brief GPU dense-solver facade specialized per backend.
 *
 * @tparam Backend `cuda_backend` or `hip_backend`.
 */
template <typename Backend>
struct solver;

#if defined(HAS_CUDA)
/**
 * @brief cuSOLVER specialization: thin `cusolverpp` forwards + native bufferSize.
 *
 * Workspace is caller-owned: query with `*_buffer_size`, allocate once via
 * `GpuExecutionContext`, pass `workspace` and `lwork` into each call.
 */
template <>
struct solver<cuda_backend>
{
    using handle_type = typename traits<cuda_backend>::solver_handle;
    using params_type = typename traits<cuda_backend>::params_type;
    using eig_mode = typename traits<cuda_backend>::eig_mode;
    using fill_mode = typename traits<cuda_backend>::fill_mode;
    using status_type = typename traits<cuda_backend>::status_type;

    /// @name Solver enum constants
    /// @{

    static constexpr eig_mode eig_mode_no_vector = CUSOLVER_EIG_MODE_NOVECTOR;
    static constexpr eig_mode eig_mode_vector = CUSOLVER_EIG_MODE_VECTOR;
    static constexpr fill_mode fill_lower = CUBLAS_FILL_MODE_LOWER;
    static constexpr fill_mode fill_upper = CUBLAS_FILL_MODE_UPPER;

    /// @}

    /// @name Handle utilities
    /// @{

    /** @brief Terminate on non-success cuSOLVER status. */
    static void check_status(status_type status)
    {
        CHECK_CUSOLVER_ERROR(status);
    }

    /** @brief Create a cuSOLVER params object (used by `geev` when enabled). */
    static status_type create_params(params_type* params)
    {
        return cusolverDnCreateParams(params);
    }

    /** @brief Query the CUDA stream bound to a cuSOLVER handle. */
    static status_type get_stream(handle_type handle, cudaStream_t* stream)
    {
        return cusolverDnGetStream(handle, stream);
    }

    /** @brief Bind a CUDA stream to a cuSOLVER handle. */
    static status_type set_stream(handle_type handle, cudaStream_t stream)
    {
        return cusolverDnSetStream(handle, stream);
    }

    /// @}

    /// @name Typed solver routines (cusolverpp forwards)
    /// @{

#if defined(XGEEV_EXISTS)
    /** @brief Query workspace for generalized eigen (`cusolverDnTgeev`). CUDA ≥ 12.6. */
    template <typename... Args>
    static status_type geev_buffer_size(Args&&... args)
    {
        return chase::linalg::cusolverpp::cusolverDnTgeev_bufferSize(
            std::forward<Args>(args)...);
    }

    /** @brief Generalized eigen solve (`cusolverDnTgeev`). */
    template <typename... Args>
    static status_type geev(Args&&... args)
    {
        return chase::linalg::cusolverpp::cusolverDnTgeev(
            std::forward<Args>(args)...);
    }
#endif

    /** @brief Query workspace elements for QR factorization (`geqrf`). */
    template <typename T>
    static cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t handle,
                                              std::size_t m, std::size_t n,
                                              T* A, std::size_t lda,
                                              int* lwork)
    {
        return chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
            handle, m, n, A, lda, lwork);
    }

    /** @brief QR factorization with caller-owned workspace. */
    template <typename T>
    static cusolverStatus_t geqrf(cusolverDnHandle_t handle, std::size_t m,
                                  std::size_t n, T* A, std::size_t lda, T* tau,
                                  T* workspace, std::size_t lwork,
                                  int* devInfo)
    {
        return chase::linalg::cusolverpp::cusolverDnTgeqrf(
            handle, m, n, A, lda, tau, workspace, lwork, devInfo);
    }

    /** @brief Query workspace elements for Q generation from QR (`gqr`). */
    template <typename T>
    static cusolverStatus_t gqr_buffer_size(cusolverDnHandle_t handle,
                                            std::size_t m, std::size_t n,
                                            std::size_t k, T* A,
                                            std::size_t lda, T* tau,
                                            int* lwork)
    {
        return chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
            handle, m, n, k, A, lda, tau, lwork);
    }

    /** @brief Form explicit Q from QR with caller-owned workspace. */
    template <typename T>
    static cusolverStatus_t gqr(cusolverDnHandle_t handle, std::size_t m,
                                std::size_t n, std::size_t k, T* A,
                                std::size_t lda, T* tau, T* workspace,
                                std::size_t lwork, int* devInfo)
    {
        return chase::linalg::cusolverpp::cusolverDnTgqr(
            handle, m, n, k, A, lda, tau, workspace, lwork, devInfo);
    }

    /** @brief Query workspace elements for Hermitian eigen-decomposition. */
    template <typename T>
    static cusolverStatus_t heevd_buffer_size(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo,
                                              std::size_t n, T* A,
                                              std::size_t lda,
                                              chase::Base<T>* W, int* lwork)
    {
        return chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
            handle, jobz, uplo, n, A, lda, W, lwork);
    }

    /** @brief Hermitian eigen-decomposition (`syevd` / `heevd`). */
    template <typename T>
    static cusolverStatus_t heevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo, std::size_t n, T* A,
                                  std::size_t lda, chase::Base<T>* W,
                                  T* workspace, std::size_t lwork,
                                  int* devInfo)
    {
        return chase::linalg::cusolverpp::cusolverDnTheevd(
            handle, jobz, uplo, n, A, lda, W, workspace, lwork, devInfo);
    }

    /** @brief Query workspace elements for Cholesky factorization. */
    template <typename T>
    static cusolverStatus_t potrf_buffer_size(
        cusolverDnHandle_t handle, cublasFillMode_t uplo, std::size_t n, T* A,
        std::size_t lda, int* lwork)
    {
        return chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
            handle, uplo, n, A, lda, lwork);
    }

    /** @brief Cholesky factorization with caller-owned workspace. */
    template <typename T>
    static cusolverStatus_t potrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, std::size_t n, T* A,
                                  std::size_t lda, T* workspace,
                                  std::size_t lwork, int* devInfo)
    {
        return chase::linalg::cusolverpp::cusolverDnTpotrf(
            handle, uplo, n, A, lda, workspace, lwork, devInfo);
    }

    /// @}
};
#endif

#if defined(HAS_HIP)
/**
 * @brief rocSOLVER specialization: rocBLAS workspace model + `rocsolverpp` dispatch.
 *
 * Public method names mirror `solver<cuda_backend>` so `SolverWorkspacePlan` and
 * shared `impl/` code stay vendor-neutral. Internally, workspace is sized via
 * rocBLAS device-memory queries and bound with `rocblas_set_workspace`.
 */
template <>
struct solver<hip_backend>
{
    using handle_type = typename traits<hip_backend>::solver_handle;
    using params_type = typename traits<hip_backend>::params_type;
    using eig_mode = typename traits<hip_backend>::eig_mode;
    using fill_mode = typename traits<hip_backend>::fill_mode;
    using status_type = typename traits<hip_backend>::status_type;

    static constexpr eig_mode eig_mode_no_vector = rocblas_evect_none;
    static constexpr eig_mode eig_mode_vector = rocblas_evect_original;
    static constexpr fill_mode fill_lower = rocblas_fill_lower;
    static constexpr fill_mode fill_upper = rocblas_fill_upper;

    static void check_status(status_type status)
    {
        CHECK_ROCSOLVER_ERROR(status);
    }

    /** @brief No-op on HIP; `params_type` is `nullptr_t`. */
    static status_type create_params(params_type* params)
    {
        *params = nullptr;
        return rocblas_status_success;
    }

    static status_type get_stream(handle_type handle, hipStream_t* stream)
    {
        return rocblas_get_stream(handle, stream);
    }

    static status_type set_stream(handle_type handle, hipStream_t stream)
    {
        return rocblas_set_stream(handle, stream);
    }

    /// @name HIP-only workspace helpers (no CUDA equivalent in this file)
    /// @{

    /** @brief Round @p bytes up to @p alignment. Used for `heevd` buffer layout. */
    static std::size_t align_bytes(std::size_t bytes, std::size_t alignment)
    {
        return ((bytes + alignment - 1) / alignment) * alignment;
    }

    /**
     * @brief Whether a rocBLAS status is acceptable during workspace query.
     *
     * ROCm 6.x may return status `9` while the byte size is still reported by
     * `rocblas_stop_device_memory_size_query`.
     */
    static bool is_workspace_query_status_ok(rocblas_status status)
    {
        return status == rocblas_status_success ||
               status == static_cast<rocblas_status>(9);
    }

    /**
     * @brief Measure device workspace bytes required by a rocSOLVER call.
     *
     * Runs @p query_call on a **temporary** rocBLAS handle so the live solver
     * handle used for computation is not left in query mode.
     */
    template <typename QueryCall>
    static rocblas_status query_workspace_bytes(rocblas_handle handle,
                                                std::size_t* bytes,
                                                QueryCall query_call)
    {
        rocblas_handle query_handle = nullptr;
        rocblas_status create_status = rocblas_create_handle(&query_handle);
        if (create_status != rocblas_status_success)
        {
            return create_status;
        }

        hipStream_t stream = nullptr;
        const rocblas_status stream_status = rocblas_get_stream(handle, &stream);
        if (stream_status == rocblas_status_success)
        {
            rocblas_set_stream(query_handle, stream);
        }

        rocblas_status status =
            rocblas_start_device_memory_size_query(query_handle);
        if (status != rocblas_status_success)
        {
            rocblas_destroy_handle(query_handle);
            return status;
        }

        status = query_call(query_handle);
        rocblas_status stop_status =
            rocblas_stop_device_memory_size_query(query_handle, bytes);
        rocblas_destroy_handle(query_handle);
        if (!is_workspace_query_status_ok(status))
        {
            return status;
        }
        return stop_status;
    }

    /** @brief Bind caller-owned device memory as rocBLAS/rocSOLVER workspace. */
    static rocblas_status set_workspace(rocblas_handle handle, void* workspace,
                                        std::size_t bytes)
    {
        if (workspace == nullptr || bytes == 0)
        {
            return rocblas_status_success;
        }
        return rocblas_set_workspace(handle, workspace, bytes);
    }

    /**
     * @brief Convert a byte workspace size to element count (`lwork`) for type @p T.
     * @param extra_bytes Additional bytes included in the element count (e.g. `heevd` `E`).
     */
    template <typename T>
    static rocblas_status workspace_bytes_to_lwork(rocblas_status status,
                                                   std::size_t bytes,
                                                   int* lwork,
                                                   std::size_t extra_bytes = 0)
    {
        if (status != rocblas_status_success)
        {
            return status;
        }

        const std::size_t elements =
            (bytes + extra_bytes + sizeof(T) - 1) / sizeof(T);
        if (elements >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            return rocblas_status_invalid_size;
        }

        *lwork = static_cast<int>(elements);
        return rocblas_status_success;
    }

    /// @}

    /// @name Typed solver routines (rocsolverpp + workspace binding)
    /// @{

    /** @brief Byte-size query for `geqrf` (invokes `query_workspace_bytes`). */
    template <typename T>
    static rocblas_status geqrf_workspace_bytes(rocblas_handle handle,
                                                std::size_t m, std::size_t n,
                                                T* A, std::size_t lda, T* tau,
                                                std::size_t* bytes)
    {
        return query_workspace_bytes(handle, bytes, [&](rocblas_handle qh) {
            return chase::linalg::rocsolverpp::rocsolverTgeqrf<T>(
                qh, static_cast<rocblas_int>(m),
                static_cast<rocblas_int>(n), A, static_cast<rocblas_int>(lda),
                tau);
        });
    }

    /** @brief QR factorization without external workspace (library-internal alloc). */
    template <typename T>
    static rocblas_status geqrf(rocblas_handle handle, std::size_t m,
                                std::size_t n, T* A, std::size_t lda, T* tau)
    {
        return chase::linalg::rocsolverpp::rocsolverTgeqrf<T>(
            handle, static_cast<rocblas_int>(m), static_cast<rocblas_int>(n), A,
            static_cast<rocblas_int>(lda), tau);
    }

    /** @brief QR factorization using caller buffer bound via `set_workspace`. */
    template <typename T>
    static rocblas_status geqrf(rocblas_handle handle, std::size_t m,
                                std::size_t n, T* A, std::size_t lda, T* tau,
                                T* workspace, std::size_t lwork)
    {
        rocblas_status status =
            set_workspace(handle, workspace, lwork * sizeof(T));
        if (status != rocblas_status_success)
        {
            return status;
        }
        return geqrf(handle, m, n, A, lda, tau);
    }

    /** @brief `gqr` with workspace pre-bound (calls workspace-less overload). */
    template <typename T>
    static rocblas_status gqr(rocblas_handle handle, std::size_t m,
                              std::size_t n, std::size_t k, T* A,
                              std::size_t lda, T* tau, T* workspace,
                              std::size_t lwork)
    {
        rocblas_status status =
            set_workspace(handle, workspace, lwork * sizeof(T));
        if (status != rocblas_status_success)
        {
            return status;
        }
        return gqr(handle, m, n, k, A, lda, tau);
    }

    /** @brief CUDA-compatible name: element workspace size for `geqrf`. */
    template <typename T>
    static rocblas_status geqrf_buffer_size(rocblas_handle handle,
                                            std::size_t m, std::size_t n,
                                            T* A, std::size_t lda, int* lwork)
    {
        std::size_t bytes = 0;
        return workspace_bytes_to_lwork<T>(
            geqrf_workspace_bytes<T>(handle, m, n, A, lda, nullptr, &bytes),
            bytes, lwork);
    }

    template <typename T>
    static rocblas_status gqr_workspace_bytes(rocblas_handle handle,
                                              std::size_t m, std::size_t n,
                                              std::size_t k, T* A,
                                              std::size_t lda, T* tau,
                                              std::size_t* bytes)
    {
        return query_workspace_bytes(handle, bytes, [&](rocblas_handle qh) {
            return chase::linalg::rocsolverpp::rocsolverTorgqr<T>(
                qh, static_cast<rocblas_int>(m),
                static_cast<rocblas_int>(n), static_cast<rocblas_int>(k),
                A, static_cast<rocblas_int>(lda), tau);
        });
    }

    template <typename T>
    static rocblas_status gqr(rocblas_handle handle, std::size_t m,
                              std::size_t n, std::size_t k, T* A,
                              std::size_t lda, T* tau)
    {
        return chase::linalg::rocsolverpp::rocsolverTorgqr<T>(
            handle, static_cast<rocblas_int>(m), static_cast<rocblas_int>(n),
            static_cast<rocblas_int>(k), A, static_cast<rocblas_int>(lda), tau);
    }

    /** @brief CUDA-compatible name: element workspace size for `gqr`. */
    template <typename T>
    static rocblas_status gqr_buffer_size(rocblas_handle handle, std::size_t m,
                                          std::size_t n, std::size_t k, T* A,
                                          std::size_t lda, T* tau, int* lwork)
    {
        std::size_t bytes = 0;
        return workspace_bytes_to_lwork<T>(
            gqr_workspace_bytes<T>(handle, m, n, k, A, lda, tau, &bytes),
            bytes, lwork);
    }

    template <typename T>
    static rocblas_status heevd_workspace_bytes(rocblas_handle handle,
                                                rocblas_evect jobz,
                                                rocblas_fill uplo,
                                                std::size_t n, T* A,
                                                std::size_t lda,
                                                chase::Base<T>* W,
                                                std::size_t* bytes)
    {
        return query_workspace_bytes(handle, bytes, [&](rocblas_handle qh) {
            auto* info = reinterpret_cast<rocblas_int*>(W);
            return chase::linalg::rocsolverpp::rocsolverTheevd<T>(
                qh, jobz, uplo, static_cast<rocblas_int>(n), A,
                static_cast<rocblas_int>(lda), W, W, info);
        });
    }

    /**
     * @brief Hermitian eigen-decomposition; @p E is the rocSOLVER eigenvalue workspace.
     *
     * Unlike CUDA, rocSOLVER requires a separate `E` buffer in addition to any
     * rocBLAS workspace bound via `set_workspace`.
     */
    template <typename T>
    static rocblas_status heevd(rocblas_handle handle, rocblas_evect jobz,
                                rocblas_fill uplo, std::size_t n, T* A,
                                std::size_t lda, chase::Base<T>* W,
                                chase::Base<T>* E, rocblas_int* devInfo)
    {
        return chase::linalg::rocsolverpp::rocsolverTheevd<T>(
            handle, jobz, uplo, static_cast<rocblas_int>(n), A,
            static_cast<rocblas_int>(lda), W, E, devInfo);
    }

    /**
     * @brief CUDA-compatible `heevd_buffer_size`; includes space for `E` via @p extra_bytes.
     */
    template <typename T>
    static rocblas_status heevd_buffer_size(rocblas_handle handle,
                                            rocblas_evect jobz,
                                            rocblas_fill uplo, std::size_t n,
                                            T* A, std::size_t lda,
                                            chase::Base<T>* W, int* lwork)
    {
        std::size_t bytes = 0;
        const std::size_t e_bytes =
            align_bytes(n * sizeof(chase::Base<T>), alignof(T));
        return workspace_bytes_to_lwork<T>(
            heevd_workspace_bytes<T>(handle, jobz, uplo, n, A, lda, W, &bytes),
            bytes, lwork, e_bytes);
    }

    /**
     * @brief `heevd` with unified caller buffer: prefix holds `E`, suffix is rocBLAS workspace.
     */
    template <typename T>
    static rocblas_status heevd(rocblas_handle handle, rocblas_evect jobz,
                                rocblas_fill uplo, std::size_t n, T* A,
                                std::size_t lda, chase::Base<T>* W,
                                T* workspace, std::size_t lwork,
                                rocblas_int* devInfo)
    {
        const std::size_t e_bytes =
            align_bytes(n * sizeof(chase::Base<T>), alignof(T));
        const std::size_t total_bytes = lwork * sizeof(T);
        if (total_bytes < e_bytes)
        {
            return rocblas_status_invalid_size;
        }

        auto* E = reinterpret_cast<chase::Base<T>*>(workspace);
        void* rocblas_workspace =
            reinterpret_cast<void*>(reinterpret_cast<char*>(workspace) +
                                    e_bytes);
        const std::size_t rocblas_workspace_bytes = total_bytes - e_bytes;

        rocblas_status status =
            set_workspace(handle, rocblas_workspace, rocblas_workspace_bytes);
        if (status != rocblas_status_success)
        {
            return status;
        }

        return heevd(handle, jobz, uplo, n, A, lda, W, E, devInfo);
    }

    template <typename T>
    static rocblas_status potrf_workspace_bytes(rocblas_handle handle,
                                                rocblas_fill uplo,
                                                std::size_t n, T* A,
                                                std::size_t lda,
                                                std::size_t* bytes)
    {
        return query_workspace_bytes(handle, bytes, [&](rocblas_handle qh) {
            auto* info = reinterpret_cast<rocblas_int*>(A);
            return chase::linalg::rocsolverpp::rocsolverTpotrf<T>(
                qh, uplo, static_cast<rocblas_int>(n), A,
                static_cast<rocblas_int>(lda), info);
        });
    }

    template <typename T>
    static rocblas_status potrf(rocblas_handle handle, rocblas_fill uplo,
                                std::size_t n, T* A, std::size_t lda,
                                T* workspace, std::size_t lwork,
                                rocblas_int* devInfo)
    {
        rocblas_status status =
            set_workspace(handle, workspace, lwork * sizeof(T));
        if (status != rocblas_status_success)
        {
            return status;
        }
        return potrf(handle, uplo, n, A, lda, devInfo);
    }

    /** @brief CUDA-compatible name: element workspace size for `potrf`. */
    template <typename T>
    static rocblas_status potrf_buffer_size(rocblas_handle handle,
                                            rocblas_fill uplo, std::size_t n,
                                            T* A, std::size_t lda, int* lwork)
    {
        std::size_t bytes = 0;
        return workspace_bytes_to_lwork<T>(
            potrf_workspace_bytes<T>(handle, uplo, n, A, lda, &bytes), bytes,
            lwork);
    }

    template <typename T>
    static rocblas_status potrf(rocblas_handle handle, rocblas_fill uplo,
                                std::size_t n, T* A, std::size_t lda,
                                rocblas_int* devInfo)
    {
        return chase::linalg::rocsolverpp::rocsolverTpotrf<T>(
            handle, uplo, static_cast<rocblas_int>(n), A,
            static_cast<rocblas_int>(lda), devInfo);
    }

    /// @}
};
#endif

} // namespace gpu_backend
} // namespace internal
} // namespace linalg
} // namespace chase
