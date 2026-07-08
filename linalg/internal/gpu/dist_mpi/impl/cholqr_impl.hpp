// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/cpu/blaspp/blaspp.hpp"
#include "external/cpu/lapackpp/lapackpp.hpp"
#include "external/scalapack/scalapackpp/scalapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"
#include "algorithm/logger.hpp"
#include "mpi.h"
#include <cmath>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

using namespace chase::linalg::blaspp;
using namespace chase::linalg::lapackpp;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{
namespace detail
{

template <typename T, typename Backend>
typename gpu_backend::handle_utils<Backend>::operation cholqr_transpose_op()
{
    using util = gpu_backend::handle_utils<Backend>;
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        return util::op_c();
    }
    else
    {
        return util::op_t();
    }
}

template <typename Backend, typename T>
void query_potrf_buffer_size(
    typename gpu_backend::handle_utils<Backend>::solver_handle handle,
    std::size_t n, T* A, std::size_t lda, int* lwork)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    util::check_solver(solver::template potrf_buffer_size<T>(
        handle, util::fill_upper(), n, A, lda, lwork));
}

template <typename Backend, typename T>
void ensure_potrf_workspace(
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    std::size_t n, T* A, T*& workspace, std::size_t& lwork, bool& owns_workspace)
{
    if (workspace != nullptr && lwork > 0)
    {
        return;
    }

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        using util = gpu_backend::handle_utils<Backend>;
        int lwork_int = 0;
        util::check_solver(
            gpu_backend::solver<Backend>::template potrf_buffer_size<T>(
                solver_handle, util::fill_upper(), n, A, n, &lwork_int));
        workspace = gpu_backend::device_malloc_array<Backend, T>(
            static_cast<std::size_t>(lwork_int));
        lwork = static_cast<std::size_t>(lwork_int);
        owns_workspace = true;
    }
}

/** CholQR POTRF: CUDA always uses caller workspace. HIP binds d_work_ only when
 *  BLAS and solver use separate rocblas handles; a single shared handle must not
 *  call rocblas_set_workspace before subsequent BLAS (e.g. trsm). */
template <typename Backend, typename T>
void run_potrf(typename gpu_backend::handle_utils<Backend>::solver_handle handle,
               std::size_t n, T* A, std::size_t lda,
               typename gpu_backend::handle_utils<Backend>::info_type* devInfo,
               T* workspace, std::size_t lwork, bool hip_bind_caller_workspace)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        util::check_solver(solver::template potrf<T>(
            handle, util::fill_upper(), n, A, lda, workspace, lwork, devInfo));
    }
    else if (workspace != nullptr && lwork > 0 && hip_bind_caller_workspace)
    {
        util::check_solver(solver::template potrf<T>(
            handle, util::fill_upper(), n, A, lda, workspace, lwork, devInfo));
    }
    else
    {
        util::check_solver(
            solver::template potrf<T>(handle, util::fill_upper(), n, A, lda, devInfo));
    }
}

template <typename Backend, typename T>
void potrf_with_optional_workspace(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    std::size_t n, T* A, std::size_t lda,
    typename gpu_backend::handle_utils<Backend>::info_type* devInfo, T* workspace,
    int lwork)
{
    T* potrf_workspace = workspace;
    std::size_t potrf_lwork = static_cast<std::size_t>(lwork);
    bool owns_potrf_workspace = false;
    ensure_potrf_workspace<Backend, T>(
        solver_handle, n, A, potrf_workspace, potrf_lwork, owns_potrf_workspace);

    bool hip_bind_caller_workspace = true;
    if constexpr (std::is_same_v<Backend, gpu_backend::hip_backend>)
    {
        hip_bind_caller_workspace = (blas_handle != solver_handle);
    }

    run_potrf<Backend, T>(solver_handle, n, A, lda, devInfo, potrf_workspace,
                          potrf_lwork, hip_bind_caller_workspace);

    if (owns_potrf_workspace)
    {
        gpu_backend::device_free<Backend>(potrf_workspace);
    }
}

} // namespace detail


/**
 * @brief Performs a distributed Cholesky QR decomposition on a matrix V.
 *
 * This function decomposes the matrix V into a QR form using Cholesky
 * factorization. It is designed to work on a multi-GPU setup where MPI is used
 * for communication between GPUs.
 *
 * @param blas_handle The cuBLAS handle to perform linear algebra operations.
 * @param solver_handle The cuSolver handle to compute Cholesky factorization.
 * @param m The number of rows in the matrix V.
 * @param n The number of columns in the matrix V.
 * @param V The input matrix to decompose, stored in column-major format.
 * @param ldv The leading dimension of the matrix V.
 * @param comm The MPI communicator used for multi-GPU collective communication.
 * @param workspace Optional workspace buffer for intermediate calculations.
 * @param lwork The size of the workspace buffer.
 * @param A Optional matrix for storing the result of the factorization.
 *
 * @return int Status code: 0 for success, non-zero for failure.
 */
template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename T>
int chol_qr1_impl(BlasHandle blas_handle, SolverHandle solver_handle, std::size_t m,
                      std::size_t n, T* V, int ldv,
                      MPI_Comm comm, T* workspace,
                      int lwork, T* A)
{
    T one = T(1.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;

    int info = 1;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    std::unique_ptr<T, typename util::device_deleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        A = gpu_backend::device_malloc_array<Backend, T>(n * n);
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        detail::query_potrf_buffer_size<Backend, T>(solver_handle, n, A, n,
                                                    &lwork);
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }

        workspace = gpu_backend::device_malloc_array<Backend, T>(lwork);
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    const auto transa =
        chase::linalg::internal::gpu_mpi::detail::
            cholqr_transpose_op<T, Backend>();

    util::check_blas(blas_dispatch::syherk(
        blas_handle, util::fill_upper(), transa, n, m, &One, V, ldv,
        &Zero, A, n));

    kernel_ops::extract_upper_triangular(A, n, workspace, n);
    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(workspace, upperTriangularSize, comm);
    kernel_ops::unpack_upper_triangular(workspace, n, A, n);
    gpu_backend::device_synchronize<Backend>();

    int* devInfo;
    devInfo = gpu_backend::device_malloc_array<Backend, int>(1);

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, n, A, n, devInfo, workspace, lwork);
    gpu_backend::memcpy_d2h<Backend>(&info, devInfo, 1);

    if (info != 0)
    {
        gpu_backend::device_free<Backend>(devInfo);
        return info;
    }
    else
    {
        util::check_blas(blas_dispatch::trsm(
            blas_handle, util::side_right(),
            util::fill_upper(), util::op_n(),
            util::diag_non_unit(), m, n, &one, A, n, V, ldv));
#ifdef CHASE_OUTPUT
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if (grank == 0)
        {
            chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
                "choldegree: 1\n", 0);
        }
#endif
        gpu_backend::device_free<Backend>(devInfo);
        return info;
    }
}

/**
 * @brief Variant of cholQR1 for InputMultiVectorType.
 *
 * This variant works with InputMultiVectorType and performs the Cholesky QR
 * decomposition in the same manner as cholQR1, but with different input data
 * type handling. The use of CUDA-Aware MPI ensures that the computation is
 * parallelized across multiple GPUs.
 *
 * @param blas_handle The cuBLAS handle for linear algebra operations.
 * @param solver_handle The cuSolver handle for Cholesky factorization.
 * @param V The input matrix to decompose.
 * @param workspace Optional workspace buffer.
 * @param lwork The size of the workspace buffer.
 * @param A Optional matrix for storing the factorization result.
 *
 * @return int Status code: 0 for success, non-zero for failure.
 */
template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename InputMultiVectorType>
int chol_qr1_impl(BlasHandle blas_handle, SolverHandle solver_handle,
                      InputMultiVectorType& V,
                      typename InputMultiVectorType::value_type* workspace,
                      int lwork, typename InputMultiVectorType::value_type* A)
{
    using T = typename InputMultiVectorType::value_type;

    T one = T(1.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;

    int info = 1;

    std::unique_ptr<T, typename util::device_deleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        A = gpu_backend::device_malloc_array<Backend, T>(V.l_cols() * V.l_cols());
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        detail::query_potrf_buffer_size<Backend, T>(
            solver_handle, V.l_cols(), A, V.l_cols(), &lwork);
        if (lwork > 0)
        {
            workspace = gpu_backend::device_malloc_array<Backend, T>(lwork);
            work_ptr.reset(workspace);
            workspace = work_ptr.get();
        }
    }

    const auto transa =
        chase::linalg::internal::gpu_mpi::detail::
            cholqr_transpose_op<T, Backend>();

    util::check_blas(blas_dispatch::syherk(
        blas_handle, util::fill_upper(), transa, V.l_cols(),
        V.l_rows(), &One, V.l_data(), V.l_ld(), &Zero, A, V.l_cols()));

    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(A, V.l_cols() * V.l_cols(),
        V.getMpiGrid()->get_col_comm());
    gpu_backend::device_synchronize<Backend>();

    int* devInfo;
    devInfo = gpu_backend::device_malloc_array<Backend, int>(1);

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, V.l_cols(), A, V.l_cols(), devInfo, workspace,
        lwork);
    gpu_backend::memcpy_d2h<Backend>(&info, devInfo, 1);

    if (info != 0)
    {
        gpu_backend::device_free<Backend>(devInfo);
        return info;
    }
    else
    {
        util::check_blas(blas_dispatch::trsm(
            blas_handle, util::side_right(),
            util::fill_upper(), util::op_n(),
            util::diag_non_unit(), V.l_rows(), V.l_cols(), &one, A,
            V.l_cols(), V.l_data(), V.l_ld()));
#ifdef CHASE_OUTPUT
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if (grank == 0)
        {
            chase::GetLogger().Log(chase::LogLevel::Info, "linalg", "choldegree: 1",
                0);
        }
#endif
        gpu_backend::device_free<Backend>(devInfo);
        return info;
    }
}

/**
 * @brief A second variant of cholQR for distributed systems using
 * CUDA-Aware-MPI.
 *
 * This function allows for Cholesky QR decomposition, and it supports multi-GPU
 * setups with MPI for synchronized operations across GPUs.
 *
 * @param blas_handle The cuBLAS handle.
 * @param solver_handle The cuSolver handle.
 * @param m The number of rows.
 * @param n The number of columns.
 * @param V The input matrix.
 * @param ldv The leading dimension of the matrix.
 * @param comm The MPI communicator.
 * @param workspace A workspace for temporary data.
 * @param lwork Size of the workspace.
 * @param A Output matrix for the factorized result.
 *
 * @return int Status code: 0 for success, non-zero for failure.
 */
template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename T>
int chol_qr2_impl(BlasHandle blas_handle, SolverHandle solver_handle, std::size_t m,
                      std::size_t n, T* V, int ldv,
                      MPI_Comm comm, T* workspace,
                      int lwork, T* A)
{
    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;

    int info = 0;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    std::unique_ptr<T, typename util::device_deleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        A = gpu_backend::device_malloc_array<Backend, T>(n * n);
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        detail::query_potrf_buffer_size<Backend, T>(solver_handle, n, A, n,
                                                    &lwork);
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }
        workspace = gpu_backend::device_malloc_array<Backend, T>(lwork);
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    const auto transa =
        chase::linalg::internal::gpu_mpi::detail::
            cholqr_transpose_op<T, Backend>();

    util::check_blas(blas_dispatch::syherk(
        blas_handle, util::fill_upper(), transa, n, m, &One, V, ldv,
        &Zero, A, n));

    kernel_ops::extract_upper_triangular(A, n, workspace, n);
    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(workspace, upperTriangularSize, comm);
    kernel_ops::unpack_upper_triangular(workspace, n, A, n);
    gpu_backend::device_synchronize<Backend>();

    int* devInfo;
    devInfo = gpu_backend::device_malloc_array<Backend, int>(1);

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, n, A, n, devInfo, workspace, lwork);
    gpu_backend::memcpy_d2h<Backend>(&info, devInfo, 1);

    if (info != 0)
    {
        gpu_backend::device_free<Backend>(devInfo);
        return info;
    }
    else
    {
        util::check_blas(blas_dispatch::trsm(
            blas_handle, util::side_right(),
            util::fill_upper(), util::op_n(),
            util::diag_non_unit(), m, n, &one, A, n, V, ldv));

        util::check_blas(blas_dispatch::syherk(
            blas_handle, util::fill_upper(), transa, n, m, &One, V, ldv,
            &Zero, A, n));

        kernel_ops::extract_upper_triangular(A, n, workspace,
                                                              n);
        gpu_backend::device_synchronize<Backend>();
        mpiAllreduceSum(workspace, upperTriangularSize, comm);
        kernel_ops::unpack_upper_triangular(workspace, n, A,
                                                             n);
        gpu_backend::device_synchronize<Backend>();

        detail::potrf_with_optional_workspace<Backend, T>(
            blas_handle, solver_handle, n, A, n, devInfo, workspace, lwork);

        util::check_blas(blas_dispatch::trsm(
            blas_handle, util::side_right(),
            util::fill_upper(), util::op_n(),
            util::diag_non_unit(), m, n, &one, A, n, V, ldv));
#ifdef CHASE_OUTPUT
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if (grank == 0)
        {
            chase::GetLogger().Log(chase::LogLevel::Info, "linalg", "choldegree: 2",
                0);
        }
#endif
        gpu_backend::device_free<Backend>(devInfo);
        return info;
    }
}

/**
 * @brief Performs a Cholesky QR decomposition on an input multi-vector type.
 *
 * This function computes the Cholesky QR decomposition of the input matrix or
 * multi-vector `V` using cuBLAS and cuSolver on the GPU. It is designed to work
 * with multi-vector input types where `V` can be a matrix or a vector, and the
 * decomposition is performed in parallel on a GPU. The function also allows for
 * memory optimization through an optional workspace buffer for intermediate
 * calculations.
 *
 * @tparam InputMultiVectorType The type of the input multi-vector (e.g., a
 * matrix or vector type).
 *
 * @param blas_handle The cuBLAS handle used for performing linear algebra
 * operations on the GPU.
 * @param solver_handle The cuSolver handle used for performing the Cholesky
 * factorization on the GPU.
 * @param V The input multi-vector (matrix or vector) to decompose. It will be
 * modified during the process.
 * @param workspace Optional workspace buffer for temporary memory usage during
 * the computation. If not provided, a buffer will be allocated automatically.
 * @param lwork The size of the workspace buffer. If not provided, the function
 * will attempt to determine the optimal size for the workspace.
 * @param A Optional matrix to store the result of the factorization. If not
 * provided, one will be allocated internally.
 *
 * @return int Status code indicating the success or failure of the computation.
 *         - 0 for success.
 *         - Non-zero value indicates failure.
 *
 * @note This function assumes the input multi-vector `V` is stored in a format
 * compatible with cuBLAS and cuSolver. The input matrix must be stored in
 * column-major format.
 *
 * @warning Make sure the appropriate GPU resources (memory and compute
 * capability) are available when calling this function, as it relies on cuBLAS
 * and cuSolver for the Cholesky decomposition and may require a large amount of
 * memory for larger matrices or vectors.
 */
template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename InputMultiVectorType>
int chol_qr2_impl(BlasHandle blas_handle, SolverHandle solver_handle,
                      InputMultiVectorType& V,
                      typename InputMultiVectorType::value_type* workspace,
                      int lwork, typename InputMultiVectorType::value_type* A)
{
    using T = typename InputMultiVectorType::value_type;

    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;

    int info = 0;

    std::unique_ptr<T, typename util::device_deleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        A = gpu_backend::device_malloc_array<Backend, T>(V.l_cols() * V.l_cols());
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        detail::query_potrf_buffer_size<Backend, T>(
            solver_handle, V.l_cols(), A, V.l_cols(), &lwork);
        if (lwork > 0)
        {
            workspace = gpu_backend::device_malloc_array<Backend, T>(lwork);
            work_ptr.reset(workspace);
            workspace = work_ptr.get();
        }
    }

#ifdef ENABLE_MIXED_PRECISION
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
#ifdef CHASE_OUTPUT
        if (grank == 0)
        {
            chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
                "In cholqr2, the first cholqr using Single Precision", 0);
        }
#endif
        V.enableSinglePrecision();
        auto V_sp = V.getSinglePrecisionMatrix();
        info = chol_qr1_impl<Backend, BlasHandle, SolverHandle,
                             typename std::remove_reference<decltype(*V_sp)>::type>(blas_handle, solver_handle, *V_sp);
        V.disableSinglePrecision(true);
    }
    else
    {
        info =
            chol_qr1_impl<Backend, BlasHandle, SolverHandle,
                          InputMultiVectorType>(
                blas_handle, solver_handle, V, workspace, lwork, A);
    }
#else
    info = chol_qr1_impl<Backend, BlasHandle, SolverHandle,
                         InputMultiVectorType>(
        blas_handle, solver_handle, V, workspace, lwork, A);
#endif
    if (info != 0)
    {
        return info;
    }

    info = chol_qr1_impl<Backend, BlasHandle, SolverHandle,
                         InputMultiVectorType>(
        blas_handle, solver_handle, V, workspace, lwork, A);

    return info;
}

/**
 * @brief Performs a shifted Cholesky QR decomposition on a matrix with optional
 * communication support.
 *
 * This function computes the shifted Cholesky QR decomposition of the input
 * matrix `V` using cuBLAS and cuSolver on the GPU. It allows for parallel
 * computation and distributed memory handling, including support for
 * communication across devices using CUDA-Aware MPI. The function can be used
 * in distributed GPU environments and enables efficient memory usage through
 * optional workspace buffers.
 *
 * @tparam T The type of the matrix elements (e.g., float, double, or complex
 * type).
 *
 * @param blas_handle The cuBLAS handle for performing linear algebra
 * operations on the GPU.
 * @param solver_handle The cuSolver handle for computing the Cholesky
 * factorization on the GPU.
 * @param N The order of the matrix `V`, representing the number of rows.
 * @param m The number of rows in the matrix `V` to process.
 * @param n The number of columns in the matrix `V` to process.
 * @param V The input matrix for the decomposition. It is modified during the
 * process.
 * @param ldv The leading dimension of the matrix `V`.
 * @param comm The MPI communicator used for distributed computation across
 * multiple devices.
 * @param workspace Optional workspace buffer for temporary memory usage during
 * computation. If not provided, a buffer will be allocated automatically.
 * @param lwork The size of the workspace buffer. If not provided, the function
 * will attempt to determine the optimal size for the workspace.
 * @param A Optional matrix to store the result of the decomposition. If not
 * provided, one will be allocated internally.
 *
 * @return int Status code indicating the success or failure of the computation.
 *         - 0 for success.
 *         - Non-zero value indicates failure.
 *
 * @note The function assumes that `V` is stored in a format compatible with
 * cuBLAS and cuSolver, and that the matrix is in column-major format. The
 * decomposition is performed on the submatrix defined by the dimensions `m` and
 * `n`.
 *
 * @warning This function requires CUDA-Aware MPI for distributed computing.
 * Ensure it is properly initialized and that multiple devices are available if
 * using distributed mode. The function also requires sufficient GPU memory and
 * computational resources to handle the matrix size and any communication
 * overhead.
 */
template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename T>
int shifted_chol_qr2_impl(BlasHandle blas_handle, SolverHandle solver_handle, std::size_t N,
                             std::size_t m, std::size_t n, T* V, int ldv,
                             MPI_Comm comm, T* workspace,
                             int lwork, T* A)
{
    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    chase::Base<T> shift;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;

    int info = 1;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    std::unique_ptr<T, typename util::device_deleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        A = gpu_backend::device_malloc_array<Backend, T>(n * n);
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        detail::query_potrf_buffer_size<Backend, T>(solver_handle, n, A, n,
                                                    &lwork);
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }

        workspace = gpu_backend::device_malloc_array<Backend, T>(lwork);
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    const auto transa =
        chase::linalg::internal::gpu_mpi::detail::
            cholqr_transpose_op<T, Backend>();

    util::check_blas(blas_dispatch::syherk(
        blas_handle, util::fill_upper(), transa, n, m, &One, V, ldv,
        &Zero, A, n));

    kernel_ops::extract_upper_triangular(A, n, workspace, n);
    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(workspace, upperTriangularSize, comm);
    kernel_ops::unpack_upper_triangular(workspace, n, A, n);
    gpu_backend::device_synchronize<Backend>();

    chase::Base<T> nrmf = 0.0;
    chase::Base<T>* d_nrmf;
    d_nrmf = gpu_backend::device_malloc_array<Backend, chase::Base<T>>(1);
    kernel_ops::abs_trace(A, d_nrmf, n, n);
    gpu_backend::memcpy_d2h<Backend>(&nrmf, d_nrmf, 1);
    if constexpr (std::is_same<chase::Base<T>, float>::value)
    {
        shift = static_cast<chase::Base<T>>(10.0) * nrmf *
                std::numeric_limits<chase::Base<T>>::epsilon();
    }
    else
    {
        shift = std::sqrt(static_cast<double>(N)) * nrmf *
                std::numeric_limits<chase::Base<T>>::epsilon();
    }

    kernel_ops::shift_diagonal(A, n, n, shift);

    int* devInfo;
    devInfo = gpu_backend::device_malloc_array<Backend, int>(1);

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, n, A, n, devInfo, workspace, lwork);
    gpu_backend::memcpy_d2h<Backend>(&info, devInfo, 1);

    if (info != 0)
    {
        return info;
    }

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_right(), util::fill_upper(),
        util::op_n(), util::diag_non_unit(), m, n, &one, A, n, V,
        ldv));

    util::check_blas(blas_dispatch::syherk(
        blas_handle, util::fill_upper(), transa, n, m, &One, V, ldv,
        &Zero, A, n));

    kernel_ops::extract_upper_triangular(A, n, workspace, n);
    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(workspace, upperTriangularSize, comm);
    kernel_ops::unpack_upper_triangular(workspace, n, A, n);
    gpu_backend::device_synchronize<Backend>();

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, n, A, n, devInfo, workspace, lwork);

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_right(), util::fill_upper(),
        util::op_n(), util::diag_non_unit(), m, n, &one, A, n, V,
        ldv));
    util::check_blas(blas_dispatch::syherk(
        blas_handle, util::fill_upper(), transa, n, m, &One, V, ldv,
        &Zero, A, n));

    kernel_ops::extract_upper_triangular(A, n, workspace, n);
    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(workspace, upperTriangularSize, comm);
    kernel_ops::unpack_upper_triangular(workspace, n, A, n);
    gpu_backend::device_synchronize<Backend>();

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, n, A, n, devInfo, workspace, lwork);

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_right(), util::fill_upper(),
        util::op_n(), util::diag_non_unit(), m, n, &one, A, n, V,
        ldv));

#ifdef CHASE_OUTPUT
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    if (grank == 0)
    {
        std::ostringstream oss;
        oss << std::setprecision(2) << "choldegree: 2, shift = " << shift;
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg", oss.str(), 0);
    }
#endif
    gpu_backend::device_free<Backend>(devInfo);

    gpu_backend::device_free<Backend>(d_nrmf);

    return info;
}

/**
 * @brief Performs a Modified Gram-Schmidt QR decomposition with Cholesky
 * updates.
 *
 * This function computes a Modified Gram-Schmidt QR decomposition of the matrix
 * `V`, using Cholesky updates to maintain numerical stability. The
 * decomposition is performed in parallel across multiple devices using cuBLAS,
 * cuSolver, and CUDA-Aware MPI for distributed communication. The matrix is
 * processed in panels, allowing for efficient memory management and
 * computation. The function can be used in a multi-GPU setting with CUDA-Aware
 * MPI.
 *
 * @tparam T The type of the matrix elements (e.g., `float`, `double`, or
 * complex types).
 *
 * @param blas_handle The cuBLAS handle for performing linear algebra
 * operations on the GPU.
 * @param solver_handle The cuSolver handle for computing the Cholesky
 * factorization on the GPU.
 * @param m The number of rows in the matrix `V`.
 * @param n The number of columns in the matrix `V`.
 * @param locked The number of columns already processed in the previous panels.
 * @param V The input matrix to decompose. It is modified during the process.
 * @param ldv The leading dimension of the matrix `V`.
 * @param comm The MPI communicator for distributed computation across devices.
 * @param workspace Optional workspace buffer for temporary memory usage. If not
 * provided, a buffer will be allocated automatically.
 * @param lwork The size of the workspace buffer. If not provided, the function
 * will attempt to determine the optimal size.
 * @param A Optional matrix to store intermediate results of the decomposition.
 * If not provided, one will be allocated internally.
 *
 * @return int Status code indicating the success or failure of the
 * decomposition.
 *         - 0 for success.
 *         - Non-zero value indicates failure.
 *
 * @note The function decomposes the matrix `V` in panels, where each panel is
 * handled separately, and the intermediate results are accumulated in the
 * matrix `A`. The decomposition is carried out in a distributed manner with the
 * help of CUDA-Aware MPI for communication between GPUs. Each panel is
 * processed using cuBLAS operations, and the Cholesky factorization is used to
 * stabilize the process.
 *
 * @note The function handles complex types (`std::complex<float>` or
 * `std::complex<double>`) by using the appropriate cuBLAS operation
 * (`CUBLAS_OP_C`). For real types, `CUBLAS_OP_T` is used for matrix
 * transposition.
 *
 * @note The function uses CUDA-Aware MPI to perform all-reduce operations
 * across devices to ensure that the results are synchronized across multiple
 * GPUs.
 *
 * @warning This function requires CUDA-Aware MPI for distributed computing.
 * Ensure it is properly initialized and that multiple devices are available if
 * using distributed mode. The function also requires sufficient GPU memory and
 * computational resources to handle the matrix size and any communication
 *          overhead.
 *
 * @warning If an error occurs during the computation, a non-zero status code is
 * returned. Ensure that cuBLAS, cuSolver, and CUDA-Aware MPI are properly
 * initialized and that all required memory is allocated.
 */
template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename T>
int modified_gram_schmidt_chol_qr_impl(BlasHandle blas_handle, SolverHandle solver_handle,
                                        std::size_t m, std::size_t n,
                                        std::size_t locked, T* V,
                                        std::size_t ldv,
                                        MPI_Comm comm,
                                        T* workspace, int lwork, T* A)
{
    T one = T(1.0);
    T negone = T(-1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    int info = 1;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;

    int number_of_panels = 4;
    size_t panel_size = ceil((double)n / number_of_panels);
    size_t panel_size_rest;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    int* devInfo;
    devInfo = gpu_backend::device_malloc_array<Backend, int>(1);

    const auto transa =
        chase::linalg::internal::gpu_mpi::detail::
            cholqr_transpose_op<T, Backend>();

    std::unique_ptr<T, typename util::device_deleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        A = gpu_backend::device_malloc_array<Backend, T>(n * n);
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        detail::query_potrf_buffer_size<Backend, T>(solver_handle, n, A, n,
                                                    &lwork);
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }

        workspace = gpu_backend::device_malloc_array<Backend, T>(lwork);
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    if (locked < panel_size)
    {
        info = chol_qr2_impl<Backend, BlasHandle, SolverHandle, T>(
            blas_handle, solver_handle, m, panel_size, V, ldv, comm, workspace,
            lwork = 0, A);
        if (info != 0)
        {
            return info;
        }
    }
    else
    {
        panel_size = locked;
        number_of_panels = ceil((double)n / locked);
    }

    for (auto j = 1; j < number_of_panels; ++j)
    {
        panel_size_rest =
            (j == number_of_panels - 1) ? n - (j)*panel_size : panel_size;

        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_c(), util::op_n(),
            panel_size_rest, n - j * panel_size, m, &one,
            V + (j - 1) * panel_size * ldv, ldv,
            V + j * panel_size * ldv, ldv, &zero, A, panel_size_rest));

        gpu_backend::device_synchronize<Backend>();
        mpiAllreduceSum(A, (n - j * panel_size) * panel_size_rest, comm);

        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_n(), util::op_n(), m,
            n - j * panel_size, panel_size_rest, &negone,
            V + (j - 1) * panel_size * ldv, ldv, A, panel_size_rest, &one,
            V + j * panel_size * ldv, ldv));

        util::check_blas(blas_dispatch::syherk(
            blas_handle, util::fill_upper(), transa, panel_size_rest, m,
            &One, V + j * panel_size * m, ldv, &Zero, A, panel_size_rest));

        std::size_t upperTriangularSize =
            std::size_t(panel_size_rest * (panel_size_rest + 1) / 2);
        kernel_ops::extract_upper_triangular(
            A, panel_size_rest, workspace, panel_size_rest);
        gpu_backend::device_synchronize<Backend>();
        mpiAllreduceSum(workspace, upperTriangularSize, comm);
        kernel_ops::unpack_upper_triangular(
            workspace, panel_size_rest, A, panel_size_rest);
        gpu_backend::device_synchronize<Backend>();

        detail::potrf_with_optional_workspace<Backend, T>(
            blas_handle, solver_handle, panel_size_rest, A, panel_size_rest,
            devInfo, workspace, lwork);
        gpu_backend::memcpy_d2h<Backend>(&info, devInfo, 1);

        if (info != 0)
        {
            return info;
        }
        else
        {
            util::check_blas(blas_dispatch::trsm(
                blas_handle, util::side_right(),
                util::fill_upper(), util::op_n(),
                util::diag_non_unit(), m, panel_size_rest, &one, A,
                panel_size_rest, V + j * panel_size * ldv, ldv));
        }

        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_c(), util::op_n(),
            j * panel_size, panel_size_rest, m, &one, V, ldv,
            V + j * panel_size * ldv, ldv, &zero, A, j * panel_size));

        gpu_backend::device_synchronize<Backend>();

        mpiAllreduceSum(A, (j * panel_size) * panel_size_rest, comm);

        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_n(), util::op_n(), m,
            panel_size_rest, j * panel_size, &negone, V, ldv, A,
            j * panel_size, &one, V + j * panel_size * ldv, ldv));

        info = chol_qr1_impl<Backend, BlasHandle, SolverHandle, T>(
            blas_handle, solver_handle, m, panel_size_rest,
                       V + j * panel_size * ldv, ldv, comm, workspace,
                       lwork = 0, A);
    }

#ifdef CHASE_OUTPUT
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    if (grank == 0)
    {
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
            "Use Modified Gram-Schmidt QR", 0);
    }
#endif
    return info;
}

/**
 * @brief Performs Householder QR factorization on a distributed matrix using
 * ScaLAPACK.
 *
 * This function computes the QR factorization of a distributed matrix \( V \)
 * using the Householder method. It first allocates and transfers the data to
 * the CPU if not already allocated, and then performs the QR factorization
 * using ScaLAPACK routines. The resulting matrix is stored back in the input
 * matrix \( V \).
 *
 * @tparam InputMultiVectorType The type of the input multi-vector (e.g., a
 * distributed matrix).
 *
 * @param[in,out] V The input distributed matrix, which will be factorized in
 * place. The matrix should be in column-major order.
 *
 * @throws std::runtime_error If ScaLAPACK is not available (i.e., for ChASE-MPI
 * builds without ScaLAPACK support).
 *
 * @note This function requires ScaLAPACK for distributed QR. If ScaLAPACK is
 * not available, a runtime error will be thrown.
 *
 * @par ScaLAPACK Functions Used:
 *   - `t_pgeqrf`: Computes the QR factorization of the matrix.
 *   - `t_pgqr`: Computes the solution of the least squares problem or the Q
 * matrix of the QR factorization.
 */
template <typename InputMultiVectorType>
void householder_qr_impl(InputMultiVectorType& V)
{
    using T = typename InputMultiVectorType::value_type;

#ifdef HAS_SCALAPACK
    V.allocate_cpu_data(); // if not allocated
    V.D2H();
    std::size_t* desc = V.scalapack_descriptor_init();
    int one = 1;
    std::vector<T> tau(V.l_cols());

    chase::linalg::scalapackpp::t_pgeqrf(V.g_rows(), V.g_cols(), V.cpu_data(),
                                         one, one, desc, tau.data());

    chase::linalg::scalapackpp::t_pgqr(V.g_rows(), V.g_cols(), V.g_cols(),
                                       V.cpu_data(), one, one, desc,
                                       tau.data());
    V.H2D();

#else
    throw std::runtime_error("For ChASE-MPI, distributed Householder QR requires "
                       "ScaLAPACK, which is not detected\n");
#endif
}


} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
