// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "mpi.h"
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "external/scalapackpp/scalapackpp.hpp"
#include "linalg/internal/cuda/absTrace.cuh"
#include "linalg/internal/cuda/shiftDiagonal.cuh"
#include "linalg/internal/nccl/nccl_kernels.hpp"

using namespace chase::linalg::blaspp;
using namespace chase::linalg::lapackpp;

namespace chase
{
namespace linalg
{
namespace internal
{
/**
 * @brief Performs a distributed Cholesky QR decomposition on a matrix V.
 *
 * This function decomposes the matrix V into a QR form using Cholesky
 * factorization. It is designed to work on a multi-GPU setup where NCCL is used
 * for communication between GPUs.
 *
 * @param cublas_handle The cuBLAS handle to perform linear algebra operations.
 * @param cusolver_handle The cuSolver handle to compute Cholesky factorization.
 * @param m The number of rows in the matrix V.
 * @param n The number of columns in the matrix V.
 * @param V The input matrix to decompose, stored in column-major format.
 * @param ldv The leading dimension of the matrix V.
 * @param comm The NCCL communicator used for multi-GPU collective
 * communication.
 * @param workspace Optional workspace buffer for intermediate calculations.
 * @param lwork The size of the workspace buffer.
 * @param A Optional matrix for storing the result of the factorization.
 *
 * @return int Status code: 0 for success, non-zero for failure.
 */
template <typename T>
int cuda_nccl::cholQR1(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle, std::size_t m,
                       std::size_t n, T* V, int ldv,
                       // MPI_Comm comm,
                       ncclComm_t comm, T* workspace, int lwork, T* A)
{
    // No-op for zero dimensions: avoid cuBLAS/cuSOLVER with n=0 or m=0
    // (can trigger driver bugs e.g. null deref in cuEGLApiInit on some systems)
    if (n == 0 || m == 0)
        return 0;

    // cholQR1 passes host scalar pointers (&One, &Zero); ensure HOST pointer
    // mode (GPU-resident Lanczos may leave the handle in DEVICE mode).
    cublasPointerMode_t prev_mode = CUBLAS_POINTER_MODE_HOST;
    CHECK_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &prev_mode));
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);

    int info = 1;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    // Detailed timing events
    cudaEvent_t ev_start, ev_syherk, ev_allreduce, ev_potrf, ev_trsm, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_syherk);
    cudaEventCreate(&ev_allreduce);
    cudaEventCreate(&ev_potrf);
    cudaEventCreate(&ev_trsm);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&A, n * n * sizeof(T)));
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, &lwork));
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }

        CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    cublasOperation_t transa;
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        transa = CUBLAS_OP_C;
    }
    else
    {
        transa = CUBLAS_OP_T;
    }

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, n, m, &One, V, ldv,
        &Zero, A, n));
    cudaEventRecord(ev_syherk);

    chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
        workspace, workspace, upperTriangularSize, ncclSum, comm));
    chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
    cudaEventRecord(ev_allreduce);

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, workspace, lwork,
        devInfo));
    cudaEventRecord(ev_potrf);

    CHECK_CUDA_ERROR(
        cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    if (info != 0)
    {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_syherk);
        cudaEventDestroy(ev_allreduce);
        cudaEventDestroy(ev_potrf);
        cudaEventDestroy(ev_trsm);
        cudaEventDestroy(ev_end);
        CHECK_CUDA_ERROR(cudaFree(devInfo));
        CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_mode));
        return info;
    }
    else
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, A, n, V, ldv));
        cudaEventRecord(ev_trsm);
        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);

        // Calculate and print breakdown timing
        float t_syherk, t_allreduce, t_potrf, t_trsm, t_total;
        cudaEventElapsedTime(&t_syherk, ev_start, ev_syherk);
        cudaEventElapsedTime(&t_allreduce, ev_syherk, ev_allreduce);
        cudaEventElapsedTime(&t_potrf, ev_allreduce, ev_potrf);
        cudaEventElapsedTime(&t_trsm, ev_potrf, ev_trsm);
        cudaEventElapsedTime(&t_total, ev_start, ev_end);

#ifdef CHASE_OUTPUT
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if (grank == 0)
        {
            std::cout << "choldegree: 1" << std::endl;
            std::cout << std::setprecision(6) << std::fixed;
            std::cout << "  [cholQR1 Breakdown] Total: " << t_total/1000.0 << " s" << std::endl;
            std::cout << "    SYHERK:       " << t_syherk/1000.0 << " s (" << 100.0*t_syherk/t_total << "%)" << std::endl;
            std::cout << "    AllReduce:    " << t_allreduce/1000.0 << " s (" << 100.0*t_allreduce/t_total << "%)" << std::endl;
            std::cout << "    Cholesky:     " << t_potrf/1000.0 << " s (" << 100.0*t_potrf/t_total << "%)" << std::endl;
            std::cout << "    TRSM:         " << t_trsm/1000.0 << " s (" << 100.0*t_trsm/t_total << "%)" << std::endl;
        }
#endif
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_syherk);
        cudaEventDestroy(ev_allreduce);
        cudaEventDestroy(ev_potrf);
        cudaEventDestroy(ev_trsm);
        cudaEventDestroy(ev_end);

        CHECK_CUDA_ERROR(cudaFree(devInfo));
        CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_mode));
        return info;
    }
}

/**
 * @brief Variant of cholQR1 for InputMultiVectorType.
 *
 * This variant works with InputMultiVectorType and performs the Cholesky QR
 * decomposition in the same manner as cholQR1, but with different input data
 * type handling. The use of NCCL ensures that the computation is parallelized
 * across multiple GPUs.
 *
 * @param cublas_handle The cuBLAS handle for linear algebra operations.
 * @param cusolver_handle The cuSolver handle for Cholesky factorization.
 * @param V The input matrix to decompose.
 * @param workspace Optional workspace buffer.
 * @param lwork The size of the workspace buffer.
 * @param A Optional matrix for storing the factorization result.
 *
 * @return int Status code: 0 for success, non-zero for failure.
 */
template <typename InputMultiVectorType>
int cuda_nccl::cholQR1(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle,
                       InputMultiVectorType& V,
                       typename InputMultiVectorType::value_type* workspace,
                       int lwork, typename InputMultiVectorType::value_type* A)
{
    using T = typename InputMultiVectorType::value_type;

    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);

    int info = 1;

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&A, V.l_cols() * V.l_cols() * sizeof(T)));
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.l_cols(), A,
                V.l_cols(), &lwork));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    cublasOperation_t transa;
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        transa = CUBLAS_OP_C;
    }
    else
    {
        transa = CUBLAS_OP_T;
    }

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, V.l_cols(), V.l_rows(),
        &One, V.l_data(), V.l_ld(), &Zero, A, V.l_cols()));

    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
        A, A, V.l_cols() * V.l_cols(), ncclSum,
        V.getMpiGrid()->get_nccl_col_comm()));

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.l_cols(), A, V.l_cols(),
        workspace, lwork, devInfo));
    
    CHECK_CUDA_ERROR(
        cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    if (info != 0)
    {
        CHECK_CUDA_ERROR(cudaFree(devInfo));
        return info;
    }
    else
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, V.l_rows(), V.l_cols(), &one, A,
            V.l_cols(), V.l_data(), V.l_ld()));

#ifdef CHASE_OUTPUT
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if (grank == 0)
        {
            std::cout << "choldegree: 1" << std::endl;
        }
#endif
        
        CHECK_CUDA_ERROR(cudaFree(devInfo));
        return info;
    }
}
/**
 * @brief A second variant of cholQR for distributed systems using NCCL.
 *
 * This function allows for Cholesky QR decomposition, and it supports multi-GPU
 * setups with NCCL for synchronized operations across GPUs.
 *
 * @param cublas_handle The cuBLAS handle.
 * @param cusolver_handle The cuSolver handle.
 * @param m The number of rows.
 * @param n The number of columns.
 * @param V The input matrix.
 * @param ldv The leading dimension of the matrix.
 * @param comm The NCCL communicator.
 * @param workspace A workspace for temporary data.
 * @param lwork Size of the workspace.
 * @param A Output matrix for the factorized result.
 *
 * @return int Status code: 0 for success, non-zero for failure.
 */
template <typename T>
int cuda_nccl::cholQR2(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle, std::size_t m,
                       std::size_t n, T* V, int ldv, ncclComm_t comm,
                       T* workspace, int lwork, T* A)
{

    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);

    int info = 0;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    // Detailed timing events for cholQR2 (2 iterations)
    cudaEvent_t ev_start, ev_syherk1, ev_allreduce1, ev_potrf1, ev_trsm1;
    cudaEvent_t ev_syherk2, ev_allreduce2, ev_potrf2, ev_trsm2, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_syherk1);
    cudaEventCreate(&ev_allreduce1);
    cudaEventCreate(&ev_potrf1);
    cudaEventCreate(&ev_trsm1);
    cudaEventCreate(&ev_syherk2);
    cudaEventCreate(&ev_allreduce2);
    cudaEventCreate(&ev_potrf2);
    cudaEventCreate(&ev_trsm2);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&A, n * n * sizeof(T)));
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, &lwork));
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }
        CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    cublasOperation_t transa;
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        transa = CUBLAS_OP_C;
    }
    else
    {
        transa = CUBLAS_OP_T;
    }

    // === Iteration 1 ===
    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, n, m, &One, V, ldv,
        &Zero, A, n));
    cudaEventRecord(ev_syherk1);

    chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
        workspace, workspace, upperTriangularSize, ncclSum, comm));
    chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
    cudaEventRecord(ev_allreduce1);

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, workspace, lwork,
        devInfo));
    cudaEventRecord(ev_potrf1);

    CHECK_CUDA_ERROR(
        cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    if (info != 0)
    {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_syherk1);
        cudaEventDestroy(ev_allreduce1);
        cudaEventDestroy(ev_potrf1);
        cudaEventDestroy(ev_trsm1);
        cudaEventDestroy(ev_syherk2);
        cudaEventDestroy(ev_allreduce2);
        cudaEventDestroy(ev_potrf2);
        cudaEventDestroy(ev_trsm2);
        cudaEventDestroy(ev_end);
        CHECK_CUDA_ERROR(cudaFree(devInfo));
        return info;
    }
    else
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, A, n, V, ldv));
        cudaEventRecord(ev_trsm1);

        // === Iteration 2 ===
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
            cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, n, m, &One, V, ldv,
            &Zero, A, n));
        cudaEventRecord(ev_syherk2);

        chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
            workspace, workspace, upperTriangularSize, ncclSum, comm));
        chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
        cudaEventRecord(ev_allreduce2);

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
            cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, workspace, lwork,
            devInfo));
        cudaEventRecord(ev_potrf2);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &one, A, n, V, ldv));
        cudaEventRecord(ev_trsm2);
        cudaEventRecord(ev_end);
        cudaEventSynchronize(ev_end);

        // Calculate and print breakdown timing
        float t_syherk1, t_allreduce1, t_potrf1, t_trsm1;
        float t_syherk2, t_allreduce2, t_potrf2, t_trsm2, t_total;
        cudaEventElapsedTime(&t_syherk1, ev_start, ev_syherk1);
        cudaEventElapsedTime(&t_allreduce1, ev_syherk1, ev_allreduce1);
        cudaEventElapsedTime(&t_potrf1, ev_allreduce1, ev_potrf1);
        cudaEventElapsedTime(&t_trsm1, ev_potrf1, ev_trsm1);
        cudaEventElapsedTime(&t_syherk2, ev_trsm1, ev_syherk2);
        cudaEventElapsedTime(&t_allreduce2, ev_syherk2, ev_allreduce2);
        cudaEventElapsedTime(&t_potrf2, ev_allreduce2, ev_potrf2);
        cudaEventElapsedTime(&t_trsm2, ev_potrf2, ev_trsm2);
        cudaEventElapsedTime(&t_total, ev_start, ev_end);

#ifdef CHASE_OUTPUT
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if (grank == 0)
        {
            std::cout << "choldegree: 2" << std::endl;
            std::cout << std::setprecision(6) << std::fixed;
            std::cout << "  [cholQR2 Breakdown] Total: " << t_total/1000.0 << " s" << std::endl;
            std::cout << "  Iter1: SYHERK=" << t_syherk1/1000.0 << "s, AllReduce=" << t_allreduce1/1000.0 
                      << "s, Cholesky=" << t_potrf1/1000.0 << "s, TRSM=" << t_trsm1/1000.0 << "s" << std::endl;
            std::cout << "  Iter2: SYHERK=" << t_syherk2/1000.0 << "s, AllReduce=" << t_allreduce2/1000.0 
                      << "s, Cholesky=" << t_potrf2/1000.0 << "s, TRSM=" << t_trsm2/1000.0 << "s" << std::endl;
        }
#endif
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_syherk1);
        cudaEventDestroy(ev_allreduce1);
        cudaEventDestroy(ev_potrf1);
        cudaEventDestroy(ev_trsm1);
        cudaEventDestroy(ev_syherk2);
        cudaEventDestroy(ev_allreduce2);
        cudaEventDestroy(ev_potrf2);
        cudaEventDestroy(ev_trsm2);
        cudaEventDestroy(ev_end);

        CHECK_CUDA_ERROR(cudaFree(devInfo));
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
 * @param cublas_handle The cuBLAS handle used for performing linear algebra
 * operations on the GPU.
 * @param cusolver_handle The cuSolver handle used for performing the Cholesky
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
template <typename InputMultiVectorType>
int cuda_nccl::cholQR2(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle,
                       InputMultiVectorType& V,
                       typename InputMultiVectorType::value_type* workspace,
                       int lwork, typename InputMultiVectorType::value_type* A)
{
    using T = typename InputMultiVectorType::value_type;

    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);

    int info = 0;

    // Detailed timing events
    cudaEvent_t ev_start, ev_syherk1, ev_allreduce1, ev_potrf1, ev_trsm1;
    cudaEvent_t ev_syherk2, ev_allreduce2, ev_potrf2, ev_trsm2, ev_end;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_syherk1);
    cudaEventCreate(&ev_allreduce1);
    cudaEventCreate(&ev_potrf1);
    cudaEventCreate(&ev_trsm1);
    cudaEventCreate(&ev_syherk2);
    cudaEventCreate(&ev_allreduce2);
    cudaEventCreate(&ev_potrf2);
    cudaEventCreate(&ev_trsm2);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&A, V.l_cols() * V.l_cols() * sizeof(T)));
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.l_cols(), A,
                V.l_cols(), &lwork));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

#ifdef ENABLE_MIXED_PRECISION
    if constexpr (std::is_same<T, double>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        std::cout << "In cholqr2, the first cholqr using Single Precision"
                  << std::endl;
        using singlePrecisionT =
            typename chase::ToSinglePrecisionTrait<T>::Type;
        V.enableSinglePrecision();
        auto V_sp = V.getSinglePrecisionMatrix();
        info = cholQR1<singlePrecisionT>(cublas_handle, cusolver_handle, *V_sp);
        V.disableSinglePrecision(true);
    }
    else
    {
        info =
            cholQR1<T>(cublas_handle, cusolver_handle, V, workspace, lwork, A);
    }
#else
    info = cholQR1<T>(cublas_handle, cusolver_handle, V, workspace, lwork, A);
#endif
    if (info != 0)
    {
        return info;
    }

    info = cholQR1<T>(cublas_handle, cusolver_handle, V, workspace, lwork, A);

    return info;
}

/**
 * @brief Performs a shifted Cholesky QR decomposition on a matrix with optional
 * communication support.
 *
 * This function computes the shifted Cholesky QR decomposition of the input
 * matrix `V` using cuBLAS and cuSolver on the GPU. It allows for parallel
 * computation and distributed memory handling, including support for
 * communication across devices using NCCL (NVIDIA Collective Communications
 * Library). The function can be used in distributed GPU environments and
 * enables efficient memory usage through optional workspace buffers.
 *
 * @tparam T The type of the matrix elements (e.g., float, double, or complex
 * type).
 *
 * @param cublas_handle The cuBLAS handle for performing linear algebra
 * operations on the GPU.
 * @param cusolver_handle The cuSolver handle for computing the Cholesky
 * factorization on the GPU.
 * @param N The order of the matrix `V`, representing the number of rows.
 * @param m The number of rows in the matrix `V` to process.
 * @param n The number of columns in the matrix `V` to process.
 * @param V The input matrix for the decomposition. It is modified during the
 * process.
 * @param ldv The leading dimension of the matrix `V`.
 * @param comm The NCCL communicator used for distributed computation across
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
 * @warning This function requires NCCL for distributed computing. Ensure that
 * the NCCL library is properly initialized and that multiple devices are
 * available if using distributed mode. The function also requires sufficient
 * GPU memory and computational resources to handle the matrix size and any
 * communication overhead.
 */
template <typename T>
int cuda_nccl::shiftedcholQR2(cublasHandle_t cublas_handle,
                              cusolverDnHandle_t cusolver_handle, std::size_t N,
                              std::size_t m, std::size_t n, T* V, int ldv,
                              ncclComm_t comm, T* workspace, int lwork, T* A)
{
    T one = T(1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    chase::Base<T> shift;

    int info = 1;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    // Detailed timing events for shiftedcholQR2 (3 iterations)
    cudaEvent_t ev_start, ev_end;
    cudaEvent_t ev_syherk1, ev_allreduce1, ev_potrf1, ev_trsm1;
    cudaEvent_t ev_syherk2, ev_allreduce2, ev_potrf2, ev_trsm2;
    cudaEvent_t ev_syherk3, ev_allreduce3, ev_potrf3, ev_trsm3;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_syherk1);
    cudaEventCreate(&ev_allreduce1);
    cudaEventCreate(&ev_potrf1);
    cudaEventCreate(&ev_trsm1);
    cudaEventCreate(&ev_syherk2);
    cudaEventCreate(&ev_allreduce2);
    cudaEventCreate(&ev_potrf2);
    cudaEventCreate(&ev_trsm2);
    cudaEventCreate(&ev_syherk3);
    cudaEventCreate(&ev_allreduce3);
    cudaEventCreate(&ev_potrf3);
    cudaEventCreate(&ev_trsm3);
    cudaEventCreate(&ev_end);
    cudaEventRecord(ev_start);

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&A, n * n * sizeof(T)));
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, &lwork));
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }

        CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    cublasOperation_t transa;
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        transa = CUBLAS_OP_C;
    }
    else
    {
        transa = CUBLAS_OP_T;
    }

    // === Iteration 1 (with shift) ===
    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, n, m, &One, V, ldv,
        &Zero, A, n));
    cudaEventRecord(ev_syherk1);

    chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
        workspace, workspace, upperTriangularSize, ncclSum, comm));
    chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
    cudaEventRecord(ev_allreduce1);

    chase::Base<T> nrmf = 0.0;
    chase::Base<T>* d_nrmf;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nrmf, sizeof(chase::Base<T>)));
    chase::linalg::internal::cuda::absTrace_gpu(A, d_nrmf, n, n,
                                                (cudaStream_t)0);
    CHECK_CUDA_ERROR(cudaMemcpy(&nrmf, d_nrmf, sizeof(chase::Base<T>),
                                cudaMemcpyDeviceToHost));
    shift =
        std::sqrt(N) * nrmf * std::numeric_limits<chase::Base<T>>::epsilon();

    chase::linalg::internal::cuda::chase_shift_matrix(A, n, n, shift,
                                                      (cudaStream_t)0);

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, workspace, lwork,
        devInfo));
    cudaEventRecord(ev_potrf1);

    CHECK_CUDA_ERROR(
        cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

    if (info != 0)
    {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_syherk1);
        cudaEventDestroy(ev_allreduce1);
        cudaEventDestroy(ev_potrf1);
        cudaEventDestroy(ev_trsm1);
        cudaEventDestroy(ev_syherk2);
        cudaEventDestroy(ev_allreduce2);
        cudaEventDestroy(ev_potrf2);
        cudaEventDestroy(ev_trsm2);
        cudaEventDestroy(ev_syherk3);
        cudaEventDestroy(ev_allreduce3);
        cudaEventDestroy(ev_potrf3);
        cudaEventDestroy(ev_trsm3);
        cudaEventDestroy(ev_end);
        CHECK_CUDA_ERROR(cudaFree(d_nrmf));
        return info;
    }

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
        cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, m, n, &one, A, n, V, ldv));
    cudaEventRecord(ev_trsm1);

    // === Iteration 2 ===
    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, n, m, &One, V, ldv,
        &Zero, A, n));
    cudaEventRecord(ev_syherk2);

    chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
        workspace, workspace, upperTriangularSize, ncclSum, comm));
    chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
    cudaEventRecord(ev_allreduce2);

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, workspace, lwork,
        devInfo));
    cudaEventRecord(ev_potrf2);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
        cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, m, n, &one, A, n, V, ldv));
    cudaEventRecord(ev_trsm2);

    // === Iteration 3 ===
    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, n, m, &One, V, ldv,
        &Zero, A, n));
    cudaEventRecord(ev_syherk3);

    chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
        workspace, workspace, upperTriangularSize, ncclSum, comm));
    chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
    cudaEventRecord(ev_allreduce3);

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, workspace, lwork,
        devInfo));
    cudaEventRecord(ev_potrf3);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
        cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT, m, n, &one, A, n, V, ldv));
    cudaEventRecord(ev_trsm3);
    cudaEventRecord(ev_end);
    cudaEventSynchronize(ev_end);

    // Calculate and print breakdown timing
    float t_syherk1, t_allreduce1, t_potrf1, t_trsm1;
    float t_syherk2, t_allreduce2, t_potrf2, t_trsm2;
    float t_syherk3, t_allreduce3, t_potrf3, t_trsm3, t_total;
    cudaEventElapsedTime(&t_syherk1, ev_start, ev_syherk1);
    cudaEventElapsedTime(&t_allreduce1, ev_syherk1, ev_allreduce1);
    cudaEventElapsedTime(&t_potrf1, ev_allreduce1, ev_potrf1);
    cudaEventElapsedTime(&t_trsm1, ev_potrf1, ev_trsm1);
    cudaEventElapsedTime(&t_syherk2, ev_trsm1, ev_syherk2);
    cudaEventElapsedTime(&t_allreduce2, ev_syherk2, ev_allreduce2);
    cudaEventElapsedTime(&t_potrf2, ev_allreduce2, ev_potrf2);
    cudaEventElapsedTime(&t_trsm2, ev_potrf2, ev_trsm2);
    cudaEventElapsedTime(&t_syherk3, ev_trsm2, ev_syherk3);
    cudaEventElapsedTime(&t_allreduce3, ev_syherk3, ev_allreduce3);
    cudaEventElapsedTime(&t_potrf3, ev_allreduce3, ev_potrf3);
    cudaEventElapsedTime(&t_trsm3, ev_potrf3, ev_trsm3);
    cudaEventElapsedTime(&t_total, ev_start, ev_end);

#ifdef CHASE_OUTPUT
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    if (grank == 0)
    {
        std::cout << std::setprecision(6) << std::fixed;
        std::cout << "choldegree: 2, shift = " << shift << std::endl;
        std::cout << "  [shiftedCholQR2 Breakdown] Total: " << t_total/1000.0 << " s" << std::endl;
        std::cout << "  Iter1(shifted): SYHERK=" << t_syherk1/1000.0 << "s, AllReduce=" << t_allreduce1/1000.0 
                  << "s, Cholesky=" << t_potrf1/1000.0 << "s, TRSM=" << t_trsm1/1000.0 << "s" << std::endl;
        std::cout << "  Iter2: SYHERK=" << t_syherk2/1000.0 << "s, AllReduce=" << t_allreduce2/1000.0 
                  << "s, Cholesky=" << t_potrf2/1000.0 << "s, TRSM=" << t_trsm2/1000.0 << "s" << std::endl;
        std::cout << "  Iter3: SYHERK=" << t_syherk3/1000.0 << "s, AllReduce=" << t_allreduce3/1000.0 
                  << "s, Cholesky=" << t_potrf3/1000.0 << "s, TRSM=" << t_trsm3/1000.0 << "s" << std::endl;
    }
#endif
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_syherk1);
    cudaEventDestroy(ev_allreduce1);
    cudaEventDestroy(ev_potrf1);
    cudaEventDestroy(ev_trsm1);
    cudaEventDestroy(ev_syherk2);
    cudaEventDestroy(ev_allreduce2);
    cudaEventDestroy(ev_potrf2);
    cudaEventDestroy(ev_trsm2);
    cudaEventDestroy(ev_syherk3);
    cudaEventDestroy(ev_allreduce3);
    cudaEventDestroy(ev_potrf3);
    cudaEventDestroy(ev_trsm3);
    cudaEventDestroy(ev_end);

    CHECK_CUDA_ERROR(cudaFree(devInfo));
    CHECK_CUDA_ERROR(cudaFree(d_nrmf));

    return info;
}

/**
 * @brief Performs a Modified Gram-Schmidt QR decomposition with Cholesky
 * updates.
 *
 * This function computes a Modified Gram-Schmidt QR decomposition of the matrix
 * `V`, using Cholesky updates to maintain numerical stability. The
 * decomposition is performed in parallel across multiple devices using cuBLAS,
 * cuSolver, and NCCL for distributed communication. The matrix is processed in
 * panels, allowing for efficient memory management and computation. The
 * function can be used in a multi-GPU setting with NCCL-based collective
 * communications.
 *
 * @tparam T The type of the matrix elements (e.g., `float`, `double`, or
 * complex types).
 *
 * @param cublas_handle The cuBLAS handle for performing linear algebra
 * operations on the GPU.
 * @param cusolver_handle The cuSolver handle for computing the Cholesky
 * factorization on the GPU.
 * @param m The number of rows in the matrix `V`.
 * @param n The number of columns in the matrix `V`.
 * @param locked The number of columns already processed in the previous panels.
 * @param V The input matrix to decompose. It is modified during the process.
 * @param ldv The leading dimension of the matrix `V`.
 * @param comm The NCCL communicator for distributed computation across devices.
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
 * help of NCCL for communication between GPUs. Each panel is processed using
 * cuBLAS operations, and the Cholesky factorization is used to stabilize the
 * process.
 *
 * @note The function handles complex types (`std::complex<float>` or
 * `std::complex<double>`) by using the appropriate cuBLAS operation
 * (`CUBLAS_OP_C`). For real types, `CUBLAS_OP_T` is used for matrix
 * transposition.
 *
 * @note The function uses NCCL to perform all-reduce operations across devices
 * to ensure that the results are synchronized across multiple GPUs.
 *
 * @warning This function requires NCCL for distributed computing. Ensure that
 * the NCCL library is properly initialized and that multiple devices are
 * available if using distributed mode. The function also requires sufficient
 * GPU memory and computational resources to handle the matrix size and any
 * communication overhead.
 *
 * @warning If an error occurs during the computation, a non-zero status code is
 * returned. Ensure that cuBLAS, cuSolver, and NCCL are properly initialized and
 * that all required memory is allocated.
 */
template <typename T>
int cuda_nccl::modifiedGramSchmidtCholQR(cublasHandle_t cublas_handle,
                                         cusolverDnHandle_t cusolver_handle,
                                         std::size_t m, std::size_t n,
                                         std::size_t locked, T* V,
                                         std::size_t ldv, ncclComm_t comm,
                                         T* workspace, int lwork, T* A)
{
    T one = T(1.0);
    T negone = T(-1.0);
    T zero = T(0.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    int info = 1;

    int number_of_panels = 6;
    size_t panel_size = ceil((double)n / number_of_panels);
    size_t panel_size_rest;
    std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    cublasOperation_t transa;
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, std::complex<double>>::value)
    {
        transa = CUBLAS_OP_C;
    }
    else
    {
        transa = CUBLAS_OP_T;
    }

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
    if (A == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMalloc(&A, n * n * sizeof(T)));
        A_ptr.reset(A);
        A = A_ptr.get();
    }

    std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, n, A, n, &lwork));
        if (upperTriangularSize > lwork)
        {
            lwork = upperTriangularSize;
        }

        CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    if (locked < panel_size)
    {
        info = cholQR2(cublas_handle, cusolver_handle, m, panel_size, V, ldv,
                       comm, workspace, lwork = 0, A);
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

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, panel_size_rest,
            n - j * panel_size, m, &one, V + (j - 1) * panel_size * ldv, ldv,
            V + j * panel_size * ldv, ldv, &zero, A, panel_size_rest));

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
            A, A, (n - j * panel_size) * panel_size_rest, ncclSum, comm));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n - j * panel_size,
            panel_size_rest, &negone, V + (j - 1) * panel_size * ldv, ldv, A,
            panel_size_rest, &one, V + j * panel_size * ldv, ldv));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
            cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, panel_size_rest, m,
            &One, V + j * panel_size * m, ldv, &Zero, A, panel_size_rest));

        std::size_t upperTriangularSize =
            std::size_t(panel_size_rest * (panel_size_rest + 1) / 2);
        chase::linalg::internal::cuda::extractUpperTriangular(
            A, panel_size_rest, workspace, panel_size_rest);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
            workspace, workspace, upperTriangularSize, ncclSum, comm));
        chase::linalg::internal::cuda::unpackUpperTriangular(
            workspace, panel_size_rest, A, panel_size_rest);

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
            cusolver_handle, CUBLAS_FILL_MODE_UPPER, panel_size_rest, A,
            panel_size_rest, workspace, lwork, devInfo));
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int),
                                    cudaMemcpyDeviceToHost));

        if (info != 0)
        {
            return info;
        }
        else
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
                cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, panel_size_rest, &one, A,
                panel_size_rest, V + j * panel_size * ldv, ldv));
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, j * panel_size,
            panel_size_rest, m, &one, V, ldv, V + j * panel_size * ldv, ldv,
            &zero, A, j * panel_size));

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
            A, A, (j * panel_size) * panel_size_rest, ncclSum, comm));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, panel_size_rest,
            j * panel_size, &negone, V, ldv, A, j * panel_size, &one,
            V + j * panel_size * ldv, ldv));

        info = cholQR1(cublas_handle, cusolver_handle, m, panel_size_rest,
                       V + j * panel_size * ldv, ldv, comm, workspace,
                       lwork = 0, A);
    }

#ifdef CHASE_OUTPUT
    int grank;
    MPI_Comm_rank(MPI_COMM_WORLD, &grank);
    if (grank == 0)
    {
        std::cout << "Use Modified Gram-Schmidt QR" << std::endl;
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
void cuda_nccl::houseHoulderQR(InputMultiVectorType& V)
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
    std::runtime_error("For ChASE-MPI, distributed Householder QR requires "
                       "ScaLAPACK, which is not detected\n");
#endif
}

} // namespace internal
} // namespace linalg
} // namespace chase
