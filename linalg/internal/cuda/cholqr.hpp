// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/cuda_utils.hpp"
#include "Impl/chase_gpu/nvtx.hpp"
#include "algorithm/logger.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "linalg/internal/cuda/absTrace.hpp"
#include "linalg/internal/cuda/shiftDiagonal.hpp"
#include "linalg/matrix/matrix.hpp"
#include <iomanip>
#include <limits>
#include <sstream>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
/**
 * @brief Performs a Cholesky-based QR factorization with a degree of 1.
 *
 * This function computes the Cholesky decomposition of the matrix \( A \),
 * and uses it for a QR factorization of a given matrix \( V \) on a GPU.
 * The process involves matrix operations using cuBLAS and cuSolver.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double, or
 * complex types).
 * @param cublas_handle The cuBLAS handle for managing operations.
 * @param cusolver_handle The cuSolver handle for solving linear systems and
 * decompositions.
 * @param V The input/output matrix \( V \) to be factored (on GPU).
 * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
 * @param lwork The size of the workspace (default: 0).
 * @param A Optional output matrix for the Cholesky factor (default: nullptr).
 *           If not provided, a new matrix is allocated.
 * @return int An error code (0 indicates success, non-zero indicates failure).
 */

template <typename T>
int cholQR1(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            chase::matrix::Matrix<T, chase::platform::GPU>& V,
            T* workspace = nullptr, int lwork = 0,
            chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
            int* external_devInfo = nullptr)
{
    SCOPED_NVTX_RANGE();
   
    T one = T(1.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);

    int info = 1;
    bool owns_A = false;
    bool owns_workspace = false;
    bool owns_devInfo = false;

    if (A == nullptr)
    {
        A = new chase::matrix::Matrix<T, chase::platform::GPU>(V.cols(), V.cols());
        owns_A = true;
    }

    cudaStream_t stream_orig_cublas = nullptr;
    cudaStream_t stream_orig_cusolver = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t evt_begin = nullptr;
    cudaEvent_t evt_end = nullptr;

    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(cusolverDnGetStream(cusolver_handle, &stream_orig_cusolver));
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig_cublas));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream));

    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(),
                V.cols(), &lwork));
        CHECK_CUDA_ERROR(cudaMallocAsync((void**)&workspace, sizeof(T) * lwork, stream));
        owns_workspace = true;
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
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, V.cols(), V.rows(), &One,
        V.data(), V.ld(), &Zero, A->data(), V.cols()));
    int* devInfo = external_devInfo;
    if (devInfo == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMallocAsync((void**)&devInfo, sizeof(int), stream));
        owns_devInfo = true;
    }

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(), V.cols(),
        workspace, lwork, devInfo));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (info == 0)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, V.rows(), V.cols(), &one,
            A->data(), V.cols(), V.data(), V.ld()));
#ifdef CHASE_OUTPUT
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
            "choldegree: 1\n", 0);
#endif
    }

    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig_cublas, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream_orig_cusolver));
    if (owns_devInfo)
        CHECK_CUDA_ERROR(cudaFreeAsync(devInfo, stream));
    if (owns_workspace)
        CHECK_CUDA_ERROR(cudaFreeAsync(workspace, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    if (owns_A)
        delete A;

    return info;
}

/**
 * @brief Performs a Cholesky-based QR factorization with a degree of 2.
 *
 * This function computes the Cholesky decomposition of the matrix \( A \),
 * and uses it for a QR factorization of a given matrix \( V \) on a GPU.
 * It involves several matrix operations using cuBLAS and cuSolver to perform
 * the QR factorization.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double, or
 * complex types).
 * @param cublas_handle The cuBLAS handle for managing operations.
 * @param cusolver_handle The cuSolver handle for solving linear systems and
 * decompositions.
 * @param V The input/output matrix \( V \) to be factored (on GPU).
 * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
 * @param lwork The size of the workspace (default: 0).
 * @param A Optional output matrix for the Cholesky factor (default: nullptr).
 *           If not provided, a new matrix is allocated.
 * @return int An error code (0 indicates success, non-zero indicates failure).
 */
template <typename T>
int cholQR2(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            chase::matrix::Matrix<T, chase::platform::GPU>& V,
            T* workspace = nullptr, int lwork = 0,
            chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
            int* external_devInfo = nullptr)
{
    SCOPED_NVTX_RANGE();

    T one = T(1.0);
    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    int info = 1;
    bool owns_A = false;
    bool owns_workspace = false;
    bool owns_devInfo = false;

    cudaStream_t stream_orig_cublas = nullptr;
    cudaStream_t stream_orig_cusolver = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t evt_begin = nullptr;
    cudaEvent_t evt_end = nullptr;

    if (A == nullptr)
    {
        A = new chase::matrix::Matrix<T, chase::platform::GPU>(V.cols(), V.cols());
        owns_A = true;
    }

    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(cusolverDnGetStream(cusolver_handle, &stream_orig_cusolver));
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig_cublas));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream));

    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(),
                V.cols(), &lwork));
        CHECK_CUDA_ERROR(cudaMallocAsync((void**)&workspace, sizeof(T) * lwork, stream));
        owns_workspace = true;
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
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, V.cols(), V.rows(), &One,
        V.data(), V.ld(), &Zero, A->data(), V.cols()));
    int* devInfo = external_devInfo;
    if (devInfo == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMallocAsync((void**)&devInfo, sizeof(int), stream));
        owns_devInfo = true;
    }

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(), V.cols(),
        workspace, lwork, devInfo));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (info == 0)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, V.rows(), V.cols(), &one,
            A->data(), V.cols(), V.data(), V.ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
            cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, V.cols(), V.rows(),
            &One, V.data(), V.ld(), &Zero, A->data(), V.cols()));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
            cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(),
            V.cols(), workspace, lwork, devInfo));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        if (info == 0)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
                cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, V.rows(), V.cols(), &one,
                A->data(), V.cols(), V.data(), V.ld()));
        }
#ifdef CHASE_OUTPUT
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg",
            "choldegree: 2", 0);
#endif
    }

    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig_cublas, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream_orig_cusolver));
    if (owns_devInfo)
        CHECK_CUDA_ERROR(cudaFreeAsync(devInfo, stream));
    if (owns_workspace)
        CHECK_CUDA_ERROR(cudaFreeAsync(workspace, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    if (owns_A)
        delete A;

    return info;
}
/**
 * @brief Performs a shifted Cholesky-based QR factorization with a degree of 2.
 *
 * This function computes the shifted Cholesky decomposition for the matrix \( A
 * \), and uses it to perform a QR factorization of the given matrix \( V \) on
 * a GPU. The function performs additional operations with shifted matrix values
 * to enhance stability.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double, or
 * complex types).
 * @param cublas_handle The cuBLAS handle for managing operations.
 * @param cusolver_handle The cuSolver handle for solving linear systems and
 * decompositions.
 * @param V The input/output matrix \( V \) to be factored (on GPU).
 * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
 * @param lwork The size of the workspace (default: 0).
 * @param A Optional output matrix for the Cholesky factor (default: nullptr).
 *           If not provided, a new matrix is allocated.
 * @return int An error code (0 indicates success, non-zero indicates failure).
 */
template <typename T>
int shiftedcholQR2(cublasHandle_t cublas_handle,
                   cusolverDnHandle_t cusolver_handle,
                   chase::matrix::Matrix<T, chase::platform::GPU>& V,
                   T* workspace = nullptr, int lwork = 0,
                   chase::matrix::Matrix<T, chase::platform::GPU>* A = nullptr,
                   int* external_devInfo = nullptr)
{
    SCOPED_NVTX_RANGE();

    T one = T(1.0);
    T zero = T(0.0);

    chase::Base<T> One = Base<T>(1.0);
    chase::Base<T> Zero = Base<T>(0.0);
    chase::Base<T> shift_scale;
    int info = 1;
    bool owns_A = false;
    bool owns_workspace = false;
    bool owns_devInfo = false;

    cudaStream_t stream_orig_cublas = nullptr;
    cudaStream_t stream_orig_cusolver = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t evt_begin = nullptr;
    cudaEvent_t evt_end = nullptr;

    if (A == nullptr)
    {
        A = new chase::matrix::Matrix<T, chase::platform::GPU>(V.cols(), V.cols());
        owns_A = true;
    }

    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(cusolverDnGetStream(cusolver_handle, &stream_orig_cusolver));
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig_cublas));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream));

    if (workspace == nullptr)
    {
        lwork = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(),
                V.cols(), &lwork));
        CHECK_CUDA_ERROR(cudaMallocAsync((void**)&workspace, sizeof(T) * lwork, stream));
        owns_workspace = true;
    }

    cublasOperation_t transa;
    if (sizeof(T) == sizeof(Base<T>))
    {
        transa = CUBLAS_OP_T;
    }
    else
    {
        transa = CUBLAS_OP_C;
    }

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
        cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, V.cols(), V.rows(), &One,
        V.data(), V.ld(), &Zero, A->data(), V.cols()));

    chase::Base<T>* d_nrmf = nullptr;
    CHECK_CUDA_ERROR(cudaMallocAsync((void**)&d_nrmf, sizeof(chase::Base<T>), stream));
    chase::linalg::internal::cuda::absTrace(*A, d_nrmf, &stream);
#ifdef CHASE_OUTPUT
    bool return_shift_enabled = false;
    if(chase::GetLogger().GetLevel() >= chase::LogLevel::Info)
    {
        return_shift_enabled = true;
    }
    chase::Base<T> nrmf_host = 0.0;
    if(return_shift_enabled)
    {       
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&nrmf_host, d_nrmf, sizeof(chase::Base<T>),
                                        cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        if constexpr (std::is_same<chase::Base<T>, float>::value)
        {
            shift_scale = static_cast<chase::Base<T>>(10.0) * nrmf_host *
                          std::numeric_limits<chase::Base<T>>::epsilon();
        }
        else
        {
            shift_scale = std::sqrt(static_cast<double>(V.rows())) * nrmf_host *
                          std::numeric_limits<chase::Base<T>>::epsilon();
        }
    }
#endif
    chase::linalg::internal::cuda::shiftDiagonalFromDeviceShift(
        A, d_nrmf, &stream, V.rows());
    CHECK_CUDA_ERROR(cudaFreeAsync(d_nrmf, stream));

    int* devInfo = external_devInfo;
    if (devInfo == nullptr)
    {
        CHECK_CUDA_ERROR(cudaMallocAsync((void**)&devInfo, sizeof(int), stream));
        owns_devInfo = true;
    }

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
        cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(), V.cols(),
        workspace, lwork, devInfo));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (info == 0)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
            cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, V.rows(), V.cols(), &one, A->data(), V.cols(),
            V.data(), V.ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
            cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, V.cols(), V.rows(), &One,
            V.data(), V.ld(), &Zero, A->data(), V.cols()));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
            cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(), V.cols(),
            workspace, lwork, devInfo));

        CHECK_CUDA_ERROR(cudaMemcpyAsync(&info, devInfo, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

        if (info == 0)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
                cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, V.rows(), V.cols(), &one, A->data(), V.cols(),
                V.data(), V.ld()));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(
                cublas_handle, CUBLAS_FILL_MODE_UPPER, transa, V.cols(), V.rows(), &One,
                V.data(), V.ld(), &Zero, A->data(), V.cols()));

            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(
                cusolver_handle, CUBLAS_FILL_MODE_UPPER, V.cols(), A->data(), V.cols(),
                workspace, lwork, devInfo));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(
                cublas_handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, V.rows(), V.cols(), &one, A->data(), V.cols(),
                V.data(), V.ld()));
        }
    }

    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig_cublas, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig_cublas));
    CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolver_handle, stream_orig_cusolver));
    if (owns_devInfo)
        CHECK_CUDA_ERROR(cudaFreeAsync(devInfo, stream));
    if (owns_workspace)
        CHECK_CUDA_ERROR(cudaFreeAsync(workspace, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    if (owns_A)
        delete A;

#ifdef CHASE_OUTPUT
    std::ostringstream oss;
    oss << std::setprecision(2)
        << "\ncholdegree: 2, shift computed on device (scale=" << shift_scale
        << ", nrmf(device->host)=" << nrmf_host
        << ", final_shift=" << (nrmf_host * shift_scale) << ")\n";
    chase::GetLogger().Log(chase::LogLevel::Info, "linalg", oss.str(), 0);
#endif
    return info;
}

/**
 * @brief Performs the Householder QR decomposition on a matrix \( V \) using
 * cuSolver and cuBLAS.
 *
 * This function computes the QR decomposition of a matrix \( V \) using the
 * Householder transformation. It performs the required operations using
 * cuSolver and cuBLAS on the GPU.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double, or
 * complex types).
 * @param cusolver_handle The cuSolver handle for solving linear systems and
 * decompositions.
 * @param V The input matrix \( V \) to be factored (on GPU).
 * @param d_tau Pointer to the array storing Householder reflectors (on GPU).
 * @param devInfo Pointer to an integer storing the result of the computation
 * (on GPU).
 * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
 * @param lwork The size of the workspace (default: 0).
 */
template <typename T>
void houseHoulderQR(cusolverDnHandle_t cusolver_handle,
                    chase::matrix::Matrix<T, chase::platform::GPU>& V, T* d_tau,
                    int* devInfo, T* workspace = nullptr, int lwork = 0)
{
    SCOPED_NVTX_RANGE();

    if (workspace == nullptr)
    {
        int lwork_geqrf = 0;
        int lwork_orgqr = 0;
        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
                cusolver_handle, V.rows(), V.cols(), V.data(), V.ld(),
                &lwork_geqrf));

        CHECK_CUSOLVER_ERROR(
            chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
                cusolver_handle, V.rows(), V.cols(), V.cols(), V.data(), V.ld(),
                d_tau, &lwork_orgqr));

        lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
    }

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf(
        cusolver_handle, V.rows(), V.cols(), V.data(), V.ld(), d_tau, workspace,
        lwork, devInfo));

    CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr(
        cusolver_handle, V.rows(), V.cols(), V.cols(), V.data(), V.ld(), d_tau,
        workspace, lwork, devInfo));
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
