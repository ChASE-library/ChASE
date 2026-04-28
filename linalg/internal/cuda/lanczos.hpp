// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/cuda_utils.hpp"
#include "Impl/chase_gpu/nvtx.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "linalg/internal/cuda/flipSign.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/matrix/matrix.hpp"

#include "linalg/internal/cuda/lanczos_kernels.hpp"

using namespace chase::linalg;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
/**
 * @brief Performs the Lanczos algorithm for eigenvalue computation on a GPU
 * using cuBLAS and CUDA.
 *
 * This function performs the Lanczos algorithm on a matrix `H` (of size `M`),
 * computing an orthonormal basis `V` and the tridiagonal matrix `T` for the
 * eigenvalue problem. It also computes Ritz values and estimates of the upper
 * bounds for the eigenvalues.
 *
 * @tparam T The data type of the matrix elements (e.g., float or double).
 * @param cublas_handle The cuBLAS handle to perform linear algebra operations
 * on the GPU.
 * @param M The number of Lanczos iterations.
 * @param numvec The number of runs of Lanczos.
 * @param H The matrix of size `M x M` that is the input for the Lanczos
 * algorithm.
 * @param V The matrix of size `M x numvec` that holds the orthonormal basis
 * vectors.
 * @param upperb Output parameter that will hold the upper bound for the Ritz
 * values.
 * @param ritzv Output array to hold the Ritz values computed during the Lanczos
 * algorithm.
 * @param Tau Output array to store the Tau values from the eigenvalue
 * decomposition.
 * @param ritzV Output array for storing the Ritz vectors.
 */
template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    SCOPED_NVTX_RANGE();
    if (M <= 0)
    {
        throw std::invalid_argument("lanczos: M, the number of Lanczos iterations, must be > 0");
    }

    if (numvec <= 0)
    {
        throw std::invalid_argument("lanczos: numvec, the number of Lanczos instances, must be >=1 ");
    }

    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // ========================================================================
    // GPU-RESIDENT VERSION: Same structure as NCCL Lanczos (batched kernels +
    // device-resident scalars), without NCCL or redistribution.
    // ========================================================================
    cudaStream_t stream_orig;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig));
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
    std::vector<RealT> d_tmp(M * numvec), e_tmp(M * numvec);

    using chase::linalg::internal::cuda::fusedNormSquaredNormalize;
    using chase::linalg::internal::cuda::fusedDotAxpyNegate;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedNormSquared;
    using chase::linalg::internal::cuda::normalizeVectors;

    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);
    std::vector<RealT> de_host(2 * M * numvec);
    std::vector<RealT> r_beta_host(numvec);

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);

    /////////////////////////////////////////////////////
    ////Lanczos internal stream wait for default stream//
    /////////////////////////////////////////////////////
    cudaEvent_t evt_begin;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));

    T* d_alpha = nullptr;
    RealT* d_real_alpha = nullptr;
    RealT* d_r_beta = nullptr;
    RealT* d_beta_prev = nullptr;
    T* d_beta_neg = nullptr;
    RealT* d_de = nullptr;
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_alpha, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_alpha, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_r_beta, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta_prev, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta_neg, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_de, 2 * M * numvec * sizeof(RealT), stream));
    d_d = d_de;
    e_d = d_de + M * numvec;
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_alpha, 0, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_alpha, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_r_beta, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_beta_prev, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(e_d, 0, M * numvec * sizeof(RealT), stream));
        
    CHECK_CUDA_ERROR(cudaMemsetAsync(v_0.data(), 0, v_0.ld() * numvec * sizeof(T), stream));

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, V.data(), V.ld(),
                                           v_1.data(), v_1.ld(), &stream);

    // Initial normalization:
    //   s_i = ||v_1^{(i)}||_2^2,   v_1^{(i)} <- v_1^{(i)} / sqrt(s_i)
    fusedNormSquaredNormalize(v_1.data(), d_real_alpha,
                              static_cast<int>(v_1.rows()),
                              static_cast<int>(numvec),
                              static_cast<int>(v_1.ld()), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                V.data() + k * V.ld(), v_1.data() + i * v_1.ld(),
                v_1.rows() * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
        

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(),
            static_cast<int>(numvec), H->cols(), &One, H->data(), H->ld(),
            v_1.data(), v_1.ld(), &Zero, v_2.data(), v_2.ld()));

        // Fused alpha update (per vector i):
        //   alpha_k^{(i)} = <v_1^{(i)}, w^{(i)}>
        //   w^{(i)} <- w^{(i)} - alpha_k^{(i)} v_1^{(i)}
        // Kernel stores -alpha internally and re-negates for recurrence consistency.
        fusedDotAxpyNegate(v_1.data(), v_2.data(), v_2.data(), v_1.data(),
                           d_alpha,
                           static_cast<int>(v_1.rows()),
                           static_cast<int>(numvec),
                           static_cast<int>(v_1.ld()), &stream);

        // Store diagonal entry:
        //   d(k, i) = Re(alpha_k^(i))
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k * numvec, d_real_alpha,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        // Previous-basis correction:
        //   w^{(i)} <- w^{(i)} - beta_{k-1}^{(i)} v_0^{(i)}
        // First iteration is a no-op because beta_{-1}=0.
        // Final-step recurrence correction:
        //   w^(i) <- w^(i) - beta_{M-2}^(i) v_0^(i)
        copyRealNegateToT(d_beta_prev, d_beta_neg, static_cast<int>(numvec),
                          &stream);
        batchedAxpy(d_beta_neg, v_0.data(), v_2.data(),
                    static_cast<int>(v_0.rows()), static_cast<int>(numvec),
                    static_cast<int>(v_0.ld()), &stream);

        // Beta computation:
        //   r_i = ||w^{(i)}||_2^2,   beta_k^{(i)} = sqrt(r_i)
        // Residual norm for upper-bound estimate:
        //   r_i = ||w^(i)||_2^2,  beta_{M-1}^(i) = sqrt(r_i)
        batchedNormSquared(v_2.data(), d_r_beta, static_cast<int>(v_2.rows()),
                           static_cast<int>(numvec), static_cast<int>(v_2.ld()),
                           &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_beta_prev, d_r_beta,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));
        batchedSqrt(d_beta_prev, static_cast<int>(numvec), &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(e_d + k * numvec, d_beta_prev,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        // Next Lanczos vector:
        //   v_{k+1}^{(i)} = w^{(i)} / beta_k^{(i)} = w^{(i)} / sqrt(r_i)
        normalizeVectors(v_2.data(), d_r_beta, static_cast<int>(v_2.rows()),
                        static_cast<int>(numvec), static_cast<int>(v_2.ld()),
                        &stream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);
    }

    {
        const std::size_t k = M - 1;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(),
            static_cast<int>(numvec), H->cols(), &One, H->data(), H->ld(),
            v_1.data(), v_1.ld(), &Zero, v_2.data(), v_2.ld()));

        // Final iteration alpha update:
        //   alpha_{M-1}^{(i)} = <v_1^{(i)}, w^{(i)}>,  w^{(i)} <- w^{(i)} - alpha_{M-1}^{(i)} v_1^{(i)}
        fusedDotAxpyNegate(v_1.data(), v_2.data(), v_2.data(), v_1.data(),
                           d_alpha, static_cast<int>(v_1.rows()),
                           static_cast<int>(numvec),
                           static_cast<int>(v_1.ld()), &stream);

        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k * numvec, d_real_alpha,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        copyRealNegateToT(d_beta_prev, d_beta_neg, static_cast<int>(numvec),
                          &stream);
        batchedAxpy(d_beta_neg, v_0.data(), v_2.data(),
                    static_cast<int>(v_0.rows()), static_cast<int>(numvec),
                    static_cast<int>(v_0.ld()), &stream);

        batchedNormSquared(v_2.data(), d_r_beta, static_cast<int>(v_2.rows()),
                           static_cast<int>(numvec), static_cast<int>(v_2.ld()),
                           &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_beta_prev, d_r_beta,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));
        batchedSqrt(d_beta_prev, static_cast<int>(numvec), &stream);
    }

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, v_1.data(), v_1.ld(),
                                           V.data(), V.ld(), &stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(de_host.data(), d_de,
                                     2 * M * numvec * sizeof(RealT),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(r_beta_host.data(), d_beta_prev,
                                numvec * sizeof(RealT), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    //std::copy(de_host.begin(), de_host.begin() + M * numvec, d.begin());
    //std::copy(de_host.begin() + M * numvec, de_host.end(), e.begin());

    // Transpose from device layout (k * numvec + i) to LAPACK layout (i * M + k).
    for (std::size_t i = 0; i < numvec; ++i)
    {
        for (std::size_t k = 0; k < M; ++k)
        {
            d_tmp[i * M + k] = de_host[k * numvec + i];
            e_tmp[i * M + k] = de_host[k * numvec + i + M * numvec];
        }
    }
    d = std::move(d_tmp);
    e = std::move(e_tmp);

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', static_cast<int>(M),
                          d.data() + i * M, e.data() + i * M, ul, ll, vl, vu,
                          &notneeded_m, ritzv + M * i, ritzV, static_cast<int>(M),
                          static_cast<int>(M), isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
    }

    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(r_beta_host[0]);
    for (auto i = 1; i < numvec; i++)
    {
        RealT max_val =
            std::max(std::abs(ritzv[i * M]), std::abs(ritzv[(i + 1) * M - 1])) +
            std::abs(r_beta_host[i]);
        *upperb = std::max(*upperb, max_val);
    }

    CHECK_CUDA_ERROR(cudaFree(d_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_real_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_r_beta));
    CHECK_CUDA_ERROR(cudaFree(d_beta_prev));
    CHECK_CUDA_ERROR(cudaFree(d_beta_neg));
    CHECK_CUDA_ERROR(cudaFree(d_de));
    cudaEvent_t evt_end;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

}

/**
 * @brief Performs the Lanczos algorithm on matrix H to compute the tridiagonal
 * matrix and eigenvalues.
 *
 * This version of the Lanczos algorithm is a simplified version that computes
 * only the upper bound of the eigenvalue spectrum and does not compute
 * eigenvectors. It operates similarly to the full Lanczos algorithm but
 * omits the eigenvector computation step.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 *
 * @param cublas_handle cuBLAS handle used to perform matrix operations.
 * @param M Number of Lanczos iterations.
 * @param H The input matrix \( H \) (square matrix).
 * @param V The input matrix \( V \) (Lanczos starting vector).
 * @param upperb Pointer to store the upper bound of the computed eigenvalues.
 *
 * @note The function modifies the input matrices `H` and `V` during the
 * computation. It computes the tridiagonal matrix and updates the eigenvalues
 * in `ritzv`, then stores the upper bound of the eigenvalues in the `upperb`
 * pointer.
 */
template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    SCOPED_NVTX_RANGE();

    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // ========================================================================
    // GPU-RESIDENT VERSION: Same structure as NCCL single-vector Lanczos.
    // ========================================================================
    cudaStream_t stream_orig;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig));
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));
    cudaEvent_t evt_begin;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));

    T* d_alpha = nullptr;
    RealT* d_real_alpha = nullptr;
    RealT* d_r_beta = nullptr;
    RealT* d_beta_prev = nullptr;
    T* d_beta_neg = nullptr;
    RealT* d_de = nullptr;
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_alpha, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_real_alpha, sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_r_beta, sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta_prev, sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_beta_neg, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_de, 2 * M * sizeof(RealT)));
    d_d = d_de;
    e_d = d_de + M;
    CHECK_CUDA_ERROR(cudaMemset(d_alpha, 0, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMemset(d_real_alpha, 0, sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMemset(d_r_beta, 0, sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMemset(d_beta_prev, 0, sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMemset(e_d, 0, M * sizeof(RealT)));

    std::vector<RealT> d(M);
    std::vector<RealT> e(M);
    std::vector<RealT> de_host(2 * M);
    RealT r_beta_host;

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);

    chase::linalg::internal::cuda::t_lacpy('A', N, 1, V.data(), V.ld(),
                                           v_1.data(), v_1.ld());

    using chase::linalg::internal::cuda::fusedNormSquaredNormalize;
    using chase::linalg::internal::cuda::fusedDotAxpyNegate;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedNormSquared;
    using chase::linalg::internal::cuda::normalizeVectors;

    // Initial normalization:
    //   s = ||v_1||_2^2,   v_1 <- v_1 / sqrt(s)
    fusedNormSquaredNormalize(v_1.data(), d_real_alpha,
                              static_cast<int>(v_1.rows()), 1,
                              static_cast<int>(v_1.ld()), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), 1, H->cols(),
            &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(),
            v_2.ld()));

        // Fused alpha update:
        //   alpha_k = <v_1, w>,   w <- w - alpha_k v_1
        // Kernel stores -alpha internally and re-negates for recurrence consistency.
        fusedDotAxpyNegate(v_1.data(), v_2.data(), v_2.data(), v_1.data(),
                           d_alpha,
                           static_cast<int>(v_1.rows()), 1,
                           static_cast<int>(v_1.ld()), &stream);
        // Store diagonal entry:
        //   d(k) = Re(alpha_k)
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        // Previous-basis correction:
        //   w <- w - beta_{k-1} v_0
        copyRealNegateToT(d_beta_prev, d_beta_neg, 1, &stream);
        batchedAxpy(d_beta_neg, v_0.data(), v_2.data(),
                    static_cast<int>(v_0.rows()), 1,
                    static_cast<int>(v_0.ld()), &stream);

        // Beta computation:
        //   r = ||w||_2^2,   beta_k = sqrt(r)
        batchedNormSquared(v_2.data(), d_r_beta, static_cast<int>(v_2.rows()),
                           1, static_cast<int>(v_2.ld()), &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_beta_prev, d_r_beta, sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));
        batchedSqrt(d_beta_prev, 1, &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(e_d + k, d_beta_prev, sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        // Next Lanczos vector:
        //   v_{k+1} = w / beta_k = w / sqrt(r)
        normalizeVectors(v_2.data(), d_r_beta, static_cast<int>(v_2.rows()), 1,
                        static_cast<int>(v_2.ld()), &stream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);
    }

    {
        const std::size_t k = M - 1;
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), 1, H->cols(),
            &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(),
            v_2.ld()));
        // Final iteration alpha update:
        //   alpha_{M-1} = <v_1, w>,  w <- w - alpha_{M-1} v_1
        fusedDotAxpyNegate(v_1.data(), v_2.data(), v_2.data(), v_1.data(),
                           d_alpha, static_cast<int>(v_1.rows()), 1,
                           static_cast<int>(v_1.ld()), &stream);
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));
    }

    CHECK_CUDA_ERROR(cudaMemcpyAsync(de_host.data(), d_de, 2 * M * sizeof(RealT),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&r_beta_host, d_beta_prev, sizeof(RealT),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (std::size_t k = 0; k < M; ++k)
    {
        d[k] = de_host[k];
        e[k] = de_host[M + k];
    }

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> ritzv(M);

    lapackpp::t_stemr<RealT>(
        LAPACK_COL_MAJOR, 'N', 'A', static_cast<int>(M), d.data(), e.data(),
        ul, ll, vl, vu, &notneeded_m, ritzv.data(), NULL, static_cast<int>(M),
        static_cast<int>(M), isuppz.data(), &tryrac);

    *upperb =
        std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
        std::abs(r_beta_host);

    CHECK_CUDA_ERROR(cudaFree(d_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_real_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_r_beta));
    CHECK_CUDA_ERROR(cudaFree(d_beta_prev));
    CHECK_CUDA_ERROR(cudaFree(d_beta_neg));
    CHECK_CUDA_ERROR(cudaFree(d_de));
    cudaEvent_t evt_end;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

}

/**
 * @brief Lanczos algorithm for eigenvalue computation of Pseudo-Hermitian
 * matrices.
 *
 * This function performs the Lanczos algorithm, which is used to estimate
 * the upper bound of spectra of the pseudo-Hermitian matrix H, i.e., where H is
 * pseudo-Hermitian and SH is Hermitian Positive Definite. The algorithm is
 * iteratively applied to the matrix H, where the input matrix `H` is a square
 * matrix of size `N x N`. The Lanczos algorithm builds an orthonormal basis of
 * the Krylov subspace, and the resulting tridiagonal matrix is diagonalized
 * using the `t_stemr` function. This pseudo-code of this implementation can be
 * found in `https://doi.org/10.1016/j.commatsci.2011.02.021`
 *
 * @tparam T The data type for the matrix elements (e.g., float, double).
 * @param cublas_handle cuBLAS handle used to perform matrix operations.
 * @param M The number of Lanczos iterations.
 * @param numvec The number of runs of Lanczos.
 * @param N The size of the input matrix `H`.
 * @param H The input matrix for the Lanczos algorithm (of size `N x N`).
 * @param V The input matrix used for storing vectors (of size `N x numvec`).
 * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
 * @param ritzv A pointer to store the Ritz eigenvalues.
 * @param Tau A pointer to store the computed Tau values.
 * @param ritzV A pointer to store the Ritz eigenvectors.
 */

template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    SCOPED_NVTX_RANGE();

    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // ========================================================================
    // GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS (numvec): Same structure as NCCL,
    // without NCCL or redistribution. Uses batched dot/axpy/scale and Sv.
    // ========================================================================
    cudaStream_t stream_orig;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig));    
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));

    using chase::linalg::internal::cuda::batchedScaleTwo;
    using chase::linalg::internal::cuda::pseudoHermitianInitBatched;
    using chase::linalg::internal::cuda::fusedDotScaleNegateAxpyPh;
    using chase::linalg::internal::cuda::lacpyFlipBatchedDot;

    std::vector<RealT> d_tmp(M * numvec), e_tmp(M * numvec);
    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);
    std::vector<RealT> de_host(2 * M * numvec);

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto Sv = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);

    /////////////////////////////////////////////////////
    ////Lanczos internal stream wait for default stream//
    /////////////////////////////////////////////////////
    cudaEvent_t evt_begin;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));

    T* d_alpha = nullptr;
    T* d_beta = nullptr;
    RealT* d_real_alpha = nullptr;
    RealT* d_real_beta = nullptr;
    RealT* d_real_beta_prev = nullptr;
    RealT* d_de = nullptr;
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_alpha, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_alpha, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_beta, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_beta_prev, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_de, 2 * M * numvec * sizeof(RealT), stream));
    d_d = d_de;
    e_d = d_de + M * numvec;
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_alpha, 0, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_beta, 0, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_alpha, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_beta, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_beta_prev, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(e_d, 0, M * numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(v_0.data(), 0, v_0.ld() * numvec * sizeof(T), stream));

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, V.data(), V.ld(),
                                           v_1.data(), v_1.ld(), &stream);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(),
        static_cast<int>(numvec), H->cols(), &One, H->data(), H->ld(),
        v_1.data(), v_1.ld(), &Zero, v_2.data(), v_2.ld()));
    // Pseudo-Hermitian initialization (per vector i):
    //   w^(i) = H v_1^(i),   Sv^(i) = S w^(i)
    //   beta_0^(i)^2 = <v_1^(i), Sv^(i)>,   d_real_beta_prev(i) <- 1 / beta_0^(i)
    //   v_1^(i), v_2^(i) are scaled so the recurrence starts in normalized form.
    pseudoHermitianInitBatched(v_1.data(), v_2.data(), Sv.data(), d_beta,
                               d_real_beta_prev, static_cast<int>(N),
                               static_cast<int>(numvec),
                               static_cast<int>(v_1.ld()), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                V.data() + k * V.ld(), v_1.data() + i * v_1.ld(),
                v_1.rows() * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }

        // Fused alpha step (per i):
        //   alpha_k^(i) = <v_2^(i), Sv^(i)>
        //   v_2^(i) <- v_2^(i) - alpha_k^(i) * (1 / beta_{k-1}^(i)) v_1^(i)
        fusedDotScaleNegateAxpyPh(v_2.data(), Sv.data(), d_alpha,
                                  v_1.data(), v_2.data(), d_real_beta_prev,
                                  static_cast<int>(v_2.rows()),
                                  static_cast<int>(numvec),
                                  static_cast<int>(v_1.ld()), &stream);
        // Store diagonal entry:
        //   d(k, i) = alpha_k^(i)
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k * numvec, d_real_alpha,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        // Three-term recurrence correction:
        //   v_2^(i) <- v_2^(i) - beta_{k-1}^(i) v_0^(i)
        // Since d_real_beta_prev = 1 / beta_{k-1}, first recover beta_{k-1}.
        realReciprocal(d_real_beta_prev, d_real_beta, static_cast<int>(numvec),
                       &stream);
        copyRealNegateToT(d_real_beta, d_beta, static_cast<int>(numvec),
                          &stream);
        batchedAxpy(d_beta, v_0.data(), v_2.data(),
                    static_cast<int>(v_0.rows()), static_cast<int>(numvec),
                    static_cast<int>(v_0.ld()), &stream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(),
            static_cast<int>(numvec), H->cols(), &One, H->data(), H->ld(),
            v_1.data(), v_1.ld(), &Zero, v_2.data(), v_2.ld()));
        // Build S*v_2 and off-diagonal coefficient (per i):
        //   Sv^(i) = S v_2^(i)
        //   beta_k^(i)^2 = <v_1^(i), Sv^(i)>
        lacpyFlipBatchedDot(v_2.data(), Sv.data(), v_1.data(), d_beta,
                            static_cast<int>(N), static_cast<int>(numvec),
                            static_cast<int>(v_2.ld()), static_cast<int>(Sv.ld()),
                            static_cast<int>(v_1.ld()), &stream);

        // Off-diagonal magnitude:
        //   beta_k^(i) = sqrt(beta_k^(i)^2)
        getRealPart(d_beta, d_real_beta, static_cast<int>(numvec), &stream);
        batchedSqrt(d_real_beta, static_cast<int>(numvec), &stream);
        // Store off-diagonal entry:
        //   e(k, i) = beta_k^(i), for k = 0..M-2
        CHECK_CUDA_ERROR(cudaMemcpyAsync(e_d + k * numvec, d_real_beta,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        // Normalize next recurrence state:
        //   beta_k^(i) = sqrt(beta_k^(i)^2)
        //   d_real_beta_prev(i) <- 1 / beta_k^(i)
        //   v_1^(i), v_2^(i) <- (1 / beta_k^(i)) * {v_1^(i), v_2^(i)}
        copyRealReciprocalToT(d_real_beta, d_beta, static_cast<int>(numvec),
                              &stream);
        realReciprocal(d_real_beta, d_real_beta_prev,
                       static_cast<int>(numvec), &stream);
        batchedScaleTwo(d_beta, v_1.data(), v_2.data(),
                        static_cast<int>(v_1.rows()), static_cast<int>(numvec),
                        static_cast<int>(v_1.ld()), &stream);
    }

    {
        const std::size_t k = M - 1;

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                V.data() + k * V.ld(), v_1.data() + i * v_1.ld(),
                v_1.rows() * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }

        // Final alpha step:
        //   alpha_{M-1}^(i) = <v_2^(i), Sv^(i)>
        //   v_2^(i) <- v_2^(i) - alpha_{M-1}^(i) * (1 / beta_{M-2}^(i)) v_1^(i)
        fusedDotScaleNegateAxpyPh(v_2.data(), Sv.data(), d_alpha,
                                  v_1.data(), v_2.data(), d_real_beta_prev,
                                  static_cast<int>(v_2.rows()),
                                  static_cast<int>(numvec),
                                  static_cast<int>(v_1.ld()), &stream);
        // Store final diagonal entry:
        //   d(M-1, i) = alpha_{M-1}^(i)
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k * numvec, d_real_alpha,
                                         numvec * sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));
    }

    chase::linalg::internal::cuda::t_lacpy('A', N, numvec, v_1.data(), v_1.ld(),
                                           V.data(), V.ld(), &stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(de_host.data(), d_de,
                                     2 * M * numvec * sizeof(RealT),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    for (std::size_t i = 0; i < numvec; ++i)
    {
        for (std::size_t k = 0; k < M; ++k)
        {
            d_tmp[i * M + k] = de_host[k * numvec + i];
            e_tmp[i * M + k] = de_host[M * numvec + k * numvec + i];
        }
    }
    d = std::move(d_tmp);
    e = std::move(e_tmp);

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', static_cast<int>(M),
                          d.data() + i * M, e.data() + i * M, ul, ll, vl, vu,
                          &notneeded_m, ritzv + M * i, ritzV, static_cast<int>(M),
                          static_cast<int>(M), isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
    }
    *upperb = ritzv[M - 1];

    CHECK_CUDA_ERROR(cudaFree(d_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_beta));
    CHECK_CUDA_ERROR(cudaFree(d_real_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_real_beta));
    CHECK_CUDA_ERROR(cudaFree(d_real_beta_prev));
    CHECK_CUDA_ERROR(cudaFree(d_de));
    cudaEvent_t evt_end;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

}

/**
 * @brief Lanczos algorithm for eigenvalue computation of pseudo-hermitian
 * matrices (simplified version).
 *
 * This version of the Lanczos algorithm is a simplified version that computes
 * only the upper bound of the eigenvalue spectrum of pseudo-hermitian matrices
 * and does not computei eigenvectors. It operates similarly to the full
 * Lanczos algorithm but omits the eigenvector computation step.
 *
 * @tparam T The data type for the matrix elements (e.g., float, double).
 * @param cublas_handle cuBLAS handle used to perform matrix operations.
 * @param M The number of Lanczos iterations.
 * @param H The input pseudo-hermitian matrix for the Lanczos algorithm (of size
 * `N x N`).
 * @param V The input matrix used for storing vectors (of size `N x 1`).
 * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
 */
template <typename T>
void lanczos(cublasHandle_t cublas_handle, std::size_t M,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    SCOPED_NVTX_RANGE();

    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // ========================================================================
    // GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS (single vector): batched kernels
    // with numvec=1, Sv and flip, no NCCL.
    // ========================================================================
    cudaStream_t stream_orig;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig));
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));


    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedScaleTwo;
    using chase::linalg::internal::cuda::pseudoHermitianInitSingle;
    using chase::linalg::internal::cuda::fusedDotScaleNegateAxpyPh;
    using chase::linalg::internal::cuda::lacpyFlipBatchedDot;
    using chase::linalg::internal::cuda::getRealPart;
    using chase::linalg::internal::cuda::realReciprocal;
    using chase::linalg::internal::cuda::copyRealNegateToT;
    using chase::linalg::internal::cuda::copyRealReciprocalToT;    

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto Sv = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);

    std::vector<RealT> d(M);
    std::vector<RealT> e(M);
    std::vector<RealT> de_host(2 * M);
    RealT real_beta_val;

    cudaEvent_t evt_begin;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));

    T* d_alpha = nullptr;
    T* d_beta = nullptr;
    RealT* d_real_alpha = nullptr;
    RealT* d_real_beta = nullptr;
    RealT* d_real_beta_prev = nullptr;
    RealT* d_de = nullptr;
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_alpha, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_alpha, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_beta, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_beta_prev, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_de, 2 * M * sizeof(RealT), stream));
    d_d = d_de;
    e_d = d_de + M;
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_alpha, 0, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_beta, 0, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_alpha, 0, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_beta, 0, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_beta_prev, 0, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(e_d, 0, M * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(v_0.data(), 0, v_0.ld() * sizeof(T), stream));




    chase::linalg::internal::cuda::t_lacpy('A', N, 1, V.data(), V.ld(),
                                           v_1.data(), v_1.ld(), &stream);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), 1, H->cols(), &One,
        H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(), v_2.ld()));
    // Pseudo-Hermitian initialization:
    //   w = H v_1,   Sv = S w
    //   beta_0^2 = <v_1, Sv>,   d_real_beta_prev <- 1 / beta_0
    //   v_1, v_2 are scaled to start the recurrence.
    pseudoHermitianInitSingle(v_1.data(), v_2.data(), Sv.data(), d_beta,
                              d_real_beta_prev, static_cast<int>(N),
                              static_cast<int>(v_1.ld()), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        CHECK_CUDA_ERROR(cudaMemcpyAsync(V.data() + k * V.ld(), v_1.data(),
                                         v_1.rows() * sizeof(T),
                                         cudaMemcpyDeviceToDevice, stream));

        // Fused alpha step:
        //   alpha_k = <v_2, Sv>
        //   v_2 <- v_2 - alpha_k * (1 / beta_{k-1}) v_1
        fusedDotScaleNegateAxpyPh(v_2.data(), Sv.data(), d_alpha,
                                  v_1.data(), v_2.data(), d_real_beta_prev,
                                  static_cast<int>(v_2.rows()), 1,
                                  static_cast<int>(v_1.ld()), &stream);
        // Store diagonal entry:
        //   d(k) = alpha_k
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));

        // Three-term recurrence correction:
        //   v_2 <- v_2 - beta_{k-1} v_0
        realReciprocal(d_real_beta_prev, d_real_beta, 1, &stream);
        copyRealNegateToT(d_real_beta, d_beta, 1, &stream);
        batchedAxpy(d_beta, v_0.data(), v_2.data(),
                    static_cast<int>(v_0.rows()), 1,
                    static_cast<int>(v_0.ld()), &stream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, H->rows(), 1, H->cols(),
            &One, H->data(), H->ld(), v_1.data(), v_1.ld(), &Zero, v_2.data(),
            v_2.ld()));
        // Build Sv and off-diagonal coefficient:
        //   Sv = S v_2,   beta_k^2 = <v_1, Sv>
        lacpyFlipBatchedDot(v_2.data(), Sv.data(), v_1.data(), d_beta,
                            static_cast<int>(N), 1,
                            static_cast<int>(v_2.ld()), static_cast<int>(Sv.ld()),
                            static_cast<int>(v_1.ld()), &stream);

        // Off-diagonal magnitude:
        //   beta_k = sqrt(beta_k^2)
        getRealPart(d_beta, d_real_beta, 1, &stream);
        batchedSqrt(d_real_beta, 1, &stream);
        // Store off-diagonal entry:
        //   e(k) = beta_k, for k = 0..M-2
        CHECK_CUDA_ERROR(cudaMemcpyAsync(e_d + k, d_real_beta, sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));
        // Normalize next recurrence state:
        //   beta_k = sqrt(beta_k^2),  d_real_beta_prev <- 1 / beta_k
        //   v_1, v_2 <- (1 / beta_k) * {v_1, v_2}
        copyRealReciprocalToT(d_real_beta, d_beta, 1, &stream);
        realReciprocal(d_real_beta, d_real_beta_prev, 1, &stream);
        batchedScaleTwo(d_beta, v_1.data(), v_2.data(),
                        static_cast<int>(v_1.rows()), 1,
                        static_cast<int>(v_1.ld()), &stream);
    }

    {
        const std::size_t k = M - 1;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(V.data() + k * V.ld(), v_1.data(),
                                         v_1.rows() * sizeof(T),
                                         cudaMemcpyDeviceToDevice, stream));
        // Final alpha step:
        //   alpha_{M-1} = <v_2, Sv>
        //   v_2 <- v_2 - alpha_{M-1} * (1 / beta_{M-2}) v_1
        fusedDotScaleNegateAxpyPh(v_2.data(), Sv.data(), d_alpha,
                                  v_1.data(), v_2.data(), d_real_beta_prev,
                                  static_cast<int>(v_2.rows()), 1,
                                  static_cast<int>(v_1.ld()), &stream);
        // Store final diagonal entry:
        //   d(M-1) = alpha_{M-1}
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT),
                                         cudaMemcpyDeviceToDevice, stream));
    }

    chase::linalg::internal::cuda::t_lacpy('A', N, 1, v_1.data(), v_1.ld(),
                                           V.data(), V.ld(), &stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(de_host.data(), d_de, 2 * M * sizeof(RealT),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&real_beta_val, d_real_beta_prev,
                                     sizeof(RealT), cudaMemcpyDeviceToHost,
                                     stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    for (std::size_t k = 0; k < M; ++k)
    {
        d[k] = de_host[k];
        e[k] = de_host[M + k];
    }

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> ritzv(M);

    lapackpp::t_stemr<RealT>(
        LAPACK_COL_MAJOR, 'N', 'A', static_cast<int>(M), d.data(), e.data(),
        ul, ll, vl, vu, &notneeded_m, ritzv.data(), NULL,
        static_cast<int>(M), static_cast<int>(M), isuppz.data(), &tryrac);
    *upperb = ritzv[M - 1];

    CHECK_CUDA_ERROR(cudaFree(d_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_beta));
    CHECK_CUDA_ERROR(cudaFree(d_real_alpha));
    CHECK_CUDA_ERROR(cudaFree(d_real_beta));
    CHECK_CUDA_ERROR(cudaFree(d_real_beta_prev));
    CHECK_CUDA_ERROR(cudaFree(d_de));
    cudaEvent_t evt_end;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

}
} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
