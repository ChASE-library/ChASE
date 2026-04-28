// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "../typeTraits.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/internal/nccl/nccl_kernels.hpp"
#include "algorithm/logger.hpp"
#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>

#include "grid/nccl_utils.hpp"
#include "linalg/internal/cuda/lanczos_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
/**
 * @brief Dispatch to the correct Lanczos procedure based on Matrix Type
 *
 * This function dispatches to the correct Lanczos procedure based on the matrix
 * type
 *
 * @tparam MatrixType Type of the input matrix, defining the value type and
 * matrix operations.
 * @tparam InputMultiVectorType Type of the multi-vector for initial Lanczos
 * basis vectors.
 *
 * @param M Number of Lanczos iterations.
 * @param numvec The number of runs of Lanczos.
 * @param H The input matrix representing the system for which eigenvalues are
 * sought.
 * @param V Initial Lanczos vectors; will be overwritten with orthonormalized
 * basis vectors.
 * @param upperb Pointer to a variable that stores the computed upper bound for
 * the largest eigenvalue.
 * @param ritzv Array storing the resulting Ritz values (eigenvalues).
 * @param Tau Array of values representing convergence estimates.
 * @param ritzV Vector storing the Ritz vectors associated with computed
 * eigenvalues.
 *
 */
template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos_dispatch(
    cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
    MatrixType& H, InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* Tau,
    chase::Base<typename MatrixType::value_type>* ritzV)
{
    using T = typename MatrixType::value_type;

    if constexpr (std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                          T, chase::platform::GPU>>::value ||
                  std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                          T, chase::platform::GPU>>::value)
    {
        cuda_nccl::pseudo_hermitian_lanczos(cublas_handle, M, numvec, H, V,
                                            upperb, ritzv, Tau, ritzV);
    }
    else
    {
        cuda_nccl::lanczos(cublas_handle, M, numvec, H, V, upperb, ritzv, Tau,
                           ritzV);
    }
}

/**
 * @brief Dispatch to the correct Simplified Lanczos procedure based on Matrix
 * Type
 *
 * This function dispatches to the correct Simplified Lanczos procedure based on
 * the matrix type
 *
 * @tparam MatrixType Type of the input matrix, defining the value type and
 * matrix operations.
 * @tparam InputMultiVectorType Type of the multi-vector for initial Lanczos
 * basis vectors.
 *
 * @param M Number of Lanczos iterations.
 * @param H The input matrix representing the system for which eigenvalues are
 * sought.
 * @param V Initial Lanczos vectors; will be overwritten with orthonormalized
 * basis vectors.
 * @param upperb Pointer to a variable that stores the computed upper bound for
 * the largest eigenvalue.
 *
 */
template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos_dispatch(
    cublasHandle_t cublas_handle, std::size_t M, MatrixType& H,
    InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb)
{
    using T = typename MatrixType::value_type;

    if constexpr (std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                          T, chase::platform::GPU>>::value ||
                  std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                          T, chase::platform::GPU>>::value)
    {
        cuda_nccl::pseudo_hermitian_lanczos(cublas_handle, M, H, V, upperb);
    }
    else
    {
        cuda_nccl::lanczos(cublas_handle, M, H, V, upperb);
    }
}

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos(cublasHandle_t cublas_handle, std::size_t M,
                        std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV)
{
    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    if (H.g_cols() != H.g_rows())
    {
        throw std::runtime_error("Lanczos requires matrix to be squared");
    }

    if (H.getMpiGrid() != V.getMpiGrid())
    {
        throw std::runtime_error("Lanczos requires H and V in same MPI grid");
    }
    if (H.g_rows() != V.g_rows())
    {
        throw std::runtime_error("Lanczos H and V have same number of rows");
    }

#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "[GPU-RESIDENT LANCZOS]: ENABLED, using NCCL + Fused Kernels" << std::endl;
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif
    // ========================================================================
    // GPU-RESIDENT VERSION: Fused kernels + NCCL collectives
    // ========================================================================

    // Get NCCL communicator from grid
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();
    using chase::linalg::internal::cuda::batchedNormSquared;
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedAxpyThenNegate;
    using chase::linalg::internal::cuda::normalizeVectors;
    using chase::linalg::internal::cuda::batchedSqrt;
    using chase::linalg::internal::cuda::copyRealNegateToT;
    using chase::linalg::internal::cuda::getRealPart;
    
    cudaStream_t stream_orig;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig));
    cudaStream_t stream = nullptr;
    //!!! SEEMS there is bug while using nonblocking stream, need to investigate!!!
    //!!! For now, using default stream (nullptr)!!!
    //CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));

    // Host buffers (only for LAPACK; filled by single D2H after loop)
    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);
    std::vector<RealT> d_tmp(M * numvec), e_tmp(M * numvec);

    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    /////////////////////////////////////////////////////
    ////Lanczos internal stream wait for default stream//
    /////////////////////////////////////////////////////
    cudaEvent_t evt_begin;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));
    
    T* d_alpha;
    RealT* d_real_alpha;
    RealT* d_r_beta;
    RealT* d_beta_prev;   // previous iteration beta (after sqrt), for k>0 axpy
    T* d_beta_neg;       // -beta for batched axpy, reused every k>0
    RealT* d_d;          // device diagonal (deferred D2H)
    RealT* e_d;          // device off-diagonal (deferred D2H)
    
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_alpha, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_alpha, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_r_beta, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta_prev, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta_neg, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_d, M * numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&e_d, M * numvec * sizeof(RealT), stream));

    // Initialize to zero to prevent garbage values
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_alpha, 0, numvec * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_alpha, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_r_beta, 0, numvec * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_beta_prev, 0, numvec * sizeof(RealT), stream));    
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_beta_neg, 0, numvec * sizeof(T), stream));
    cudaMemsetAsync(d_d, 0, M * numvec * sizeof(RealT), stream);
    cudaMemsetAsync(e_d, 0, M * numvec * sizeof(RealT), stream);
    cudaMemsetAsync(v_0.l_data(), 0, v_0.l_rows() * numvec * sizeof(T), stream);

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld(), &stream);
//    CHECK_CUDA_ERROR(cudaPeekAtLastError());  // launch/config errors
//    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));  // runtime errors

    // ========================================================================
    // Initial normalization (GPU-resident)
    // ========================================================================
    
    // Compute local norms (per vector i):
    //   s_i^(loc) = ||v_1^(i)||_2^2
    batchedNormSquared(v_1.l_data(), d_real_alpha, v_1.l_rows(), numvec, v_1.l_ld(), &stream);

    // Global reduction across the column communicator:
    //   s_i = sum_r s_i^(loc,r)
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_alpha, d_real_alpha, numvec, ncclSum, nccl_comm, &stream));

    // Initial normalization:
    //   v_1^(i) <- v_1^(i) / sqrt(s_i)
    normalizeVectors(v_1.l_data(), d_real_alpha, v_1.l_rows(), numvec, v_1.l_ld(), &stream);

    // ========================================================================
    // Main Lanczos iteration loop
    // ========================================================================
    
    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(),
                       v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }

        // Krylov expansion (distributed matvec):
        //   w^(i) = H v_k^(i)
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);
        
        // Redistribution to local panel layout:
        //   v_2^(i) <- redistribute(w^(i))
        // Next kernels run on same stream, preserving this dependency.
        v_w.redistributeImplAsync(&v_2, &stream);

        // Local alpha contribution (stored negated for fused update):
        //   alpha_k^(i,loc) = <v_1^(i), w^(i)>_loc
        //   d_alpha(i) <- -alpha_k^(i,loc)
        batchedDotProduct(v_1.l_data(), v_2.l_data(), d_alpha, 
                         v_1.l_rows(), numvec, v_1.l_ld(), true, &stream);  // true = negate

        // Global alpha reduction:
        //   d_alpha(i) <- -alpha_k^(i) = -sum_r alpha_k^(i,loc,r)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_alpha, d_alpha, numvec, ncclSum, nccl_comm, &stream));

        // Fused alpha correction (per i):
        //   w^(i) <- w^(i) - alpha_k^(i) v_1^(i)
        // and store alpha_k^(i) sign-corrected for tridiagonal diagonal entry d(k,i).
        batchedAxpyThenNegate(d_alpha, v_1.l_data(), v_2.l_data(),
                              v_1.l_rows(), numvec, v_1.l_ld(), &stream);

        // Store tridiagonal diagonal entry:
        //   d(k, i) = Re(alpha_k^(i))
        getRealPart(d_alpha, d_real_alpha, numvec, &stream);
        cudaMemcpyAsync(d_d + k * numvec, d_real_alpha, numvec * sizeof(RealT),
                       cudaMemcpyDeviceToDevice, stream);

        // Previous-basis correction:
        //   w^(i) <- w^(i) - beta_{k-1}^(i) v_0^(i)
        // First iteration is a no-op because beta_{-1}=0.
        copyRealNegateToT(d_beta_prev, d_beta_neg, numvec, &stream);
        batchedAxpy(d_beta_neg, v_0.l_data(), v_2.l_data(),
                    v_0.l_rows(), numvec, v_0.l_ld(), &stream);

        // Local beta contribution:
        //   r_i^(loc) = ||w^(i)||_2^2_loc
        batchedNormSquared(v_2.l_data(), d_r_beta, v_2.l_rows(), numvec, v_2.l_ld(), &stream);

        // Global beta norm:
        //   r_i = sum_r r_i^(loc,r)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_r_beta, d_r_beta, numvec, ncclSum, nccl_comm, &stream));

        // Beta value:
        //   beta_k^(i) = sqrt(r_i)
        // Keep beta_k on device for next recurrence step.
        cudaMemcpyAsync(d_beta_prev, d_r_beta, numvec * sizeof(RealT),
                       cudaMemcpyDeviceToDevice, stream);
        batchedSqrt(d_beta_prev, numvec, &stream);

        // Next Lanczos vector:
        //   v_{k+1}^(i) = w^(i) / beta_k^(i) = w^(i) / sqrt(r_i)
        normalizeVectors(v_2.l_data(), d_r_beta, v_2.l_rows(), numvec, v_2.l_ld(), &stream);

        // Store tridiagonal off-diagonal entry:
        //   e(k, i) = beta_k^(i),  for k = 0..M-2
        cudaMemcpyAsync(e_d + k * numvec, d_beta_prev, numvec * sizeof(RealT),
                        cudaMemcpyDeviceToDevice, stream);

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    {
        const std::size_t k = M - 1;

        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(),
                            v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }

        // Krylov expansion (distributed matvec):
        //   w^(i) = H v_{M-1}^(i)
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        // Redistribution to local panel layout:
        //   v_2^(i) <- redistribute(w^(i))
        v_w.redistributeImplAsync(&v_2, &stream);

        // Local alpha contribution (stored negated for fused update):
        //   alpha_{M-1}^(i,loc) = <v_1^(i), w^(i)>_loc
        //   d_alpha(i) <- -alpha_{M-1}^(i,loc)
        batchedDotProduct(v_1.l_data(), v_2.l_data(), d_alpha,
                         v_1.l_rows(), numvec, v_1.l_ld(), true, &stream);

        // Global alpha reduction:
        //   d_alpha(i) <- -alpha_{M-1}^(i) = -sum_r alpha_{M-1}^(i,loc,r)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_alpha, d_alpha, numvec, ncclSum, nccl_comm, &stream));

        // Fused alpha correction (per i):
        //   w^(i) <- w^(i) - alpha_{M-1}^(i) v_1^(i)
        // and store alpha_{M-1}^(i) sign-corrected for d(M-1,i).
        batchedAxpyThenNegate(d_alpha, v_1.l_data(), v_2.l_data(),
                              v_1.l_rows(), numvec, v_1.l_ld(), &stream);

        // Store final tridiagonal diagonal entry:
        //   d(M-1, i) = Re(alpha_{M-1}^(i))
        getRealPart(d_alpha, d_real_alpha, numvec, &stream);
        cudaMemcpyAsync(d_d + k * numvec, d_real_alpha, numvec * sizeof(RealT),
                        cudaMemcpyDeviceToDevice, stream);
    }

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           v_1.l_data(), v_1.l_ld(), V.l_data(),
                                           V.l_ld(), &stream);

    // Ensure all GPU operations complete; single D2H for d and e (was per-iteration sync)
    cudaMemcpyAsync(d.data(), d_d, M * numvec * sizeof(RealT), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(e.data(), e_d, M * numvec * sizeof(RealT), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_d, stream);
    cudaFreeAsync(e_d, stream);

    // Last beta per run for upper-bound (e_d layout: e[k*numvec+i]; last row is k=M-2)
    std::vector<RealT> r_beta_last(numvec);
    for (auto i = 0; i < numvec; i++)
        r_beta_last[i] = e[(M - 2) * numvec + i];

    // Transpose d, e from device layout (k*numvec+i) to LAPACK layout (i*M+k)
    for (auto i = 0; i < numvec; i++)
        for (std::size_t k = 0; k < M; k++)
        {
            d_tmp[i * M + k] = d[k * numvec + i];
            e_tmp[i * M + k] = e[k * numvec + i];
        }
    d = std::move(d_tmp);
    e = std::move(e_tmp);

    // ========================================================================
    // Eigenvalue computation (LAPACK on CPU)
    // ========================================================================
    
    auto lapack_start = std::chrono::high_resolution_clock::now();
    
    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', M, d.data() + i * M,
                          e.data() + i * M, ul, ll, vl, vu, &notneeded_m,
                          ritzv + M * i, ritzV, M, M, isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
        {
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
        }
    }
    
    auto lapack_end = std::chrono::high_resolution_clock::now();
    auto lapack_duration = std::chrono::duration_cast<std::chrono::microseconds>(lapack_end - lapack_start);
#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "[LANCZOS TIMING] LAPACK t_stemr (CPU sequential):" << std::endl;
        oss << "  numvec: " << numvec << ", M: " << M << std::endl;
        oss << "  Total time: " << lapack_duration.count() / 1000.0 << " ms" << std::endl;
        oss << "  Avg per solve: " << lapack_duration.count() / (double)numvec / 1000.0 << " ms" << std::endl;
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif
    RealT max;
    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(r_beta_last[0]);

    for (auto i = 1; i < numvec; i++)
    {
        max =
            std::max(std::abs(ritzv[i * M]), std::abs(ritzv[(i + 1) * M - 1])) +
            std::abs(r_beta_last[i]);
        *upperb = std::max(max, *upperb);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_alpha);
    cudaFree(d_real_alpha);
    cudaFree(d_r_beta);
    cudaFree(d_beta_prev);
    cudaFree(d_beta_neg);

    cudaEvent_t evt_end;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig));
//    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::lanczos(cublasHandle_t cublas_handle, std::size_t M,
                        MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb)
{
    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    if (H.g_cols() != H.g_rows())
    {
        throw std::runtime_error("Lanczos requires matrix to be squared");
    }

    if (H.getMpiGrid() != V.getMpiGrid())
    {
        throw std::runtime_error("Lanczos requires H and V in same MPI grid");
    }

    if (H.g_rows() != V.g_rows())
    {
        throw std::runtime_error("Lanczos H and V have same number of rows");
    }
    
    // Use default stream (nullptr) so performance.hpp CUDA events see Lanczos work.
    // TODO(nonblocking-stream): investigate ordering hazards and re-enable a
    // dedicated non-blocking stream once dependencies are fully validated.
    cudaStream_t stream_orig;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_orig));
    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream));

#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "[GPU-RESIDENT LANCZOS (single-vec)]: ENABLED, using NCCL + Fused Kernels" << std::endl;
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif
    // ========================================================================
    // GPU-RESIDENT VERSION: Fused kernels + NCCL collectives
    // ========================================================================

    // Get NCCL communicator from grid
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();

    // ========================================================================
    // GPU-Resident: Allocate device buffers (once, no allocation in loop)
    // ========================================================================
    T* d_alpha;
    RealT* d_real_alpha;
    RealT* d_r_beta;
    RealT* d_beta_prev;   // previous iteration beta (after sqrt), for k>0 axpy
    T* d_beta_neg;       // -beta for axpy, reused every k>0
    RealT* d_d;          // device diagonal (deferred D2H)
    RealT* e_d;          // device off-diagonal (deferred D2H)
    /////////////////////////////////////////////////////
    ////Lanczos internal stream wait for default stream//
    /////////////////////////////////////////////////////
    cudaEvent_t evt_begin;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_begin));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_begin, stream_orig));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, evt_begin, 0));

    CHECK_CUDA_ERROR(cudaMallocAsync(&d_alpha, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_real_alpha, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_r_beta, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta_prev, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_beta_neg, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&d_d, M * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMallocAsync(&e_d, M * sizeof(RealT), stream));

    CHECK_CUDA_ERROR(cudaMemsetAsync(d_alpha, 0, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_alpha, 0, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_r_beta, 0, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_beta_prev, 0, sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_beta_neg, 0, sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_d, 0, M * sizeof(RealT), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(e_d, 0, M * sizeof(RealT), stream));

    // Host buffers (filled by single D2H after loop)
    std::vector<RealT> d(M);
    std::vector<RealT> e(M);

    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    CHECK_CUDA_ERROR(cudaMemsetAsync(v_0.l_data(), 0, v_0.l_rows() * sizeof(T), stream));

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), 1, V.l_data(),
                                           V.l_ld(), v_1.l_data(), v_1.l_ld(),
                                           &stream);

    // ========================================================================
    // Initial normalization (GPU-resident)
    // ========================================================================
    
    using chase::linalg::internal::cuda::batchedNormSquared;
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedAxpyThenNegate;
    using chase::linalg::internal::cuda::normalizeVectors;
    using chase::linalg::internal::cuda::batchedSqrt;
    using chase::linalg::internal::cuda::copyRealNegateToT;
    using chase::linalg::internal::cuda::getRealPart;
    
    // Local norm:
    //   s^(loc) = ||v_1||_2^2_loc
    batchedNormSquared(v_1.l_data(), d_real_alpha, v_1.l_rows(), 1, v_1.l_ld(), &stream);

    // Global norm:
    //   s = sum_r s^(loc,r)
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_alpha, d_real_alpha, 1, ncclSum, nccl_comm, &stream));

    // Initial normalization:
    //   v_1 <- v_1 / sqrt(s)
    normalizeVectors(v_1.l_data(), d_real_alpha, v_1.l_rows(), 1, v_1.l_ld(), &stream);

    // ========================================================================
    // Main Lanczos iteration loop
    // ========================================================================
    
    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        // Krylov expansion:
        //   w = H v_k
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        // Redistribution to local layout:
        //   v_2 <- redistribute(w)
        v_w.redistributeImplAsync(&v_2, &stream);

        // Local alpha contribution (negated storage):
        //   d_alpha <- -<v_1, w>_loc
        batchedDotProduct(v_1.l_data(), v_2.l_data(), d_alpha,
                         v_1.l_rows(), 1, v_1.l_ld(), true, &stream);  // true = negate

        // Global alpha:
        //   d_alpha <- -alpha_k = -sum_r <v_1, w>_loc,r
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_alpha, d_alpha, 1, ncclSum, nccl_comm, &stream));

        // Fused alpha correction:
        //   w <- w - alpha_k v_1
        // and store sign-correct alpha_k for d(k).
        batchedAxpyThenNegate(d_alpha, v_1.l_data(), v_2.l_data(),
                             v_1.l_rows(), 1, v_1.l_ld(), &stream);

        // Store diagonal entry:
        //   d(k) = Re(alpha_k)
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT), cudaMemcpyDeviceToDevice, stream);

        // Previous-basis correction:
        //   w <- w - beta_{k-1} v_0
        // First iteration is a no-op because beta_{-1}=0.
        copyRealNegateToT(d_beta_prev, d_beta_neg, 1, &stream);
        batchedAxpy(d_beta_neg, v_0.l_data(), v_2.l_data(),
                    v_0.l_rows(), 1, v_0.l_ld(), &stream);

        // Local beta contribution:
        //   r^(loc) = ||w||_2^2_loc
        batchedNormSquared(v_2.l_data(), d_r_beta, v_2.l_rows(), 1, v_2.l_ld(), &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_r_beta, d_r_beta, 1, ncclSum, nccl_comm, &stream));

        // Beta value:
        //   beta_k = sqrt(sum_r r^(loc,r))
        cudaMemcpyAsync(d_beta_prev, d_r_beta, sizeof(RealT), cudaMemcpyDeviceToDevice, stream);
        batchedSqrt(d_beta_prev, 1, &stream);

        // Next Lanczos vector:
        //   v_{k+1} = w / beta_k = w / sqrt(r)
        normalizeVectors(v_2.l_data(), d_r_beta, v_2.l_rows(), 1, v_2.l_ld(), &stream);

        // Store off-diagonal entry:
        //   e(k) = beta_k, for k = 0..M-2
        cudaMemcpyAsync(e_d + k, d_beta_prev, sizeof(RealT),
                        cudaMemcpyDeviceToDevice, stream);

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    {
        const std::size_t k = M - 1;
        cudaMemcpyAsync(V.l_data() + k * V.l_ld(), v_1.l_data(),
                        v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice,
                        stream);

        // Krylov expansion:
        //   w = H v_{M-1}
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        // Redistribution to local layout:
        //   v_2 <- redistribute(w)
        v_w.redistributeImplAsync(&v_2, &stream);

        // Local alpha contribution (negated storage):
        //   d_alpha <- -<v_1, w>_loc
        batchedDotProduct(v_1.l_data(), v_2.l_data(), d_alpha,
                          v_1.l_rows(), 1, v_1.l_ld(), true, &stream);

        // Global alpha:
        //   d_alpha <- -alpha_{M-1} = -sum_r <v_1, w>_loc,r
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_alpha, d_alpha, 1, ncclSum, nccl_comm, &stream));

        // Fused alpha correction:
        //   w <- w - alpha_{M-1} v_1
        batchedAxpyThenNegate(d_alpha, v_1.l_data(), v_2.l_data(),
                              v_1.l_rows(), 1, v_1.l_ld(), &stream);

        // Store final diagonal entry:
        //   d(M-1) = Re(alpha_{M-1})
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // Ensure all GPU operations complete; single D2H for d and e (was per-iteration sync)
    cudaMemcpyAsync(d.data(), d_d, M * sizeof(RealT), cudaMemcpyDeviceToHost,
                    stream);
    cudaMemcpyAsync(e.data(), e_d, M * sizeof(RealT), cudaMemcpyDeviceToHost,
                    stream);
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_d, stream);
    cudaFreeAsync(e_d, stream);

    // ========================================================================
    // Eigenvalue computation (LAPACK on CPU)
    // ========================================================================
    
    auto lapack_start = std::chrono::high_resolution_clock::now();
    
    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> ritzv(M);

    // Validate tridiagonal matrix elements before calling LAPACK
#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        bool has_nan = false;
        for (std::size_t i = 0; i < M; i++) {
            if (std::isnan(d[i]) || std::isinf(d[i])) {
                oss << "[LANCZOS WARNING] d[" << i << "] = " << d[i] << std::endl;
                has_nan = true;
            }
            if (i < M - 1 && (std::isnan(e[i]) || std::isinf(e[i]))) {
                oss << "[LANCZOS WARNING] e[" << i << "] = " << e[i] << std::endl;
                has_nan = true;
            }
        }
        if (has_nan)
            oss << "[LANCZOS WARNING] NaN/Inf detected in tridiagonal matrix!" << std::endl;
        if (!oss.str().empty())
            chase::GetLogger().Log(chase::LogLevel::Warn, "linalg", oss.str(),
                                   H.getMpiGrid()->get_myRank());
    }
#endif

    lapackpp::t_stemr<RealT>(
        LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
        &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);
    
    auto lapack_end = std::chrono::high_resolution_clock::now();
    auto lapack_duration = std::chrono::duration_cast<std::chrono::microseconds>(lapack_end - lapack_start);
    
#ifdef CHASE_OUTPUT
    {
        std::ostringstream oss;
        oss << "[LANCZOS TIMING - SINGLE VECTOR] LAPACK t_stemr (CPU):" << std::endl;
        oss << "  M: " << M << std::endl;
        oss << "  Time: " << lapack_duration.count() / 1000.0 << " ms" << std::endl;
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif
    RealT last_beta = (M > 1) ? e[M - 2] : RealT(0);
    *upperb =
        std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) + std::abs(last_beta);

    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_alpha);
    cudaFree(d_real_alpha);
    cudaFree(d_r_beta);
    cudaFree(d_beta_prev);
    cudaFree(d_beta_neg);

    cudaEvent_t evt_end;
    CHECK_CUDA_ERROR(cudaEventCreate(&evt_end));
    CHECK_CUDA_ERROR(cudaEventRecord(evt_end, stream));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_orig, evt_end, 0));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_begin));
    CHECK_CUDA_ERROR(cudaEventDestroy(evt_end));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, stream_orig));
    // stream is nullptr (default stream) in this overload; do not destroy it.
}

} // namespace internal
} // namespace linalg
} // namespace chase
