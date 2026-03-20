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
#include <cstring>
#include <chrono>
#include <iostream>

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
#include "grid/nccl_utils.hpp"
#include "linalg/internal/cuda/lanczos_kernels.hpp"
#endif

namespace chase
{
namespace linalg
{
namespace internal
{
/**
 * @brief Executes the Lanczos algorithm to generate a tridiagonal matrix
 * representation.
 *
 * This function performs the Lanczos algorithm, which is used to estimate
 * the upper bound of spectra of symmetric/Hermitian matrix.
 * The algorithm is iteratively applied to the matrix H, where the input
 * matrix `H` is a square matrix of size `N x N`. The Lanczos algorithm
 * builds an orthonormal basis of the Krylov subspace, and the resulting
 * tridiagonal matrix is diagonalized using the `t_stemr` function.
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
 * @throws std::runtime_error if the matrix `H` is not square or if `H` and `V`
 * are not in the same MPI grid.
 */
template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::pseudo_hermitian_lanczos(
    cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
    MatrixType& H, InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* Tau,
    chase::Base<typename MatrixType::value_type>* ritzV)
{
    using T = typename MatrixType::value_type;
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

    // Use default stream (nullptr) so performance.hpp CUDA events see Lanczos work
    cudaStream_t stream = nullptr;

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss; 
        oss << "[GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS]: ENABLED, using NCCL + batched dot/AXPY/scale (DEVICE mode)\n";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), H.getMpiGrid()->get_myRank());
    }
#endif
    // ========================================================================
    // GPU-RESIDENT VERSION: Fully batched kernels + NCCL collectives
    // Uses DEVICE pointer mode with batchedScale for vector normalization
    // ========================================================================

    // Get NCCL communicator from grid
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();


    // GPU-resident buffers for scalar results
    using RealT = chase::Base<T>;
    T* d_alpha;
    T* d_beta;
    RealT* d_real_alpha;
    RealT* d_real_beta;
    RealT* d_real_beta_prev;  // 1/sqrt(real_dot) for alpha scaling, no host round-trip
    T* d_beta_neg;           // -beta_prev for k>0 AXPY, allocated once
    cudaMalloc(&d_alpha, numvec * sizeof(T));
    cudaMalloc(&d_beta, numvec * sizeof(T));
    cudaMalloc(&d_real_alpha, numvec * sizeof(RealT));
    cudaMalloc(&d_real_beta, numvec * sizeof(RealT));
    cudaMalloc(&d_real_beta_prev, numvec * sizeof(RealT));
    cudaMalloc(&d_beta_neg, numvec * sizeof(T));
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;
    cudaMalloc(&d_d, M * numvec * sizeof(RealT));
    cudaMalloc(&e_d, M * numvec * sizeof(RealT));

    // Host buffers (RealT only: for LAPACK d/e and final scalars)
    std::vector<RealT> real_beta(numvec);
    std::vector<RealT> real_alpha(numvec);
#else
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss; 
        oss << "[ORIGINAL MPI PSEUDO-HERMITIAN LANCZOS]: ENABLED, using MPI_Allreduce\n";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), H.getMpiGrid()->get_myRank());
    }
#endif
    std::vector<chase::Base<T>> real_beta(numvec);
    std::vector<chase::Base<T>> real_alpha(numvec);
#endif

    std::vector<chase::Base<T>> d(M * numvec);
    std::vector<chase::Base<T>> e(M * numvec);

    auto lanczos_start = std::chrono::high_resolution_clock::now();

    // Enable fine-grain CUDA event timing only when high-verbosity logging is requested.
    bool enable_fine_timers = false;
#ifdef CHASE_OUTPUT
    enable_fine_timers =
        chase::GetLogger().GetLevel() >= chase::LogLevel::Trace;
#endif

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // Include fused kernel utilities (no host T for alpha/beta)
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedScale;
    using chase::linalg::internal::cuda::batchedScaleTwo;
    using chase::linalg::internal::cuda::batchedSqrt;
    using chase::linalg::internal::cuda::getRealPart;
    using chase::linalg::internal::cuda::copyRealToT;
    using chase::linalg::internal::cuda::realReciprocal;
    using chase::linalg::internal::cuda::copyRealReciprocalToT;
    using chase::linalg::internal::cuda::copyRealNegateToT;
    using chase::linalg::internal::cuda::scaleComplexByRealNegate;
    
    cublasSetStream(cublas_handle, stream);
#else
    std::vector<T> alpha(numvec, T(1.0));
    std::vector<T> beta(numvec, T(0.0));
#endif

    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto Sv = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    // Fine-grain timing via CUDA events (no sync in hot path)
    auto matvec_time_us = 0.0;
    auto redist_time_us = 0.0;
    auto sync_alpha_time_us = 0.0;
    auto sync_beta_time_us = 0.0;
    auto alpha_step_time_us = 0.0;
    auto beta_step_time_us = 0.0;
    auto init_beta_time_us = 0.0;
    std::vector<cudaEvent_t> ev_matvec_start;
    std::vector<cudaEvent_t> ev_matvec_end;
    std::vector<cudaEvent_t> ev_redist_start;
    std::vector<cudaEvent_t> ev_redist_end;
    std::vector<cudaEvent_t> ev_alpha_start;
    std::vector<cudaEvent_t> ev_alpha_end;
    std::vector<cudaEvent_t> ev_beta_start;
    std::vector<cudaEvent_t> ev_beta_end;
    cudaEvent_t ev_init_beta_start = nullptr, ev_init_beta_end = nullptr;
    if (enable_fine_timers)
    {
        ev_matvec_start.resize(M + 1);
        ev_matvec_end.resize(M + 1);
        ev_redist_start.resize(M + 1);
        ev_redist_end.resize(M + 1);
        for (std::size_t i = 0; i <= M; i++)
        {
            cudaEventCreate(&ev_matvec_start[i]);
            cudaEventCreate(&ev_matvec_end[i]);
            cudaEventCreate(&ev_redist_start[i]);
            cudaEventCreate(&ev_redist_end[i]);
        }
#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        cudaEventCreate(&ev_init_beta_start);
        cudaEventCreate(&ev_init_beta_end);
        ev_alpha_start.resize(M);
        ev_alpha_end.resize(M);
        ev_beta_start.resize(M);
        ev_beta_end.resize(M);
        for (std::size_t i = 0; i < M; i++)
        {
            cudaEventCreate(&ev_alpha_start[i]);
            cudaEventCreate(&ev_alpha_end[i]);
            cudaEventCreate(&ev_beta_start[i]);
            cudaEventCreate(&ev_beta_end[i]);
        }
#endif
    }

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld());

    if (enable_fine_timers)
        cudaEventRecord(ev_matvec_start[0], stream);
    chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
        cublas_handle, &One, H, v_1, &Zero, v_w);
    if (enable_fine_timers)
        cudaEventRecord(ev_matvec_end[0], stream);

    if (enable_fine_timers)
        cudaEventRecord(ev_redist_start[0], stream);
    v_w.redistributeImplAsync(&v_2, &stream);
    if (enable_fine_timers)
        cudaEventRecord(ev_redist_end[0], stream);

    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), numvec,
                                           v_2.l_data(), v_2.l_ld(),
                                           Sv.l_data(), Sv.l_ld());

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv, stream);

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // ========================================================================
    // INITIAL NORMALIZATION: real(v_1·Sv) on device, Allreduce RealT, beta on device
    // ========================================================================
    if (enable_fine_timers)
        cudaEventRecord(ev_init_beta_start, stream);
    batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                     v_1.l_rows(), numvec, v_1.l_ld(), false, &stream);
    getRealPart(d_beta, d_real_beta, static_cast<int>(numvec), &stream);
    auto comm_time_us = 0.0;
    auto comm_start = std::chrono::high_resolution_clock::now();
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_beta, d_real_beta, numvec, ncclSum, nccl_comm, &stream));
    auto comm_end = std::chrono::high_resolution_clock::now();
    comm_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                        comm_end - comm_start)
                        .count();
    batchedSqrt(d_real_beta, static_cast<int>(numvec), &stream);
    realReciprocal(d_real_beta, d_real_beta_prev, static_cast<int>(numvec), &stream);
    copyRealReciprocalToT(d_real_beta, d_beta, static_cast<int>(numvec), &stream);
    batchedScale(d_beta, v_1.l_data(), v_1.l_rows(), numvec, v_1.l_ld(), &stream);
    batchedScale(d_beta, v_2.l_data(), v_2.l_rows(), numvec, v_2.l_ld(), &stream);
    if (enable_fine_timers)
        cudaEventRecord(ev_init_beta_end, stream);
#else
    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_1.l_rows(), v_1.l_data() + i * v_1.l_ld(), 1,
            Sv.l_data() + i * Sv.l_ld(), 1, &beta[i]));
    }

    for (auto i = 0; i < numvec; i++)
    {
        real_beta[i] = std::real(beta[i]);
    }

    MPI_Allreduce(MPI_IN_PLACE, real_beta.data(), numvec,
                  chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                  H.getMpiGrid()->get_col_comm());

    for (auto i = 0; i < numvec; i++)
    {
        beta[i] = One / std::sqrt(real_beta[i]);
    }

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_1.l_rows(), &beta[i],
            v_1.l_data() + i * v_1.l_ld(), 1));
    }

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_2.l_rows(), &beta[i],
            v_2.l_data() + i * v_2.l_ld(), 1));
    }
#endif

    // ========================================================================
    // MAIN LANCZOS LOOP
    // Following EXACT algorithm structure as original MPI version
    // ========================================================================
    
    for (std::size_t k = 0; k < M; k = k + 1)
    {
        // Store v_1 into V (output)
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(),
                       v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice, 0);
        }

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        // Step 1: real(v_2·Sv) on device, Allreduce RealT; alpha = -real_alpha*beta_prev on device
        if (enable_fine_timers)
            cudaEventRecord(ev_alpha_start[k], stream);
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha,
                         v_2.l_rows(), numvec, v_2.l_ld(), false, &stream);
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        comm_start = std::chrono::high_resolution_clock::now();
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, numvec, ncclSum, nccl_comm, &stream));
        comm_end = std::chrono::high_resolution_clock::now();
        comm_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            comm_end - comm_start)
                            .count();
        copyRealToT(d_real_alpha, d_alpha, static_cast<int>(numvec), &stream);
        scaleComplexByRealNegate(d_alpha, d_real_beta_prev, static_cast<int>(numvec), &stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(),
                   v_1.l_rows(), numvec, v_1.l_ld(), &stream);
        // Store diagonal d[k] = -real_alpha*beta_prev on device (deferred D2H to avoid sync per iter)
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(d_d + (k + M * i), d_real_alpha + i, sizeof(RealT),
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (enable_fine_timers)
            cudaEventRecord(ev_alpha_end[k], stream);
#else
        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, v_2.l_rows(), v_2.l_data() + i * v_2.l_ld(), 1,
                Sv.l_data() + i * Sv.l_ld(), 1, &alpha[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = std::real(alpha[i]);
        }

        MPI_Allreduce(MPI_IN_PLACE, real_alpha.data(), numvec,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -real_alpha[i] * beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_1.l_rows(), &alpha[i],
                v_1.l_data() + i * v_1.l_ld(), 1, v_2.l_data() + i * v_2.l_ld(),
                1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha[i]);
        }
#endif

        // Step 4: Check if last iteration
        if (k == M - 1)
            break;

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        // Step 5: v_2 += (-sqrt(real_dot)) * v_0 = (-d_real_beta) * v_0 (same as ref: beta = -1/beta so coeff = -sqrt)
        copyRealNegateToT(d_real_beta, d_beta_neg, static_cast<int>(numvec), &stream);
        batchedAxpy(d_beta_neg, v_0.l_data(), v_2.l_data(),
                   v_0.l_rows(), numvec, v_0.l_ld(), &stream);
#else
        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -One / beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_0.l_rows(), &beta[i],
                v_0.l_data() + i * v_0.l_ld(), 1, v_2.l_data() + i * v_2.l_ld(),
                1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -beta[i];
        }
#endif

        v_1.swap(v_0);
        v_1.swap(v_2);

        std::size_t ev_idx = k + 1;
        if (enable_fine_timers)
            cudaEventRecord(ev_matvec_start[ev_idx], stream);
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);
        if (enable_fine_timers)
            cudaEventRecord(ev_matvec_end[ev_idx], stream);

        if (enable_fine_timers)
            cudaEventRecord(ev_redist_start[ev_idx], stream);
        v_w.redistributeImplAsync(&v_2, &stream);
        if (enable_fine_timers)
            cudaEventRecord(ev_redist_end[ev_idx], stream);

        chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), numvec,
                                               v_2.l_data(), v_2.l_ld(),
                                               Sv.l_data(), Sv.l_ld());

        chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv, stream);

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        // Step 9: real(v_1·Sv) on device, Allreduce, sqrt; e[k]=sqrt; beta_prev=1/sqrt on device
        if (enable_fine_timers)
            cudaEventRecord(ev_beta_start[k], stream);
        batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                         v_1.l_rows(), numvec, v_1.l_ld(), false, &stream);
        getRealPart(d_beta, d_real_beta, static_cast<int>(numvec), &stream);
        comm_start = std::chrono::high_resolution_clock::now();
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_beta, d_real_beta, numvec, ncclSum, nccl_comm, &stream));
        comm_end = std::chrono::high_resolution_clock::now();
        comm_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            comm_end - comm_start)
                            .count();
        batchedSqrt(d_real_beta, static_cast<int>(numvec), &stream);
        realReciprocal(d_real_beta, d_real_beta_prev, static_cast<int>(numvec), &stream);
        copyRealReciprocalToT(d_real_beta, d_beta, static_cast<int>(numvec), &stream);
        batchedScaleTwo(d_beta, v_1.l_data(), v_2.l_data(),
                       v_1.l_rows(), numvec, v_1.l_ld(), &stream);
        // Store e[k] on device (deferred D2H)
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(e_d + (k + M * i), d_real_beta + i, sizeof(RealT),
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (enable_fine_timers)
            cudaEventRecord(ev_beta_end[k], stream);
#else
        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, v_1.l_rows(), v_1.l_data() + i * v_1.l_ld(), 1,
                Sv.l_data() + i * Sv.l_ld(), 1, &beta[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            real_beta[i] = std::real(beta[i]);
        }

        MPI_Allreduce(MPI_IN_PLACE, real_beta.data(), numvec,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            real_beta[i] = std::sqrt(real_beta[i]);
        }

        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = real_beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = One / real_beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, v_1.l_rows(), &beta[i],
                v_1.l_data() + i * v_1.l_ld(), 1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, v_2.l_rows(), &beta[i],
                v_2.l_data() + i * v_2.l_ld(), 1));
        }
#endif
    }

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           v_1.l_data(), v_1.l_ld(), V.l_data(),
                                           V.l_ld(),&stream);

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // Single D2H for d and e (was per-iteration sync; now one sync after loop)
    cudaStreamSynchronize(stream);
#endif

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    if (enable_fine_timers)
    {
        // Single D2H for d and e (was per-iteration sync; now one sync after loop)
        float ev_ms = 0.0f;
        cudaEventElapsedTime(&ev_ms, ev_init_beta_start, ev_init_beta_end);
        init_beta_time_us = ev_ms * 1000.0;
        for (std::size_t i = 0; i <= M; i++)
        {
            cudaEventElapsedTime(&ev_ms, ev_matvec_start[i], ev_matvec_end[i]);
            matvec_time_us += ev_ms * 1000.0;
            cudaEventElapsedTime(&ev_ms, ev_redist_start[i], ev_redist_end[i]);
            redist_time_us += ev_ms * 1000.0;
        }
        for (std::size_t i = 0; i < M; i++)
        {
            cudaEventElapsedTime(&ev_ms, ev_alpha_start[i], ev_alpha_end[i]);
            alpha_step_time_us += ev_ms * 1000.0;
        }
        for (std::size_t i = 0; i < M - 1; i++)
        {
            cudaEventElapsedTime(&ev_ms, ev_beta_start[i], ev_beta_end[i]);
            beta_step_time_us += ev_ms * 1000.0;
        }
        for (std::size_t i = 0; i <= M; i++)
        {
            cudaEventDestroy(ev_matvec_start[i]);
            cudaEventDestroy(ev_matvec_end[i]);
            cudaEventDestroy(ev_redist_start[i]);
            cudaEventDestroy(ev_redist_end[i]);
        }
        cudaEventDestroy(ev_init_beta_start);
        cudaEventDestroy(ev_init_beta_end);
        for (std::size_t i = 0; i < M; i++)
        {
            cudaEventDestroy(ev_alpha_start[i]);
            cudaEventDestroy(ev_alpha_end[i]);
            cudaEventDestroy(ev_beta_start[i]);
            cudaEventDestroy(ev_beta_end[i]);
        }
    }
    cudaMemcpy(d.data(), d_d, M * numvec * sizeof(RealT), cudaMemcpyDeviceToHost);
    cudaMemcpy(e.data(), e_d, M * numvec * sizeof(RealT), cudaMemcpyDeviceToHost);
    cudaFree(d_d);
    cudaFree(e_d);
    // Clean up GPU resources
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_real_alpha);
    cudaFree(d_real_beta);
    cudaFree(d_real_beta_prev);
    cudaFree(d_beta_neg);
#endif

    auto lapack_start = std::chrono::high_resolution_clock::now();
    
    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    chase::Base<T> ul = 0;
    chase::Base<T> ll = 0;
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
    auto lapack_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(lapack_end - lapack_start);

    *upperb = ritzv[M - 1];

    // End-to-end time matching performance.hpp Lanczos scope (GPU+NCCL+D2H+LAPACK)
    auto lanczos_end = std::chrono::high_resolution_clock::now();
    auto lanczos_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(lanczos_end - lanczos_start);
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss;
        double lanczos_ms = lanczos_duration.count() / 1000.0;
        double comm_ms = 0.0;//comm_time_us / 1000.0;
        double redist_ms = redist_time_us / 1000.0;
        double matvec_ms = matvec_time_us / 1000.0;
        double sync_alpha_ms = sync_alpha_time_us / 1000.0;
        double sync_beta_ms = sync_beta_time_us / 1000.0;
#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        double init_beta_ms = init_beta_time_us / 1000.0;
        double alpha_step_ms = alpha_step_time_us / 1000.0;
        double beta_step_ms = beta_step_time_us / 1000.0;
        double other_ms = lanczos_ms - comm_ms - redist_ms - matvec_ms
            - init_beta_ms - alpha_step_ms - beta_step_ms
            - sync_alpha_ms - sync_beta_ms;
        oss << "[PSEUDO-HERMITIAN LANCZOS TIMING] (NCCL, multi-vector)\n"
            << "  numvec: " << numvec << ", M: " << M << "\n"
            << "  Lanczos total (matches performance.hpp): " << lanczos_ms << " ms\n"
            << "    Matvec (H*v, events):      " << matvec_ms << " ms\n"
            << "    Redistribute:              " << redist_ms << " ms\n"
            << "    Init norm (dot+scale):     " << init_beta_ms << " ms\n"
            << "    Alpha step (dot+AXPY+d[k]): " << alpha_step_ms << " ms\n"
            << "    Beta step (dot+scale+e[k]): " << beta_step_ms << " ms\n"
            << "    NCCL AllReduce (CPU wall): " << comm_ms << " ms\n"
            << "    Sync+D2H (alpha, d[k]):    " << sync_alpha_ms << " ms\n"
            << "    Sync+D2H (beta, e[k]):     " << sync_beta_ms << " ms\n"
            << "    Other (lacpy, flip, swap): " << other_ms << " ms\n"
            << "  LAPACK t_stemr (CPU) total: " << lapack_duration.count() / 1000.0
            << " ms\n"
            << "  Avg LAPACK per solve: "
            << lapack_duration.count() / (double)numvec / 1000.0 << " ms\n";
#else
        double other_ms = lanczos_ms - comm_ms - redist_ms - matvec_ms - sync_alpha_ms - sync_beta_ms;
        oss << "[PSEUDO-HERMITIAN LANCZOS TIMING] (NCCL, multi-vector)\n"
            << "  numvec: " << numvec << ", M: " << M << "\n"
            << "  Lanczos total (matches performance.hpp): " << lanczos_ms << " ms\n"
            << "    NCCL AllReduce time:       " << comm_ms << " ms\n"
            << "    Redistribute:              " << redist_ms << " ms\n"
            << "    Matvec (H*v, events):      " << matvec_ms << " ms\n"
            << "    Sync+D2H (alpha, d[k]):    " << sync_alpha_ms << " ms\n"
            << "    Sync+D2H (beta, e[k]):     " << sync_beta_ms << " ms\n"
            << "    Other (BLAS, no sync):     " << other_ms << " ms\n"
            << "  LAPACK t_stemr (CPU) total: " << lapack_duration.count() / 1000.0
            << " ms\n"
            << "  Avg LAPACK per solve: "
            << lapack_duration.count() / (double)numvec / 1000.0 << " ms\n";
#endif
        chase::GetLogger().Log(chase::LogLevel::Trace, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif
}

/**
 * @brief Lanczos algorithm for eigenvalue computation (simplified version).
 *
 * This version of the Lanczos algorithm is a simplified version that computes
 * only the upper bound of the eigenvalue spectrum and does not compute
 * eigenvectors. It operates similarly to the full Lanczos algorithm but
 * omits the eigenvector computation step.
 *
 * @tparam MatrixType Type of the input matrix, defining the value type and
 * matrix operations.
 * @tparam InputMultiVectorType Type of the multi-vector for initial Lanczos
 * basis vectors.
 *
 * @param M Number of Lanczos iterations.
 * @param H The input matrix representing the system for which eigenvalues are
 * sought.
 * @param V Initial Lanczos vector; will be overwritten with orthonormalized
 * basis vectors.
 * @param upperb Pointer to a variable that stores the computed upper bound for
 * the largest eigenvalue.
 *
 * @throws std::runtime_error if the matrix `H` is not square or if `H` and `V`
 * are not in the same MPI grid.
 */
template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::pseudo_hermitian_lanczos(
    cublasHandle_t cublas_handle, std::size_t M, MatrixType& H,
    InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb)
{
    using T = typename MatrixType::value_type;
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

    std::vector<chase::Base<T>> d(M);
    std::vector<chase::Base<T>> e(M);

    // Use default stream (nullptr) so performance.hpp CUDA events see Lanczos work
    cudaStream_t stream = nullptr;
#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss; 
        oss << "[GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS (SINGLE)]: ENABLED, using NCCL + batched dot/AXPY/scale (DEVICE mode)";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), H.getMpiGrid()->get_myRank());
    }
#endif
    // GPU-resident setup
    using RealT = chase::Base<T>;
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();


    // Device buffers for scalar results
    T* d_alpha;
    T* d_beta;
    RealT* d_real_alpha;
    RealT* d_real_beta;
    RealT* d_real_beta_prev;
    T* d_beta_neg;
    RealT* d_d;
    RealT* e_d;
    cudaMalloc(&d_alpha, sizeof(T));
    cudaMalloc(&d_beta, sizeof(T));
    cudaMalloc(&d_real_alpha, sizeof(RealT));
    cudaMalloc(&d_real_beta, sizeof(RealT));
    cudaMalloc(&d_real_beta_prev, sizeof(RealT));
    cudaMalloc(&d_beta_neg, sizeof(T));
    cudaMalloc(&d_d, M * sizeof(RealT));
    cudaMalloc(&e_d, M * sizeof(RealT));
    
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedScale;
    using chase::linalg::internal::cuda::batchedScaleTwo;
    using chase::linalg::internal::cuda::batchedSqrt;
    using chase::linalg::internal::cuda::getRealPart;
    using chase::linalg::internal::cuda::copyRealToT;
    using chase::linalg::internal::cuda::realReciprocal;
    using chase::linalg::internal::cuda::copyRealReciprocalToT;
    using chase::linalg::internal::cuda::copyRealNegateToT;
    using chase::linalg::internal::cuda::scaleComplexByRealNegate;
    
    cublasSetStream(cublas_handle, stream);
    
    chase::Base<T> real_alpha;
    chase::Base<T> real_beta;
#else
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss; 
        oss << "[ORIGINAL MPI PSEUDO-HERMITIAN LANCZOS (SINGLE)]: ENABLED, using MPI_Allreduce";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), H.getMpiGrid()->get_myRank());
    }
#endif
    chase::Base<T> real_alpha;
    chase::Base<T> real_beta;
#endif

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto Sv = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), 1, V.l_data(),
                                           V.l_ld(), v_1.l_data(), v_1.l_ld());

    chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
        cublas_handle, &One, H, v_1, &Zero, v_w);

    auto redist_time_us = 0.0;
    v_w.redistributeImplAsync(&v_2, &stream);

    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1, v_2.l_data(),
                                           v_2.l_ld(), Sv.l_data(), Sv.l_ld());

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv, stream);

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // Init: real(v_1·Sv) on device, Allreduce, beta = 1/sqrt on device
    batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                     v_1.l_rows(), 1, v_1.l_ld(), false, &stream);
    getRealPart(d_beta, d_real_beta, 1, &stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_beta, d_real_beta, 1, ncclSum, nccl_comm, &stream));
    batchedSqrt(d_real_beta, 1, &stream);
    realReciprocal(d_real_beta, d_real_beta_prev, 1, &stream);
    copyRealReciprocalToT(d_real_beta, d_beta, 1, &stream);
    batchedScale(d_beta, v_1.l_data(), v_1.l_rows(), 1, v_1.l_ld(), &stream);
    batchedScale(d_beta, v_2.l_data(), v_2.l_rows(), 1, v_2.l_ld(), &stream);
#else
    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
        cublas_handle, v_1.l_rows(), v_1.l_data(), 1, Sv.l_data(), 1, &beta));

    real_beta = std::real(beta);

    MPI_Allreduce(MPI_IN_PLACE, &real_beta, 1,
                  chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                  H.getMpiGrid()->get_col_comm());

    beta = One / std::sqrt(real_beta);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
        cublas_handle, v_1.l_rows(), &beta, v_1.l_data(), 1));

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
        cublas_handle, v_2.l_rows(), &beta, v_2.l_data(), 1));
#endif

    for (std::size_t k = 0; k < M; k = k + 1)
    {
        cudaMemcpy(V.l_data() + k * V.l_ld(), v_1.l_data(),
                   v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice);

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        // Alpha: real(v_2·Sv) on device, Allreduce; alpha = -real_alpha*beta_prev on device
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha,
                         v_2.l_rows(), 1, v_2.l_ld(), false, &stream);
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, 1, ncclSum, nccl_comm, &stream));
        cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT), cudaMemcpyDeviceToDevice, stream);
        copyRealToT(d_real_alpha, d_alpha, 1, &stream);
        scaleComplexByRealNegate(d_alpha, d_real_beta_prev, 1, &stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(),
                   v_1.l_rows(), 1, v_1.l_ld(), &stream);
#else
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_2.l_rows(), v_2.l_data(), 1, Sv.l_data(), 1,
            &alpha));

        real_alpha = std::real(alpha);

        MPI_Allreduce(MPI_IN_PLACE, &real_alpha, 1,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        alpha = -real_alpha * beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
            cublas_handle, v_1.l_rows(), &alpha, v_1.l_data(), 1, v_2.l_data(),
            1));

        alpha = -alpha;

        d[k] = std::real(alpha);
#endif

        if (k == M - 1)
            break;

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        copyRealNegateToT(d_real_beta_prev, d_beta_neg, 1, &stream);
        batchedAxpy(d_beta_neg, v_0.l_data(), v_2.l_data(),
                   v_0.l_rows(), 1, v_0.l_ld(), &stream);
#else
        beta = -One / beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
            cublas_handle, v_0.l_rows(), &beta, v_0.l_data(), 1, v_2.l_data(),
            1));
        beta = -beta;
#endif

        v_1.swap(v_0);
        v_1.swap(v_2);

        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        v_w.redistributeImplAsync(&v_2, &stream);

        chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1,
                                               v_2.l_data(), v_2.l_ld(),
                                               Sv.l_data(), Sv.l_ld());

        chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv, stream);

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        // real(v_1·Sv) on device, Allreduce, sqrt; e[k]=sqrt; beta_prev=1/sqrt on device
        batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                         v_1.l_rows(), 1, v_1.l_ld(), false, &stream);
        getRealPart(d_beta, d_real_beta, 1, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_beta, d_real_beta, 1, ncclSum, nccl_comm, &stream));
        batchedSqrt(d_real_beta, 1, &stream);
        cudaMemcpyAsync(e_d + k, d_real_beta, sizeof(RealT), cudaMemcpyDeviceToDevice, stream);
        realReciprocal(d_real_beta, d_real_beta_prev, 1, &stream);
        copyRealReciprocalToT(d_real_beta, d_beta, 1, &stream);
        batchedScaleTwo(d_beta, v_1.l_data(), v_2.l_data(),
                       v_1.l_rows(), 1, v_1.l_ld(), &stream);
#else
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_1.l_rows(), v_1.l_data(), 1, Sv.l_data(), 1,
            &beta));
        real_beta = std::real(beta);

        MPI_Allreduce(MPI_IN_PLACE, &real_beta, 1,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        real_beta = std::sqrt(real_beta);

        e[k] = real_beta;

        beta = One / real_beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_1.l_rows(), &beta, v_1.l_data(), 1));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_2.l_rows(), &beta, v_2.l_data(), 1));
#endif
    }

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // Single D2H for d and e (was per-iteration sync; now one sync after loop)
    cudaStreamSynchronize(stream);
    cudaMemcpy(d.data(), d_d, M * sizeof(RealT), cudaMemcpyDeviceToHost);
    cudaMemcpy(e.data(), e_d, M * sizeof(RealT), cudaMemcpyDeviceToHost);
    for (std::size_t k = 0; k < M; k++)
        d[k] = -d[k];
    cudaFree(d_d);
    cudaFree(e_d);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_real_alpha);
    cudaFree(d_real_beta);
    cudaFree(d_real_beta_prev);
    cudaFree(d_beta_neg);
#endif

    auto lapack_start = std::chrono::high_resolution_clock::now();
    
    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    chase::Base<T> ul = 0;
    chase::Base<T> ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<chase::Base<T>> ritzv(M);

    lapackpp::t_stemr<chase::Base<T>>(
        LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
        &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);
    
    auto lapack_end = std::chrono::high_resolution_clock::now();
    auto lapack_duration = std::chrono::duration_cast<std::chrono::microseconds>(lapack_end - lapack_start);
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss; 
        oss << "[PSEUDO-HERMITIAN LANCZOS TIMING - SINGLE VECTOR] LAPACK t_stemr (CPU):";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), H.getMpiGrid()->get_myRank());
        oss << "  M: " << M;
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), H.getMpiGrid()->get_myRank());
        oss << "  Time: " << lapack_duration.count() / 1000.0 << " ms";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), H.getMpiGrid()->get_myRank());
    }
#endif
    *upperb = ritzv[M - 1];
}
} // namespace internal
} // namespace linalg
} // namespace chase
