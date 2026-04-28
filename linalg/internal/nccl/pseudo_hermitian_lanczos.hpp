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
#include "linalg/internal/nccl/pseudo_hermitian_lanczos_diag.hpp"
#include "linalg/internal/nccl/nccl_kernels.hpp"
#include <chrono>
#include <cstring>
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

    std::vector<chase::Base<T>> d(M * numvec);
    std::vector<chase::Base<T>> e(M * numvec);

    auto lanczos_start = std::chrono::high_resolution_clock::now();

    bool enable_fine_timers = false;
#ifdef CHASE_OUTPUT
    enable_fine_timers =
        chase::GetLogger().GetLevel() >= chase::LogLevel::Trace;
#endif

    auto matvec_time_us = 0.0;
    auto redist_time_us = 0.0;
    auto sync_alpha_time_us = 0.0;
    auto sync_beta_time_us = 0.0;
    auto alpha_step_time_us = 0.0;
    auto beta_step_time_us = 0.0;
    auto init_beta_time_us = 0.0;

    // ========================================================================
    // GPU-resident Lanczos: NCCL + batched dot / AXPY / scale (device scalars)
    // ========================================================================
    cudaStream_t stream = nullptr;

#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss;
        oss << "[GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS]: ENABLED, using NCCL "
               "+ batched dot/AXPY/scale (DEVICE mode)\n";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif

    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();

    using RealT = chase::Base<T>;
    T* d_alpha;
    T* d_beta;
    RealT* d_real_alpha;
    RealT* d_real_beta;
    RealT* d_real_beta_prev;
    T* d_beta_neg;
    cudaMalloc(&d_alpha, numvec * sizeof(T));
    cudaMalloc(&d_beta, numvec * sizeof(T));
    cudaMalloc(&d_real_alpha, numvec * sizeof(RealT));
    cudaMalloc(&d_real_beta, numvec * sizeof(RealT));
    cudaMalloc(&d_real_beta_prev, numvec * sizeof(RealT));
    cudaMalloc(&d_beta_neg, numvec * sizeof(RealT));
    RealT* d_d = nullptr;
    RealT* e_d = nullptr;
    cudaMalloc(&d_d, M * numvec * sizeof(RealT));
    cudaMalloc(&e_d, M * numvec * sizeof(RealT));

    std::vector<RealT> real_beta(numvec);
    std::vector<RealT> real_alpha(numvec);

    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedScale;
    using chase::linalg::internal::cuda::batchedScaleTwo;
    using chase::linalg::internal::cuda::batchedSqrt;
    using chase::linalg::internal::cuda::copyRealNegateToT;
    using chase::linalg::internal::cuda::copyRealReciprocalToT;
    using chase::linalg::internal::cuda::copyRealToT;
    using chase::linalg::internal::cuda::getRealPart;
    using chase::linalg::internal::cuda::realReciprocal;
    using chase::linalg::internal::cuda::scaleComplexByRealNegate;

    cublasSetStream(cublas_handle, stream);

    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto Sv = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

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
    }

    // Copy initial block of vectors:
    //   v_1^(i) <- V(:, i),  i = 1..numvec
    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld());

    if (enable_fine_timers)
        cudaEventRecord(ev_matvec_start[0], stream);
    // Initial matvec:
    //   w^(i) = H v_1^(i)
    chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
        cublas_handle, &One, H, v_1, &Zero, v_w);
    if (enable_fine_timers)
        cudaEventRecord(ev_matvec_end[0], stream);

    if (enable_fine_timers)
        cudaEventRecord(ev_redist_start[0], stream);
    // Redistribute distributed result to match local layout of v_2:
    //   v_2^(i) <- redistribute(w^(i))
    v_w.redistributeImplAsync(&v_2, &stream);
    if (enable_fine_timers)
        cudaEventRecord(ev_redist_end[0], stream);

    // Build S v_2 for pseudo-Hermitian inner products:
    //   Sv^(i) = S v_2^(i)
    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), numvec,
                                           v_2.l_data(), v_2.l_ld(),
                                           Sv.l_data(), Sv.l_ld());
    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv, stream);

    if (enable_fine_timers)
        cudaEventRecord(ev_init_beta_start, stream);
    // Initial off-diagonal coefficient (local):
    //   beta_0^(i,loc)^2 = <v_1^(i), S v_2^(i)>_loc
    batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta, v_1.l_rows(), numvec,
                      v_1.l_ld(), false, &stream);
    getRealPart(d_beta, d_real_beta, static_cast<int>(numvec), &stream);
    auto comm_time_us = 0.0;
    auto comm_start = std::chrono::high_resolution_clock::now();
    // Global reduction:
    //   beta_0^(i)^2 = sum_r beta_0^(i,loc,r)^2
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_beta, d_real_beta, numvec, ncclSum, nccl_comm, &stream));
    auto comm_end = std::chrono::high_resolution_clock::now();
    comm_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                        comm_end - comm_start)
                        .count();
    // beta_0^(i) = sqrt(beta_0^(i)^2), and keep reciprocal for alpha step:
    //   d_real_beta_prev(i) <- 1 / beta_0^(i)
    batchedSqrt(d_real_beta, static_cast<int>(numvec), &stream);
    realReciprocal(d_real_beta, d_real_beta_prev, static_cast<int>(numvec),
                   &stream);
    copyRealReciprocalToT(d_real_beta, d_beta, static_cast<int>(numvec),
                          &stream);
    // Normalize recurrence pair:
    //   v_1^(i) <- v_1^(i) / beta_0^(i),  v_2^(i) <- v_2^(i) / beta_0^(i)
    batchedScale(d_beta, v_1.l_data(), v_1.l_rows(), numvec, v_1.l_ld(),
                 &stream);
    batchedScale(d_beta, v_2.l_data(), v_2.l_rows(), numvec, v_2.l_ld(),
                 &stream);
    if (enable_fine_timers)
        cudaEventRecord(ev_init_beta_end, stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(V.l_data() + k * V.l_ld(),
                            v_1.l_data() + i * v_1.l_ld(),
                            v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice,
                            0);
        }

        if (enable_fine_timers)
            cudaEventRecord(ev_alpha_start[k], stream);
        // Diagonal coefficient (local part):
        //   alpha_k^(i,loc) = <v_2^(i), S v_2^(i)>_loc
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha, v_2.l_rows(),
                          numvec, v_2.l_ld(), false, &stream);
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        comm_start = std::chrono::high_resolution_clock::now();
        // Global alpha:
        //   alpha_k^(i) = sum_r alpha_k^(i,loc,r)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, numvec, ncclSum, nccl_comm, &stream));
        comm_end = std::chrono::high_resolution_clock::now();
        comm_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            comm_end - comm_start)
                            .count();
        // Convert to complex/scalar format and apply normalized correction:
        //   v_2^(i) <- v_2^(i) - alpha_k^(i) * (1 / beta_{k-1}^(i)) v_1^(i)
        copyRealToT(d_real_alpha, d_alpha, static_cast<int>(numvec), &stream);
        scaleComplexByRealNegate(d_alpha, d_real_beta_prev,
                                 static_cast<int>(numvec), &stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(), v_1.l_rows(), numvec,
                    v_1.l_ld(), &stream);
        // Store diagonal entry:
        //   d(k, i) = alpha_k^(i)
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(d_d + (k + M * i), d_real_alpha + i,
                            sizeof(RealT), cudaMemcpyDeviceToDevice, stream);
        }
        if (enable_fine_timers)
            cudaEventRecord(ev_alpha_end[k], stream);

        // Three-term recurrence term:
        //   v_2^(i) <- v_2^(i) - beta_k^(i) v_0^(i)
        copyRealNegateToT(d_real_beta, d_beta_neg, static_cast<int>(numvec),
                          &stream);
        batchedAxpy(d_beta_neg, v_0.l_data(), v_2.l_data(), v_0.l_rows(),
                    numvec, v_0.l_ld(), &stream);

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
        chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv,
                                                                      stream);

        if (enable_fine_timers)
            cudaEventRecord(ev_beta_start[k], stream);
        // Off-diagonal coefficient (local part):
        //   beta_k^(i,loc)^2 = <v_1^(i), S v_2^(i)>_loc
        batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta, v_1.l_rows(),
                          numvec, v_1.l_ld(), false, &stream);
        getRealPart(d_beta, d_real_beta, static_cast<int>(numvec), &stream);
        comm_start = std::chrono::high_resolution_clock::now();
        // Global beta:
        //   beta_k^(i)^2 = sum_r beta_k^(i,loc,r)^2
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_beta, d_real_beta, numvec, ncclSum, nccl_comm, &stream));
        comm_end = std::chrono::high_resolution_clock::now();
        comm_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            comm_end - comm_start)
                            .count();
        // beta_k^(i) = sqrt(sum_r beta_k^(i,loc)^2_r), then form 1/beta_k^(i)
        batchedSqrt(d_real_beta, static_cast<int>(numvec), &stream);
        realReciprocal(d_real_beta, d_real_beta_prev, static_cast<int>(numvec),
                       &stream);
        copyRealReciprocalToT(d_real_beta, d_beta, static_cast<int>(numvec),
                              &stream);
        // Re-normalize recurrence pair:
        //   v_1^(i) <- v_1^(i) / beta_k^(i),   v_2^(i) <- v_2^(i) / beta_k^(i)
        batchedScaleTwo(d_beta, v_1.l_data(), v_2.l_data(), v_1.l_rows(),
                        numvec, v_1.l_ld(), &stream);
        // Store off-diagonal entry:
        //   e(k, i) = beta_k^(i)
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(e_d + (k + M * i), d_real_beta + i, sizeof(RealT),
                            cudaMemcpyDeviceToDevice, stream);
        }
        if (enable_fine_timers)
            cudaEventRecord(ev_beta_end[k], stream);
    }

    {
        const std::size_t k = M - 1;
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(V.l_data() + k * V.l_ld(),
                            v_1.l_data() + i * v_1.l_ld(),
                            v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice,
                            0);
        }

        if (enable_fine_timers)
            cudaEventRecord(ev_alpha_start[k], stream);
        // Diagonal coefficient (local part):
        //   alpha_{M-1}^(i,loc) = <v_2^(i), S v_2^(i)>_loc
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha, v_2.l_rows(),
                          numvec, v_2.l_ld(), false, &stream);
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        comm_start = std::chrono::high_resolution_clock::now();
        // Global alpha:
        //   alpha_{M-1}^(i) = sum_r alpha_{M-1}^(i,loc,r)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, numvec, ncclSum, nccl_comm, &stream));
        comm_end = std::chrono::high_resolution_clock::now();
        comm_time_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            comm_end - comm_start)
                            .count();
        // Apply normalized alpha correction:
        //   v_2^(i) <- v_2^(i) - alpha_{M-1}^(i) * (1 / beta_{M-2}^(i)) v_1^(i)
        copyRealToT(d_real_alpha, d_alpha, static_cast<int>(numvec), &stream);
        scaleComplexByRealNegate(d_alpha, d_real_beta_prev,
                                 static_cast<int>(numvec), &stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(), v_1.l_rows(), numvec,
                    v_1.l_ld(), &stream);
        // Store diagonal entry:
        //   d(M-1, i) = alpha_{M-1}^(i)
        getRealPart(d_alpha, d_real_alpha, static_cast<int>(numvec), &stream);
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(d_d + (k + M * i), d_real_alpha + i,
                            sizeof(RealT), cudaMemcpyDeviceToDevice, stream);
        }
        if (enable_fine_timers)
            cudaEventRecord(ev_alpha_end[k], stream);
    }

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           v_1.l_data(), v_1.l_ld(), V.l_data(),
                                           V.l_ld(), &stream);

    cudaStreamSynchronize(stream);

    if (enable_fine_timers)
    {
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
    cudaMemcpy(d.data(), d_d, M * numvec * sizeof(RealT),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(e.data(), e_d, M * numvec * sizeof(RealT),
               cudaMemcpyDeviceToHost);
    cudaFree(d_d);
    cudaFree(e_d);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_real_alpha);
    cudaFree(d_real_beta);
    cudaFree(d_real_beta_prev);
    cudaFree(d_beta_neg);

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
        std::chrono::duration_cast<std::chrono::microseconds>(lapack_end -
                                                               lapack_start);

    *upperb = ritzv[M - 1];

    // Pseudo-Hermitian coupling diag: off unless CHASE_PH_LANCZOS_DIAG=1|true|on|yes
    if (detail_nccl_ph_diag::ph_lanczos_diag_enabled())
    {
        detail_nccl_ph_diag::maybe_nccl_ph_lanczos_coupling_report(H, ritzv,
                                                                   M * numvec);
    }

    auto lanczos_end = std::chrono::high_resolution_clock::now();
    auto lanczos_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(lanczos_end -
                                                               lanczos_start);
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss;
        double lanczos_ms = lanczos_duration.count() / 1000.0;
        double comm_ms = 0.0;
        double redist_ms = redist_time_us / 1000.0;
        double matvec_ms = matvec_time_us / 1000.0;
        double sync_alpha_ms = sync_alpha_time_us / 1000.0;
        double sync_beta_ms = sync_beta_time_us / 1000.0;
        double init_beta_ms = init_beta_time_us / 1000.0;
        double alpha_step_ms = alpha_step_time_us / 1000.0;
        double beta_step_ms = beta_step_time_us / 1000.0;
        double other_ms = lanczos_ms - comm_ms - redist_ms - matvec_ms
            - init_beta_ms - alpha_step_ms - beta_step_ms - sync_alpha_ms
            - sync_beta_ms;
        oss << "[PSEUDO-HERMITIAN LANCZOS TIMING] (NCCL, multi-vector)\n"
            << "  numvec: " << numvec << ", M: " << M << "\n"
            << "  Lanczos total (matches performance.hpp): " << lanczos_ms
            << " ms\n"
            << "    Matvec (H*v, events):      " << matvec_ms << " ms\n"
            << "    Redistribute:              " << redist_ms << " ms\n"
            << "    Init norm (dot+scale):     " << init_beta_ms << " ms\n"
            << "    Alpha step (dot+AXPY+d[k]): " << alpha_step_ms << " ms\n"
            << "    Beta step (dot+scale+e[k]): " << beta_step_ms << " ms\n"
            << "    NCCL AllReduce (CPU wall): " << comm_ms << " ms\n"
            << "    Sync+D2H (alpha, d[k]):    " << sync_alpha_ms << " ms\n"
            << "    Sync+D2H (beta, e[k]):     " << sync_beta_ms << " ms\n"
            << "    Other (lacpy, flip, swap): " << other_ms << " ms\n"
            << "  LAPACK t_stemr (CPU) total: "
            << lapack_duration.count() / 1000.0 << " ms\n"
            << "  Avg LAPACK per solve: "
            << lapack_duration.count() / (double)numvec / 1000.0 << " ms\n";
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

    // GPU-resident Lanczos (single vector)
    cudaStream_t stream = nullptr;

#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss;
        oss << "[GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS (SINGLE)]: ENABLED, "
               "using NCCL + batched dot/AXPY/scale (DEVICE mode)";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif

    using RealT = chase::Base<T>;
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();

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

    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedScale;
    using chase::linalg::internal::cuda::batchedScaleTwo;
    using chase::linalg::internal::cuda::batchedSqrt;
    using chase::linalg::internal::cuda::copyRealNegateToT;
    using chase::linalg::internal::cuda::copyRealReciprocalToT;
    using chase::linalg::internal::cuda::copyRealToT;
    using chase::linalg::internal::cuda::getRealPart;
    using chase::linalg::internal::cuda::realReciprocal;
    using chase::linalg::internal::cuda::scaleComplexByRealNegate;

    cublasSetStream(cublas_handle, stream);

    chase::Base<T> real_alpha;
    chase::Base<T> real_beta;

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

    // Copy initial vector:
    //   v_1 <- V(:, 0)
    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), 1, V.l_data(),
                                           V.l_ld(), v_1.l_data(), v_1.l_ld());

    // Initial matvec:
    //   w = H v_1
    chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
        cublas_handle, &One, H, v_1, &Zero, v_w);

    auto redist_time_us = 0.0;
    // Redistribute distributed result:
    //   v_2 <- redistribute(w)
    v_w.redistributeImplAsync(&v_2, &stream);

    // Build S v_2:
    //   Sv = S v_2
    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1, v_2.l_data(),
                                           v_2.l_ld(), Sv.l_data(), Sv.l_ld());
    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv, stream);

    // Initial off-diagonal coefficient (local):
    //   beta_0^(loc)^2 = <v_1, S v_2>_loc
    batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta, v_1.l_rows(), 1,
                      v_1.l_ld(), false, &stream);
    getRealPart(d_beta, d_real_beta, 1, &stream);
    // Global reduction:
    //   beta_0^2 = sum_r beta_0^(loc,r)^2
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_beta, d_real_beta, 1, ncclSum, nccl_comm, &stream));
    // beta_0 = sqrt(beta_0^2), keep reciprocal:
    //   d_real_beta_prev <- 1 / beta_0
    batchedSqrt(d_real_beta, 1, &stream);
    realReciprocal(d_real_beta, d_real_beta_prev, 1, &stream);
    copyRealReciprocalToT(d_real_beta, d_beta, 1, &stream);
    // Normalize recurrence pair:
    //   v_1 <- v_1 / beta_0,  v_2 <- v_2 / beta_0
    batchedScale(d_beta, v_1.l_data(), v_1.l_rows(), 1, v_1.l_ld(), &stream);
    batchedScale(d_beta, v_2.l_data(), v_2.l_rows(), 1, v_2.l_ld(), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        // Diagonal coefficient:
        //   alpha_k^(loc) = <v_2, S v_2>_loc
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha, v_2.l_rows(), 1,
                          v_2.l_ld(), false, &stream);
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        // Global alpha:
        //   alpha_k = sum_r alpha_k^(loc,r)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, 1, ncclSum, nccl_comm, &stream));
        // Store diagonal entry:
        //   d(k) = alpha_k
        cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT),
                        cudaMemcpyDeviceToDevice, stream);
        // Apply alpha correction:
        //   v_2 <- v_2 - alpha_k * (1 / beta_{k-1}) v_1
        copyRealToT(d_real_alpha, d_alpha, 1, &stream);
        scaleComplexByRealNegate(d_alpha, d_real_beta_prev, 1, &stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(), v_1.l_rows(), 1,
                    v_1.l_ld(), &stream);

        // Three-term recurrence term:
        //   v_2 <- v_2 - beta_{k-1} v_0
        copyRealNegateToT(d_real_beta_prev, d_beta_neg, 1, &stream);
        batchedAxpy(d_beta_neg, v_0.l_data(), v_2.l_data(), v_0.l_rows(), 1,
                    v_0.l_ld(), &stream);

        v_1.swap(v_0);
        v_1.swap(v_2);

        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        v_w.redistributeImplAsync(&v_2, &stream);

        chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1,
                                               v_2.l_data(), v_2.l_ld(),
                                               Sv.l_data(), Sv.l_ld());
        chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv, stream);

        // Off-diagonal coefficient:
        //   beta_k^(loc)^2 = <v_1, S v_2>_loc
        batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta, v_1.l_rows(), 1,
                          v_1.l_ld(), false, &stream);
        getRealPart(d_beta, d_real_beta, 1, &stream);
        // Global beta:
        //   beta_k^2 = sum_r beta_k^(loc,r)^2
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_beta, d_real_beta, 1, ncclSum, nccl_comm, &stream));
        batchedSqrt(d_real_beta, 1, &stream);
        // Store off-diagonal entry:
        //   e(k) = beta_k
        cudaMemcpyAsync(e_d + k, d_real_beta, sizeof(RealT),
                        cudaMemcpyDeviceToDevice, stream);
        realReciprocal(d_real_beta, d_real_beta_prev, 1, &stream);
        copyRealReciprocalToT(d_real_beta, d_beta, 1, &stream);
        // Re-normalize recurrence pair:
        //   v_1 <- v_1 / beta_k,   v_2 <- v_2 / beta_k
        batchedScaleTwo(d_beta, v_1.l_data(), v_2.l_data(), v_1.l_rows(), 1,
                        v_1.l_ld(), &stream);
    }

    {
        const std::size_t k = M - 1;
        cudaMemcpy(V.l_data() + k * V.l_ld(), v_1.l_data(),
                   v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice);

        // Diagonal coefficient:
        //   alpha_{M-1}^(loc) = <v_2, S v_2>_loc
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha, v_2.l_rows(), 1,
                          v_2.l_ld(), false, &stream);
        getRealPart(d_alpha, d_real_alpha, 1, &stream);
        // Global alpha:
        //   alpha_{M-1} = sum_r alpha_{M-1}^(loc,r)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, 1, ncclSum, nccl_comm, &stream));
        // Store diagonal entry:
        //   d(M-1) = alpha_{M-1}
        cudaMemcpyAsync(d_d + k, d_real_alpha, sizeof(RealT),
                        cudaMemcpyDeviceToDevice, stream);
        // Apply alpha correction:
        //   v_2 <- v_2 - alpha_{M-1} * (1 / beta_{M-2}) v_1
        copyRealToT(d_real_alpha, d_alpha, 1, &stream);
        scaleComplexByRealNegate(d_alpha, d_real_beta_prev, 1, &stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(), v_1.l_rows(), 1,
                    v_1.l_ld(), &stream);
    }

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
    auto lapack_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(lapack_end -
                                                               lapack_start);
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::ostringstream oss;
        oss << "[PSEUDO-HERMITIAN LANCZOS TIMING - SINGLE VECTOR] LAPACK t_stemr "
               "(CPU):";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
        oss << "  M: " << M;
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
        oss << "  Time: " << lapack_duration.count() / 1000.0 << " ms";
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(),
                               H.getMpiGrid()->get_myRank());
    }
#endif
    *upperb = ritzv[M - 1];

    if (detail_nccl_ph_diag::ph_lanczos_diag_enabled())
    {
        detail_nccl_ph_diag::maybe_nccl_ph_lanczos_coupling_report(H, ritzv.data(),
                                                                   M);
    }
}

} // namespace internal
} // namespace linalg
} // namespace chase
