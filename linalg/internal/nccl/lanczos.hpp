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

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[GPU-RESIDENT LANCZOS]: ENABLED, using NCCL + Fused Kernels" << std::endl;
    }
#endif
    // ========================================================================
    // GPU-RESIDENT VERSION: Fused kernels + NCCL collectives
    // ========================================================================

    // Get NCCL communicator from grid
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();
    
    // Create stream for Lanczos operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ========================================================================
    // GPU-Resident: Allocate device buffers
    // ========================================================================
    T* d_alpha;
    RealT* d_real_alpha;
    RealT* d_r_beta;
    cudaMalloc(&d_alpha, numvec * sizeof(T));
    cudaMalloc(&d_real_alpha, numvec * sizeof(RealT));
    cudaMalloc(&d_r_beta, numvec * sizeof(RealT));
    
    // Initialize to zero to prevent garbage values
    cudaMemset(d_alpha, 0, numvec * sizeof(T));
    cudaMemset(d_real_alpha, 0, numvec * sizeof(RealT));
    cudaMemset(d_r_beta, 0, numvec * sizeof(RealT));

    // Host buffers (only for LAPACK and final results)
    std::vector<RealT> r_beta_host(numvec);
    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);

    // Note: cublas_handle is already configured with CUBLAS_POINTER_MODE_DEVICE
    // in the pchase_gpu constructor, so all scalar pointers must be on device
    cublasSetStream(cublas_handle, stream);

    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld());

    // ========================================================================
    // Initial normalization (GPU-resident)
    // ========================================================================
    
    // Compute norms squared on GPU (results stay on GPU)
    using chase::linalg::internal::cuda::batchedNormSquared;
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::normalizeVectors;
    using chase::linalg::internal::cuda::negate;
    using chase::linalg::internal::cuda::sqrtInplace;
    
    batchedNormSquared(v_1.l_data(), d_real_alpha, v_1.l_rows(), numvec, v_1.l_ld(), &stream);
    CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors

    // NCCL Allreduce: sum norms across column communicator (GPU-GPU)
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_alpha, d_real_alpha, numvec, ncclSum, nccl_comm, &stream));

    // Normalize vectors: fused 1/sqrt + scale (single kernel launch)
    normalizeVectors(v_1.l_data(), d_real_alpha, v_1.l_rows(), numvec, v_1.l_ld(), &stream);

    // ========================================================================
    // Main Lanczos iteration loop
    // ========================================================================
    
    for (std::size_t k = 0; k < M; k = k + 1)
    {
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpyAsync(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(),
                       v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }

        // Matrix-vector multiply
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);
        
        // Pass stream to ensure redistribution uses same stream as other operations
        v_w.redistributeImplAsync(&v_2, &stream);

        // Batched dot products with negation (single kernel launch, GPU-resident)
        // Computes d_alpha[i] = -conj(v_1[:,i]) · v_2[:,i] for all i in one pass
        batchedDotProduct(v_1.l_data(), v_2.l_data(), d_alpha, 
                         v_1.l_rows(), numvec, v_1.l_ld(), true, &stream);  // true = negate
        CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors

        // NCCL Allreduce (GPU-GPU) - sum the negative alphas
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_alpha, d_alpha, numvec, ncclSum, nccl_comm, &stream));

        // Batched AXPY with NEGATIVE alpha: v_2[:,i] += (-alpha[i]) * v_1[:,i] for all i
        // Single kernel launch replaces numvec separate cublasTaxpy calls
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(), 
                   v_1.l_rows(), numvec, v_1.l_ld(), &stream);

        // Negate back to get positive alpha for storage
        negate(d_alpha, numvec, &stream);

        // Copy alpha to host for d array (needed for LAPACK later)
        std::vector<T> alpha_host(numvec);
        cudaMemcpyAsync(alpha_host.data(), d_alpha, numvec * sizeof(T),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha_host[i]);
        }

        // Subtract previous beta contribution (if k > 0)
        if (k > 0)
        {
            // Prepare negative beta values on GPU for batched AXPY
            T* d_beta_neg;
            cudaMalloc(&d_beta_neg, numvec * sizeof(T));
            
            // Convert RealT beta to T and copy to device
            std::vector<T> beta_host_T(numvec);
            for (auto i = 0; i < numvec; i++)
            {
                beta_host_T[i] = T(-r_beta_host[i]);
            }
            cudaMemcpyAsync(d_beta_neg, beta_host_T.data(), numvec * sizeof(T),
                           cudaMemcpyHostToDevice, stream);
            
            // Batched AXPY: v_2[:,i] += (-beta[i]) * v_0[:,i] for all i
            batchedAxpy(d_beta_neg, v_0.l_data(), v_2.l_data(),
                       v_0.l_rows(), numvec, v_0.l_ld(), &stream);
            
            cudaFree(d_beta_neg);
        }

        // Compute norms squared (GPU-resident)
        batchedNormSquared(v_2.l_data(), d_r_beta, v_2.l_rows(), numvec, v_2.l_ld(), &stream);
        CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors

        // NCCL Allreduce: sum norm squared
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_r_beta, d_r_beta, numvec, ncclSum, nccl_comm, &stream));

        // Copy norm squared to host and compute sqrt on host
        cudaMemcpyAsync(r_beta_host.data(), d_r_beta, numvec * sizeof(RealT),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        for (auto i = 0; i < numvec; i++)
        {
            r_beta_host[i] = std::sqrt(r_beta_host[i]);
        }

        if (k == M - 1)
        {
            break;
        }

        // Normalize v_2: v_2 *= 1/||v_2|| using norm² still on GPU
        normalizeVectors(v_2.l_data(), d_r_beta, v_2.l_rows(), numvec, v_2.l_ld(), &stream);

        // Store e[k] for LAPACK
        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = r_beta_host[i];
        }

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           v_1.l_data(), v_1.l_ld(), V.l_data(),
                                           V.l_ld());

    // Ensure all GPU operations complete
    cudaStreamSynchronize(stream);

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
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[LANCZOS TIMING] LAPACK t_stemr (CPU sequential):" << std::endl;
        std::cout << "  numvec: " << numvec << ", M: " << M << std::endl;
        std::cout << "  Total time: " << lapack_duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Avg per solve: " << lapack_duration.count() / (double)numvec / 1000.0 << " ms" << std::endl;
    }
#endif
    RealT max;
    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(r_beta_host[0]);

    for (auto i = 1; i < numvec; i++)
    {
        max =
            std::max(std::abs(ritzv[i * M]), std::abs(ritzv[(i + 1) * M - 1])) +
            std::abs(r_beta_host[i]);
        *upperb = std::max(max, *upperb);
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_alpha);
    cudaFree(d_real_alpha);
    cudaFree(d_r_beta);
    cudaStreamDestroy(stream);

#else
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[ORIGINAL MPI LANCZOS]: ENABLED, using MPI_Allreduce (not GPU-resident)" << std::endl;
    }
#endif
    // ========================================================================
    // ORIGINAL VERSION: MPI-based collectives (default)
    // ========================================================================

    std::vector<RealT> r_beta(numvec);

    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);

    std::vector<RealT> real_alpha(numvec);
    std::vector<T> alpha(numvec, T(1.0));
    std::vector<T> beta(numvec, T(0.0));

    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld());
    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
            cublas_handle, v_1.l_rows(), v_1.l_data() + i * v_1.l_ld(), 1,
            &real_alpha[i]));
        real_alpha[i] = std::pow(real_alpha[i], 2);
    }

    MPI_Allreduce(MPI_IN_PLACE, real_alpha.data(), numvec,
                  chase::mpi::getMPI_Type<RealT>(), MPI_SUM,
                  H.getMpiGrid()->get_col_comm());

    for (auto i = 0; i < numvec; i++)
    {
        real_alpha[i] = std::sqrt(real_alpha[i]);
        alpha[i] = T(1 / real_alpha[i]);
    }

    for (auto i = 0; i < numvec; i++)
    {
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_1.l_rows(), &alpha[i],
            v_1.l_data() + i * v_1.l_ld(), 1));
    }

    for (std::size_t k = 0; k < M; k = k + 1)
    {
        for (auto i = 0; i < numvec; i++)
        {
            cudaMemcpy(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(),
                       v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice);
        }

        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);
        
        // Pass stream to ensure redistribution uses same stream as other operations  
        v_w.redistributeImplAsync(&v_2, &stream);

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, v_1.l_rows(), v_1.l_data() + i * v_1.l_ld(), 1,
                v_2.l_data() + i * v_2.l_ld(), 1, &alpha[i]));

            alpha[i] = -alpha[i];
        }

        MPI_Allreduce(MPI_IN_PLACE, alpha.data(), numvec,
                      chase::mpi::getMPI_Type<T>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_1.l_rows(), &alpha[i],
                v_1.l_data() + i * v_1.l_ld(), 1, v_2.l_data() + i * v_2.l_ld(),
                1));
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha[i]);
        }
        if (k > 0)
        {
            for (auto i = 0; i < numvec; i++)
            {
                beta[i] = T(-r_beta[i]);
            }
            for (auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                    cublas_handle, v_0.l_rows(), &beta[i],
                    v_0.l_data() + i * v_0.l_ld(), 1,
                    v_2.l_data() + i * v_2.l_ld(), 1));
            }
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
                cublas_handle, v_2.l_rows(), v_2.l_data() + i * v_2.l_ld(), 1,
                &r_beta[i]));

            r_beta[i] = std::pow(r_beta[i], 2);
        }

        MPI_Allreduce(MPI_IN_PLACE, r_beta.data(), numvec,
                      chase::mpi::getMPI_Type<RealT>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            r_beta[i] = std::sqrt(r_beta[i]);
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = T(1 / r_beta[i]);
        }

        if (k == M - 1)
        {
            break;
        }

        for (auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, v_2.l_rows(), &beta[i],
                v_2.l_data() + i * v_2.l_ld(), 1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = r_beta[i];
        }

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           v_1.l_data(), v_1.l_ld(), V.l_data(),
                                           V.l_ld());

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

    RealT max;
    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(r_beta[0]);

    for (auto i = 1; i < numvec; i++)
    {
        max =
            std::max(std::abs(ritzv[i * M]), std::abs(ritzv[(i + 1) * M - 1])) +
            std::abs(r_beta[i]);
        *upperb = std::max(max, *upperb);
    }
#endif // CHASE_ENABLE_GPU_RESIDENT_LANCZOS
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

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[GPU-RESIDENT LANCZOS (single-vec)]: ENABLED, using NCCL + Fused Kernels" << std::endl;
    }
#endif
    // ========================================================================
    // GPU-RESIDENT VERSION: Fused kernels + NCCL collectives
    // ========================================================================

    // Get NCCL communicator from grid
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();
    
    // Create stream for Lanczos operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // ========================================================================
    // GPU-Resident: Allocate device buffers
    // ========================================================================
    T* d_alpha;
    RealT* d_real_alpha;
    RealT* d_r_beta;
    cudaMalloc(&d_alpha, sizeof(T));
    cudaMalloc(&d_real_alpha, sizeof(RealT));
    cudaMalloc(&d_r_beta, sizeof(RealT));
    
    // Initialize to zero to prevent garbage values
    cudaMemset(d_alpha, 0, sizeof(T));
    cudaMemset(d_real_alpha, 0, sizeof(RealT));
    cudaMemset(d_r_beta, 0, sizeof(RealT));

    // Host buffers
    std::vector<RealT> d(M);
    std::vector<RealT> e(M);
    RealT r_beta_host;

    // Note: cublas_handle is already configured with CUBLAS_POINTER_MODE_DEVICE
    // in the pchase_gpu constructor, so all scalar pointers must be on device
    cublasSetStream(cublas_handle, stream);

    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), 1, V.l_data(),
                                           V.l_ld(), v_1.l_data(), v_1.l_ld());

    // ========================================================================
    // Initial normalization (GPU-resident)
    // ========================================================================
    
    using chase::linalg::internal::cuda::batchedNormSquared;
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::normalizeVectors;
    using chase::linalg::internal::cuda::negate;
    using chase::linalg::internal::cuda::sqrtInplace;
    using chase::linalg::internal::cuda::squareInplace;
    
    // Compute norm squared on GPU (single vector version)
    batchedNormSquared(v_1.l_data(), d_real_alpha, v_1.l_rows(), 1, v_1.l_ld(), &stream);
    CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors

    // NCCL Allreduce
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_alpha, d_real_alpha, 1, ncclSum, nccl_comm, &stream));

    // Normalize: fused sqrt + reciprocal + scale
    normalizeVectors(v_1.l_data(), d_real_alpha, v_1.l_rows(), 1, v_1.l_ld(), &stream);

    // ========================================================================
    // Main Lanczos iteration loop
    // ========================================================================
    
    for (std::size_t k = 0; k < M; k = k + 1)
    {
        // Matrix-vector multiply
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        // Pass stream to ensure redistribution uses same stream as other operations
        v_w.redistributeImplAsync(&v_2, &stream);

        // Dot product with negation (single-vector, but using batched kernel for consistency)
        batchedDotProduct(v_1.l_data(), v_2.l_data(), d_alpha,
                         v_1.l_rows(), 1, v_1.l_ld(), true, &stream);  // true = negate
        CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors

        // NCCL Allreduce (sum the negative alpha)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_alpha, d_alpha, 1, ncclSum, nccl_comm, &stream));

        // AXPY with NEGATIVE alpha: v_2 = v_2 + (-alpha)*v_1
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
            cublas_handle, v_1.l_rows(), d_alpha, v_1.l_data(), 1, v_2.l_data(),
            1));

        // Negate back to get positive alpha for storage
        negate(d_alpha, 1, &stream);

        // Copy alpha to host for d array
        T alpha_host;
        cudaMemcpyAsync(&alpha_host, d_alpha, sizeof(T),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        d[k] = std::real(alpha_host);
        
#ifdef CHASE_OUTPUT
        if (std::isnan(d[k]) || std::isinf(d[k])) {
            if (H.getMpiGrid()->get_myRank() == 0) {
                std::cout << "[LANCZOS DEBUG] d[" << k << "] = " << d[k] 
                         << " (alpha_host = " << alpha_host << ")" << std::endl;
            }
        }
#endif

        if (k > 0)
        {
            T beta_host = T(-r_beta_host);
            // Use synchronous copy to ensure d_alpha is ready before cuBLAS uses it
            cudaMemcpy(d_alpha, &beta_host, sizeof(T), cudaMemcpyHostToDevice);
            
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_0.l_rows(), d_alpha, v_0.l_data(), 1,
                v_2.l_data(), 1));
        }

        // Compute norm squared (GPU-resident)
        batchedNormSquared(v_2.l_data(), d_r_beta, v_2.l_rows(), 1, v_2.l_ld(), &stream);
        CHECK_CUDA_ERROR(cudaGetLastError());  // Check for kernel launch errors

        // NCCL Allreduce: sum norm squared
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_r_beta, d_r_beta, 1, ncclSum, nccl_comm, &stream));

        // Copy norm squared to host and compute sqrt on host
        cudaMemcpyAsync(&r_beta_host, d_r_beta, sizeof(RealT),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        r_beta_host = std::sqrt(r_beta_host);
        
#ifdef CHASE_OUTPUT
        if (std::isnan(r_beta_host) || std::isinf(r_beta_host) || r_beta_host < 0) {
            if (H.getMpiGrid()->get_myRank() == 0) {
                std::cout << "[LANCZOS DEBUG] k=" << k << " r_beta_host = " << r_beta_host << std::endl;
            }
        }
#endif

        if (k == M - 1)
            break;

        // Normalize v_2: v_2 *= 1/||v_2|| using norm² still on GPU
        normalizeVectors(v_2.l_data(), d_r_beta, v_2.l_rows(), 1, v_2.l_ld(), &stream);
        
        e[k] = r_beta_host;
        
#ifdef CHASE_OUTPUT
        if (std::isnan(e[k]) || std::isinf(e[k])) {
            if (H.getMpiGrid()->get_myRank() == 0) {
                std::cout << "[LANCZOS DEBUG] e[" << k << "] = " << e[k] << std::endl;
            }
        }
#endif

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    // Ensure all GPU operations complete
    cudaStreamSynchronize(stream);

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
    bool has_nan = false;
    for (std::size_t i = 0; i < M; i++) {
        if (std::isnan(d[i]) || std::isinf(d[i])) {
            if (H.getMpiGrid()->get_myRank() == 0) {
                std::cout << "[LANCZOS WARNING] d[" << i << "] = " << d[i] << std::endl;
            }
            has_nan = true;
        }
        if (i < M - 1 && (std::isnan(e[i]) || std::isinf(e[i]))) {
            if (H.getMpiGrid()->get_myRank() == 0) {
                std::cout << "[LANCZOS WARNING] e[" << i << "] = " << e[i] << std::endl;
            }
            has_nan = true;
        }
    }
    if (has_nan && H.getMpiGrid()->get_myRank() == 0) {
        std::cout << "[LANCZOS WARNING] NaN/Inf detected in tridiagonal matrix!" << std::endl;
    }
#endif

    lapackpp::t_stemr<RealT>(
        LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
        &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);
    
    auto lapack_end = std::chrono::high_resolution_clock::now();
    auto lapack_duration = std::chrono::duration_cast<std::chrono::microseconds>(lapack_end - lapack_start);
    
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[LANCZOS TIMING - SINGLE VECTOR] LAPACK t_stemr (CPU):" << std::endl;
        std::cout << "  M: " << M << std::endl;
        std::cout << "  Time: " << lapack_duration.count() / 1000.0 << " ms" << std::endl;
    }
#endif
    *upperb =
        std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) + std::abs(r_beta_host);

    // ========================================================================
    // Cleanup
    // ========================================================================
    
    cudaFree(d_alpha);
    cudaFree(d_real_alpha);
    cudaFree(d_r_beta);
    cudaStreamDestroy(stream);

#else
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "========================================" << std::endl;
        std::cout << "ORIGINAL MPI LANCZOS (single-vec): ENABLED" << std::endl;
        std::cout << "Using MPI_Allreduce (not GPU-resident)" << std::endl;
        std::cout << "========================================" << std::endl;
    }
#endif
    // ========================================================================
    // ORIGINAL VERSION: MPI-based collectives (default)
    // ========================================================================

    std::vector<RealT> d(M);
    std::vector<RealT> e(M);

    RealT real_alpha;
    RealT r_beta;

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), 1, V.l_data(),
                                           V.l_ld(), v_1.l_data(), v_1.l_ld());

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
        cublas_handle, v_1.l_rows(), v_1.l_data(), 1, &real_alpha));
    real_alpha = std::pow(real_alpha, 2);

    MPI_Allreduce(MPI_IN_PLACE, &real_alpha, 1,
                  chase::mpi::getMPI_Type<RealT>(), MPI_SUM,
                  H.getMpiGrid()->get_col_comm());

    real_alpha = std::sqrt(real_alpha);
    alpha = T(1 / real_alpha);

    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
        cublas_handle, v_1.l_rows(), &alpha, v_1.l_data(), 1));
    
    for (std::size_t k = 0; k < M; k = k + 1)
    {
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        // Pass stream to ensure redistribution uses same stream as other operations
        v_w.redistributeImplAsync(&v_2, &stream);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
            cublas_handle, v_1.l_rows(), v_1.l_data(), 1, v_2.l_data(), 1,
            &alpha));
        alpha = -alpha;

        MPI_Allreduce(MPI_IN_PLACE, &alpha, 1, chase::mpi::getMPI_Type<T>(),
                      MPI_SUM, H.getMpiGrid()->get_col_comm());

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
            cublas_handle, v_1.l_rows(), &alpha, v_1.l_data(), 1, v_2.l_data(),
            1));
        alpha = -alpha;

        d[k] = std::real(alpha);

        if (k > 0)
        {
            beta = T(-r_beta);
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(
                cublas_handle, v_0.l_rows(), &beta, v_0.l_data(), 1,
                v_2.l_data(), 1));
        }

        beta = -beta;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(
            cublas_handle, v_2.l_rows(), v_2.l_data(), 1, &r_beta));

        r_beta = std::pow(r_beta, 2);

        MPI_Allreduce(MPI_IN_PLACE, &r_beta, 1,
                      chase::mpi::getMPI_Type<RealT>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        r_beta = std::sqrt(r_beta);

        beta = T(1 / r_beta);

        if (k == M - 1)
            break;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
            cublas_handle, v_2.l_rows(), &beta, v_2.l_data(), 1));
        e[k] = r_beta;

        v_1.swap(v_0);
        v_1.swap(v_2);
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
        LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
        &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);

    *upperb =
        std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) + std::abs(r_beta);
#endif // CHASE_ENABLE_GPU_RESIDENT_LANCZOS
}

} // namespace internal
} // namespace linalg
} // namespace chase
