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

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS]: ENABLED, using NCCL + batched dot/AXPY/scale (DEVICE mode)" << std::endl;
    }
#endif
    // ========================================================================
    // GPU-RESIDENT VERSION: Fully batched kernels + NCCL collectives
    // Uses DEVICE pointer mode with batchedScale for vector normalization
    // ========================================================================

    // Get NCCL communicator from grid
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();
    
    // Create stream for Lanczos operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // GPU-resident buffers for scalar results
    using RealT = chase::Base<T>;
    T* d_alpha;
    T* d_beta;
    RealT* d_real_alpha;
    RealT* d_real_beta;
    cudaMalloc(&d_alpha, numvec * sizeof(T));
    cudaMalloc(&d_beta, numvec * sizeof(T));
    cudaMalloc(&d_real_alpha, numvec * sizeof(RealT));
    cudaMalloc(&d_real_beta, numvec * sizeof(RealT));

    // Host buffers (needed for LAPACK and some operations)
    std::vector<chase::Base<T>> real_beta(numvec);
    std::vector<chase::Base<T>> real_alpha(numvec);
#else
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[ORIGINAL MPI PSEUDO-HERMITIAN LANCZOS]: ENABLED, using MPI_Allreduce" << std::endl;  
    }
#endif
    std::vector<chase::Base<T>> real_beta(numvec);
    std::vector<chase::Base<T>> real_alpha(numvec);
#endif

    std::vector<chase::Base<T>> d(M * numvec);
    std::vector<chase::Base<T>> e(M * numvec);

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // Host-side arrays for LAPACK and HOST pointer mode operations
    std::vector<T> alpha(numvec, T(1.0));
    std::vector<T> beta(numvec, T(0.0));
    
    // Include fused kernel utilities
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedScale;
    
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

    chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld());

    chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
        cublas_handle, &One, H, v_1, &Zero, v_w);

    v_w.redistributeImplAsync(&v_2);

    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), numvec,
                                           v_2.l_data(), v_2.l_ld(),
                                           Sv.l_data(), Sv.l_ld());

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
    // Ensure flip operation completes (it uses default stream)
    cudaDeviceSynchronize();

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // ========================================================================
    // INITIAL NORMALIZATION: Compute real(v_1 · Sv) then normalize
    // Following EXACT algorithm as original MPI version
    // ========================================================================
    
    // Batched dot products: d_beta[i] = conj(v_1[:,i]) · Sv[:,i]
    batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                     v_1.l_rows(), numvec, v_1.l_ld(), false, &stream);
    
    // Copy complex results to host, extract real parts
    std::vector<T> beta_host_complex(numvec);
    cudaMemcpyAsync(beta_host_complex.data(), d_beta, numvec * sizeof(T),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    for (auto i = 0; i < numvec; i++)
    {
        real_beta[i] = std::real(beta_host_complex[i]);
    }
    
    // Copy real parts to device for NCCL Allreduce
    cudaMemcpyAsync(d_real_beta, real_beta.data(), numvec * sizeof(RealT),
                   cudaMemcpyHostToDevice, stream);
    
    // NCCL Allreduce (sum real parts)
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_beta, d_real_beta, numvec, ncclSum, nccl_comm, &stream));
    
    // Copy back to host for sqrt computation
    cudaMemcpyAsync(real_beta.data(), d_real_beta, numvec * sizeof(RealT),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    // Compute beta = 1/sqrt(real_beta) for normalization
    for (auto i = 0; i < numvec; i++)
    {
        beta[i] = One / std::sqrt(real_beta[i]);
    }
    
    // Copy beta to device for batched scaling
    cudaMemcpyAsync(d_beta, beta.data(), numvec * sizeof(T),
                   cudaMemcpyHostToDevice, stream);
    
    // Scale v_1 and v_2 by beta (normalization)
    batchedScale(d_beta, v_1.l_data(), v_1.l_rows(), numvec, v_1.l_ld(), &stream);
    batchedScale(d_beta, v_2.l_data(), v_2.l_rows(), numvec, v_2.l_ld(), &stream);
    cudaStreamSynchronize(stream);
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
        // Step 1: Compute alpha = real(v_2 · Sv), allreduce, then alpha = -real_alpha * beta
        
        // Batched dot products: d_alpha[i] = conj(v_2[:,i]) · Sv[:,i]
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha,
                         v_2.l_rows(), numvec, v_2.l_ld(), false, &stream);
        
        // Copy to host and extract real parts
        std::vector<T> alpha_host_complex(numvec);
        cudaMemcpyAsync(alpha_host_complex.data(), d_alpha, numvec * sizeof(T),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        for (auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = std::real(alpha_host_complex[i]);
        }
        
        // Copy real parts to device for NCCL Allreduce
        cudaMemcpyAsync(d_real_alpha, real_alpha.data(), numvec * sizeof(RealT),
                       cudaMemcpyHostToDevice, stream);
        
        // NCCL Allreduce (sum real parts)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, numvec, ncclSum, nccl_comm, &stream));
        
        // Copy back to host
        cudaMemcpyAsync(real_alpha.data(), d_real_alpha, numvec * sizeof(RealT),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Compute alpha = -real_alpha * beta (beta from previous iteration/init)
        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -real_alpha[i] * beta[i];
        }
        
        // Step 2: AXPY: v_2[:,i] += alpha[i] * v_1[:,i]
        cudaMemcpyAsync(d_alpha, alpha.data(), numvec * sizeof(T),
                       cudaMemcpyHostToDevice, stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(),
                   v_1.l_rows(), numvec, v_1.l_ld(), &stream);
        cudaStreamSynchronize(stream);
        
        // Step 3: Negate alpha for storage in d array
        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha[i]);
        }
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
        // Step 5: Compute -1/beta, AXPY with v_0, restore beta
        // (Subtract previous vector contribution)
        
        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -One / beta[i];
        }
        
        // Copy beta to device
        cudaMemcpyAsync(d_beta, beta.data(), numvec * sizeof(T),
                       cudaMemcpyHostToDevice, stream);
        
        // Batched AXPY: v_2[:,i] += beta[i] * v_0[:,i]
        batchedAxpy(d_beta, v_0.l_data(), v_2.l_data(),
                   v_0.l_rows(), numvec, v_0.l_ld(), &stream);
        cudaStreamSynchronize(stream);
        
        // Restore beta sign
        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -beta[i];
        }
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

        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(
            cublas_handle, &One, H, v_1, &Zero, v_w);

        v_w.redistributeImplAsync(&v_2);

        chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), numvec,
                                               v_2.l_data(), v_2.l_ld(),
                                               Sv.l_data(), Sv.l_ld());

        chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
        // Ensure flip operation completes (it uses default stream)
        cudaDeviceSynchronize();

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        // Step 9: Compute beta = real(v_1 · Sv), allreduce, sqrt
        
        // Batched dot products: d_beta[i] = conj(v_1[:,i]) · Sv[:,i]
        batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                         v_1.l_rows(), numvec, v_1.l_ld(), false, &stream);
        
        // Copy to host and extract real parts
        std::vector<T> beta_host_complex_loop(numvec);
        cudaMemcpyAsync(beta_host_complex_loop.data(), d_beta, numvec * sizeof(T),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        for (auto i = 0; i < numvec; i++)
        {
            real_beta[i] = std::real(beta_host_complex_loop[i]);
        }
        
        // Copy real parts to device for NCCL Allreduce
        cudaMemcpyAsync(d_real_beta, real_beta.data(), numvec * sizeof(RealT),
                       cudaMemcpyHostToDevice, stream);
        
        // NCCL Allreduce (sum real parts)
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_beta, d_real_beta, numvec, ncclSum, nccl_comm, &stream));
        
        // Copy back to host for sqrt
        cudaMemcpyAsync(real_beta.data(), d_real_beta, numvec * sizeof(RealT),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Step 10: Compute sqrt and store in e array
        for (auto i = 0; i < numvec; i++)
        {
            real_beta[i] = std::sqrt(real_beta[i]);
        }

        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = real_beta[i];
        }

        // Step 11: Compute beta = 1/sqrt(real_beta) for next iteration
        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = One / real_beta[i];
        }
        
        // Step 12: Scale v_1 and v_2 by beta (normalization)
        cudaMemcpyAsync(d_beta, beta.data(), numvec * sizeof(T),
                       cudaMemcpyHostToDevice, stream);
        
        batchedScale(d_beta, v_1.l_data(), v_1.l_rows(), numvec, v_1.l_ld(), &stream);
        batchedScale(d_beta, v_2.l_data(), v_2.l_rows(), numvec, v_2.l_ld(), &stream);
        cudaStreamSynchronize(stream);
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
                                           V.l_ld());

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // Clean up GPU resources
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_real_alpha);
    cudaFree(d_real_beta);
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
    auto lapack_duration = std::chrono::duration_cast<std::chrono::microseconds>(lapack_end - lapack_start);
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[PSEUDO-HERMITIAN LANCZOS TIMING] LAPACK t_stemr (CPU sequential):" << std::endl;
        std::cout << "  numvec: " << numvec << ", M: " << M << std::endl;
        std::cout << "  Total time: " << lapack_duration.count() / 1000.0 << " ms" << std::endl;
        std::cout << "  Avg per solve: " << lapack_duration.count() / (double)numvec / 1000.0 << " ms" << std::endl;
    }
#endif
    *upperb = ritzv[M - 1];
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

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[GPU-RESIDENT PSEUDO-HERMITIAN LANCZOS (SINGLE)]: ENABLED, using NCCL + batched dot/AXPY/scale (DEVICE mode)" << std::endl;
    }
#endif
    // GPU-resident setup
    using RealT = chase::Base<T>;
    ncclComm_t nccl_comm = H.getMpiGrid()->get_nccl_col_comm();
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Device buffers for scalar results
    T* d_alpha;
    T* d_beta;
    RealT* d_real_alpha;
    RealT* d_real_beta;
    cudaMalloc(&d_alpha, sizeof(T));
    cudaMalloc(&d_beta, sizeof(T));
    cudaMalloc(&d_real_alpha, sizeof(RealT));
    cudaMalloc(&d_real_beta, sizeof(RealT));
    
    // Include fused kernel utilities
    using chase::linalg::internal::cuda::batchedDotProduct;
    using chase::linalg::internal::cuda::batchedAxpy;
    using chase::linalg::internal::cuda::batchedScale;
    
    cublasSetStream(cublas_handle, stream);
    
    chase::Base<T> real_alpha;
    chase::Base<T> real_beta;
#else
#ifdef CHASE_OUTPUT
    if (H.getMpiGrid()->get_myRank() == 0)
    {
        std::cout << "[ORIGINAL MPI PSEUDO-HERMITIAN LANCZOS (SINGLE)]: ENABLED, using MPI_Allreduce" << std::endl;
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

    v_w.redistributeImplAsync(&v_2);

    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1, v_2.l_data(),
                                           v_2.l_ld(), Sv.l_data(), Sv.l_ld());

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
    // Ensure flip operation completes (it uses default stream)
    cudaDeviceSynchronize();

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
    // Dot product: beta = v_1 · Sv
    batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                     v_1.l_rows(), 1, v_1.l_ld(), false, &stream);
    
    // Copy complex result to host and extract real part
    T beta_complex_val;
    cudaMemcpyAsync(&beta_complex_val, d_beta, sizeof(T),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    real_beta = std::real(beta_complex_val);
    
    // Copy real part to device for NCCL Allreduce
    cudaMemcpyAsync(d_real_beta, &real_beta, sizeof(RealT),
                   cudaMemcpyHostToDevice, stream);
    
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_real_beta, d_real_beta, 1, ncclSum, nccl_comm, &stream));
    
    // Copy allreduced value back to host
    cudaMemcpyAsync(&real_beta, d_real_beta, sizeof(RealT),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    beta = One / std::sqrt(real_beta);

    // Copy beta to device for batched scaling
    cudaMemcpyAsync(d_beta, &beta, sizeof(T), cudaMemcpyHostToDevice, stream);
    
    // Scale vectors using GPU-resident batched kernel
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
        // Dot product: alpha = v_2 · Sv
        batchedDotProduct(v_2.l_data(), Sv.l_data(), d_alpha,
                         v_2.l_rows(), 1, v_2.l_ld(), false, &stream);
        
        // Copy complex result to host and extract real part
        T alpha_complex_val;
        cudaMemcpyAsync(&alpha_complex_val, d_alpha, sizeof(T),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        real_alpha = std::real(alpha_complex_val);
        
        // Copy real part to device for NCCL Allreduce
        cudaMemcpyAsync(d_real_alpha, &real_alpha, sizeof(RealT),
                       cudaMemcpyHostToDevice, stream);
        
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_alpha, d_real_alpha, 1, ncclSum, nccl_comm, &stream));
        
        // Copy allreduced value back to host
        cudaMemcpyAsync(&real_alpha, d_real_alpha, sizeof(RealT),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        alpha = -real_alpha * beta;
        
        // AXPY: v_2 += alpha * v_1
        cudaMemcpyAsync(d_alpha, &alpha, sizeof(T),
                       cudaMemcpyHostToDevice, stream);
        batchedAxpy(d_alpha, v_1.l_data(), v_2.l_data(),
                   v_1.l_rows(), 1, v_1.l_ld(), &stream);
        cudaStreamSynchronize(stream);
        
        alpha = -alpha;

        d[k] = std::real(alpha);
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
        beta = -One / beta;
        
        // AXPY: v_2 += beta * v_0
        cudaMemcpyAsync(d_beta, &beta, sizeof(T),
                       cudaMemcpyHostToDevice, stream);
        batchedAxpy(d_beta, v_0.l_data(), v_2.l_data(),
                   v_0.l_rows(), 1, v_0.l_ld(), &stream);
        cudaStreamSynchronize(stream);
        
        beta = -beta;
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

        v_w.redistributeImplAsync(&v_2);

        chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1,
                                               v_2.l_data(), v_2.l_ld(),
                                               Sv.l_data(), Sv.l_ld());

        chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
        // Ensure flip operation completes (it uses default stream)
        cudaDeviceSynchronize();

#ifdef CHASE_ENABLE_GPU_RESIDENT_LANCZOS
        // Dot product: beta = v_1 · Sv
        batchedDotProduct(v_1.l_data(), Sv.l_data(), d_beta,
                         v_1.l_rows(), 1, v_1.l_ld(), false, &stream);
        
        // Copy complex result to host and extract real part
        T beta_complex_loop_val;
        cudaMemcpyAsync(&beta_complex_loop_val, d_beta, sizeof(T),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        real_beta = std::real(beta_complex_loop_val);
        
        // Copy real part to device for NCCL Allreduce
        cudaMemcpyAsync(d_real_beta, &real_beta, sizeof(RealT),
                       cudaMemcpyHostToDevice, stream);
        
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            d_real_beta, d_real_beta, 1, ncclSum, nccl_comm, &stream));
        
        // Copy allreduced value back to host for sqrt computation
        cudaMemcpyAsync(&real_beta, d_real_beta, sizeof(RealT),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        real_beta = std::sqrt(real_beta);
        e[k] = real_beta;
        
        // Compute beta = 1/||v|| for use in next iteration
        beta = One / real_beta;

        // Copy beta to device for batched scaling
        cudaMemcpyAsync(d_beta, &beta, sizeof(T), cudaMemcpyHostToDevice, stream);
        
        // Scale vectors using GPU-resident batched kernel
        batchedScale(d_beta, v_1.l_data(), v_1.l_rows(), 1, v_1.l_ld(), &stream);
        batchedScale(d_beta, v_2.l_data(), v_2.l_rows(), 1, v_2.l_ld(), &stream);
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
    // Clean up GPU resources
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_real_alpha);
    cudaFree(d_real_beta);
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
        std::cout << "[PSEUDO-HERMITIAN LANCZOS TIMING - SINGLE VECTOR] LAPACK t_stemr (CPU):" << std::endl;
        std::cout << "  M: " << M << std::endl;
        std::cout << "  Time: " << lapack_duration.count() / 1000.0 << " ms" << std::endl;
    }
#endif
    *upperb = ritzv[M - 1];
}
} // namespace internal
} // namespace linalg
} // namespace chase
