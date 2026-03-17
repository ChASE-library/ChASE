// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup nccl_kernels chase::linalg::internal::nccl Namespace
 * \brief The `chase::linalg::internal::nccl` namespace contains
 * kernels required by ChASE for the distributed-memory GPU using NCCL
 * for communications.
 * @{
 */
#include "../typeTraits.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include <nccl.h>
/** @} */

// Forward declaration of CUDA stream type to avoid pulling CUDA headers here.
using cudaStream_t = struct CUstream_st*;

namespace chase
{
namespace linalg
{
namespace internal
{
struct cuda_nccl
{
    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector);

    // Hierarchical HEMM matvec (uses hierarchical NCCL reduction)
    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsHierarchical(
        cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsHierarchical(
        cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector);

    // this operation do: W1<-1.0 * H * V1, while redistribute V2 to W2
    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistributeAsync(
        cublasHandle_t cublas_handle, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistributeAsync(
        cublasHandle_t cublas_handle, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector);

    template <typename MatrixType, typename InputMultiVectorType>
    static void
    lanczos_dispatch(cublasHandle_t cublas_handle, std::size_t M,
                     std::size_t numvec, MatrixType& H, InputMultiVectorType& V,
                     chase::Base<typename MatrixType::value_type>* upperb,
                     chase::Base<typename MatrixType::value_type>* ritzv,
                     chase::Base<typename MatrixType::value_type>* Tau,
                     chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void
    lanczos_dispatch(cublasHandle_t cublas_handle, std::size_t M, MatrixType& H,
                     InputMultiVectorType& V,
                     chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(cublasHandle_t cublas_handle, std::size_t M,
                        std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(cublasHandle_t cublas_handle, std::size_t M,
                        MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(
        cublasHandle_t cublas_handle, std::size_t M, std::size_t numvec,
        MatrixType& H, InputMultiVectorType& V,
        chase::Base<typename MatrixType::value_type>* upperb,
        chase::Base<typename MatrixType::value_type>* ritzv,
        chase::Base<typename MatrixType::value_type>* Tau,
        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(
        cublasHandle_t cublas_handle, std::size_t M, MatrixType& H,
        InputMultiVectorType& V,
        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename T>
    static int cholQR1(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle, std::size_t m,
                       std::size_t n, T* V, int ldv, ncclComm_t comm,
                       T* workspace = nullptr, int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static int
    cholQR1(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            InputMultiVectorType& V,
            typename InputMultiVectorType::value_type* workspace = nullptr,
            int lwork = 0,
            typename InputMultiVectorType::value_type* A = nullptr);

    template <typename T>
    static int cholQR2(cublasHandle_t cublas_handle,
                       cusolverDnHandle_t cusolver_handle, std::size_t m,
                       std::size_t n, T* V, int ldv, ncclComm_t comm,
                       T* workspace = nullptr, int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static int
    cholQR2(cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
            InputMultiVectorType& V,
            typename InputMultiVectorType::value_type* workspace = nullptr,
            int lwork = 0,
            typename InputMultiVectorType::value_type* A = nullptr);

    template <typename T>
    static int shiftedcholQR2(cublasHandle_t cublas_handle,
                              cusolverDnHandle_t cusolver_handle, std::size_t N,
                              std::size_t m, std::size_t n, T* V, int ldv,
                              ncclComm_t comm, T* workspace = nullptr,
                              int lwork = 0, T* A = nullptr);

    template <typename T>
    static int modifiedGramSchmidtCholQR(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        std::size_t m, std::size_t n, std::size_t locked, T* V, std::size_t ldv,
        ncclComm_t comm, T* workspace = nullptr, int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static void houseHoulderQR(InputMultiVectorType& V);

    /** Householder QR (geqrf + form Q) using NCCL; result Q in V1. Uses V2 as temp.
     *  nb: column block size for the blocked algorithm (default 64; set to 0
     *  or any value >= n to use the unblocked variant). */
    template <typename InputMultiVectorType>
    static void houseQR1_formQ(cublasHandle_t cublas_handle,
                               InputMultiVectorType& V1,
                               InputMultiVectorType& V2,
                               typename InputMultiVectorType::value_type* workspace,
                               int lwork,
                               std::size_t nb = 32,
                               cusolverDnHandle_t cusolver_handle = nullptr);

    /** Distributed Householder QR + form Q. V must be on device; uses NCCL for collectives. */
    template <typename T>
    static void distributed_houseQR_formQ(
        std::size_t m_global, std::size_t n, std::size_t l_rows,
        std::size_t g_off, std::size_t ldv, T* V, MPI_Comm mpi_comm,
        cublasHandle_t cublas_handle, T* d_workspace, std::size_t lwork_elems,
        ncclComm_t nccl_col_comm);

    /** Blocked distributed Householder QR + form Q. V on device; NCCL for collectives. */
    template <typename T>
    static void distributed_blocked_houseQR_formQ(
        std::size_t m_global, std::size_t n, std::size_t l_rows,
        std::size_t g_off, std::size_t ldv, T* V, MPI_Comm mpi_comm,
        std::size_t nb,
        cublasHandle_t cublas_handle, T* d_workspace, std::size_t lwork_elems,
        ncclComm_t nccl_col_comm,
        cusolverDnHandle_t cusolver_handle = nullptr);

    /** Lightweight timing container for Householder panel factorization. */
    struct HouseholderPanelTiming
    {
        float norm_ms          = 0.f;
        float scalar_kernel_ms = 0.f;
        float allreduce_tau_ms = 0.f;
        float scal_ms          = 0.f;
        float trail_ms         = 0.f;
    };

    /** Panel factorization for columns [k, k+jb). Used by blocked/unblocked QR; swap for different strategy. */
    template <typename T>
    static void distributed_houseQR_panel_factor(
        std::size_t n, std::size_t l_rows, std::size_t g_off, std::size_t ldv,
        T* V, std::size_t k, std::size_t jb, T* d_tau,
        cublasHandle_t cublas_handle,
        chase::Base<T>* d_real_scalar, T* d_T_scalar,
        T* d_one, T* d_zero, T* d_minus_one, T* d_panel_scalars, T* d_w,
        ncclComm_t nccl_col_comm,
        HouseholderPanelTiming* panel_timing = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void rayleighRitz(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::distMatrix::RedundantMatrix<
            chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
            ritzv,
        std::size_t offset, std::size_t subSize, int* devInfo,
        typename MatrixType::value_type* workspace = nullptr,
        int lwork_heevd = 0,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::GPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_rayleighRitz(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        cusolverDnParams_t params, MatrixType& H, InputMultiVectorType& V1,
        InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::distMatrix::RedundantMatrix<
            chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
            ritzv,
        std::size_t offset, std::size_t subSize, int* devInfo,
        typename MatrixType::value_type* d_workspace = nullptr, int d_lwork = 0,
        typename MatrixType::value_type* h_workspace = nullptr, int h_lwork = 0,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::GPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_rayleighRitz_v2(
        cublasHandle_t cublas_handle, cusolverDnHandle_t cusolver_handle,
        cusolverDnParams_t params, MatrixType& H, InputMultiVectorType& V1,
        InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::distMatrix::RedundantMatrix<
            chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
            ritzv,
        std::size_t offset, std::size_t subSize, int* devInfo,
        typename MatrixType::value_type* d_workspace = nullptr, int lwork = 0,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::GPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void residuals(
        cublasHandle_t cublas_handle, MatrixType& H, InputMultiVectorType& V1,
        InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                              typename MatrixType::platform_type>& ritzv,
        chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                              typename MatrixType::platform_type>& resids,
        std::size_t offset, std::size_t subSize);

    template <typename MatrixType>
    static bool checkSymmetryEasy(cublasHandle_t cublas_handle, MatrixType& H);

    template <typename T>
    static void symOrHermMatrix(
        char uplo,
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H);

    template <typename T>
    static void symOrHermMatrix(
        char uplo,
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H);

    template <typename MatrixType>
    static void
    shiftDiagonal(MatrixType& H, std::size_t* d_off_m, std::size_t* d_off_n,
                  std::size_t offsize,
                  chase::Base<typename MatrixType::value_type> shift);

    template <typename T>
    static void flipLowerHalfMatrixSign(
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H);

    template <typename T>
    static void flipLowerHalfMatrixSign(
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H);

    template <typename InputMultiVectorType>
    static void flipLowerHalfMatrixSign(InputMultiVectorType& V);

    template <typename InputMultiVectorType>
    static void flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                        cudaStream_t stream);

    template <typename InputMultiVectorType>
    static void flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                        std::size_t offset,
                                        std::size_t subSize);
};

} // namespace internal
} // namespace linalg
} // namespace chase

#include "linalg/internal/nccl/cholqr.hpp"
#include "linalg/internal/nccl/householder_qr.hpp"
#include "linalg/internal/nccl/flipSign.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/nccl/lanczos.hpp"
#include "linalg/internal/nccl/pseudo_hermitian_lanczos.hpp"
#include "linalg/internal/nccl/pseudo_hermitian_rayleighRitz.hpp"
#include "linalg/internal/nccl/rayleighRitz.hpp"
#include "linalg/internal/nccl/residuals.hpp"
#include "linalg/internal/nccl/shiftDiagonal.hpp"
#include "linalg/internal/nccl/symOrHerm.hpp"
