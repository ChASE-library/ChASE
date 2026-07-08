// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup hip_mpi_kernels chase::linalg::internal::hip_mpi
 * Namespace
 * \brief The `chase::linalg::internal::hip_mpi` namespace contains kernels
 * required by ChASE for distributed-memory GPU using HIP-aware MPI for
 * communications.
 * @{
 */
#include "linalg/internal/typeTraits.hpp"
#include "external/amd_gpu/rocblaspp/rocblaspp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include <mpi.h>
/** @} */

namespace chase
{
namespace linalg
{
namespace internal
{
struct hip_mpi
{
    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        rocblas_handle rocblas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        rocblas_handle rocblas_handle, T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector);

    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistribute(
        rocblas_handle rocblas_handle, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistribute(
        rocblas_handle rocblas_handle, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector);

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
                                        std::size_t offset,
                                        std::size_t subSize);

    template <typename MatrixType, typename InputMultiVectorType>
    static void residuals(
        rocblas_handle rocblas_handle, MatrixType& H, InputMultiVectorType& V1,
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
    static bool checkSymmetryEasy(rocblas_handle rocblas_handle, MatrixType& H);

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

    template <typename MatrixType, typename InputMultiVectorType>
    static void
    lanczos_dispatch(rocblas_handle rocblas_handle, std::size_t M,
                     std::size_t numvec, MatrixType& H, InputMultiVectorType& V,
                     chase::Base<typename MatrixType::value_type>* upperb,
                     chase::Base<typename MatrixType::value_type>* ritzv,
                     chase::Base<typename MatrixType::value_type>* Tau,
                     chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void
    lanczos_dispatch(rocblas_handle rocblas_handle, std::size_t M, MatrixType& H,
                     InputMultiVectorType& V,
                     chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(rocblas_handle rocblas_handle, std::size_t M,
                        std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(rocblas_handle rocblas_handle, std::size_t M,
                        MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(
        rocblas_handle rocblas_handle, std::size_t M, std::size_t numvec,
        MatrixType& H, InputMultiVectorType& V,
        chase::Base<typename MatrixType::value_type>* upperb,
        chase::Base<typename MatrixType::value_type>* ritzv,
        chase::Base<typename MatrixType::value_type>* Tau,
        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(
        rocblas_handle rocblas_handle, std::size_t M, MatrixType& H,
        InputMultiVectorType& V,
        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename T>
    static int cholQR1(rocblas_handle blas_handle,
                       rocblas_handle solver_handle, std::size_t m, std::size_t n,
                       T* V, int ldv, MPI_Comm comm, T* workspace = nullptr,
                       int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static int cholQR1(rocblas_handle blas_handle,
                       rocblas_handle solver_handle, InputMultiVectorType& V,
                       typename InputMultiVectorType::value_type* workspace =
                           nullptr,
                       int lwork = 0,
                       typename InputMultiVectorType::value_type* A = nullptr);

    template <typename T>
    static int cholQR2(rocblas_handle blas_handle,
                       rocblas_handle solver_handle, std::size_t m, std::size_t n,
                       T* V, int ldv, MPI_Comm comm, T* workspace = nullptr,
                       int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static int cholQR2(rocblas_handle blas_handle,
                       rocblas_handle solver_handle, InputMultiVectorType& V,
                       typename InputMultiVectorType::value_type* workspace =
                           nullptr,
                       int lwork = 0,
                       typename InputMultiVectorType::value_type* A = nullptr);

    template <typename T>
    static int shiftedcholQR2(rocblas_handle blas_handle,
                              rocblas_handle solver_handle, std::size_t N,
                              std::size_t m, std::size_t n, T* V, int ldv,
                              MPI_Comm comm, T* workspace = nullptr,
                              int lwork = 0, T* A = nullptr);

    template <typename T>
    static int modifiedGramSchmidtCholQR(
        rocblas_handle blas_handle, rocblas_handle solver_handle,
        std::size_t m, std::size_t n, std::size_t locked, T* V, std::size_t ldv,
        MPI_Comm comm, T* workspace = nullptr, int lwork = 0, T* A = nullptr);

    template <typename InputMultiVectorType>
    static void houseHoulderQR(InputMultiVectorType& V);

    template <typename MatrixType, typename InputMultiVectorType>
    static void rayleighRitz(
        rocblas_handle blas_handle, rocblas_handle solver_handle,
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
        rocblas_handle blas_handle, rocblas_handle solver_handle, void* params,
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
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
        rocblas_handle blas_handle, rocblas_handle solver_handle, void* params,
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::distMatrix::RedundantMatrix<
            chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
            ritzv,
        std::size_t offset, std::size_t subSize, int* devInfo,
        typename MatrixType::value_type* d_workspace = nullptr, int d_lwork = 0,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::GPU>* A = nullptr);
};

} // namespace internal
} // namespace linalg
} // namespace chase

#include "linalg/internal/gpu/dist_mpi/hip/flipSign.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hemm.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/residuals.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/shiftDiagonal.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/symOrHerm.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/lanczos.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/pseudo_hermitian_lanczos.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/cholqr.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/rayleighRitz.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/pseudo_hermitian_rayleighRitz.hpp"
