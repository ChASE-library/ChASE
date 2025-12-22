// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup mpi_kernels chase::linalg::internal::mpi Namespace
 * \brief The `chase::linalg::internal::mpi` namespace contains
 * kernels required by ChASE for the distributed-memory CPU
 * @{
 */
#include "../typeTraits.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
/** @} */

namespace chase
{
namespace linalg
{
namespace internal
{
struct cpu_mpi
{
    template <typename T>
    static int cholQR1(std::size_t m, std::size_t n, T* V, int ldv,
                       MPI_Comm comm, T* A = nullptr);

    template <typename T>
    static int cholQR2(std::size_t m, std::size_t n, T* V, int ldv,
                       MPI_Comm comm, T* A = nullptr);

    template <typename InputMultiVectorType>
    static int cholQR1(InputMultiVectorType& V, 
                      typename InputMultiVectorType::value_type *A = nullptr);

    template <typename InputMultiVectorType>
    static int cholQR2(InputMultiVectorType& V, 
                      typename InputMultiVectorType::value_type *A = nullptr);

    template <typename T>
    static int shiftedcholQR2(std::size_t N, std::size_t m, std::size_t n, T* V,
                              int ldv, MPI_Comm comm, T* A = nullptr);

    template <typename InputMultiVectorType>
    static void houseHoulderQR(InputMultiVectorType& V);

    template <typename InputMultiVectorType>
    static chase::Base<typename InputMultiVectorType::value_type> computeConditionNumber(InputMultiVectorType& V);

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename T, typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectors(
        T* alpha, MatrixType& blockMatrix,
        InputMultiVectorType& input_multiVector, T* beta,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector);

    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistributeAsync(
        MatrixType& blockMatrix, InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector,
        std::size_t offset, std::size_t subSize);

    template <typename MatrixType, typename InputMultiVectorType>
    static void MatrixMultiplyMultiVectorsAndRedistributeAsync(
        MatrixType& blockMatrix, InputMultiVectorType& input_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            result_multiVector,
        InputMultiVectorType& src_multiVector,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            target_multiVector);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos_dispatch(std::size_t M, std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(std::size_t M, std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV);
    
    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(std::size_t M, std::size_t numvec, MatrixType& H,
                        InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb,
                        chase::Base<typename MatrixType::value_type>* ritzv,
                        chase::Base<typename MatrixType::value_type>* Tau,
                        chase::Base<typename MatrixType::value_type>* ritzV);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos_dispatch(std::size_t M, MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void lanczos(std::size_t M, MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb);
    
    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_lanczos(std::size_t M, MatrixType& H, InputMultiVectorType& V,
                        chase::Base<typename MatrixType::value_type>* upperb);

    template <typename MatrixType, typename InputMultiVectorType>
    static void rayleighRitz_dispatch(
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::Base<typename MatrixType::value_type>* ritzv, std::size_t offset,
        std::size_t subSize,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::CPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void rayleighRitz(
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::Base<typename MatrixType::value_type>* ritzv, std::size_t offset,
        std::size_t subSize,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::CPU>* A = nullptr);
    
    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_rayleighRitz(
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::Base<typename MatrixType::value_type>* ritzv, std::size_t offset,
        std::size_t subSize,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::CPU>* A = nullptr);
    
    template <typename MatrixType, typename InputMultiVectorType>
    static void pseudo_hermitian_rayleighRitz_v2(
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::Base<typename MatrixType::value_type>* ritzv, std::size_t offset,
        std::size_t subSize,
        chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                           chase::platform::CPU>* A = nullptr);

    template <typename MatrixType, typename InputMultiVectorType>
    static void residuals(
        MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W1,
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
            W2,
        chase::Base<typename MatrixType::value_type>* ritzv,
        chase::Base<typename MatrixType::value_type>* resids,
        std::size_t offset, std::size_t subSize);

    template <typename T>
    static void shiftDiagonal(
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>& H,
        T shift);

    template <typename T>
    static void shiftDiagonal(
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>& H,
        T shift);

    template <typename T>
    static void flipLowerHalfMatrixSign(
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>& H);
    
    template <typename T>
    static void flipLowerHalfMatrixSign(
        chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>& H);

    template <typename T, chase::distMultiVector::CommunicatorType comm_type>
    static void flipLowerHalfMatrixSign(
        chase::distMultiVector::DistMultiVector1D<T, comm_type,
                                                  chase::platform::CPU>& V, 
						  std::size_t offset,
						  std::size_t subSize);
    
    template <typename T, chase::distMultiVector::CommunicatorType comm_type>
    static void flipLowerHalfMatrixSign(
        chase::distMultiVector::DistMultiVector1D<T, comm_type,
                                                  chase::platform::CPU>& V);
    
    template <typename T, chase::distMultiVector::CommunicatorType comm_type>
    static void flipLowerHalfMatrixSign(
        chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, comm_type,
                                                  chase::platform::CPU>& V,
						  std::size_t offset,
						  std::size_t subSize);
    
    template <typename T, chase::distMultiVector::CommunicatorType comm_type>
    static void flipLowerHalfMatrixSign(
        chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, comm_type,
                                                  chase::platform::CPU>& V);

    template <typename MatrixType>
    static bool checkSymmetryEasy(MatrixType& H);

    template <typename T>
    static void symOrHermMatrix(char uplo,
                                chase::distMatrix::BlockBlockMatrix<T>& H);

    template <typename T>
    static void symOrHermMatrix(char uplo,
                                chase::distMatrix::BlockCyclicMatrix<T>& H);
};
} // namespace internal
} // namespace linalg
} // namespace chase

#include "linalg/internal/mpi/cholqr.hpp"
#include "linalg/internal/mpi/hemm.hpp"
#include "linalg/internal/mpi/lanczos.hpp"
#include "linalg/internal/mpi/pseudo_hermitian_lanczos.hpp"
#include "linalg/internal/mpi/rayleighRitz.hpp"
#include "linalg/internal/mpi/pseudo_hermitian_rayleighRitz.hpp"
#include "linalg/internal/mpi/residuals.hpp"
#include "linalg/internal/mpi/shiftDiagonal.hpp"
#include "linalg/internal/mpi/symOrHerm.hpp"
