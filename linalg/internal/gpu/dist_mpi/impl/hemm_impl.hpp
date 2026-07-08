// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/typeTraits.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include <mpi.h>

namespace chase {
namespace linalg {
namespace internal {
namespace gpu_mpi
{

template <typename Backend, typename MpiKernelNamespace, typename BlasHandle,
          typename T, typename MatrixType, typename InputMultiVectorType>
void matrix_multiply_multi_vectors_impl(
    BlasHandle blas_handle, T* alpha, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector, T* beta,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    std::size_t offset, std::size_t subSize)
{
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;

    // Ensure the platform type is chase::platform::GPU
    static_assert(std::is_same<typename MatrixType::platform_type,
                               chase::platform::GPU>::value,
                  "Matrix type must be chase::platform::GPU");

    static_assert(std::is_same<typename InputMultiVectorType::platform_type,
                               chase::platform::GPU>::value,
                  "Multivector type must be chase::platform::GPU");

    if (input_multiVector.l_rows() !=
        (ExtractCommType<InputMultiVectorType>::value ==
                 chase::distMultiVector::CommunicatorType::row
             ? blockMatrix.l_cols()
             : blockMatrix.l_rows()))
    {
        throw std::runtime_error("Dimension mismatch: Input multiVector rows "
                                 "must match blockMatrix rows or columns.");
    }

    if (result_multiVector.l_rows() !=
        (ExtractCommType<InputMultiVectorType>::value ==
                 chase::distMultiVector::CommunicatorType::row
             ? blockMatrix.l_rows()
             : blockMatrix.l_cols()))
    {
        throw std::runtime_error("Dimension mismatch: Result multiVector rows "
                                 "must match blockMatrix rows or columns.");
    }

    // Check MPI grid compatibility
    if (input_multiVector.getMpiGrid() != result_multiVector.getMpiGrid() ||
        input_multiVector.getMpiGrid() != blockMatrix.getMpiGrid())
    {
        throw std::runtime_error(
            "MPI Grid mismatch between input and result multiVectors.");
    }

    // Check offset and subSize
    std::size_t inputCols = input_multiVector.l_cols();
    if (offset < 0 || offset >= inputCols)
    {
        throw std::invalid_argument("Offset must be between 0 and the number "
                                    "of input multiVector columns - 1.");
    }

    if (subSize <= 0 || subSize > inputCols)
    {
        throw std::invalid_argument(
            "SubSize must be greater than 0 and less than or equal to the "
            "number of input multiVector columns.");
    }
    if (offset + subSize > inputCols)
    {
        throw std::invalid_argument(
            "Offset plus SubSize must be less than or equal to the number of "
            "input multiVector columns.");
    }

    int* coords = input_multiVector.getMpiGrid()->get_coords();
    T beta_tmp;
    typename util::stream_type used_stream = util::get_blas_stream(blas_handle);

    if constexpr (ExtractCommType<InputMultiVectorType>::value ==
                  chase::distMultiVector::CommunicatorType::column)
    {
        if (coords[0] != 0)
        {
            beta_tmp = T(0.0); // If not the first row, set beta_tmp to 0
        }
        else
        {
            beta_tmp = *beta; // If the first row, use the provided beta value
        }

        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {

            MpiKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize);
            gpu_backend::device_synchronize<Backend>();
        }

        if (beta_tmp != T(0.0))
        {

            if constexpr (
                std::is_same<MatrixType,
                             chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                                 T, chase::platform::GPU>>::value ||
                std::is_same<
                    MatrixType,
                    chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                        T, chase::platform::GPU>>::value)
            {

                MpiKernelNamespace::flipLowerHalfMatrixSign(
                    result_multiVector, offset, subSize);
                gpu_backend::device_synchronize<Backend>();
            }
        }

        // Perform the matrix multiplication using BLAS
        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_c(), util::op_n(), blockMatrix.l_cols(),
            subSize, blockMatrix.l_rows(), alpha, blockMatrix.l_data(),
            blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &beta_tmp,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));
        gpu_mpi::mpiAllreduceSum(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize,
            input_multiVector.getMpiGrid()->get_col_comm(), used_stream);

        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {

            MpiKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize);
            gpu_backend::device_synchronize<Backend>();
        }

        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {

            MpiKernelNamespace::flipLowerHalfMatrixSign(
                result_multiVector, offset, subSize);
            gpu_backend::device_synchronize<Backend>();
        }
    }
    else // InputCommType is CommunicatorType::row
    {
        if (coords[1] != 0)
        {
            beta_tmp = T(0.0); // If not the first column, set beta_tmp to 0
        }
        else
        {
            beta_tmp =
                *beta; // If the first column, use the provided beta value
        }

        // Perform the matrix multiplication using BLAS
        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_n(), util::op_n(), blockMatrix.l_rows(),
            subSize, blockMatrix.l_cols(), alpha, blockMatrix.l_data(),
            blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &beta_tmp,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));

        // Perform reduction across the row communicator
        gpu_mpi::mpiAllreduceSum(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize,
            input_multiVector.getMpiGrid()->get_row_comm(), used_stream);
    }
}

// W1 <- H * V1 while redistributing V2 to W2.
template <typename Backend, typename MpiKernelNamespace, typename BlasHandle,
          typename MatrixType, typename InputMultiVectorType>
void matrix_multiply_multi_vectors_and_redistribute_impl(
    BlasHandle blas_handle, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    InputMultiVectorType& src_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        target_multiVector,
    std::size_t offset, std::size_t subSize)

{
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using T = typename MatrixType::value_type;

    // Ensure the platform type is chase::platform::GPU
    static_assert(std::is_same<typename MatrixType::platform_type,
                               chase::platform::GPU>::value,
                  "Matrix type must be chase::platform::GPU");

    static_assert(std::is_same<typename InputMultiVectorType::platform_type,
                               chase::platform::GPU>::value,
                  "Multivector type must be chase::platform::GPU");

    if (input_multiVector.l_rows() !=
        (ExtractCommType<InputMultiVectorType>::value ==
                 chase::distMultiVector::CommunicatorType::row
             ? blockMatrix.l_cols()
             : blockMatrix.l_rows()))
    {
        throw std::runtime_error("Dimension mismatch: Result multiVector rows "
                                 "must match blockMatrix rows or columns.");
    }

    if (result_multiVector.l_rows() !=
        (ExtractCommType<InputMultiVectorType>::value ==
                 chase::distMultiVector::CommunicatorType::row
             ? blockMatrix.l_rows()
             : blockMatrix.l_cols()))
    {
        throw std::runtime_error("Dimension mismatch: Result multiVector rows "
                                 "must match blockMatrix rows or columns.");
    }

    if (src_multiVector.l_rows() !=
        (ExtractCommType<InputMultiVectorType>::value ==
                 chase::distMultiVector::CommunicatorType::row
             ? blockMatrix.l_cols()
             : blockMatrix.l_rows()))
    {
        throw std::runtime_error("Dimension mismatch: Result multiVector rows "
                                 "must match blockMatrix rows or columns.");
    }

    if (target_multiVector.l_rows() !=
        (ExtractCommType<InputMultiVectorType>::value ==
                 chase::distMultiVector::CommunicatorType::row
             ? blockMatrix.l_rows()
             : blockMatrix.l_cols()))
    {
        throw std::runtime_error("Dimension mismatch: Result multiVector rows "
                                 "must match blockMatrix rows or columns.");
    }

    // Check MPI grid compatibility
    if (input_multiVector.getMpiGrid() != result_multiVector.getMpiGrid() ||
        input_multiVector.getMpiGrid() != blockMatrix.getMpiGrid())
    {
        throw std::runtime_error(
            "MPI Grid mismatch between input and result multiVectors.");
    }

    // Check MPI grid compatibility
    if (src_multiVector.getMpiGrid() != target_multiVector.getMpiGrid())
    {
        throw std::runtime_error(
            "MPI Grid mismatch between input and result multiVectors.");
    }

    // Check offset and subSize
    std::size_t inputCols = input_multiVector.l_cols();
    if (offset < 0 || offset >= inputCols)
    {
        throw std::invalid_argument("Offset must be between 0 and the number "
                                    "of input multiVector columns - 1.");
    }

    if (subSize <= 0 || subSize > inputCols)
    {
        throw std::invalid_argument(
            "SubSize must be greater than 0 and less than or equal to the "
            "number of input multiVector columns.");
    }
    if (offset + subSize > inputCols)
    {
        throw std::invalid_argument(
            "Offset plus SubSize must be less than or equal to the number of "
            "input multiVector columns.");
    }

    int* coords = input_multiVector.getMpiGrid()->get_coords();

    T One = T(1.0);
    T Zero = T(0.0);
    typename util::stream_type used_stream = util::get_blas_stream(blas_handle);

    if constexpr (ExtractCommType<InputMultiVectorType>::value ==
                  chase::distMultiVector::CommunicatorType::column)
    {
        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {

            MpiKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize);
            gpu_backend::device_synchronize<Backend>();
        }

        // Perform the matrix multiplication using BLAS
        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_c(), util::op_n(), blockMatrix.l_cols(),
            subSize, blockMatrix.l_rows(), &One, blockMatrix.l_data(),
            blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &Zero,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));

        // Perform reduction across the column communicator
        gpu_mpi::mpiAllreduceSum(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize,
            input_multiVector.getMpiGrid()->get_col_comm(), used_stream);

        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {

            MpiKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize);
            gpu_backend::device_synchronize<Backend>();
        }

        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {

            MpiKernelNamespace::flipLowerHalfMatrixSign(
                result_multiVector, offset, subSize);
            gpu_backend::device_synchronize<Backend>();
        }
    }
    else // InputCommType is CommunicatorType::row
    {
        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_n(), util::op_n(), blockMatrix.l_rows(),
            subSize, blockMatrix.l_cols(), &One, blockMatrix.l_data(),
            blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &Zero,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));
        // Perform reduction across the row communicator
        gpu_mpi::mpiAllreduceSum(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize,
            input_multiVector.getMpiGrid()->get_row_comm(), used_stream);
    }

    src_multiVector.redistributeImpl(
        &target_multiVector, offset,
        subSize); // computation of gemm, and redistribute can be overlapped,
                  // wiil do later
}


} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
