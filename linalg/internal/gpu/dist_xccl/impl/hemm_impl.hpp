// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/typeTraits.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "grid/nccl_utils.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"

#include <stdexcept>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_xccl
{

// W1 <- H * V1 while redistributing V2 to W2 (async device collective path).
template <typename GpuBackend, typename CollectiveKernelNamespace, typename BlasHandle,
          typename MatrixType, typename InputMultiVectorType>
void matrix_multiply_multi_vectors_and_redistribute_async_impl(
    BlasHandle blas_handle, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    InputMultiVectorType& src_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        target_multiVector,
    std::size_t offset, std::size_t subSize)
{
    using T = typename MatrixType::value_type;
    using util = gpu_backend::handle_utils<GpuBackend>;
    using blas_dispatch = gpu_backend::blas<GpuBackend>;

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

    if (input_multiVector.getMpiGrid() != result_multiVector.getMpiGrid() ||
        input_multiVector.getMpiGrid() != blockMatrix.getMpiGrid())
    {
        throw std::runtime_error(
            "MPI Grid mismatch between input and result multiVectors.");
    }

    if (src_multiVector.getMpiGrid() != target_multiVector.getMpiGrid())
    {
        throw std::runtime_error(
            "MPI Grid mismatch between input and result multiVectors.");
    }

    std::size_t inputCols = input_multiVector.l_cols();
    if (offset >= inputCols)
    {
        throw std::invalid_argument("Offset must be between 0 and the number "
                                    "of input multiVector columns - 1.");
    }

    if (subSize == 0 || subSize > inputCols)
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

    T One = T(1.0);
    T Zero = T(0.0);
    typename util::stream_t used_stream =
        util::get_blas_stream(blas_handle);

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
            CollectiveKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize, used_stream);
        }

        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_c(), util::op_n(),
            blockMatrix.l_cols(), subSize, blockMatrix.l_rows(), &One,
            blockMatrix.l_data(), blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &Zero,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize, ncclSum,
            input_multiVector.getMpiGrid()->get_nccl_col_comm(), &used_stream));

        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {
            CollectiveKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize, used_stream);
            CollectiveKernelNamespace::flipLowerHalfMatrixSign(
                result_multiVector, offset, subSize, used_stream);
        }
    }
    else
    {
        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_n(), util::op_n(),
            blockMatrix.l_rows(), subSize, blockMatrix.l_cols(), &One,
            blockMatrix.l_data(), blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &Zero,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize, ncclSum,
            input_multiVector.getMpiGrid()->get_nccl_row_comm(), &used_stream));
    }

    src_multiVector.redistributeImplAsync(&target_multiVector, offset, subSize,
                                          &used_stream);
}

template <typename GpuBackend, typename CollectiveKernelNamespace, typename BlasHandle,
          typename T, typename MatrixType, typename InputMultiVectorType>
void matrix_multiply_multi_vectors_impl(
    BlasHandle blas_handle, T* alpha, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector, T* beta,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    std::size_t offset, std::size_t subSize)
{
    using util = gpu_backend::handle_utils<GpuBackend>;
    using blas_dispatch = gpu_backend::blas<GpuBackend>;

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

    if (input_multiVector.getMpiGrid() != result_multiVector.getMpiGrid() ||
        input_multiVector.getMpiGrid() != blockMatrix.getMpiGrid())
    {
        throw std::runtime_error(
            "MPI Grid mismatch between input and result multiVectors.");
    }

    std::size_t inputCols = input_multiVector.l_cols();
    if (offset >= inputCols)
    {
        throw std::invalid_argument("Offset must be between 0 and the number "
                                    "of input multiVector columns - 1.");
    }

    if (subSize == 0 || subSize > inputCols)
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
    typename util::stream_t used_stream =
        util::get_blas_stream(blas_handle);

    if constexpr (ExtractCommType<InputMultiVectorType>::value ==
                  chase::distMultiVector::CommunicatorType::column)
    {
        if (coords[0] != 0)
        {
            beta_tmp = T(0.0);
        }
        else
        {
            beta_tmp = *beta;
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
            CollectiveKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize, used_stream);
        }

        if (beta_tmp != T(0.0))
        {
            if constexpr (std::is_same<
                              MatrixType,
                              chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                                  T, chase::platform::GPU>>::value ||
                          std::is_same<MatrixType,
                                       chase::distMatrix::
                                           PseudoHermitianBlockCyclicMatrix<
                                               T, chase::platform::GPU>>::value)
            {
                CollectiveKernelNamespace::flipLowerHalfMatrixSign(
                    result_multiVector, offset, subSize, used_stream);
            }
        }

        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_c(), util::op_n(),
            blockMatrix.l_cols(), subSize, blockMatrix.l_rows(), alpha,
            blockMatrix.l_data(), blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &beta_tmp,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize, ncclSum,
            input_multiVector.getMpiGrid()->get_nccl_col_comm(), &used_stream));

        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {
            CollectiveKernelNamespace::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize, used_stream);
            CollectiveKernelNamespace::flipLowerHalfMatrixSign(
                result_multiVector, offset, subSize, used_stream);
        }
    }
    else
    {
        if (coords[1] != 0)
        {
            beta_tmp = T(0.0);
        }
        else
        {
            beta_tmp = *beta;
        }

        util::check_blas(blas_dispatch::gemm(
            blas_handle, util::op_n(), util::op_n(),
            blockMatrix.l_rows(), subSize, blockMatrix.l_cols(), alpha,
            blockMatrix.l_data(), blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &beta_tmp,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld()));

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize, ncclSum,
            input_multiVector.getMpiGrid()->get_nccl_row_comm(), &used_stream));
    }
}

} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
