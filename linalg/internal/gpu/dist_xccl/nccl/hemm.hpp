// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/typeTraits.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_xccl/impl/hemm_impl.hpp"
#include "linalg/internal/gpu/dist_xccl/nccl/nccl_kernels.hpp"
#ifdef CHASE_ENABLE_OZAKI
#include "external/gemmul8pp/gemmul8pp.hpp"
#endif

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename T, typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::MatrixMultiplyMultiVectors(
    cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector, T* beta,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    std::size_t offset, std::size_t subSize)
{
    gpu_xccl::matrix_multiply_multi_vectors_impl<
        gpu_backend::cuda_backend, cuda_nccl, cublasHandle_t, T,
        MatrixType, InputMultiVectorType>(
        cublas_handle, alpha, blockMatrix, input_multiVector, beta,
        result_multiVector, offset, subSize);
}

template <typename T, typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::MatrixMultiplyMultiVectors(
    cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector, T* beta,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector)
{
    cuda_nccl::MatrixMultiplyMultiVectors(
        cublas_handle, alpha, blockMatrix, input_multiVector, beta,
        result_multiVector, 0, input_multiVector.l_cols());
}

// this operation do: W1<-1.0 * H * V1, while redistribute V2 to W2
template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
    cublasHandle_t cublas_handle, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    InputMultiVectorType& src_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        target_multiVector,
    std::size_t offset, std::size_t subSize)
{
    gpu_xccl::matrix_multiply_multi_vectors_and_redistribute_async_impl<
        gpu_backend::cuda_backend, cuda_nccl, cublasHandle_t>(
        cublas_handle, blockMatrix, input_multiVector, result_multiVector,
        src_multiVector, target_multiVector, offset, subSize);
}

template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
    cublasHandle_t cublas_handle, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    InputMultiVectorType& src_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        target_multiVector)
{
    MatrixMultiplyMultiVectorsAndRedistributeAsync(
        cublas_handle, blockMatrix, input_multiVector, result_multiVector,
        src_multiVector, target_multiVector, 0, input_multiVector.l_cols());
}

#ifdef CHASE_ENABLE_OZAKI
template <typename T, typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::MatrixMultiplyMultiVectorsWithOzakiII(
    cublasHandle_t cublas_handle, T* alpha, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector, T* beta,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    std::size_t offset, std::size_t subSize,
    unsigned ozaki_num_moduli_, bool ozaki_fastmode_,
    int8_t* ozaki_workspace_blockmatrix, int8_t* ozaki_workspace_input_multiVector, int8_t* ozaki_workspace_result_multiVector)
{
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
    cudaStream_t used_stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &used_stream));

    if constexpr (ExtractCommType<InputMultiVectorType>::value ==
                  chase::distMultiVector::CommunicatorType::column)
    {
        beta_tmp = (coords[0] != 0) ? T(0.0) : *beta;
        if constexpr (std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                              T, chase::platform::GPU>>::value ||
                      std::is_same<
                          MatrixType,
                          chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                              T, chase::platform::GPU>>::value)
        {
            cuda_nccl::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize, used_stream);
        }
        if (beta_tmp != T(0.0))
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
                cuda_nccl::flipLowerHalfMatrixSign(
                    result_multiVector, offset, subSize, used_stream);
            }
        }
        CHECK_CUBLAS_ERROR(chase::linalg::gemmul8pp::gemmul8Tgemm(
            cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N, blockMatrix.l_cols(),
            subSize, blockMatrix.l_rows(), alpha, blockMatrix.l_data(),
            blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &beta_tmp,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld(),
            ozaki_num_moduli_, ozaki_fastmode_,
            static_cast<void *>(ozaki_workspace_result_multiVector),
            static_cast<void *>(ozaki_workspace_blockmatrix),
            static_cast<void *>(ozaki_workspace_input_multiVector),
            true, true, false, false));
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
            cuda_nccl::flipLowerHalfMatrixSign(
                input_multiVector, offset, subSize, used_stream);
            cuda_nccl::flipLowerHalfMatrixSign(
                result_multiVector, offset, subSize, used_stream);
        }
    }
    else
    {
        beta_tmp = (coords[1] != 0) ? T(0.0) : *beta;
        CHECK_CUBLAS_ERROR(chase::linalg::gemmul8pp::gemmul8Tgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, blockMatrix.l_rows(),
            subSize, blockMatrix.l_cols(), alpha, blockMatrix.l_data(),
            blockMatrix.l_ld(),
            input_multiVector.l_data() + offset * input_multiVector.l_ld(),
            input_multiVector.l_ld(), &beta_tmp,
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld(),
            ozaki_num_moduli_, ozaki_fastmode_,
            static_cast<void *>(ozaki_workspace_result_multiVector),
            static_cast<void *>(ozaki_workspace_blockmatrix),
            static_cast<void *>(ozaki_workspace_input_multiVector),
            true, true, false, false));
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_data() + offset * result_multiVector.l_ld(),
            result_multiVector.l_ld() * subSize, ncclSum,
            input_multiVector.getMpiGrid()->get_nccl_row_comm(), &used_stream));
    }
}
#endif

} // namespace internal
} // namespace linalg
} // namespace chase
