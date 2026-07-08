// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/dist_xccl/impl/hemm_impl.hpp"
#include "linalg/internal/gpu/dist_xccl/rccl/rccl_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename T, typename MatrixType, typename InputMultiVectorType>
void hip_rccl::MatrixMultiplyMultiVectors(
    rocblas_handle blas_handle, T* alpha, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector, T* beta,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    std::size_t offset, std::size_t subSize)
{
    gpu_xccl::matrix_multiply_multi_vectors_impl<
        gpu_backend::hip_backend, hip_rccl, rocblas_handle, T,
        MatrixType, InputMultiVectorType>(
        blas_handle, alpha, blockMatrix, input_multiVector, beta,
        result_multiVector, offset, subSize);
}

template <typename T, typename MatrixType, typename InputMultiVectorType>
void hip_rccl::MatrixMultiplyMultiVectors(
    rocblas_handle blas_handle, T* alpha, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector, T* beta,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector)
{
    hip_rccl::MatrixMultiplyMultiVectors(
        blas_handle, alpha, blockMatrix, input_multiVector, beta,
        result_multiVector, 0, input_multiVector.l_cols());
}

template <typename MatrixType, typename InputMultiVectorType>
void hip_rccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
    rocblas_handle blas_handle, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    InputMultiVectorType& src_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        target_multiVector,
    std::size_t offset, std::size_t subSize)
{
    gpu_xccl::matrix_multiply_multi_vectors_and_redistribute_async_impl<
        gpu_backend::hip_backend, hip_rccl, rocblas_handle>(
        blas_handle, blockMatrix, input_multiVector, result_multiVector,
        src_multiVector, target_multiVector, offset, subSize);
}

template <typename MatrixType, typename InputMultiVectorType>
void hip_rccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
    rocblas_handle blas_handle, MatrixType& blockMatrix,
    InputMultiVectorType& input_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        result_multiVector,
    InputMultiVectorType& src_multiVector,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type&
        target_multiVector)
{
    hip_rccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
        blas_handle, blockMatrix, input_multiVector, result_multiVector,
        src_multiVector, target_multiVector, 0, input_multiVector.l_cols());
}

} // namespace internal
} // namespace linalg
} // namespace chase
