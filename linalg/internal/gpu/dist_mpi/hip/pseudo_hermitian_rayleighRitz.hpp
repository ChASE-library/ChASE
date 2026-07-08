// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/dist_mpi/impl/pseudo_hermitian_rayleighRitz_impl.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename MatrixType, typename InputMultiVectorType>
void hip_mpi::pseudo_hermitian_rayleighRitz(
    rocblas_handle blas_handle, rocblas_handle solver_handle, void* params,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* d_workspace, int d_lwork,
    typename MatrixType::value_type* h_workspace, int h_lwork,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    gpu_mpi::pseudo_hermitian_rayleighRitz_impl<
        gpu_backend::hip_backend, rocblas_handle, rocblas_handle, void*,
        hip_mpi>(blas_handle, solver_handle, params, H, V1, V2, W1, W2, ritzv,
                 offset, subSize, devInfo, d_workspace, d_lwork, h_workspace,
                 h_lwork, A);
}

template <typename MatrixType, typename InputMultiVectorType>
void hip_mpi::pseudo_hermitian_rayleighRitz_v2(
    rocblas_handle blas_handle, rocblas_handle solver_handle, void* params,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* workspace, int lwork,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    gpu_mpi::pseudo_hermitian_rayleighRitz_v2_impl<
        gpu_backend::hip_backend, rocblas_handle, rocblas_handle, void*,
        hip_mpi>(blas_handle, solver_handle, params, H, V1, V2, W1, W2, ritzv,
                 offset, subSize, devInfo, workspace, lwork, A);
}

} // namespace internal
} // namespace linalg
} // namespace chase
