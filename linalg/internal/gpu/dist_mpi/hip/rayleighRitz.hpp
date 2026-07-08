// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/rayleighRitz_impl.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{

template <typename MatrixType, typename InputMultiVectorType>
void hip_mpi::rayleighRitz(
    rocblas_handle blas_handle, rocblas_handle /*solver_handle*/,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* workspace, int lwork_heevd,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    chase::linalg::internal::gpu_mpi::rayleigh_ritz_impl<
        chase::linalg::internal::gpu_backend::hip_backend,
        chase::linalg::internal::hip_mpi>(
        blas_handle, blas_handle, H, V1, V2, W1, W2, ritzv, offset, subSize,
        devInfo, workspace, lwork_heevd, A);
}

} // namespace internal
} // namespace linalg
} // namespace chase
