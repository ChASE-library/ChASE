// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_xccl/impl/residuals_impl.hpp"
#include "linalg/internal/gpu/dist_xccl/nccl/nccl_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::residuals(
    cublasHandle_t cublas_handle, MatrixType& H, InputMultiVectorType& V1,
    InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                          typename MatrixType::platform_type>& ritzv,
    chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                          typename MatrixType::platform_type>& resids,
    std::size_t offset, std::size_t subSize)
{
    gpu_xccl::residuals_impl<gpu_backend::cuda_backend, cuda_nccl>(
        cublas_handle, H, V1, V2, W1, W2, ritzv, resids, offset, subSize);
}

} // namespace internal
} // namespace linalg
} // namespace chase
