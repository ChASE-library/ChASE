// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/dist_xccl/impl/matrix_inf_norm_impl.hpp"
#include "linalg/internal/gpu/dist_xccl/nccl/nccl_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename MatrixType>
void cuda_nccl::matrixInfNorm(
    MatrixType& H,
    chase::Base<typename MatrixType::value_type>* d_row_sums,
    chase::Base<typename MatrixType::value_type>* d_result, cudaStream_t stream)
{
    gpu_xccl::matrixInfNorm_impl<gpu_backend::cuda_backend>(H, d_row_sums,
                                                            d_result, stream);
}
} // namespace internal
} // namespace linalg
} // namespace chase
