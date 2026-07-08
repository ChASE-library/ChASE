// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/dist_xccl/nccl/nccl_kernels.hpp"
#include <omp.h>

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename T>
void cuda_nccl::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H)
{
    gpu_device::kernels<gpu_backend::cuda_backend>::flip_lower_block_sign(
        H.l_data(), H.l_ld(), H.l_half(), H.l_rows() - H.l_half(), H.l_cols());
}

template <typename T>
void cuda_nccl::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H)
{
    gpu_device::kernels<gpu_backend::cuda_backend>::flip_lower_block_sign(
        H.l_data(), H.l_ld(), H.l_half(), H.l_rows() - H.l_half(), H.l_cols());
}

template <typename InputMultiVectorType>
void cuda_nccl::flipLowerHalfMatrixSign(InputMultiVectorType& V)
{
    gpu_device::kernels<gpu_backend::cuda_backend>::flip_lower_block_sign(
        V.l_data(), V.l_ld(), V.l_half(), V.l_rows() - V.l_half(), V.l_cols());
}

template <typename InputMultiVectorType>
void cuda_nccl::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                        cudaStream_t stream)
{
    gpu_device::kernels<gpu_backend::cuda_backend>::flip_lower_block_sign(
        V.l_data(), V.l_ld(), V.l_half(), V.l_rows() - V.l_half(), V.l_cols(),
        stream);
}

template <typename InputMultiVectorType>
void cuda_nccl::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                        std::size_t offset, std::size_t subSize)
{
    gpu_device::kernels<gpu_backend::cuda_backend>::flip_lower_block_sign(
        V.l_data() + offset * V.l_ld(), V.l_ld(), V.l_half(),
        V.l_rows() - V.l_half(), subSize);
}

template <typename InputMultiVectorType>
void cuda_nccl::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                        std::size_t offset, std::size_t subSize,
                                        cudaStream_t stream)
{
    gpu_device::kernels<gpu_backend::cuda_backend>::flip_lower_block_sign(
        V.l_data() + offset * V.l_ld(), V.l_ld(), V.l_half(),
        V.l_rows() - V.l_half(), subSize, stream);
}

} // namespace internal
} // namespace linalg
} // namespace chase
