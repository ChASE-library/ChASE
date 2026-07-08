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
#include "linalg/internal/gpu/dist_xccl/rccl/rccl_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace detail
{

template <typename T>
inline void rccl_flip_lower_half_block(T* data, std::size_t ld,
                                       std::size_t row_offset,
                                       std::size_t num_rows, std::size_t num_cols,
                                       hipStream_t stream = nullptr)
{
    gpu_device::kernels<gpu_backend::hip_backend>::flip_lower_block_sign(
        data, ld, row_offset, num_rows, num_cols, stream);
}

} // namespace detail

template <typename T>
void hip_rccl::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H)
{
    detail::rccl_flip_lower_half_block(H.l_data(), H.l_ld(), H.l_half(),
                                       H.l_rows() - H.l_half(), H.l_cols());
}

template <typename T>
void hip_rccl::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H)
{
    detail::rccl_flip_lower_half_block(H.l_data(), H.l_ld(), H.l_half(),
                                       H.l_rows() - H.l_half(), H.l_cols());
}

template <typename InputMultiVectorType>
void hip_rccl::flipLowerHalfMatrixSign(InputMultiVectorType& V)
{
    using T = typename InputMultiVectorType::value_type;
    detail::rccl_flip_lower_half_block(
        V.l_data(), V.l_ld(), V.l_half(), V.l_rows() - V.l_half(), V.l_cols());
}

template <typename InputMultiVectorType>
void hip_rccl::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                       hipStream_t stream)
{
    using T = typename InputMultiVectorType::value_type;
    detail::rccl_flip_lower_half_block(
        V.l_data(), V.l_ld(), V.l_half(), V.l_rows() - V.l_half(), V.l_cols(),
        stream);
}

template <typename InputMultiVectorType>
void hip_rccl::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                       std::size_t offset, std::size_t subSize)
{
    using T = typename InputMultiVectorType::value_type;
    detail::rccl_flip_lower_half_block(
        V.l_data() + offset * V.l_ld(), V.l_ld(), V.l_half(),
        V.l_rows() - V.l_half(), subSize);
}

template <typename InputMultiVectorType>
void hip_rccl::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                       std::size_t offset, std::size_t subSize,
                                       hipStream_t stream)
{
    using T = typename InputMultiVectorType::value_type;
    detail::rccl_flip_lower_half_block(
        V.l_data() + offset * V.l_ld(), V.l_ld(), V.l_half(),
        V.l_rows() - V.l_half(), subSize, stream);
}

} // namespace internal
} // namespace linalg
} // namespace chase
