// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{
template <typename Backend, typename T>
void flip_lower_half_matrix_sign_impl(
    chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H)
{
    gpu_device::kernels<Backend>::flip_lower_block_sign(
        H.l_data(), H.l_ld(), H.l_half(), H.l_rows() - H.l_half(), H.l_cols());
}

template <typename Backend, typename T>
void flip_lower_half_matrix_sign_impl(
    chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H)
{
    gpu_device::kernels<Backend>::flip_lower_block_sign(
        H.l_data(), H.l_ld(), H.l_half(), H.l_rows() - H.l_half(), H.l_cols());
}

template <typename Backend, typename InputMultiVectorType>
void flip_lower_half_matrix_sign_impl(InputMultiVectorType& V)
{
    using T = typename InputMultiVectorType::value_type;
    gpu_device::kernels<Backend>::flip_lower_block_sign(
        V.l_data(), V.l_ld(), V.l_half(), V.l_rows() - V.l_half(), V.l_cols());
}

template <typename Backend, typename InputMultiVectorType>
void flip_lower_half_matrix_sign_impl(InputMultiVectorType& V, std::size_t offset,
                                      std::size_t subSize)
{
    using T = typename InputMultiVectorType::value_type;
    gpu_device::kernels<Backend>::flip_lower_block_sign(
        V.l_data() + offset * V.l_ld(), V.l_ld(), V.l_half(),
        V.l_rows() - V.l_half(), subSize);
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
