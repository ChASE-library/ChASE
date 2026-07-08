// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/flip_sign_impl.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename T>
void hip_mpi::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H)
{
    gpu_mpi::flip_lower_half_matrix_sign_impl<gpu_backend::hip_backend>(
        H);
}

template <typename T>
void hip_mpi::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H)
{
    gpu_mpi::flip_lower_half_matrix_sign_impl<gpu_backend::hip_backend>(
        H);
}

template <typename InputMultiVectorType>
void hip_mpi::flipLowerHalfMatrixSign(InputMultiVectorType& V)
{
    gpu_mpi::flip_lower_half_matrix_sign_impl<gpu_backend::hip_backend>(
        V);
}

template <typename InputMultiVectorType>
void hip_mpi::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                      std::size_t offset, std::size_t subSize)
{
    gpu_mpi::flip_lower_half_matrix_sign_impl<gpu_backend::hip_backend>(
        V, offset, subSize);
}

} // namespace internal
} // namespace linalg
} // namespace chase
