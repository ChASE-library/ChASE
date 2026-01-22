// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda/flipSign.cuh"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#include <omp.h>

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename T>
void cuda_mpi::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H)
{
    chase::linalg::internal::cuda::chase_flipMatrixSign(
        H.l_data() + H.l_half(), H.l_rows() - H.l_half(), H.l_cols(), H.l_ld(),
        (cudaStream_t)0);
}

template <typename T>
void cuda_mpi::flipLowerHalfMatrixSign(
    chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H)
{
    chase::linalg::internal::cuda::chase_flipMatrixSign(
        H.l_data() + H.l_half(), H.l_rows() - H.l_half(), H.l_cols(), H.l_ld(),
        (cudaStream_t)0);
}

template <typename InputMultiVectorType>
void cuda_mpi::flipLowerHalfMatrixSign(InputMultiVectorType& V)
{
    chase::linalg::internal::cuda::chase_flipMatrixSign(
        V.l_data() + V.l_half(), V.l_rows() - V.l_half(), V.l_cols(), V.l_ld(),
        (cudaStream_t)0);
}

template <typename InputMultiVectorType>
void cuda_mpi::flipLowerHalfMatrixSign(InputMultiVectorType& V,
                                       std::size_t offset, std::size_t subSize)
{
    chase::linalg::internal::cuda::chase_flipMatrixSign(
        V.l_data() + V.l_half() + offset * V.l_ld(), V.l_rows() - V.l_half(),
        subSize, V.l_ld(), (cudaStream_t)0);
}
} // namespace internal
} // namespace linalg
} // namespace chase
