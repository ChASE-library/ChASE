// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/dist_mpi/impl/sym_or_herm_impl.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename MatrixType>
bool hip_mpi::checkSymmetryEasy(rocblas_handle blas_handle, MatrixType& H)
{
    return gpu_mpi::check_symmetry_easy_impl<
        gpu_backend::hip_backend, hip_mpi, rocblas_handle>(blas_handle, H);
}

template <typename T>
void hip_mpi::symOrHermMatrix(
    char uplo, chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H)
{
    gpu_mpi::sym_or_herm_matrix_impl(uplo, H);
}

template <typename T>
void hip_mpi::symOrHermMatrix(
    char uplo, chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H)
{
    gpu_mpi::sym_or_herm_matrix_impl(uplo, H);
}

} // namespace internal
} // namespace linalg
} // namespace chase
