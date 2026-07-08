// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/shift_diagonal_impl.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename MatrixType>
void hip_mpi::shiftDiagonal(MatrixType& H, std::size_t* d_off_m,
                            std::size_t* d_off_n, std::size_t offsize,
                            chase::Base<typename MatrixType::value_type> shift)
{
    gpu_mpi::shift_diagonal_impl<gpu_backend::hip_backend>(
        H, d_off_m, d_off_n, offsize, shift);
}

} // namespace internal
} // namespace linalg
} // namespace chase
