// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{

template <typename Backend, typename MatrixType>
void shift_diagonal_impl(MatrixType& H, std::size_t* d_off_m,
                         std::size_t* d_off_n, std::size_t offsize,
                         chase::Base<typename MatrixType::value_type> shift)
{
    gpu_device::kernels<Backend>::shift(
        H.l_data(), d_off_m, d_off_n, offsize, H.l_ld(), shift);
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
