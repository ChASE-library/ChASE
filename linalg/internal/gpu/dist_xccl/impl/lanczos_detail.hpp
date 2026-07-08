// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <stdexcept>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_xccl
{
namespace detail
{

template <typename MatrixType, typename InputMultiVectorType>
void validate_lanczos_inputs(MatrixType& H, InputMultiVectorType& V)
{
    if (H.g_cols() != H.g_rows())
    {
        throw std::runtime_error("Lanczos requires matrix to be squared");
    }
    if (H.getMpiGrid() != V.getMpiGrid())
    {
        throw std::runtime_error("Lanczos requires H and V in same MPI grid");
    }
    if (H.g_rows() != V.g_rows())
    {
        throw std::runtime_error("Lanczos H and V have same number of rows");
    }
}

} // namespace detail
} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
