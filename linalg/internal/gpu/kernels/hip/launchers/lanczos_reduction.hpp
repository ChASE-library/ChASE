// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{

void lanczos_dot_float_double_accum(std::size_t n, const float* x,
                                    std::size_t incx, const float* y,
                                    std::size_t incy, double* d_result,
                                    hipStream_t stream = nullptr);

void lanczos_nrm2_float_double_accum(std::size_t n, const float* x,
                                     std::size_t incx, double* d_result,
                                     hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase
