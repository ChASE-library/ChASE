// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cstddef>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{

void chase_rand_normal(unsigned long long seed,
                       hiprandStatePhilox4_32_10_t* states, float* v,
                       std::size_t n, hipStream_t stream = nullptr);

void chase_rand_normal(unsigned long long seed,
                       hiprandStatePhilox4_32_10_t* states, double* v,
                       std::size_t n, hipStream_t stream = nullptr);

void chase_rand_normal(unsigned long long seed,
                       hiprandStatePhilox4_32_10_t* states,
                       std::complex<float>* v, std::size_t n,
                       hipStream_t stream = nullptr);

void chase_rand_normal(unsigned long long seed,
                       hiprandStatePhilox4_32_10_t* states,
                       std::complex<double>* v, std::size_t n,
                       hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase
