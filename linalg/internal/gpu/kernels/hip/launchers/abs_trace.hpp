// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
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

void absTrace_gpu(float* d_matrix, float* d_trace, std::size_t n, std::size_t ld,
                  hipStream_t stream = nullptr);

void absTrace_gpu(double* d_matrix, double* d_trace, std::size_t n,
                  std::size_t ld, hipStream_t stream = nullptr);

void absTrace_gpu(std::complex<float>* d_matrix, float* d_trace, std::size_t n,
                  std::size_t ld, hipStream_t stream = nullptr);

void absTrace_gpu(std::complex<double>* d_matrix, double* d_trace, std::size_t n,
                  std::size_t ld, hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase
