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

void convert_DP_TO_SP_GPU(double* d_input, float* d_output, std::size_t size,
                          hipStream_t stream = nullptr);

void convert_DP_TO_SP_GPU(std::complex<double>* d_input,
                          std::complex<float>* d_output, std::size_t size,
                          hipStream_t stream = nullptr);

void convert_SP_TO_DP_GPU(float* d_input, double* d_output, std::size_t size,
                          hipStream_t stream = nullptr);

void convert_SP_TO_DP_GPU(std::complex<float>* d_input,
                          std::complex<double>* d_output, std::size_t size,
                          hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase
