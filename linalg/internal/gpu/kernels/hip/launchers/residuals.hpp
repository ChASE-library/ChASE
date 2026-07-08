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

void residual_gpu(std::size_t m, std::size_t n, std::complex<double>* dA,
                  std::size_t lda, std::complex<double>* dB, std::size_t ldb,
                  double* d_ritzv, double* d_resids, bool is_sqrt,
                  hipStream_t stream = nullptr);

void residual_gpu(std::size_t m, std::size_t n, std::complex<float>* dA,
                  std::size_t lda, std::complex<float>* dB, std::size_t ldb,
                  float* d_ritzv, float* d_resids, bool is_sqrt,
                  hipStream_t stream = nullptr);

void residual_gpu(std::size_t m, std::size_t n, double* dA, std::size_t lda,
                  double* dB, std::size_t ldb, double* d_ritzv,
                  double* d_resids, bool is_sqrt, hipStream_t stream = nullptr);

void residual_gpu(std::size_t m, std::size_t n, float* dA, std::size_t lda,
                  float* dB, std::size_t ldb, float* d_ritzv, float* d_resids,
                  bool is_sqrt, hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase
