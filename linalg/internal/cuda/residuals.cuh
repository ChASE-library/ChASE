// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iostream>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
template <int n, typename T>
__device__ void cuda_sum_reduce(int i, T* x);
__global__ void c_resids_kernel(std::size_t m, std::size_t n,
                                const cuComplex* A, std::size_t lda,
                                const cuComplex* B, std::size_t ldb,
                                float* ritzv, float* resids, bool is_sqrt);

__global__ void z_resids_kernel(std::size_t m, std::size_t n,
                                const cuDoubleComplex* A, std::size_t lda,
                                const cuDoubleComplex* B, std::size_t ldb,
                                double* ritzv, double* resids, bool is_sqrt);

__global__ void d_resids_kernel(std::size_t m, std::size_t n, const double* A,
                                std::size_t lda, const double* B,
                                std::size_t ldb, double* ritzv, double* resids,
                                bool is_sqrt);

__global__ void s_resids_kernel(std::size_t m, std::size_t n, const float* A,
                                std::size_t lda, const float* B,
                                std::size_t ldb, float* ritzv, float* resids,
                                bool is_sqrt);

void residual_gpu(std::size_t m, std::size_t n, std::complex<double>* dA,
                  std::size_t lda, std::complex<double>* dB, std::size_t ldb,
                  double* d_ritzv, double* d_resids, bool is_sqrt,
                  cudaStream_t stream_);

void residual_gpu(std::size_t m, std::size_t n, std::complex<float>* dA,
                  std::size_t lda, std::complex<float>* dB, std::size_t ldb,
                  float* d_ritzv, float* d_resids, bool is_sqrt,
                  cudaStream_t stream_);

void residual_gpu(std::size_t m, std::size_t n, double* dA, std::size_t lda,
                  double* dB, std::size_t ldb, double* d_ritzv,
                  double* d_resids, bool is_sqrt, cudaStream_t stream_);

void residual_gpu(std::size_t m, std::size_t n, float* dA, std::size_t lda,
                  float* dB, std::size_t ldb, float* d_ritzv, float* d_resids,
                  bool is_sqrt, cudaStream_t stream_);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase