// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cstddef>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
// ----- In-place conjugation (no transposition). Real types: no-op. -----
__global__ void sconjugate_inplace(float* A, std::size_t m, std::size_t n,
                                   std::size_t lda);
__global__ void dconjugate_inplace(double* A, std::size_t m, std::size_t n,
                                   std::size_t lda);
__global__ void cconjugate_inplace(cuComplex* A, std::size_t m, std::size_t n,
                                   std::size_t lda);
__global__ void zconjugate_inplace(cuDoubleComplex* A, std::size_t m,
                                   std::size_t n, std::size_t lda);

// Host launchers (stream may be 0 for default)
void chase_conjugate_inplace(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, cudaStream_t stream_);
void chase_conjugate_inplace(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, cudaStream_t stream_);
void chase_conjugate_inplace(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             cudaStream_t stream_);
void chase_conjugate_inplace(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda,
                             cudaStream_t stream_);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
