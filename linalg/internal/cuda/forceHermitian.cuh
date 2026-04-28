// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

void force_hermitian_gpu(float* A, std::size_t n, std::size_t lda,
                         cudaStream_t stream);
void force_hermitian_gpu(double* A, std::size_t n, std::size_t lda,
                          cudaStream_t stream);
void force_hermitian_gpu(std::complex<float>* A, std::size_t n,
                         std::size_t lda, cudaStream_t stream);
void force_hermitian_gpu(std::complex<double>* A, std::size_t n,
                         std::size_t lda, cudaStream_t stream);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
