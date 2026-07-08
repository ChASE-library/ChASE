// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once
#include <cfloat>
#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

// generate `n` random float numbers on GPU
__global__ void s_normal_kernel(unsigned long long seed,
                                curandStatePhilox4_32_10_t* states, float* v,
                                std::size_t n);
// generate `n` random double numbers on GPU
__global__ void d_normal_kernel(unsigned long long seed,
                                curandStatePhilox4_32_10_t* states, double* v,
                                std::size_t n);
// generate `n` random complex single numbers on GPU
__global__ void c_normal_kernel(unsigned long long seed,
                                curandStatePhilox4_32_10_t* states,
                                cuComplex* v, std::size_t n);

// generate `n` random complex double numbers on GPU
__global__ void z_normal_kernel(unsigned long long seed,
                                curandStatePhilox4_32_10_t* states,
                                cuDoubleComplex* v, std::size_t n);

void chase_rand_normal(unsigned long long seed,
                       curandStatePhilox4_32_10_t* states, float* v,
                       std::size_t n, cudaStream_t stream_);

void chase_rand_normal(unsigned long long seed,
                       curandStatePhilox4_32_10_t* states, double* v,
                       std::size_t n, cudaStream_t stream_);

void chase_rand_normal(unsigned long long seed,
                       curandStatePhilox4_32_10_t* states,
                       std::complex<float>* v, std::size_t n,
                       cudaStream_t stream_);

void chase_rand_normal(unsigned long long seed,
                       curandStatePhilox4_32_10_t* states,
                       std::complex<double>* v, std::size_t n,
                       cudaStream_t stream_);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase