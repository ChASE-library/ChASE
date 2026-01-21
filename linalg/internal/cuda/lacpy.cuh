// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
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
static __device__ void dlacpy_full_device(std::size_t m, std::size_t n,
                                          const double* dA, std::size_t ldda,
                                          double* dB, std::size_t lddb);

__global__ void dlacpy_full_kernel(std::size_t m, std::size_t n,
                                   const double* dA, std::size_t ldda,
                                   double* dB, std::size_t lddb);

static __device__ void slacpy_full_device(std::size_t m, std::size_t n,
                                          const float* dA, std::size_t ldda,
                                          float* dB, std::size_t lddb);

__global__ void slacpy_full_kernel(std::size_t m, std::size_t n,
                                   const float* dA, std::size_t ldda, float* dB,
                                   std::size_t lddb);

static __device__ void zlacpy_full_device(std::size_t m, std::size_t n,
                                          const cuDoubleComplex* dA,
                                          std::size_t ldda, cuDoubleComplex* dB,
                                          std::size_t lddb);

__global__ void zlacpy_full_kernel(std::size_t m, std::size_t n,
                                   const cuDoubleComplex* dA, std::size_t ldda,
                                   cuDoubleComplex* dB, std::size_t lddb);

static __device__ void clacpy_full_device(std::size_t m, std::size_t n,
                                          const cuComplex* dA, std::size_t ldda,
                                          cuComplex* dB, std::size_t lddb);

__global__ void clacpy_full_kernel(std::size_t m, std::size_t n,
                                   const cuComplex* dA, std::size_t ldda,
                                   cuComplex* dB, std::size_t lddb);

static __device__ void dlacpy_upper_device(std::size_t m, std::size_t n,
                                           const double* dA, std::size_t ldda,
                                           double* dB, std::size_t lddb);

__global__ void dlacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const double* dA, std::size_t ldda,
                                    double* dB, std::size_t lddb);

static __device__ void slacpy_upper_device(std::size_t m, std::size_t n,
                                           const float* dA, std::size_t ldda,
                                           float* dB, std::size_t lddb);

__global__ void slacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const float* dA, std::size_t ldda,
                                    float* dB, std::size_t lddb);

static __device__ void clacpy_upper_device(std::size_t m, std::size_t n,
                                           const cuComplex* dA,
                                           std::size_t ldda, cuComplex* dB,
                                           std::size_t lddb);

__global__ void clacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const cuComplex* dA, std::size_t ldda,
                                    cuComplex* dB, std::size_t lddb);

static __device__ void
zlacpy_upper_device(std::size_t m, std::size_t n, const cuDoubleComplex* dA,
                    std::size_t ldda, cuDoubleComplex* dB, std::size_t lddb);

__global__ void zlacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const cuDoubleComplex* dA, std::size_t ldda,
                                    cuDoubleComplex* dB, std::size_t lddb);

static __device__ void dlacpy_lower_device(std::size_t m, std::size_t n,
                                           const double* dA, std::size_t ldda,
                                           double* dB, std::size_t lddb);

__global__ void dlacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const double* dA, std::size_t ldda,
                                    double* dB, std::size_t lddb);

static __device__ void slacpy_lower_device(std::size_t m, std::size_t n,
                                           const float* dA, std::size_t ldda,
                                           float* dB, std::size_t lddb);

__global__ void slacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const float* dA, std::size_t ldda,
                                    float* dB, std::size_t lddb);

static __device__ void clacpy_lower_device(std::size_t m, std::size_t n,
                                           const cuComplex* dA,
                                           std::size_t ldda, cuComplex* dB,
                                           std::size_t lddb);

__global__ void clacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const cuComplex* dA, std::size_t ldda,
                                    cuComplex* dB, std::size_t lddb);

static __device__ void
zlacpy_lower_device(std::size_t m, std::size_t n, const cuDoubleComplex* dA,
                    std::size_t ldda, cuDoubleComplex* dB, std::size_t lddb);

__global__ void zlacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const cuDoubleComplex* dA, std::size_t ldda,
                                    cuDoubleComplex* dB, std::size_t lddb);

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, float* dA,
                 std::size_t ldda, float* dB, std::size_t lddb,
                 cudaStream_t stream_);
void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, double* dA,
                 std::size_t ldda, double* dB, std::size_t lddb,
                 cudaStream_t stream_);
void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n,
                 std::complex<double>* ddA, std::size_t ldda,
                 std::complex<double>* ddB, std::size_t lddb,
                 cudaStream_t stream_);
void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n,
                 std::complex<float>* ddA, std::size_t ldda,
                 std::complex<float>* ddB, std::size_t lddb,
                 cudaStream_t stream_);

__global__ void d_extractUpperTriangularKernel(const double* matrix,
                                               std::size_t ld,
                                               double* upperTriangular,
                                               std::size_t n);
__global__ void s_extractUpperTriangularKernel(const float* matrix,
                                               std::size_t ld,
                                               float* upperTriangular,
                                               std::size_t n);
__global__ void c_extractUpperTriangularKernel(const cuComplex* matrix,
                                               std::size_t ld,
                                               cuComplex* upperTriangular,
                                               std::size_t n);
__global__ void z_extractUpperTriangularKernel(const cuDoubleComplex* matrix,
                                               std::size_t ld,
                                               cuDoubleComplex* upperTriangular,
                                               std::size_t n);

void extractUpperTriangular(float* d_matrix, std::size_t ld,
                            float* d_upperTriangular, std::size_t n,
                            cudaStream_t stream_);
void extractUpperTriangular(double* d_matrix, std::size_t ld,
                            double* d_upperTriangular, std::size_t n,
                            cudaStream_t stream_);
void extractUpperTriangular(std::complex<double>* d_matrix, std::size_t ld,
                            std::complex<double>* d_upperTriangular,
                            std::size_t n, cudaStream_t stream_);
void extractUpperTriangular(std::complex<float>* d_matrix, std::size_t ld,
                            std::complex<float>* d_upperTriangular,
                            std::size_t n, cudaStream_t stream_);

__global__ void s_unpackUpperTriangularKernel(const float* upperTriangular,
                                              std::size_t ld, float* matrix,
                                              std::size_t n);
__global__ void d_unpackUpperTriangularKernel(const double* upperTriangular,
                                              std::size_t ld, double* matrix,
                                              std::size_t n);
__global__ void c_unpackUpperTriangularKernel(const cuComplex* upperTriangular,
                                              std::size_t ld, cuComplex* matrix,
                                              std::size_t n);
__global__ void
z_unpackUpperTriangularKernel(const cuDoubleComplex* upperTriangular,
                              std::size_t ld, cuDoubleComplex* matrix,
                              std::size_t n);

void unpackUpperTriangular(float* d_matrix, std::size_t ld,
                           float* d_upperTriangular, std::size_t n,
                           cudaStream_t stream_);
void unpackUpperTriangular(double* d_matrix, std::size_t ld,
                           double* d_upperTriangular, std::size_t n,
                           cudaStream_t stream_);
void unpackUpperTriangular(std::complex<double>* d_matrix, std::size_t ld,
                           std::complex<double>* d_upperTriangular,
                           std::size_t n, cudaStream_t stream_);
void unpackUpperTriangular(std::complex<float>* d_matrix, std::size_t ld,
                           std::complex<float>* d_upperTriangular,
                           std::size_t n, cudaStream_t stream_);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase