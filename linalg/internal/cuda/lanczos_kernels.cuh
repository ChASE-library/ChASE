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

// Forward declarations of CUDA kernels

/**
 * @brief Square elements in-place on GPU
 */
template<typename T>
__global__ void square_inplace_kernel(T* data, int n);

/**
 * @brief Negate array elements in-place
 */
template<typename T>
__global__ void negate_kernel(T* data, int n);

/**
 * @brief Take square root in-place
 */
template<typename T>
__global__ void sqrt_inplace_kernel(T* data, int n);

/**
 * @brief Fused normalization: compute 1/sqrt(norm_squared) and scale vectors
 */
template<typename T, typename RealT>
__global__ void normalize_vectors_kernel(
    T* vectors,
    const RealT* norms_squared,
    int rows,
    int numvec,
    int ld);

/**
 * @brief Batched dot product for multiple vector pairs
 */
template<typename T>
__global__ void batched_dot_product_kernel(
    const T* v1,
    const T* v2,
    T* results,
    int rows,
    int numvec,
    int ld,
    bool negate);

/**
 * @brief Batched norm computation (squared)
 */
template<typename T, typename RealT>
__global__ void batched_norm_squared_kernel(
    const T* v,
    RealT* norms_squared,
    int rows,
    int numvec,
    int ld);

// Host callable functions for launching kernels

void square_inplace_gpu(double* data, int n, cudaStream_t stream);
void square_inplace_gpu(float* data, int n, cudaStream_t stream);

void negate_gpu(double* data, int n, cudaStream_t stream);
void negate_gpu(float* data, int n, cudaStream_t stream);
void negate_gpu(cuDoubleComplex* data, int n, cudaStream_t stream);
void negate_gpu(cuComplex* data, int n, cudaStream_t stream);

void sqrt_inplace_gpu(double* data, int n, cudaStream_t stream);
void sqrt_inplace_gpu(float* data, int n, cudaStream_t stream);

// Batched sqrt: data[i] = sqrt(data[i])
void batched_sqrt_gpu(double* data, int n, cudaStream_t stream);
void batched_sqrt_gpu(float* data, int n, cudaStream_t stream);

void normalize_vectors_gpu(cuDoubleComplex* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);
void normalize_vectors_gpu(cuComplex* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);
void normalize_vectors_gpu(double* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);
void normalize_vectors_gpu(float* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);

void batched_dot_product_gpu(const cuDoubleComplex* v1, const cuDoubleComplex* v2,
                             cuDoubleComplex* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);
void batched_dot_product_gpu(const cuComplex* v1, const cuComplex* v2,
                             cuComplex* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);
void batched_dot_product_gpu(const double* v1, const double* v2,
                             double* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);
void batched_dot_product_gpu(const float* v1, const float* v2,
                             float* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);

// Batched scaling: v[:,i] *= scale[i]
void batched_scale_gpu(const cuDoubleComplex* scale, cuDoubleComplex* v,
                      int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_gpu(const cuComplex* scale, cuComplex* v,
                      int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_gpu(const double* scale, double* v,
                      int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_gpu(const float* scale, float* v,
                      int rows, int numvec, int ld, cudaStream_t stream);

// Batched AXPY: y[:,i] += alpha[i] * x[:,i]
void batched_axpy_gpu(const cuDoubleComplex* alpha, const cuDoubleComplex* x,
                     cuDoubleComplex* y, int rows, int numvec, int ld,
                     cudaStream_t stream);
void batched_axpy_gpu(const cuComplex* alpha, const cuComplex* x,
                     cuComplex* y, int rows, int numvec, int ld,
                     cudaStream_t stream);
void batched_axpy_gpu(const double* alpha, const double* x,
                     double* y, int rows, int numvec, int ld,
                     cudaStream_t stream);
void batched_axpy_gpu(const float* alpha, const float* x,
                     float* y, int rows, int numvec, int ld,
                     cudaStream_t stream);

void batched_norm_squared_gpu(const cuDoubleComplex* v, double* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);
void batched_norm_squared_gpu(const cuComplex* v, float* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);
void batched_norm_squared_gpu(const double* v, double* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);
void batched_norm_squared_gpu(const float* v, float* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
