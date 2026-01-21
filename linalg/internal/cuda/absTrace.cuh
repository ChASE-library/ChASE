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
/**
 * @brief CUDA device function to reduce the sum of elements in a vector.
 *
 * This device function is used to perform a sum reduction on the elements of
 * the input vector `x`. It is typically called within a CUDA kernel to reduce
 * the elements of the vector `x` across multiple threads in a warp.
 *
 * @tparam n The number of elements to reduce.
 * @tparam T The type of the elements in the vector (e.g., float, double).
 * @param[in] i The index of the current thread.
 * @param[in,out] x The input vector whose elements will be summed. The sum
 * result will be stored in `x`.
 */
template <std::size_t n, typename T>
__device__ void cuda_sum_reduce(std::size_t i, T* x);
/**
 * @brief CUDA kernel to compute the absolute trace of a single-precision
 * matrix.
 *
 * This kernel computes the absolute trace of a single-precision matrix by
 * summing the absolute values of the diagonal elements. The result is stored in
 * the variable `d_trace`.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 */
__global__ void s_absTraceKernel(float* d_matrix, float* d_trace, std::size_t n,
                                 std::size_t ld);
/**
 * @brief CUDA kernel to compute the absolute trace of a double-precision
 * matrix.
 *
 * This kernel computes the absolute trace of a double-precision matrix by
 * summing the absolute values of the diagonal elements. The result is stored in
 * the variable `d_trace`.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 */
__global__ void d_absTraceKernel(double* d_matrix, double* d_trace,
                                 std::size_t n, std::size_t ld);
/**
 * @brief CUDA kernel to compute the absolute trace of a single-precision
 * complex matrix.
 *
 * This kernel computes the absolute trace of a single-precision complex matrix
 * by summing the absolute values of the diagonal elements. The result is stored
 * in the variable `d_trace`.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 */
__global__ void c_absTraceKernel(cuComplex* d_matrix, float* d_trace,
                                 std::size_t n, std::size_t ld);
/**
 * @brief CUDA kernel to compute the absolute trace of a double-precision
 * complex matrix.
 *
 * This kernel computes the absolute trace of a double-precision complex matrix
 * by summing the absolute values of the diagonal elements. The result is stored
 * in the variable `d_trace`.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 */
__global__ void z_absTraceKernel(cuDoubleComplex* d_matrix, double* d_trace,
                                 std::size_t n, std::size_t ld);

/**
 * @brief Computes the absolute trace of a single-precision matrix on the GPU.
 *
 * This function launches a CUDA kernel to compute the absolute trace of a
 * single-precision matrix. The absolute trace is computed by summing the
 * absolute values of the diagonal elements.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 * @param[in] stream_ The CUDA stream to execute the kernel.
 */
void absTrace_gpu(float* d_matrix, float* d_trace, std::size_t n,
                  std::size_t ld, cudaStream_t stream_);
/**
 * @brief Computes the absolute trace of a double-precision matrix on the GPU.
 *
 * This function launches a CUDA kernel to compute the absolute trace of a
 * double-precision matrix. The absolute trace is computed by summing the
 * absolute values of the diagonal elements.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 * @param[in] stream_ The CUDA stream to execute the kernel.
 */
void absTrace_gpu(double* d_matrix, double* d_trace, std::size_t n,
                  std::size_t ld, cudaStream_t stream_);
/**
 * @brief Computes the absolute trace of a single-precision complex matrix on
 * the GPU.
 *
 * This function launches a CUDA kernel to compute the absolute trace of a
 * single-precision complex matrix. The absolute trace is computed by summing
 * the absolute values of the diagonal elements.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 * @param[in] stream_ The CUDA stream to execute the kernel.
 */
void absTrace_gpu(std::complex<float>* d_matrix, float* d_trace, std::size_t n,
                  std::size_t ld, cudaStream_t stream_);
/**
 * @brief Computes the absolute trace of a double-precision complex matrix on
 * the GPU.
 *
 * This function launches a CUDA kernel to compute the absolute trace of a
 * double-precision complex matrix. The absolute trace is computed by summing
 * the absolute values of the diagonal elements.
 *
 * @param[in] d_matrix The input matrix of size `n x ld` in device memory.
 * @param[out] d_trace The resulting absolute trace of the matrix.
 * @param[in] n The number of rows in the matrix.
 * @param[in] ld The leading dimension of the matrix.
 * @param[in] stream_ The CUDA stream to execute the kernel.
 */
void absTrace_gpu(std::complex<double>* d_matrix, double* d_trace,
                  std::size_t n, std::size_t ld, cudaStream_t stream_);
} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase