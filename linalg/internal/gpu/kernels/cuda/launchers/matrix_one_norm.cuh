// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cstddef>
#include <cuda_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
/**
 * @brief Computes the 1-norm (maximum absolute column sum) of a
 * column-major matrix on the GPU.
 *
 * For complex matrices the entry magnitude is `|Re| + |Im|` (BLAS `asum`
 * convention), which bounds `|z|` from above.
 *
 * The result is written asynchronously to device memory on @p stream_. When
 * the row range is split across thread blocks, a temporary buffer of
 * `splits * cols` reals is taken from the stream-ordered allocator.
 *
 * @param[in] d_matrix The input matrix of size `rows x cols` in device memory.
 * @param[in] rows The number of rows of the matrix.
 * @param[in] cols The number of columns of the matrix.
 * @param[in] ld The leading dimension of the matrix.
 * @param[out] d_norm Device pointer receiving the 1-norm.
 * @param[in] stream_ The CUDA stream to execute the kernels.
 */
void matrixOneNorm_gpu(const float* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       cudaStream_t stream_);
/** @copydoc matrixOneNorm_gpu(const float*, std::size_t, std::size_t, std::size_t, float*, cudaStream_t) */
void matrixOneNorm_gpu(const double* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       cudaStream_t stream_);
/** @copydoc matrixOneNorm_gpu(const float*, std::size_t, std::size_t, std::size_t, float*, cudaStream_t) */
void matrixOneNorm_gpu(const std::complex<float>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       cudaStream_t stream_);
/** @copydoc matrixOneNorm_gpu(const float*, std::size_t, std::size_t, std::size_t, float*, cudaStream_t) */
void matrixOneNorm_gpu(const std::complex<double>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       cudaStream_t stream_);
} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
