// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#if defined(CHASE_GPU_DEVICE_BACKEND_HIP)

#include "linalg/internal/gpu/kernels/hip/launchers/matrix_diag.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{
namespace
{

constexpr int block_size = 256;

__global__ void set_diagonal_float_kernel(float* A, std::size_t n,
                                          std::size_t lda, float coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[idx * lda + idx] = coef;
}

__global__ void set_diagonal_double_kernel(double* A, std::size_t n,
                                           std::size_t lda, double coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[idx * lda + idx] = coef;
}

__global__ void set_diagonal_complex_float_kernel(CHASE_GPU_FLOAT_COMPLEX* A,
                                                  std::size_t n, std::size_t lda,
                                                  CHASE_GPU_FLOAT_COMPLEX coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[idx * lda + idx] = coef;
}

__global__ void set_diagonal_complex_double_kernel(CHASE_GPU_DOUBLE_COMPLEX* A,
                                                   std::size_t n, std::size_t lda,
                                                   CHASE_GPU_DOUBLE_COMPLEX coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[idx * lda + idx] = coef;
}

__global__ void inverse_entries_float_kernel(float* vector, std::size_t n)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        vector[idx] = 1.0f / vector[idx];
}

__global__ void inverse_entries_double_kernel(double* vector, std::size_t n)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        vector[idx] = 1.0 / vector[idx];
}

__global__ void plus_inverse_diagonal_float_kernel(float* A, std::size_t n,
                                                   std::size_t lda, float coef,
                                                   float* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0f / (coef + A[idx * lda + idx]);
}

__global__ void plus_inverse_diagonal_double_kernel(double* A, std::size_t n,
                                                    std::size_t lda,
                                                    double coef,
                                                    double* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef + A[idx * lda + idx]);
}

__global__ void plus_inverse_diagonal_complex_float_kernel(
    CHASE_GPU_FLOAT_COMPLEX* A, std::size_t n, std::size_t lda, float coef,
    float* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0f / (coef + A[idx * lda + idx].x);
}

__global__ void plus_inverse_diagonal_complex_double_kernel(
    CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t n, std::size_t lda, double coef,
    double* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef + A[idx * lda + idx].x);
}

__global__ void subtract_inverse_diagonal_float_kernel(
    float* A, std::size_t n, std::size_t lda, float coef, float* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0f / (coef - A[idx * lda + idx]);
}

__global__ void subtract_inverse_diagonal_double_kernel(
    double* A, std::size_t n, std::size_t lda, double coef, double* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef - A[idx * lda + idx]);
}

__global__ void subtract_inverse_diagonal_complex_float_kernel(
    CHASE_GPU_FLOAT_COMPLEX* A, std::size_t n, std::size_t lda, float coef,
    float* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0f / (coef - A[idx * lda + idx].x);
}

__global__ void subtract_inverse_diagonal_complex_double_kernel(
    CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t n, std::size_t lda, double coef,
    double* new_diag)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef - A[idx * lda + idx].x);
}

__global__ void scale_rows_matrix_float_kernel(float* A, std::size_t m,
                                               std::size_t n, std::size_t lda,
                                               float* coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = idx % m;
    const std::size_t col = idx / m;
    if (idx < m * n)
        A[row + lda * col] *= coef[row];
}

__global__ void scale_rows_matrix_double_kernel(double* A, std::size_t m,
                                                std::size_t n, std::size_t lda,
                                                double* coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = idx % m;
    const std::size_t col = idx / m;
    if (idx < m * n)
        A[row + lda * col] *= coef[row];
}

__global__ void scale_rows_matrix_complex_float_kernel(
    CHASE_GPU_FLOAT_COMPLEX* A, std::size_t m, std::size_t n, std::size_t lda,
    float* coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = idx % m;
    const std::size_t col = idx / m;
    if (idx < m * n)
    {
        A[row + lda * col].x *= coef[row];
        A[row + lda * col].y *= coef[row];
    }
}

__global__ void scale_rows_matrix_complex_double_kernel(
    CHASE_GPU_DOUBLE_COMPLEX* A, std::size_t m, std::size_t n, std::size_t lda,
    double* coef)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t row = idx % m;
    const std::size_t col = idx / m;
    if (idx < m * n)
    {
        A[row + lda * col].x *= coef[row];
        A[row + lda * col].y *= coef[row];
    }
}

} // namespace

void chase_set_diagonal(float* A, std::size_t n, std::size_t lda, float coef,
                        CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(set_diagonal_float_kernel, num_blocks, block_size, 0,
                       stream, A, n, lda, coef);
}

void chase_set_diagonal(double* A, std::size_t n, std::size_t lda, double coef,
                        CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(set_diagonal_double_kernel, num_blocks, block_size, 0,
                       stream, A, n, lda, coef);
}

void chase_set_diagonal(std::complex<float>* A, std::size_t n, std::size_t lda,
                        std::complex<float> coef, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    const CHASE_GPU_FLOAT_COMPLEX hcoef =
        CHASE_GPU_MAKE_FLOAT_COMPLEX(coef.real(), coef.imag());
    hipLaunchKernelGGL(set_diagonal_complex_float_kernel, num_blocks, block_size,
                       0, stream,
                       reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), n, lda,
                       hcoef);
}

void chase_set_diagonal(std::complex<double>* A, std::size_t n, std::size_t lda,
                        std::complex<double> coef, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    const CHASE_GPU_DOUBLE_COMPLEX hcoef =
        CHASE_GPU_MAKE_DOUBLE_COMPLEX(coef.real(), coef.imag());
    hipLaunchKernelGGL(set_diagonal_complex_double_kernel, num_blocks,
                       block_size, 0, stream,
                       reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), n, lda,
                       hcoef);
}

void chase_inverse_entries(float* vector, std::size_t n, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(inverse_entries_float_kernel, num_blocks, block_size, 0,
                       stream, vector, n);
}

void chase_inverse_entries(double* vector, std::size_t n, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(inverse_entries_double_kernel, num_blocks, block_size, 0,
                       stream, vector, n);
}

void chase_plus_inverse_diagonal(float* A, std::size_t n, std::size_t lda,
                                 float coef, float* new_diag,
                                 CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(plus_inverse_diagonal_float_kernel, num_blocks,
                       block_size, 0, stream, A, n, lda, coef, new_diag);
}

void chase_plus_inverse_diagonal(double* A, std::size_t n, std::size_t lda,
                                 double coef, double* new_diag,
                                 CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(plus_inverse_diagonal_double_kernel, num_blocks,
                       block_size, 0, stream, A, n, lda, coef, new_diag);
}

void chase_plus_inverse_diagonal(std::complex<float>* A, std::size_t n,
                                 std::size_t lda, float coef,
                                 float* new_diag, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(plus_inverse_diagonal_complex_float_kernel, num_blocks,
                       block_size, 0, stream,
                       reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), n, lda,
                       coef, new_diag);
}

void chase_plus_inverse_diagonal(std::complex<double>* A, std::size_t n,
                                 std::size_t lda, double coef,
                                 double* new_diag, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(plus_inverse_diagonal_complex_double_kernel, num_blocks,
                       block_size, 0, stream,
                       reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), n, lda,
                       coef, new_diag);
}

void subtract_inverse_diagonal(float* A, std::size_t n, std::size_t lda,
                               float coef, float* new_diag,
                               CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(subtract_inverse_diagonal_float_kernel, num_blocks,
                       block_size, 0, stream, A, n, lda, coef, new_diag);
}

void subtract_inverse_diagonal(double* A, std::size_t n, std::size_t lda,
                               double coef, double* new_diag,
                               CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(subtract_inverse_diagonal_double_kernel, num_blocks,
                       block_size, 0, stream, A, n, lda, coef, new_diag);
}

void subtract_inverse_diagonal(std::complex<float>* A, std::size_t n,
                               std::size_t lda, float coef, float* new_diag,
                               CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(subtract_inverse_diagonal_complex_float_kernel,
                       num_blocks, block_size, 0, stream,
                       reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), n, lda,
                       coef, new_diag);
}

void subtract_inverse_diagonal(std::complex<double>* A, std::size_t n,
                               std::size_t lda, double coef, double* new_diag,
                               CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (n + block_size - 1) / block_size;
    hipLaunchKernelGGL(subtract_inverse_diagonal_complex_double_kernel,
                       num_blocks, block_size, 0, stream,
                       reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), n, lda,
                       coef, new_diag);
}

void chase_scale_rows_matrix(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, float* coef, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (m * n + block_size - 1) / block_size;
    hipLaunchKernelGGL(scale_rows_matrix_float_kernel, num_blocks, block_size,
                       0, stream, A, m, n, lda, coef);
}

void chase_scale_rows_matrix(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, double* coef, CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (m * n + block_size - 1) / block_size;
    hipLaunchKernelGGL(scale_rows_matrix_double_kernel, num_blocks, block_size,
                       0, stream, A, m, n, lda, coef);
}

void chase_scale_rows_matrix(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda, float* coef,
                             CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (m * n + block_size - 1) / block_size;
    hipLaunchKernelGGL(scale_rows_matrix_complex_float_kernel, num_blocks,
                       block_size, 0, stream,
                       reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(A), m, n, lda,
                       coef);
}

void chase_scale_rows_matrix(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda, double* coef,
                             CHASE_GPU_STREAM_T stream)
{
    const std::size_t num_blocks = (m * n + block_size - 1) / block_size;
    hipLaunchKernelGGL(scale_rows_matrix_complex_double_kernel, num_blocks,
                       block_size, 0, stream,
                       reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(A), m, n, lda,
                       coef);
}

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase

#endif // defined(CHASE_GPU_DEVICE_BACKEND_HIP)
