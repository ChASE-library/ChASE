// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "shiftDiagonal.cuh"

#define blockSize 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
// ---------------------------------------- CUDA kernels
// ---------------------------------------------- //

__global__ void sshift_matrix(float* A, std::size_t n, std::size_t lda,
                              float shift)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*lda + idx] += shift;
}
__global__ void dshift_matrix(double* A, std::size_t n, std::size_t lda,
                              double shift)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*lda + idx] += shift;
}
__global__ void cshift_matrix(cuComplex* A, std::size_t n, std::size_t lda,
                              float shift)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*lda + idx].x += shift;
}
__global__ void zshift_matrix(cuDoubleComplex* A, std::size_t n,
                              std::size_t lda, double shift)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*lda + idx].x += shift;
}

__global__ void sshift_mgpu_matrix(float* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, float shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind] += shift;
    }
}

__global__ void dshift_mgpu_matrix(double* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, double shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind] += shift;
    }
}

__global__ void cshift_mgpu_matrix(cuComplex* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, float shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind].x += shift;
    }
}

__global__ void zshift_mgpu_matrix(cuDoubleComplex* A, std::size_t* off_m,
                                   std::size_t* off_n, std::size_t offsize,
                                   std::size_t ldH, double shift)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t ind;
    if (i < offsize)
    {
        ind = off_n[i] * ldH + off_m[i];
        A[ind].x += shift;
    }
}
__global__ void sinverse_entries(float* vector, std::size_t n)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        vector[idx] = 1.0 / vector[idx];
}
__global__ void dinverse_entries(double* vector, std::size_t n)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        vector[idx] = 1.0 / vector[idx];
}

__global__ void ssubtract_inverse_diagonal(float* A, std::size_t n,
                                           std::size_t lda, float coef,
                                           float* new_diag)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef - A[(idx)*lda + idx]);
}
__global__ void dsubtract_inverse_diagonal(double* A, std::size_t n,
                                           std::size_t lda, double coef,
                                           double* new_diag)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef - A[(idx)*lda + idx]);
}
__global__ void csubtract_inverse_diagonal(cuComplex* A, std::size_t n,
                                           std::size_t lda, float coef,
                                           float* new_diag)
{
    // We assume the diagonal of A is real, and coef real. We quite new_diag
    // complex for later operations
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef - A[(idx)*lda + idx].x);
}
__global__ void zsubtract_inverse_diagonal(cuDoubleComplex* A, std::size_t n,
                                           std::size_t lda, double coef,
                                           double* new_diag)
{
    // We assume the diagonal of A is real, and coef real. We quite new_diag
    // complex for later operations
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef - A[(idx)*lda + idx].x);
}

__global__ void splus_inverse_diagonal(float* A, std::size_t n, std::size_t lda,
                                       float coef, float* new_diag)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef + A[(idx)*lda + idx]);
}
__global__ void dplus_inverse_diagonal(double* A, std::size_t n,
                                       std::size_t lda, double coef,
                                       double* new_diag)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef + A[(idx)*lda + idx]);
}
__global__ void cplus_inverse_diagonal(cuComplex* A, std::size_t n,
                                       std::size_t lda, float coef,
                                       float* new_diag)
{
    // We assume the diagonal of A is real, and coef real. We quite new_diag
    // complex for later operations
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef + A[(idx)*lda + idx].x);
}
__global__ void zplus_inverse_diagonal(cuDoubleComplex* A, std::size_t n,
                                       std::size_t lda, double coef,
                                       double* new_diag)
{
    // We assume the diagonal of A is real, and coef real. We quite new_diag
    // complex for later operations
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        new_diag[idx] = 1.0 / (coef + A[(idx)*lda + idx].x);
}

__global__ void sset_diagonal(float* A, std::size_t n, std::size_t lda,
                              float coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*lda + idx] = coef;
}
__global__ void dset_diagonal(double* A, std::size_t n, std::size_t lda,
                              double coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        A[(idx)*lda + idx] = coef;
}
__global__ void cset_diagonal(cuComplex* A, std::size_t n, std::size_t lda,
                              cuComplex coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        A[(idx)*lda + idx].x = coef.x;
        A[(idx)*lda + idx].y = coef.y;
    }
}
__global__ void zset_diagonal(cuDoubleComplex* A, std::size_t n,
                              std::size_t lda, cuDoubleComplex coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        A[(idx)*lda + idx].x = coef.x;
        A[(idx)*lda + idx].y = coef.y;
    }
}
__global__ void sscale_rows_matrix(float* A, std::size_t m, std::size_t n,
                                   std::size_t lda, float* coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t row = idx % m;
    std::size_t col = idx / m;
    if (idx < m * n)
        A[row + lda * col] *= coef[row];
}
__global__ void dscale_rows_matrix(double* A, std::size_t m, std::size_t n,
                                   std::size_t lda, double* coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t row = idx % m;
    std::size_t col = idx / m;
    if (idx < m * n)
        A[row + lda * col] *= coef[row];
}
__global__ void cscale_rows_matrix(cuComplex* A, std::size_t m, std::size_t n,
                                   std::size_t lda, float* coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t row = idx % m;
    std::size_t col = idx / m;
    if (idx < m * n)
    {
        A[row + lda * col].x *= coef[row];
        A[row + lda * col].y *= coef[row];
    }
}
__global__ void zscale_rows_matrix(cuDoubleComplex* A, std::size_t m,
                                   std::size_t n, std::size_t lda, double* coef)
{
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t row = idx % m;
    std::size_t col = idx / m;
    if (idx < m * n)
    {
        A[row + lda * col].x *= coef[row];
        A[row + lda * col].y *= coef[row];
    }
}

// ------------------------------ ChASE templated calls to kernels
// ----------------------------------- //

void chase_shift_matrix(float* A, std::size_t n, std::size_t lda, float shift,
                        cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    sshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda, shift);
}
void chase_shift_matrix(double* A, std::size_t n, std::size_t lda, double shift,
                        cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    dshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda, shift);
}
void chase_shift_matrix(std::complex<float>* A, std::size_t n, std::size_t lda,
                        float shift, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    cshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), n, lda, shift);
}
void chase_shift_matrix(std::complex<double>* A, std::size_t n, std::size_t lda,
                        double shift, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    zshift_matrix<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), n, lda, shift);
}

void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, float shift,
                             cudaStream_t stream_)
{
    unsigned int grid = (offsize + blockSize - 1) / blockSize;
    if (grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(blockSize, 1);
    dim3 numBlocks(grid, 1);
    sshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        A, off_m, off_n, offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(double* A, std::size_t* off_m, std::size_t* off_n,
                             std::size_t offsize, std::size_t ldH, double shift,
                             cudaStream_t stream_)
{
    unsigned int grid = (offsize + blockSize - 1) / blockSize;
    if (grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(blockSize, 1);
    dim3 numBlocks(grid, 1);
    dshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        A, off_m, off_n, offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(std::complex<float>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, float shift, cudaStream_t stream_)
{
    unsigned int grid = (offsize + blockSize - 1) / blockSize;
    if (grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(blockSize, 1);
    dim3 numBlocks(grid, 1);
    cshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        reinterpret_cast<cuComplex*>(A), off_m, off_n,              //
        offsize, ldH, shift);
}

void chase_shift_mgpu_matrix(std::complex<double>* A, std::size_t* off_m,
                             std::size_t* off_n, std::size_t offsize,
                             std::size_t ldH, double shift,
                             cudaStream_t stream_)
{
    unsigned int grid = (offsize + blockSize - 1) / blockSize;
    if (grid == 0)
    {
        grid = 1;
    }
    dim3 threadsPerBlock(blockSize, 1);
    dim3 numBlocks(grid, 1);
    zshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
        reinterpret_cast<cuDoubleComplex*>(A), off_m, off_n,        //
        offsize, ldH, shift);
}
void chase_inverse_entries(float* vector, std::size_t n, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    sinverse_entries<<<num_blocks, blockSize, 0, stream_>>>(vector, n);
}
void chase_inverse_entries(double* vector, std::size_t n, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    dinverse_entries<<<num_blocks, blockSize, 0, stream_>>>(vector, n);
}
void chase_subtract_inverse_diagonal(float* A, std::size_t n, std::size_t lda,
                                     float coef, float* new_diag,
                                     cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    ssubtract_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        A, n, lda, coef, new_diag);
}
void chase_subtract_inverse_diagonal(double* A, std::size_t n, std::size_t lda,
                                     double coef, double* new_diag,
                                     cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    dsubtract_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        A, n, lda, coef, new_diag);
}
void chase_subtract_inverse_diagonal(std::complex<float>* A, std::size_t n,
                                     std::size_t lda, float coef,
                                     float* new_diag, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    csubtract_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), n, lda, coef, new_diag);
}
void chase_subtract_inverse_diagonal(std::complex<double>* A, std::size_t n,
                                     std::size_t lda, double coef,
                                     double* new_diag, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    zsubtract_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), n, lda, coef, new_diag);
}

void chase_plus_inverse_diagonal(float* A, std::size_t n, std::size_t lda,
                                 float coef, float* new_diag,
                                 cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    splus_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        A, n, lda, coef, new_diag);
}
void chase_plus_inverse_diagonal(double* A, std::size_t n, std::size_t lda,
                                 double coef, double* new_diag,
                                 cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    dplus_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        A, n, lda, coef, new_diag);
}
void chase_plus_inverse_diagonal(std::complex<float>* A, std::size_t n,
                                 std::size_t lda, float coef, float* new_diag,
                                 cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    cplus_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), n, lda, coef, new_diag);
}
void chase_plus_inverse_diagonal(std::complex<double>* A, std::size_t n,
                                 std::size_t lda, double coef, double* new_diag,
                                 cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    zplus_inverse_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), n, lda, coef, new_diag);
}

void chase_set_diagonal(float* A, std::size_t n, std::size_t lda, float coef,
                        cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    sset_diagonal<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda, coef);
}
void chase_set_diagonal(double* A, std::size_t n, std::size_t lda, double coef,
                        cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    dset_diagonal<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda, coef);
}
void chase_set_diagonal(std::complex<float>* A, std::size_t n, std::size_t lda,
                        std::complex<float> coef, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    cset_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), n, lda,
        make_cuComplex(std::real(coef), std::imag(coef)));
}
void chase_set_diagonal(std::complex<double>* A, std::size_t n, std::size_t lda,
                        std::complex<double> coef, cudaStream_t stream_)
{
    std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
    zset_diagonal<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), n, lda,
        make_cuDoubleComplex(std::real(coef), std::imag(coef)));
}

void chase_scale_rows_matrix(float* A, std::size_t m, std::size_t n,
                             std::size_t lda, float* coef, cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    sscale_rows_matrix<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda,
                                                              coef);
}
void chase_scale_rows_matrix(double* A, std::size_t m, std::size_t n,
                             std::size_t lda, double* coef,
                             cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    dscale_rows_matrix<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda,
                                                              coef);
}
void chase_scale_rows_matrix(std::complex<float>* A, std::size_t m,
                             std::size_t n, std::size_t lda, float* coef,
                             cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    cscale_rows_matrix<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), m, n, lda, coef);
}
void chase_scale_rows_matrix(std::complex<double>* A, std::size_t m,
                             std::size_t n, std::size_t lda, double* coef,
                             cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    zscale_rows_matrix<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), m, n, lda, coef);
}
} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
