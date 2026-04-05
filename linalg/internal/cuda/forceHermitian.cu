// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "forceHermitian.cuh"

#define FORCE_HERM_BLOCK 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
namespace
{
__device__ void decode_upper_tri_index(std::size_t k, std::size_t n,
                                       std::size_t& i, std::size_t& j)
{
    std::size_t rem = k;
    for (i = 0; i < n; ++i)
    {
        std::size_t row_len = n - i;
        if (rem < row_len)
        {
            j = i + rem;
            return;
        }
        rem -= row_len;
    }
    i = n;
    j = n;
}

__global__ void s_force_hermitian_kernel(float* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    float val = 0.5f * (A[i + j * lda] + A[j + i * lda]);
    A[i + j * lda] = val;
    A[j + i * lda] = val;
}

__global__ void d_force_hermitian_kernel(double* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    double val = 0.5 * (A[i + j * lda] + A[j + i * lda]);
    A[i + j * lda] = val;
    A[j + i * lda] = val;
}

__global__ void c_force_hermitian_kernel(cuComplex* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    cuComplex aij = A[i + j * lda];
    cuComplex aji = A[j + i * lda];
    cuComplex sum = cuCaddf(aij, cuConjf(aji));
    cuComplex val = make_cuComplex(0.5f * sum.x, 0.5f * sum.y);
    A[i + j * lda] = val;
    A[j + i * lda] = cuConjf(val);
}

__global__ void z_force_hermitian_kernel(cuDoubleComplex* A, std::size_t n,
                                         std::size_t lda)
{
    std::size_t k = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t total = n * (n + 1) / 2;
    if (k >= total)
        return;
    std::size_t i, j;
    decode_upper_tri_index(k, n, i, j);
    cuDoubleComplex aij = A[i + j * lda];
    cuDoubleComplex aji = A[j + i * lda];
    cuDoubleComplex sum = cuCadd(aij, cuConj(aji));
    cuDoubleComplex val =
        make_cuDoubleComplex(0.5 * sum.x, 0.5 * sum.y);
    A[i + j * lda] = val;
    A[j + i * lda] = cuConj(val);
}
} // namespace

void force_hermitian_gpu(float* A, std::size_t n, std::size_t lda,
                         cudaStream_t stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    s_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(A, n, lda);
}

void force_hermitian_gpu(double* A, std::size_t n, std::size_t lda,
                         cudaStream_t stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    d_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(A, n, lda);
}

void force_hermitian_gpu(std::complex<float>* A, std::size_t n, std::size_t lda,
                         cudaStream_t stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    c_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(
        reinterpret_cast<cuComplex*>(A), n, lda);
}

void force_hermitian_gpu(std::complex<double>* A, std::size_t n,
                         std::size_t lda, cudaStream_t stream)
{
    std::size_t total = n * (n + 1) / 2;
    std::size_t grid = (total + FORCE_HERM_BLOCK - 1) / FORCE_HERM_BLOCK;
    z_force_hermitian_kernel<<<grid, FORCE_HERM_BLOCK, 0, stream>>>(
        reinterpret_cast<cuDoubleComplex*>(A), n, lda);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
