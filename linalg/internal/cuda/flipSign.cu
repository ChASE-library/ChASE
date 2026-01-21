// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "flipSign.cuh"

const int blockSize = 256;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
__global__ void sflipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n,
                                         std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void dflipLowerHalfMatrixSign(double* A, std::size_t m,
                                         std::size_t n, std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void cflipLowerHalfMatrixSign(cuComplex* A, std::size_t m,
                                         std::size_t n, std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        cuComplex& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}
__global__ void zflipLowerHalfMatrixSign(cuDoubleComplex* A, std::size_t m,
                                         std::size_t n, std::size_t lda)
{
    const std::size_t half_m = m / 2;
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / half_m;
    const std::size_t row_offset = idx % half_m;

    if (col < n)
    {
        const std::size_t row = half_m + row_offset;
        cuDoubleComplex& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}

__global__ void sflipMatrixSign(float* A, std::size_t m, std::size_t n,
                                std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void dflipMatrixSign(double* A, std::size_t m, std::size_t n,
                                std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        A[row + lda * col] = -A[row + lda * col];
    }
}
__global__ void cflipMatrixSign(cuComplex* A, std::size_t m, std::size_t n,
                                std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        cuComplex& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}
__global__ void zflipMatrixSign(cuDoubleComplex* A, std::size_t m,
                                std::size_t n, std::size_t lda)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t col = idx / m;
    const std::size_t row = idx % m;

    if (col < n)
    {
        cuDoubleComplex& element = A[row + lda * col];
        element.x = -element.x;
        element.y = -element.y;
    }
}

void chase_flipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n,
                                   std::size_t lda, cudaStream_t stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    sflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n,
                                                                    lda);
}
void chase_flipLowerHalfMatrixSign(double* A, std::size_t m, std::size_t n,
                                   std::size_t lda, cudaStream_t stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    dflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n,
                                                                    lda);
}
void chase_flipLowerHalfMatrixSign(std::complex<float>* A, std::size_t m,
                                   std::size_t n, std::size_t lda,
                                   cudaStream_t stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    cflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), m, n, lda);
}
void chase_flipLowerHalfMatrixSign(std::complex<double>* A, std::size_t m,
                                   std::size_t n, std::size_t lda,
                                   cudaStream_t stream_)
{
    std::size_t num_blocks = (n * (m / 2) + (blockSize - 1)) / blockSize;
    zflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), m, n, lda);
}

void chase_flipMatrixSign(float* A, std::size_t m, std::size_t n,
                          std::size_t lda, cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    sflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda);
}
void chase_flipMatrixSign(double* A, std::size_t m, std::size_t n,
                          std::size_t lda, cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    dflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda);
}
void chase_flipMatrixSign(std::complex<float>* A, std::size_t m, std::size_t n,
                          std::size_t lda, cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    cflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuComplex*>(A), m, n, lda);
}
void chase_flipMatrixSign(std::complex<double>* A, std::size_t m, std::size_t n,
                          std::size_t lda, cudaStream_t stream_)
{
    std::size_t num_blocks = (n * m + (blockSize - 1)) / blockSize;
    zflipMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
        reinterpret_cast<cuDoubleComplex*>(A), m, n, lda);
}
} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
