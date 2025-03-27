// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "flipSign.cuh"

#define blockSize 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    __global__ void sflipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	std::size_t row = idx % (m / 2);
	std::size_t col = idx / (m / 2);
	if(idx < (m/2) * n)
            A[m / 2 + row + lda * col] = -1.0 * A[m / 2 + row + lda * col];
    }
    __global__ void dflipLowerHalfMatrixSign(double* A, std::size_t m, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	std::size_t row = idx % (m / 2);
	std::size_t col = idx / (m / 2);
	if(idx < (m/2) * n)
            A[m / 2 + row + lda * col] = -1.0 * A[m / 2 + row + lda * col];
	
    }
    __global__ void cflipLowerHalfMatrixSign(cuComplex* A, std::size_t m, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	std::size_t row = idx % (m / 2);
	std::size_t col = idx / (m / 2);
	if(idx < (m/2) * n){
            A[m / 2 + row + lda * col].x = -1.0 * A[m / 2 + row + lda * col].x;
            A[m / 2 + row + lda * col].y = -1.0 * A[m / 2 + row + lda * col].y;
	}
    }
    __global__ void zflipLowerHalfMatrixSign(cuDoubleComplex* A, std::size_t m, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	std::size_t row = idx % (m / 2);
	std::size_t col = idx / (m / 2);
	if(idx < (m/2) * n){
            A[m / 2 + row + lda * col].x = -1.0 * A[m / 2 + row + lda * col].x;
            A[m / 2 + row + lda * col].y = -1.0 * A[m / 2 + row + lda * col].y;
	}
    }

    void chase_flipLowerHalfMatrixSign(float* A, std::size_t m, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n * (m/2) + (blockSize - 1)) / blockSize;
        sflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda);
    }
    void chase_flipLowerHalfMatrixSign(double* A, std::size_t m, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n * (m/2) + (blockSize - 1)) / blockSize;
        dflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, m, n, lda);
    }
    void chase_flipLowerHalfMatrixSign(std::complex<float>* A, std::size_t m, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n * (m/2) + (blockSize - 1)) / blockSize;
        cflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
            reinterpret_cast<cuComplex*>(A), m, n, lda);
    }
    void chase_flipLowerHalfMatrixSign(std::complex<double>* A, std::size_t m, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n * (m/2) + (blockSize - 1)) / blockSize;
        zflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
            reinterpret_cast<cuDoubleComplex*>(A), m, n, lda);
    }
}
}
}
}
