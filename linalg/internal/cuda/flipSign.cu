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
    __global__ void sflipLowerHalfMatrixSign(float* A, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ((idx % lda) >= (lda / 2) && idx < lda*n)
            A[idx] = -1.0 * A[idx];
    }
    __global__ void dflipLowerHalfMatrixSign(double* A, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ((idx % lda) >= (lda / 2) && idx < lda*n)
            A[idx] = -1.0 * A[idx];
    }
    __global__ void cflipLowerHalfMatrixSign(cuComplex* A, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ((idx % lda) >= (lda / 2) && idx < lda*n)
            A[idx].x = -1.0 * A[idx].x;
    }
    __global__ void zflipLowerHalfMatrixSign(cuDoubleComplex* A, std::size_t n, std::size_t lda)
    {
        std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if ((idx % lda) >= (lda / 2) && idx < lda*n)
            A[idx].x = -1.0 * A[idx].x;
    }

/*
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
*/
    void chase_flipLowerHalfMatrixSign(float* A, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        sflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda);
    }
    void chase_flipLowerHalfMatrixSign(double* A, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        dflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(A, n, lda);
    }
    void chase_flipLowerHalfMatrixSign(std::complex<float>* A, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        cflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
            reinterpret_cast<cuComplex*>(A), n, lda);
    }
    void chase_flipLowerHalfMatrixSign(std::complex<double>* A, std::size_t n, std::size_t lda, cudaStream_t stream_)
    {
        std::size_t num_blocks = (n + (blockSize - 1)) / blockSize;
        zflipLowerHalfMatrixSign<<<num_blocks, blockSize, 0, stream_>>>(
            reinterpret_cast<cuDoubleComplex*>(A), n, lda);
    }
/*
    void chase_shift_mgpu_matrix(float* A, std::size_t* off_m, std::size_t* off_n,
                                std::size_t offsize, std::size_t ldH, float shift,
                                cudaStream_t stream_)
    {
        unsigned int grid = (offsize + blockSize - 1) / blockSize;
        if(grid == 0)
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
        if(grid == 0)
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
        if(grid == 0)
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
        if(grid == 0)
        {
            grid = 1;
        }
        dim3 threadsPerBlock(blockSize, 1);
        dim3 numBlocks(grid, 1);
        zshift_mgpu_matrix<<<numBlocks, threadsPerBlock, 0, stream_>>>( //
            reinterpret_cast<cuDoubleComplex*>(A), off_m, off_n,        //
            offsize, ldH, shift);
    }
*/
}
}
}
}
