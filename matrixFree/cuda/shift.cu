/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCKDIM 256

__global__ void shiftMatrixGpu(cuComplex* A, int lda, int n, cuComplex shift, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offset && idx < n) //TODO verify correctness in multi-gpu case
        A[(idx - offset) * lda + idx].x += shift.x;
}

__global__ void shiftMatrixGpu(cuDoubleComplex* A, int lda, int n, cuDoubleComplex shift, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offset && idx < n) //TODO verify correctness in multi-gpu case
        A[(idx - offset) * lda + idx].x += shift.x;
}

__global__ void shiftMatrixGpu(float* A, int lda, int n, float shift, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offset && idx < n) //TODO verify correctness in multi-gpu case
        A[(idx - offset) * lda + idx] += shift;
}

__global__ void shiftMatrixGpu(double* A, int lda, int n, double shift, int offset)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offset && idx < n) //TODO verify correctness in multi-gpu case
        A[(idx - offset) * lda + idx] += shift;
}

void shiftMatrixGPU(float* A, int lda, int n, float shift, int offset, cudaStream_t stream)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    shiftMatrixGpu<<<num_blocks, BLOCKDIM, 0, stream>>>(A, lda, n, shift, offset);
}

void shiftMatrixGPU(double* A, int lda, int n, double shift, int offset, cudaStream_t stream)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    shiftMatrixGpu<<<num_blocks, BLOCKDIM, 0, stream>>>(A, lda, n, shift, offset);
}

void shiftMatrixGPU(std::complex<float>* A, int lda, int n, std::complex<float> shift, int offset, cudaStream_t stream)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    cuComplex cuShift = make_cuComplex(shift.real(), shift.imag());
    shiftMatrixGpu<<<num_blocks, BLOCKDIM, 0, stream>>>(reinterpret_cast<cuComplex*>(A),
                                                        lda, n, cuShift, offset);
}

void shiftMatrixGPU(std::complex<double>* A, int lda, int n, std::complex<double> shift, int offset, cudaStream_t stream)
{
    int num_blocks = (n + (BLOCKDIM - 1)) / BLOCKDIM;
    cuDoubleComplex cuShift = make_cuDoubleComplex(shift.real(), shift.imag());
    shiftMatrixGpu<<<num_blocks, BLOCKDIM, 0, stream>>>(reinterpret_cast<cuDoubleComplex*>(A),
                                                        lda, n, cuShift, offset);
}
