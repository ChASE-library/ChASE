// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "residuals.cuh"

#define NB_X 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
template <int n, typename T>
__device__ void cuda_sum_reduce(int i, T* x)
{
    __syncthreads();
    if (n > 1024)
    {
        if (i < 1024 && i + 1024 < n)
        {
            x[i] += x[i + 1024];
        }
        __syncthreads();
    }
    if (n > 512)
    {
        if (i < 512 && i + 512 < n)
        {
            x[i] += x[i + 512];
        }
        __syncthreads();
    }
    if (n > 256)
    {
        if (i < 256 && i + 256 < n)
        {
            x[i] += x[i + 256];
        }
        __syncthreads();
    }
    if (n > 128)
    {
        if (i < 128 && i + 128 < n)
        {
            x[i] += x[i + 128];
        }
        __syncthreads();
    }
    if (n > 64)
    {
        if (i < 64 && i + 64 < n)
        {
            x[i] += x[i + 64];
        }
        __syncthreads();
    }
    if (n > 32)
    {
        if (i < 32 && i + 32 < n)
        {
            x[i] += x[i + 32];
        }
        __syncthreads();
    }
    if (n > 16)
    {
        if (i < 16 && i + 16 < n)
        {
            x[i] += x[i + 16];
        }
        __syncthreads();
    }
    if (n > 8)
    {
        if (i < 8 && i + 8 < n)
        {
            x[i] += x[i + 8];
        }
        __syncthreads();
    }
    if (n > 4)
    {
        if (i < 4 && i + 4 < n)
        {
            x[i] += x[i + 4];
        }
        __syncthreads();
    }
    if (n > 2)
    {
        if (i < 2 && i + 2 < n)
        {
            x[i] += x[i + 2];
        }
        __syncthreads();
    }
    if (n > 1)
    {
        if (i < 1 && i + 1 < n)
        {
            x[i] += x[i + 1];
        }
        __syncthreads();
    }
}

__global__ void c_resids_kernel(std::size_t m, std::size_t n,
                                const cuComplex* A, std::size_t lda,
                                const cuComplex* B, std::size_t ldb,
                                float* ritzv, float* resids, bool is_sqrt)
{
    __shared__ float ssum[NB_X];
    std::size_t tx = threadIdx.x;
    A += blockIdx.x * lda;
    B += blockIdx.x * lda;

    ssum[tx] = 0;
    for (std::size_t i = tx; i < m; i += NB_X)
    {
        cuComplex alpha;
        alpha.x = ritzv[blockIdx.x];
        alpha.y = 0.0;
        cuComplex a = cuCmulf(alpha, B[i]);
        cuComplex b = cuCsubf(A[i], a);
        float nrm = cuCabsf(b);
        ssum[tx] += nrm * nrm;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if (tx == 0)
    {
        if (is_sqrt)
        {
            resids[blockIdx.x] = sqrtf(ssum[0]);
        }
        else
        {
            resids[blockIdx.x] = ssum[0];
        }
    }
}

__global__ void z_resids_kernel(std::size_t m, std::size_t n,
                                const cuDoubleComplex* A, std::size_t lda,
                                const cuDoubleComplex* B, std::size_t ldb,
                                double* ritzv, double* resids, bool is_sqrt)
{
    __shared__ double ssum[NB_X];
    std::size_t tx = threadIdx.x;
    A += blockIdx.x * lda;
    B += blockIdx.x * lda;

    ssum[tx] = 0;
    for (std::size_t i = tx; i < m; i += NB_X)
    {
        cuDoubleComplex alpha;
        alpha.x = ritzv[blockIdx.x];
        alpha.y = 0.0;
        cuDoubleComplex a = cuCmul(alpha, B[i]);
        cuDoubleComplex b = cuCsub(A[i], a);
        double nrm = cuCabs(b);
        ssum[tx] += nrm * nrm;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if (tx == 0)
    {
        if (is_sqrt)
        {
            resids[blockIdx.x] = sqrt(ssum[0]);
        }
        else
        {
            resids[blockIdx.x] = ssum[0];
        }
    }
}

__global__ void d_resids_kernel(std::size_t m, std::size_t n, const double* A,
                                std::size_t lda, const double* B,
                                std::size_t ldb, double* ritzv, double* resids,
                                bool is_sqrt)
{
    __shared__ double ssum[NB_X];
    std::size_t tx = threadIdx.x;
    A += blockIdx.x * lda;
    B += blockIdx.x * lda;

    ssum[tx] = 0;
    for (std::size_t i = tx; i < m; i += NB_X)
    {
        double alpha;
        alpha = ritzv[blockIdx.x];
        double a = alpha * B[i];
        double b = A[i] - a;
        ssum[tx] += b * b;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if (tx == 0)
    {
        if (is_sqrt)
        {
            resids[blockIdx.x] = sqrt(ssum[0]);
        }
        else
        {
            resids[blockIdx.x] = ssum[0];
        }
    }
}

__global__ void s_resids_kernel(std::size_t m, std::size_t n, const float* A,
                                std::size_t lda, const float* B,
                                std::size_t ldb, float* ritzv, float* resids,
                                bool is_sqrt)
{
    __shared__ float ssum[NB_X];
    std::size_t tx = threadIdx.x;
    A += blockIdx.x * lda;
    B += blockIdx.x * lda;

    ssum[tx] = 0;
    for (std::size_t i = tx; i < m; i += NB_X)
    {
        float alpha;
        alpha = ritzv[blockIdx.x];
        float a = alpha * B[i];
        float b = A[i] - a;
        ssum[tx] += b * b;
    }

    cuda_sum_reduce<NB_X>(tx, ssum);
    if (tx == 0)
    {
        if (is_sqrt)
        {
            resids[blockIdx.x] = sqrtf(ssum[0]);
        }
        else
        {
            resids[blockIdx.x] = ssum[0];
        }
    }
}

void residual_gpu(std::size_t m, std::size_t n, std::complex<double>* dA,
                  std::size_t lda, std::complex<double>* dB, std::size_t ldb,
                  double* d_ritzv, double* d_resids, bool is_sqrt,
                  cudaStream_t stream_)
{
    dim3 threads(NB_X);
    dim3 grid(n);
    z_resids_kernel<<<grid, threads, 0, stream_>>>(
        m, n, reinterpret_cast<cuDoubleComplex*>(dA), lda,
        reinterpret_cast<cuDoubleComplex*>(dB), ldb, d_ritzv, d_resids,
        is_sqrt);
}

void residual_gpu(std::size_t m, std::size_t n, std::complex<float>* dA,
                  std::size_t lda, std::complex<float>* dB, std::size_t ldb,
                  float* d_ritzv, float* d_resids, bool is_sqrt,
                  cudaStream_t stream_)
{
    dim3 threads(NB_X);
    dim3 grid(n);
    c_resids_kernel<<<grid, threads, 0, stream_>>>(
        m, n, reinterpret_cast<cuComplex*>(dA), lda,
        reinterpret_cast<cuComplex*>(dB), ldb, d_ritzv, d_resids, is_sqrt);
}

void residual_gpu(std::size_t m, std::size_t n, double* dA, std::size_t lda,
                  double* dB, std::size_t ldb, double* d_ritzv,
                  double* d_resids, bool is_sqrt, cudaStream_t stream_)
{
    dim3 threads(NB_X);
    dim3 grid(n);
    d_resids_kernel<<<grid, threads, 0, stream_>>>(m, n, dA, lda, dB, ldb,
                                                   d_ritzv, d_resids, is_sqrt);
}

void residual_gpu(std::size_t m, std::size_t n, float* dA, std::size_t lda,
                  float* dB, std::size_t ldb, float* d_ritzv, float* d_resids,
                  bool is_sqrt, cudaStream_t stream_)
{
    dim3 threads(NB_X);
    dim3 grid(n);
    s_resids_kernel<<<grid, threads, 0, stream_>>>(m, n, dA, lda, dB, ldb,
                                                   d_ritzv, d_resids, is_sqrt);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase