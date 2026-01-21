// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "lacpy.cuh"
#include <algorithm>

#define BLK_X 64
#define BLK_Y BLK_X
const std::size_t max_blocks = 65535;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

__device__ inline std::size_t device_min(std::size_t a, std::size_t b)
{
    return (a < b) ? a : b;
}

__device__ inline std::size_t device_max(std::size_t a, std::size_t b)
{
    return (a > b) ? a : b;
}

static __device__ void dlacpy_full_device(std::size_t m, std::size_t n,
                                          const double* dA, std::size_t ldda,
                                          double* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
    }
}

__global__ void dlacpy_full_kernel(std::size_t m, std::size_t n,
                                   const double* dA, std::size_t ldda,
                                   double* dB, std::size_t lddb)
{
    dlacpy_full_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void slacpy_full_device(std::size_t m, std::size_t n,
                                          const float* dA, std::size_t ldda,
                                          float* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
    }
}

__global__ void slacpy_full_kernel(std::size_t m, std::size_t n,
                                   const float* dA, std::size_t ldda, float* dB,
                                   std::size_t lddb)
{
    slacpy_full_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void zlacpy_full_device(std::size_t m, std::size_t n,
                                          const cuDoubleComplex* dA,
                                          std::size_t ldda, cuDoubleComplex* dB,
                                          std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
    }
}

__global__ void zlacpy_full_kernel(std::size_t m, std::size_t n,
                                   const cuDoubleComplex* dA, std::size_t ldda,
                                   cuDoubleComplex* dB, std::size_t lddb)
{
    zlacpy_full_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_full_device(std::size_t m, std::size_t n,
                                          const cuComplex* dA, std::size_t ldda,
                                          cuComplex* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = dA[j * ldda];
            }
        }
    }
}

__global__ void clacpy_full_kernel(std::size_t m, std::size_t n,
                                   const cuComplex* dA, std::size_t ldda,
                                   cuComplex* dB, std::size_t lddb)
{
    clacpy_full_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void dlacpy_upper_device(std::size_t m, std::size_t n,
                                           const double* dA, std::size_t ldda,
                                           double* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyLimit =
            device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y));
        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j < copyLimit) ? dA[j * ldda] : 0.0;
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j < copyLimit) ? dA[j * ldda] : 0.0;
            }
        }
    }
}

__global__ void dlacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const double* dA, std::size_t ldda,
                                    double* dB, std::size_t lddb)
{
    dlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void slacpy_upper_device(std::size_t m, std::size_t n,
                                           const float* dA, std::size_t ldda,
                                           float* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyLimit =
            device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y));
        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j < copyLimit) ? dA[j * ldda] : 0.0;
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j < copyLimit) ? dA[j * ldda] : 0.0;
            }
        }
    }
}

__global__ void slacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const float* dA, std::size_t ldda,
                                    float* dB, std::size_t lddb)
{
    slacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_upper_device(std::size_t m, std::size_t n,
                                           const cuComplex* dA,
                                           std::size_t ldda, cuComplex* dB,
                                           std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyLimit =
            device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y));

        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j < copyLimit)
                                   ? dA[j * ldda]
                                   : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j < copyLimit)
                                   ? dA[j * ldda]
                                   : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
    }
}

__global__ void clacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const cuComplex* dA, std::size_t ldda,
                                    cuComplex* dB, std::size_t lddb)
{
    clacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void
zlacpy_upper_device(std::size_t m, std::size_t n, const cuDoubleComplex* dA,
                    std::size_t ldda, cuDoubleComplex* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyLimit =
            device_min(ind - iby + 1, static_cast<std::size_t>(BLK_Y));

        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j < copyLimit) ? dA[j * ldda]
                                               : make_cuDoubleComplex(0.0, 0.0);
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j < copyLimit) ? dA[j * ldda]
                                               : make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
}

__global__ void zlacpy_upper_kernel(std::size_t m, std::size_t n,
                                    const cuDoubleComplex* dA, std::size_t ldda,
                                    cuDoubleComplex* dB, std::size_t lddb)
{
    zlacpy_upper_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void dlacpy_lower_device(std::size_t m, std::size_t n,
                                           const double* dA, std::size_t ldda,
                                           double* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyStart =
            device_max(ind - iby, static_cast<std::size_t>(0));

        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j >= copyStart) ? dA[j * ldda] : 0.0;
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j >= copyStart) ? dA[j * ldda] : 0.0;
            }
        }
    }
}

__global__ void dlacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const double* dA, std::size_t ldda,
                                    double* dB, std::size_t lddb)
{
    dlacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void slacpy_lower_device(std::size_t m, std::size_t n,
                                           const float* dA, std::size_t ldda,
                                           float* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyStart =
            device_max(ind - iby, static_cast<std::size_t>(0));
        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j >= copyStart) ? dA[j * ldda] : 0.0;
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j >= copyStart) ? dA[j * ldda] : 0.0;
            }
        }
    }
}

__global__ void slacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const float* dA, std::size_t ldda,
                                    float* dB, std::size_t lddb)
{
    slacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void clacpy_lower_device(std::size_t m, std::size_t n,
                                           const cuComplex* dA,
                                           std::size_t ldda, cuComplex* dB,
                                           std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyStart =
            device_max(ind - iby, static_cast<std::size_t>(0));

        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j >= copyStart)
                                   ? dA[j * ldda]
                                   : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j >= copyStart)
                                   ? dA[j * ldda]
                                   : make_cuFloatComplex(0.0f, 0.0f);
            }
        }
    }
}

__global__ void clacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const cuComplex* dA, std::size_t ldda,
                                    cuComplex* dB, std::size_t lddb)
{
    clacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

static __device__ void
zlacpy_lower_device(std::size_t m, std::size_t n, const cuDoubleComplex* dA,
                    std::size_t ldda, cuDoubleComplex* dB, std::size_t lddb)
{
    std::size_t ind = blockIdx.x * BLK_X + threadIdx.x;
    std::size_t iby = blockIdx.y * BLK_Y;
    bool full = (iby + BLK_Y <= n);
    if (ind < m)
    {
        dA += ind + iby * ldda;
        dB += ind + iby * lddb;
        std::size_t copyStart =
            device_max(ind - iby, static_cast<std::size_t>(0));

        if (full)
        {
#pragma unroll
            for (std::size_t j = 0; j < BLK_Y; ++j)
            {
                dB[j * lddb] = (j >= copyStart)
                                   ? dA[j * ldda]
                                   : make_cuDoubleComplex(0.0, 0.0);
            }
        }
        else
        {
            for (std::size_t j = 0; j < BLK_Y && iby + j < n; ++j)
            {
                dB[j * lddb] = (j >= copyStart)
                                   ? dA[j * ldda]
                                   : make_cuDoubleComplex(0.0, 0.0);
            }
        }
    }
}

__global__ void zlacpy_lower_kernel(std::size_t m, std::size_t n,
                                    const cuDoubleComplex* dA, std::size_t ldda,
                                    cuDoubleComplex* dB, std::size_t lddb)
{
    zlacpy_lower_device(m, n, dA, ldda, dB, lddb);
}

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, float* dA,
                 std::size_t ldda, float* dB, std::size_t lddb,
                 cudaStream_t stream_)
{
#define dA(i_, j_) (dA + (i_) + (j_) * ldda)
#define dB(i_, j_) (dB + (i_) + (j_) * lddb)
    std::size_t super_NB = max_blocks * BLK_X;
    dim3 super_grid((m + super_NB - 1) / super_NB,
                    (n + super_NB - 1) / super_NB);

    dim3 threads(BLK_X, 1);
    dim3 grid;

    std::size_t mm, nn;
    if (uplo == 'L')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    slacpy_upper_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    slacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else if (uplo == 'U')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    slacpy_lower_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    slacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                slacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                    mm, nn, dA(i * super_NB, j * super_NB), ldda,
                    dB(i * super_NB, j * super_NB), lddb);
            }
        }
    }
}

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n, double* dA,
                 std::size_t ldda, double* dB, std::size_t lddb,
                 cudaStream_t stream_)
{
#define dA(i_, j_) (dA + (i_) + (j_) * ldda)
#define dB(i_, j_) (dB + (i_) + (j_) * lddb)
    std::size_t super_NB = max_blocks * BLK_X;
    dim3 super_grid((m + super_NB - 1) / super_NB,
                    (n + super_NB - 1) / super_NB);

    dim3 threads(BLK_X, 1);
    dim3 grid;

    std::size_t mm, nn;
    if (uplo == 'L')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    dlacpy_upper_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    dlacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else if (uplo == 'U')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    dlacpy_lower_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    dlacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                dlacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                    mm, nn, dA(i * super_NB, j * super_NB), ldda,
                    dB(i * super_NB, j * super_NB), lddb);
            }
        }
    }
}

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n,
                 std::complex<double>* ddA, std::size_t ldda,
                 std::complex<double>* ddB, std::size_t lddb,
                 cudaStream_t stream_)
{
    cuDoubleComplex* dA = reinterpret_cast<cuDoubleComplex*>(ddA);
    cuDoubleComplex* dB = reinterpret_cast<cuDoubleComplex*>(ddB);
#define dA(i_, j_) (dA + (i_) + (j_) * ldda)
#define dB(i_, j_) (dB + (i_) + (j_) * lddb)
    std::size_t super_NB = max_blocks * BLK_X;
    dim3 super_grid((m + super_NB - 1) / super_NB,
                    (n + super_NB - 1) / super_NB);

    dim3 threads(BLK_X, 1);
    dim3 grid;

    std::size_t mm, nn;
    if (uplo == 'L')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    zlacpy_upper_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    zlacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else if (uplo == 'U')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    zlacpy_lower_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    zlacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                zlacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                    mm, nn, dA(i * super_NB, j * super_NB), ldda,
                    dB(i * super_NB, j * super_NB), lddb);
            }
        }
    }
}

void t_lacpy_gpu(char uplo, std::size_t m, std::size_t n,
                 std::complex<float>* ddA, std::size_t ldda,
                 std::complex<float>* ddB, std::size_t lddb,
                 cudaStream_t stream_)
{
    cuComplex* dA = reinterpret_cast<cuComplex*>(ddA);
    cuComplex* dB = reinterpret_cast<cuComplex*>(ddB);
#define dA(i_, j_) (dA + (i_) + (j_) * ldda)
#define dB(i_, j_) (dB + (i_) + (j_) * lddb)
    std::size_t super_NB = max_blocks * BLK_X;
    dim3 super_grid((m + super_NB - 1) / super_NB,
                    (n + super_NB - 1) / super_NB);

    dim3 threads(BLK_X, 1);
    dim3 grid;

    std::size_t mm, nn;
    if (uplo == 'L')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    clacpy_upper_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    clacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else if (uplo == 'U')
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                if (i == j)
                { // diagonal super block
                    clacpy_lower_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
                else // off diagonal super block
                {
                    clacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                        mm, nn, dA(i * super_NB, j * super_NB), ldda,
                        dB(i * super_NB, j * super_NB), lddb);
                }
            }
        }
    }
    else
    {
        for (std::size_t i = 0; i < super_grid.x; ++i)
        {
            mm = (i == super_grid.x - 1 ? m % super_NB : super_NB);
            grid.x = (mm + BLK_X - 1) / BLK_X;
            for (std::size_t j = 0; j < super_grid.y; ++j)
            { // full row
                nn = (j == super_grid.y - 1 ? n % super_NB : super_NB);
                grid.y = (nn + BLK_X - 1) / BLK_Y;
                clacpy_full_kernel<<<grid, threads, 0, stream_>>>(
                    mm, nn, dA(i * super_NB, j * super_NB), ldda,
                    dB(i * super_NB, j * super_NB), lddb);
            }
        }
    }
}

__global__ void d_extractUpperTriangularKernel(const double* matrix,
                                               std::size_t ld,
                                               double* upperTriangular,
                                               std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n && row <= col)
    {
        std::size_t linearIndex = row + col * ld; // Column-major indexing
        std::size_t upperTriIndex =
            (row * (2 * n - row + 1)) / 2 + (col - row); // Packed index
        upperTriangular[upperTriIndex] = matrix[linearIndex];
    }
}

__global__ void s_extractUpperTriangularKernel(const float* matrix,
                                               std::size_t ld,
                                               float* upperTriangular,
                                               std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n && row <= col)
    {
        std::size_t linearIndex = row + col * ld; // Column-major indexing
        std::size_t upperTriIndex =
            (row * (2 * n - row + 1)) / 2 + (col - row); // Packed index
        upperTriangular[upperTriIndex] = matrix[linearIndex];
    }
}

__global__ void c_extractUpperTriangularKernel(const cuComplex* matrix,
                                               std::size_t ld,
                                               cuComplex* upperTriangular,
                                               std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n && row <= col)
    {
        std::size_t linearIndex = row + col * ld; // Column-major indexing
        std::size_t upperTriIndex =
            (row * (2 * n - row + 1)) / 2 + (col - row); // Packed index
        upperTriangular[upperTriIndex] = matrix[linearIndex];
    }
}

__global__ void z_extractUpperTriangularKernel(const cuDoubleComplex* matrix,
                                               std::size_t ld,
                                               cuDoubleComplex* upperTriangular,
                                               std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n && row <= col)
    {
        std::size_t linearIndex = row + col * ld; // Column-major indexing
        std::size_t upperTriIndex =
            (row * (2 * n - row + 1)) / 2 + (col - row); // Packed index
        upperTriangular[upperTriIndex] = matrix[linearIndex];
    }
}

void extractUpperTriangular(float* d_matrix, std::size_t ld,
                            float* d_upperTriangular, std::size_t n,
                            cudaStream_t stream_)
{
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    s_extractUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        d_matrix, ld, d_upperTriangular, n);
}
/*
 */
void extractUpperTriangular(double* d_matrix, std::size_t ld,
                            double* d_upperTriangular, std::size_t n,
                            cudaStream_t stream_)
{
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    d_extractUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        d_matrix, ld, d_upperTriangular, n);
}
void extractUpperTriangular(std::complex<double>* d_matrix, std::size_t ld,
                            std::complex<double>* d_upperTriangular,
                            std::size_t n, cudaStream_t stream_)
{
    cuDoubleComplex* dd_matrix = reinterpret_cast<cuDoubleComplex*>(d_matrix);
    cuDoubleComplex* dd_upperTriangular =
        reinterpret_cast<cuDoubleComplex*>(d_upperTriangular);

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    z_extractUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        dd_matrix, ld, dd_upperTriangular, n);
}
void extractUpperTriangular(std::complex<float>* d_matrix, std::size_t ld,
                            std::complex<float>* d_upperTriangular,
                            std::size_t n, cudaStream_t stream_)
{
    cuComplex* dd_matrix = reinterpret_cast<cuComplex*>(d_matrix);
    cuComplex* dd_upperTriangular =
        reinterpret_cast<cuComplex*>(d_upperTriangular);

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    c_extractUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        dd_matrix, ld, dd_upperTriangular, n);
}

__global__ void s_unpackUpperTriangularKernel(const float* upperTriangular,
                                              std::size_t ld, float* matrix,
                                              std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        if (row <= col)
        {
            std::size_t index =
                (row * (2 * n - row + 1)) / 2 +
                (col - row); // Packed index for upper triangular
            matrix[row + col * ld] =
                upperTriangular[index]; // Column-major indexing with leading
                                        // dimension
        }
    }
}

__global__ void d_unpackUpperTriangularKernel(const double* upperTriangular,
                                              std::size_t ld, double* matrix,
                                              std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        if (row <= col)
        {
            std::size_t index =
                (row * (2 * n - row + 1)) / 2 +
                (col - row); // Packed index for upper triangular
            matrix[row + col * ld] =
                upperTriangular[index]; // Column-major indexing with leading
                                        // dimension
        }
    }
}

__global__ void c_unpackUpperTriangularKernel(const cuComplex* upperTriangular,
                                              std::size_t ld, cuComplex* matrix,
                                              std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        if (row <= col)
        {
            std::size_t index =
                (row * (2 * n - row + 1)) / 2 +
                (col - row); // Packed index for upper triangular
            matrix[row + col * ld] =
                upperTriangular[index]; // Column-major indexing with leading
                                        // dimension
        }
    }
}

__global__ void
z_unpackUpperTriangularKernel(const cuDoubleComplex* upperTriangular,
                              std::size_t ld, cuDoubleComplex* matrix,
                              std::size_t n)
{
    std::size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    std::size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n)
    {
        if (row <= col)
        {
            std::size_t index =
                (row * (2 * n - row + 1)) / 2 +
                (col - row); // Packed index for upper triangular
            matrix[row + col * ld] =
                upperTriangular[index]; // Column-major indexing with leading
                                        // dimension
        }
    }
}

void unpackUpperTriangular(float* d_matrix, std::size_t ld,
                           float* d_upperTriangular, std::size_t n,
                           cudaStream_t stream_)
{
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    s_unpackUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        d_upperTriangular, ld, d_matrix, n);
}

void unpackUpperTriangular(double* d_matrix, std::size_t ld,
                           double* d_upperTriangular, std::size_t n,
                           cudaStream_t stream_)
{
    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    d_unpackUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        d_upperTriangular, ld, d_matrix, n);
}

void unpackUpperTriangular(std::complex<double>* d_matrix, std::size_t ld,
                           std::complex<double>* d_upperTriangular,
                           std::size_t n, cudaStream_t stream_)
{
    cuDoubleComplex* dd_matrix = reinterpret_cast<cuDoubleComplex*>(d_matrix);
    cuDoubleComplex* dd_upperTriangular =
        reinterpret_cast<cuDoubleComplex*>(d_upperTriangular);

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    z_unpackUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        dd_upperTriangular, ld, dd_matrix, n);
}
void unpackUpperTriangular(std::complex<float>* d_matrix, std::size_t ld,
                           std::complex<float>* d_upperTriangular,
                           std::size_t n, cudaStream_t stream_)
{
    cuComplex* dd_matrix = reinterpret_cast<cuComplex*>(d_matrix);
    cuComplex* dd_upperTriangular =
        reinterpret_cast<cuComplex*>(d_upperTriangular);

    dim3 threads(16, 16);
    dim3 blocks((n + threads.x - 1) / threads.x,
                (n + threads.y - 1) / threads.y);
    c_unpackUpperTriangularKernel<<<blocks, threads, 0, stream_>>>(
        dd_upperTriangular, ld, dd_matrix, n);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase