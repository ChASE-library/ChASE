// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "pseudo_hermitian_lanczos_diag.cuh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda_nccl_ph_diag
{

namespace
{
inline void ph_diag_cuda_check(cudaError_t err, const char* expr,
                               const char* file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ":" << line << ' ' << expr
                  << " -> " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}
} // namespace

#define PH_DIAG_CUDA_CHECK(x) ph_diag_cuda_check((x), #x, __FILE__, __LINE__)

constexpr int kDiagBlockThreads = 256;

namespace
{
__device__ __forceinline__ void atomic_min_double(double* address, double val)
{
    auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        const double cur = __longlong_as_double(static_cast<long long>(assumed));
        const double newv = fmin(val, cur);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(newv));
    } while (assumed != old);
}

// Local tile matches LAPACK/BLAS column-major (same as BlockBlockMatrix::l_data):
//   entry (local row i, local column j) is A[i + j * ld], 0<=i<lrows, 0<=j<lcols, ld>=lrows.
// Linear index t = i + j * lrows visits columns in order with row i varying fastest.

__global__ void sum_sq_B_block_float(const float* A, std::size_t ld,
                                     std::size_t lrows, std::size_t lcols,
                                     std::size_t g_row_off, std::size_t g_col_off,
                                     std::size_t half_m, std::size_t N,
                                     double* d_sum)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows; // local column (column-major)
        const std::size_t i = t % lrows; // local row
        const std::size_t gj = g_col_off + j;
        const std::size_t gi = g_row_off + i;
        if (gj < half_m || gj >= N || gi >= half_m)
            continue;
        const float v = A[i + j * ld];
        atomicAdd(d_sum, static_cast<double>(v) * static_cast<double>(v));
    }
}

__global__ void sum_sq_B_block_double(const double* A, std::size_t ld,
                                      std::size_t lrows, std::size_t lcols,
                                      std::size_t g_row_off, std::size_t g_col_off,
                                      std::size_t half_m, std::size_t N,
                                      double* d_sum)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows;
        const std::size_t i = t % lrows;
        const std::size_t gj = g_col_off + j;
        const std::size_t gi = g_row_off + i;
        if (gj < half_m || gj >= N || gi >= half_m)
            continue;
        const double v = A[i + j * ld];
        atomicAdd(d_sum, v * v);
    }
}

__global__ void sum_sq_B_block_cfloat(const cuComplex* A, std::size_t ld,
                                     std::size_t lrows, std::size_t lcols,
                                     std::size_t g_row_off, std::size_t g_col_off,
                                     std::size_t half_m, std::size_t N,
                                     double* d_sum)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows;
        const std::size_t i = t % lrows;
        const std::size_t gj = g_col_off + j;
        const std::size_t gi = g_row_off + i;
        if (gj < half_m || gj >= N || gi >= half_m)
            continue;
        const cuComplex z = A[i + j * ld];
        const double re = static_cast<double>(z.x);
        const double im = static_cast<double>(z.y);
        atomicAdd(d_sum, re * re + im * im);
    }
}

__global__ void sum_sq_B_block_zdouble(const cuDoubleComplex* A, std::size_t ld,
                                       std::size_t lrows, std::size_t lcols,
                                       std::size_t g_row_off, std::size_t g_col_off,
                                       std::size_t half_m, std::size_t N,
                                       double* d_sum)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows;
        const std::size_t i = t % lrows;
        const std::size_t gj = g_col_off + j;
        const std::size_t gi = g_row_off + i;
        if (gj < half_m || gj >= N || gi >= half_m)
            continue;
        const cuDoubleComplex z = A[i + j * ld];
        atomicAdd(d_sum, z.x * z.x + z.y * z.y);
    }
}

__global__ void min_abs_global_diag_float(const float* A, std::size_t ld,
                                          std::size_t lrows, std::size_t lcols,
                                          std::size_t g_row_off,
                                          std::size_t g_col_off, std::size_t N,
                                          double* d_min)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows;
        const std::size_t i = t % lrows;
        const std::size_t gi = g_row_off + i;
        const std::size_t gj = g_col_off + j;
        if (gi != gj || gi >= N)
            continue;
        const float v = A[i + j * ld];
        atomic_min_double(d_min, static_cast<double>(fabsf(v)));
    }
}

__global__ void min_abs_global_diag_double(const double* A, std::size_t ld,
                                           std::size_t lrows, std::size_t lcols,
                                           std::size_t g_row_off,
                                           std::size_t g_col_off, std::size_t N,
                                           double* d_min)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows;
        const std::size_t i = t % lrows;
        const std::size_t gi = g_row_off + i;
        const std::size_t gj = g_col_off + j;
        if (gi != gj || gi >= N)
            continue;
        const double v = A[i + j * ld];
        atomic_min_double(d_min, fabs(v));
    }
}

__global__ void min_abs_global_diag_cfloat(const cuComplex* A, std::size_t ld,
                                           std::size_t lrows, std::size_t lcols,
                                           std::size_t g_row_off,
                                           std::size_t g_col_off, std::size_t N,
                                           double* d_min)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows;
        const std::size_t i = t % lrows;
        const std::size_t gi = g_row_off + i;
        const std::size_t gj = g_col_off + j;
        if (gi != gj || gi >= N)
            continue;
        const cuComplex z = A[i + j * ld];
        const double ab =
            hypot(static_cast<double>(z.x), static_cast<double>(z.y));
        atomic_min_double(d_min, ab);
    }
}

__global__ void min_abs_global_diag_zdouble(const cuDoubleComplex* A,
                                            std::size_t ld, std::size_t lrows,
                                            std::size_t lcols,
                                            std::size_t g_row_off,
                                            std::size_t g_col_off,
                                            std::size_t N, double* d_min)
{
    const std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t nelt = lrows * lcols;
    for (std::size_t t = tid; t < nelt; t += blockDim.x * gridDim.x)
    {
        const std::size_t j = t / lrows;
        const std::size_t i = t % lrows;
        const std::size_t gi = g_row_off + i;
        const std::size_t gj = g_col_off + j;
        if (gi != gj || gi >= N)
            continue;
        const cuDoubleComplex z = A[i + j * ld];
        const double ab = hypot(z.x, z.y);
        atomic_min_double(d_min, ab);
    }
}

} // namespace

void chase_ph_diag_B_fro_sq_float(const float* dA, std::size_t ld,
                                  std::size_t lrows, std::size_t lcols,
                                  std::size_t g_row_off, std::size_t g_col_off,
                                  std::size_t half_m, std::size_t N,
                                  cudaStream_t stream, double* h_sum_sq)
{
    double* d_sum = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    PH_DIAG_CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    sum_sq_B_block_float<<<nb, kDiagBlockThreads, 0, stream>>>(
        dA, ld, lrows, lcols, g_row_off, g_col_off, half_m, N, d_sum);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_sum_sq, d_sum, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_sum));
}

void chase_ph_diag_B_fro_sq_double(const double* dA, std::size_t ld,
                                   std::size_t lrows, std::size_t lcols,
                                   std::size_t g_row_off, std::size_t g_col_off,
                                   std::size_t half_m, std::size_t N,
                                   cudaStream_t stream, double* h_sum_sq)
{
    double* d_sum = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    PH_DIAG_CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    sum_sq_B_block_double<<<nb, kDiagBlockThreads, 0, stream>>>(
        dA, ld, lrows, lcols, g_row_off, g_col_off, half_m, N, d_sum);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_sum_sq, d_sum, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_sum));
}

void chase_ph_diag_B_fro_sq_complex_float(
    const std::complex<float>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t half_m, std::size_t N, cudaStream_t stream, double* h_sum_sq)
{
    double* d_sum = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    PH_DIAG_CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    sum_sq_B_block_cfloat<<<nb, kDiagBlockThreads, 0, stream>>>(
        reinterpret_cast<const cuComplex*>(dA), ld, lrows, lcols, g_row_off,
        g_col_off, half_m, N, d_sum);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_sum_sq, d_sum, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_sum));
}

void chase_ph_diag_B_fro_sq_complex_double(
    const std::complex<double>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t half_m, std::size_t N, cudaStream_t stream, double* h_sum_sq)
{
    double* d_sum = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_sum, sizeof(double)));
    PH_DIAG_CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    sum_sq_B_block_zdouble<<<nb, kDiagBlockThreads, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(dA), ld, lrows, lcols,
        g_row_off, g_col_off, half_m, N, d_sum);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_sum_sq, d_sum, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_sum));
}

void chase_ph_diag_min_abs_diag_float(const float* dA, std::size_t ld,
                                      std::size_t lrows, std::size_t lcols,
                                      std::size_t g_row_off, std::size_t g_col_off,
                                      std::size_t N, cudaStream_t stream,
                                      double* h_min_abs)
{
    double* d_min = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_min, sizeof(double)));
    const double inf = std::numeric_limits<double>::infinity();
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(d_min, &inf, sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    min_abs_global_diag_float<<<nb, kDiagBlockThreads, 0, stream>>>(
        dA, ld, lrows, lcols, g_row_off, g_col_off, N, d_min);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_min_abs, d_min, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_min));
}

void chase_ph_diag_min_abs_diag_double(const double* dA, std::size_t ld,
                                       std::size_t lrows, std::size_t lcols,
                                       std::size_t g_row_off, std::size_t g_col_off,
                                       std::size_t N, cudaStream_t stream,
                                       double* h_min_abs)
{
    double* d_min = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_min, sizeof(double)));
    const double inf = std::numeric_limits<double>::infinity();
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(d_min, &inf, sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    min_abs_global_diag_double<<<nb, kDiagBlockThreads, 0, stream>>>(
        dA, ld, lrows, lcols, g_row_off, g_col_off, N, d_min);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_min_abs, d_min, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_min));
}

void chase_ph_diag_min_abs_diag_complex_float(
    const std::complex<float>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t N, cudaStream_t stream, double* h_min_abs)
{
    double* d_min = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_min, sizeof(double)));
    const double inf = std::numeric_limits<double>::infinity();
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(d_min, &inf, sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    min_abs_global_diag_cfloat<<<nb, kDiagBlockThreads, 0, stream>>>(
        reinterpret_cast<const cuComplex*>(dA), ld, lrows, lcols, g_row_off,
        g_col_off, N, d_min);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_min_abs, d_min, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_min));
}

void chase_ph_diag_min_abs_diag_complex_double(
    const std::complex<double>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t N, cudaStream_t stream, double* h_min_abs)
{
    double* d_min = nullptr;
    PH_DIAG_CUDA_CHECK(cudaMalloc(&d_min, sizeof(double)));
    const double inf = std::numeric_limits<double>::infinity();
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(d_min, &inf, sizeof(double),
                                       cudaMemcpyHostToDevice, stream));
    const std::size_t nelt = lrows * lcols;
    const int blocks =
        static_cast<int>(std::min<std::size_t>(
            (nelt + kDiagBlockThreads - 1) / kDiagBlockThreads,
            std::size_t(4096)));
    const int nb = std::max(blocks, 1);
    min_abs_global_diag_zdouble<<<nb, kDiagBlockThreads, 0, stream>>>(
        reinterpret_cast<const cuDoubleComplex*>(dA), ld, lrows, lcols,
        g_row_off, g_col_off, N, d_min);
    PH_DIAG_CUDA_CHECK(cudaMemcpyAsync(h_min_abs, d_min, sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
    PH_DIAG_CUDA_CHECK(cudaStreamSynchronize(stream));
    PH_DIAG_CUDA_CHECK(cudaFree(d_min));
}

#undef PH_DIAG_CUDA_CHECK

} // namespace cuda_nccl_ph_diag
} // namespace internal
} // namespace linalg
} // namespace chase
