// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "lanczos_reduction.hpp"

#include "linalg/internal/gpu/backend/error.hpp"

#include <hip/hip_runtime.h>

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

__device__ void reduce_block_sum(double* values)
{
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        if (threadIdx.x < stride)
        {
            values[threadIdx.x] += values[threadIdx.x + stride];
        }
    }
    __syncthreads();
}

__global__ void lanczos_dot_float_double_accum_kernel(
    std::size_t n, const float* x, std::size_t incx, const float* y,
    std::size_t incy, double* result)
{
    __shared__ double partial[block_size];
    double sum = 0.0;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n;
         idx += stride)
    {
        sum += static_cast<double>(x[idx * incx]) *
               static_cast<double>(y[idx * incy]);
    }

    partial[threadIdx.x] = sum;
    reduce_block_sum(partial);
    if (threadIdx.x == 0)
    {
        atomicAdd(result, partial[0]);
    }
}

__global__ void lanczos_nrm2_float_double_accum_kernel(
    std::size_t n, const float* x, std::size_t incx, double* result)
{
    __shared__ double partial[block_size];
    double sum = 0.0;
    const std::size_t stride =
        static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < n;
         idx += stride)
    {
        const double xi = static_cast<double>(x[idx * incx]);
        sum += xi * xi;
    }

    partial[threadIdx.x] = sum;
    reduce_block_sum(partial);
    if (threadIdx.x == 0)
    {
        atomicAdd(result, partial[0]);
    }
}

dim3 reduction_grid(std::size_t n)
{
    const std::size_t blocks = (n + block_size - 1) / block_size;
    const std::size_t capped = blocks < 1024 ? blocks : 1024;
    return dim3(static_cast<unsigned int>(capped == 0 ? 1 : capped));
}

} // namespace

void lanczos_dot_float_double_accum(std::size_t n, const float* x,
                                    std::size_t incx, const float* y,
                                    std::size_t incy, double* d_result,
                                    hipStream_t stream)
{
    CHECK_HIP_ERROR(hipMemsetAsync(d_result, 0, sizeof(double), stream));
    lanczos_dot_float_double_accum_kernel<<<reduction_grid(n), block_size, 0,
                                            stream>>>(
        n, x, incx, y, incy, d_result);
    CHECK_HIP_ERROR(hipPeekAtLastError());
}

void lanczos_nrm2_float_double_accum(std::size_t n, const float* x,
                                     std::size_t incx, double* d_result,
                                     hipStream_t stream)
{
    CHECK_HIP_ERROR(hipMemsetAsync(d_result, 0, sizeof(double), stream));
    lanczos_nrm2_float_double_accum_kernel<<<reduction_grid(n), block_size, 0,
                                             stream>>>(n, x, incx, d_result);
    CHECK_HIP_ERROR(hipPeekAtLastError());
}

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase
