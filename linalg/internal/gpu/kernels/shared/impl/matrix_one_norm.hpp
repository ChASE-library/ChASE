// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <algorithm>
#include <complex>
#include <cstddef>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace CHASE_GPU_DEVICE_NAMESPACE
{
namespace matrix_one_norm_detail
{

// One warp owns one column and walks it contiguously. A block holds
// kTileCols such columns. Tall columns are split along rows.
constexpr unsigned kRowLanes = 32;
constexpr unsigned kTileCols = 8;
constexpr unsigned kUnroll = 4;
constexpr unsigned kFinalizeThreads = 256;
constexpr unsigned kTargetBlocksPerSM = 8;
constexpr std::size_t kMinRowsPerSplit = 256;
constexpr std::size_t kMaxSplits = 65535; // gridDim.y limit

// |Re| + |Im| for complex values (BLAS asum convention), as cuBLAS asum does.
__device__ __forceinline__ float abs1(float x) { return fabsf(x); }
__device__ __forceinline__ double abs1(double x) { return fabs(x); }
__device__ __forceinline__ float abs1(CHASE_GPU_FLOAT_COMPLEX x)
{
    return fabsf(CHASE_GPU_CREAL_FLOAT(x)) + fabsf(CHASE_GPU_CIMAG_FLOAT(x));
}
__device__ __forceinline__ double abs1(CHASE_GPU_DOUBLE_COMPLEX x)
{
    return fabs(CHASE_GPU_CREAL_DOUBLE(x)) + fabs(CHASE_GPU_CIMAG_DOUBLE(x));
}

template <typename R>
__device__ __forceinline__ R nan_max(R a, R b)
{
    return (b > a || b != b) ? b : a;
}

// Non-negative IEEE values (and NaN with cleared sign) order like their bits.
__device__ __forceinline__ void atomic_max_nonneg(float* addr, float v)
{
    atomicMax(reinterpret_cast<unsigned int*>(addr), __float_as_uint(v));
}
__device__ __forceinline__ void atomic_max_nonneg(double* addr, double v)
{
    atomicMax(reinterpret_cast<unsigned long long*>(addr),
              static_cast<unsigned long long>(__double_as_longlong(v)));
}

// kFused: the block sees whole columns and folds its column maxima into `result`.
// Otherwise it writes partial[blockIdx.y * cols + col] for a later pass.
template <bool kFused, typename T, typename R>
__global__ void __launch_bounds__(kRowLanes* kTileCols)
    col_abs_sum_kernel(const T* __restrict__ A, std::size_t rows,
                       std::size_t cols, std::size_t lda,
                       std::size_t rows_per_split, R* __restrict__ partial,
                       R* __restrict__ result)
{
    __shared__ R lane_sums[kTileCols][kRowLanes];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const std::size_t col =
        static_cast<std::size_t>(blockIdx.x) * kTileCols + ty;
    const std::size_t row_begin =
        static_cast<std::size_t>(blockIdx.y) * rows_per_split + tx;
    const std::size_t row_end =
        (static_cast<std::size_t>(blockIdx.y) + 1) * rows_per_split < rows
            ? (static_cast<std::size_t>(blockIdx.y) + 1) * rows_per_split
            : rows;

    R acc[kUnroll];
#pragma unroll
    for (unsigned u = 0; u < kUnroll; ++u)
    {
        acc[u] = R(0);
    }

    if (col < cols && row_begin < row_end)
    {
        const std::size_t count =
            (row_end - row_begin + kRowLanes - 1) / kRowLanes;
        const T* p = A + row_begin + col * lda;

        std::size_t k = 0;
        for (; k + kUnroll <= count; k += kUnroll)
        {
#pragma unroll
            for (unsigned u = 0; u < kUnroll; ++u)
            {
                acc[u] += abs1(p[u * kRowLanes]);
            }
            p += kUnroll * kRowLanes;
        }
        for (; k < count; ++k)
        {
            acc[0] += abs1(*p);
            p += kRowLanes;
        }
    }

    R lane_sum = acc[0];
#pragma unroll
    for (unsigned u = 1; u < kUnroll; ++u)
    {
        lane_sum += acc[u];
    }
    lane_sums[ty][tx] = lane_sum;
    __syncthreads();

    for (unsigned offset = kRowLanes / 2; offset > 0; offset >>= 1)
    {
        if (tx < offset)
        {
            lane_sums[ty][tx] += lane_sums[ty][tx + offset];
        }
        __syncthreads();
    }

    if constexpr (kFused)
    {
        if (tx == 0 && ty == 0)
        {
            R block_max = lane_sums[0][0];
            for (unsigned c = 1; c < kTileCols; ++c)
            {
                block_max = nan_max(block_max, lane_sums[c][0]);
            }
            atomic_max_nonneg(result, block_max);
        }
    }
    else if (tx == 0 && col < cols)
    {
        partial[static_cast<std::size_t>(blockIdx.y) * cols + col] =
            lane_sums[ty][0];
    }
}

// Sums the per-split partials of each column in a fixed order, then max-reduces.
template <typename R>
__global__ void __launch_bounds__(kFinalizeThreads)
    reduce_splits_max_kernel(const R* __restrict__ partial, std::size_t cols,
                             std::size_t splits, R* __restrict__ result)
{
    __shared__ R block_max[kFinalizeThreads];

    R thread_max = R(0);
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * kFinalizeThreads;
    for (std::size_t col =
             static_cast<std::size_t>(blockIdx.x) * kFinalizeThreads +
             threadIdx.x;
         col < cols; col += stride)
    {
        R column_sum = R(0);
        for (std::size_t s = 0; s < splits; ++s)
        {
            column_sum += partial[s * cols + col];
        }
        thread_max = nan_max(thread_max, column_sum);
    }
    block_max[threadIdx.x] = thread_max;
    __syncthreads();

#pragma unroll
    for (unsigned offset = kFinalizeThreads / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            block_max[threadIdx.x] =
                nan_max(block_max[threadIdx.x], block_max[threadIdx.x + offset]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomic_max_nonneg(result, block_max[0]);
    }
}

inline std::size_t multiprocessor_count()
{
    int device = 0;
    int count = 0;
    if (CHASE_GPU_GET_DEVICE(&device) != CHASE_GPU_SUCCESS ||
        CHASE_GPU_MULTIPROCESSOR_COUNT(&count, device) != CHASE_GPU_SUCCESS ||
        count <= 0)
    {
        return 1;
    }
    return static_cast<std::size_t>(count);
}

template <typename T, typename R>
void matrix_one_norm_launch(const T* A, std::size_t rows, std::size_t cols,
                            std::size_t lda, R* d_result,
                            CHASE_GPU_STREAM_T stream)
{
    static_cast<void>(CHASE_GPU_MEMSET_ASYNC(d_result, 0, sizeof(R), stream));
    if (rows == 0 || cols == 0)
    {
        return;
    }

    const std::size_t sms = multiprocessor_count();
    const std::size_t column_tiles = (cols + kTileCols - 1) / kTileCols;
    const std::size_t target_blocks = sms * kTargetBlocksPerSM;

    std::size_t splits = (target_blocks + column_tiles - 1) / column_tiles;
    splits = std::min(splits, (rows + kMinRowsPerSplit - 1) / kMinRowsPerSplit);
    splits = std::min(splits, kMaxSplits);
    splits = std::max<std::size_t>(splits, 1);

    R* partial = nullptr;
    if (splits > 1 &&
        CHASE_GPU_MALLOC_ASYNC(reinterpret_cast<void**>(&partial),
                               splits * cols * sizeof(R),
                               stream) != CHASE_GPU_SUCCESS)
    {
        static_cast<void>(CHASE_GPU_GET_LAST_ERROR());
        partial = nullptr;
        splits = 1;
    }

    const dim3 block(kRowLanes, kTileCols);

    if (splits == 1)
    {
        const dim3 grid(static_cast<unsigned>(column_tiles), 1);
        col_abs_sum_kernel<true><<<grid, block, 0, stream>>>(
            A, rows, cols, lda, rows, static_cast<R*>(nullptr), d_result);
        return;
    }

    const std::size_t rows_per_split = (rows + splits - 1) / splits;
    const dim3 grid(static_cast<unsigned>(column_tiles),
                    static_cast<unsigned>(splits));
    col_abs_sum_kernel<false><<<grid, block, 0, stream>>>(
        A, rows, cols, lda, rows_per_split, partial, static_cast<R*>(nullptr));

    const std::size_t finalize_blocks =
        std::min((cols + kFinalizeThreads - 1) / kFinalizeThreads, sms * 4);
    reduce_splits_max_kernel<<<static_cast<unsigned>(finalize_blocks),
                               kFinalizeThreads, 0, stream>>>(
        partial, cols, splits, d_result);

    static_cast<void>(CHASE_GPU_FREE_ASYNC(partial, stream));
}

} // namespace matrix_one_norm_detail

void matrixOneNorm_gpu(const float* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_one_norm_detail::matrix_one_norm_launch(d_matrix, rows, cols, ld,
                                                   d_norm, stream_);
}

void matrixOneNorm_gpu(const double* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_one_norm_detail::matrix_one_norm_launch(d_matrix, rows, cols, ld,
                                                   d_norm, stream_);
}

void matrixOneNorm_gpu(const std::complex<float>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_one_norm_detail::matrix_one_norm_launch(
        reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_matrix), rows, cols,
        ld, d_norm, stream_);
}

void matrixOneNorm_gpu(const std::complex<double>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_one_norm_detail::matrix_one_norm_launch(
        reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_matrix), rows, cols,
        ld, d_norm, stream_);
}

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase
