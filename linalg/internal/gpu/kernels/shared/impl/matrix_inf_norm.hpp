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
namespace matrix_inf_norm_detail
{

// A block covers kTileRows consecutive rows (one coalesced segment per
// column) and splits its column range over kColLanes lanes.
constexpr unsigned kTileRows = 32;
constexpr unsigned kColLanes = 8;
constexpr unsigned kUnroll = 4;
constexpr unsigned kFinalizeThreads = 256;
constexpr unsigned kTargetBlocksPerSM = 8;
constexpr std::size_t kMinColsPerSplit = 256;
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

// kFused: the block sees whole rows and folds its row maxima into `result`.
// Otherwise it writes partial[blockIdx.y * rows + row] for a later pass.
template <bool kFused, typename T, typename R>
__global__ void __launch_bounds__(kTileRows* kColLanes)
    row_abs_sum_kernel(const T* __restrict__ A, std::size_t rows,
                       std::size_t cols, std::size_t lda,
                       std::size_t cols_per_split, R* __restrict__ partial,
                       R* __restrict__ result)
{
    __shared__ R lane_sums[kColLanes][kTileRows];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const std::size_t row =
        static_cast<std::size_t>(blockIdx.x) * kTileRows + tx;
    const std::size_t col_begin =
        static_cast<std::size_t>(blockIdx.y) * cols_per_split + ty;
    const std::size_t col_end =
        (static_cast<std::size_t>(blockIdx.y) + 1) * cols_per_split < cols
            ? (static_cast<std::size_t>(blockIdx.y) + 1) * cols_per_split
            : cols;

    R acc[kUnroll];
#pragma unroll
    for (unsigned u = 0; u < kUnroll; ++u)
    {
        acc[u] = R(0);
    }

    if (row < rows && col_begin < col_end)
    {
        const std::size_t count = (col_end - col_begin + kColLanes - 1) / kColLanes;
        const std::size_t step = static_cast<std::size_t>(kColLanes) * lda;
        const T* p = A + row + col_begin * lda;

        std::size_t k = 0;
        for (; k + kUnroll <= count; k += kUnroll)
        {
#pragma unroll
            for (unsigned u = 0; u < kUnroll; ++u)
            {
                acc[u] += abs1(p[u * step]);
            }
            p += kUnroll * step;
        }
        for (; k < count; ++k)
        {
            acc[0] += abs1(*p);
            p += step;
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

    if (ty == 0)
    {
        R row_sum = lane_sums[0][tx];
#pragma unroll
        for (unsigned l = 1; l < kColLanes; ++l)
        {
            row_sum += lane_sums[l][tx];
        }

        if constexpr (kFused)
        {
            lane_sums[0][tx] = row_sum;
        }
        else if (row < rows)
        {
            partial[static_cast<std::size_t>(blockIdx.y) * rows + row] = row_sum;
        }
    }

    if constexpr (kFused)
    {
        __syncthreads();
        if (tx == 0 && ty == 0)
        {
            R block_max = lane_sums[0][0];
            for (unsigned r = 1; r < kTileRows; ++r)
            {
                block_max = nan_max(block_max, lane_sums[0][r]);
            }
            atomic_max_nonneg(result, block_max);
        }
    }
}

// Sums the per-split partials of each row into `row_sums` (no max).
template <typename R>
__global__ void __launch_bounds__(kFinalizeThreads)
    reduce_splits_sum_kernel(const R* __restrict__ partial, std::size_t rows,
                             std::size_t splits, R* __restrict__ row_sums)
{
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * kFinalizeThreads;
    for (std::size_t row =
             static_cast<std::size_t>(blockIdx.x) * kFinalizeThreads +
             threadIdx.x;
         row < rows; row += stride)
    {
        R row_sum = R(0);
        for (std::size_t s = 0; s < splits; ++s)
        {
            row_sum += partial[s * rows + row];
        }
        row_sums[row] = row_sum;
    }
}

// Sums the per-split partials of each row in a fixed order, then max-reduces.
template <typename R>
__global__ void __launch_bounds__(kFinalizeThreads)
    reduce_splits_max_kernel(const R* __restrict__ partial, std::size_t rows,
                             std::size_t splits, R* __restrict__ result)
{
    __shared__ R block_max[kFinalizeThreads];

    R thread_max = R(0);
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * kFinalizeThreads;
    for (std::size_t row =
             static_cast<std::size_t>(blockIdx.x) * kFinalizeThreads +
             threadIdx.x;
         row < rows; row += stride)
    {
        R row_sum = R(0);
        for (std::size_t s = 0; s < splits; ++s)
        {
            row_sum += partial[s * rows + row];
        }
        thread_max = nan_max(thread_max, row_sum);
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
void matrix_inf_norm_launch(const T* A, std::size_t rows, std::size_t cols,
                            std::size_t lda, R* d_result,
                            CHASE_GPU_STREAM_T stream)
{
    static_cast<void>(CHASE_GPU_MEMSET_ASYNC(d_result, 0, sizeof(R), stream));
    if (rows == 0 || cols == 0)
    {
        return;
    }

    const std::size_t sms = multiprocessor_count();
    const std::size_t row_tiles = (rows + kTileRows - 1) / kTileRows;
    const std::size_t target_blocks = sms * kTargetBlocksPerSM;

    std::size_t splits = (target_blocks + row_tiles - 1) / row_tiles;
    splits = std::min(splits, (cols + kMinColsPerSplit - 1) / kMinColsPerSplit);
    splits = std::min(splits, kMaxSplits);
    splits = std::max<std::size_t>(splits, 1);

    R* partial = nullptr;
    if (splits > 1 &&
        CHASE_GPU_MALLOC_ASYNC(reinterpret_cast<void**>(&partial),
                               splits * rows * sizeof(R),
                               stream) != CHASE_GPU_SUCCESS)
    {
        static_cast<void>(CHASE_GPU_GET_LAST_ERROR());
        partial = nullptr;
        splits = 1;
    }

    const dim3 block(kTileRows, kColLanes);

    if (splits == 1)
    {
        const dim3 grid(static_cast<unsigned>(row_tiles), 1);
        row_abs_sum_kernel<true><<<grid, block, 0, stream>>>(
            A, rows, cols, lda, cols, static_cast<R*>(nullptr), d_result);
        return;
    }

    const std::size_t cols_per_split = (cols + splits - 1) / splits;
    const dim3 grid(static_cast<unsigned>(row_tiles),
                    static_cast<unsigned>(splits));
    row_abs_sum_kernel<false><<<grid, block, 0, stream>>>(
        A, rows, cols, lda, cols_per_split, partial, static_cast<R*>(nullptr));

    const std::size_t finalize_blocks =
        std::min((rows + kFinalizeThreads - 1) / kFinalizeThreads, sms * 4);
    reduce_splits_max_kernel<<<static_cast<unsigned>(finalize_blocks),
                               kFinalizeThreads, 0, stream>>>(
        partial, rows, splits, d_result);

    static_cast<void>(CHASE_GPU_FREE_ASYNC(partial, stream));
}

// Local abs row-sums only (vector of length `rows`). Used by distributed
// ||A||_∞ (Allreduce SUM on row_sums, then max).
template <typename T, typename R>
void matrix_row_abs_sums_launch(const T* A, std::size_t rows, std::size_t cols,
                                std::size_t lda, R* d_row_sums,
                                CHASE_GPU_STREAM_T stream)
{
    if (rows == 0)
    {
        return;
    }
    static_cast<void>(
        CHASE_GPU_MEMSET_ASYNC(d_row_sums, 0, rows * sizeof(R), stream));
    if (cols == 0)
    {
        return;
    }

    const std::size_t sms = multiprocessor_count();
    const std::size_t row_tiles = (rows + kTileRows - 1) / kTileRows;
    const std::size_t target_blocks = sms * kTargetBlocksPerSM;

    std::size_t splits = (target_blocks + row_tiles - 1) / row_tiles;
    splits = std::min(splits, (cols + kMinColsPerSplit - 1) / kMinColsPerSplit);
    splits = std::min(splits, kMaxSplits);
    splits = std::max<std::size_t>(splits, 1);

    const dim3 block(kTileRows, kColLanes);

    if (splits == 1)
    {
        const dim3 grid(static_cast<unsigned>(row_tiles), 1);
        row_abs_sum_kernel<false><<<grid, block, 0, stream>>>(
            A, rows, cols, lda, cols, d_row_sums, static_cast<R*>(nullptr));
        return;
    }

    R* partial = nullptr;
    if (CHASE_GPU_MALLOC_ASYNC(reinterpret_cast<void**>(&partial),
                               splits * rows * sizeof(R),
                               stream) != CHASE_GPU_SUCCESS)
    {
        static_cast<void>(CHASE_GPU_GET_LAST_ERROR());
        const dim3 grid(static_cast<unsigned>(row_tiles), 1);
        row_abs_sum_kernel<false><<<grid, block, 0, stream>>>(
            A, rows, cols, lda, cols, d_row_sums, static_cast<R*>(nullptr));
        return;
    }

    const std::size_t cols_per_split = (cols + splits - 1) / splits;
    const dim3 grid(static_cast<unsigned>(row_tiles),
                    static_cast<unsigned>(splits));
    row_abs_sum_kernel<false><<<grid, block, 0, stream>>>(
        A, rows, cols, lda, cols_per_split, partial, static_cast<R*>(nullptr));

    const std::size_t finalize_blocks =
        std::min((rows + kFinalizeThreads - 1) / kFinalizeThreads, sms * 4);
    reduce_splits_sum_kernel<<<static_cast<unsigned>(finalize_blocks),
                              kFinalizeThreads, 0, stream>>>(
        partial, rows, splits, d_row_sums);

    static_cast<void>(CHASE_GPU_FREE_ASYNC(partial, stream));
}

template <typename R>
void matrix_row_sums_max_launch(const R* d_row_sums, std::size_t rows,
                                R* d_result, CHASE_GPU_STREAM_T stream)
{
    static_cast<void>(CHASE_GPU_MEMSET_ASYNC(d_result, 0, sizeof(R), stream));
    if (rows == 0)
    {
        return;
    }
    const std::size_t sms = multiprocessor_count();
    const std::size_t finalize_blocks =
        std::min((rows + kFinalizeThreads - 1) / kFinalizeThreads, sms * 4);
    reduce_splits_max_kernel<<<static_cast<unsigned>(finalize_blocks),
                               kFinalizeThreads, 0, stream>>>(
        d_row_sums, rows, /*splits=*/1, d_result);
}

} // namespace matrix_inf_norm_detail

void matrixInfNorm_gpu(const float* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_inf_norm_launch(d_matrix, rows, cols, ld,
                                                   d_norm, stream_);
}

void matrixInfNorm_gpu(const double* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_inf_norm_launch(d_matrix, rows, cols, ld,
                                                   d_norm, stream_);
}

void matrixInfNorm_gpu(const std::complex<float>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, float* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_inf_norm_launch(
        reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_matrix), rows, cols,
        ld, d_norm, stream_);
}

void matrixInfNorm_gpu(const std::complex<double>* d_matrix, std::size_t rows,
                       std::size_t cols, std::size_t ld, double* d_norm,
                       CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_inf_norm_launch(
        reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_matrix), rows, cols,
        ld, d_norm, stream_);
}

void matrixRowAbsSums_gpu(const float* d_matrix, std::size_t rows,
                          std::size_t cols, std::size_t ld, float* d_row_sums,
                          CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_row_abs_sums_launch(
        d_matrix, rows, cols, ld, d_row_sums, stream_);
}

void matrixRowAbsSums_gpu(const double* d_matrix, std::size_t rows,
                          std::size_t cols, std::size_t ld, double* d_row_sums,
                          CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_row_abs_sums_launch(
        d_matrix, rows, cols, ld, d_row_sums, stream_);
}

void matrixRowAbsSums_gpu(const std::complex<float>* d_matrix, std::size_t rows,
                          std::size_t cols, std::size_t ld, float* d_row_sums,
                          CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_row_abs_sums_launch(
        reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_matrix), rows, cols,
        ld, d_row_sums, stream_);
}

void matrixRowAbsSums_gpu(const std::complex<double>* d_matrix,
                          std::size_t rows, std::size_t cols, std::size_t ld,
                          double* d_row_sums, CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_row_abs_sums_launch(
        reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_matrix), rows, cols,
        ld, d_row_sums, stream_);
}

void matrixRowSumsMax_gpu(const float* d_row_sums, std::size_t rows,
                          float* d_result, CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_row_sums_max_launch(d_row_sums, rows,
                                                       d_result, stream_);
}

void matrixRowSumsMax_gpu(const double* d_row_sums, std::size_t rows,
                          double* d_result, CHASE_GPU_STREAM_T stream_)
{
    matrix_inf_norm_detail::matrix_row_sums_max_launch(d_row_sums, rows,
                                                       d_result, stream_);
}

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase
