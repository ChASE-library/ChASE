// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "lanczos_kernels.cuh"

#define LANCZOS_BLOCK_SIZE 256

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

// ============================================================================
// Kernel Implementations
// ============================================================================

/**
 * @brief Batched scaling: v[:,i] = scale[i] * v[:,i] for all i
 * Generic template for real types
 */
template<typename T>
__global__ void batched_scale_kernel(
    const T* __restrict__ scale,  // Scalars on GPU [numvec]
    T* __restrict__ v,             // Input/output vectors [rows × numvec]
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        T s = scale[vec_idx];
        int idx = row_idx + vec_idx * ld;
        v[idx] *= s;
    }
}

/**
 * @brief Batched scaling: Specialization for cuDoubleComplex
 */
template<>
__global__ void batched_scale_kernel<cuDoubleComplex>(
    const cuDoubleComplex* __restrict__ scale,
    cuDoubleComplex* __restrict__ v,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        cuDoubleComplex s = scale[vec_idx];
        int idx = row_idx + vec_idx * ld;
        v[idx] = cuCmul(s, v[idx]);
    }
}

/**
 * @brief Batched scaling: Specialization for cuComplex
 */
template<>
__global__ void batched_scale_kernel<cuComplex>(
    const cuComplex* __restrict__ scale,
    cuComplex* __restrict__ v,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        cuComplex s = scale[vec_idx];
        int idx = row_idx + vec_idx * ld;
        v[idx] = cuCmulf(s, v[idx]);
    }
}

/**
 * @brief Batched AXPY: y[:,i] = y[:,i] + alpha[i] * x[:,i] for all i
 * Generic template for real types
 */
template<typename T>
__global__ void batched_axpy_kernel(
    const T* __restrict__ alpha,  // Scalars on GPU [numvec]
    const T* __restrict__ x,       // Input vectors [rows × numvec]
    T* __restrict__ y,             // Input/output vectors [rows × numvec]
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        T scale = alpha[vec_idx];
        int idx = row_idx + vec_idx * ld;
        y[idx] = y[idx] + scale * x[idx];
    }
}

/**
 * @brief Batched AXPY: Specialization for cuDoubleComplex
 */
template<>
__global__ void batched_axpy_kernel<cuDoubleComplex>(
    const cuDoubleComplex* __restrict__ alpha,
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ y,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        cuDoubleComplex scale = alpha[vec_idx];
        int idx = row_idx + vec_idx * ld;
        // y[idx] = y[idx] + scale * x[idx]
        y[idx] = cuCadd(y[idx], cuCmul(scale, x[idx]));
    }
}

/**
 * @brief Batched AXPY: Specialization for cuComplex
 */
template<>
__global__ void batched_axpy_kernel<cuComplex>(
    const cuComplex* __restrict__ alpha,
    const cuComplex* __restrict__ x,
    cuComplex* __restrict__ y,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        cuComplex scale = alpha[vec_idx];
        int idx = row_idx + vec_idx * ld;
        // y[idx] = y[idx] + scale * x[idx]
        y[idx] = cuCaddf(y[idx], cuCmulf(scale, x[idx]));
    }
}

template<typename T>
__global__ void square_inplace_kernel(T* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= data[i];
    }
}

template<typename T>
__global__ void negate_kernel(T* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = -data[i];
    }
}

// Specialization for cuComplex
template<>
__global__ void negate_kernel<cuComplex>(cuComplex* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = make_cuComplex(-cuCrealf(data[i]), -cuCimagf(data[i]));
    }
}

// Specialization for cuDoubleComplex
template<>
__global__ void negate_kernel<cuDoubleComplex>(cuDoubleComplex* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = make_cuDoubleComplex(-cuCreal(data[i]), -cuCimag(data[i]));
    }
}

template<typename T>
__global__ void sqrt_inplace_kernel(T* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = sqrt(data[i]);
    }
}

/**
 * @brief Batched sqrt: data[i] = sqrt(data[i]) for all i in parallel
 */
template<typename T>
__global__ void batched_sqrt_kernel(T* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = sqrt(data[i]);
    }
}

// Explicit kernels for all types (no generic template to avoid CUDA issues)

__global__ void normalize_vectors_double_kernel(
    double* vectors,
    const double* norms_squared,
    int rows,
    int numvec,
    int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        double norm_sq = norms_squared[vec_idx];
        double safe_norm_sq = fmax(norm_sq, 1e-100);
        double scale = rsqrt(safe_norm_sq);
        vectors[row_idx + vec_idx * ld] *= scale;
    }
}

__global__ void normalize_vectors_float_kernel(
    float* vectors,
    const float* norms_squared,
    int rows,
    int numvec,
    int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        float norm_sq = norms_squared[vec_idx];
        float safe_norm_sq = fmaxf(norm_sq, 1e-30f);
        float scale = rsqrtf(safe_norm_sq);
        vectors[row_idx + vec_idx * ld] *= scale;
    }
}

__global__ void normalize_vectors_complex_double_kernel(
    cuDoubleComplex* vectors,
    const double* norms_squared,
    int rows,
    int numvec,
    int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        double norm_sq = norms_squared[vec_idx];
        // Add epsilon to prevent division by zero or numerical instability
        double scale = rsqrt(fmax(norm_sq, 1e-100));
        int idx = row_idx + vec_idx * ld;
        vectors[idx].x *= scale;
        vectors[idx].y *= scale;
    }
}

__global__ void normalize_vectors_complex_float_kernel(
    cuComplex* vectors,
    const float* norms_squared,
    int rows,
    int numvec,
    int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vec_idx < numvec && row_idx < rows) {
        float norm_sq = norms_squared[vec_idx];
        // Protect against division by zero and numerical instability
        // For float: machine epsilon ~1.19e-7, use conservative threshold
        // This prevents rsqrtf from producing Inf/NaN on zero or tiny values
        float safe_norm_sq = fmaxf(norm_sq, 1e-30f);
        float scale = rsqrtf(safe_norm_sq);
        int idx = row_idx + vec_idx * ld;
        vectors[idx].x *= scale;
        vectors[idx].y *= scale;
    }
}

template<typename T>
__global__ void batched_dot_product_kernel(
    const T* v1,
    const T* v2,
    T* results,
    int rows,
    int numvec,
    int ld,
    bool negate)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    const T* v1_vec = v1 + vec_idx * ld;
    const T* v2_vec = v2 + vec_idx * ld;
    
    // Parallel reduction for dot product
    T sum = T(0);
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum += v1_vec[i] * v2_vec[i];
    }
    
    // Store in shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (threadIdx.x == 0) {
        results[vec_idx] = negate ? -sdata[0] : sdata[0];
    }
}

// Specialization for complex double
__global__ void batched_dot_product_complex_double_kernel(
    const cuDoubleComplex* v1,
    const cuDoubleComplex* v2,
    cuDoubleComplex* results,
    int rows,
    int numvec,
    int ld,
    bool negate)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    
    extern __shared__ double shared_mem_double[];
    double* sdata_real = shared_mem_double;
    double* sdata_imag = &shared_mem_double[blockDim.x];
    
    const cuDoubleComplex* v1_vec = v1 + vec_idx * ld;
    const cuDoubleComplex* v2_vec = v2 + vec_idx * ld;
    
    double sum_real = 0.0;
    double sum_imag = 0.0;
    
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        // conj(v1) * v2 = (a - bi)(c + di) = (ac + bd) + (ad - bc)i
        sum_real += v1_vec[i].x * v2_vec[i].x + v1_vec[i].y * v2_vec[i].y;
        sum_imag += v1_vec[i].x * v2_vec[i].y - v1_vec[i].y * v2_vec[i].x;
    }
    
    sdata_real[threadIdx.x] = sum_real;
    sdata_imag[threadIdx.x] = sum_imag;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_real[threadIdx.x] += sdata_real[threadIdx.x + s];
            sdata_imag[threadIdx.x] += sdata_imag[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        if (negate) {
            results[vec_idx].x = -sdata_real[0];
            results[vec_idx].y = -sdata_imag[0];
        } else {
            results[vec_idx].x = sdata_real[0];
            results[vec_idx].y = sdata_imag[0];
        }
    }
}

// Specialization for complex float
__global__ void batched_dot_product_complex_float_kernel(
    const cuComplex* v1,
    const cuComplex* v2,
    cuComplex* results,
    int rows,
    int numvec,
    int ld,
    bool negate)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    
    extern __shared__ float shared_mem_float[];
    float* sdata_real = shared_mem_float;
    float* sdata_imag = &shared_mem_float[blockDim.x];
    
    const cuComplex* v1_vec = v1 + vec_idx * ld;
    const cuComplex* v2_vec = v2 + vec_idx * ld;
    
    float sum_real = 0.0f;
    float sum_imag = 0.0f;
    
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_real += v1_vec[i].x * v2_vec[i].x + v1_vec[i].y * v2_vec[i].y;
        sum_imag += v1_vec[i].x * v2_vec[i].y - v1_vec[i].y * v2_vec[i].x;
    }
    
    sdata_real[threadIdx.x] = sum_real;
    sdata_imag[threadIdx.x] = sum_imag;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata_real[threadIdx.x] += sdata_real[threadIdx.x + s];
            sdata_imag[threadIdx.x] += sdata_imag[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        if (negate) {
            results[vec_idx].x = -sdata_real[0];
            results[vec_idx].y = -sdata_imag[0];
        } else {
            results[vec_idx].x = sdata_real[0];
            results[vec_idx].y = sdata_imag[0];
        }
    }
}

template<typename T, typename RealT>
__global__ void batched_norm_squared_kernel(
    const T* v,
    RealT* norms_squared,
    int rows,
    int numvec,
    int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    
    extern __shared__ char shared_mem[];
    RealT* sdata = reinterpret_cast<RealT*>(shared_mem);
    
    const T* v_vec = v + vec_idx * ld;
    
    RealT sum = RealT(0);
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum += v_vec[i] * v_vec[i];
    }
    
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        norms_squared[vec_idx] = sdata[0];
    }
}

// Specialization for complex types
__global__ void batched_norm_squared_complex_double_kernel(
    const cuDoubleComplex* v,
    double* norms_squared,
    int rows,
    int numvec,
    int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    
    extern __shared__ double shared_norm_double[];
    
    const cuDoubleComplex* v_vec = v + vec_idx * ld;
    
    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum += v_vec[i].x * v_vec[i].x + v_vec[i].y * v_vec[i].y;
    }
    
    shared_norm_double[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_norm_double[threadIdx.x] += shared_norm_double[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        norms_squared[vec_idx] = shared_norm_double[0];
    }
}

__global__ void batched_norm_squared_complex_float_kernel(
    const cuComplex* v,
    float* norms_squared,
    int rows,
    int numvec,
    int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    
    extern __shared__ float shared_norm_float[];
    
    const cuComplex* v_vec = v + vec_idx * ld;
    
    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum += v_vec[i].x * v_vec[i].x + v_vec[i].y * v_vec[i].y;
    }
    
    shared_norm_float[threadIdx.x] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_norm_float[threadIdx.x] += shared_norm_float[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        norms_squared[vec_idx] = shared_norm_float[0];
    }
}

// ============================================================================
// Host Wrapper Functions
// ============================================================================

void batched_scale_gpu(const cuDoubleComplex* scale, cuDoubleComplex* v,
                      int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_kernel<<<blocks, threads, 0, stream>>>(scale, v, rows, numvec, ld);
}

void batched_scale_gpu(const cuComplex* scale, cuComplex* v,
                      int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_kernel<<<blocks, threads, 0, stream>>>(scale, v, rows, numvec, ld);
}

void batched_scale_gpu(const double* scale, double* v,
                      int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_kernel<<<blocks, threads, 0, stream>>>(scale, v, rows, numvec, ld);
}

void batched_scale_gpu(const float* scale, float* v,
                      int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_kernel<<<blocks, threads, 0, stream>>>(scale, v, rows, numvec, ld);
}

void batched_axpy_gpu(const cuDoubleComplex* alpha, const cuDoubleComplex* x,
                     cuDoubleComplex* y, int rows, int numvec, int ld,
                     cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_axpy_kernel<<<blocks, threads, 0, stream>>>(alpha, x, y, rows, numvec, ld);
}

void batched_axpy_gpu(const cuComplex* alpha, const cuComplex* x,
                     cuComplex* y, int rows, int numvec, int ld,
                     cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_axpy_kernel<<<blocks, threads, 0, stream>>>(alpha, x, y, rows, numvec, ld);
}

void batched_axpy_gpu(const double* alpha, const double* x,
                     double* y, int rows, int numvec, int ld,
                     cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_axpy_kernel<<<blocks, threads, 0, stream>>>(alpha, x, y, rows, numvec, ld);
}

void batched_axpy_gpu(const float* alpha, const float* x,
                     float* y, int rows, int numvec, int ld,
                     cudaStream_t stream)
{
    dim3 threads(256);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_axpy_kernel<<<blocks, threads, 0, stream>>>(alpha, x, y, rows, numvec, ld);
}

void square_inplace_gpu(double* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    square_inplace_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void square_inplace_gpu(float* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    square_inplace_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void negate_gpu(double* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    negate_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void negate_gpu(float* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    negate_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void negate_gpu(cuDoubleComplex* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    negate_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void negate_gpu(cuComplex* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    negate_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void sqrt_inplace_gpu(double* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    sqrt_inplace_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void sqrt_inplace_gpu(float* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    sqrt_inplace_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void batched_sqrt_gpu(double* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    batched_sqrt_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void batched_sqrt_gpu(float* data, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    batched_sqrt_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

void normalize_vectors_gpu(cuDoubleComplex* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    normalize_vectors_complex_double_kernel<<<blocks, threads, 0, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void normalize_vectors_gpu(cuComplex* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    normalize_vectors_complex_float_kernel<<<blocks, threads, 0, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void normalize_vectors_gpu(double* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    normalize_vectors_double_kernel<<<blocks, threads, 0, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void normalize_vectors_gpu(float* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    normalize_vectors_float_kernel<<<blocks, threads, 0, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void batched_dot_product_gpu(const cuDoubleComplex* v1, const cuDoubleComplex* v2,
                             cuDoubleComplex* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double) * 2;  // real + imag
    batched_dot_product_complex_double_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, results, rows, numvec, ld, negate);
}

void batched_dot_product_gpu(const cuComplex* v1, const cuComplex* v2,
                             cuComplex* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float) * 2;
    batched_dot_product_complex_float_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, results, rows, numvec, ld, negate);
}

void batched_dot_product_gpu(const double* v1, const double* v2,
                             double* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double);
    batched_dot_product_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, results, rows, numvec, ld, negate);
}

void batched_dot_product_gpu(const float* v1, const float* v2,
                             float* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float);
    batched_dot_product_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, results, rows, numvec, ld, negate);
}

void batched_norm_squared_gpu(const cuDoubleComplex* v, double* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double);
    batched_norm_squared_complex_double_kernel<<<blocks, threads, shared_mem, stream>>>(
        v, norms_squared, rows, numvec, ld);
}

void batched_norm_squared_gpu(const cuComplex* v, float* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float);
    batched_norm_squared_complex_float_kernel<<<blocks, threads, shared_mem, stream>>>(
        v, norms_squared, rows, numvec, ld);
}

void batched_norm_squared_gpu(const double* v, double* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double);
    batched_norm_squared_kernel<<<blocks, threads, shared_mem, stream>>>(
        v, norms_squared, rows, numvec, ld);
}

void batched_norm_squared_gpu(const float* v, float* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float);
    batched_norm_squared_kernel<<<blocks, threads, shared_mem, stream>>>(
        v, norms_squared, rows, numvec, ld);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
