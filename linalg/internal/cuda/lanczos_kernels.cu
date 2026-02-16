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
//
// Naming convention:
//   - Device kernels: <operation>_kernel (template) or <operation>_<type>_kernel
//     (e.g. batched_scale_kernel, real_part_double_kernel).
//   - Host launchers: <operation>_gpu in .cu/.cuh; C++ wrappers in .hpp use
//     CamelCase (e.g. batchedSqrt, getRealPart).
//   - Pseudo-Hermitian (PH) kernels: ph_<operation>_<type>_kernel and
//     ph_<operation>_gpu (e.g. ph_fused_dot_scale_negate_axpy_double_kernel,
//     ph_lacpy_flip_batched_dot_gpu). Single/batched init: pseudo_hermitian_init_*
//
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

/**
 * @brief Batched AXPY then negate: y[:,i] += alpha[i]*x[:,i], then alpha[i] = -alpha[i]
 * One block per vector so we can negate alpha after axpy without race (alpha cached in register).
 */
template<typename T>
__global__ void batched_axpy_then_negate_kernel(
    T* __restrict__ alpha,
    const T* __restrict__ x,
    T* __restrict__ y,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    T scale = alpha[vec_idx];
    const T* x_vec = x + vec_idx * ld;
    T* y_vec = y + vec_idx * ld;
    for (int row_idx = threadIdx.x; row_idx < rows; row_idx += blockDim.x) {
        y_vec[row_idx] = y_vec[row_idx] + scale * x_vec[row_idx];
    }
    __syncthreads();
    if (threadIdx.x == 0)
        alpha[vec_idx] = -scale;
}

template<>
__global__ void batched_axpy_then_negate_kernel<cuDoubleComplex>(
    cuDoubleComplex* __restrict__ alpha,
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ y,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    cuDoubleComplex scale = alpha[vec_idx];
    const cuDoubleComplex* x_vec = x + vec_idx * ld;
    cuDoubleComplex* y_vec = y + vec_idx * ld;
    for (int row_idx = threadIdx.x; row_idx < rows; row_idx += blockDim.x) {
        y_vec[row_idx] = cuCadd(y_vec[row_idx], cuCmul(scale, x_vec[row_idx]));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        alpha[vec_idx] = make_cuDoubleComplex(-scale.x, -scale.y);
}

template<>
__global__ void batched_axpy_then_negate_kernel<cuComplex>(
    cuComplex* __restrict__ alpha,
    const cuComplex* __restrict__ x,
    cuComplex* __restrict__ y,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    cuComplex scale = alpha[vec_idx];
    const cuComplex* x_vec = x + vec_idx * ld;
    cuComplex* y_vec = y + vec_idx * ld;
    for (int row_idx = threadIdx.x; row_idx < rows; row_idx += blockDim.x) {
        y_vec[row_idx] = cuCaddf(y_vec[row_idx], cuCmulf(scale, x_vec[row_idx]));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        alpha[vec_idx] = make_cuComplex(-cuCrealf(scale), -cuCimagf(scale));
}

/**
 * @brief Copy real array to T with negate: out[i] = T(-in[i]). For complex T: out[i].x = -in[i], .y = 0.
 */
__global__ void copy_real_negate_to_T_double_kernel(const double* in, double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

__global__ void copy_real_negate_to_T_float_kernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

__global__ void copy_real_negate_to_T_complex_double_kernel(const double* in, cuDoubleComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = -in[i];
        out[i].y = 0.0;
    }
}

__global__ void copy_real_negate_to_T_complex_float_kernel(const float* in, cuComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = -in[i];
        out[i].y = 0.0f;
    }
}

/**
 * @brief Extract real part of T into RealT: out[i] = real(in[i]). For real T this is copy; for complex, out[i] = in[i].x.
 */
__global__ void real_part_double_kernel(const double* in, double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void real_part_float_kernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void real_part_complex_double_kernel(const cuDoubleComplex* in, double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i].x;
}

__global__ void real_part_complex_float_kernel(const cuComplex* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i].x;
}

/**
 * @brief Copy real to T: out[i] = T(in[i]). For complex T: out[i].x = in[i], .y = 0.
 */
__global__ void copy_real_to_T_double_kernel(const double* in, double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void copy_real_to_T_float_kernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}

__global__ void copy_real_to_T_complex_double_kernel(const double* in, cuDoubleComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = in[i];
        out[i].y = 0.0;
    }
}

__global__ void copy_real_to_T_complex_float_kernel(const float* in, cuComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i].x = in[i];
        out[i].y = 0.0f;
    }
}

/**
 * @brief Real reciprocal: out[i] = 1/in[i] (with safe divisor).
 */
__global__ void real_reciprocal_double_kernel(const double* in, double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in[i];
        out[i] = (x != 0.0) ? (1.0 / x) : 1.0;
    }
}

__global__ void real_reciprocal_float_kernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = (x != 0.0f) ? (1.0f / x) : 1.0f;
    }
}

/**
 * @brief Copy real reciprocal to T: out[i] = T(1/in[i]). For complex: .x = 1/in[i], .y = 0.
 */
__global__ void copy_real_reciprocal_to_T_double_kernel(const double* in, double* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in[i];
        out[i] = (x != 0.0) ? (1.0 / x) : 1.0;
    }
}

__global__ void copy_real_reciprocal_to_T_float_kernel(const float* in, float* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = (x != 0.0f) ? (1.0f / x) : 1.0f;
    }
}

__global__ void copy_real_reciprocal_to_T_complex_double_kernel(const double* in, cuDoubleComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in[i];
        double r = (x != 0.0) ? (1.0 / x) : 1.0;
        out[i].x = r;
        out[i].y = 0.0;
    }
}

__global__ void copy_real_reciprocal_to_T_complex_float_kernel(const float* in, cuComplex* out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        float r = (x != 0.0f) ? (1.0f / x) : 1.0f;
        out[i].x = r;
        out[i].y = 0.0f;
    }
}

/**
 * @brief Element-wise sqrt in-place: data[i] = sqrt(data[i]).
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
// Fused kernel: norm_squared + normalize (one block per vector)
// ============================================================================

__global__ void fused_norm_squared_normalize_double_kernel(
    double* vectors, double* norms_squared,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ double shared_fns_d[];
    const double* v_vec = vectors + vec_idx * ld;

    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_vec[i] * v_vec[i];
    shared_fns_d[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_fns_d[threadIdx.x] += shared_fns_d[threadIdx.x + s];
        __syncthreads();
    }

    double norm_sq = shared_fns_d[0];
    if (threadIdx.x == 0) {
        norms_squared[vec_idx] = norm_sq;
    }
    double scale = rsqrt(fmax(norm_sq, 1e-100));
    __syncthreads();

    double* out_vec = vectors + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        out_vec[i] *= scale;
}

__global__ void fused_norm_squared_normalize_float_kernel(
    float* vectors, float* norms_squared,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ float shared_fns_f[];
    const float* v_vec = vectors + vec_idx * ld;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_vec[i] * v_vec[i];
    shared_fns_f[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_fns_f[threadIdx.x] += shared_fns_f[threadIdx.x + s];
        __syncthreads();
    }

    float norm_sq = shared_fns_f[0];
    if (threadIdx.x == 0)
        norms_squared[vec_idx] = norm_sq;
    float scale = rsqrtf(fmaxf(norm_sq, 1e-30f));
    __syncthreads();

    float* out_vec = vectors + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        out_vec[i] *= scale;
}

__global__ void fused_norm_squared_normalize_complex_double_kernel(
    cuDoubleComplex* vectors, double* norms_squared,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ double shared_fns_cd[];
    const cuDoubleComplex* v_vec = vectors + vec_idx * ld;

    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_vec[i].x * v_vec[i].x + v_vec[i].y * v_vec[i].y;
    shared_fns_cd[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_fns_cd[threadIdx.x] += shared_fns_cd[threadIdx.x + s];
        __syncthreads();
    }

    double norm_sq = shared_fns_cd[0];
    if (threadIdx.x == 0)
        norms_squared[vec_idx] = norm_sq;
    double scale = rsqrt(fmax(norm_sq, 1e-100));
    __syncthreads();

    cuDoubleComplex* out_vec = vectors + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        out_vec[i].x *= scale;
        out_vec[i].y *= scale;
    }
}

__global__ void fused_norm_squared_normalize_complex_float_kernel(
    cuComplex* vectors, float* norms_squared,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ float shared_fns_cf[];
    const cuComplex* v_vec = vectors + vec_idx * ld;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_vec[i].x * v_vec[i].x + v_vec[i].y * v_vec[i].y;
    shared_fns_cf[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_fns_cf[threadIdx.x] += shared_fns_cf[threadIdx.x + s];
        __syncthreads();
    }

    float norm_sq = shared_fns_cf[0];
    if (threadIdx.x == 0)
        norms_squared[vec_idx] = norm_sq;
    float scale = rsqrtf(fmaxf(norm_sq, 1e-30f));
    __syncthreads();

    cuComplex* out_vec = vectors + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        out_vec[i].x *= scale;
        out_vec[i].y *= scale;
    }
}

// ============================================================================
// Fused kernel: dot(conj(v1)·v2), alpha=-dot, y+=alpha*x, then alpha_out=dot
// ============================================================================

__global__ void fused_dot_axpy_negate_double_kernel(
    const double* v1, const double* v2, double* y, const double* x,
    double* alpha_out, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ double shared_fdan_d[];
    const double* v1_vec = v1 + vec_idx * ld;
    const double* v2_vec = v2 + vec_idx * ld;

    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v1_vec[i] * v2_vec[i];
    shared_fdan_d[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_fdan_d[threadIdx.x] += shared_fdan_d[threadIdx.x + s];
        __syncthreads();
    }
    double dot_val = shared_fdan_d[0];
    double alpha = -dot_val;
    if (threadIdx.x == 0)
        alpha_out[vec_idx] = alpha;
    __syncthreads();

    double* y_vec = y + vec_idx * ld;
    const double* x_vec = x + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        y_vec[i] += alpha * x_vec[i];

    if (threadIdx.x == 0)
        alpha_out[vec_idx] = dot_val;
}

__global__ void fused_dot_axpy_negate_float_kernel(
    const float* v1, const float* v2, float* y, const float* x,
    float* alpha_out, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ float shared_fdan_f[];
    const float* v1_vec = v1 + vec_idx * ld;
    const float* v2_vec = v2 + vec_idx * ld;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v1_vec[i] * v2_vec[i];
    shared_fdan_f[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_fdan_f[threadIdx.x] += shared_fdan_f[threadIdx.x + s];
        __syncthreads();
    }
    float dot_val = shared_fdan_f[0];
    float alpha = -dot_val;
    if (threadIdx.x == 0)
        alpha_out[vec_idx] = alpha;
    __syncthreads();

    float* y_vec = y + vec_idx * ld;
    const float* x_vec = x + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        y_vec[i] += alpha * x_vec[i];

    if (threadIdx.x == 0)
        alpha_out[vec_idx] = dot_val;
}

__global__ void fused_dot_axpy_negate_complex_double_kernel(
    const cuDoubleComplex* v1, const cuDoubleComplex* v2,
    cuDoubleComplex* y, const cuDoubleComplex* x,
    cuDoubleComplex* alpha_out, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ double shared_fdan_cd[];  // [0]=real, [blockDim.x]=imag
    double* sreal = shared_fdan_cd;
    double* simag = &shared_fdan_cd[blockDim.x];

    const cuDoubleComplex* v1_vec = v1 + vec_idx * ld;
    const cuDoubleComplex* v2_vec = v2 + vec_idx * ld;

    double sum_r = 0.0, sum_i = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v1_vec[i].x * v2_vec[i].x + v1_vec[i].y * v2_vec[i].y;
        sum_i += v1_vec[i].x * v2_vec[i].y - v1_vec[i].y * v2_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    double dr = sreal[0], di = simag[0];
    double ar = -dr, ai = -di;
    if (threadIdx.x == 0) {
        alpha_out[vec_idx].x = ar;
        alpha_out[vec_idx].y = ai;
    }
    __syncthreads();

    cuDoubleComplex* y_vec = y + vec_idx * ld;
    const cuDoubleComplex* x_vec = x + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        y_vec[i] = cuCadd(y_vec[i], make_cuDoubleComplex(ar * x_vec[i].x - ai * x_vec[i].y,
                                                        ar * x_vec[i].y + ai * x_vec[i].x));

    if (threadIdx.x == 0) {
        alpha_out[vec_idx].x = dr;
        alpha_out[vec_idx].y = di;
    }
}

__global__ void fused_dot_axpy_negate_complex_float_kernel(
    const cuComplex* v1, const cuComplex* v2,
    cuComplex* y, const cuComplex* x,
    cuComplex* alpha_out, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;

    extern __shared__ float shared_fdan_cf[];
    float* sreal = shared_fdan_cf;
    float* simag = &shared_fdan_cf[blockDim.x];

    const cuComplex* v1_vec = v1 + vec_idx * ld;
    const cuComplex* v2_vec = v2 + vec_idx * ld;

    float sum_r = 0.0f, sum_i = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v1_vec[i].x * v2_vec[i].x + v1_vec[i].y * v2_vec[i].y;
        sum_i += v1_vec[i].x * v2_vec[i].y - v1_vec[i].y * v2_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    float dr = sreal[0], di = simag[0];
    float ar = -dr, ai = -di;
    if (threadIdx.x == 0) {
        alpha_out[vec_idx].x = ar;
        alpha_out[vec_idx].y = ai;
    }
    __syncthreads();

    cuComplex* y_vec = y + vec_idx * ld;
    const cuComplex* x_vec = x + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        y_vec[i] = cuCaddf(y_vec[i], make_cuComplex(ar * x_vec[i].x - ai * x_vec[i].y,
                                                    ar * x_vec[i].y + ai * x_vec[i].x));

    if (threadIdx.x == 0) {
        alpha_out[vec_idx].x = dr;
        alpha_out[vec_idx].y = di;
    }
}

// ============================================================================
// Batched scale two: v1[:,i] *= scale[i], v2[:,i] *= scale[i]
// ============================================================================

__global__ void batched_scale_two_double_kernel(
    const double* scale, double* v1, double* v2,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < numvec && row_idx < rows) {
        double s = scale[vec_idx];
        int i = row_idx + vec_idx * ld;
        v1[i] *= s;
        v2[i] *= s;
    }
}

__global__ void batched_scale_two_float_kernel(
    const float* scale, float* v1, float* v2,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < numvec && row_idx < rows) {
        float s = scale[vec_idx];
        int i = row_idx + vec_idx * ld;
        v1[i] *= s;
        v2[i] *= s;
    }
}

__global__ void batched_scale_two_complex_double_kernel(
    const cuDoubleComplex* scale, cuDoubleComplex* v1, cuDoubleComplex* v2,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < numvec && row_idx < rows) {
        cuDoubleComplex s = scale[vec_idx];
        int i = row_idx + vec_idx * ld;
        v1[i] = cuCmul(s, v1[i]);
        v2[i] = cuCmul(s, v2[i]);
    }
}

__global__ void batched_scale_two_complex_float_kernel(
    const cuComplex* scale, cuComplex* v1, cuComplex* v2,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.y;
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx < numvec && row_idx < rows) {
        cuComplex s = scale[vec_idx];
        int i = row_idx + vec_idx * ld;
        v1[i] = cuCmulf(s, v1[i]);
        v2[i] = cuCmulf(s, v2[i]);
    }
}

// ============================================================================
// Pseudo-Hermitian: alpha[i] = -real(alpha[i]) * real_scale[i] (in-place)
// ============================================================================

__global__ void scale_complex_by_real_negate_double_kernel(
    cuDoubleComplex* alpha, const double* real_scale, int numvec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numvec) {
        double r = alpha[i].x;
        alpha[i].x = -r * real_scale[i];
        alpha[i].y = 0.0;
    }
}

__global__ void scale_complex_by_real_negate_float_kernel(
    cuComplex* alpha, const float* real_scale, int numvec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numvec) {
        float r = alpha[i].x;
        alpha[i].x = -r * real_scale[i];
        alpha[i].y = 0.0f;
    }
}

__global__ void scale_real_by_real_negate_double_kernel(
    double* alpha, const double* real_scale, int numvec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numvec)
        alpha[i] = -alpha[i] * real_scale[i];
}

__global__ void scale_real_by_real_negate_float_kernel(
    float* alpha, const float* real_scale, int numvec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numvec)
        alpha[i] = -alpha[i] * real_scale[i];
}

// ============================================================================
// Pseudo-Hermitian single-vector init: copy+flip, dot, scale=1/sqrt(real(dot)), scale v1/v2
// ============================================================================

__global__ void pseudo_hermitian_init_single_double_kernel(
    double* v_1, double* v_2, double* Sv,
    double* d_beta, double* d_real_beta_prev, int rows, int ld)
{
    int half_m = rows / 2;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        double val = v_2[i];
        Sv[i] = (i >= half_m) ? -val : val;
    }
    __syncthreads();

    extern __shared__ double sh_phis_d[];
    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_1[i] * Sv[i];
    sh_phis_d[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_phis_d[threadIdx.x] += sh_phis_d[threadIdx.x + s];
        __syncthreads();
    }
    double real_dot = sh_phis_d[0];
    double scale = 1.0 / sqrt(real_dot);
    if (threadIdx.x == 0) {
        *d_beta = scale;
        *d_real_beta_prev = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1[i] *= scale;
        v_2[i] *= scale;
    }
}

__global__ void pseudo_hermitian_init_single_float_kernel(
    float* v_1, float* v_2, float* Sv,
    float* d_beta, float* d_real_beta_prev, int rows, int ld)
{
    int half_m = rows / 2;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        float val = v_2[i];
        Sv[i] = (i >= half_m) ? -val : val;
    }
    __syncthreads();

    extern __shared__ float sh_phis_f[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_1[i] * Sv[i];
    sh_phis_f[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_phis_f[threadIdx.x] += sh_phis_f[threadIdx.x + s];
        __syncthreads();
    }
    float real_dot = sh_phis_f[0];
    float scale = 1.0f / sqrtf(real_dot);
    if (threadIdx.x == 0) {
        *d_beta = scale;
        *d_real_beta_prev = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1[i] *= scale;
        v_2[i] *= scale;
    }
}

__global__ void pseudo_hermitian_init_single_complex_double_kernel(
    cuDoubleComplex* v_1, cuDoubleComplex* v_2, cuDoubleComplex* Sv,
    cuDoubleComplex* d_beta, double* d_real_beta_prev, int rows, int ld)
{
    int half_m = rows / 2;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        cuDoubleComplex val = v_2[i];
        if (i >= half_m) {
            Sv[i].x = -val.x;
            Sv[i].y = -val.y;
        } else {
            Sv[i] = val;
        }
    }
    __syncthreads();

    extern __shared__ double sh_phis_cd[];  // [0..blockDim.x-1]=real, [blockDim.x..]=imag
    double* sreal = sh_phis_cd;
    double* simag = &sh_phis_cd[blockDim.x];
    double sum_r = 0.0, sum_i = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_1[i].x * Sv[i].x + v_1[i].y * Sv[i].y;
        sum_i += v_1[i].x * Sv[i].y - v_1[i].y * Sv[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    double real_dot = sreal[0];
    double scale = 1.0 / sqrt(real_dot);
    if (threadIdx.x == 0) {
        d_beta->x = scale;
        d_beta->y = 0.0;
        *d_real_beta_prev = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1[i].x *= scale;
        v_1[i].y *= scale;
        v_2[i].x *= scale;
        v_2[i].y *= scale;
    }
}

__global__ void pseudo_hermitian_init_single_complex_float_kernel(
    cuComplex* v_1, cuComplex* v_2, cuComplex* Sv,
    cuComplex* d_beta, float* d_real_beta_prev, int rows, int ld)
{
    int half_m = rows / 2;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        cuComplex val = v_2[i];
        if (i >= half_m) {
            Sv[i].x = -val.x;
            Sv[i].y = -val.y;
        } else {
            Sv[i] = val;
        }
    }
    __syncthreads();

    extern __shared__ float sh_phis_cf[];
    float* sreal = sh_phis_cf;
    float* simag = &sh_phis_cf[blockDim.x];
    float sum_r = 0.0f, sum_i = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_1[i].x * Sv[i].x + v_1[i].y * Sv[i].y;
        sum_i += v_1[i].x * Sv[i].y - v_1[i].y * Sv[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    float real_dot = sreal[0];
    float scale = 1.0f / sqrtf(real_dot);
    if (threadIdx.x == 0) {
        d_beta->x = scale;
        d_beta->y = 0.0f;
        *d_real_beta_prev = scale;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1[i].x *= scale;
        v_1[i].y *= scale;
        v_2[i].x *= scale;
        v_2[i].y *= scale;
    }
}

// ============================================================================
// Pseudo-Hermitian batched fused init: lacpyFlip + dot + scale from dot + scale v1,v2 (one block per column)
// ============================================================================

__global__ void pseudo_hermitian_init_batched_double_kernel(
    double* v_1, double* v_2, double* Sv,
    double* d_beta, double* d_real_beta_prev, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    double* v_2_vec = v_2 + vec_idx * ld;
    double* Sv_vec = Sv + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        double val = v_2_vec[i];
        Sv_vec[i] = (i >= half_m) ? -val : val;
    }
    __syncthreads();
    extern __shared__ double sh_phib_d[];
    double* v_1_vec = v_1 + vec_idx * ld;
    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_1_vec[i] * Sv_vec[i];
    sh_phib_d[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_phib_d[threadIdx.x] += sh_phib_d[threadIdx.x + s];
        __syncthreads();
    }
    double scale = 1.0 / sqrt(sh_phib_d[0]);
    if (threadIdx.x == 0) {
        d_beta[vec_idx] = scale;
        d_real_beta_prev[vec_idx] = scale;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1_vec[i] *= scale;
        v_2_vec[i] *= scale;
    }
}

__global__ void pseudo_hermitian_init_batched_float_kernel(
    float* v_1, float* v_2, float* Sv,
    float* d_beta, float* d_real_beta_prev, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    float* v_2_vec = v_2 + vec_idx * ld;
    float* Sv_vec = Sv + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        float val = v_2_vec[i];
        Sv_vec[i] = (i >= half_m) ? -val : val;
    }
    __syncthreads();
    extern __shared__ float sh_phib_f[];
    float* v_1_vec = v_1 + vec_idx * ld;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_1_vec[i] * Sv_vec[i];
    sh_phib_f[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_phib_f[threadIdx.x] += sh_phib_f[threadIdx.x + s];
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(sh_phib_f[0]);
    if (threadIdx.x == 0) {
        d_beta[vec_idx] = scale;
        d_real_beta_prev[vec_idx] = scale;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1_vec[i] *= scale;
        v_2_vec[i] *= scale;
    }
}

__global__ void pseudo_hermitian_init_batched_complex_double_kernel(
    cuDoubleComplex* v_1, cuDoubleComplex* v_2, cuDoubleComplex* Sv,
    cuDoubleComplex* d_beta, double* d_real_beta_prev, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    cuDoubleComplex* v_2_vec = v_2 + vec_idx * ld;
    cuDoubleComplex* Sv_vec = Sv + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        cuDoubleComplex val = v_2_vec[i];
        if (i >= half_m) {
            Sv_vec[i].x = -val.x;
            Sv_vec[i].y = -val.y;
        } else {
            Sv_vec[i] = val;
        }
    }
    __syncthreads();
    extern __shared__ double sh_phib_cd[];
    double* sreal = sh_phib_cd;
    double* simag = &sh_phib_cd[blockDim.x];
    cuDoubleComplex* v_1_vec = v_1 + vec_idx * ld;
    double sum_r = 0.0, sum_i = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_1_vec[i].x * Sv_vec[i].x + v_1_vec[i].y * Sv_vec[i].y;
        sum_i += v_1_vec[i].x * Sv_vec[i].y - v_1_vec[i].y * Sv_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    double scale = 1.0 / sqrt(sreal[0]);
    if (threadIdx.x == 0) {
        d_beta[vec_idx].x = scale;
        d_beta[vec_idx].y = 0.0;
        d_real_beta_prev[vec_idx] = scale;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1_vec[i].x *= scale;
        v_1_vec[i].y *= scale;
        v_2_vec[i].x *= scale;
        v_2_vec[i].y *= scale;
    }
}

__global__ void pseudo_hermitian_init_batched_complex_float_kernel(
    cuComplex* v_1, cuComplex* v_2, cuComplex* Sv,
    cuComplex* d_beta, float* d_real_beta_prev, int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    cuComplex* v_2_vec = v_2 + vec_idx * ld;
    cuComplex* Sv_vec = Sv + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        cuComplex val = v_2_vec[i];
        if (i >= half_m) {
            Sv_vec[i].x = -val.x;
            Sv_vec[i].y = -val.y;
        } else {
            Sv_vec[i] = val;
        }
    }
    __syncthreads();
    extern __shared__ float sh_phib_cf[];
    float* sreal = sh_phib_cf;
    float* simag = &sh_phib_cf[blockDim.x];
    cuComplex* v_1_vec = v_1 + vec_idx * ld;
    float sum_r = 0.0f, sum_i = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_1_vec[i].x * Sv_vec[i].x + v_1_vec[i].y * Sv_vec[i].y;
        sum_i += v_1_vec[i].x * Sv_vec[i].y - v_1_vec[i].y * Sv_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    float scale = 1.0f / sqrtf(sreal[0]);
    if (threadIdx.x == 0) {
        d_beta[vec_idx].x = scale;
        d_beta[vec_idx].y = 0.0f;
        d_real_beta_prev[vec_idx] = scale;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        v_1_vec[i].x *= scale;
        v_1_vec[i].y *= scale;
        v_2_vec[i].x *= scale;
        v_2_vec[i].y *= scale;
    }
}

// ============================================================================
// Pseudo-Hermitian fused: dot(v_2,Sv)->alpha, alpha=-real(alpha)*real_beta_prev, v_2+=alpha*v_1
// ============================================================================

__global__ void ph_fused_dot_scale_negate_axpy_double_kernel(
    const double* v_2, const double* Sv, double* d_alpha, const double* v_1,
    double* v_2_out, const double* d_real_beta_prev,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    extern __shared__ double sh_fdsna_d[];
    const double* v_2_vec = v_2 + vec_idx * ld;
    const double* Sv_vec = Sv + vec_idx * ld;
    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_2_vec[i] * Sv_vec[i];
    sh_fdsna_d[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_fdsna_d[threadIdx.x] += sh_fdsna_d[threadIdx.x + s];
        __syncthreads();
    }
    double alpha_raw = sh_fdsna_d[0];
    double rbp = d_real_beta_prev[vec_idx];
    double alpha = -alpha_raw * rbp;
    if (threadIdx.x == 0)
        d_alpha[vec_idx] = alpha;
    __syncthreads();
    const double* v_1_vec = v_1 + vec_idx * ld;
    double* v_2_out_vec = v_2_out + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        v_2_out_vec[i] += alpha * v_1_vec[i];
}

__global__ void ph_fused_dot_scale_negate_axpy_float_kernel(
    const float* v_2, const float* Sv, float* d_alpha, const float* v_1,
    float* v_2_out, const float* d_real_beta_prev,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    extern __shared__ float sh_fdsna_f[];
    const float* v_2_vec = v_2 + vec_idx * ld;
    const float* Sv_vec = Sv + vec_idx * ld;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_2_vec[i] * Sv_vec[i];
    sh_fdsna_f[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_fdsna_f[threadIdx.x] += sh_fdsna_f[threadIdx.x + s];
        __syncthreads();
    }
    float alpha_raw = sh_fdsna_f[0];
    float rbp = d_real_beta_prev[vec_idx];
    float alpha = -alpha_raw * rbp;
    if (threadIdx.x == 0)
        d_alpha[vec_idx] = alpha;
    __syncthreads();
    const float* v_1_vec = v_1 + vec_idx * ld;
    float* v_2_out_vec = v_2_out + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        v_2_out_vec[i] += alpha * v_1_vec[i];
}

__global__ void ph_fused_dot_scale_negate_axpy_complex_double_kernel(
    const cuDoubleComplex* v_2, const cuDoubleComplex* Sv,
    cuDoubleComplex* d_alpha, const cuDoubleComplex* v_1,
    cuDoubleComplex* v_2_out, const double* d_real_beta_prev,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    extern __shared__ double sh_fdsna_cd[];
    double* sreal = sh_fdsna_cd;
    double* simag = &sh_fdsna_cd[blockDim.x];
    const cuDoubleComplex* v_2_vec = v_2 + vec_idx * ld;
    const cuDoubleComplex* Sv_vec = Sv + vec_idx * ld;
    double sum_r = 0.0, sum_i = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_2_vec[i].x * Sv_vec[i].x + v_2_vec[i].y * Sv_vec[i].y;
        sum_i += v_2_vec[i].x * Sv_vec[i].y - v_2_vec[i].y * Sv_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    double real_dot = sreal[0];
    double rbp = d_real_beta_prev[vec_idx];
    double alpha_r = -real_dot * rbp;
    if (threadIdx.x == 0) {
        d_alpha[vec_idx].x = alpha_r;
        d_alpha[vec_idx].y = 0.0;
    }
    __syncthreads();
    const cuDoubleComplex* v_1_vec = v_1 + vec_idx * ld;
    cuDoubleComplex* v_2_out_vec = v_2_out + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        v_2_out_vec[i] = cuCadd(v_2_out_vec[i],
                                 make_cuDoubleComplex(alpha_r * v_1_vec[i].x - 0.0 * v_1_vec[i].y,
                                                      alpha_r * v_1_vec[i].y + 0.0 * v_1_vec[i].x));
}

__global__ void ph_fused_dot_scale_negate_axpy_complex_float_kernel(
    const cuComplex* v_2, const cuComplex* Sv,
    cuComplex* d_alpha, const cuComplex* v_1,
    cuComplex* v_2_out, const float* d_real_beta_prev,
    int rows, int numvec, int ld)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    extern __shared__ float sh_fdsna_cf[];
    float* sreal = sh_fdsna_cf;
    float* simag = &sh_fdsna_cf[blockDim.x];
    const cuComplex* v_2_vec = v_2 + vec_idx * ld;
    const cuComplex* Sv_vec = Sv + vec_idx * ld;
    float sum_r = 0.0f, sum_i = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_2_vec[i].x * Sv_vec[i].x + v_2_vec[i].y * Sv_vec[i].y;
        sum_i += v_2_vec[i].x * Sv_vec[i].y - v_2_vec[i].y * Sv_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    float real_dot = sreal[0];
    float rbp = d_real_beta_prev[vec_idx];
    float alpha_r = -real_dot * rbp;
    if (threadIdx.x == 0) {
        d_alpha[vec_idx].x = alpha_r;
        d_alpha[vec_idx].y = 0.0f;
    }
    __syncthreads();
    const cuComplex* v_1_vec = v_1 + vec_idx * ld;
    cuComplex* v_2_out_vec = v_2_out + vec_idx * ld;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        v_2_out_vec[i] = cuCaddf(v_2_out_vec[i],
                                  make_cuComplex(alpha_r * v_1_vec[i].x, alpha_r * v_1_vec[i].y));
}

// ============================================================================
// Fused: lacpyFlipLowerHalf(v_2->Sv) + batchedDotProduct(v_1, Sv, d_beta)
// ============================================================================

__global__ void ph_lacpy_flip_batched_dot_double_kernel(
    const double* v_2, double* Sv, const double* v_1, double* d_beta,
    int rows, int numvec, int ld_v2, int ld_sv, int ld_v1)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    const double* v_2_vec = v_2 + vec_idx * ld_v2;
    double* Sv_vec = Sv + vec_idx * ld_sv;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        double val = v_2_vec[i];
        Sv_vec[i] = (i >= half_m) ? -val : val;
    }
    __syncthreads();
    const double* v_1_vec = v_1 + vec_idx * ld_v1;
    extern __shared__ double sh_lfbd_d[];
    double sum = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_1_vec[i] * Sv_vec[i];
    sh_lfbd_d[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_lfbd_d[threadIdx.x] += sh_lfbd_d[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_beta[vec_idx] = sh_lfbd_d[0];
}

__global__ void ph_lacpy_flip_batched_dot_float_kernel(
    const float* v_2, float* Sv, const float* v_1, float* d_beta,
    int rows, int numvec, int ld_v2, int ld_sv, int ld_v1)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    const float* v_2_vec = v_2 + vec_idx * ld_v2;
    float* Sv_vec = Sv + vec_idx * ld_sv;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        float val = v_2_vec[i];
        Sv_vec[i] = (i >= half_m) ? -val : val;
    }
    __syncthreads();
    const float* v_1_vec = v_1 + vec_idx * ld_v1;
    extern __shared__ float sh_lfbd_f[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x)
        sum += v_1_vec[i] * Sv_vec[i];
    sh_lfbd_f[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sh_lfbd_f[threadIdx.x] += sh_lfbd_f[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        d_beta[vec_idx] = sh_lfbd_f[0];
}

__global__ void ph_lacpy_flip_batched_dot_complex_double_kernel(
    const cuDoubleComplex* v_2, cuDoubleComplex* Sv,
    const cuDoubleComplex* v_1, cuDoubleComplex* d_beta,
    int rows, int numvec, int ld_v2, int ld_sv, int ld_v1)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    const cuDoubleComplex* v_2_vec = v_2 + vec_idx * ld_v2;
    cuDoubleComplex* Sv_vec = Sv + vec_idx * ld_sv;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        cuDoubleComplex val = v_2_vec[i];
        if (i >= half_m) {
            Sv_vec[i].x = -val.x;
            Sv_vec[i].y = -val.y;
        } else {
            Sv_vec[i] = val;
        }
    }
    __syncthreads();
    const cuDoubleComplex* v_1_vec = v_1 + vec_idx * ld_v1;
    extern __shared__ double sh_lfbd_cd[];
    double* sreal = sh_lfbd_cd;
    double* simag = &sh_lfbd_cd[blockDim.x];
    double sum_r = 0.0, sum_i = 0.0;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_1_vec[i].x * Sv_vec[i].x + v_1_vec[i].y * Sv_vec[i].y;
        sum_i += v_1_vec[i].x * Sv_vec[i].y - v_1_vec[i].y * Sv_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_beta[vec_idx].x = sreal[0];
        d_beta[vec_idx].y = simag[0];
    }
}

__global__ void ph_lacpy_flip_batched_dot_complex_float_kernel(
    const cuComplex* v_2, cuComplex* Sv,
    const cuComplex* v_1, cuComplex* d_beta,
    int rows, int numvec, int ld_v2, int ld_sv, int ld_v1)
{
    int vec_idx = blockIdx.x;
    if (vec_idx >= numvec) return;
    int half_m = rows / 2;
    const cuComplex* v_2_vec = v_2 + vec_idx * ld_v2;
    cuComplex* Sv_vec = Sv + vec_idx * ld_sv;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        cuComplex val = v_2_vec[i];
        if (i >= half_m) {
            Sv_vec[i].x = -val.x;
            Sv_vec[i].y = -val.y;
        } else {
            Sv_vec[i] = val;
        }
    }
    __syncthreads();
    const cuComplex* v_1_vec = v_1 + vec_idx * ld_v1;
    extern __shared__ float sh_lfbd_cf[];
    float* sreal = sh_lfbd_cf;
    float* simag = &sh_lfbd_cf[blockDim.x];
    float sum_r = 0.0f, sum_i = 0.0f;
    for (int i = threadIdx.x; i < rows; i += blockDim.x) {
        sum_r += v_1_vec[i].x * Sv_vec[i].x + v_1_vec[i].y * Sv_vec[i].y;
        sum_i += v_1_vec[i].x * Sv_vec[i].y - v_1_vec[i].y * Sv_vec[i].x;
    }
    sreal[threadIdx.x] = sum_r;
    simag[threadIdx.x] = sum_i;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sreal[threadIdx.x] += sreal[threadIdx.x + s];
            simag[threadIdx.x] += simag[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        d_beta[vec_idx].x = sreal[0];
        d_beta[vec_idx].y = simag[0];
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

void batched_axpy_then_negate_gpu(double* alpha, const double* x, double* y,
                                   int rows, int numvec, int ld,
                                   cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    batched_axpy_then_negate_kernel<<<blocks, threads, 0, stream>>>(
        alpha, x, y, rows, numvec, ld);
}

void batched_axpy_then_negate_gpu(float* alpha, const float* x, float* y,
                                   int rows, int numvec, int ld,
                                   cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    batched_axpy_then_negate_kernel<<<blocks, threads, 0, stream>>>(
        alpha, x, y, rows, numvec, ld);
}

void batched_axpy_then_negate_gpu(cuDoubleComplex* alpha, const cuDoubleComplex* x,
                                   cuDoubleComplex* y, int rows, int numvec, int ld,
                                   cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    batched_axpy_then_negate_kernel<<<blocks, threads, 0, stream>>>(
        alpha, x, y, rows, numvec, ld);
}

void batched_axpy_then_negate_gpu(cuComplex* alpha, const cuComplex* x,
                                   cuComplex* y, int rows, int numvec, int ld,
                                   cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    batched_axpy_then_negate_kernel<<<blocks, threads, 0, stream>>>(
        alpha, x, y, rows, numvec, ld);
}

void copy_real_negate_to_T_gpu(const double* in, double* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_negate_to_T_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_negate_to_T_gpu(const float* in, float* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_negate_to_T_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_negate_to_T_gpu(const double* in, cuDoubleComplex* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_negate_to_T_complex_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_negate_to_T_gpu(const float* in, cuComplex* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_negate_to_T_complex_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void real_part_gpu(const double* in, double* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    real_part_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void real_part_gpu(const float* in, float* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    real_part_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void real_part_gpu(const cuDoubleComplex* in, double* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    real_part_complex_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void real_part_gpu(const cuComplex* in, float* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    real_part_complex_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_to_T_gpu(const double* in, double* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_to_T_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_to_T_gpu(const float* in, float* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_to_T_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_to_T_gpu(const double* in, cuDoubleComplex* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_to_T_complex_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_to_T_gpu(const float* in, cuComplex* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_to_T_complex_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void real_reciprocal_gpu(const double* in, double* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    real_reciprocal_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void real_reciprocal_gpu(const float* in, float* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    real_reciprocal_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_reciprocal_to_T_gpu(const double* in, double* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_reciprocal_to_T_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_reciprocal_to_T_gpu(const float* in, float* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_reciprocal_to_T_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_reciprocal_to_T_gpu(const double* in, cuDoubleComplex* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_reciprocal_to_T_complex_double_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void copy_real_reciprocal_to_T_gpu(const float* in, cuComplex* out, int n, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    copy_real_reciprocal_to_T_complex_float_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
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

void fused_norm_squared_normalize_gpu(double* vectors, double* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double);
    fused_norm_squared_normalize_double_kernel<<<blocks, threads, shared_mem, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void fused_norm_squared_normalize_gpu(float* vectors, float* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float);
    fused_norm_squared_normalize_float_kernel<<<blocks, threads, shared_mem, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void fused_norm_squared_normalize_gpu(cuDoubleComplex* vectors,
                                      double* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double);
    fused_norm_squared_normalize_complex_double_kernel<<<blocks, threads, shared_mem, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void fused_norm_squared_normalize_gpu(cuComplex* vectors, float* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float);
    fused_norm_squared_normalize_complex_float_kernel<<<blocks, threads, shared_mem, stream>>>(
        vectors, norms_squared, rows, numvec, ld);
}

void fused_dot_axpy_negate_gpu(const double* v1, const double* v2, double* y,
                               const double* x, double* alpha_out,
                               int rows, int numvec, int ld,
                               cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double);
    fused_dot_axpy_negate_double_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, y, x, alpha_out, rows, numvec, ld);
}

void fused_dot_axpy_negate_gpu(const float* v1, const float* v2, float* y,
                               const float* x, float* alpha_out,
                               int rows, int numvec, int ld,
                               cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float);
    fused_dot_axpy_negate_float_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, y, x, alpha_out, rows, numvec, ld);
}

void fused_dot_axpy_negate_gpu(const cuDoubleComplex* v1, const cuDoubleComplex* v2,
                               cuDoubleComplex* y, const cuDoubleComplex* x,
                               cuDoubleComplex* alpha_out,
                               int rows, int numvec, int ld,
                               cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(double) * 2;
    fused_dot_axpy_negate_complex_double_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, y, x, alpha_out, rows, numvec, ld);
}

void fused_dot_axpy_negate_gpu(const cuComplex* v1, const cuComplex* v2,
                               cuComplex* y, const cuComplex* x,
                               cuComplex* alpha_out,
                               int rows, int numvec, int ld,
                               cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = numvec;
    size_t shared_mem = threads * sizeof(float) * 2;
    fused_dot_axpy_negate_complex_float_kernel<<<blocks, threads, shared_mem, stream>>>(
        v1, v2, y, x, alpha_out, rows, numvec, ld);
}

void batched_scale_two_gpu(const double* scale, double* v1, double* v2,
                           int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_two_double_kernel<<<blocks, threads, 0, stream>>>(
        scale, v1, v2, rows, numvec, ld);
}

void batched_scale_two_gpu(const float* scale, float* v1, float* v2,
                           int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_two_float_kernel<<<blocks, threads, 0, stream>>>(
        scale, v1, v2, rows, numvec, ld);
}

void batched_scale_two_gpu(const cuDoubleComplex* scale,
                           cuDoubleComplex* v1, cuDoubleComplex* v2,
                           int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_two_complex_double_kernel<<<blocks, threads, 0, stream>>>(
        scale, v1, v2, rows, numvec, ld);
}

void batched_scale_two_gpu(const cuComplex* scale, cuComplex* v1, cuComplex* v2,
                           int rows, int numvec, int ld, cudaStream_t stream)
{
    dim3 threads(LANCZOS_BLOCK_SIZE);
    dim3 blocks((rows + threads.x - 1) / threads.x, numvec);
    batched_scale_two_complex_float_kernel<<<blocks, threads, 0, stream>>>(
        scale, v1, v2, rows, numvec, ld);
}

void scale_complex_by_real_negate_gpu(cuDoubleComplex* alpha,
                                      const double* real_scale,
                                      int numvec, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (numvec + threads - 1) / threads;
    scale_complex_by_real_negate_double_kernel<<<blocks, threads, 0, stream>>>(
        alpha, real_scale, numvec);
}

void scale_complex_by_real_negate_gpu(cuComplex* alpha,
                                      const float* real_scale,
                                      int numvec, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (numvec + threads - 1) / threads;
    scale_complex_by_real_negate_float_kernel<<<blocks, threads, 0, stream>>>(
        alpha, real_scale, numvec);
}

void scale_real_by_real_negate_gpu(double* alpha, const double* real_scale,
                                   int numvec, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (numvec + threads - 1) / threads;
    scale_real_by_real_negate_double_kernel<<<blocks, threads, 0, stream>>>(
        alpha, real_scale, numvec);
}

void scale_real_by_real_negate_gpu(float* alpha, const float* real_scale,
                                   int numvec, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    int blocks = (numvec + threads - 1) / threads;
    scale_real_by_real_negate_float_kernel<<<blocks, threads, 0, stream>>>(
        alpha, real_scale, numvec);
}

void pseudo_hermitian_init_single_gpu(double* v_1, double* v_2,
                                      double* Sv, double* d_beta,
                                      double* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double);
    pseudo_hermitian_init_single_double_kernel<<<1, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, ld);
}

void pseudo_hermitian_init_single_gpu(float* v_1, float* v_2,
                                      float* Sv, float* d_beta,
                                      float* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float);
    pseudo_hermitian_init_single_float_kernel<<<1, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, ld);
}

void pseudo_hermitian_init_single_gpu(cuDoubleComplex* v_1,
                                      cuDoubleComplex* v_2,
                                      cuDoubleComplex* Sv,
                                      cuDoubleComplex* d_beta,
                                      double* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double) * 2;
    pseudo_hermitian_init_single_complex_double_kernel<<<1, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, ld);
}

void pseudo_hermitian_init_single_gpu(cuComplex* v_1, cuComplex* v_2,
                                      cuComplex* Sv, cuComplex* d_beta,
                                      float* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float) * 2;
    pseudo_hermitian_init_single_complex_float_kernel<<<1, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, ld);
}

void pseudo_hermitian_init_batched_gpu(double* v_1, double* v_2, double* Sv,
                                       double* d_beta, double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double);
    pseudo_hermitian_init_batched_double_kernel<<<numvec, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, numvec, ld);
}

void pseudo_hermitian_init_batched_gpu(float* v_1, float* v_2, float* Sv,
                                       float* d_beta, float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float);
    pseudo_hermitian_init_batched_float_kernel<<<numvec, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, numvec, ld);
}

void pseudo_hermitian_init_batched_gpu(cuDoubleComplex* v_1, cuDoubleComplex* v_2,
                                       cuDoubleComplex* Sv,
                                       cuDoubleComplex* d_beta,
                                       double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double) * 2;
    pseudo_hermitian_init_batched_complex_double_kernel<<<numvec, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, numvec, ld);
}

void pseudo_hermitian_init_batched_gpu(cuComplex* v_1, cuComplex* v_2,
                                       cuComplex* Sv, cuComplex* d_beta,
                                       float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float) * 2;
    pseudo_hermitian_init_batched_complex_float_kernel<<<numvec, threads, shmem, stream>>>(
        v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, numvec, ld);
}

void ph_fused_dot_scale_negate_axpy_gpu(const double* v_2, const double* Sv,
                                         double* d_alpha, const double* v_1,
                                         double* v_2_out, const double* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double);
    ph_fused_dot_scale_negate_axpy_double_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, d_alpha, v_1, v_2_out, d_real_beta_prev, rows, numvec, ld);
}

void ph_fused_dot_scale_negate_axpy_gpu(const float* v_2, const float* Sv,
                                         float* d_alpha, const float* v_1,
                                         float* v_2_out, const float* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float);
    ph_fused_dot_scale_negate_axpy_float_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, d_alpha, v_1, v_2_out, d_real_beta_prev, rows, numvec, ld);
}

void ph_fused_dot_scale_negate_axpy_gpu(const cuDoubleComplex* v_2,
                                         const cuDoubleComplex* Sv,
                                         cuDoubleComplex* d_alpha,
                                         const cuDoubleComplex* v_1,
                                         cuDoubleComplex* v_2_out,
                                         const double* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double) * 2;
    ph_fused_dot_scale_negate_axpy_complex_double_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, d_alpha, v_1, v_2_out, d_real_beta_prev, rows, numvec, ld);
}

void ph_fused_dot_scale_negate_axpy_gpu(const cuComplex* v_2, const cuComplex* Sv,
                                         cuComplex* d_alpha, const cuComplex* v_1,
                                         cuComplex* v_2_out, const float* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float) * 2;
    ph_fused_dot_scale_negate_axpy_complex_float_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, d_alpha, v_1, v_2_out, d_real_beta_prev, rows, numvec, ld);
}

void ph_lacpy_flip_batched_dot_gpu(const double* v_2, double* Sv,
                                 const double* v_1, double* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double);
    ph_lacpy_flip_batched_dot_double_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, v_1, d_beta, rows, numvec, ld_v2, ld_sv, ld_v1);
}

void ph_lacpy_flip_batched_dot_gpu(const float* v_2, float* Sv,
                                 const float* v_1, float* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float);
    ph_lacpy_flip_batched_dot_float_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, v_1, d_beta, rows, numvec, ld_v2, ld_sv, ld_v1);
}

void ph_lacpy_flip_batched_dot_gpu(const cuDoubleComplex* v_2, cuDoubleComplex* Sv,
                                 const cuDoubleComplex* v_1, cuDoubleComplex* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(double) * 2;
    ph_lacpy_flip_batched_dot_complex_double_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, v_1, d_beta, rows, numvec, ld_v2, ld_sv, ld_v1);
}

void ph_lacpy_flip_batched_dot_gpu(const cuComplex* v_2, cuComplex* Sv,
                                 const cuComplex* v_1, cuComplex* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream)
{
    int threads = LANCZOS_BLOCK_SIZE;
    size_t shmem = threads * sizeof(float) * 2;
    ph_lacpy_flip_batched_dot_complex_float_kernel<<<numvec, threads, shmem, stream>>>(
        v_2, Sv, v_1, d_beta, rows, numvec, ld_v2, ld_sv, ld_v1);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
