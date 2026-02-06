// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "Impl/chase_gpu/nvtx.hpp"
#include "algorithm/types.hpp"
#include "lanczos_kernels.cuh"
#include "linalg/matrix/matrix.hpp"
#include <complex>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

/**
 * @brief Square elements in-place on GPU
 * 
 * @tparam T Data type (double or float)
 * @param data Input/output array on GPU
 * @param n Number of elements
 * @param stream_ Optional CUDA stream. If nullptr, uses default stream.
 */
template <typename T>
void squareInplace(T* data, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    square_inplace_gpu(data, n, usedStream);
}

/**
 * @brief Negate array elements in-place on GPU
 * 
 * @tparam T Data type
 * @param data Input/output array on GPU
 * @param n Number of elements
 * @param stream_ Optional CUDA stream
 */
template <typename T>
void negate(T* data, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    negate_gpu(data, n, usedStream);
}

template <typename T>
void negate(std::complex<T>* data, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    
    if constexpr (std::is_same_v<T, double>) {
        negate_gpu(reinterpret_cast<cuDoubleComplex*>(data), n, usedStream);
    } else {
        negate_gpu(reinterpret_cast<cuComplex*>(data), n, usedStream);
    }
}

/**
 * @brief Take square root in-place on GPU
 * 
 * @tparam T Data type (double or float)
 * @param data Input/output array on GPU
 * @param n Number of elements
 * @param stream_ Optional CUDA stream
 */
template <typename T>
void sqrtInplace(T* data, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    sqrt_inplace_gpu(data, n, usedStream);
}

/**
 * @brief Batched square root: data[i] = sqrt(data[i]) for all i in parallel
 * 
 * @tparam T Data type (double or float)
 * @param data Input/output array on GPU
 * @param n Number of elements
 * @param stream_ Optional CUDA stream
 */
template <typename T>
void batchedSqrt(T* data, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    batched_sqrt_gpu(data, n, usedStream);
}

/**
 * @brief Fused normalization: compute 1/sqrt(norm_squared) and scale vectors
 * 
 * This function performs a fused operation that:
 * 1. Computes scale factor: 1/sqrt(norm_squared[i]) for each vector
 * 2. Scales all elements of vector i by the scale factor
 * 
 * @tparam T Vector element type
 * @tparam RealT Real type for norms (double or float)
 * @param vectors Input/output vectors [rows × numvec] on GPU
 * @param norms_squared Input: squared norms for each vector [numvec] on GPU
 * @param rows Number of rows per vector
 * @param numvec Number of vectors
 * @param ld Leading dimension (stride between vectors)
 * @param stream_ Optional CUDA stream
 */
template <typename T, typename RealT>
void normalizeVectors(T* vectors, const RealT* norms_squared,
                     int rows, int numvec, int ld,
                     cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    normalize_vectors_gpu(vectors, norms_squared, rows, numvec, ld, usedStream);
}

template <typename T>
void normalizeVectors(std::complex<T>* vectors, const T* norms_squared,
                     int rows, int numvec, int ld,
                     cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    
    if constexpr (std::is_same_v<T, double>) {
        normalize_vectors_gpu(reinterpret_cast<cuDoubleComplex*>(vectors),
                             norms_squared, rows, numvec, ld, usedStream);
    } else {
        normalize_vectors_gpu(reinterpret_cast<cuComplex*>(vectors),
                             norms_squared, rows, numvec, ld, usedStream);
    }
}

/**
 * @brief Batched dot product for multiple vector pairs
 * 
 * Computes dot[i] = conj(v1[:,i]) · v2[:,i] for i = 0..numvec-1
 * 
 * @tparam T Vector element type
 * @param v1 First set of vectors [rows × numvec] on GPU
 * @param v2 Second set of vectors [rows × numvec] on GPU
 * @param results Output dot products [numvec] on GPU
 * @param rows Number of rows per vector
 * @param numvec Number of vector pairs
 * @param ld Leading dimension
 * @param negate If true, negate results
 * @param stream_ Optional CUDA stream
 */
template <typename T>
void batchedDotProduct(const T* v1, const T* v2, T* results,
                      int rows, int numvec, int ld, bool negate = false,
                      cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    batched_dot_product_gpu(v1, v2, results, rows, numvec, ld, negate, usedStream);
}

template <typename T>
void batchedDotProduct(const std::complex<T>* v1, const std::complex<T>* v2,
                      std::complex<T>* results, int rows, int numvec, int ld,
                      bool negate = false, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    
    if constexpr (std::is_same_v<T, double>) {
        batched_dot_product_gpu(
            reinterpret_cast<const cuDoubleComplex*>(v1),
            reinterpret_cast<const cuDoubleComplex*>(v2),
            reinterpret_cast<cuDoubleComplex*>(results),
            rows, numvec, ld, negate, usedStream);
    } else {
        batched_dot_product_gpu(
            reinterpret_cast<const cuComplex*>(v1),
            reinterpret_cast<const cuComplex*>(v2),
            reinterpret_cast<cuComplex*>(results),
            rows, numvec, ld, negate, usedStream);
    }
}

/**
 * @brief Batched scaling operation
 * 
 * Computes v[:,i] *= scale[i] for i = 0..numvec-1
 * 
 * @tparam T Vector element type
 * @param scale Scalar array [numvec] on GPU
 * @param v Input/output vectors [rows × numvec] on GPU (modified in place)
 * @param rows Number of rows per vector
 * @param numvec Number of vectors
 * @param ld Leading dimension
 * @param stream_ Optional CUDA stream
 */
template <typename T>
void batchedScale(const T* scale, T* v,
                 int rows, int numvec, int ld,
                 cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    batched_scale_gpu(scale, v, rows, numvec, ld, usedStream);
}

template <typename T>
void batchedScale(const std::complex<T>* scale, std::complex<T>* v,
                 int rows, int numvec, int ld,
                 cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    
    if constexpr (std::is_same_v<T, double>) {
        batched_scale_gpu(
            reinterpret_cast<const cuDoubleComplex*>(scale),
            reinterpret_cast<cuDoubleComplex*>(v),
            rows, numvec, ld, usedStream);
    } else {
        batched_scale_gpu(
            reinterpret_cast<const cuComplex*>(scale),
            reinterpret_cast<cuComplex*>(v),
            rows, numvec, ld, usedStream);
    }
}

/**
 * @brief Batched AXPY operation
 * 
 * Computes y[:,i] += alpha[i] * x[:,i] for i = 0..numvec-1
 * 
 * @tparam T Vector element type
 * @param alpha Scalar array [numvec] on GPU
 * @param x Input vectors [rows × numvec] on GPU
 * @param y Input/output vectors [rows × numvec] on GPU (modified in place)
 * @param rows Number of rows per vector
 * @param numvec Number of vectors
 * @param ld Leading dimension
 * @param stream_ Optional CUDA stream
 */
template <typename T>
void batchedAxpy(const T* alpha, const T* x, T* y,
                int rows, int numvec, int ld,
                cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    batched_axpy_gpu(alpha, x, y, rows, numvec, ld, usedStream);
}

template <typename T>
void batchedAxpy(const std::complex<T>* alpha, const std::complex<T>* x,
                std::complex<T>* y, int rows, int numvec, int ld,
                cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    
    if constexpr (std::is_same_v<T, double>) {
        batched_axpy_gpu(
            reinterpret_cast<const cuDoubleComplex*>(alpha),
            reinterpret_cast<const cuDoubleComplex*>(x),
            reinterpret_cast<cuDoubleComplex*>(y),
            rows, numvec, ld, usedStream);
    } else {
        batched_axpy_gpu(
            reinterpret_cast<const cuComplex*>(alpha),
            reinterpret_cast<const cuComplex*>(x),
            reinterpret_cast<cuComplex*>(y),
            rows, numvec, ld, usedStream);
    }
}

/**
 * @brief Batched norm computation (squared)
 * 
 * Computes norms_sq[i] = ||v[:,i]||² for i = 0..numvec-1
 * 
 * @tparam T Vector element type
 * @tparam RealT Real type for norm output
 * @param v Input vectors [rows × numvec] on GPU
 * @param norms_squared Output squared norms [numvec] on GPU
 * @param rows Number of rows per vector
 * @param numvec Number of vectors
 * @param ld Leading dimension
 * @param stream_ Optional CUDA stream
 */
template <typename T, typename RealT>
void batchedNormSquared(const T* v, RealT* norms_squared,
                       int rows, int numvec, int ld,
                       cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    batched_norm_squared_gpu(v, norms_squared, rows, numvec, ld, usedStream);
}

template <typename T>
void batchedNormSquared(const std::complex<T>* v, T* norms_squared,
                       int rows, int numvec, int ld,
                       cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    
    if constexpr (std::is_same_v<T, double>) {
        batched_norm_squared_gpu(reinterpret_cast<const cuDoubleComplex*>(v),
                                norms_squared, rows, numvec, ld, usedStream);
    } else {
        batched_norm_squared_gpu(reinterpret_cast<const cuComplex*>(v),
                                norms_squared, rows, numvec, ld, usedStream);
    }
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
