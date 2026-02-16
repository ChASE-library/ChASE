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
 * @brief Element-wise sqrt in-place: data[i] = sqrt(data[i]) for all i.
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
 * @brief Fused batched AXPY then negate: y[:,i] += alpha[i]*x[:,i], then alpha[i] = -alpha[i]
 */
template <typename T>
void batchedAxpyThenNegate(T* alpha, const T* x, T* y, int rows, int numvec, int ld,
                          cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    batched_axpy_then_negate_gpu(alpha, x, y, rows, numvec, ld, usedStream);
}

template <typename T>
void batchedAxpyThenNegate(std::complex<T>* alpha, const std::complex<T>* x,
                          std::complex<T>* y, int rows, int numvec, int ld,
                          cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<T, double>) {
        batched_axpy_then_negate_gpu(reinterpret_cast<cuDoubleComplex*>(alpha),
                                     reinterpret_cast<const cuDoubleComplex*>(x),
                                     reinterpret_cast<cuDoubleComplex*>(y),
                                     rows, numvec, ld, usedStream);
    } else {
        batched_axpy_then_negate_gpu(reinterpret_cast<cuComplex*>(alpha),
                                     reinterpret_cast<const cuComplex*>(x),
                                     reinterpret_cast<cuComplex*>(y),
                                     rows, numvec, ld, usedStream);
    }
}

/**
 * @brief Copy real array to T with negate on GPU: out[i] = T(-in[i]). For complex T: .x = -in[i], .y = 0.
 */
template <typename T>
void copyRealNegateToT(const T* in, T* out, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    copy_real_negate_to_T_gpu(in, out, n, usedStream);
}

template <typename T>
void copyRealNegateToT(const T* in, std::complex<T>* out, int n,
                      cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<T, double>) {
        copy_real_negate_to_T_gpu(in, reinterpret_cast<cuDoubleComplex*>(out), n, usedStream);
    } else {
        copy_real_negate_to_T_gpu(in, reinterpret_cast<cuComplex*>(out), n, usedStream);
    }
}

/**
 * @brief Extract real part of T array to RealT array on GPU: out[i] = real(alpha[i]). Avoids T on host.
 */
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>>
getRealPart(const T* alpha, T* out, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    real_part_gpu(alpha, out, n, usedStream);
}

template <typename RealT>
void getRealPart(const std::complex<RealT>* alpha, RealT* out, int n,
                 cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<RealT, double>) {
        real_part_gpu(reinterpret_cast<const cuDoubleComplex*>(alpha), out, n, usedStream);
    } else {
        real_part_gpu(reinterpret_cast<const cuComplex*>(alpha), out, n, usedStream);
    }
}

/**
 * @brief Copy real array to T on GPU: out[i] = T(in[i]). For complex T: .x = in[i], .y = 0.
 */
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>>
copyRealToT(const T* in, T* out, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    copy_real_to_T_gpu(in, out, n, usedStream);
}

template <typename RealT>
void copyRealToT(const RealT* in, std::complex<RealT>* out, int n,
                 cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<RealT, double>) {
        copy_real_to_T_gpu(in, reinterpret_cast<cuDoubleComplex*>(out), n, usedStream);
    } else {
        copy_real_to_T_gpu(in, reinterpret_cast<cuComplex*>(out), n, usedStream);
    }
}

/**
 * @brief Real reciprocal on GPU: out[i] = 1/in[i].
 */
template <typename RealT>
std::enable_if_t<std::is_floating_point_v<RealT>>
realReciprocal(const RealT* in, RealT* out, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    real_reciprocal_gpu(in, out, n, usedStream);
}

/**
 * @brief Copy real reciprocal to T on GPU: out[i] = T(1/in[i]).
 */
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>>
copyRealReciprocalToT(const T* in, T* out, int n, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    copy_real_reciprocal_to_T_gpu(in, out, n, usedStream);
}

template <typename RealT>
void copyRealReciprocalToT(const RealT* in, std::complex<RealT>* out, int n,
                          cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<RealT, double>) {
        copy_real_reciprocal_to_T_gpu(in, reinterpret_cast<cuDoubleComplex*>(out), n, usedStream);
    } else {
        copy_real_reciprocal_to_T_gpu(in, reinterpret_cast<cuComplex*>(out), n, usedStream);
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

/**
 * @brief Fused: compute norm² per vector and normalize vectors in one kernel
 */
template <typename T, typename RealT>
void fusedNormSquaredNormalize(T* vectors, RealT* norms_squared,
                               int rows, int numvec, int ld,
                               cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    fused_norm_squared_normalize_gpu(vectors, norms_squared, rows, numvec, ld,
                                     usedStream);
}

template <typename T>
void fusedNormSquaredNormalize(std::complex<T>* vectors, T* norms_squared,
                               int rows, int numvec, int ld,
                               cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<T, double>) {
        fused_norm_squared_normalize_gpu(
            reinterpret_cast<cuDoubleComplex*>(vectors), norms_squared,
            rows, numvec, ld, usedStream);
    } else {
        fused_norm_squared_normalize_gpu(
            reinterpret_cast<cuComplex*>(vectors), norms_squared,
            rows, numvec, ld, usedStream);
    }
}

/**
 * @brief Fused Lanczos step: dot=conj(v1)·v2, alpha=-dot, y+=alpha*x, alpha_out=dot
 */
template <typename T>
void fusedDotAxpyNegate(const T* v1, const T* v2, T* y, const T* x, T* alpha_out,
                        int rows, int numvec, int ld,
                        cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    fused_dot_axpy_negate_gpu(v1, v2, y, x, alpha_out, rows, numvec, ld,
                              usedStream);
}

template <typename T>
void fusedDotAxpyNegate(const std::complex<T>* v1, const std::complex<T>* v2,
                        std::complex<T>* y, const std::complex<T>* x,
                        std::complex<T>* alpha_out,
                        int rows, int numvec, int ld,
                        cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<T, double>) {
        fused_dot_axpy_negate_gpu(
            reinterpret_cast<const cuDoubleComplex*>(v1),
            reinterpret_cast<const cuDoubleComplex*>(v2),
            reinterpret_cast<cuDoubleComplex*>(y),
            reinterpret_cast<const cuDoubleComplex*>(x),
            reinterpret_cast<cuDoubleComplex*>(alpha_out),
            rows, numvec, ld, usedStream);
    } else {
        fused_dot_axpy_negate_gpu(
            reinterpret_cast<const cuComplex*>(v1),
            reinterpret_cast<const cuComplex*>(v2),
            reinterpret_cast<cuComplex*>(y),
            reinterpret_cast<const cuComplex*>(x),
            reinterpret_cast<cuComplex*>(alpha_out),
            rows, numvec, ld, usedStream);
    }
}

/**
 * @brief Scale two vector sets by the same scale: v1[:,i] *= scale[i], v2[:,i] *= scale[i]
 */
template <typename T>
void batchedScaleTwo(const T* scale, T* v1, T* v2, int rows, int numvec, int ld,
                     cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    batched_scale_two_gpu(scale, v1, v2, rows, numvec, ld, usedStream);
}

template <typename T>
void batchedScaleTwo(const std::complex<T>* scale, std::complex<T>* v1,
                     std::complex<T>* v2, int rows, int numvec, int ld,
                     cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<T, double>) {
        batched_scale_two_gpu(
            reinterpret_cast<const cuDoubleComplex*>(scale),
            reinterpret_cast<cuDoubleComplex*>(v1),
            reinterpret_cast<cuDoubleComplex*>(v2),
            rows, numvec, ld, usedStream);
    } else {
        batched_scale_two_gpu(
            reinterpret_cast<const cuComplex*>(scale),
            reinterpret_cast<cuComplex*>(v1),
            reinterpret_cast<cuComplex*>(v2),
            rows, numvec, ld, usedStream);
    }
}

/**
 * @brief Pseudo-Hermitian: alpha[i] = -real(alpha[i]) * real_scale[i] (in-place, output as complex)
 */
template <typename T>
void scaleComplexByRealNegate(std::complex<T>* alpha, const T* real_scale,
                              int numvec, cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    if constexpr (std::is_same_v<T, double>) {
        scale_complex_by_real_negate_gpu(
            reinterpret_cast<cuDoubleComplex*>(alpha), real_scale,
            numvec, usedStream);
    } else {
        scale_complex_by_real_negate_gpu(
            reinterpret_cast<cuComplex*>(alpha), real_scale,
            numvec, usedStream);
    }
}

/**
 * @brief Pseudo-Hermitian real T: alpha[i] = -alpha[i] * real_scale[i] (in-place)
 */
template <typename T>
std::enable_if_t<std::is_floating_point_v<T>>
scaleComplexByRealNegate(T* alpha, const T* real_scale, int numvec,
                         cudaStream_t* stream_ = nullptr)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t usedStream = (stream_ == nullptr) ? 0 : *stream_;
    scale_real_by_real_negate_gpu(alpha, real_scale, numvec, usedStream);
}

/**
 * @brief Pseudo-Hermitian single-vector init (one kernel): v_2->Sv (flip lower half),
 * dot(v_1,Sv), scale=1/sqrt(real(dot)), write d_beta/d_real_beta_prev, scale v_1 and v_2.
 */
inline void pseudoHermitianInitSingle(double* v_1, double* v_2, double* Sv,
                                      double* d_beta, double* d_real_beta_prev,
                                      int rows, int ld, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_single_gpu(v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, ld, s);
}

inline void pseudoHermitianInitSingle(float* v_1, float* v_2, float* Sv,
                                      float* d_beta, float* d_real_beta_prev,
                                      int rows, int ld, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_single_gpu(v_1, v_2, Sv, d_beta, d_real_beta_prev, rows, ld, s);
}

inline void pseudoHermitianInitSingle(std::complex<double>* v_1,
                                      std::complex<double>* v_2,
                                      std::complex<double>* Sv,
                                      std::complex<double>* d_beta,
                                      double* d_real_beta_prev,
                                      int rows, int ld, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_single_gpu(
        reinterpret_cast<cuDoubleComplex*>(v_1),
        reinterpret_cast<cuDoubleComplex*>(v_2),
        reinterpret_cast<cuDoubleComplex*>(Sv),
        reinterpret_cast<cuDoubleComplex*>(d_beta),
        d_real_beta_prev, rows, ld, s);
}

inline void pseudoHermitianInitSingle(std::complex<float>* v_1,
                                      std::complex<float>* v_2,
                                      std::complex<float>* Sv,
                                      std::complex<float>* d_beta,
                                      float* d_real_beta_prev,
                                      int rows, int ld, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_single_gpu(
        reinterpret_cast<cuComplex*>(v_1),
        reinterpret_cast<cuComplex*>(v_2),
        reinterpret_cast<cuComplex*>(Sv),
        reinterpret_cast<cuComplex*>(d_beta),
        d_real_beta_prev, rows, ld, s);
}

inline void pseudoHermitianInitBatched(double* v_1, double* v_2, double* Sv,
                                        double* d_beta, double* d_real_beta_prev,
                                        int rows, int numvec, int ld,
                                        cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_batched_gpu(v_1, v_2, Sv, d_beta, d_real_beta_prev,
                                       rows, numvec, ld, s);
}
inline void pseudoHermitianInitBatched(float* v_1, float* v_2, float* Sv,
                                        float* d_beta, float* d_real_beta_prev,
                                        int rows, int numvec, int ld,
                                        cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_batched_gpu(v_1, v_2, Sv, d_beta, d_real_beta_prev,
                                       rows, numvec, ld, s);
}
inline void pseudoHermitianInitBatched(std::complex<double>* v_1,
                                        std::complex<double>* v_2,
                                        std::complex<double>* Sv,
                                        std::complex<double>* d_beta,
                                        double* d_real_beta_prev,
                                        int rows, int numvec, int ld,
                                        cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_batched_gpu(
        reinterpret_cast<cuDoubleComplex*>(v_1),
        reinterpret_cast<cuDoubleComplex*>(v_2),
        reinterpret_cast<cuDoubleComplex*>(Sv),
        reinterpret_cast<cuDoubleComplex*>(d_beta),
        d_real_beta_prev, rows, numvec, ld, s);
}
inline void pseudoHermitianInitBatched(std::complex<float>* v_1,
                                        std::complex<float>* v_2,
                                        std::complex<float>* Sv,
                                        std::complex<float>* d_beta,
                                        float* d_real_beta_prev,
                                        int rows, int numvec, int ld,
                                        cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    pseudo_hermitian_init_batched_gpu(
        reinterpret_cast<cuComplex*>(v_1),
        reinterpret_cast<cuComplex*>(v_2),
        reinterpret_cast<cuComplex*>(Sv),
        reinterpret_cast<cuComplex*>(d_beta),
        d_real_beta_prev, rows, numvec, ld, s);
}

inline void fusedDotScaleNegateAxpyPh(const double* v_2, const double* Sv,
                                       double* d_alpha, const double* v_1,
                                       double* v_2_out,
                                       const double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_fused_dot_scale_negate_axpy_gpu(v_2, Sv, d_alpha, v_1, v_2_out,
                                        d_real_beta_prev, rows, numvec, ld, s);
}
inline void fusedDotScaleNegateAxpyPh(const float* v_2, const float* Sv,
                                       float* d_alpha, const float* v_1,
                                       float* v_2_out,
                                       const float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_fused_dot_scale_negate_axpy_gpu(v_2, Sv, d_alpha, v_1, v_2_out,
                                        d_real_beta_prev, rows, numvec, ld, s);
}
inline void fusedDotScaleNegateAxpyPh(const std::complex<double>* v_2,
                                       const std::complex<double>* Sv,
                                       std::complex<double>* d_alpha,
                                       const std::complex<double>* v_1,
                                       std::complex<double>* v_2_out,
                                       const double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_fused_dot_scale_negate_axpy_gpu(
        reinterpret_cast<const cuDoubleComplex*>(v_2),
        reinterpret_cast<const cuDoubleComplex*>(Sv),
        reinterpret_cast<cuDoubleComplex*>(d_alpha),
        reinterpret_cast<const cuDoubleComplex*>(v_1),
        reinterpret_cast<cuDoubleComplex*>(v_2_out),
        d_real_beta_prev, rows, numvec, ld, s);
}
inline void fusedDotScaleNegateAxpyPh(const std::complex<float>* v_2,
                                       const std::complex<float>* Sv,
                                       std::complex<float>* d_alpha,
                                       const std::complex<float>* v_1,
                                       std::complex<float>* v_2_out,
                                       const float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_fused_dot_scale_negate_axpy_gpu(
        reinterpret_cast<const cuComplex*>(v_2),
        reinterpret_cast<const cuComplex*>(Sv),
        reinterpret_cast<cuComplex*>(d_alpha),
        reinterpret_cast<const cuComplex*>(v_1),
        reinterpret_cast<cuComplex*>(v_2_out),
        d_real_beta_prev, rows, numvec, ld, s);
}

inline void lacpyFlipBatchedDot(const double* v_2, double* Sv,
                                 const double* v_1, double* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_lacpy_flip_batched_dot_gpu(v_2, Sv, v_1, d_beta, rows, numvec,
                                ld_v2, ld_sv, ld_v1, s);
}
inline void lacpyFlipBatchedDot(const float* v_2, float* Sv,
                                 const float* v_1, float* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_lacpy_flip_batched_dot_gpu(v_2, Sv, v_1, d_beta, rows, numvec,
                                ld_v2, ld_sv, ld_v1, s);
}
inline void lacpyFlipBatchedDot(const std::complex<double>* v_2,
                                 std::complex<double>* Sv,
                                 const std::complex<double>* v_1,
                                 std::complex<double>* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_lacpy_flip_batched_dot_gpu(
        reinterpret_cast<const cuDoubleComplex*>(v_2),
        reinterpret_cast<cuDoubleComplex*>(Sv),
        reinterpret_cast<const cuDoubleComplex*>(v_1),
        reinterpret_cast<cuDoubleComplex*>(d_beta),
        rows, numvec, ld_v2, ld_sv, ld_v1, s);
}
inline void lacpyFlipBatchedDot(const std::complex<float>* v_2,
                                 std::complex<float>* Sv,
                                 const std::complex<float>* v_1,
                                 std::complex<float>* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t* stream_)
{
    SCOPED_NVTX_RANGE();
    cudaStream_t s = (stream_ == nullptr) ? 0 : *stream_;
    ph_lacpy_flip_batched_dot_gpu(
        reinterpret_cast<const cuComplex*>(v_2),
        reinterpret_cast<cuComplex*>(Sv),
        reinterpret_cast<const cuComplex*>(v_1),
        reinterpret_cast<cuComplex*>(d_beta),
        rows, numvec, ld_v2, ld_sv, ld_v1, s);
}

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
