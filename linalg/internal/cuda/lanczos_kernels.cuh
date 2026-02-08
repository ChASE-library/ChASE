// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cuComplex.h>
#include <cuda_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

// Forward declarations of CUDA kernels

/**
 * @brief Square elements in-place on GPU
 */
template<typename T>
__global__ void square_inplace_kernel(T* data, int n);

/**
 * @brief Negate array elements in-place
 */
template<typename T>
__global__ void negate_kernel(T* data, int n);

/**
 * @brief Take square root in-place
 */
template<typename T>
__global__ void sqrt_inplace_kernel(T* data, int n);

/**
 * @brief Fused normalization: compute 1/sqrt(norm_squared) and scale vectors
 */
template<typename T, typename RealT>
__global__ void normalize_vectors_kernel(
    T* vectors,
    const RealT* norms_squared,
    int rows,
    int numvec,
    int ld);

/**
 * @brief Batched dot product for multiple vector pairs
 */
template<typename T>
__global__ void batched_dot_product_kernel(
    const T* v1,
    const T* v2,
    T* results,
    int rows,
    int numvec,
    int ld,
    bool negate);

/**
 * @brief Batched norm computation (squared)
 */
template<typename T, typename RealT>
__global__ void batched_norm_squared_kernel(
    const T* v,
    RealT* norms_squared,
    int rows,
    int numvec,
    int ld);

// Host callable functions for launching kernels

void square_inplace_gpu(double* data, int n, cudaStream_t stream);
void square_inplace_gpu(float* data, int n, cudaStream_t stream);

void negate_gpu(double* data, int n, cudaStream_t stream);
void negate_gpu(float* data, int n, cudaStream_t stream);
void negate_gpu(cuDoubleComplex* data, int n, cudaStream_t stream);
void negate_gpu(cuComplex* data, int n, cudaStream_t stream);

void sqrt_inplace_gpu(double* data, int n, cudaStream_t stream);
void sqrt_inplace_gpu(float* data, int n, cudaStream_t stream);

// Batched sqrt: data[i] = sqrt(data[i])
void batched_sqrt_gpu(double* data, int n, cudaStream_t stream);
void batched_sqrt_gpu(float* data, int n, cudaStream_t stream);

void normalize_vectors_gpu(cuDoubleComplex* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);
void normalize_vectors_gpu(cuComplex* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);
void normalize_vectors_gpu(double* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);
void normalize_vectors_gpu(float* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, cudaStream_t stream);

void batched_dot_product_gpu(const cuDoubleComplex* v1, const cuDoubleComplex* v2,
                             cuDoubleComplex* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);
void batched_dot_product_gpu(const cuComplex* v1, const cuComplex* v2,
                             cuComplex* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);
void batched_dot_product_gpu(const double* v1, const double* v2,
                             double* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);
void batched_dot_product_gpu(const float* v1, const float* v2,
                             float* results, int rows, int numvec, int ld,
                             bool negate, cudaStream_t stream);

// Batched scaling: v[:,i] *= scale[i]
void batched_scale_gpu(const cuDoubleComplex* scale, cuDoubleComplex* v,
                      int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_gpu(const cuComplex* scale, cuComplex* v,
                      int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_gpu(const double* scale, double* v,
                      int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_gpu(const float* scale, float* v,
                      int rows, int numvec, int ld, cudaStream_t stream);

// Batched AXPY: y[:,i] += alpha[i] * x[:,i]
void batched_axpy_gpu(const cuDoubleComplex* alpha, const cuDoubleComplex* x,
                     cuDoubleComplex* y, int rows, int numvec, int ld,
                     cudaStream_t stream);
void batched_axpy_gpu(const cuComplex* alpha, const cuComplex* x,
                     cuComplex* y, int rows, int numvec, int ld,
                     cudaStream_t stream);
void batched_axpy_gpu(const double* alpha, const double* x,
                     double* y, int rows, int numvec, int ld,
                     cudaStream_t stream);
void batched_axpy_gpu(const float* alpha, const float* x,
                     float* y, int rows, int numvec, int ld,
                     cudaStream_t stream);

void batched_norm_squared_gpu(const cuDoubleComplex* v, double* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);
void batched_norm_squared_gpu(const cuComplex* v, float* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);
void batched_norm_squared_gpu(const double* v, double* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);
void batched_norm_squared_gpu(const float* v, float* norms_squared,
                              int rows, int numvec, int ld, cudaStream_t stream);

// Fused: norm_squared + normalize (one kernel)
void fused_norm_squared_normalize_gpu(double* vectors, double* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream);
void fused_norm_squared_normalize_gpu(float* vectors, float* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream);
void fused_norm_squared_normalize_gpu(cuDoubleComplex* vectors,
                                      double* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream);
void fused_norm_squared_normalize_gpu(cuComplex* vectors, float* norms_squared,
                                      int rows, int numvec, int ld,
                                      cudaStream_t stream);

// Fused: dot(conj(v1)·v2) + alpha=-dot, y+=alpha*x, then alpha=dot for output (Lanczos step)
void fused_dot_axpy_negate_gpu(const double* v1, const double* v2, double* y,
                                const double* x, double* alpha_out,
                                int rows, int numvec, int ld,
                                cudaStream_t stream);
void fused_dot_axpy_negate_gpu(const float* v1, const float* v2, float* y,
                               const float* x, float* alpha_out,
                               int rows, int numvec, int ld,
                               cudaStream_t stream);
void fused_dot_axpy_negate_gpu(const cuDoubleComplex* v1, const cuDoubleComplex* v2,
                               cuDoubleComplex* y, const cuDoubleComplex* x,
                               cuDoubleComplex* alpha_out,
                               int rows, int numvec, int ld,
                               cudaStream_t stream);
void fused_dot_axpy_negate_gpu(const cuComplex* v1, const cuComplex* v2,
                               cuComplex* y, const cuComplex* x,
                               cuComplex* alpha_out,
                               int rows, int numvec, int ld,
                               cudaStream_t stream);

// Fused for pseudo-Hermitian: copy src -> dst and flip sign of lower half of dst
void lacpy_flip_lower_half_gpu(const double* src, double* dst,
                               int m, int n, int ld_src, int ld_dst,
                               cudaStream_t stream);
void lacpy_flip_lower_half_gpu(const float* src, float* dst,
                               int m, int n, int ld_src, int ld_dst,
                               cudaStream_t stream);
void lacpy_flip_lower_half_gpu(const cuDoubleComplex* src, cuDoubleComplex* dst,
                               int m, int n, int ld_src, int ld_dst,
                               cudaStream_t stream);
void lacpy_flip_lower_half_gpu(const cuComplex* src, cuComplex* dst,
                               int m, int n, int ld_src, int ld_dst,
                               cudaStream_t stream);

// Scale two vector sets by the same scale array: v1 *= scale, v2 *= scale
void batched_scale_two_gpu(const double* scale, double* v1, double* v2,
                           int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_two_gpu(const float* scale, float* v1, float* v2,
                           int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_two_gpu(const cuDoubleComplex* scale,
                           cuDoubleComplex* v1, cuDoubleComplex* v2,
                           int rows, int numvec, int ld, cudaStream_t stream);
void batched_scale_two_gpu(const cuComplex* scale, cuComplex* v1, cuComplex* v2,
                           int rows, int numvec, int ld, cudaStream_t stream);

// Pseudo-Hermitian: alpha[i] = -real(alpha[i]) * real_scale[i] (in-place, output as complex)
void scale_complex_by_real_negate_gpu(cuDoubleComplex* alpha, const double* real_scale,
                                      int numvec, cudaStream_t stream);
void scale_complex_by_real_negate_gpu(cuComplex* alpha, const float* real_scale,
                                      int numvec, cudaStream_t stream);

// Pseudo-Hermitian real T: alpha[i] = -alpha[i] * real_scale[i] (in-place)
void scale_real_by_real_negate_gpu(double* alpha, const double* real_scale,
                                   int numvec, cudaStream_t stream);
void scale_real_by_real_negate_gpu(float* alpha, const float* real_scale,
                                   int numvec, cudaStream_t stream);

// Pseudo-Hermitian single-vector init: v_2->Sv (flip lower half), dot(v_1,Sv), scale=1/sqrt(real(dot)), write d_beta/d_real_beta_prev, scale v_1 and v_2 (one kernel)
void pseudo_hermitian_init_single_gpu(double* v_1, double* v_2,
                                      double* Sv, double* d_beta,
                                      double* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream);
void pseudo_hermitian_init_single_gpu(float* v_1, float* v_2,
                                      float* Sv, float* d_beta,
                                      float* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream);
void pseudo_hermitian_init_single_gpu(cuDoubleComplex* v_1,
                                      cuDoubleComplex* v_2,
                                      cuDoubleComplex* Sv,
                                      cuDoubleComplex* d_beta,
                                      double* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream);
void pseudo_hermitian_init_single_gpu(cuComplex* v_1, cuComplex* v_2,
                                      cuComplex* Sv, cuComplex* d_beta,
                                      float* d_real_beta_prev, int rows, int ld,
                                      cudaStream_t stream);

// Pseudo-Hermitian batched init: scale[i]=1/sqrt(real(d_beta[i])), write d_beta, d_real_beta_prev (no scaling of v)
void init_scale_from_dot_batched_gpu(double* d_beta, double* d_real_beta_prev,
                                      int numvec, cudaStream_t stream);
void init_scale_from_dot_batched_gpu(float* d_beta, float* d_real_beta_prev,
                                      int numvec, cudaStream_t stream);
void init_scale_from_dot_batched_gpu(cuDoubleComplex* d_beta,
                                      double* d_real_beta_prev,
                                      int numvec, cudaStream_t stream);
void init_scale_from_dot_batched_gpu(cuComplex* d_beta,
                                      float* d_real_beta_prev,
                                      int numvec, cudaStream_t stream);

// Pseudo-Hermitian batched fused init: lacpyFlip(v_2->Sv) + dot(v_1,Sv) + scale from dot + scale v_1,v_2 (one kernel)
void pseudo_hermitian_init_batched_gpu(double* v_1, double* v_2, double* Sv,
                                       double* d_beta, double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream);
void pseudo_hermitian_init_batched_gpu(float* v_1, float* v_2, float* Sv,
                                       float* d_beta, float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream);
void pseudo_hermitian_init_batched_gpu(cuDoubleComplex* v_1, cuDoubleComplex* v_2,
                                       cuDoubleComplex* Sv,
                                       cuDoubleComplex* d_beta,
                                       double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream);
void pseudo_hermitian_init_batched_gpu(cuComplex* v_1, cuComplex* v_2,
                                       cuComplex* Sv, cuComplex* d_beta,
                                       float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       cudaStream_t stream);

// Pseudo-Hermitian fused: dot(v_2,Sv)->alpha, alpha=-real(alpha)*real_beta_prev, v_2_out+=alpha*v_1, output d_alpha
void fused_dot_scale_negate_axpy_ph_gpu(const double* v_2, const double* Sv,
                                         double* d_alpha, const double* v_1,
                                         double* v_2_out, const double* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream);
void fused_dot_scale_negate_axpy_ph_gpu(const float* v_2, const float* Sv,
                                         float* d_alpha, const float* v_1,
                                         float* v_2_out, const float* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream);
void fused_dot_scale_negate_axpy_ph_gpu(const cuDoubleComplex* v_2,
                                         const cuDoubleComplex* Sv,
                                         cuDoubleComplex* d_alpha,
                                         const cuDoubleComplex* v_1,
                                         cuDoubleComplex* v_2_out,
                                         const double* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream);
void fused_dot_scale_negate_axpy_ph_gpu(const cuComplex* v_2, const cuComplex* Sv,
                                         cuComplex* d_alpha, const cuComplex* v_1,
                                         cuComplex* v_2_out, const float* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         cudaStream_t stream);

// Fused: lacpyFlipLowerHalf(v_2->Sv) + batchedDotProduct(v_1, Sv, d_beta)
void lacpy_flip_batched_dot_gpu(const double* v_2, double* Sv,
                                 const double* v_1, double* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream);
void lacpy_flip_batched_dot_gpu(const float* v_2, float* Sv,
                                 const float* v_1, float* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream);
void lacpy_flip_batched_dot_gpu(const cuDoubleComplex* v_2, cuDoubleComplex* Sv,
                                 const cuDoubleComplex* v_1, cuDoubleComplex* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream);
void lacpy_flip_batched_dot_gpu(const cuComplex* v_2, cuComplex* Sv,
                                 const cuComplex* v_1, cuComplex* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, cudaStream_t stream);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
