// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace hip
{

// Host callable functions for launching kernels

// Element-wise sqrt in-place: data[i] = sqrt(data[i])
void batched_sqrt_gpu(double* data, int n, hipStream_t stream = nullptr);
void batched_sqrt_gpu(float* data, int n, hipStream_t stream = nullptr);

void normalize_vectors_gpu(hipDoubleComplex* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, hipStream_t stream = nullptr);
void normalize_vectors_gpu(hipFloatComplex* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, hipStream_t stream = nullptr);
void normalize_vectors_gpu(double* vectors, const double* norms_squared,
                          int rows, int numvec, int ld, hipStream_t stream = nullptr);
void normalize_vectors_gpu(float* vectors, const float* norms_squared,
                          int rows, int numvec, int ld, hipStream_t stream = nullptr);

void batched_dot_product_gpu(const hipDoubleComplex* v1, const hipDoubleComplex* v2,
                             hipDoubleComplex* results, int rows, int numvec, int ld,
                             bool negate, hipStream_t stream = nullptr);
void batched_dot_product_gpu(const hipFloatComplex* v1, const hipFloatComplex* v2,
                             hipFloatComplex* results, int rows, int numvec, int ld,
                             bool negate, hipStream_t stream = nullptr);
void batched_dot_product_gpu(const double* v1, const double* v2,
                             double* results, int rows, int numvec, int ld,
                             bool negate, hipStream_t stream = nullptr);
void batched_dot_product_gpu(const float* v1, const float* v2,
                             float* results, int rows, int numvec, int ld,
                             bool negate, hipStream_t stream = nullptr);

// Batched scaling: v[:,i] *= scale[i]
void batched_scale_gpu(const hipDoubleComplex* scale, hipDoubleComplex* v,
                      int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_scale_gpu(const hipFloatComplex* scale, hipFloatComplex* v,
                      int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_scale_gpu(const double* scale, double* v,
                      int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_scale_gpu(const float* scale, float* v,
                      int rows, int numvec, int ld, hipStream_t stream = nullptr);

// Batched AXPY: y[:,i] += alpha[i] * x[:,i]
void batched_axpy_gpu(const hipDoubleComplex* alpha, const hipDoubleComplex* x,
                     hipDoubleComplex* y, int rows, int numvec, int ld,
                     hipStream_t stream = nullptr);
void batched_axpy_gpu(const hipFloatComplex* alpha, const hipFloatComplex* x,
                     hipFloatComplex* y, int rows, int numvec, int ld,
                     hipStream_t stream = nullptr);
void batched_axpy_gpu(const double* alpha, const double* x,
                     double* y, int rows, int numvec, int ld,
                     hipStream_t stream = nullptr);
void batched_axpy_gpu(const float* alpha, const float* x,
                     float* y, int rows, int numvec, int ld,
                     hipStream_t stream = nullptr);

// Batched AXPY with single scalar: y[:,i] += alpha * x[:,i] (ldx, ldy may differ)
void batched_axpy_scalar_gpu(hipDoubleComplex alpha, const hipDoubleComplex* x,
                             hipDoubleComplex* y, int rows, int numvec, int ldx, int ldy,
                             hipStream_t stream = nullptr);
void batched_axpy_scalar_gpu(hipFloatComplex alpha, const hipFloatComplex* x,
                             hipFloatComplex* y, int rows, int numvec, int ldx, int ldy,
                             hipStream_t stream = nullptr);
void batched_axpy_scalar_gpu(double alpha, const double* x,
                             double* y, int rows, int numvec, int ldx, int ldy,
                             hipStream_t stream = nullptr);
void batched_axpy_scalar_gpu(float alpha, const float* x,
                             float* y, int rows, int numvec, int ldx, int ldy,
                             hipStream_t stream = nullptr);

// Batched AXPY then negate: y[:,i] += alpha[i]*x[:,i], then alpha[i] = -alpha[i]
void batched_axpy_then_negate_gpu(double* alpha, const double* x, double* y,
                                  int rows, int numvec, int ld,
                                  hipStream_t stream = nullptr);
void batched_axpy_then_negate_gpu(float* alpha, const float* x, float* y,
                                  int rows, int numvec, int ld,
                                  hipStream_t stream = nullptr);
void batched_axpy_then_negate_gpu(hipDoubleComplex* alpha, const hipDoubleComplex* x,
                                  hipDoubleComplex* y, int rows, int numvec, int ld,
                                  hipStream_t stream = nullptr);
void batched_axpy_then_negate_gpu(hipFloatComplex* alpha, const hipFloatComplex* x,
                                  hipFloatComplex* y, int rows, int numvec, int ld,
                                  hipStream_t stream = nullptr);

// Copy real array to T with negate: out[i] = T(-in[i])
void copy_real_negate_to_T_gpu(const double* in, double* out, int n, hipStream_t stream = nullptr);
void copy_real_negate_to_T_gpu(const float* in, float* out, int n, hipStream_t stream = nullptr);
void copy_real_negate_to_T_gpu(const double* in, hipDoubleComplex* out, int n, hipStream_t stream = nullptr);
void copy_real_negate_to_T_gpu(const float* in, hipFloatComplex* out, int n, hipStream_t stream = nullptr);

// Extract real part: out[i] = real(in[i]) (RealT output; for complex T, out[i] = in[i].x)
void real_part_gpu(const double* in, double* out, int n, hipStream_t stream = nullptr);
void real_part_gpu(const float* in, float* out, int n, hipStream_t stream = nullptr);
void real_part_gpu(const hipDoubleComplex* in, double* out, int n, hipStream_t stream = nullptr);
void real_part_gpu(const hipFloatComplex* in, float* out, int n, hipStream_t stream = nullptr);

// Copy real to T: out[i] = T(in[i])
void copy_real_to_T_gpu(const double* in, double* out, int n, hipStream_t stream = nullptr);
void copy_real_to_T_gpu(const float* in, float* out, int n, hipStream_t stream = nullptr);
void copy_real_to_T_gpu(const double* in, hipDoubleComplex* out, int n, hipStream_t stream = nullptr);
void copy_real_to_T_gpu(const float* in, hipFloatComplex* out, int n, hipStream_t stream = nullptr);

// Real reciprocal: out[i] = 1/in[i]
void real_reciprocal_gpu(const double* in, double* out, int n, hipStream_t stream = nullptr);
void real_reciprocal_gpu(const float* in, float* out, int n, hipStream_t stream = nullptr);

// Copy real reciprocal to T: out[i] = T(1/in[i])
void copy_real_reciprocal_to_T_gpu(const double* in, double* out, int n, hipStream_t stream = nullptr);
void copy_real_reciprocal_to_T_gpu(const float* in, float* out, int n, hipStream_t stream = nullptr);
void copy_real_reciprocal_to_T_gpu(const double* in, hipDoubleComplex* out, int n, hipStream_t stream = nullptr);
void copy_real_reciprocal_to_T_gpu(const float* in, hipFloatComplex* out, int n, hipStream_t stream = nullptr);

void batched_norm_squared_gpu(const hipDoubleComplex* v, double* norms_squared,
                              int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_norm_squared_gpu(const hipFloatComplex* v, float* norms_squared,
                              int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_norm_squared_gpu(const double* v, double* norms_squared,
                              int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_norm_squared_gpu(const float* v, float* norms_squared,
                              int rows, int numvec, int ld, hipStream_t stream = nullptr);

// Fused: norm_squared + normalize (one kernel)
void fused_norm_squared_normalize_gpu(double* vectors, double* norms_squared,
                                      int rows, int numvec, int ld,
                                      hipStream_t stream = nullptr);
void fused_norm_squared_normalize_gpu(float* vectors, float* norms_squared,
                                      int rows, int numvec, int ld,
                                      hipStream_t stream = nullptr);
void fused_norm_squared_normalize_gpu(hipDoubleComplex* vectors,
                                      double* norms_squared,
                                      int rows, int numvec, int ld,
                                      hipStream_t stream = nullptr);
void fused_norm_squared_normalize_gpu(hipFloatComplex* vectors, float* norms_squared,
                                      int rows, int numvec, int ld,
                                      hipStream_t stream = nullptr);

// Fused: dot(conj(v1)·v2) + alpha=-dot, y+=alpha*x, then alpha=dot for output (Lanczos step)
void fused_dot_axpy_negate_gpu(const double* v1, const double* v2, double* y,
                                const double* x, double* alpha_out,
                                int rows, int numvec, int ld,
                                hipStream_t stream = nullptr);
void fused_dot_axpy_negate_gpu(const float* v1, const float* v2, float* y,
                               const float* x, float* alpha_out,
                               int rows, int numvec, int ld,
                               hipStream_t stream = nullptr);
void fused_dot_axpy_negate_gpu(const hipDoubleComplex* v1, const hipDoubleComplex* v2,
                               hipDoubleComplex* y, const hipDoubleComplex* x,
                               hipDoubleComplex* alpha_out,
                               int rows, int numvec, int ld,
                               hipStream_t stream = nullptr);
void fused_dot_axpy_negate_gpu(const hipFloatComplex* v1, const hipFloatComplex* v2,
                               hipFloatComplex* y, const hipFloatComplex* x,
                               hipFloatComplex* alpha_out,
                               int rows, int numvec, int ld,
                               hipStream_t stream = nullptr);

// Scale two vector sets by the same scale array: v1 *= scale, v2 *= scale
void batched_scale_two_gpu(const double* scale, double* v1, double* v2,
                           int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_scale_two_gpu(const float* scale, float* v1, float* v2,
                           int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_scale_two_gpu(const hipDoubleComplex* scale,
                           hipDoubleComplex* v1, hipDoubleComplex* v2,
                           int rows, int numvec, int ld, hipStream_t stream = nullptr);
void batched_scale_two_gpu(const hipFloatComplex* scale, hipFloatComplex* v1, hipFloatComplex* v2,
                           int rows, int numvec, int ld, hipStream_t stream = nullptr);

// Pseudo-Hermitian: alpha[i] = -real(alpha[i]) * real_scale[i] (in-place, output as complex)
void scale_complex_by_real_negate_gpu(hipDoubleComplex* alpha, const double* real_scale,
                                      int numvec, hipStream_t stream = nullptr);
void scale_complex_by_real_negate_gpu(hipFloatComplex* alpha, const float* real_scale,
                                      int numvec, hipStream_t stream = nullptr);

// Pseudo-Hermitian real T: alpha[i] = -alpha[i] * real_scale[i] (in-place)
void scale_real_by_real_negate_gpu(double* alpha, const double* real_scale,
                                   int numvec, hipStream_t stream = nullptr);
void scale_real_by_real_negate_gpu(float* alpha, const float* real_scale,
                                   int numvec, hipStream_t stream = nullptr);

// Pseudo-Hermitian single-vector init: v_2->Sv (flip lower half), dot(v_1,Sv), scale=1/sqrt(real(dot)), write d_beta/d_real_beta_prev, scale v_1 and v_2 (one kernel)
void pseudo_hermitian_init_single_gpu(double* v_1, double* v_2,
                                      double* Sv, double* d_beta,
                                      double* d_real_beta_prev, int rows, int ld,
                                      hipStream_t stream = nullptr);
void pseudo_hermitian_init_single_gpu(float* v_1, float* v_2,
                                      float* Sv, float* d_beta,
                                      float* d_real_beta_prev, int rows, int ld,
                                      hipStream_t stream = nullptr);
void pseudo_hermitian_init_single_gpu(hipDoubleComplex* v_1,
                                      hipDoubleComplex* v_2,
                                      hipDoubleComplex* Sv,
                                      hipDoubleComplex* d_beta,
                                      double* d_real_beta_prev, int rows, int ld,
                                      hipStream_t stream = nullptr);
void pseudo_hermitian_init_single_gpu(hipFloatComplex* v_1, hipFloatComplex* v_2,
                                      hipFloatComplex* Sv, hipFloatComplex* d_beta,
                                      float* d_real_beta_prev, int rows, int ld,
                                      hipStream_t stream = nullptr);

// Pseudo-Hermitian batched fused init: lacpyFlip(v_2->Sv) + dot(v_1,Sv) + scale from dot + scale v_1,v_2 (one kernel)
void pseudo_hermitian_init_batched_gpu(double* v_1, double* v_2, double* Sv,
                                       double* d_beta, double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       hipStream_t stream = nullptr);
void pseudo_hermitian_init_batched_gpu(float* v_1, float* v_2, float* Sv,
                                       float* d_beta, float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       hipStream_t stream = nullptr);
void pseudo_hermitian_init_batched_gpu(hipDoubleComplex* v_1, hipDoubleComplex* v_2,
                                       hipDoubleComplex* Sv,
                                       hipDoubleComplex* d_beta,
                                       double* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       hipStream_t stream = nullptr);
void pseudo_hermitian_init_batched_gpu(hipFloatComplex* v_1, hipFloatComplex* v_2,
                                       hipFloatComplex* Sv, hipFloatComplex* d_beta,
                                       float* d_real_beta_prev,
                                       int rows, int numvec, int ld,
                                       hipStream_t stream = nullptr);

// Pseudo-Hermitian fused: dot(v_2,Sv)->alpha, alpha=-real(alpha)*real_beta_prev, v_2_out+=alpha*v_1
void ph_fused_dot_scale_negate_axpy_gpu(const double* v_2, const double* Sv,
                                         double* d_alpha, const double* v_1,
                                         double* v_2_out, const double* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         hipStream_t stream = nullptr);
void ph_fused_dot_scale_negate_axpy_gpu(const float* v_2, const float* Sv,
                                         float* d_alpha, const float* v_1,
                                         float* v_2_out, const float* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         hipStream_t stream = nullptr);
void ph_fused_dot_scale_negate_axpy_gpu(const hipDoubleComplex* v_2,
                                         const hipDoubleComplex* Sv,
                                         hipDoubleComplex* d_alpha,
                                         const hipDoubleComplex* v_1,
                                         hipDoubleComplex* v_2_out,
                                         const double* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         hipStream_t stream = nullptr);
void ph_fused_dot_scale_negate_axpy_gpu(const hipFloatComplex* v_2, const hipFloatComplex* Sv,
                                         hipFloatComplex* d_alpha, const hipFloatComplex* v_1,
                                         hipFloatComplex* v_2_out, const float* d_real_beta_prev,
                                         int rows, int numvec, int ld,
                                         hipStream_t stream = nullptr);

// Fused: lacpyFlipLowerHalf(v_2->Sv) + batchedDotProduct(v_1, Sv, d_beta)
void ph_lacpy_flip_batched_dot_gpu(const double* v_2, double* Sv,
                                 const double* v_1, double* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, hipStream_t stream = nullptr);
void ph_lacpy_flip_batched_dot_gpu(const float* v_2, float* Sv,
                                 const float* v_1, float* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, hipStream_t stream = nullptr);
void ph_lacpy_flip_batched_dot_gpu(const hipDoubleComplex* v_2, hipDoubleComplex* Sv,
                                 const hipDoubleComplex* v_1, hipDoubleComplex* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, hipStream_t stream = nullptr);
void ph_lacpy_flip_batched_dot_gpu(const hipFloatComplex* v_2, hipFloatComplex* Sv,
                                 const hipFloatComplex* v_1, hipFloatComplex* d_beta,
                                 int rows, int numvec, int ld_v2, int ld_sv,
                                 int ld_v1, hipStream_t stream = nullptr);

} // namespace hip
} // namespace internal
} // namespace linalg
} // namespace chase