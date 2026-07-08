// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"

#if defined(HAS_CUDA)
#include "linalg/internal/gpu/kernels/cuda/launchers/lacpy.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/orthog_kernels.cuh"
#endif

#if defined(HAS_HIP)
#include "linalg/internal/gpu/kernels/hip/launchers/orthog_launchers.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/lacpy.hpp"
#endif

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_device
{
namespace detail
{

template <typename Backend>
struct orthog_kernels;

#if defined(HAS_CUDA)
template <>
struct orthog_kernels<gpu_backend::cuda_backend>
{
    using stream_t =
        typename gpu_backend::handle_utils<gpu_backend::cuda_backend>::stream_type;

    template <typename T>
    static void lacpy(char uplo, std::size_t m, std::size_t n, T* src,
                      std::size_t lds, T* dst, std::size_t ldd,
                      stream_t* stream)
    {
        stream_t used_stream = stream == nullptr ? nullptr : *stream;
        chase::linalg::internal::cuda::t_lacpy_gpu(
            uplo, m, n, src, lds, dst, ldd, used_stream);
    }

    template <typename T>
    static void batched_norm_squared(const T* v, chase::Base<T>* result,
                                     int rows, int numvec, int ld,
                                     stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedNormSquared(v, result, rows,
                                                          numvec, ld, stream);
    }

    template <typename T>
    static void batched_dot(const T* x, const T* y, T* result, int rows,
                            int numvec, int ld, bool negate, stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedDotProduct(
            x, y, result, rows, numvec, ld, negate, stream);
    }

    template <typename T>
    static void batched_axpy(const T* alpha, const T* x, T* y, int rows,
                             int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedAxpy(alpha, x, y, rows, numvec,
                                                   ld, stream);
    }

    template <typename T>
    static void batched_axpy_then_negate(T* alpha, const T* x, T* y, int rows,
                                         int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedAxpyThenNegate(
            alpha, x, y, rows, numvec, ld, stream);
    }

    template <typename T, typename RealT>
    static void normalize(T* v, const RealT* norms_squared, int rows,
                          int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::cuda::normalizeVectors(
            v, norms_squared, rows, numvec, ld, stream);
    }

    template <typename RealT>
    static void sqrt(RealT* data, int n, stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedSqrt(data, n, stream);
    }

    template <typename T>
    static void get_real(const T* in, chase::Base<T>* out, int n,
                         stream_t* stream)
    {
        chase::linalg::internal::cuda::getRealPart(in, out, n, stream);
    }

    template <typename RealT>
    static void real_reciprocal(const RealT* in, RealT* out, int n,
                                stream_t* stream)
    {
        chase::linalg::internal::cuda::realReciprocal(in, out, n, stream);
    }

    template <typename T>
    static void copy_real_reciprocal_to_t(const chase::Base<T>* in, T* out,
                                          int n, stream_t* stream)
    {
        chase::linalg::internal::cuda::copyRealReciprocalToT(in, out, n,
                                                             stream);
    }

    template <typename T>
    static void copy_real_to_t(const chase::Base<T>* in, T* out, int n,
                               stream_t* stream)
    {
        chase::linalg::internal::cuda::copyRealToT(in, out, n, stream);
    }

    template <typename T>
    static void copy_real_negate_to_t(const chase::Base<T>* in, T* out, int n,
                                      stream_t* stream)
    {
        chase::linalg::internal::cuda::copyRealNegateToT(in, out, n, stream);
    }

    template <typename T>
    static void scale_by_real_negate(T* alpha, const chase::Base<T>* scale,
                                     int n, stream_t* stream)
    {
        chase::linalg::internal::cuda::scaleComplexByRealNegate(alpha, scale,
                                                                n, stream);
    }

    template <typename T>
    static void batched_scale(const T* scale, T* v, int rows, int numvec,
                              int ld, stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedScale(scale, v, rows, numvec, ld,
                                                    stream);
    }

    template <typename T>
    static void batched_scale_two(const T* scale, T* v1, T* v2, int rows,
                                  int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedScaleTwo(
            scale, v1, v2, rows, numvec, ld, stream);
    }

    template <typename T, typename RealT>
    static void fused_norm_squared_normalize(T* vectors,
                                             RealT* norms_squared, int rows,
                                             int numvec, int ld,
                                             stream_t* stream)
    {
        chase::linalg::internal::cuda::fusedNormSquaredNormalize(
            vectors, norms_squared, rows, numvec, ld, stream);
    }

    template <typename T>
    static void fused_dot_axpy_negate(const T* v1, const T* v2, T* y,
                                      const T* x, T* alpha_out, int rows,
                                      int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::cuda::fusedDotAxpyNegate(
            v1, v2, y, x, alpha_out, rows, numvec, ld, stream);
    }

    template <typename T>
    static void batched_axpy_scalar(T alpha, const T* x, T* y, int rows,
                                    int numvec, int ldx, int ldy,
                                    stream_t* stream)
    {
        chase::linalg::internal::cuda::batchedAxpyScalar(
            alpha, x, y, rows, numvec, ldx, ldy, stream);
    }

    template <typename T, typename RealT>
    static void pseudo_hermitian_init_batched(T* v1, T* v2, T* sv, T* beta,
                                              RealT* beta_prev, int rows,
                                              int numvec, int ld,
                                              stream_t* stream)
    {
        chase::linalg::internal::cuda::pseudoHermitianInitBatched(
            v1, v2, sv, beta, beta_prev, rows, numvec, ld, stream);
    }

    template <typename T, typename RealT>
    static void pseudo_hermitian_init_single(T* v1, T* v2, T* sv, T* beta,
                                             RealT* beta_prev, int rows,
                                             int ld, stream_t* stream)
    {
        chase::linalg::internal::cuda::pseudoHermitianInitSingle(
            v1, v2, sv, beta, beta_prev, rows, ld, stream);
    }

    template <typename T, typename RealT>
    static void fused_dot_scale_negate_axpy_ph(const T* v2, const T* sv,
                                               T* alpha, const T* v1,
                                               T* v2_out, RealT* beta_prev,
                                               int rows, int numvec, int ld,
                                               stream_t* stream)
    {
        chase::linalg::internal::cuda::fusedDotScaleNegateAxpyPh(
            v2, sv, alpha, v1, v2_out, beta_prev, rows, numvec, ld, stream);
    }

    template <typename T>
    static void lacpy_flip_batched_dot(const T* v2, T* sv, const T* v1,
                                       T* beta, int rows, int numvec,
                                       int ldv2, int ldsv, int ldv1,
                                       stream_t* stream)
    {
        chase::linalg::internal::cuda::lacpyFlipBatchedDot(
            v2, sv, v1, beta, rows, numvec, ldv2, ldsv, ldv1, stream);
    }
};
#endif

#if defined(HAS_HIP)
template <>
struct orthog_kernels<gpu_backend::hip_backend>
{
    using stream_t =
        typename gpu_backend::handle_utils<gpu_backend::hip_backend>::stream_type;

    template <typename T>
    static void lacpy(char uplo, std::size_t m, std::size_t n, T* src,
                      std::size_t lds, T* dst, std::size_t ldd,
                      stream_t* stream)
    {
        stream_t used_stream = stream == nullptr ? nullptr : *stream;
        chase::linalg::internal::hip::t_lacpy_gpu(uplo, m, n, src, lds, dst,
                                                  ldd, used_stream);
    }

    template <typename T>
    static void batched_norm_squared(const T* v, chase::Base<T>* result,
                                     int rows, int numvec, int ld,
                                     stream_t* stream)
    {
        chase::linalg::internal::hip::batchedNormSquared(v, result, rows,
                                                         numvec, ld, stream);
    }

    template <typename T>
    static void batched_dot(const T* x, const T* y, T* result, int rows,
                            int numvec, int ld, bool negate, stream_t* stream)
    {
        chase::linalg::internal::hip::batchedDotProduct(
            x, y, result, rows, numvec, ld, negate, stream);
    }

    template <typename T>
    static void batched_axpy(const T* alpha, const T* x, T* y, int rows,
                             int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::hip::batchedAxpy(alpha, x, y, rows, numvec,
                                                  ld, stream);
    }

    template <typename T>
    static void batched_axpy_then_negate(T* alpha, const T* x, T* y, int rows,
                                         int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::hip::batchedAxpyThenNegate(
            alpha, x, y, rows, numvec, ld, stream);
    }

    template <typename T, typename RealT>
    static void normalize(T* v, const RealT* norms_squared, int rows,
                          int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::hip::normalizeVectors(v, norms_squared, rows,
                                                       numvec, ld, stream);
    }

    template <typename RealT>
    static void sqrt(RealT* data, int n, stream_t* stream)
    {
        chase::linalg::internal::hip::batchedSqrt(data, n, stream);
    }

    template <typename T>
    static void get_real(const T* in, chase::Base<T>* out, int n,
                         stream_t* stream)
    {
        chase::linalg::internal::hip::getRealPart(in, out, n, stream);
    }

    template <typename RealT>
    static void real_reciprocal(const RealT* in, RealT* out, int n,
                                stream_t* stream)
    {
        chase::linalg::internal::hip::realReciprocal(in, out, n, stream);
    }

    template <typename T>
    static void copy_real_reciprocal_to_t(const chase::Base<T>* in, T* out,
                                          int n, stream_t* stream)
    {
        chase::linalg::internal::hip::copyRealReciprocalToT(in, out, n,
                                                            stream);
    }

    template <typename T>
    static void copy_real_to_t(const chase::Base<T>* in, T* out, int n,
                               stream_t* stream)
    {
        chase::linalg::internal::hip::copyRealToT(in, out, n, stream);
    }

    template <typename T>
    static void copy_real_negate_to_t(const chase::Base<T>* in, T* out, int n,
                                      stream_t* stream)
    {
        chase::linalg::internal::hip::copyRealNegateToT(in, out, n, stream);
    }

    template <typename T>
    static void scale_by_real_negate(T* alpha, const chase::Base<T>* scale,
                                     int n, stream_t* stream)
    {
        chase::linalg::internal::hip::scaleComplexByRealNegate(alpha, scale, n,
                                                               stream);
    }

    template <typename T>
    static void batched_scale(const T* scale, T* v, int rows, int numvec,
                              int ld, stream_t* stream)
    {
        chase::linalg::internal::hip::batchedScale(scale, v, rows, numvec, ld,
                                                   stream);
    }

    template <typename T>
    static void batched_scale_two(const T* scale, T* v1, T* v2, int rows,
                                  int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::hip::batchedScaleTwo(scale, v1, v2, rows,
                                                      numvec, ld, stream);
    }

    template <typename T, typename RealT>
    static void fused_norm_squared_normalize(T* vectors,
                                             RealT* norms_squared, int rows,
                                             int numvec, int ld,
                                             stream_t* stream)
    {
        chase::linalg::internal::hip::fusedNormSquaredNormalize(
            vectors, norms_squared, rows, numvec, ld, stream);
    }

    template <typename T>
    static void fused_dot_axpy_negate(const T* v1, const T* v2, T* y,
                                      const T* x, T* alpha_out, int rows,
                                      int numvec, int ld, stream_t* stream)
    {
        chase::linalg::internal::hip::fusedDotAxpyNegate(
            v1, v2, y, x, alpha_out, rows, numvec, ld, stream);
    }

    template <typename T>
    static void batched_axpy_scalar(T alpha, const T* x, T* y, int rows,
                                    int numvec, int ldx, int ldy,
                                    stream_t* stream)
    {
        chase::linalg::internal::hip::batchedAxpyScalar(
            alpha, x, y, rows, numvec, ldx, ldy, stream);
    }

    template <typename T, typename RealT>
    static void pseudo_hermitian_init_batched(T* v1, T* v2, T* sv, T* beta,
                                              RealT* beta_prev, int rows,
                                              int numvec, int ld,
                                              stream_t* stream)
    {
        chase::linalg::internal::hip::pseudoHermitianInitBatched(
            v1, v2, sv, beta, beta_prev, rows, numvec, ld, stream);
    }

    template <typename T, typename RealT>
    static void pseudo_hermitian_init_single(T* v1, T* v2, T* sv, T* beta,
                                             RealT* beta_prev, int rows,
                                             int ld, stream_t* stream)
    {
        chase::linalg::internal::hip::pseudoHermitianInitSingle(
            v1, v2, sv, beta, beta_prev, rows, ld, stream);
    }

    template <typename T, typename RealT>
    static void fused_dot_scale_negate_axpy_ph(const T* v2, const T* sv,
                                               T* alpha, const T* v1,
                                               T* v2_out, RealT* beta_prev,
                                               int rows, int numvec, int ld,
                                               stream_t* stream)
    {
        chase::linalg::internal::hip::fusedDotScaleNegateAxpyPh(
            v2, sv, alpha, v1, v2_out, beta_prev, rows, numvec, ld, stream);
    }

    template <typename T>
    static void lacpy_flip_batched_dot(const T* v2, T* sv, const T* v1,
                                       T* beta, int rows, int numvec,
                                       int ldv2, int ldsv, int ldv1,
                                       stream_t* stream)
    {
        chase::linalg::internal::hip::lacpyFlipBatchedDot(
            v2, sv, v1, beta, rows, numvec, ldv2, ldsv, ldv1, stream);
    }
};
#endif

} // namespace detail
} // namespace gpu_device
} // namespace internal
} // namespace linalg
} // namespace chase
