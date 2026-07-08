// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#if defined(HAS_CUDA) || defined(HAS_HIP)

#include "algorithm/types.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/kernels/shared/orthog_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>

#if defined(HAS_CUDA)
#include "linalg/internal/gpu/kernels/cuda/launchers/abs_trace.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/conjugate.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/flip_sign.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/force_hermitian.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/lacpy.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/precision_conversion.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/random_normal_distribution.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/residuals.cuh"
#include "linalg/internal/gpu/kernels/cuda/launchers/shift_diagonal.cuh"
#endif

#if defined(HAS_HIP)
#include "linalg/internal/gpu/kernels/hip/launchers/abs_trace.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/conjugate.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/flip_sign.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/lacpy.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/lanczos_reduction.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/matrix_diag.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/precision_conversion.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/random_normal_distribution.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/residuals.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/shift_diagonal.hpp"
#endif

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_device
{

/**
 * @brief Shared dispatch for ChASE-owned CUDA/HIP device kernels.
 *
 * This layer is intentionally backend-neutral at the call site: serial,
 * GPU-aware MPI, and NCCL/RCCL code call `kernels<Backend>` for ChASE custom
 * kernels such as residual, diagonal shifts, triangular packing, and Lanczos
 * batched helpers.
 *
 * Plain runtime operations (allocation, stream management, raw copies) belong
 * in `gpu_backend::runtime.hpp`. Solver operations belong in
 * `gpu_backend::solver<Backend>`.
 */
template <typename Backend>
struct kernels;

namespace detail
{

template <typename RealT>
RealT lanczos_tridiagonal_upper_bound(const RealT* d, std::size_t m,
                                      RealT beta)
{
    RealT upperb = std::abs(beta);
    for (std::size_t k = 0; k < m; ++k)
    {
        upperb = std::max(upperb, std::abs(d[k]));
    }
    return upperb;
}

template <typename Backend>
struct lanczos_pointer_mode_guard;

template <typename Backend, typename T>
class lanczos_scalar_workspace
{
public:
    explicit lanczos_scalar_workspace(
        typename gpu_backend::handle_utils<Backend>::blas_handle)
    {
    }

    ~lanczos_scalar_workspace() = default;

    lanczos_scalar_workspace(const lanczos_scalar_workspace&) = delete;
    lanczos_scalar_workspace&
    operator=(const lanczos_scalar_workspace&) = delete;

    double* double_accum() const { return nullptr; }
};

#if defined(HAS_CUDA)
template <>
struct lanczos_pointer_mode_guard<gpu_backend::cuda_backend>
{
    using handle_t =
        typename gpu_backend::handle_utils<gpu_backend::cuda_backend>::blas_handle;

    explicit lanczos_pointer_mode_guard(handle_t) {}
};
#endif

#if defined(HAS_HIP)
template <>
class lanczos_scalar_workspace<gpu_backend::hip_backend, float>
{
public:
    using backend = gpu_backend::hip_backend;
    using handle_t = typename gpu_backend::handle_utils<backend>::blas_handle;

    explicit lanczos_scalar_workspace(handle_t)
        : d_double_accum_(gpu_backend::device_malloc_array<backend, double>(1))
    {
    }

    ~lanczos_scalar_workspace()
    {
        gpu_backend::device_free<backend>(d_double_accum_);
    }

    lanczos_scalar_workspace(const lanczos_scalar_workspace&) = delete;
    lanczos_scalar_workspace&
    operator=(const lanczos_scalar_workspace&) = delete;

    double* double_accum() const { return d_double_accum_; }

private:
    double* d_double_accum_;
};

template <>
struct lanczos_pointer_mode_guard<gpu_backend::hip_backend>
{
    using handle_t =
        typename gpu_backend::handle_utils<gpu_backend::hip_backend>::blas_handle;

    handle_t handle_;
    rocblas_pointer_mode previous_;

    explicit lanczos_pointer_mode_guard(handle_t handle) : handle_(handle)
    {
        CHECK_ROCBLAS_ERROR(rocblas_get_pointer_mode(handle_, &previous_));
        CHECK_ROCBLAS_ERROR(
            rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host));
    }

    ~lanczos_pointer_mode_guard()
    {
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle_, previous_));
    }

    lanczos_pointer_mode_guard(const lanczos_pointer_mode_guard&) = delete;
    lanczos_pointer_mode_guard&
    operator=(const lanczos_pointer_mode_guard&) = delete;
};
#endif

template <typename Backend, typename T, typename BlasHandle>
void lanczos_dot(BlasHandle handle, std::size_t n, const T* x, std::size_t incx,
                 const T* y, std::size_t incy, T* result,
                 double* d_double_accum = nullptr)
{
#if defined(HAS_HIP)
    if constexpr (std::is_same_v<Backend, gpu_backend::hip_backend>)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            auto stream =
                gpu_backend::handle_utils<Backend>::get_blas_stream(handle);
            const bool owns_workspace = d_double_accum == nullptr;
            if (owns_workspace)
            {
                d_double_accum =
                    gpu_backend::device_malloc_array<Backend, double>(1);
            }
            chase::linalg::internal::hip::lanczos_dot_float_double_accum(
                n, x, incx, y, incy, d_double_accum, stream);
            double host_accum = 0.0;
            gpu_backend::stream_synchronize<Backend>(stream);
            gpu_backend::copy_to_host<Backend>(&host_accum, d_double_accum, 1);
            if (owns_workspace)
            {
                gpu_backend::device_free<Backend>(d_double_accum);
            }
            *result = static_cast<float>(host_accum);
            return;
        }
    }
#endif
    (void)d_double_accum;
    gpu_backend::blas<Backend>::dot(handle, n, x, incx, y, incy, result);
}

template <typename Backend, typename T, typename BlasHandle>
void lanczos_nrm2(BlasHandle handle, std::size_t n, const T* x,
                  std::size_t incx, chase::Base<T>* result,
                  double* d_double_accum = nullptr)
{
#if defined(HAS_HIP)
    if constexpr (std::is_same_v<Backend, gpu_backend::hip_backend>)
    {
        if constexpr (std::is_same_v<T, float>)
        {
            auto stream =
                gpu_backend::handle_utils<Backend>::get_blas_stream(handle);
            const bool owns_workspace = d_double_accum == nullptr;
            if (owns_workspace)
            {
                d_double_accum =
                    gpu_backend::device_malloc_array<Backend, double>(1);
            }
            chase::linalg::internal::hip::lanczos_nrm2_float_double_accum(
                n, x, incx, d_double_accum, stream);
            double host_accum = 0.0;
            gpu_backend::stream_synchronize<Backend>(stream);
            gpu_backend::copy_to_host<Backend>(&host_accum, d_double_accum, 1);
            if (owns_workspace)
            {
                gpu_backend::device_free<Backend>(d_double_accum);
            }
            *result = static_cast<float>(std::sqrt(host_accum));
            return;
        }
    }
#endif
    (void)d_double_accum;
    gpu_backend::blas<Backend>::nrm2(handle, n, x, incx, result);
}

} // namespace detail

#if defined(HAS_CUDA)
template <>
struct kernels<gpu_backend::cuda_backend>
    : detail::orthog_kernels<gpu_backend::cuda_backend>
{
    using backend = gpu_backend::cuda_backend;
    using util = gpu_backend::handle_utils<backend>;
    using runtime_type = gpu_backend::runtime<backend>;
    using stream_type = typename runtime_type::stream_type;
    using blas_handle = typename util::blas_handle;
    using rand_state = typename runtime_type::rand_state_type;

    static rand_state* allocate_states()
    {
        return gpu_backend::device_malloc_array<backend, rand_state>(256 * 32);
    }

    template <typename T>
    static void fill_random(unsigned long long seed, rand_state* states, T* v,
                            std::size_t n)
    {
        chase::linalg::internal::cuda::chase_rand_normal(
            seed, states, v, n, runtime_type::default_stream());
    }

    template <typename T>
    static void init_random_vectors(T* v, std::size_t n)
    {
        rand_state* states = allocate_states();
        fill_random(24141ULL, states, v, n);
        gpu_backend::device_free<backend>(states);
    }

    template <typename T>
    static void lacpy(char uplo, std::size_t m, std::size_t n, T* src,
                      std::size_t lds, T* dst, std::size_t ldd,
                      stream_type* stream = nullptr)
    {
        chase::linalg::internal::cuda::t_lacpy_gpu(
            uplo, m, n, src, lds, dst, ldd,
            runtime_type::resolve_stream(stream));
    }

    static void convert_dp_to_sp(double* input, float* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(input, output, size,
                                                            stream);
    }

    static void convert_dp_to_sp(std::complex<double>* input,
                                 std::complex<float>* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::cuda::convert_DP_TO_SP_GPU(input, output, size,
                                                            stream);
    }

    static void convert_sp_to_dp(float* input, double* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(input, output, size,
                                                            stream);
    }

    static void convert_sp_to_dp(std::complex<float>* input,
                                 std::complex<double>* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(input, output, size,
                                                            stream);
    }

    template <typename T>
    static void conjugate_inplace(T* input, std::size_t rows, std::size_t cols,
                                  std::size_t ld,
                                  stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::chase_conjugate_inplace(
            input, rows, cols, ld, stream);
    }

    template <typename T>
    static void extract_upper_triangular(
        T* matrix, std::size_t n, T* workspace, std::size_t ld,
        stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::extractUpperTriangular(matrix, ld,
                                                              workspace, n,
                                                              stream);
    }

    template <typename T>
    static void unpack_upper_triangular(
        T* workspace, std::size_t n, T* matrix, std::size_t ld,
        stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::unpackUpperTriangular(matrix, ld,
                                                             workspace, n,
                                                             stream);
    }

    template <typename T>
    static void extract_upper(T* matrix, std::size_t ld, T* upper,
                              std::size_t n, blas_handle handle)
    {
        extract_upper_triangular(matrix, n, upper, ld,
                                 util::get_blas_stream(handle));
    }

    template <typename T>
    static void unpack_upper(T* upper, std::size_t n, T* matrix, std::size_t ld,
                             blas_handle handle)
    {
        unpack_upper_triangular(upper, n, matrix, ld,
                                util::get_blas_stream(handle));
    }

    template <typename T>
    static void abs_trace(T* matrix, chase::Base<T>* d_trace, std::size_t n,
                          std::size_t ld,
                          stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::absTrace_gpu(matrix, d_trace, n, ld,
                                                    stream);
    }

    template <typename T>
    static void residual_gpu(std::size_t rows, std::size_t cols, T* W1,
                             std::size_t ldW1, T* W2, std::size_t ldW2,
                             chase::Base<T>* ritzv, chase::Base<T>* resids,
                             bool is_sqrt = false,
                             stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::residual_gpu(
            rows, cols, W1, ldW1, W2, ldW2, ritzv, resids, is_sqrt, stream);
    }

    template <typename T>
    static void shift(T* data, std::size_t* d_off_m, std::size_t* d_off_n,
                      std::size_t offsize, std::size_t ld,
                      chase::Base<T> shift)
    {
        chase::linalg::internal::cuda::chase_shift_mgpu_matrix(
            data, d_off_m, d_off_n, offsize, ld, shift,
            runtime_type::default_stream());
    }

    template <typename T>
    static void shift_diagonal(T* matrix, std::size_t n, std::size_t ld,
                               chase::Base<T> shift,
                               stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::chase_shift_matrix(matrix, n, ld, shift,
                                                          stream);
    }

    template <typename T>
    static void shift_matrix_from_device_shift(
        T* matrix, std::size_t n, std::size_t ld,
        const chase::Base<T>* d_shift, std::size_t shift_scale_num_rows,
        stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::chase_shift_matrix_from_device_shift(
            matrix, n, ld, d_shift, shift_scale_num_rows, stream);
    }

    template <typename T>
    static void plus_inverse_diagonal(T* matrix, std::size_t n, std::size_t lda,
                                      chase::Base<T> coef, chase::Base<T>* diag,
                                      stream_type stream)
    {
        chase::linalg::internal::cuda::chase_plus_inverse_diagonal(
            matrix, n, lda, coef, diag, stream);
    }

    template <typename T>
    static void subtract_inverse_diagonal(T* matrix, std::size_t n,
                                          std::size_t lda,
                                          chase::Base<T> coef,
                                          chase::Base<T>* diag,
                                          stream_type stream =
                                              runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::chase_subtract_inverse_diagonal(
            matrix, n, lda, coef, diag, stream);
    }

    template <typename T>
    static void flip_lower_half_matrix_sign(std::size_t m, std::size_t n,
                                            T* data, std::size_t ld,
                                            stream_type stream)
    {
        chase::linalg::internal::cuda::chase_flipLowerHalfMatrixSign(
            data, m, n, ld, stream);
    }

    template <typename T>
    static void flip_lower_block_sign(T* data, std::size_t ld,
                                      std::size_t row_offset,
                                      std::size_t num_rows,
                                      std::size_t num_cols,
                                      stream_type stream =
                                          runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::chase_flipMatrixSign(
            data + row_offset, num_rows, num_cols, ld, stream);
    }

    template <typename T>
    static void scale_lower_block_rows(T* data, std::size_t ld,
                                       std::size_t row_offset,
                                       std::size_t num_rows,
                                       std::size_t num_cols, T scale,
                                       stream_type stream =
                                           runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::chase_scaleLowerBlockRows(
            data, ld, row_offset, num_rows, num_cols, scale, stream);
    }

    template <typename T>
    static void force_hermitian(T* matrix, std::size_t n, std::size_t ld,
                                stream_type stream =
                                    runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::force_hermitian_gpu(matrix, n, ld,
                                                           stream);
    }

    template <typename T>
    static void set_diagonal(T* matrix, std::size_t n, std::size_t lda, T coef,
                             stream_type stream)
    {
        chase::linalg::internal::cuda::chase_set_diagonal(matrix, n, lda, coef,
                                                          stream);
    }

    template <typename T>
    static void scale_matrix_rows(T* matrix, std::size_t rows, std::size_t cols,
                                  std::size_t ld, const chase::Base<T>* values,
                                  stream_type stream =
                                      runtime_type::default_stream())
    {
        chase::linalg::internal::cuda::chase_scale_rows_matrix(
            matrix, rows, cols, ld, values, stream);
    }

    template <typename T>
    static void scale_rows_matrix(T* matrix, std::size_t m, std::size_t n,
                                  std::size_t lda, chase::Base<T>* weights,
                                  stream_type stream)
    {
        chase::linalg::internal::cuda::chase_scale_rows_matrix(
            matrix, m, n, lda, weights, stream);
    }

    template <typename RealT>
    static void inverse_entries(RealT* vector, std::size_t n,
                                stream_type stream)
    {
        chase::linalg::internal::cuda::chase_inverse_entries(vector, n, stream);
    }
};
#endif

#if defined(HAS_HIP)
template <>
struct kernels<gpu_backend::hip_backend>
    : detail::orthog_kernels<gpu_backend::hip_backend>
{
    using backend = gpu_backend::hip_backend;
    using util = gpu_backend::handle_utils<backend>;
    using runtime_type = gpu_backend::runtime<backend>;
    using stream_type = typename runtime_type::stream_type;
    using blas_handle = typename util::blas_handle;
    using rand_state = typename runtime_type::rand_state_type;

    static rand_state* allocate_states()
    {
        return gpu_backend::device_malloc_array<backend, rand_state>(256 * 32);
    }

    template <typename T>
    static void fill_random(unsigned long long seed, rand_state* states, T* v,
                            std::size_t n)
    {
        chase::linalg::internal::hip::chase_rand_normal(
            seed, states, v, n, runtime_type::default_stream());
    }

    template <typename T>
    static void init_random_vectors(T* v, std::size_t n)
    {
        rand_state* states = allocate_states();
        fill_random(24141ULL, states, v, n);
        gpu_backend::device_free<backend>(states);
    }

    template <typename T>
    static void lacpy(char uplo, std::size_t m, std::size_t n, T* src,
                      std::size_t lds, T* dst, std::size_t ldd,
                      stream_type* stream = nullptr)
    {
        chase::linalg::internal::hip::t_lacpy_gpu(
            uplo, m, n, src, lds, dst, ldd,
            runtime_type::resolve_stream(stream));
    }

    static void convert_dp_to_sp(double* input, float* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::hip::convert_DP_TO_SP_GPU(
            input, output, size, runtime_type::resolve_stream(stream));
    }

    static void convert_dp_to_sp(std::complex<double>* input,
                                 std::complex<float>* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::hip::convert_DP_TO_SP_GPU(
            input, output, size, runtime_type::resolve_stream(stream));
    }

    static void convert_sp_to_dp(float* input, double* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::hip::convert_SP_TO_DP_GPU(
            input, output, size, runtime_type::resolve_stream(stream));
    }

    static void convert_sp_to_dp(std::complex<float>* input,
                                 std::complex<double>* output,
                                 std::size_t size,
                                 stream_type* stream = nullptr)
    {
        chase::linalg::internal::hip::convert_SP_TO_DP_GPU(
            input, output, size, runtime_type::resolve_stream(stream));
    }

    template <typename T>
    static void conjugate_inplace(T* input, std::size_t rows, std::size_t cols,
                                  std::size_t ld,
                                  stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::hip::conjugateInPlace(input, rows, cols, ld,
                                                       stream);
    }

    template <typename T>
    static void extract_upper_triangular(
        T* matrix, std::size_t n, T* workspace, std::size_t ld,
        stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::hip::extractUpperTriangular(matrix, ld,
                                                             workspace, n,
                                                             stream);
    }

    template <typename T>
    static void unpack_upper_triangular(
        T* workspace, std::size_t n, T* matrix, std::size_t ld,
        stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::hip::unpackUpperTriangular(matrix, ld,
                                                            workspace, n,
                                                            stream);
    }

    template <typename T>
    static void extract_upper(T* matrix, std::size_t ld, T* upper,
                              std::size_t n, blas_handle handle)
    {
        extract_upper_triangular(matrix, n, upper, ld,
                                 util::get_blas_stream(handle));
    }

    template <typename T>
    static void unpack_upper(T* upper, std::size_t n, T* matrix, std::size_t ld,
                             blas_handle handle)
    {
        unpack_upper_triangular(upper, n, matrix, ld,
                                util::get_blas_stream(handle));
    }

    template <typename T>
    static void abs_trace(T* matrix, chase::Base<T>* d_trace, std::size_t n,
                          std::size_t ld,
                          stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::hip::absTrace_gpu(matrix, d_trace, n, ld,
                                                   stream);
    }

    template <typename T>
    static void residual_gpu(std::size_t rows, std::size_t cols, T* W1,
                             std::size_t ldW1, T* W2, std::size_t ldW2,
                             chase::Base<T>* ritzv, chase::Base<T>* resids,
                             bool is_sqrt = false,
                             stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::hip::residual_gpu(
            rows, cols, W1, ldW1, W2, ldW2, ritzv, resids, is_sqrt, stream);
    }

    template <typename T>
    static void shift(T* data, std::size_t* d_off_m, std::size_t* d_off_n,
                      std::size_t offsize, std::size_t ld,
                      chase::Base<T> shift)
    {
        chase::linalg::internal::hip::shift_mgpu_matrix(
            data, d_off_m, d_off_n, offsize, ld, shift,
            runtime_type::default_stream());
    }

    template <typename T>
    static void shift_diagonal(T* matrix, std::size_t n, std::size_t ld,
                               chase::Base<T> shift,
                               stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::hip::shiftDiagonal(matrix, n, ld, shift,
                                                    stream);
    }

    template <typename T>
    static void shift_matrix_from_device_shift(
        T* matrix, std::size_t n, std::size_t ld,
        const chase::Base<T>* d_shift, std::size_t shift_scale_num_rows,
        stream_type stream = runtime_type::default_stream())
    {
        chase::linalg::internal::hip::shift_matrix_from_device_shift(
            matrix, n, ld, d_shift, shift_scale_num_rows, stream);
    }

    template <typename T>
    static void plus_inverse_diagonal(T* matrix, std::size_t n, std::size_t lda,
                                      chase::Base<T> coef, chase::Base<T>* diag,
                                      stream_type stream)
    {
        chase::linalg::internal::hip::chase_plus_inverse_diagonal(
            matrix, n, lda, coef, diag, stream);
    }

    template <typename T>
    static void subtract_inverse_diagonal(T* matrix, std::size_t n,
                                          std::size_t lda,
                                          chase::Base<T> coef,
                                          chase::Base<T>* diag,
                                          stream_type stream =
                                              runtime_type::default_stream())
    {
        chase::linalg::internal::hip::subtract_inverse_diagonal(
            matrix, n, lda, coef, diag, stream);
    }

    template <typename T>
    static void flip_lower_half_matrix_sign(std::size_t m, std::size_t n,
                                            T* data, std::size_t ld,
                                            stream_type stream)
    {
        chase::linalg::internal::hip::flipLowerHalfMatrixSign(data, m, n, ld,
                                                              stream);
    }

    template <typename T>
    static void flip_lower_block_sign(T* data, std::size_t ld,
                                      std::size_t row_offset,
                                      std::size_t num_rows,
                                      std::size_t num_cols,
                                      stream_type stream =
                                          runtime_type::default_stream())
    {
        chase::linalg::internal::hip::scaleLowerBlockRows(
            data, ld, row_offset, num_rows, num_cols, T(-1.0), stream);
    }

    template <typename T>
    static void scale_lower_block_rows(T* data, std::size_t ld,
                                       std::size_t row_offset,
                                       std::size_t num_rows,
                                       std::size_t num_cols, T scale,
                                       stream_type stream =
                                           runtime_type::default_stream())
    {
        chase::linalg::internal::hip::scaleLowerBlockRows(
            data, ld, row_offset, num_rows, num_cols, scale, stream);
    }

    template <typename T>
    static void set_diagonal(T* matrix, std::size_t n, std::size_t lda, T coef,
                             stream_type stream)
    {
        chase::linalg::internal::hip::chase_set_diagonal(matrix, n, lda, coef,
                                                         stream);
    }

    template <typename T>
    static void scale_matrix_rows(T* matrix, std::size_t rows, std::size_t cols,
                                  std::size_t ld, const chase::Base<T>* values,
                                  stream_type stream =
                                      runtime_type::default_stream())
    {
        chase::linalg::internal::hip::chase_scale_rows_matrix(
            matrix, rows, cols, ld, values, stream);
    }

    template <typename T>
    static void scale_rows_matrix(T* matrix, std::size_t m, std::size_t n,
                                  std::size_t lda, chase::Base<T>* weights,
                                  stream_type stream)
    {
        chase::linalg::internal::hip::chase_scale_rows_matrix(
            matrix, m, n, lda, weights, stream);
    }

    template <typename RealT>
    static void inverse_entries(RealT* vector, std::size_t n,
                                stream_type stream)
    {
        chase::linalg::internal::hip::chase_inverse_entries(vector, n, stream);
    }
};
#endif

} // namespace gpu_device
} // namespace internal
} // namespace linalg
} // namespace chase

#endif // defined(HAS_CUDA) || defined(HAS_HIP)
