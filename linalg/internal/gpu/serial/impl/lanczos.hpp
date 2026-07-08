// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "external/cpu/lapackpp/lapackpp.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/kernels/shared/lanczos_workspace.hpp"
#include "linalg/matrix/matrix.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace chase::linalg::internal::gpu_serial::impl
{

using gpu_device::lanczos_hermitian_workspace;
using gpu_device::lanczos_pseudo_hermitian_workspace;

#if defined(HAS_CUDA) || defined(HAS_HIP)

template <typename Backend, typename BlasHandle, typename T>
inline void lanczos_gemm(BlasHandle handle, int rows, int num_cols, int inner,
                         const T* alpha, T* A, int lda, T* B, int ldb,
                         const T* beta, T* C, int ldc)
{
    using util = gpu_backend::handle_utils<Backend>;
    using info_t = typename util::info_t;
    util::check_blas(gpu_backend::blas<Backend>::gemm(
        handle, util::op_none(), util::op_none(),
        static_cast<info_t>(rows), static_cast<info_t>(num_cols),
        static_cast<info_t>(inner), alpha, A, static_cast<info_t>(lda), B,
        static_cast<info_t>(ldb), beta, C, static_cast<info_t>(ldc)));
}

#endif

using namespace chase::linalg;

/**
 * @brief Performs the Lanczos algorithm for eigenvalue computation on a GPU
 * using the selected CUDA or HIP backend.
 *
 * This function performs the Lanczos algorithm on a matrix `H` (of size `M`),
 * computing an orthonormal basis `V` and the tridiagonal matrix `T` for the
 * eigenvalue problem. It also computes Ritz values and estimates of the upper
 * bounds for the eigenvalues.
 *
 * @tparam T The data type of the matrix elements (e.g., float or double).
 * @param blas_handle Backend BLAS handle used for GPU linear algebra.
 * @param M The number of Lanczos iterations.
 * @param numvec The number of runs of Lanczos.
 * @param H The matrix of size `M x M` that is the input for the Lanczos
 * algorithm.
 * @param V The matrix of size `M x numvec` that holds the orthonormal basis
 * vectors.
 * @param upperb Output parameter that will hold the upper bound for the Ritz
 * values.
 * @param ritzv Output array to hold the Ritz values computed during the Lanczos
 * algorithm.
 * @param Tau Output array to store the Tau values from the eigenvalue
 * decomposition.
 * @param ritzV Output array for storing the Ritz vectors.
 */
template <typename Backend, typename T, typename BlasHandle>
void lanczos(BlasHandle blas_handle, std::size_t M, std::size_t numvec,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    if (M <= 0)
    {
        throw std::invalid_argument(
            "lanczos: M, the number of Lanczos iterations, must be > 0");
    }

    if (numvec <= 0)
    {
        throw std::invalid_argument(
            "lanczos: numvec, the number of Lanczos instances, must be >= 1");
    }

    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // GPU-resident serial path: batched kernels and device-resident scalars,
    // mirroring the xCCL Lanczos data flow without collectives.
    using util = gpu_backend::handle_utils<Backend>;
    using kernels = gpu_device::kernels<Backend>;
    gpu_backend::blas_side_stream_scope<Backend> stream_scope(blas_handle);
    auto stream = stream_scope.stream();
    auto kstream = gpu_backend::optional_stream_ptr(stream);
    std::vector<RealT> d_tmp(M * numvec);
    std::vector<RealT> e_tmp(M * numvec);
    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);
    std::vector<RealT> de_host(2 * M * numvec);
    std::vector<RealT> r_beta_host(numvec);

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);

    lanczos_hermitian_workspace<Backend, T> ws(
        M, numvec, stream, true);
    ws.zero_scalars(M, numvec, stream);

    gpu_backend::memset_async<Backend>(v_0.data(), 0, v_0.ld() * numvec * sizeof(T), stream);

    kernels::template lacpy<T>('A', N, numvec, V.data(), V.ld(), v_1.data(),
                               v_1.ld(), kstream);

    // Initial normalization:
    //   s_i = ||v_1^{(i)}||_2^2,   v_1^{(i)} <- v_1^{(i)} / sqrt(s_i)
    kernels::template fused_norm_squared_normalize<T, RealT>(
        v_1.data(), ws.d_real_alpha, static_cast<int>(v_1.rows()),
        static_cast<int>(numvec), static_cast<int>(v_1.ld()), kstream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (auto i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d_async<Backend, T>(
                V.data() + k * V.ld(), v_1.data() + i * v_1.ld(), v_1.rows(),
                stream);
        }
        lanczos_gemm<Backend>(
            blas_handle, static_cast<int>(H->rows()),
            static_cast<int>(numvec), static_cast<int>(H->cols()), &One,
            H->data(), static_cast<int>(H->ld()), v_1.data(),
            static_cast<int>(v_1.ld()), &Zero, v_2.data(),
            static_cast<int>(v_2.ld()));

        // Fused alpha update (per vector i):
        //   alpha_k^{(i)} = <v_1^{(i)}, w^{(i)}>
        //   w^{(i)} <- w^{(i)} - alpha_k^{(i)} v_1^{(i)}
        // Kernel stores -alpha internally and re-negates for recurrence
        // consistency.
        kernels::template fused_dot_axpy_negate<T>(
            v_1.data(), v_2.data(), v_2.data(), v_1.data(), ws.d_alpha,
            static_cast<int>(v_1.rows()), static_cast<int>(numvec),
            static_cast<int>(v_1.ld()), kstream);

        // Store diagonal entry:
        //   d(k, i) = Re(alpha_k^(i))
        kernels::template get_real<T>(
            ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k * numvec, ws.d_real_alpha,
                                           numvec, stream);

        // Previous-basis correction:
        //   w^{(i)} <- w^{(i)} - beta_{k-1}^{(i)} v_0^{(i)}
        // First iteration is a no-op because beta_{-1}=0.
        // Final-step recurrence correction:
        //   w^(i) <- w^(i) - beta_{M-2}^(i) v_0^(i)
        kernels::template copy_real_negate_to_t<T>(
            ws.d_beta_prev, ws.d_beta_neg, static_cast<int>(numvec), kstream);
        kernels::template batched_axpy<T>(ws.d_beta_neg, v_0.data(), v_2.data(),
                                          static_cast<int>(v_0.rows()),
                                          static_cast<int>(numvec),
                                          static_cast<int>(v_0.ld()), kstream);

        // Beta computation:
        //   r_i = ||w^{(i)}||_2^2,   beta_k^{(i)} = sqrt(r_i)
        // Residual norm for upper-bound estimate:
        //   r_i = ||w^(i)||_2^2,  beta_{M-1}^(i) = sqrt(r_i)
        kernels::template batched_norm_squared<T>(
            v_2.data(), ws.d_r_beta, static_cast<int>(v_2.rows()),
            static_cast<int>(numvec), static_cast<int>(v_2.ld()), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_beta_prev, ws.d_r_beta, numvec,
                                           stream);
        kernels::template sqrt<RealT>(
            ws.d_beta_prev, static_cast<int>(numvec), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.e_d + k * numvec, ws.d_beta_prev,
                                           numvec, stream);

        // Next Lanczos vector:
        //   v_{k+1}^{(i)} = w^{(i)} / beta_k^{(i)} = w^{(i)} / sqrt(r_i)
        kernels::template normalize<T, RealT>(
            v_2.data(), ws.d_r_beta, static_cast<int>(v_2.rows()),
            static_cast<int>(numvec), static_cast<int>(v_2.ld()), kstream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);
    }

    {
        const std::size_t k = M - 1;

        lanczos_gemm<Backend>(
            blas_handle, static_cast<int>(H->rows()),
            static_cast<int>(numvec), static_cast<int>(H->cols()), &One,
            H->data(), static_cast<int>(H->ld()), v_1.data(),
            static_cast<int>(v_1.ld()), &Zero, v_2.data(),
            static_cast<int>(v_2.ld()));

        // Final iteration alpha update:
        //   alpha_{M-1}^{(i)} = <v_1^{(i)}, w^{(i)}>
        //   w^{(i)} <- w^{(i)} - alpha_{M-1}^{(i)} v_1^{(i)}
        kernels::template fused_dot_axpy_negate<T>(
            v_1.data(), v_2.data(), v_2.data(), v_1.data(), ws.d_alpha,
            static_cast<int>(v_1.rows()), static_cast<int>(numvec),
            static_cast<int>(v_1.ld()), kstream);

        kernels::template get_real<T>(
            ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k * numvec, ws.d_real_alpha,
                                           numvec, stream);

        kernels::template copy_real_negate_to_t<T>(
            ws.d_beta_prev, ws.d_beta_neg, static_cast<int>(numvec), kstream);
        kernels::template batched_axpy<T>(ws.d_beta_neg, v_0.data(), v_2.data(),
                                          static_cast<int>(v_0.rows()),
                                          static_cast<int>(numvec),
                                          static_cast<int>(v_0.ld()), kstream);

        kernels::template batched_norm_squared<T>(
            v_2.data(), ws.d_r_beta, static_cast<int>(v_2.rows()),
            static_cast<int>(numvec), static_cast<int>(v_2.ld()), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_beta_prev, ws.d_r_beta, numvec,
                                           stream);
        kernels::template sqrt<RealT>(
            ws.d_beta_prev, static_cast<int>(numvec), kstream);
    }

    kernels::template lacpy<T>('A', N, numvec, v_1.data(), v_1.ld(), V.data(),
                               V.ld(), kstream);
    gpu_backend::memcpy_d2h_async<Backend, RealT>(de_host.data(), ws.d_de, 2 * M * numvec,
                                      stream);
    gpu_backend::memcpy_d2h_async<Backend, RealT>(r_beta_host.data(), ws.d_beta_prev, numvec,
                                      stream);
    gpu_backend::stream_synchronize<Backend>(stream);

    // Transpose from device layout (k * numvec + i) to LAPACK layout
    // (i * M + k).
    for (std::size_t i = 0; i < numvec; ++i)
    {
        for (std::size_t k = 0; k < M; ++k)
        {
            d_tmp[i * M + k] = de_host[k * numvec + i];
            e_tmp[i * M + k] = de_host[k * numvec + i + M * numvec];
        }
    }
    d = std::move(d_tmp);
    e = std::move(e_tmp);

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', static_cast<int>(M),
                          d.data() + i * M, e.data() + i * M, ul, ll, vl, vu,
                          &notneeded_m, ritzv + M * i, ritzV,
                          static_cast<int>(M), static_cast<int>(M),
                          isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
        {
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
        }
    }

    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(r_beta_host[0]);
    for (auto i = 1; i < numvec; i++)
    {
        RealT max_val =
            std::max(std::abs(ritzv[i * M]), std::abs(ritzv[(i + 1) * M - 1])) +
            std::abs(r_beta_host[i]);
        *upperb = std::max(*upperb, max_val);
    }

    stream_scope.finish();
}

/**
 * @brief Performs the Lanczos algorithm on matrix H to compute the tridiagonal
 * matrix and eigenvalues.
 *
 * This version of the Lanczos algorithm is a simplified version that computes
 * only the upper bound of the eigenvalue spectrum and does not compute
 * eigenvectors. It operates similarly to the full Lanczos algorithm but
 * omits the eigenvector computation step.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 *
 * @param blas_handle Backend BLAS handle used to perform matrix operations.
 * @param M Number of Lanczos iterations.
 * @param H The input matrix \( H \) (square matrix).
 * @param V The input matrix \( V \) (Lanczos starting vector).
 * @param upperb Pointer to store the upper bound of the computed eigenvalues.
 *
 * @note The function modifies the input matrices `H` and `V` during the
 * computation. It computes the tridiagonal matrix and updates the eigenvalues
 * in `ritzv`, then stores the upper bound of the eigenvalues in the `upperb`
 * pointer.
 */
template <typename Backend, typename T, typename BlasHandle>
void lanczos(BlasHandle blas_handle, std::size_t M,
             chase::matrix::Matrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // GPU-resident serial path with the same single-vector data flow as xCCL,
    // but without collectives.
    using util = gpu_backend::handle_utils<Backend>;
    using kernels = gpu_device::kernels<Backend>;
    gpu_backend::blas_side_stream_scope<Backend> stream_scope(blas_handle);
    auto stream = stream_scope.stream();
    auto kstream = gpu_backend::optional_stream_ptr(stream);

    lanczos_hermitian_workspace<Backend, T> ws(M, 1, stream, false);
    ws.zero_scalars(M, 1, stream);

    std::vector<RealT> d(M);
    std::vector<RealT> e(M);
    std::vector<RealT> de_host(2 * M);
    RealT r_beta_host;

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);

    kernels::template lacpy<T>('A', N, 1, V.data(), V.ld(), v_1.data(),
                               v_1.ld(), kstream);

    // Initial normalization:
    //   s = ||v_1||_2^2,   v_1 <- v_1 / sqrt(s)
    kernels::template fused_norm_squared_normalize<T, RealT>(
        v_1.data(), ws.d_real_alpha, static_cast<int>(v_1.rows()), 1,
        static_cast<int>(v_1.ld()), kstream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        lanczos_gemm<Backend>(
            blas_handle, static_cast<int>(H->rows()), 1,
            static_cast<int>(H->cols()), &One, H->data(),
            static_cast<int>(H->ld()), v_1.data(), static_cast<int>(v_1.ld()),
            &Zero, v_2.data(), static_cast<int>(v_2.ld()));

        // Fused alpha update:
        //   alpha_k = <v_1, w>,   w <- w - alpha_k v_1
        // Kernel stores -alpha internally and re-negates for recurrence
        // consistency.
        kernels::template fused_dot_axpy_negate<T>(
            v_1.data(), v_2.data(), v_2.data(), v_1.data(), ws.d_alpha,
            static_cast<int>(v_1.rows()), 1, static_cast<int>(v_1.ld()),
            kstream);
        // Store diagonal entry:
        //   d(k) = Re(alpha_k)
        kernels::template get_real<T>(ws.d_alpha, ws.d_real_alpha,
                                                  1, kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k, ws.d_real_alpha, 1, stream);

        // Previous-basis correction:
        //   w <- w - beta_{k-1} v_0
        kernels::template copy_real_negate_to_t<T>(
            ws.d_beta_prev, ws.d_beta_neg, 1, kstream);
        kernels::template batched_axpy<T>(ws.d_beta_neg, v_0.data(), v_2.data(),
                                          static_cast<int>(v_0.rows()), 1,
                                          static_cast<int>(v_0.ld()), kstream);

        // Beta computation:
        //   r = ||w||_2^2,   beta_k = sqrt(r)
        kernels::template batched_norm_squared<T>(
            v_2.data(), ws.d_r_beta, static_cast<int>(v_2.rows()), 1,
            static_cast<int>(v_2.ld()), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_beta_prev, ws.d_r_beta, 1, stream);
        kernels::template sqrt<RealT>(ws.d_beta_prev, 1, kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.e_d + k, ws.d_beta_prev, 1, stream);

        // Next Lanczos vector:
        //   v_{k+1} = w / beta_k = w / sqrt(r)
        kernels::template normalize<T, RealT>(
            v_2.data(), ws.d_r_beta, static_cast<int>(v_2.rows()), 1,
            static_cast<int>(v_2.ld()), kstream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);
    }

    {
        const std::size_t k = M - 1;
        lanczos_gemm<Backend>(
            blas_handle, static_cast<int>(H->rows()), 1,
            static_cast<int>(H->cols()), &One, H->data(),
            static_cast<int>(H->ld()), v_1.data(), static_cast<int>(v_1.ld()),
            &Zero, v_2.data(), static_cast<int>(v_2.ld()));
        // Final iteration alpha update:
        //   alpha_{M-1} = <v_1, w>,  w <- w - alpha_{M-1} v_1
        kernels::template fused_dot_axpy_negate<T>(
            v_1.data(), v_2.data(), v_2.data(), v_1.data(), ws.d_alpha,
            static_cast<int>(v_1.rows()), 1, static_cast<int>(v_1.ld()),
            kstream);
        kernels::template get_real<T>(ws.d_alpha, ws.d_real_alpha,
                                                  1, kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k, ws.d_real_alpha, 1, stream);
    }

    gpu_backend::memcpy_d2h_async<Backend, RealT>(de_host.data(), ws.d_de, 2 * M, stream);
    gpu_backend::memcpy_d2h_async<Backend, RealT>(&r_beta_host, ws.d_beta_prev, 1, stream);
    gpu_backend::stream_synchronize<Backend>(stream);

    for (std::size_t k = 0; k < M; ++k)
    {
        d[k] = de_host[k];
        e[k] = de_host[M + k];
    }

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> ritzv(M);

    lapackpp::t_stemr<RealT>(
        LAPACK_COL_MAJOR, 'N', 'A', static_cast<int>(M), d.data(), e.data(),
        ul, ll, vl, vu, &notneeded_m, ritzv.data(), NULL, static_cast<int>(M),
        static_cast<int>(M), isuppz.data(), &tryrac);

    *upperb =
        std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
        std::abs(r_beta_host);

    stream_scope.finish();

}

/**
 * @brief Lanczos algorithm for eigenvalue computation of Pseudo-Hermitian
 * matrices.
 *
 * This function performs the Lanczos algorithm, which is used to estimate
 * the upper bound of spectra of the pseudo-Hermitian matrix H, i.e., where H is
 * pseudo-Hermitian and SH is Hermitian Positive Definite. The algorithm is
 * iteratively applied to the matrix H, where the input matrix `H` is a square
 * matrix of size `N x N`. The Lanczos algorithm builds an orthonormal basis of
 * the Krylov subspace, and the resulting tridiagonal matrix is diagonalized
 * using the `t_stemr` function. This pseudo-code of this implementation can be
 * found in `https://doi.org/10.1016/j.commatsci.2011.02.021`
 *
 * @tparam T The data type for the matrix elements (e.g., float, double).
 * @param blas_handle Backend BLAS handle used to perform matrix operations.
 * @param M The number of Lanczos iterations.
 * @param numvec The number of runs of Lanczos.
 * @param N The size of the input matrix `H`.
 * @param H The input matrix for the Lanczos algorithm (of size `N x N`).
 * @param V The input matrix used for storing vectors (of size `N x numvec`).
 * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
 * @param ritzv A pointer to store the Ritz eigenvalues.
 * @param Tau A pointer to store the computed Tau values.
 * @param ritzV A pointer to store the Ritz eigenvectors.
 */

template <typename Backend, typename T, typename BlasHandle>
void lanczos(BlasHandle blas_handle, std::size_t M, std::size_t numvec,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb, chase::Base<T>* ritzv, chase::Base<T>* Tau,
             chase::Base<T>* ritzV)
{
    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // GPU-resident pseudo-Hermitian path: batched dot/axpy/scale kernels and
    // S*v updates, mirroring xCCL without collectives.
    using util = gpu_backend::handle_utils<Backend>;
    using kernels = gpu_device::kernels<Backend>;
    gpu_backend::blas_side_stream_scope<Backend> stream_scope(blas_handle);
    auto stream = stream_scope.stream();
    auto kstream = gpu_backend::optional_stream_ptr(stream);


    std::vector<RealT> d_tmp(M * numvec);
    std::vector<RealT> e_tmp(M * numvec);
    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);
    std::vector<RealT> de_host(2 * M * numvec);

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);
    auto Sv = chase::matrix::Matrix<T, chase::platform::GPU>(N, numvec);

    lanczos_pseudo_hermitian_workspace<Backend, T> ws(
        M, numvec, stream, true);
    ws.zero_scalars(M, numvec, stream);
    gpu_backend::memset_async<Backend>(v_0.data(), 0, v_0.ld() * numvec * sizeof(T), stream);

    kernels::template lacpy<T>('A', N, numvec, V.data(), V.ld(),
                                           v_1.data(), v_1.ld(), kstream);

    lanczos_gemm<Backend>(blas_handle, static_cast<int>(H->rows()), static_cast<int>(numvec), static_cast<int>(H->cols()), &One, H->data(), static_cast<int>(H->ld()), v_1.data(), static_cast<int>(v_1.ld()), &Zero, v_2.data(), static_cast<int>(v_2.ld()));
    // Pseudo-Hermitian initialization (per vector i):
    //   w^(i) = H v_1^(i),   Sv^(i) = S w^(i)
    //   beta_0^(i)^2 = <v_1^(i), Sv^(i)>,   d_real_beta_prev(i) <- 1 / beta_0^(i)
    //   v_1^(i), v_2^(i) are scaled so the recurrence starts in normalized form.
    kernels::template pseudo_hermitian_init_batched<T, RealT>(v_1.data(), v_2.data(), Sv.data(), ws.d_beta,
                               ws.d_real_beta_prev, static_cast<int>(N),
                               static_cast<int>(numvec),
                               static_cast<int>(v_1.ld()), kstream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (auto i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d_async<Backend, T>(
                V.data() + k * V.ld(), v_1.data() + i * v_1.ld(), v_1.rows(),
                stream);
        }

        // Fused alpha step (per i):
        //   alpha_k^(i) = <v_2^(i), Sv^(i)>
        //   v_2^(i) <- v_2^(i) - alpha_k^(i) * (1 / beta_{k-1}^(i)) v_1^(i)
        kernels::template fused_dot_scale_negate_axpy_ph<T, RealT>(v_2.data(), Sv.data(), ws.d_alpha,
                                  v_1.data(), v_2.data(), ws.d_real_beta_prev,
                                  static_cast<int>(v_2.rows()),
                                  static_cast<int>(numvec),
                                  static_cast<int>(v_1.ld()), kstream);
        // Store diagonal entry:
        //   d(k, i) = alpha_k^(i)
        kernels::template get_real<T>(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k * numvec, ws.d_real_alpha,
                                           numvec, stream);

        // Three-term recurrence correction:
        //   v_2^(i) <- v_2^(i) - beta_{k-1}^(i) v_0^(i)
        // Since d_real_beta_prev = 1 / beta_{k-1}, first recover beta_{k-1}.
        kernels::template real_reciprocal<RealT>(ws.d_real_beta_prev, ws.d_real_beta,
                       static_cast<int>(numvec), kstream);
        kernels::template copy_real_negate_to_t<T>(ws.d_real_beta, ws.d_beta, static_cast<int>(numvec),
                          kstream);
        kernels::template batched_axpy<T>(ws.d_beta, v_0.data(), v_2.data(),
                    static_cast<int>(v_0.rows()), static_cast<int>(numvec),
                    static_cast<int>(v_0.ld()), kstream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);

        lanczos_gemm<Backend>(blas_handle, static_cast<int>(H->rows()), static_cast<int>(numvec), static_cast<int>(H->cols()), &One, H->data(), static_cast<int>(H->ld()), v_1.data(), static_cast<int>(v_1.ld()), &Zero, v_2.data(), static_cast<int>(v_2.ld()));
        // Build S*v_2 and off-diagonal coefficient (per i):
        //   Sv^(i) = S v_2^(i)
        //   beta_k^(i)^2 = <v_1^(i), Sv^(i)>
        kernels::template lacpy_flip_batched_dot<T>(v_2.data(), Sv.data(), v_1.data(), ws.d_beta,
                            static_cast<int>(N), static_cast<int>(numvec),
                            static_cast<int>(v_2.ld()), static_cast<int>(Sv.ld()),
                            static_cast<int>(v_1.ld()), kstream);

        // Off-diagonal magnitude:
        //   beta_k^(i) = sqrt(beta_k^(i)^2)
        kernels::template get_real<T>(ws.d_beta, ws.d_real_beta, static_cast<int>(numvec), kstream);
        kernels::template sqrt<RealT>(ws.d_real_beta, static_cast<int>(numvec), kstream);
        // Store off-diagonal entry:
        //   e(k, i) = beta_k^(i), for k = 0..M-2
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.e_d + k * numvec, ws.d_real_beta,
                                           numvec, stream);

        // Normalize next recurrence state:
        //   beta_k^(i) = sqrt(beta_k^(i)^2)
        //   d_real_beta_prev(i) <- 1 / beta_k^(i)
        //   v_1^(i), v_2^(i) <- (1 / beta_k^(i)) * {v_1^(i), v_2^(i)}
        kernels::template copy_real_reciprocal_to_t<T>(ws.d_real_beta, ws.d_beta, static_cast<int>(numvec),
                              kstream);
        kernels::template real_reciprocal<RealT>(ws.d_real_beta, ws.d_real_beta_prev,
                       static_cast<int>(numvec), kstream);
        kernels::template batched_scale_two<T>(ws.d_beta, v_1.data(), v_2.data(),
                        static_cast<int>(v_1.rows()), static_cast<int>(numvec),
                        static_cast<int>(v_1.ld()), kstream);
    }

    {
        const std::size_t k = M - 1;

        for (auto i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d_async<Backend, T>(
                V.data() + k * V.ld(), v_1.data() + i * v_1.ld(), v_1.rows(),
                stream);
        }

        // Final alpha step:
        //   alpha_{M-1}^(i) = <v_2^(i), Sv^(i)>
        //   v_2^(i) <- v_2^(i) - alpha_{M-1}^(i) * (1 / beta_{M-2}^(i)) v_1^(i)
        kernels::template fused_dot_scale_negate_axpy_ph<T, RealT>(v_2.data(), Sv.data(), ws.d_alpha,
                                  v_1.data(), v_2.data(), ws.d_real_beta_prev,
                                  static_cast<int>(v_2.rows()),
                                  static_cast<int>(numvec),
                                  static_cast<int>(v_1.ld()), kstream);
        // Store final diagonal entry:
        //   d(M-1, i) = alpha_{M-1}^(i)
        kernels::template get_real<T>(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec), kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k * numvec, ws.d_real_alpha,
                                           numvec, stream);
    }

    kernels::template lacpy<T>('A', N, numvec, v_1.data(), v_1.ld(),
                                           V.data(), V.ld(), kstream);
    gpu_backend::memcpy_d2h_async<Backend, RealT>(de_host.data(), ws.d_de, 2 * M * numvec,
                                      stream);
    gpu_backend::stream_synchronize<Backend>(stream);

    for (std::size_t i = 0; i < numvec; ++i)
    {
        for (std::size_t k = 0; k < M; ++k)
        {
            d_tmp[i * M + k] = de_host[k * numvec + i];
            e_tmp[i * M + k] = de_host[M * numvec + k * numvec + i];
        }
    }
    d = std::move(d_tmp);
    e = std::move(e_tmp);

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', static_cast<int>(M),
                          d.data() + i * M, e.data() + i * M, ul, ll, vl, vu,
                          &notneeded_m, ritzv + M * i, ritzV, static_cast<int>(M),
                          static_cast<int>(M), isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
    }
    *upperb = ritzv[M - 1];

    stream_scope.finish();

}

/**
 * @brief Lanczos algorithm for eigenvalue computation of pseudo-Hermitian
 * matrices (simplified version).
 *
 * This version of the Lanczos algorithm is a simplified version that computes
 * only the upper bound of the eigenvalue spectrum of pseudo-Hermitian matrices
 * and does not compute eigenvectors. It operates similarly to the full
 * Lanczos algorithm but omits the eigenvector computation step.
 *
 * @tparam T The data type for the matrix elements (e.g., float, double).
 * @param blas_handle Backend BLAS handle used to perform matrix operations.
 * @param M The number of Lanczos iterations.
 * @param H The input pseudo-Hermitian matrix for the Lanczos algorithm (of size
 * `N x N`).
 * @param V The input matrix used for storing vectors (of size `N x 1`).
 * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
 */
template <typename Backend, typename T, typename BlasHandle>
void lanczos(BlasHandle blas_handle, std::size_t M,
             chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H,
             chase::matrix::Matrix<T, chase::platform::GPU>& V,
             chase::Base<T>* upperb)
{
    using RealT = chase::Base<T>;
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t N = H->rows();

    // GPU-resident pseudo-Hermitian path with numvec=1, S*v and flip kernels,
    // and no collectives.
    using util = gpu_backend::handle_utils<Backend>;
    using kernels = gpu_device::kernels<Backend>;
    gpu_backend::blas_side_stream_scope<Backend> stream_scope(blas_handle);
    auto stream = stream_scope.stream();
    auto kstream = gpu_backend::optional_stream_ptr(stream);

    auto v_0 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_1 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto v_2 = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);
    auto Sv = chase::matrix::Matrix<T, chase::platform::GPU>(N, 1);

    std::vector<RealT> d(M);
    std::vector<RealT> e(M);
    std::vector<RealT> de_host(2 * M);
    RealT real_beta_val;

    lanczos_pseudo_hermitian_workspace<Backend, T> ws(M, 1, stream, true);
    ws.zero_scalars(M, 1, stream);
    gpu_backend::memset_async<Backend>(v_0.data(), 0, v_0.ld() * sizeof(T), stream);

    kernels::template lacpy<T>('A', N, 1, V.data(), V.ld(),
                                           v_1.data(), v_1.ld(), kstream);

    lanczos_gemm<Backend>(blas_handle, static_cast<int>(H->rows()), 1, static_cast<int>(H->cols()), &One, H->data(), static_cast<int>(H->ld()), v_1.data(), static_cast<int>(v_1.ld()), &Zero, v_2.data(), static_cast<int>(v_2.ld()));
    // Pseudo-Hermitian initialization:
    //   w = H v_1,   Sv = S w
    //   beta_0^2 = <v_1, Sv>,   d_real_beta_prev <- 1 / beta_0
    //   v_1, v_2 are scaled to start the recurrence.
    kernels::template pseudo_hermitian_init_single<T, RealT>(v_1.data(), v_2.data(), Sv.data(), ws.d_beta,
                              ws.d_real_beta_prev, static_cast<int>(N),
                              static_cast<int>(v_1.ld()), kstream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        gpu_backend::memcpy_d2d_async<Backend, T>(V.data() + k * V.ld(), v_1.data(),
                                       v_1.rows(), stream);

        kernels::template fused_dot_scale_negate_axpy_ph<T, RealT>(v_2.data(), Sv.data(), ws.d_alpha,
                                  v_1.data(), v_2.data(), ws.d_real_beta_prev,
                                  static_cast<int>(v_2.rows()), 1,
                                  static_cast<int>(v_1.ld()), kstream);
        kernels::template get_real<T>(ws.d_alpha, ws.d_real_alpha, 1, kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k, ws.d_real_alpha, 1, stream);

        kernels::template real_reciprocal<RealT>(ws.d_real_beta_prev, ws.d_real_beta, 1, kstream);
        kernels::template copy_real_negate_to_t<T>(ws.d_real_beta, ws.d_beta, 1, kstream);
        kernels::template batched_axpy<T>(ws.d_beta, v_0.data(), v_2.data(),
                    static_cast<int>(v_0.rows()), 1,
                    static_cast<int>(v_0.ld()), kstream);

        v_1.swapDataPointer(v_0);
        v_1.swapDataPointer(v_2);

        lanczos_gemm<Backend>(blas_handle, static_cast<int>(H->rows()), 1, static_cast<int>(H->cols()), &One, H->data(), static_cast<int>(H->ld()), v_1.data(), static_cast<int>(v_1.ld()), &Zero, v_2.data(), static_cast<int>(v_2.ld()));
        kernels::template lacpy_flip_batched_dot<T>(v_2.data(), Sv.data(), v_1.data(), ws.d_beta,
                            static_cast<int>(N), 1,
                            static_cast<int>(v_2.ld()), static_cast<int>(Sv.ld()),
                            static_cast<int>(v_1.ld()), kstream);

        kernels::template get_real<T>(ws.d_beta, ws.d_real_beta, 1, kstream);
        kernels::template sqrt<RealT>(ws.d_real_beta, 1, kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.e_d + k, ws.d_real_beta, 1, stream);
        kernels::template copy_real_reciprocal_to_t<T>(ws.d_real_beta, ws.d_beta, 1, kstream);
        kernels::template real_reciprocal<RealT>(ws.d_real_beta, ws.d_real_beta_prev, 1, kstream);
        kernels::template batched_scale_two<T>(ws.d_beta, v_1.data(), v_2.data(),
                        static_cast<int>(v_1.rows()), 1,
                        static_cast<int>(v_1.ld()), kstream);
    }

    {
        const std::size_t k = M - 1;
        gpu_backend::memcpy_d2d_async<Backend, T>(V.data() + k * V.ld(), v_1.data(),
                                       v_1.rows(), stream);
        kernels::template fused_dot_scale_negate_axpy_ph<T, RealT>(v_2.data(), Sv.data(), ws.d_alpha,
                                  v_1.data(), v_2.data(), ws.d_real_beta_prev,
                                  static_cast<int>(v_2.rows()), 1,
                                  static_cast<int>(v_1.ld()), kstream);
        kernels::template get_real<T>(ws.d_alpha, ws.d_real_alpha, 1, kstream);
        gpu_backend::memcpy_d2d_async<Backend, RealT>(ws.d_d + k, ws.d_real_alpha, 1, stream);
    }

    kernels::template lacpy<T>('A', N, 1, v_1.data(), v_1.ld(),
                                           V.data(), V.ld(), kstream);
    gpu_backend::memcpy_d2h_async<Backend, RealT>(de_host.data(), ws.d_de, 2 * M, stream);
    gpu_backend::memcpy_d2h_async<Backend, RealT>(&real_beta_val, ws.d_real_beta_prev, 1,
                                      stream);
    gpu_backend::stream_synchronize<Backend>(stream);
    for (std::size_t k = 0; k < M; ++k)
    {
        d[k] = de_host[k];
        e[k] = de_host[M + k];
    }

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> ritzv(M);

    lapackpp::t_stemr<RealT>(
        LAPACK_COL_MAJOR, 'N', 'A', static_cast<int>(M), d.data(), e.data(),
        ul, ll, vl, vu, &notneeded_m, ritzv.data(), NULL,
        static_cast<int>(M), static_cast<int>(M), isuppz.data(), &tryrac);
    *upperb = ritzv[M - 1];

    stream_scope.finish();

}


} // namespace chase::linalg::internal::gpu_serial::impl
