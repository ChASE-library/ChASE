// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "external/cpu/lapackpp/lapackpp.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/matrix/matrix.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(HAS_CUDA)
#include <cusolverDn.h>
#endif

namespace chase::linalg::internal::gpu_serial::impl
{

#if defined(HAS_CUDA)

/**
 * @brief Pseudo-Hermitian Rayleigh–Ritz v1 (GEEV / host-LAPACK fallback path).
 *
 * CUDA-only: used when XGEEV is available or when falling back to host GEEV.
 */
template <typename T>
void pseudo_hermitian_rayleighRitz_v1(
    cublasHandle_t blas_handle, cusolverDnHandle_t solver_handle,
    cusolverDnParams_t params,
    ::chase::matrix::PseudoHermitianMatrix<T, ::chase::platform::GPU>* H,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V1,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V2,
    ::chase::matrix::Matrix<::chase::Base<T>, ::chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    T* d_workspace = nullptr, int d_lwork = 0, T* h_workspace = nullptr,
    int h_lwork = 0,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr)
{
    using Backend = gpu_backend::cuda_backend;
    using util = gpu_backend::handle_utils<Backend>;
    using Scalar = ::chase::Base<T>;
    using device_kernels = gpu_device::kernels<Backend>;
    using solver_dispatch = gpu_backend::solver<Backend>;

    const std::size_t N = H->rows();
    const std::size_t ldh = H->ld();
    const std::size_t k = N / 2;
    const std::size_t n = subSize;

    typename util::stream_t stream{};
    util::get_stream(blas_handle, &stream);

    bool owns_A = false;
    if (A == nullptr)
    {
        A = new ::chase::matrix::Matrix<T, ::chase::platform::GPU>(3 * n, n);
        owns_A = true;
    }

    std::vector<T> ritzvs_cmplex_cpu(n);
    T* ritzv_complex = A->data() + n * n;

#ifdef XGEEV_EXISTS
    T* owned_d_workspace = d_workspace;
    T* owned_h_workspace = h_workspace;
    bool owns_d_workspace = false;
    bool owns_h_workspace = false;

    if (d_workspace == nullptr || h_workspace == nullptr)
    {
        std::size_t temp_d_lwork = 0;
        std::size_t temp_h_lwork = 0;

        util::check_solver(solver_dispatch::geev_buffer_size(
            solver_handle, params, solver_dispatch::eig_mode_no_vector,
            solver_dispatch::eig_mode_vector, n, A->data(), n, ritzv_complex,
            nullptr, 1, V1.data() + offset * V1.ld(),
            static_cast<int>(V1.ld()), &temp_d_lwork, &temp_h_lwork));

        if (d_workspace == nullptr)
        {
            d_lwork = static_cast<int>(temp_d_lwork);
            gpu_backend::device_malloc<Backend, T>(
                &owned_d_workspace, static_cast<std::size_t>(d_lwork));
            owns_d_workspace = true;
        }

        if (h_workspace == nullptr)
        {
            h_lwork = static_cast<int>(temp_h_lwork);
            owned_h_workspace = new T[static_cast<std::size_t>(h_lwork)]();
            owns_h_workspace = true;
        }
    }
    else
    {
        owned_d_workspace = d_workspace;
        owned_h_workspace = h_workspace;
    }
#endif

    const T one = T(1.0);
    const T zero = T(0.0);
    const T neg_one = T(-1.0);
    const T negative_two = T(-2.0);
    const Scalar real_one = Scalar(1.0);

    auto diag = ::chase::matrix::Matrix<Scalar, ::chase::platform::GPU>(n, 1);

    T* M = A->data() + n * n;
    T* W = A->data() + 2 * n * n;
    T* d_V1_block = V1.data() + offset * V1.ld();
    T* d_V2_block = V2.data() + offset * V2.ld();
    Scalar* ptx = ritzv.cpu_data() + offset;

    const auto N_int = static_cast<int>(N);
    const auto n_int = static_cast<int>(n);
    const auto k_int = static_cast<int>(k);
    const auto ldh_int = static_cast<int>(ldh);
    const auto ldv_int = static_cast<int>(V1.ld());

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_none(), util::op_none(), N_int, n_int, N_int,
        &one, H->data(), ldh_int, d_V1_block, ldv_int, &zero, d_V2_block,
        ldv_int));

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_conj_transpose(), util::op_none(), n_int,
        n_int, N_int, &one, d_V1_block, ldv_int, d_V2_block, ldv_int, &zero,
        W, n_int));

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_conj_transpose(), util::op_none(), n_int,
        n_int, k_int, &negative_two, d_V1_block + k, ldv_int, d_V1_block + k,
        ldv_int, &zero, M, n_int));

    device_kernels::template plus_inverse_diagonal<T>(M, n, n, real_one, diag.data(),
                                                  stream);
    device_kernels::template set_diagonal<T>(M, n, n, zero, stream);

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_none(), util::op_none(), n_int, n_int, n_int,
        &neg_one, M, n_int, W, n_int, &zero, A->data(), n_int));

    device_kernels::template flip_lower_half_matrix_sign<T>(N, n, d_V2_block,
                                                        V2.ld(), stream);

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_conj_transpose(), util::op_none(), n_int,
        n_int, N_int, &one, d_V1_block, ldv_int, d_V2_block, ldv_int, &one,
        A->data(), n_int));

    device_kernels::template scale_rows_matrix<T>(A->data(), n, n, n, diag.data(),
                                              stream);

#ifdef XGEEV_EXISTS
    util::check_solver(solver_dispatch::geev(
        solver_handle, params, solver_dispatch::eig_mode_no_vector,
        solver_dispatch::eig_mode_vector, n, A->data(), n, ritzv_complex,
        nullptr, 1, W, n_int, owned_d_workspace, d_lwork, owned_h_workspace,
        h_lwork, devInfo));

    int info = -1;
    gpu_backend::memcpy_d2h<Backend, int>(&info, devInfo, 1);
    if (info != 0)
    {
        if (owns_d_workspace)
        {
            gpu_backend::device_free<Backend>(owned_d_workspace);
        }
        if (owns_h_workspace)
        {
            delete[] owned_h_workspace;
        }
        if (owns_A)
        {
            delete A;
        }
        throw std::runtime_error("cusolver GEEV failed in RayleighRitz");
    }

    if constexpr (std::is_same_v<T, std::complex<Scalar>>)
    {
        gpu_backend::memcpy_d2h<Backend, T>(ritzvs_cmplex_cpu.data(), ritzv_complex,
                                       n);
        for (std::size_t i = 0; i < n; ++i)
        {
            ptx[i] = std::real(ritzvs_cmplex_cpu[i]);
        }
    }
    else
    {
        gpu_backend::memcpy_d2h<Backend, Scalar>(ptx, ritzv_complex, n);
    }
#else
#ifdef CHASE_OUTPUT
    std::cout << "WARNING! XGeev not found in cuda. Compute Geev on CPU with "
                 "Lapack..."
              << std::endl;
#endif
    std::vector<Scalar> ptx_imag(n, Scalar(0.0));

    A->allocate_cpu_data();
    A->D2H();

    T* W_cpu = A->cpu_data() + 2 * n * n;

    lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', n, A->cpu_data(), n, ptx,
                     ptx_imag.data(), W_cpu, n);

    A->H2D();
#endif

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [ptx](std::size_t i1, std::size_t i2)
              { return ptx[i1] < ptx[i2]; });

    std::vector<Scalar> sorted_ritzv(n);
    T* d_sorted_W = A->data();
    for (std::size_t i = 0; i < n; ++i)
    {
        sorted_ritzv[i] = ptx[indices[i]];
        gpu_backend::memcpy_d2d_async<Backend, T>(d_sorted_W + i * n,
                                             W + indices[i] * n, n, stream);
    }
    gpu_backend::stream_synchronize<Backend>(stream);
    std::copy(sorted_ritzv.begin(), sorted_ritzv.end(), ptx);

    ritzv.H2D();

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_none(), util::op_none(),
        static_cast<int>(V2.rows()), n_int, n_int, &one, d_V1_block, ldv_int,
        d_sorted_W, n_int, &zero, d_V2_block, ldv_int));

#ifdef XGEEV_EXISTS
    if (owns_d_workspace)
    {
        gpu_backend::device_free<Backend>(owned_d_workspace);
    }
    if (owns_h_workspace)
    {
        delete[] owned_h_workspace;
    }
#endif

    if (owns_A)
    {
        delete A;
    }
}

#endif

} // namespace chase::linalg::internal::gpu_serial::impl
