// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/matrix/matrix.hpp"
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace chase::linalg::internal::gpu_serial::impl
{

template <typename Backend, typename T>
void ensure_pseudo_rr_workspace(
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    std::size_t n, T* d_A, ::chase::Base<T>* d_ritzv_block, T*& workspace,
    std::size_t& lwork, bool& owns_workspace)
{
    if (workspace != nullptr && lwork > 0)
    {
        return;
    }

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        using util = gpu_backend::handle_utils<Backend>;
        int lwork_heevd = 0;
        int lwork_potrf = 0;
        util::check_solver(
            gpu_backend::solver<Backend>::template heevd_buffer_size<T>(
                solver_handle, util::eig_original(), util::fill_lower(), n,
                d_A, n, d_ritzv_block, &lwork_heevd));
        util::check_solver(
            gpu_backend::solver<Backend>::template potrf_buffer_size<T>(
                solver_handle, util::fill_lower(), n, d_A, n, &lwork_potrf));
        const int lwork_int = std::max(lwork_heevd, lwork_potrf);
        gpu_backend::device_malloc<Backend, T>(
            &workspace, static_cast<std::size_t>(lwork_int));
        lwork = static_cast<std::size_t>(lwork_int);
        owns_workspace = true;
    }
}

template <typename Backend, typename T>
void pseudo_hermitian_rayleigh_ritz_blas(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    T* d_H, std::size_t N, std::size_t ldh, T* d_V1, T* d_V2, std::size_t ldv,
    ::chase::Base<T>* d_ritzv, ::chase::Base<T>* h_ritzv, std::size_t offset,
    std::size_t subSize, typename gpu_backend::handle_utils<Backend>::info_t* d_info,
    T* d_A, typename gpu_backend::handle_utils<Backend>::stream_t stream =
                  typename gpu_backend::handle_utils<Backend>::stream_t{},
    T* workspace = nullptr, std::size_t lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;
    using Scalar = ::chase::Base<T>;
    using info_t = typename util::info_t;
    using device_kernels = gpu_device::kernels<Backend>;

    const std::size_t k = N / 2;
    const std::size_t n = subSize;
    const T one = T(1);
    const T zero = T(0);
    const T negative_two = T(-2.0);
    const T negative_one = T(-1.0);
    const auto N_int = static_cast<info_t>(N);
    const auto n_int = static_cast<info_t>(n);
    const auto k_int = static_cast<info_t>(k);
    const auto ldh_int = static_cast<info_t>(ldh);
    const auto ldv_int = static_cast<info_t>(ldv);
    const auto lda_int = n_int;

    T* d_V1_block = d_V1 + offset * ldv;
    T* d_V2_block = d_V2 + offset * ldv;
    Scalar* d_ritzv_block = d_ritzv + offset;
    T* M = d_A + n * n;

    T* solver_workspace = workspace;
    std::size_t solver_lwork = lwork;
    bool owns_solver_workspace = false;
    ensure_pseudo_rr_workspace<Backend, T>(solver_handle, n, d_A, d_ritzv_block,
                                           solver_workspace, solver_lwork,
                                           owns_solver_workspace);

    util::set_stream(blas_handle, stream);
    util::set_solver_stream(solver_handle, stream);

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_none(), util::op_none(), N_int, n_int, N_int,
        &one, d_H, ldh_int, d_V1_block, ldv_int, &zero, d_V2_block, ldv_int));

    device_kernels::template flip_lower_half_matrix_sign<T>(N, n, d_V2_block, ldv,
                                                            stream);

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_conj_transpose(), util::op_none(), n_int,
        n_int, N_int, &one, d_V1_block, ldv_int, d_V2_block, ldv_int, &zero,
        d_A, lda_int));

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        util::check_solver(gpu_backend::solver<Backend>::template potrf<T>(
            solver_handle, util::fill_lower(), n, d_A, n, solver_workspace,
            solver_lwork, d_info));
    }
    else if (solver_workspace != nullptr && solver_lwork > 0)
    {
        util::check_solver(gpu_backend::solver<Backend>::template potrf<T>(
            solver_handle, util::fill_lower(), n, d_A, n, solver_workspace,
            solver_lwork, d_info));
    }
    else
    {
        util::check_solver(gpu_backend::solver<Backend>::template potrf<T>(
            solver_handle, util::fill_lower(), n, d_A, n, d_info));
    }

    info_t potrf_info = -1;
    gpu_backend::memcpy_d2h<Backend, info_t>(&potrf_info, d_info, 1);
    if (potrf_info != 0)
    {
        if (owns_solver_workspace)
        {
            gpu_backend::device_free<Backend>(solver_workspace);
        }
        throw std::runtime_error(
            "POTRF failed in Pseudo-Hermitian RayleighRitz");
    }

    gpu_backend::memset_async<Backend>(M, 0, sizeof(T) * n * n, stream);
    device_kernels::template set_diagonal<T>(M, n, n, one, stream);

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_conj_transpose(), util::op_none(), n_int,
        n_int, k_int, &negative_two, d_V1_block + k, ldv_int, d_V1_block + k,
        ldv_int, &one, M, lda_int));

    util::check_blas(gpu_backend::blas<Backend>::trsm(
        blas_handle, util::side_left(), util::fill_lower(),
        util::op_none(), util::diag_non_unit(), n_int, n_int, &one, d_A,
        lda_int, M, lda_int));

    util::check_blas(gpu_backend::blas<Backend>::trsm(
        blas_handle, util::side_right(), util::fill_lower(),
        util::op_conj_transpose(), util::diag_non_unit(), n_int, n_int,
        &one, d_A, lda_int, M, lda_int));

    util::check_blas(gpu_backend::blas<Backend>::scal(
        blas_handle, n_int * n_int, &negative_one, M, 1));

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        util::check_solver(gpu_backend::solver<Backend>::template heevd<T>(
            solver_handle, util::eig_original(), util::fill_lower(), n, M,
            n, d_ritzv_block, solver_workspace, solver_lwork, d_info));
    }
    else if (solver_workspace != nullptr && solver_lwork > 0)
    {
        util::check_solver(gpu_backend::solver<Backend>::template heevd<T>(
            solver_handle, util::eig_original(), util::fill_lower(), n, M,
            n, d_ritzv_block, solver_workspace, solver_lwork, d_info));
    }
    else
    {
        Scalar* d_E = nullptr;
        gpu_backend::device_malloc<Backend, Scalar>(&d_E, n);
        util::check_solver(gpu_backend::solver<Backend>::template heevd<T>(
            solver_handle, util::eig_original(), util::fill_lower(), n, M,
            n, d_ritzv_block, d_E, d_info));
        gpu_backend::device_free<Backend>(d_E);
    }

    info_t heevd_info = -1;
    gpu_backend::memcpy_d2h<Backend, info_t>(&heevd_info, d_info, 1);
    if (heevd_info != 0)
    {
        if (owns_solver_workspace)
        {
            gpu_backend::device_free<Backend>(solver_workspace);
        }
        throw std::runtime_error(
            "HEEVD failed in Pseudo-Hermitian RayleighRitz");
    }

    Scalar ritz_minus_one = Scalar(-1.0);
    util::check_blas(gpu_backend::blas<Backend>::scal(
        blas_handle, n_int, &ritz_minus_one, d_ritzv_block, 1));

    util::check_blas(gpu_backend::blas<Backend>::trsm(
        blas_handle, util::side_left(), util::fill_lower(),
        util::op_conj_transpose(), util::diag_non_unit(), n_int, n_int,
        &one, d_A, lda_int, M, lda_int));

    device_kernels::template inverse_entries<Scalar>(d_ritzv_block, n, stream);

    const auto copy_stream = gpu_backend::create_nonblocking_stream<Backend>();
    const auto evt_ritz_ready = gpu_backend::create_event<Backend>();
    gpu_backend::record_event<Backend>(evt_ritz_ready, stream);
    gpu_backend::stream_wait_event<Backend>(copy_stream, evt_ritz_ready);
    gpu_backend::memcpy_d2h_async<Backend, Scalar>(h_ritzv + offset, d_ritzv_block,
                                              n, copy_stream);

    Scalar* d_col_norms = nullptr;
    gpu_backend::device_malloc<Backend, Scalar>(&d_col_norms, n / 2);
    device_kernels::template fused_norm_squared_normalize<T, Scalar>(
        M, d_col_norms, static_cast<int>(n), static_cast<int>(n / 2),
        static_cast<int>(n), gpu_backend::optional_stream_ptr(stream));
    gpu_backend::device_free<Backend>(d_col_norms);

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_none(), util::op_none(),
        static_cast<info_t>(N), static_cast<info_t>(n / 2), n_int, &one,
        d_V1_block, ldv_int, M, lda_int, &zero, d_V2_block, ldv_int));

    gpu_backend::stream_synchronize<Backend>(copy_stream);
    gpu_backend::destroy_event<Backend>(evt_ritz_ready);
    gpu_backend::destroy_stream<Backend>(copy_stream);

    if (owns_solver_workspace)
    {
        gpu_backend::device_free<Backend>(solver_workspace);
    }
}


/**
 * @brief Pseudo-Hermitian Rayleigh–Ritz (POTRF + HEEVD path) on Matrix types.
 */
template <typename Backend, typename T>
void pseudo_hermitian_rayleighRitz_v2(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    ::chase::matrix::PseudoHermitianMatrix<T, ::chase::platform::GPU>* H,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V1,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V2,
    ::chase::matrix::Matrix<::chase::Base<T>, ::chase::platform::GPU>& ritzv,
    std::size_t offset, std::size_t subSize,
    typename gpu_backend::handle_utils<Backend>::info_t* devInfo,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr,
    T* workspace = nullptr, int lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;

    bool owns_A = false;
    if (A == nullptr)
    {
        A = new ::chase::matrix::Matrix<T, ::chase::platform::GPU>(2 * subSize,
                                                                   subSize);
        owns_A = true;
    }

    typename util::stream_t stream{};
    util::get_stream(blas_handle, &stream);

    pseudo_hermitian_rayleigh_ritz_blas<Backend, T>(
        blas_handle, solver_handle, H->data(), H->rows(), H->ld(), V1.data(),
        V2.data(), V1.ld(), ritzv.data(), ritzv.cpu_data(), offset, subSize,
        devInfo, A->data(), stream, workspace, static_cast<std::size_t>(lwork));

    if (owns_A)
    {
        delete A;
    }
}

} // namespace chase::linalg::internal::gpu_serial::impl
