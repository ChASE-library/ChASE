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
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/matrix/matrix.hpp"
#include <algorithm>
#include <cstddef>

namespace chase::linalg::internal::gpu_serial::impl
{

template <typename Backend, typename T>
void ensure_potrf_workspace(
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    std::size_t n, T* A, T*& workspace, std::size_t& lwork, bool& owns_workspace)
{
    if (workspace != nullptr && lwork > 0)
    {
        return;
    }

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        using util = gpu_backend::handle_utils<Backend>;
        int lwork_int = 0;
        util::check_solver(
            gpu_backend::solver<Backend>::template potrf_buffer_size<T>(
                solver_handle, util::fill_upper(), n, A, n, &lwork_int));
        gpu_backend::device_malloc<Backend, T>(
            &workspace, static_cast<std::size_t>(lwork_int));
        lwork = static_cast<std::size_t>(lwork_int);
        owns_workspace = true;
    }
}

template <typename Backend, typename T>
void ensure_householder_workspace(
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    std::size_t m, std::size_t n, std::size_t ldv, T* d_V, T* d_tau,
    T*& workspace, std::size_t& lwork, bool& owns_workspace)
{
    if (workspace != nullptr && lwork > 0)
    {
        return;
    }

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        using util = gpu_backend::handle_utils<Backend>;
        int lwork_geqrf = 0;
        int lwork_gqr = 0;
        util::check_solver(
            gpu_backend::solver<Backend>::template geqrf_buffer_size<T>(
                solver_handle, m, n, d_V, ldv, &lwork_geqrf));
        util::check_solver(
            gpu_backend::solver<Backend>::template gqr_buffer_size<T>(
                solver_handle, m, n, n, d_V, ldv, d_tau, &lwork_gqr));
        const int lwork_int = std::max(lwork_geqrf, lwork_gqr);
        gpu_backend::device_malloc<Backend, T>(
            &workspace, static_cast<std::size_t>(lwork_int));
        lwork = static_cast<std::size_t>(lwork_int);
        owns_workspace = true;
    }
}

namespace detail
{

/** CholQR POTRF: CUDA always uses caller workspace. HIP binds d_work_ only when
 *  BLAS and solver use separate rocblas handles; a single shared handle must not
 *  call rocblas_set_workspace before subsequent BLAS (e.g. trsm). */
template <typename Backend, typename T>
void run_potrf(typename gpu_backend::handle_utils<Backend>::solver_handle handle,
               std::size_t n, T* A, std::size_t lda,
               typename gpu_backend::handle_utils<Backend>::info_t* devInfo,
               T* workspace, std::size_t lwork, bool hip_bind_caller_workspace)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        util::check_solver(solver::template potrf<T>(
            handle, util::fill_upper(), n, A, lda, workspace, lwork, devInfo));
    }
    else if (workspace != nullptr && lwork > 0 && hip_bind_caller_workspace)
    {
        util::check_solver(solver::template potrf<T>(
            handle, util::fill_upper(), n, A, lda, workspace, lwork, devInfo));
    }
    else
    {
        util::check_solver(solver::template potrf<T>(
            handle, util::fill_upper(), n, A, lda, devInfo));
    }
}

template <typename Backend, typename T>
void potrf_with_optional_workspace(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    std::size_t n, T* A, std::size_t lda,
    typename gpu_backend::handle_utils<Backend>::info_t* devInfo, T* workspace,
    int lwork)
{
    using util = gpu_backend::handle_utils<Backend>;
    T* potrf_workspace = workspace;
    std::size_t potrf_lwork = static_cast<std::size_t>(lwork);
    bool owns_potrf_workspace = false;
    ensure_potrf_workspace<Backend, T>(
        solver_handle, n, A, potrf_workspace, potrf_lwork, owns_potrf_workspace);

    bool hip_bind_caller_workspace = true;
    if constexpr (std::is_same_v<Backend, gpu_backend::hip_backend>)
    {
        hip_bind_caller_workspace = (blas_handle != solver_handle);
    }

    run_potrf<Backend, T>(solver_handle, n, A, lda, devInfo, potrf_workspace,
                          potrf_lwork, hip_bind_caller_workspace);

    if (owns_potrf_workspace)
    {
        gpu_backend::device_free<Backend>(potrf_workspace);
    }
}

} // namespace detail

template <typename Backend, typename T>
int cholqr_pass(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle, T* d_V,
    std::size_t m, std::size_t n, std::size_t ldv, T* d_A,
    typename gpu_backend::handle_utils<Backend>::info_t* d_info,
    typename gpu_backend::handle_utils<Backend>::stream_t stream, T* workspace = nullptr,
    std::size_t lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;
    const T one = T(1);
    const ::chase::Base<T> alpha = ::chase::Base<T>(1.0);
    const ::chase::Base<T> beta = ::chase::Base<T>(0.0);
    const auto n_int = static_cast<typename util::info_t>(n);
    const auto m_int = static_cast<typename util::info_t>(m);
    const auto ldv_int = static_cast<typename util::info_t>(ldv);
    const auto lda_int = n_int;

    util::set_stream(blas_handle, stream);
    util::set_solver_stream(solver_handle, stream);
    util::check_blas(gpu_backend::blas<Backend>::syherk(
        blas_handle, util::fill_upper(), util::template herk_transpose<T>(),
        n_int, m_int, &alpha, d_V, ldv_int, &beta, d_A, lda_int));

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, n, d_A, n, d_info, workspace,
        static_cast<int>(lwork));

    typename util::info_t info = 1;
    gpu_backend::memcpy_d2h<Backend, typename util::info_t>(&info, d_info, 1);
    if (info != 0)
    {
        return static_cast<int>(info);
    }

    util::check_blas(gpu_backend::blas<Backend>::trsm(
        blas_handle, util::side_right(), util::fill_upper(), util::op_none(),
        util::diag_non_unit(), m_int, n_int, &one, d_A, lda_int, d_V,
        ldv_int));

    return 0;
}

template <typename Backend, typename T>
int cholqr1_blas(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle, T* d_V,
    std::size_t m, std::size_t n, std::size_t ldv, T* d_A,
    typename gpu_backend::handle_utils<Backend>::info_t* d_info,
    typename gpu_backend::handle_utils<Backend>::stream_t stream =
        typename gpu_backend::handle_utils<Backend>::stream_t{},
    T* workspace = nullptr, std::size_t lwork = 0)
{
    return cholqr_pass<Backend, T>(blas_handle, solver_handle, d_V, m, n, ldv,
                                   d_A, d_info, stream, workspace, lwork);
}

template <typename Backend, typename T>
int cholqr2_blas(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle, T* d_V,
    std::size_t m, std::size_t n, std::size_t ldv, T* d_A,
    typename gpu_backend::handle_utils<Backend>::info_t* d_info,
    typename gpu_backend::handle_utils<Backend>::stream_t stream =
        typename gpu_backend::handle_utils<Backend>::stream_t{},
    T* workspace = nullptr, std::size_t lwork = 0)
{
    int info = cholqr_pass<Backend, T>(blas_handle, solver_handle, d_V, m, n,
                                       ldv, d_A, d_info, stream, workspace,
                                       lwork);
    if (info != 0)
    {
        return info;
    }
    return cholqr_pass<Backend, T>(blas_handle, solver_handle, d_V, m, n, ldv,
                                   d_A, d_info, stream, workspace, lwork);
}

template <typename Backend, typename T>
void householder_qr_blas(
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle, T* d_V,
    std::size_t m, std::size_t n, std::size_t ldv, T* d_tau,
    typename gpu_backend::handle_utils<Backend>::info_t* devInfo,
    typename gpu_backend::handle_utils<Backend>::stream_t stream =
        typename gpu_backend::handle_utils<Backend>::stream_t{},
    T* workspace = nullptr, std::size_t lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;
    using info_t = typename util::info_t;

    info_t* info = devInfo;
    bool owns_info = false;
    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        if (info == nullptr)
        {
            gpu_backend::device_malloc<Backend, info_t>(&info, 1);
            owns_info = true;
        }
    }

    T* qr_workspace = workspace;
    std::size_t qr_lwork = lwork;
    bool owns_qr_workspace = false;
    ensure_householder_workspace<Backend, T>(
        solver_handle, m, n, ldv, d_V, d_tau, qr_workspace, qr_lwork,
        owns_qr_workspace);

    util::set_solver_stream(solver_handle, stream);

    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        util::check_solver(gpu_backend::solver<Backend>::template geqrf<T>(
            solver_handle, m, n, d_V, ldv, d_tau, qr_workspace, qr_lwork,
            info));
        util::check_solver(gpu_backend::solver<Backend>::template gqr<T>(
            solver_handle, m, n, n, d_V, ldv, d_tau, qr_workspace, qr_lwork,
            info));
    }
    else if (qr_workspace != nullptr && qr_lwork > 0)
    {
        util::check_solver(gpu_backend::solver<Backend>::template geqrf<T>(
            solver_handle, m, n, d_V, ldv, d_tau, qr_workspace, qr_lwork));
        util::check_solver(gpu_backend::solver<Backend>::template gqr<T>(
            solver_handle, m, n, n, d_V, ldv, d_tau, qr_workspace, qr_lwork));
    }
    else
    {
        util::check_solver(gpu_backend::solver<Backend>::template geqrf<T>(
            solver_handle, m, n, d_V, ldv, d_tau));
        util::check_solver(gpu_backend::solver<Backend>::template gqr<T>(
            solver_handle, m, n, n, d_V, ldv, d_tau));
    }

    if (owns_qr_workspace)
    {
        gpu_backend::device_free<Backend>(qr_workspace);
    }
    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        if (owns_info)
        {
            gpu_backend::device_free<Backend>(info);
        }
    }
}


template <typename Backend, typename T>
int cholQR1(typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
            typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
            ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V,
            ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr,
            typename gpu_backend::handle_utils<Backend>::info_t* external_devInfo =
                nullptr,
            T* workspace = nullptr, int lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;
    using info_t = typename util::info_t;

    bool owns_A = false;
    bool owns_devInfo = false;

    if (A == nullptr)
    {
        A = new ::chase::matrix::Matrix<T, ::chase::platform::GPU>(V.cols(), V.cols());
        owns_A = true;
    }

    info_t* devInfo = external_devInfo;
    if (devInfo == nullptr)
    {
        gpu_backend::device_malloc<Backend, info_t>(&devInfo, 1);
        owns_devInfo = true;
    }

    typename util::stream_t stream{};
    util::get_stream(blas_handle, &stream);

    const int info = cholqr1_blas<Backend, T>(
        blas_handle, solver_handle, V.data(), V.rows(), V.cols(), V.ld(),
        A->data(), devInfo, stream, workspace, static_cast<std::size_t>(lwork));

    if (owns_devInfo)
    {
        gpu_backend::device_free<Backend>(devInfo);
    }
    if (owns_A)
    {
        delete A;
    }

    return info;
}

template <typename Backend, typename T>
int cholQR2(typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
            typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
            ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V,
            ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr,
            typename gpu_backend::handle_utils<Backend>::info_t* external_devInfo =
                nullptr,
            T* workspace = nullptr, int lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;
    using info_t = typename util::info_t;

    bool owns_A = false;
    bool owns_devInfo = false;

    if (A == nullptr)
    {
        A = new ::chase::matrix::Matrix<T, ::chase::platform::GPU>(V.cols(), V.cols());
        owns_A = true;
    }

    info_t* devInfo = external_devInfo;
    if (devInfo == nullptr)
    {
        gpu_backend::device_malloc<Backend, info_t>(&devInfo, 1);
        owns_devInfo = true;
    }

    typename util::stream_t stream{};
    util::get_stream(blas_handle, &stream);

    const int info = cholqr2_blas<Backend, T>(
        blas_handle, solver_handle, V.data(), V.rows(), V.cols(), V.ld(),
        A->data(), devInfo, stream, workspace, static_cast<std::size_t>(lwork));

    if (owns_devInfo)
    {
        gpu_backend::device_free<Backend>(devInfo);
    }
    if (owns_A)
    {
        delete A;
    }

    return info;
}

template <typename Backend, typename T>
void houseHoulderQR(
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V, T* d_tau,
    typename gpu_backend::handle_utils<Backend>::info_t* devInfo = nullptr,
    T* workspace = nullptr, int lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;

    typename util::stream_t stream{};
    util::get_solver_stream(solver_handle, &stream);
    householder_qr_blas<Backend, T>(
        solver_handle, V.data(), V.rows(), V.cols(), V.ld(), d_tau, devInfo,
        stream, workspace, static_cast<std::size_t>(lwork));
}

template <typename Backend, typename T>
int shiftedcholQR2(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A = nullptr,
    typename gpu_backend::handle_utils<Backend>::info_t* external_devInfo = nullptr,
    T* workspace = nullptr, int lwork = 0)
{
    using util = gpu_backend::handle_utils<Backend>;
    using info_t = typename util::info_t;

    bool owns_A = false;
    bool owns_devInfo = false;

    if (A == nullptr)
    {
        A = new ::chase::matrix::Matrix<T, ::chase::platform::GPU>(V.cols(), V.cols());
        owns_A = true;
    }

    info_t* devInfo = external_devInfo;
    if (devInfo == nullptr)
    {
        gpu_backend::device_malloc<Backend, info_t>(&devInfo, 1);
        owns_devInfo = true;
    }

    typename util::stream_t stream{};
    util::get_stream(blas_handle, &stream);

    const T one = T(1);
    const ::chase::Base<T> alpha = ::chase::Base<T>(1.0);
    const ::chase::Base<T> beta = ::chase::Base<T>(0.0);
    const auto n_int = static_cast<info_t>(V.cols());
    const auto m_int = static_cast<info_t>(V.rows());
    const auto ldv_int = static_cast<info_t>(V.ld());
    const auto lda_int = n_int;
    info_t info = 1;

    util::set_stream(blas_handle, stream);
    util::set_solver_stream(solver_handle, stream);
    util::check_blas(gpu_backend::blas<Backend>::syherk(
        blas_handle, util::fill_upper(), util::template herk_transpose<T>(),
        n_int, m_int, &alpha, V.data(), ldv_int, &beta, A->data(), lda_int));

    ::chase::Base<T>* d_nrmf = nullptr;
    gpu_backend::device_malloc<Backend, ::chase::Base<T>>(&d_nrmf, 1);
    const std::size_t n = std::min(A->rows(), A->cols());
    gpu_device::kernels<Backend>::abs_trace(A->data(), d_nrmf, n, A->ld(),
                                            stream);
    gpu_device::kernels<Backend>::shift_matrix_from_device_shift(
        A->data(), n, A->ld(), d_nrmf, V.rows(), stream);
    gpu_backend::device_free<Backend>(d_nrmf);

    detail::potrf_with_optional_workspace<Backend, T>(
        blas_handle, solver_handle, V.cols(), A->data(), V.cols(), devInfo,
        workspace, lwork);
    gpu_backend::memcpy_d2h<Backend, info_t>(&info, devInfo, 1);

    if (info == 0)
    {
        util::check_blas(gpu_backend::blas<Backend>::trsm(
            blas_handle, util::side_right(), util::fill_upper(),
            util::op_none(), util::diag_non_unit(), m_int, n_int, &one,
            A->data(), lda_int, V.data(), ldv_int));

        util::check_blas(gpu_backend::blas<Backend>::syherk(
            blas_handle, util::fill_upper(), util::template herk_transpose<T>(),
            n_int, m_int, &alpha, V.data(), ldv_int, &beta, A->data(),
            lda_int));

        detail::potrf_with_optional_workspace<Backend, T>(
            blas_handle, solver_handle, V.cols(), A->data(), V.cols(), devInfo,
            workspace, lwork);
        gpu_backend::memcpy_d2h<Backend, info_t>(&info, devInfo, 1);

        if (info == 0)
        {
            util::check_blas(gpu_backend::blas<Backend>::trsm(
                blas_handle, util::side_right(), util::fill_upper(),
                util::op_none(), util::diag_non_unit(), m_int, n_int, &one,
                A->data(), lda_int, V.data(), ldv_int));

            util::check_blas(gpu_backend::blas<Backend>::syherk(
                blas_handle, util::fill_upper(),
                util::template herk_transpose<T>(), n_int, m_int, &alpha,
                V.data(), ldv_int, &beta, A->data(), lda_int));

            detail::potrf_with_optional_workspace<Backend, T>(
                blas_handle, solver_handle, V.cols(), A->data(), V.cols(),
                devInfo, workspace, lwork);

            util::check_blas(gpu_backend::blas<Backend>::trsm(
                blas_handle, util::side_right(), util::fill_upper(),
                util::op_none(), util::diag_non_unit(), m_int, n_int, &one,
                A->data(), lda_int, V.data(), ldv_int));
        }
    }

    if (owns_devInfo)
    {
        gpu_backend::device_free<Backend>(devInfo);
    }
    if (owns_A)
    {
        delete A;
    }

    return static_cast<int>(info);
}

} // namespace chase::linalg::internal::gpu_serial::impl
