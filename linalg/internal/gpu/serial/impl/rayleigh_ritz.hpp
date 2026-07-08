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
#include "linalg/matrix/matrix.hpp"
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace chase::linalg::internal::gpu_serial::impl
{

template <typename Backend, typename T>
void ensure_heevd_workspace(
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    std::size_t subSize, T* d_A, ::chase::Base<T>* d_ritzv_block,
    T*& workspace, std::size_t& lwork, bool& owns_workspace)
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
            gpu_backend::solver<Backend>::template heevd_buffer_size<T>(
                solver_handle, util::eig_original(), util::fill_lower(),
                subSize, d_A, subSize, d_ritzv_block, &lwork_int));
        gpu_backend::device_malloc<Backend, T>(
            &workspace, static_cast<std::size_t>(lwork_int));
        lwork = static_cast<std::size_t>(lwork_int);
        owns_workspace = true;
    }
}

/**
 * @brief Standard Hermitian Rayleigh–Ritz on device pointers (backend-neutral).
 */
template <typename Backend, typename T>
void rayleigh_ritz_blas(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle, T* d_H,
    std::size_t N, std::size_t ldh, T* d_V1, T* d_V2, std::size_t ldv,
    ::chase::Base<T>* d_ritzv, ::chase::Base<T>* h_ritzv, std::size_t offset,
    std::size_t subSize, typename gpu_backend::handle_utils<Backend>::info_t* d_info,
    T* d_A, typename gpu_backend::handle_utils<Backend>::stream_t stream =
                  typename gpu_backend::handle_utils<Backend>::stream_t{},
    T* workspace = nullptr, std::size_t lwork = 0,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* A_matrix = nullptr,
    ::chase::matrix::Matrix<::chase::Base<T>, ::chase::platform::GPU>*
        ritzv_matrix = nullptr)
{
    using util = gpu_backend::handle_utils<Backend>;
    using Scalar = ::chase::Base<T>;
    using info_t = typename util::info_t;

    const T one = T(1);
    const T zero = T(0);
    const auto N_int = static_cast<info_t>(N);
    const auto subSize_int = static_cast<info_t>(subSize);
    const auto ldh_int = static_cast<info_t>(ldh);
    const auto ldv_int = static_cast<info_t>(ldv);
    const auto lda_int = subSize_int;
    T* d_V1_block = d_V1 + offset * ldv;
    T* d_V2_block = d_V2 + offset * ldv;
    Scalar* d_ritzv_block = d_ritzv + offset;

    T* heevd_workspace = workspace;
    std::size_t heevd_lwork = lwork;
    bool owns_heevd_workspace = false;
    ensure_heevd_workspace<Backend, T>(solver_handle, subSize, d_A,
                                       d_ritzv_block, heevd_workspace,
                                       heevd_lwork, owns_heevd_workspace);

    util::set_stream(blas_handle, stream);
    util::set_solver_stream(solver_handle, stream);

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_conj_transpose(), util::op_none(), N_int,
        subSize_int, N_int, &one, d_H, ldh_int, d_V1_block, ldv_int, &zero,
        d_V2_block, ldv_int));

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_conj_transpose(), util::op_none(), subSize_int,
        subSize_int, N_int, &one, d_V2_block, ldv_int, d_V1_block, ldv_int,
        &zero, d_A, lda_int));

    bool heevd_done = false;
#ifdef RR_DOUBLE_PRECISION
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, float>)
    {
        if (A_matrix != nullptr && ritzv_matrix != nullptr)
        {
            typename util::stream_t* stream_ptr = &stream;
            if (A_matrix->isDoublePrecisionEnabled())
            {
                A_matrix->copyToSubBlockAsync(0, subSize * subSize, stream_ptr);
            }
            else
            {
                A_matrix->enableDoublePrecisionAsync(stream_ptr);
            }
            if (!ritzv_matrix->isDoublePrecisionEnabled())
            {
                ritzv_matrix->enableDoublePrecisionAsync(stream_ptr);
            }

            auto* A_d = A_matrix->getDoublePrecisionMatrix();
            auto* ritzv_d = ritzv_matrix->getDoublePrecisionMatrix();
            using DT = typename chase::ToDoublePrecisionTrait<T>::Type;
            using DScalar = chase::Base<DT>;

            if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
            {
                DT* workspace_d = nullptr;
                gpu_backend::device_malloc<Backend, DT>(&workspace_d, heevd_lwork);
                util::check_solver(
                    gpu_backend::solver<Backend>::template heevd<DT>(
                        solver_handle, util::eig_original(),
                        util::fill_lower(), subSize, A_d->data(), subSize,
                        ritzv_d->data() + offset, workspace_d, heevd_lwork,
                        d_info));
                gpu_backend::device_free<Backend>(workspace_d);
            }
            else
            {
                DScalar* d_E = nullptr;
                gpu_backend::device_malloc<Backend, DScalar>(&d_E, subSize);
                util::check_solver(
                    gpu_backend::solver<Backend>::template heevd<DT>(
                        solver_handle, util::eig_original(),
                        util::fill_lower(), subSize, A_d->data(), subSize,
                        ritzv_d->data() + offset, d_E, d_info));
                gpu_backend::device_free<Backend>(d_E);
            }

            ritzv_matrix->disableDoublePrecisionAsync(true, stream_ptr);
            A_matrix->copyBackSubBlockAsync(0, subSize * subSize, stream_ptr);
            heevd_done = true;
        }
    }
#endif

    if (!heevd_done)
    {
        if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
        {
            util::check_solver(gpu_backend::solver<Backend>::template heevd<T>(
                solver_handle, util::eig_original(), util::fill_lower(),
                subSize, d_A, subSize, d_ritzv_block, heevd_workspace,
                heevd_lwork, d_info));
        }
        else if (heevd_workspace != nullptr && heevd_lwork > 0)
        {
            util::check_solver(gpu_backend::solver<Backend>::template heevd<T>(
                solver_handle, util::eig_original(), util::fill_lower(),
                subSize, d_A, subSize, d_ritzv_block, heevd_workspace,
                heevd_lwork, d_info));
        }
        else
        {
            Scalar* d_E = nullptr;
            gpu_backend::device_malloc<Backend, Scalar>(&d_E, subSize);

            util::check_solver(gpu_backend::solver<Backend>::template heevd<T>(
                solver_handle, util::eig_original(), util::fill_lower(),
                subSize, d_A, subSize, d_ritzv_block, d_E, d_info));

            gpu_backend::device_free<Backend>(d_E);
        }
    }

    if (owns_heevd_workspace)
    {
        gpu_backend::device_free<Backend>(heevd_workspace);
    }

    info_t info = -1;
    gpu_backend::memcpy_d2h<Backend, info_t>(&info, d_info, 1);
    if (info != 0)
    {
        throw std::runtime_error("HEEVD failed in gpu_serial::rayleigh_ritz_blas");
    }

    typename util::stream_t copy_stream{};
    typename util::event_t evt_ritz_ready{};
    const bool copy_host_ritz = (h_ritzv != nullptr);
    if (copy_host_ritz)
    {
        copy_stream = gpu_backend::create_nonblocking_stream<Backend>();
        evt_ritz_ready = gpu_backend::create_event<Backend>();
        gpu_backend::record_event<Backend>(evt_ritz_ready, stream);
        gpu_backend::stream_wait_event<Backend>(copy_stream, evt_ritz_ready);
        gpu_backend::memcpy_d2h_async<Backend, Scalar>(
            h_ritzv + offset, d_ritzv_block, subSize, copy_stream);
    }

    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_none(), util::op_none(), N_int, subSize_int,
        subSize_int, &one, d_V1_block, ldv_int, d_A, lda_int, &zero,
        d_V2_block, ldv_int));

    if (copy_host_ritz)
    {
        gpu_backend::stream_synchronize<Backend>(copy_stream);
        gpu_backend::destroy_event<Backend>(evt_ritz_ready);
        gpu_backend::destroy_stream<Backend>(copy_stream);
    }
}


/**
 * @brief Standard Hermitian Rayleigh–Ritz on Matrix<T, GPU>.
 */
template <typename Backend, typename T>
void rayleighRitz(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    typename gpu_backend::handle_utils<Backend>::solver_handle solver_handle,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* H,
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
        A = new ::chase::matrix::Matrix<T, ::chase::platform::GPU>(subSize,
                                                                   subSize);
        owns_A = true;
    }

    typename util::stream_t stream{};
    util::get_stream(blas_handle, &stream);

    rayleigh_ritz_blas<Backend, T>(
        blas_handle, solver_handle, H->data(), H->rows(), H->ld(), V1.data(),
        V2.data(), V1.ld(), ritzv.data(), ritzv.cpu_data(), offset, subSize,
        devInfo, A->data(), stream, workspace, static_cast<std::size_t>(lwork),
        A, &ritzv);

    if (owns_A)
    {
        delete A;
    }
}

} // namespace chase::linalg::internal::gpu_serial::impl
