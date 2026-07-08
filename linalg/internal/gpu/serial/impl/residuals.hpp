// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/matrix/matrix.hpp"
#include <cstddef>

namespace chase::linalg::internal::gpu_serial::impl
{

/**
 * @brief Compute eigenvector residuals on device pointers (backend-neutral).
 */
template <typename Backend, typename T>
void residuals_blas(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle, T* d_H,
    std::size_t ldh, T* d_V, std::size_t ldv, ::chase::Base<T>* d_ritzv,
    ::chase::Base<T>* d_resids, std::size_t m, std::size_t n, std::size_t k,
    T* d_work, std::size_t ldw,
    typename gpu_backend::handle_utils<Backend>::stream_t stream =
        typename gpu_backend::handle_utils<Backend>::stream_t{})
{
    using util = gpu_backend::handle_utils<Backend>;
    using device_ops = gpu_device::kernels<Backend>;

    const T alpha = T(1);
    const T beta = T(0);
    const auto m_int = static_cast<typename util::info_t>(m);
    const auto n_int = static_cast<typename util::info_t>(n);
    const auto k_int = static_cast<typename util::info_t>(k);
    const auto ldh_int = static_cast<typename util::info_t>(ldh);
    const auto ldv_int = static_cast<typename util::info_t>(ldv);
    const auto ldw_int = static_cast<typename util::info_t>(ldw);

    util::set_stream(blas_handle, stream);
    util::check_blas(gpu_backend::blas<Backend>::gemm(
        blas_handle, util::op_none(), util::op_none(), m_int, n_int, k_int,
        &alpha, d_H, ldh_int, d_V, ldv_int, &beta, d_work, ldw_int));

    device_ops::template residual_gpu<T>(m, n, d_work, ldw, d_V, ldv, d_ritzv,
                                         d_resids, true, stream);
}


/**
 * @brief Compute eigenvector residuals on Matrix<T, GPU>.
 */
template <typename Backend, typename T>
void residuals(
    typename gpu_backend::handle_utils<Backend>::blas_handle blas_handle,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* H,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>& V1,
    ::chase::Base<T>* d_ritzv, ::chase::Base<T>* d_resids, std::size_t offset,
    std::size_t subSize,
    ::chase::matrix::Matrix<T, ::chase::platform::GPU>* V2 = nullptr)
{
    using util = gpu_backend::handle_utils<Backend>;

    bool owns_V2 = false;
    if (V2 == nullptr)
    {
        V2 = new ::chase::matrix::Matrix<T, ::chase::platform::GPU>(V1.rows(),
                                                                    V1.cols());
        owns_V2 = true;
    }

    typename util::stream_t stream{};
    util::get_stream(blas_handle, &stream);

    residuals_blas<Backend, T>(
        blas_handle, H->data(), H->ld(), V1.data() + offset * V1.ld(), V1.ld(),
        d_ritzv + offset, d_resids + offset, H->rows(), subSize, H->cols(),
        V2->data() + offset * V2->ld(), V2->ld(), stream);

    if (owns_V2)
    {
        delete V2;
    }
}

} // namespace chase::linalg::internal::gpu_serial::impl
