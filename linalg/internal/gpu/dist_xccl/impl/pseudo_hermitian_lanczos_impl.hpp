// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/typeTraits.hpp"
#include "external/cpu/lapackpp/lapackpp.hpp"
#include "grid/nccl_utils.hpp"
#include "linalg/internal/gpu/dist_xccl/impl/lanczos_detail.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/kernels/shared/lanczos_workspace.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_xccl
{

template <typename Backend, typename BlasHandle, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void pseudo_hermitian_lanczos_impl(
    BlasHandle blas_handle, std::size_t M, std::size_t numvec, MatrixType& H,
    InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* Tau,
    chase::Base<typename MatrixType::value_type>* ritzV)
{
    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using kernels = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    detail::validate_lanczos_inputs(H, V);

    typename util::stream_type original_stream = util::get_blas_stream(blas_handle);
    typename util::stream_type stream = nullptr;
    util::set_blas_stream(blas_handle, stream);

    ncclComm_t comm = H.getMpiGrid()->get_nccl_col_comm();
    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);
    std::vector<RealT> d_tmp(M * numvec);
    std::vector<RealT> e_tmp(M * numvec);

    T One = T(1.0);
    T Zero = T(0.0);
    const std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto Sv = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    gpu_device::lanczos_pseudo_hermitian_workspace<Backend, T> ws(M, numvec, stream,
                                                                 true);
    ws.zero_scalars(M, numvec, stream);

    kernels::lacpy('A', v_1.l_rows(), numvec, V.l_data(), V.l_ld(),
                   v_1.l_data(), v_1.l_ld(), &stream);

    CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
        blas_handle, &One, H, v_1, &Zero, v_w);
    v_w.redistributeImplAsync(&v_2, &stream);
    kernels::lacpy('A', Sv.l_rows(), numvec, v_2.l_data(), v_2.l_ld(),
                   Sv.l_data(), Sv.l_ld(), &stream);
    CollectiveKernelNamespace::flipLowerHalfMatrixSign(Sv, stream);

    kernels::batched_dot(v_1.l_data(), Sv.l_data(), ws.d_beta,
                         static_cast<int>(v_1.l_rows()),
                         static_cast<int>(numvec),
                         static_cast<int>(v_1.l_ld()), false, &stream);
    kernels::get_real(ws.d_beta, ws.d_real_beta, static_cast<int>(numvec), &stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_beta, ws.d_real_beta, numvec, ncclSum, comm, &stream));
    kernels::sqrt(ws.d_real_beta, static_cast<int>(numvec), &stream);
    kernels::real_reciprocal(ws.d_real_beta, ws.d_real_beta_prev,
                             static_cast<int>(numvec), &stream);
    kernels::copy_real_reciprocal_to_t(ws.d_real_beta, ws.d_beta,
                                       static_cast<int>(numvec), &stream);
    kernels::batched_scale(ws.d_beta, v_1.l_data(),
                           static_cast<int>(v_1.l_rows()),
                           static_cast<int>(numvec),
                           static_cast<int>(v_1.l_ld()), &stream);
    kernels::batched_scale(ws.d_beta, v_2.l_data(),
                           static_cast<int>(v_2.l_rows()),
                           static_cast<int>(numvec),
                           static_cast<int>(v_2.l_ld()), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (std::size_t i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d_async<Backend>(V.l_data() + k * V.l_ld(),
                                v_1.l_data() + i * v_1.l_ld(), v_1.l_rows(),
                                stream);
        }

        kernels::batched_dot(v_2.l_data(), Sv.l_data(), ws.d_alpha,
                             static_cast<int>(v_2.l_rows()),
                             static_cast<int>(numvec),
                             static_cast<int>(v_2.l_ld()), false, &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec),
                          &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_alpha, ws.d_real_alpha, numvec, ncclSum, comm, &stream));
        kernels::copy_real_to_t(ws.d_real_alpha, ws.d_alpha, static_cast<int>(numvec),
                                &stream);
        kernels::scale_by_real_negate(ws.d_alpha, ws.d_real_beta_prev,
                                      static_cast<int>(numvec), &stream);
        kernels::batched_axpy(ws.d_alpha, v_1.l_data(), v_2.l_data(),
                              static_cast<int>(v_1.l_rows()),
                              static_cast<int>(numvec),
                              static_cast<int>(v_1.l_ld()), &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec),
                          &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k * numvec, ws.d_real_alpha,
                                numvec, stream);

        kernels::copy_real_negate_to_t(ws.d_real_beta, ws.d_beta,
                                       static_cast<int>(numvec), &stream);
        kernels::batched_axpy(ws.d_beta, v_0.l_data(), v_2.l_data(),
                              static_cast<int>(v_0.l_rows()),
                              static_cast<int>(numvec),
                              static_cast<int>(v_0.l_ld()), &stream);

        v_1.swap(v_0);
        v_1.swap(v_2);

        CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);
        v_w.redistributeImplAsync(&v_2, &stream);
        kernels::lacpy('A', Sv.l_rows(), numvec, v_2.l_data(), v_2.l_ld(),
                       Sv.l_data(), Sv.l_ld(), &stream);
        CollectiveKernelNamespace::flipLowerHalfMatrixSign(Sv, stream);

        kernels::batched_dot(v_1.l_data(), Sv.l_data(), ws.d_beta,
                             static_cast<int>(v_1.l_rows()),
                             static_cast<int>(numvec),
                             static_cast<int>(v_1.l_ld()), false, &stream);
        kernels::get_real(ws.d_beta, ws.d_real_beta, static_cast<int>(numvec),
                          &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_beta, ws.d_real_beta, numvec, ncclSum, comm, &stream));
        kernels::sqrt(ws.d_real_beta, static_cast<int>(numvec), &stream);
        kernels::real_reciprocal(ws.d_real_beta, ws.d_real_beta_prev,
                                 static_cast<int>(numvec), &stream);
        kernels::copy_real_reciprocal_to_t(ws.d_real_beta, ws.d_beta,
                                           static_cast<int>(numvec), &stream);
        kernels::batched_scale_two(ws.d_beta, v_1.l_data(), v_2.l_data(),
                                   static_cast<int>(v_1.l_rows()),
                                   static_cast<int>(numvec),
                                   static_cast<int>(v_1.l_ld()), &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.e_d + k * numvec, ws.d_real_beta,
                                numvec, stream);
    }

    {
        const std::size_t k = M - 1;
        for (std::size_t i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d_async<Backend>(V.l_data() + k * V.l_ld(),
                                v_1.l_data() + i * v_1.l_ld(), v_1.l_rows(),
                                stream);
        }

        kernels::batched_dot(v_2.l_data(), Sv.l_data(), ws.d_alpha,
                             static_cast<int>(v_2.l_rows()),
                             static_cast<int>(numvec),
                             static_cast<int>(v_2.l_ld()), false, &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec),
                          &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_alpha, ws.d_real_alpha, numvec, ncclSum, comm, &stream));
        kernels::copy_real_to_t(ws.d_real_alpha, ws.d_alpha, static_cast<int>(numvec),
                                &stream);
        kernels::scale_by_real_negate(ws.d_alpha, ws.d_real_beta_prev,
                                      static_cast<int>(numvec), &stream);
        kernels::batched_axpy(ws.d_alpha, v_1.l_data(), v_2.l_data(),
                              static_cast<int>(v_1.l_rows()),
                              static_cast<int>(numvec),
                              static_cast<int>(v_1.l_ld()), &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec),
                          &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k * numvec, ws.d_real_alpha,
                                numvec, stream);
    }

    kernels::lacpy('A', v_1.l_rows(), numvec, v_1.l_data(), v_1.l_ld(),
                   V.l_data(), V.l_ld(), &stream);
    gpu_backend::memcpy_async_device_to_host<Backend>(d.data(), ws.d_d, M * numvec, stream);
    gpu_backend::memcpy_async_device_to_host<Backend>(e.data(), ws.e_d, M * numvec, stream);
    gpu_backend::stream_synchronize<Backend>(stream);

    util::set_blas_stream(blas_handle, original_stream);

    for (std::size_t i = 0; i < numvec; i++)
    {
        for (std::size_t k = 0; k < M; k++)
        {
            d_tmp[i * M + k] = d[k * numvec + i];
            e_tmp[i * M + k] = e[k * numvec + i];
        }
    }
    d = std::move(d_tmp);
    e = std::move(e_tmp);

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    lapack_logical tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (std::size_t i = 0; i < numvec; i++)
    {
        lapackpp::t_stemr<RealT>(LAPACK_COL_MAJOR, 'V', 'A', M,
                                 d.data() + i * M, e.data() + i * M, ul, ll,
                                 vl, vu, &notneeded_m, ritzv + M * i, ritzV,
                                 M, M, isuppz.data(), &tryrac);
        for (std::size_t k = 0; k < M; ++k)
        {
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
        }
    }

    *upperb = ritzv[M - 1];
}

template <typename Backend, typename BlasHandle, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void pseudo_hermitian_lanczos_impl(
    BlasHandle blas_handle, std::size_t M, MatrixType& H,
    InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb)
{
    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using kernels = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    detail::validate_lanczos_inputs(H, V);

    typename util::stream_type original_stream = util::get_blas_stream(blas_handle);
    typename util::stream_type stream = nullptr;
    util::set_blas_stream(blas_handle, stream);

    ncclComm_t comm = H.getMpiGrid()->get_nccl_col_comm();
    std::vector<RealT> d(M);
    std::vector<RealT> e(M);

    T One = T(1.0);
    T Zero = T(0.0);
    const std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto Sv = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    gpu_device::lanczos_pseudo_hermitian_workspace<Backend, T> ws(M, 1, stream, true);
    ws.zero_scalars(M, 1, stream);

    kernels::lacpy('A', v_1.l_rows(), 1, V.l_data(), V.l_ld(), v_1.l_data(),
                   v_1.l_ld(), &stream);
    CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
        blas_handle, &One, H, v_1, &Zero, v_w);
    v_w.redistributeImplAsync(&v_2, &stream);
    kernels::lacpy('A', Sv.l_rows(), 1, v_2.l_data(), v_2.l_ld(),
                   Sv.l_data(), Sv.l_ld(), &stream);
    CollectiveKernelNamespace::flipLowerHalfMatrixSign(Sv, stream);
    kernels::batched_dot(v_1.l_data(), Sv.l_data(), ws.d_beta,
                         static_cast<int>(v_1.l_rows()), 1,
                         static_cast<int>(v_1.l_ld()), false, &stream);
    kernels::get_real(ws.d_beta, ws.d_real_beta, 1, &stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_beta, ws.d_real_beta, 1, ncclSum, comm, &stream));
    kernels::sqrt(ws.d_real_beta, 1, &stream);
    kernels::real_reciprocal(ws.d_real_beta, ws.d_real_beta_prev, 1, &stream);
    kernels::copy_real_reciprocal_to_t(ws.d_real_beta, ws.d_beta, 1, &stream);
    kernels::batched_scale(ws.d_beta, v_1.l_data(),
                           static_cast<int>(v_1.l_rows()), 1,
                           static_cast<int>(v_1.l_ld()), &stream);
    kernels::batched_scale(ws.d_beta, v_2.l_data(),
                           static_cast<int>(v_2.l_rows()), 1,
                           static_cast<int>(v_2.l_ld()), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        kernels::batched_dot(v_2.l_data(), Sv.l_data(), ws.d_alpha,
                             static_cast<int>(v_2.l_rows()), 1,
                             static_cast<int>(v_2.l_ld()), false, &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, 1, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_alpha, ws.d_real_alpha, 1, ncclSum, comm, &stream));
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k, ws.d_real_alpha, 1, stream);
        kernels::copy_real_to_t(ws.d_real_alpha, ws.d_alpha, 1, &stream);
        kernels::scale_by_real_negate(ws.d_alpha, ws.d_real_beta_prev, 1, &stream);
        kernels::batched_axpy(ws.d_alpha, v_1.l_data(), v_2.l_data(),
                              static_cast<int>(v_1.l_rows()), 1,
                              static_cast<int>(v_1.l_ld()), &stream);
        kernels::copy_real_negate_to_t(ws.d_real_beta, ws.d_beta, 1, &stream);
        kernels::batched_axpy(ws.d_beta, v_0.l_data(), v_2.l_data(),
                              static_cast<int>(v_0.l_rows()), 1,
                              static_cast<int>(v_0.l_ld()), &stream);

        v_1.swap(v_0);
        v_1.swap(v_2);
        CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);
        v_w.redistributeImplAsync(&v_2, &stream);
        kernels::lacpy('A', Sv.l_rows(), 1, v_2.l_data(), v_2.l_ld(),
                       Sv.l_data(), Sv.l_ld(), &stream);
        CollectiveKernelNamespace::flipLowerHalfMatrixSign(Sv, stream);
        kernels::batched_dot(v_1.l_data(), Sv.l_data(), ws.d_beta,
                             static_cast<int>(v_1.l_rows()), 1,
                             static_cast<int>(v_1.l_ld()), false, &stream);
        kernels::get_real(ws.d_beta, ws.d_real_beta, 1, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_beta, ws.d_real_beta, 1, ncclSum, comm, &stream));
        kernels::sqrt(ws.d_real_beta, 1, &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.e_d + k, ws.d_real_beta, 1, stream);
        kernels::real_reciprocal(ws.d_real_beta, ws.d_real_beta_prev, 1, &stream);
        kernels::copy_real_reciprocal_to_t(ws.d_real_beta, ws.d_beta, 1, &stream);
        kernels::batched_scale_two(ws.d_beta, v_1.l_data(), v_2.l_data(),
                                   static_cast<int>(v_1.l_rows()), 1,
                                   static_cast<int>(v_1.l_ld()), &stream);
    }

    {
        const std::size_t k = M - 1;
        gpu_backend::memcpy_d2d_async<Backend>(V.l_data() + k * V.l_ld(), v_1.l_data(),
                            v_1.l_rows(), stream);
        kernels::batched_dot(v_2.l_data(), Sv.l_data(), ws.d_alpha,
                             static_cast<int>(v_2.l_rows()), 1,
                             static_cast<int>(v_2.l_ld()), false, &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, 1, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_alpha, ws.d_real_alpha, 1, ncclSum, comm, &stream));
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k, ws.d_real_alpha, 1, stream);
        kernels::copy_real_to_t(ws.d_real_alpha, ws.d_alpha, 1, &stream);
        kernels::scale_by_real_negate(ws.d_alpha, ws.d_real_beta_prev, 1, &stream);
        kernels::batched_axpy(ws.d_alpha, v_1.l_data(), v_2.l_data(),
                              static_cast<int>(v_1.l_rows()), 1,
                              static_cast<int>(v_1.l_ld()), &stream);
    }

    gpu_backend::memcpy_async_device_to_host<Backend>(d.data(), ws.d_d, M, stream);
    gpu_backend::memcpy_async_device_to_host<Backend>(e.data(), ws.e_d, M, stream);
    gpu_backend::stream_synchronize<Backend>(stream);
    for (std::size_t k = 0; k < M; k++)
    {
        d[k] = -d[k];
    }

    util::set_blas_stream(blas_handle, original_stream);

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    lapack_logical tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> ritzv(M);

    lapackpp::t_stemr<RealT>(LAPACK_COL_MAJOR, 'N', 'A', M, d.data(),
                             e.data(), ul, ll, vl, vu, &notneeded_m,
                             ritzv.data(), static_cast<RealT*>(nullptr), M, M,
                             isuppz.data(), &tryrac);
    *upperb = ritzv[M - 1];
}

} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
