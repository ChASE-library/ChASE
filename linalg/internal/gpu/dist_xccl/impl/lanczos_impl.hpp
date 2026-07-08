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
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
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
void lanczos_impl(BlasHandle blas_handle, std::size_t M, std::size_t numvec,
                  MatrixType& H, InputMultiVectorType& V,
                  chase::Base<typename MatrixType::value_type>* upperb,
                  chase::Base<typename MatrixType::value_type>* ritzv,
                  chase::Base<typename MatrixType::value_type>* Tau,
                  chase::Base<typename MatrixType::value_type>* ritzV);

template <typename Backend, typename BlasHandle, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_impl(BlasHandle blas_handle, std::size_t M, MatrixType& H,
                  InputMultiVectorType& V,
                  chase::Base<typename MatrixType::value_type>* upperb);

template <typename Backend, typename BlasHandle, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_dispatch_impl(
    BlasHandle blas_handle, std::size_t M, std::size_t numvec, MatrixType& H,
    InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* Tau,
    chase::Base<typename MatrixType::value_type>* ritzV)
{
    using T = typename MatrixType::value_type;

    if constexpr (std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                          T, chase::platform::GPU>>::value ||
                  std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                          T, chase::platform::GPU>>::value)
    {
        CollectiveKernelNamespace::pseudo_hermitian_lanczos(
            blas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
    }
    else
    {
        lanczos_impl<Backend, BlasHandle, CollectiveKernelNamespace>(
            blas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
    }
}

template <typename Backend, typename BlasHandle, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_dispatch_impl(BlasHandle blas_handle, std::size_t M, MatrixType& H,
                           InputMultiVectorType& V,
                           chase::Base<typename MatrixType::value_type>* upperb)
{
    using T = typename MatrixType::value_type;

    if constexpr (std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                          T, chase::platform::GPU>>::value ||
                  std::is_same<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
                          T, chase::platform::GPU>>::value)
    {
        CollectiveKernelNamespace::pseudo_hermitian_lanczos(blas_handle, M, H,
                                                            V, upperb);
    }
    else
    {
        lanczos_impl<Backend, BlasHandle, CollectiveKernelNamespace>(
            blas_handle, M, H, V, upperb);
    }
}

template <typename Backend, typename BlasHandle, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_impl(BlasHandle blas_handle, std::size_t M, std::size_t numvec,
                  MatrixType& H, InputMultiVectorType& V,
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
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    gpu_device::lanczos_hermitian_workspace<Backend, T> ws(M, numvec, stream, true);
    ws.zero_scalars(M, numvec, stream);
    gpu_backend::memset_zero_async<Backend>(v_0.l_data(), v_0.l_rows() * numvec, stream);

    kernels::lacpy('A', v_1.l_rows(), numvec, V.l_data(), V.l_ld(),
                   v_1.l_data(), v_1.l_ld(), &stream);

    kernels::batched_norm_squared(v_1.l_data(), ws.d_real_alpha,
                                  static_cast<int>(v_1.l_rows()),
                                  static_cast<int>(numvec),
                                  static_cast<int>(v_1.l_ld()), &stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_alpha, ws.d_real_alpha, numvec, ncclSum, comm, &stream));
    kernels::normalize(v_1.l_data(), ws.d_real_alpha,
                       static_cast<int>(v_1.l_rows()),
                       static_cast<int>(numvec), static_cast<int>(v_1.l_ld()),
                       &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        for (std::size_t i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d_async<Backend>(V.l_data() + k * V.l_ld(),
                                v_1.l_data() + i * v_1.l_ld(), v_1.l_rows(),
                                stream);
        }

        CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);
        v_w.redistributeImplAsync(&v_2, &stream);

        kernels::batched_dot(v_1.l_data(), v_2.l_data(), ws.d_alpha,
                             static_cast<int>(v_1.l_rows()),
                             static_cast<int>(numvec),
                             static_cast<int>(v_1.l_ld()), true, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_alpha, ws.d_alpha, numvec, ncclSum, comm, &stream));

        kernels::batched_axpy_then_negate(
            ws.d_alpha, v_1.l_data(), v_2.l_data(),
            static_cast<int>(v_1.l_rows()), static_cast<int>(numvec),
            static_cast<int>(v_1.l_ld()), &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec),
                          &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k * numvec, ws.d_real_alpha, numvec, stream);

        kernels::copy_real_negate_to_t(ws.d_beta_prev, ws.d_beta_neg,
                                       static_cast<int>(numvec), &stream);
        kernels::batched_axpy(ws.d_beta_neg, v_0.l_data(), v_2.l_data(),
                              static_cast<int>(v_0.l_rows()),
                              static_cast<int>(numvec),
                              static_cast<int>(v_0.l_ld()), &stream);

        kernels::batched_norm_squared(v_2.l_data(), ws.d_r_beta,
                                      static_cast<int>(v_2.l_rows()),
                                      static_cast<int>(numvec),
                                      static_cast<int>(v_2.l_ld()), &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_r_beta, ws.d_r_beta, numvec, ncclSum, comm, &stream));
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_beta_prev, ws.d_r_beta, numvec, stream);
        kernels::sqrt(ws.d_beta_prev, static_cast<int>(numvec), &stream);
        kernels::normalize(v_2.l_data(), ws.d_r_beta,
                           static_cast<int>(v_2.l_rows()),
                           static_cast<int>(numvec),
                           static_cast<int>(v_2.l_ld()), &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.e_d + k * numvec, ws.d_beta_prev, numvec, stream);

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    {
        const std::size_t k = M - 1;
        for (std::size_t i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d_async<Backend>(V.l_data() + k * V.l_ld(),
                                v_1.l_data() + i * v_1.l_ld(), v_1.l_rows(),
                                stream);
        }

        CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);
        v_w.redistributeImplAsync(&v_2, &stream);

        kernels::batched_dot(v_1.l_data(), v_2.l_data(), ws.d_alpha,
                             static_cast<int>(v_1.l_rows()),
                             static_cast<int>(numvec),
                             static_cast<int>(v_1.l_ld()), true, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_alpha, ws.d_alpha, numvec, ncclSum, comm, &stream));
        kernels::batched_axpy_then_negate(
            ws.d_alpha, v_1.l_data(), v_2.l_data(),
            static_cast<int>(v_1.l_rows()), static_cast<int>(numvec),
            static_cast<int>(v_1.l_ld()), &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, static_cast<int>(numvec),
                          &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k * numvec, ws.d_real_alpha, numvec, stream);
    }

    kernels::lacpy('A', v_1.l_rows(), numvec, v_1.l_data(), v_1.l_ld(),
                   V.l_data(), V.l_ld(), &stream);
    gpu_backend::memcpy_async_device_to_host<Backend>(d.data(), ws.d_d, M * numvec, stream);
    gpu_backend::memcpy_async_device_to_host<Backend>(e.data(), ws.e_d, M * numvec, stream);
    gpu_backend::stream_synchronize<Backend>(stream);

    util::set_blas_stream(blas_handle, original_stream);

    std::vector<RealT> r_beta_last(numvec);
    for (std::size_t i = 0; i < numvec; i++)
    {
        r_beta_last[i] = M > 1 ? e[(M - 2) * numvec + i] : RealT(0);
    }

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

    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(r_beta_last[0]);
    for (std::size_t i = 1; i < numvec; i++)
    {
        RealT candidate =
            std::max(std::abs(ritzv[i * M]), std::abs(ritzv[(i + 1) * M - 1])) +
            std::abs(r_beta_last[i]);
        *upperb = std::max(candidate, *upperb);
    }
}

template <typename Backend, typename BlasHandle, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_impl(BlasHandle blas_handle, std::size_t M, MatrixType& H,
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
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    gpu_device::lanczos_hermitian_workspace<Backend, T> ws(M, 1, stream, true);
    ws.zero_scalars(M, 1, stream);
    gpu_backend::memset_zero_async<Backend>(v_0.l_data(), v_0.l_rows(), stream);

    kernels::lacpy('A', v_1.l_rows(), 1, V.l_data(), V.l_ld(), v_1.l_data(),
                   v_1.l_ld(), &stream);

    kernels::batched_norm_squared(v_1.l_data(), ws.d_real_alpha,
                                  static_cast<int>(v_1.l_rows()), 1,
                                  static_cast<int>(v_1.l_ld()), &stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_real_alpha, ws.d_real_alpha, 1, ncclSum, comm, &stream));
    kernels::normalize(v_1.l_data(), ws.d_real_alpha,
                       static_cast<int>(v_1.l_rows()), 1,
                       static_cast<int>(v_1.l_ld()), &stream);

    for (std::size_t k = 0; k + 1 < M; ++k)
    {
        CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);
        v_w.redistributeImplAsync(&v_2, &stream);
        kernels::batched_dot(v_1.l_data(), v_2.l_data(), ws.d_alpha,
                             static_cast<int>(v_1.l_rows()), 1,
                             static_cast<int>(v_1.l_ld()), true, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_alpha, ws.d_alpha, 1, ncclSum, comm, &stream));
        kernels::batched_axpy_then_negate(
            ws.d_alpha, v_1.l_data(), v_2.l_data(),
            static_cast<int>(v_1.l_rows()), 1, static_cast<int>(v_1.l_ld()),
            &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, 1, &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k, ws.d_real_alpha, 1, stream);
        kernels::copy_real_negate_to_t(ws.d_beta_prev, ws.d_beta_neg, 1, &stream);
        kernels::batched_axpy(ws.d_beta_neg, v_0.l_data(), v_2.l_data(),
                              static_cast<int>(v_0.l_rows()), 1,
                              static_cast<int>(v_0.l_ld()), &stream);
        kernels::batched_norm_squared(v_2.l_data(), ws.d_r_beta,
                                      static_cast<int>(v_2.l_rows()), 1,
                                      static_cast<int>(v_2.l_ld()), &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_r_beta, ws.d_r_beta, 1, ncclSum, comm, &stream));
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_beta_prev, ws.d_r_beta, 1, stream);
        kernels::sqrt(ws.d_beta_prev, 1, &stream);
        kernels::normalize(v_2.l_data(), ws.d_r_beta,
                           static_cast<int>(v_2.l_rows()), 1,
                           static_cast<int>(v_2.l_ld()), &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.e_d + k, ws.d_beta_prev, 1, stream);
        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    {
        const std::size_t k = M - 1;
        gpu_backend::memcpy_d2d_async<Backend>(V.l_data() + k * V.l_ld(), v_1.l_data(),
                            v_1.l_rows(), stream);
        CollectiveKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);
        v_w.redistributeImplAsync(&v_2, &stream);
        kernels::batched_dot(v_1.l_data(), v_2.l_data(), ws.d_alpha,
                             static_cast<int>(v_1.l_rows()), 1,
                             static_cast<int>(v_1.l_ld()), true, &stream);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        ws.d_alpha, ws.d_alpha, 1, ncclSum, comm, &stream));
        kernels::batched_axpy_then_negate(
            ws.d_alpha, v_1.l_data(), v_2.l_data(),
            static_cast<int>(v_1.l_rows()), 1, static_cast<int>(v_1.l_ld()),
            &stream);
        kernels::get_real(ws.d_alpha, ws.d_real_alpha, 1, &stream);
        gpu_backend::memcpy_d2d_async<Backend>(ws.d_d + k, ws.d_real_alpha, 1, stream);
    }

    gpu_backend::memcpy_async_device_to_host<Backend>(d.data(), ws.d_d, M, stream);
    gpu_backend::memcpy_async_device_to_host<Backend>(e.data(), ws.e_d, M, stream);
    gpu_backend::stream_synchronize<Backend>(stream);

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

    const RealT last_beta = M > 1 ? e[M - 2] : RealT(0);
    *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
              std::abs(last_beta);
}

} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
