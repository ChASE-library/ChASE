// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/cpu/lapackpp/lapackpp.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/typeTraits.hpp"
#include <vector>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{

template <typename Backend, typename BlasHandle, typename MpiKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_impl(BlasHandle blas_handle, std::size_t M, std::size_t numvec,
                  MatrixType& H, InputMultiVectorType& V,
                  chase::Base<typename MatrixType::value_type>* upperb,
                  chase::Base<typename MatrixType::value_type>* ritzv,
                  chase::Base<typename MatrixType::value_type>* Tau,
                  chase::Base<typename MatrixType::value_type>* ritzV);

template <typename Backend, typename BlasHandle, typename MpiKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_impl(BlasHandle blas_handle, std::size_t M, MatrixType& H,
                  InputMultiVectorType& V,
                  chase::Base<typename MatrixType::value_type>* upperb);

template <typename Backend, typename BlasHandle, typename MpiKernelNamespace,
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
        MpiKernelNamespace::pseudo_hermitian_lanczos(
            blas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
    }
    else
    {
        lanczos_impl<Backend, BlasHandle, MpiKernelNamespace>(
            blas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
    }
}

template <typename Backend, typename BlasHandle, typename MpiKernelNamespace,
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
        MpiKernelNamespace::pseudo_hermitian_lanczos(blas_handle, M, H, V,
                                                     upperb);
    }
    else
    {
        lanczos_impl<Backend, BlasHandle, MpiKernelNamespace>(blas_handle, M,
                                                              H, V, upperb);
    }
}

template <typename Backend, typename BlasHandle, typename MpiKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_impl(BlasHandle blas_handle, std::size_t M,
                       std::size_t numvec, MatrixType& H,
                       InputMultiVectorType& V,
                       chase::Base<typename MatrixType::value_type>* upperb,
                       chase::Base<typename MatrixType::value_type>* ritzv,
                       chase::Base<typename MatrixType::value_type>* Tau,
                       chase::Base<typename MatrixType::value_type>* ritzV)
{
    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    if (H.g_cols() != H.g_rows())
    {
        throw std::runtime_error("Lanczos requires matrix to be squared");
    }

    if (H.getMpiGrid() != V.getMpiGrid())
    {
        throw std::runtime_error("Lanczos requires H and V in same MPI grid");
    }
    if (H.g_rows() != V.g_rows())
    {
        throw std::runtime_error("Lanczos H and V have same number of rows");
    }

    gpu_device::detail::lanczos_pointer_mode_guard<Backend> pointer_guard(blas_handle);
    gpu_device::detail::lanczos_scalar_workspace<Backend, T>
        scalar_workspace(blas_handle);

    std::vector<RealT> r_beta(numvec);

    std::vector<RealT> d(M * numvec);
    std::vector<RealT> e(M * numvec);

    std::vector<RealT> real_alpha(numvec);
    std::vector<T> alpha(numvec, T(1.0));
    std::vector<T> beta(numvec, T(0.0));

    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    kernel_ops::lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld());
    for (auto i = 0; i < numvec; i++)
    {
        gpu_device::detail::lanczos_nrm2<Backend, T>(blas_handle, v_1.l_rows(),
                                v_1.l_data() + i * v_1.l_ld(), 1,
                                &real_alpha[i],
                                scalar_workspace.double_accum());
        real_alpha[i] = std::pow(real_alpha[i], 2);
    }

    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(real_alpha.data(), numvec, H.getMpiGrid()->get_col_comm());

    for (auto i = 0; i < numvec; i++)
    {
        real_alpha[i] = std::sqrt(real_alpha[i]);
        alpha[i] = T(1 / real_alpha[i]);
    }

    for (auto i = 0; i < numvec; i++)
    {
        util::check_blas(blas_dispatch::scal(
            blas_handle, v_1.l_rows(), &alpha[i],
            v_1.l_data() + i * v_1.l_ld(), 1));
    }

    for (std::size_t k = 0; k < M; k = k + 1)
    {
        for (auto i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d<Backend>(V.l_data() + k * V.l_ld(),
                                    v_1.l_data() + i * v_1.l_ld(),
                                    v_1.l_rows());
        }

        MpiKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);
        v_w.redistributeImpl(&v_2);

        for (auto i = 0; i < numvec; i++)
        {
            gpu_device::detail::lanczos_dot<Backend, T>(blas_handle, v_1.l_rows(), v_1.l_data() + i * v_1.l_ld(), 1,
                v_2.l_data() + i * v_2.l_ld(), 1, &alpha[i],
                scalar_workspace.double_accum());

            alpha[i] = -alpha[i];
        }

        gpu_backend::device_synchronize<Backend>();

        mpiAllreduceSum(alpha.data(), numvec, H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::axpy(
                blas_handle, v_1.l_rows(), &alpha[i],
                v_1.l_data() + i * v_1.l_ld(), 1, v_2.l_data() + i * v_2.l_ld(),
                1));
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha[i]);
        }
        if (k > 0)
        {
            for (auto i = 0; i < numvec; i++)
            {
                beta[i] = T(-r_beta[i]);
            }
            for (auto i = 0; i < numvec; i++)
            {
                util::check_blas(blas_dispatch::axpy(
                    blas_handle, v_0.l_rows(), &beta[i],
                    v_0.l_data() + i * v_0.l_ld(), 1,
                    v_2.l_data() + i * v_2.l_ld(), 1));
            }
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            gpu_device::detail::lanczos_nrm2<Backend, T>(blas_handle, v_2.l_rows(), v_2.l_data() + i * v_2.l_ld(), 1,
                &r_beta[i], scalar_workspace.double_accum());

            r_beta[i] = std::pow(r_beta[i], 2);
        }

        gpu_backend::device_synchronize<Backend>();
        mpiAllreduceSum(r_beta.data(), numvec, H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            r_beta[i] = std::sqrt(r_beta[i]);
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = T(1 / r_beta[i]);
        }

        if (k == M - 1)
        {
            break;
        }

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::scal(
                blas_handle, v_2.l_rows(), &beta[i],
                v_2.l_data() + i * v_2.l_ld(), 1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = r_beta[i];
        }

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    kernel_ops::lacpy('A', v_1.l_rows(), numvec,
                                           v_1.l_data(), v_1.l_ld(), V.l_data(),
                                           V.l_ld());

    RealT upperb_tridiag = 0;
    for (auto i = 0; i < numvec; ++i)
    {
        upperb_tridiag = std::max(
            upperb_tridiag,
            gpu_device::detail::lanczos_tridiagonal_upper_bound(d.data() + i * M, M,
                                                    r_beta[i]));
    }

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    lapack_logical tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> d_stemr(M);
    std::vector<RealT> e_stemr(M);

    for (auto i = 0; i < numvec; i++)
    {
        std::copy(d.data() + i * M, d.data() + (i + 1) * M, d_stemr.begin());
        std::copy(e.data() + i * M, e.data() + (i + 1) * M, e_stemr.begin());

        lapackpp::t_stemr<RealT>(LAPACK_COL_MAJOR, 'V', 'A', M, d_stemr.data(),
                                 e_stemr.data(), ul, ll, vl, vu, &notneeded_m,
                                 ritzv + M * i, ritzV, M, M, isuppz.data(),
                                 &tryrac);
        for (std::size_t k = 0; k < M; ++k)
        {
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
        }
    }

    RealT upperb_ritz = 0;
    for (auto i = 0; i < numvec; ++i)
    {
        RealT max_ritz = 0;
        for (std::size_t k = 0; k < M; ++k)
        {
            max_ritz = std::max(max_ritz, std::abs(ritzv[i * M + k]));
        }
        upperb_ritz =
            std::max(upperb_ritz, max_ritz + std::abs(r_beta[i]));
    }
    *upperb = std::max(upperb_ritz, upperb_tridiag);
}

template <typename Backend, typename BlasHandle, typename MpiKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void lanczos_impl(BlasHandle blas_handle, std::size_t M,
                       MatrixType& H, InputMultiVectorType& V,
                       chase::Base<typename MatrixType::value_type>* upperb)
{
    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    if (H.g_cols() != H.g_rows())
    {
        throw std::runtime_error("Lanczos requires matrix to be squared");
    }

    if (H.getMpiGrid() != V.getMpiGrid())
    {
        throw std::runtime_error("Lanczos requires H and V in same MPI grid");
    }

    if (H.g_rows() != V.g_rows())
    {
        throw std::runtime_error("Lanczos H and V have same number of rows");
    }

    gpu_device::detail::lanczos_pointer_mode_guard<Backend> pointer_guard(blas_handle);
    gpu_device::detail::lanczos_scalar_workspace<Backend, T>
        scalar_workspace(blas_handle);

    std::vector<RealT> d(M);
    std::vector<RealT> e(M);

    RealT real_alpha;
    RealT r_beta;

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    kernel_ops::lacpy('A', v_1.l_rows(), 1, V.l_data(),
                                           V.l_ld(), v_1.l_data(), v_1.l_ld());

    gpu_device::detail::lanczos_nrm2<Backend, T>(blas_handle, v_1.l_rows(), v_1.l_data(), 1,
                            &real_alpha, scalar_workspace.double_accum());
    real_alpha = std::pow(real_alpha, 2);

    gpu_backend::device_synchronize<Backend>();
    mpiAllreduceSum(&real_alpha, 1, H.getMpiGrid()->get_col_comm());

    real_alpha = std::sqrt(real_alpha);
    alpha = T(1 / real_alpha);

    util::check_blas(blas_dispatch::scal(
        blas_handle, v_1.l_rows(), &alpha, v_1.l_data(), 1));
    for (std::size_t k = 0; k < M; k = k + 1)
    {
        MpiKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);

        v_w.redistributeImpl(&v_2);

        gpu_device::detail::lanczos_dot<Backend, T>(blas_handle, v_1.l_rows(), v_1.l_data(), 1,
                               v_2.l_data(), 1, &alpha,
                               scalar_workspace.double_accum());
        alpha = -alpha;

        gpu_backend::device_synchronize<Backend>();
        mpiAllreduceSum(&alpha, 1, H.getMpiGrid()->get_col_comm());

        util::check_blas(blas_dispatch::axpy(
            blas_handle, v_1.l_rows(), &alpha, v_1.l_data(), 1, v_2.l_data(),
            1));
        alpha = -alpha;

        d[k] = std::real(alpha);

        if (k > 0)
        {
            beta = T(-r_beta);
            util::check_blas(blas_dispatch::axpy(
                blas_handle, v_0.l_rows(), &beta, v_0.l_data(), 1,
                v_2.l_data(), 1));
        }

        beta = -beta;

        gpu_device::detail::lanczos_nrm2<Backend, T>(blas_handle, v_2.l_rows(), v_2.l_data(), 1,
                                &r_beta, scalar_workspace.double_accum());

        r_beta = std::pow(r_beta, 2);

        gpu_backend::device_synchronize<Backend>();
        mpiAllreduceSum(&r_beta, 1, H.getMpiGrid()->get_col_comm());

        r_beta = std::sqrt(r_beta);

        beta = T(1 / r_beta);

        if (k == M - 1)
            break;

        util::check_blas(blas_dispatch::scal(
            blas_handle, v_2.l_rows(), &beta, v_2.l_data(), 1));
        e[k] = r_beta;

        v_1.swap(v_0);
        v_1.swap(v_2);
    }

    const RealT upperb_tridiag =
        gpu_device::detail::lanczos_tridiagonal_upper_bound(d.data(), M, r_beta);

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    RealT ul = 0;
    RealT ll = 0;
    lapack_logical tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<RealT> ritzv(M);
    std::vector<RealT> d_stemr(M);
    std::vector<RealT> e_stemr(M);

    std::copy(d.begin(), d.end(), d_stemr.begin());
    std::copy(e.begin(), e.end(), e_stemr.begin());

    lapackpp::t_stemr<RealT>(LAPACK_COL_MAJOR, 'N', 'A', M, d_stemr.data(),
                             e_stemr.data(), ul, ll, vl, vu, &notneeded_m,
                             ritzv.data(), static_cast<RealT*>(nullptr), M, M,
                             isuppz.data(), &tryrac);

    RealT upperb_ritz = 0;
    for (std::size_t k = 0; k < M; ++k)
    {
        upperb_ritz = std::max(upperb_ritz, std::abs(ritzv[k]));
    }
    upperb_ritz += std::abs(r_beta);

    *upperb = std::max(upperb_ritz, upperb_tridiag);
}


} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
