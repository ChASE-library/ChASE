// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/cpu/lapackpp/lapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
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
void pseudo_hermitian_lanczos_impl(
    BlasHandle blas_handle, std::size_t M, std::size_t numvec,
    MatrixType& H, InputMultiVectorType& V,
    chase::Base<typename MatrixType::value_type>* upperb,
    chase::Base<typename MatrixType::value_type>* ritzv,
    chase::Base<typename MatrixType::value_type>* Tau,
    chase::Base<typename MatrixType::value_type>* ritzV)
{
    using T = typename MatrixType::value_type;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    gpu_device::detail::lanczos_pointer_mode_guard<Backend> pointer_guard(blas_handle);

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

    std::vector<chase::Base<T>> real_beta(numvec);
    std::vector<chase::Base<T>> real_alpha(numvec);

    std::vector<chase::Base<T>> d(M * numvec);
    std::vector<chase::Base<T>> e(M * numvec);

    std::vector<T> alpha(numvec, T(1.0));
    std::vector<T> beta(numvec, T(0.0));

    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto Sv = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

    kernel_ops::lacpy('A', v_1.l_rows(), numvec,
                                           V.l_data(), V.l_ld(), v_1.l_data(),
                                           v_1.l_ld());

    gpu_backend::device_synchronize<Backend>();
    MpiKernelNamespace::MatrixMultiplyMultiVectors(
        blas_handle, &One, H, v_1, &Zero, v_w);

    gpu_backend::device_synchronize<Backend>();
    v_w.redistributeImpl(&v_2);

    kernel_ops::lacpy('A', Sv.l_rows(), numvec,
                                           v_2.l_data(), v_2.l_ld(),
                                           Sv.l_data(), Sv.l_ld());

    MpiKernelNamespace::flipLowerHalfMatrixSign(Sv);
    gpu_backend::device_synchronize<Backend>();

    for (auto i = 0; i < numvec; i++)
    {
        util::check_blas(blas_dispatch::dot(
            blas_handle, v_1.l_rows(), v_1.l_data() + i * v_1.l_ld(), 1,
            Sv.l_data() + i * Sv.l_ld(), 1, &beta[i]));
    }

    for (auto i = 0; i < numvec; i++)
    {
        real_beta[i] = std::real(beta[i]);
    }

    gpu_backend::device_synchronize<Backend>();
    MPI_Allreduce(MPI_IN_PLACE, real_beta.data(), numvec,
                  chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                  H.getMpiGrid()->get_col_comm());

    for (auto i = 0; i < numvec; i++)
    {
        beta[i] = One / std::sqrt(real_beta[i]);
    }

    for (auto i = 0; i < numvec; i++)
    {
        util::check_blas(blas_dispatch::scal(
            blas_handle, v_1.l_rows(), &beta[i],
            v_1.l_data() + i * v_1.l_ld(), 1));
    }

    for (auto i = 0; i < numvec; i++)
    {
        util::check_blas(blas_dispatch::scal(
            blas_handle, v_2.l_rows(), &beta[i],
            v_2.l_data() + i * v_2.l_ld(), 1));
    }

    for (std::size_t k = 0; k < M; k = k + 1)
    {
        for (auto i = 0; i < numvec; i++)
        {
            gpu_backend::memcpy_d2d<Backend>(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(), v_1.l_rows());
        }

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::dot(
                blas_handle, v_2.l_rows(), v_2.l_data() + i * v_2.l_ld(), 1,
                Sv.l_data() + i * Sv.l_ld(), 1, &alpha[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = std::real(alpha[i]);
        }

        gpu_backend::device_synchronize<Backend>();
        MPI_Allreduce(MPI_IN_PLACE, real_alpha.data(), numvec,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -real_alpha[i] * beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::axpy(
                blas_handle, v_1.l_rows(), &alpha[i],
                v_1.l_data() + i * v_1.l_ld(), 1, v_2.l_data() + i * v_2.l_ld(),
                1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            alpha[i] = -alpha[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            d[k + M * i] = std::real(alpha[i]);
        }

        if (k == M - 1)
            break;

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -One / beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::axpy(
                blas_handle, v_0.l_rows(), &beta[i],
                v_0.l_data() + i * v_0.l_ld(), 1, v_2.l_data() + i * v_2.l_ld(),
                1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = -beta[i];
        }

        v_1.swap(v_0);
        v_1.swap(v_2);

        MpiKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);

        gpu_backend::device_synchronize<Backend>();
        v_w.redistributeImpl(&v_2);

        kernel_ops::lacpy('A', Sv.l_rows(), numvec,
                                               v_2.l_data(), v_2.l_ld(),
                                               Sv.l_data(), Sv.l_ld());

        MpiKernelNamespace::flipLowerHalfMatrixSign(Sv);
    gpu_backend::device_synchronize<Backend>();

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::dot(
                blas_handle, v_1.l_rows(), v_1.l_data() + i * v_1.l_ld(), 1,
                Sv.l_data() + i * Sv.l_ld(), 1, &beta[i]));
        }

        for (auto i = 0; i < numvec; i++)
        {
            real_beta[i] = std::real(beta[i]);
        }

        gpu_backend::device_synchronize<Backend>();
        MPI_Allreduce(MPI_IN_PLACE, real_beta.data(), numvec,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        for (auto i = 0; i < numvec; i++)
        {
            real_beta[i] = std::sqrt(real_beta[i]);
        }

        for (auto i = 0; i < numvec; i++)
        {
            e[k + M * i] = real_beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            beta[i] = One / real_beta[i];
        }

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::scal(
                blas_handle, v_1.l_rows(), &beta[i],
                v_1.l_data() + i * v_1.l_ld(), 1));
        }

        for (auto i = 0; i < numvec; i++)
        {
            util::check_blas(blas_dispatch::scal(
                blas_handle, v_2.l_rows(), &beta[i],
                v_2.l_data() + i * v_2.l_ld(), 1));
        }
    }

    kernel_ops::lacpy('A', v_1.l_rows(), numvec,
                                           v_1.l_data(), v_1.l_ld(), V.l_data(),
                                           V.l_ld());

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    chase::Base<T> ul = 0;
    chase::Base<T> ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);

    for (auto i = 0; i < numvec; i++)
    {

        lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', M, d.data() + i * M,
                          e.data() + i * M, ul, ll, vl, vu, &notneeded_m,
                          ritzv + M * i, ritzV, M, M, isuppz.data(), &tryrac);

        for (std::size_t k = 0; k < M; ++k)
        {
            Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
        }
    }

    *upperb = ritzv[M - 1];
}

template <typename Backend, typename BlasHandle, typename MpiKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void pseudo_hermitian_lanczos_impl(BlasHandle blas_handle, std::size_t M,
                                   MatrixType& H, InputMultiVectorType& V,
                                   chase::Base<typename MatrixType::value_type>* upperb)
{
    using T = typename MatrixType::value_type;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using ResultMultiVectorType =
        typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

    gpu_device::detail::lanczos_pointer_mode_guard<Backend> pointer_guard(blas_handle);

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

    std::vector<chase::Base<T>> d(M);
    std::vector<chase::Base<T>> e(M);

    chase::Base<T> real_alpha;
    chase::Base<T> real_beta;

    T alpha = T(1.0);
    T beta = T(0.0);
    T One = T(1.0);
    T Zero = T(0.0);

    std::size_t N = H.g_rows();

    auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
    auto v_1 = v_0.template clone<InputMultiVectorType>();
    auto v_2 = v_0.template clone<InputMultiVectorType>();
    auto Sv = v_0.template clone<InputMultiVectorType>();
    auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

    kernel_ops::lacpy('A', v_1.l_rows(), 1, V.l_data(),
                                           V.l_ld(), v_1.l_data(), v_1.l_ld());

    gpu_backend::device_synchronize<Backend>();

    MpiKernelNamespace::MatrixMultiplyMultiVectors(
        blas_handle, &One, H, v_1, &Zero, v_w);

    gpu_backend::device_synchronize<Backend>();
    v_w.redistributeImpl(&v_2);

    kernel_ops::lacpy('A', Sv.l_rows(), 1, v_2.l_data(),
                                           v_2.l_ld(), Sv.l_data(), Sv.l_ld());

    MpiKernelNamespace::flipLowerHalfMatrixSign(Sv);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::dot(
        blas_handle, v_1.l_rows(), v_1.l_data(), 1, Sv.l_data(), 1, &beta));

    real_beta = std::real(beta);

    gpu_backend::device_synchronize<Backend>();
    MPI_Allreduce(MPI_IN_PLACE, &real_beta, 1,
                  chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                  H.getMpiGrid()->get_col_comm());

    beta = One / std::sqrt(real_beta);

    util::check_blas(blas_dispatch::scal(
        blas_handle, v_1.l_rows(), &beta, v_1.l_data(), 1));

    util::check_blas(blas_dispatch::scal(
        blas_handle, v_2.l_rows(), &beta, v_2.l_data(), 1));

    for (std::size_t k = 0; k < M; k = k + 1)
    {
        gpu_backend::memcpy_d2d<Backend>(V.l_data() + k * V.l_ld(), v_1.l_data(), v_1.l_rows());

        util::check_blas(blas_dispatch::dot(
            blas_handle, v_2.l_rows(), v_2.l_data(), 1, Sv.l_data(), 1,
            &alpha));

        real_alpha = std::real(alpha);

        gpu_backend::device_synchronize<Backend>();
        MPI_Allreduce(MPI_IN_PLACE, &real_alpha, 1,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        alpha = -real_alpha * beta;

        util::check_blas(blas_dispatch::axpy(
            blas_handle, v_1.l_rows(), &alpha, v_1.l_data(), 1, v_2.l_data(),
            1));

        alpha = -alpha;

        d[k] = std::real(alpha);

        if (k == M - 1)
            break;

        beta = -One / beta;

        util::check_blas(blas_dispatch::axpy(
            blas_handle, v_0.l_rows(), &beta, v_0.l_data(), 1, v_2.l_data(),
            1));
        beta = -beta;

        v_1.swap(v_0);
        v_1.swap(v_2);

        MpiKernelNamespace::MatrixMultiplyMultiVectors(
            blas_handle, &One, H, v_1, &Zero, v_w);

        gpu_backend::device_synchronize<Backend>();
        v_w.redistributeImpl(&v_2);

        kernel_ops::lacpy('A', Sv.l_rows(), 1,
                                               v_2.l_data(), v_2.l_ld(),
                                               Sv.l_data(), Sv.l_ld());

        MpiKernelNamespace::flipLowerHalfMatrixSign(Sv);
    gpu_backend::device_synchronize<Backend>();

        util::check_blas(blas_dispatch::dot(
            blas_handle, v_1.l_rows(), v_1.l_data(), 1, Sv.l_data(), 1,
            &beta));
        real_beta = std::real(beta);

        gpu_backend::device_synchronize<Backend>();
        MPI_Allreduce(MPI_IN_PLACE, &real_beta, 1,
                      chase::mpi::getMPI_Type<chase::Base<T>>(), MPI_SUM,
                      H.getMpiGrid()->get_col_comm());

        real_beta = std::sqrt(real_beta);

        e[k] = real_beta;

        beta = One / real_beta;

        util::check_blas(blas_dispatch::scal(
            blas_handle, v_1.l_rows(), &beta, v_1.l_data(), 1));

        util::check_blas(blas_dispatch::scal(
            blas_handle, v_2.l_rows(), &beta, v_2.l_data(), 1));
    }

    int notneeded_m;
    std::size_t vl = 0;
    std::size_t vu = 0;
    chase::Base<T> ul = 0;
    chase::Base<T> ll = 0;
    int tryrac = 0;
    std::vector<int> isuppz(2 * M);
    std::vector<chase::Base<T>> ritzv(M);

    lapackpp::t_stemr<chase::Base<T>>(
        LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
        &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);

    *upperb = ritzv[M - 1];
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
