// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/cpu/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/rayleighRitz_impl.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "grid/nccl_utils.hpp"
#include "linalg/internal/typeTraits.hpp"

#include <algorithm>
#include <complex>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_xccl
{

template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename SolverParams, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void pseudo_hermitian_rayleigh_ritz_impl(
    BlasHandle blas_handle, SolverHandle solver_handle, SolverParams params,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* d_workspace, int d_lwork,
    typename MatrixType::value_type* h_workspace, int h_lwork,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    (void)solver_handle;
    (void)params;
    (void)devInfo;
    (void)d_workspace;
    (void)d_lwork;
    (void)h_workspace;
    (void)h_lwork;

    using T = typename MatrixType::value_type;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using kernels = gpu_device::kernels<Backend>;

    auto usedStream = util::get_blas_stream(blas_handle);

    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_ptr;
    if (A == nullptr)
    {
        A_ptr = std::make_unique<
            chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
            3 * subSize, subSize, V1.getMpiGrid_shared_ptr());
        A = A_ptr.get();
    }

    T One = T(1.0);
    T Zero = T(0.0);
    T NegOne = T(-1.0);
    chase::Base<T> real_Zero = chase::Base<T>(0.0);

    T* M = A->l_data() + subSize * subSize;
    T* W = A->l_data() + 2 * subSize * subSize;

    chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>
        diag(1, subSize, V1.getMpiGrid_shared_ptr());
    std::vector<chase::Base<T>> ptx_imag(subSize, chase::Base<T>(0.0));
    chase::Base<T>* ptx = ritzv.cpu_data() + offset;

    A->allocate_cpu_data();

    CollectiveKernelNamespace::MatrixMultiplyMultiVectorsAndRedistributeAsync(
        blas_handle, H, V1, W1, V2, W2, offset, subSize);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize,
        subSize, W2.l_rows(), &One, W2.l_data() + offset * W2.l_ld(),
        W2.l_ld(), W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero, W,
        subSize));
    gpu_backend::stream_synchronize<Backend>(usedStream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        W, W, subSize * subSize, ncclSum, A->getMpiGrid()->get_nccl_row_comm(), &usedStream));
    gpu_backend::stream_synchronize<Backend>(usedStream);

    CollectiveKernelNamespace::flipLowerHalfMatrixSign(V2, offset, subSize);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize,
        subSize, V2.l_rows(), &One, V2.l_data() + offset * V2.l_ld(),
        V2.l_ld(), V1.l_data() + offset * V1.l_ld(), V1.l_ld(), &Zero, M,
        subSize));
    gpu_backend::stream_synchronize<Backend>(usedStream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        M, M, subSize * subSize, ncclSum, A->getMpiGrid()->get_nccl_col_comm(), &usedStream));
    gpu_backend::stream_synchronize<Backend>(usedStream);

    kernels::plus_inverse_diagonal(M, subSize, subSize, real_Zero,
                                       diag.l_data(), usedStream);
    kernels::set_diagonal(M, subSize, subSize, Zero, usedStream);
    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), subSize,
        subSize, subSize, &NegOne, M, subSize, W, subSize, &Zero, A->l_data(),
        subSize));

    CollectiveKernelNamespace::flipLowerHalfMatrixSign(W1, offset, subSize);
    gpu_backend::device_synchronize<Backend>();

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize,
        subSize, W2.l_rows(), &One, W2.l_data() + offset * W2.l_ld(),
        W2.l_ld(), W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero, M,
        subSize));
    gpu_backend::stream_synchronize<Backend>(usedStream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        M, M, subSize * subSize, ncclSum, A->getMpiGrid()->get_nccl_row_comm(), &usedStream));
    gpu_backend::stream_synchronize<Backend>(usedStream);

    util::check_blas(blas_dispatch::axpy(
        blas_handle, subSize * subSize, &One, M, 1, A->l_data(), 1));
    kernels::scale_rows_matrix(A->l_data(), subSize, subSize, subSize,
                                   diag.l_data(), usedStream);
    gpu_backend::stream_synchronize<Backend>(usedStream);

    A->D2H();
    T* W_cpu = A->cpu_data() + 2 * subSize * subSize;
    lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', subSize, A->cpu_data(), subSize,
                     ptx, ptx_imag.data(), W_cpu, subSize);
    A->H2D();

    std::vector<std::size_t> indices(subSize);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&ptx](std::size_t i1, std::size_t i2)
              { return ptx[i1] < ptx[i2]; });

    std::vector<chase::Base<T>> sorted_ritzv(subSize);
    T* d_sorted_W = A->l_data();
    for (std::size_t i = 0; i < subSize; ++i)
    {
        sorted_ritzv[i] = ptx[indices[i]];
        gpu_backend::memcpy_d2d<Backend>(d_sorted_W + i * subSize,
                                W + indices[i] * subSize, subSize);
    }

    std::copy(sorted_ritzv.begin(), sorted_ritzv.end(), ptx);
    ritzv.H2D();

    CollectiveKernelNamespace::flipLowerHalfMatrixSign(V2, offset, subSize);
    gpu_backend::device_synchronize<Backend>();
    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), V2.l_rows(),
        subSize, subSize, &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(),
        d_sorted_W, subSize, &Zero, V1.l_data() + offset * V1.l_ld(),
        V1.l_ld()));
}

template <typename Backend, typename BlasHandle, typename SolverHandle,
          typename SolverParams, typename CollectiveKernelNamespace,
          typename MatrixType, typename InputMultiVectorType>
void pseudo_hermitian_rayleigh_ritz_v2_impl(
    BlasHandle blas_handle, SolverHandle solver_handle, SolverParams params,
    MatrixType& H, InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* workspace, int lwork,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    (void)params;

    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using blas_dispatch = gpu_backend::blas<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using kernels = gpu_device::kernels<Backend>;

    auto stream = util::get_blas_stream(blas_handle);
    T One = T(1.0);
    T Zero = T(0.0);
    T NegOne = T(-1.0);
    RealT ritz_neg_one = RealT(-1.0);
    RealT shift =
        std::numeric_limits<RealT>::epsilon() * RealT(10.0);

    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_ptr;
    if (A == nullptr)
    {
        A_ptr = std::make_unique<
            chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
            2 * subSize, subSize, V1.getMpiGrid_shared_ptr());
        A = A_ptr.get();
    }
    T* M = A->l_data() + subSize * subSize;

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork = 0;
        gpu_mpi::detail::query_heevd_lower_buffer_size<Backend, T>(
            solver_handle, subSize, A->l_data(), subSize,
            ritzv.l_data() + offset, &lwork);

        int lwork_potrf = 0;
        gpu_mpi::detail::query_potrf_lower_buffer_size<Backend, T>(
            solver_handle, subSize, A->l_data(), subSize, &lwork_potrf);
        lwork = std::max(lwork, lwork_potrf);
        const std::size_t upperTriangularSize =
            std::size_t(subSize * (subSize + 1) / 2);
        if (upperTriangularSize > static_cast<std::size_t>(lwork))
            lwork = static_cast<int>(upperTriangularSize);

        gpu_backend::device_malloc<Backend>(&workspace, static_cast<std::size_t>(lwork));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    CollectiveKernelNamespace::MatrixMultiplyMultiVectorsAndRedistributeAsync(
        blas_handle, H, V1, W1, V2, W2, offset, subSize);

    CollectiveKernelNamespace::flipLowerHalfMatrixSign(W1, offset, subSize,
                                                       stream);
    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize,
        subSize, W2.l_rows(), &One, W2.l_data() + offset * W2.l_ld(),
        W2.l_ld(), W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero,
        A->l_data(), subSize));
    gpu_backend::stream_synchronize<Backend>(stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        A->l_data(), A->l_data(), subSize * subSize, ncclSum, A->getMpiGrid()->get_nccl_row_comm(), &stream));
    gpu_backend::stream_synchronize<Backend>(stream);

    kernels::shift_diagonal(A->l_data(), subSize, subSize, shift, stream);
    gpu_mpi::detail::run_potrf_lower<Backend, T>(
        solver_handle, subSize, A->l_data(), subSize,
        reinterpret_cast<typename util::info_type*>(devInfo), workspace, lwork);
    int info = 0;
    gpu_backend::copy_to_host<Backend>(&info, devInfo, 1);
    if (info != 0)
    {
        throw std::runtime_error(
            "solver POTRF failed in Pseudo-Hermitian RayleighRitz, return "
            "value: " +
            std::to_string(info));
    }

    CollectiveKernelNamespace::flipLowerHalfMatrixSign(V1, offset, subSize,
                                                       stream);
    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize,
        subSize, V2.l_rows(), &One, V2.l_data() + offset * V2.l_ld(),
        V2.l_ld(), V1.l_data() + offset * V1.l_ld(), V1.l_ld(), &Zero, M,
        subSize));
    gpu_backend::stream_synchronize<Backend>(stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        M, M, subSize * subSize, ncclSum, A->getMpiGrid()->get_nccl_col_comm(), &stream));
    gpu_backend::stream_synchronize<Backend>(stream);

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_left(), util::fill_lower(),
        util::op_n(), util::diag_non_unit(), subSize, subSize,
        &One, A->l_data(), subSize, M, subSize));
    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_right(), util::fill_lower(),
        util::op_c(), util::diag_non_unit(), subSize, subSize,
        &One, A->l_data(), subSize, M, subSize));
    util::check_blas(blas_dispatch::scal(
        blas_handle, subSize * subSize, &NegOne, M, 1));

#ifdef RR_DOUBLE_PRECISION
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, float>::value)
    {
        if (A->isDoublePrecisionEnabled())
            A->copyToSubBlock(subSize * subSize, subSize * subSize);
        else
            A->enableDoublePrecision();

        if (!ritzv.isDoublePrecisionEnabled())
            ritzv.enableDoublePrecision();

        auto A_d = A->getDoublePrecisionMatrix();
        auto ritzv_d = ritzv.getDoublePrecisionMatrix();
        using DT = typename chase::ToDoublePrecisionTrait<T>::Type;
        DT* workspace_d = nullptr;
        gpu_backend::device_malloc<Backend>(&workspace_d,
                                  static_cast<std::size_t>(lwork));
        std::unique_ptr<DT, typename util::device_deleter>
            workspace_d_ptr(workspace_d);

        gpu_mpi::detail::run_heevd_lower<Backend, DT>(
            solver_handle, subSize, A_d->l_data() + subSize * subSize,
            subSize, ritzv_d->l_data() + offset, workspace_d, lwork,
            reinterpret_cast<typename util::info_type*>(devInfo));
        ritzv.disableDoublePrecision(true);
        A->copyBackSubBlock(subSize * subSize, subSize * subSize);
        gpu_backend::device_synchronize<Backend>();
    }
    else
    {
#endif
        gpu_mpi::detail::run_heevd_lower<Backend, T>(
            solver_handle, subSize, M, subSize, ritzv.l_data() + offset,
            workspace, lwork,
            reinterpret_cast<typename util::info_type*>(devInfo));
#ifdef RR_DOUBLE_PRECISION
    }
#endif

    gpu_backend::copy_to_host<Backend>(&info, devInfo, 1);
    if (info != 0)
    {
        throw std::runtime_error("solver HEEVD failed in Pseudo-Hermitian "
                                 "RayleighRitz, return value: " +
                                 std::to_string(info));
    }

    util::check_blas(blas_dispatch::trsm(
        blas_handle, util::side_left(), util::fill_lower(),
        util::op_c(), util::diag_non_unit(), subSize, subSize,
        &One, A->l_data(), subSize, M, subSize));

    kernels::template inverse_entries<RealT>(
        ritzv.l_data() + offset, subSize, stream);
    util::check_blas(blas_dispatch::scal(
        blas_handle, subSize, &ritz_neg_one, ritzv.l_data() + offset, 1));
    gpu_backend::stream_synchronize<Backend>(stream);
    gpu_backend::copy_to_host<Backend>(ritzv.cpu_data() + offset,
                              ritzv.l_data() + offset, subSize);

    RealT* d_col_norms = nullptr;
    gpu_backend::device_malloc<Backend>(&d_col_norms, subSize / 2);
    std::unique_ptr<RealT, typename util::device_deleter>
        col_norms_ptr(d_col_norms);
    kernels::fused_norm_squared_normalize(
        M, d_col_norms, static_cast<int>(subSize),
        static_cast<int>(subSize / 2), static_cast<int>(subSize), &stream);

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), V2.l_rows(),
        subSize / 2, subSize, &One, V2.l_data() + offset * V2.l_ld(),
        V2.l_ld(), M, subSize, &Zero, V1.l_data() + offset * V1.l_ld(),
        V1.l_ld()));
}

} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
