// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/typeTraits.hpp"

#include <cstddef>
#include <complex>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <type_traits>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_mpi
{
namespace detail
{

template <typename Backend, typename T>
void query_heevd_upper_buffer_size(
    typename gpu_backend::handle_utils<Backend>::solver_handle handle,
    std::size_t n, T* A, std::size_t lda, chase::Base<T>* W, int* lwork)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        util::check_solver(solver::template heevd_buffer_size<T>(
            handle, util::eig_original(), util::fill_upper(), n, A, lda, W,
            lwork));
    }
}

template <typename Backend, typename T>
void run_heevd_upper(
    typename gpu_backend::handle_utils<Backend>::solver_handle handle,
    std::size_t n, T* A, std::size_t lda, chase::Base<T>* W, T* workspace,
    int lwork, typename gpu_backend::handle_utils<Backend>::info_type* devInfo)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        util::check_solver(solver::template heevd<T>(
            handle, util::eig_original(), util::fill_upper(), n, A, lda, W,
            workspace, static_cast<std::size_t>(lwork), devInfo));
    }
    else
    {
        (void)workspace;
        (void)lwork;
        using Scalar = chase::Base<T>;
        Scalar* d_E = nullptr;
        gpu_backend::device_malloc<Backend>(&d_E, n);
        std::unique_ptr<Scalar, typename util::device_deleter> d_E_ptr(d_E);
        util::check_solver(solver::template heevd<T>(
            handle, util::eig_original(), util::fill_upper(), n, A, lda, W, d_E,
            devInfo));
    }
}

template <typename Backend, typename T>
void query_heevd_lower_buffer_size(
    typename gpu_backend::handle_utils<Backend>::solver_handle handle,
    std::size_t n, T* A, std::size_t lda, chase::Base<T>* W, int* lwork)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    util::check_solver(solver::template heevd_buffer_size<T>(
        handle, util::eig_original(), util::fill_lower(), n, A, lda, W, lwork));
}

template <typename Backend, typename T>
void query_potrf_lower_buffer_size(
    typename gpu_backend::handle_utils<Backend>::solver_handle handle,
    std::size_t n, T* A, std::size_t lda, int* lwork)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    util::check_solver(solver::template potrf_buffer_size<T>(
        handle, util::fill_lower(), n, A, lda, lwork));
}

template <typename Backend, typename T>
void run_potrf_lower(
    typename gpu_backend::handle_utils<Backend>::solver_handle handle,
    std::size_t n, T* A, std::size_t lda,
    typename gpu_backend::handle_utils<Backend>::info_type* devInfo,
    T* workspace, int lwork)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    util::check_solver(solver::template potrf<T>(
        handle, util::fill_lower(), n, A, lda, workspace,
        static_cast<std::size_t>(lwork), devInfo));
}

template <typename Backend, typename T>
void run_heevd_lower(
    typename gpu_backend::handle_utils<Backend>::solver_handle handle,
    std::size_t n, T* A, std::size_t lda, chase::Base<T>* W, T* workspace,
    int lwork, typename gpu_backend::handle_utils<Backend>::info_type* devInfo)
{
    using util = gpu_backend::handle_utils<Backend>;
    using solver = gpu_backend::solver<Backend>;
    util::check_solver(solver::template heevd<T>(
        handle, util::eig_original(), util::fill_lower(), n, A, lda, W,
        workspace, static_cast<std::size_t>(lwork), devInfo));
}

template <typename Backend>
void sync_after_rayleigh_blas(typename gpu_backend::handle_utils<Backend>::blas_handle handle)
{
    using util = gpu_backend::handle_utils<Backend>;
#if defined(HAS_CUDA)
    if constexpr (std::is_same_v<Backend, gpu_backend::cuda_backend>)
    {
        (void)handle;
        gpu_backend::device_synchronize<Backend>();
    }
    else
#endif
    {
        util::synchronize_blas_stream(handle);
    }
}

} // namespace detail

template <typename Backend, typename MpiKernelNamespace, typename BlasHandle,
          typename SolverHandle, typename MatrixType,
          typename InputMultiVectorType>
void rayleigh_ritz_impl(
    BlasHandle blas_handle, SolverHandle solver_handle, MatrixType& H,
    InputMultiVectorType& V1, InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::distMatrix::RedundantMatrix<
        chase::Base<typename MatrixType::value_type>, chase::platform::GPU>&
        ritzv,
    std::size_t offset, std::size_t subSize, int* devInfo,
    typename MatrixType::value_type* workspace, int lwork_heevd,
    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type,
                                       chase::platform::GPU>* A)
{
    using T = typename MatrixType::value_type;
    using kernels = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;
    using blas_dispatch = gpu_backend::blas<Backend>;

    std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>
        A_ptr;
    const std::size_t upperTriangularSize =
        std::size_t(subSize * (subSize + 1) / 2);

    if (A == nullptr)
    {
        A_ptr = std::make_unique<
            chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(
            subSize, subSize, V1.getMpiGrid_shared_ptr());
        A = A_ptr.get();
    }

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork_heevd = 0;
        detail::query_heevd_upper_buffer_size<Backend, T>(
            solver_handle, subSize, A->l_data(), subSize,
            ritzv.l_data() + offset, &lwork_heevd);

        if (upperTriangularSize > static_cast<std::size_t>(lwork_heevd))
        {
            lwork_heevd = static_cast<int>(upperTriangularSize);
        }

        gpu_backend::device_malloc<Backend>(&workspace, static_cast<std::size_t>(lwork_heevd));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    MpiKernelNamespace::MatrixMultiplyMultiVectorsAndRedistribute(
        blas_handle, H, V1, W1, V2, W2, offset, subSize);

    const T One = T(1.0);
    const T Zero = T(0.0);

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize,
        subSize, W2.l_rows(), &One, W2.l_data() + offset * W2.l_ld(),
        W2.l_ld(), W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero,
        A->l_data(), subSize));

    detail::sync_after_rayleigh_blas<Backend>(blas_handle);
    kernels::extract_upper(A->l_data(), subSize, workspace, subSize,
                           blas_handle);
    detail::sync_after_rayleigh_blas<Backend>(blas_handle);
    MPI_Allreduce(MPI_IN_PLACE, workspace, upperTriangularSize,
                  chase::mpi::getMPI_Type<T>(), MPI_SUM,
                  A->getMpiGrid()->get_row_comm());
    kernels::unpack_upper(workspace, subSize, A->l_data(), subSize,
                          blas_handle);
    detail::sync_after_rayleigh_blas<Backend>(blas_handle);

    auto* backend_info =
        reinterpret_cast<typename util::info_type*>(devInfo);

#ifdef RR_DOUBLE_PRECISION
    if constexpr (std::is_same<T, std::complex<float>>::value ||
                  std::is_same<T, float>::value)
    {
        if (A->isDoublePrecisionEnabled())
        {
            A->copyToSubBlock(0, subSize * subSize);
        }
        else
        {
            A->enableDoublePrecision();
        }

        if (!ritzv.isDoublePrecisionEnabled())
        {
            ritzv.enableDoublePrecision();
        }

        auto A_d = A->getDoublePrecisionMatrix();
        auto ritzv_d = ritzv.getDoublePrecisionMatrix();
        using DT = typename chase::ToDoublePrecisionTrait<T>::Type;
        DT* workspace_d = nullptr;
        gpu_backend::device_malloc<Backend>(&workspace_d, static_cast<std::size_t>(lwork_heevd));
        std::unique_ptr<DT, typename util::device_deleter> workspace_d_ptr(
            workspace_d);

        detail::run_heevd_upper<Backend, DT>(
            solver_handle, subSize, A_d->l_data(), subSize,
            ritzv_d->l_data() + offset, workspace_d, lwork_heevd, backend_info);
        ritzv.disableDoublePrecision(true);
        A->copyBackSubBlock(0, subSize * subSize);
        gpu_backend::device_synchronize<Backend>();
    }
    else
    {
#endif
        detail::run_heevd_upper<Backend, T>(
            solver_handle, subSize, A->l_data(), subSize,
            ritzv.l_data() + offset, workspace, lwork_heevd, backend_info);
#ifdef RR_DOUBLE_PRECISION
    }
#endif

    typename util::info_type info = -1;
    gpu_backend::copy_to_host<Backend>(&info, backend_info, 1);
    if (info != 0)
    {
        throw std::runtime_error("GPU HEEVD failed in RayleighRitz");
    }

    gpu_backend::copy_to_host<Backend>(ritzv.cpu_data() + offset, ritzv.l_data() + offset,
                      subSize);

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), V2.l_rows(),
        subSize, subSize, &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(),
        A->l_data(), subSize, &Zero, V1.l_data() + offset * V1.l_ld(),
        V1.l_ld()));
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
