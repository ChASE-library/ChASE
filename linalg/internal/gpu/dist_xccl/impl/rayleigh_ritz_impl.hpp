// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_mpi/impl/rayleighRitz_impl.hpp"
#include "grid/nccl_utils.hpp"
#include "linalg/internal/gpu/backend/blas.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/typeTraits.hpp"

#include <memory>
#include <stdexcept>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_xccl
{

template <typename GpuBackend, typename CollectiveKernelNamespace, typename BlasHandle,
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
    using gpu_backend_type = GpuBackend;
    using util = gpu_backend::handle_utils<gpu_backend_type>;
    using kernels = gpu_device::kernels<gpu_backend_type>;
    using blas_dispatch = gpu_backend::blas<gpu_backend_type>;

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

    typename util::stream_type stream = util::get_blas_stream(blas_handle);

    std::unique_ptr<T, typename util::device_deleter> work_ptr = nullptr;
    if (workspace == nullptr)
    {
        lwork_heevd = 0;
        gpu_mpi::detail::query_heevd_upper_buffer_size<gpu_backend_type, T>(
            solver_handle, subSize, A->l_data(), subSize,
            ritzv.l_data() + offset, &lwork_heevd);

        if (upperTriangularSize > static_cast<std::size_t>(lwork_heevd))
        {
            lwork_heevd = static_cast<int>(upperTriangularSize);
        }

        gpu_backend::device_malloc<gpu_backend_type>(&workspace, static_cast<std::size_t>(lwork_heevd));
        work_ptr.reset(workspace);
        workspace = work_ptr.get();
    }

    CollectiveKernelNamespace::MatrixMultiplyMultiVectorsAndRedistributeAsync(
        blas_handle, H, V1, W1, V2, W2, offset, subSize);
    gpu_backend::stream_synchronize<gpu_backend_type>(stream);

    const T One = T(1.0);
    const T Zero = T(0.0);

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_c(), util::op_n(), subSize, subSize,
        W2.l_rows(), &One, W2.l_data() + offset * W2.l_ld(), W2.l_ld(),
        W1.l_data() + offset * W1.l_ld(), W1.l_ld(), &Zero, A->l_data(),
        subSize));

    kernels::extract_upper(A->l_data(), subSize, workspace, subSize,
                           blas_handle);
    gpu_backend::stream_synchronize<gpu_backend_type>(stream);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        workspace, workspace, upperTriangularSize, ncclSum,
        A->getMpiGrid()->get_nccl_row_comm(), &stream));
    gpu_backend::stream_synchronize<gpu_backend_type>(stream);
    kernels::unpack_upper(workspace, subSize, A->l_data(), subSize,
                          blas_handle);
    gpu_backend::stream_synchronize<gpu_backend_type>(stream);

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
        gpu_backend::device_malloc<gpu_backend_type>(&workspace_d, static_cast<std::size_t>(lwork_heevd));
        std::unique_ptr<DT, typename util::device_deleter> workspace_d_ptr(
            workspace_d);

        gpu_mpi::detail::run_heevd_upper<gpu_backend_type, DT>(
            solver_handle, subSize, A_d->l_data(), subSize,
            ritzv_d->l_data() + offset, workspace_d, lwork_heevd, backend_info);
        ritzv.disableDoublePrecision(true);
        A->copyBackSubBlock(0, subSize * subSize);
        gpu_backend::device_synchronize<gpu_backend_type>();
    }
    else
    {
#endif
        gpu_mpi::detail::run_heevd_upper<gpu_backend_type, T>(
            solver_handle, subSize, A->l_data(), subSize,
            ritzv.l_data() + offset, workspace, lwork_heevd, backend_info);
#ifdef RR_DOUBLE_PRECISION
    }
#endif

    typename util::info_type info = -1;
    gpu_backend::copy_to_host<gpu_backend_type>(&info, backend_info, 1);
    if (info != 0)
    {
        throw std::runtime_error("GPU HEEVD failed in RayleighRitz");
    }

    gpu_backend::copy_to_host<gpu_backend_type>(ritzv.cpu_data() + offset, ritzv.l_data() + offset,
                      subSize);

    util::check_blas(blas_dispatch::gemm(
        blas_handle, util::op_n(), util::op_n(), V2.l_rows(),
        subSize, subSize, &One, V2.l_data() + offset * V2.l_ld(), V2.l_ld(),
        A->l_data(), subSize, &Zero, V1.l_data() + offset * V1.l_ld(),
        V1.l_ld()));
}

} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
