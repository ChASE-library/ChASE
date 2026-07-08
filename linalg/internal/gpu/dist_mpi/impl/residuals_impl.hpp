// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/internal/gpu/dist_mpi/impl/mpi_collectives.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/typeTraits.hpp"
#include "linalg/matrix/matrix.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <mpi.h>

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

template <typename Backend>
struct residuals_backend
{
    template <typename Real>
    static void allreduce_sum_row(Real* device_buf, std::size_t count,
                                  MPI_Comm comm)
    {
        gpu_mpi::mpiAllreduceSum(device_buf, count, comm);
    }
};

} // namespace detail

template <typename Backend, typename MpiKernelNamespace, typename BlasHandle,
          typename MatrixType, typename InputMultiVectorType>
void residuals_impl(
    BlasHandle blas_handle, MatrixType& H, InputMultiVectorType& V1,
    InputMultiVectorType& V2,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
    chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                          typename MatrixType::platform_type>& ritzv,
    chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>,
                          typename MatrixType::platform_type>& resids,
    std::size_t offset, std::size_t subSize,
    bool broadcast_resids_to_all_ranks)
{
    using T = typename MatrixType::value_type;
    using Real = chase::Base<T>;
    using kernel_ops = gpu_device::kernels<Backend>;
    using util = gpu_backend::handle_utils<Backend>;

    MpiKernelNamespace::MatrixMultiplyMultiVectorsAndRedistribute(
        blas_handle, H, V1, W1, V2, W2, offset, subSize);

    kernel_ops::residual_gpu(
        W1.l_rows(), subSize, W1.l_data() + offset * W1.l_ld(), W1.l_ld(),
        W2.l_data() + offset * W2.l_ld(), W2.l_ld(), ritzv.data() + offset,
        resids.data() + offset);
    gpu_backend::device_synchronize<Backend>();

    detail::residuals_backend<Backend>::allreduce_sum_row(
        resids.data() + offset, subSize, V1.getMpiGrid()->get_row_comm());

    gpu_backend::copy_to_host<Backend>(resids.cpu_data() + offset,
                              resids.data() + offset, subSize);

    for (std::size_t i = 0; i < subSize; ++i)
    {
        resids.cpu_data()[i + offset] =
            std::sqrt(resids.cpu_data()[i + offset]);
    }

    if (broadcast_resids_to_all_ranks)
    {
        MPI_Bcast(resids.cpu_data() + offset, static_cast<int>(subSize),
                  chase::mpi::getMPI_Type<Real>(), 0,
                  V1.getMpiGrid()->get_comm());
        gpu_backend::copy_to_device<Backend>(resids.data() + offset,
                                    resids.cpu_data() + offset, subSize);
    }
}

} // namespace gpu_mpi
} // namespace internal
} // namespace linalg
} // namespace chase
