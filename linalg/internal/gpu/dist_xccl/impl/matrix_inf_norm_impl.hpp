// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "grid/gpu_collective_config.hpp"
#include "grid/nccl_utils.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_xccl
{

/**
 * Global ||H||_∞ for a 2D-distributed GPU matrix:
 *   local abs row-sums → NCCL/RCCL SUM on row_comm → local max → MAX on col_comm.
 *
 * @param d_row_sums scratch of length H.l_rows() (device, Base<T>)
 * @param d_result   scalar infinity-norm (device, Base<T>), same on all ranks
 */
template <typename GpuBackend, typename MatrixType>
void matrixInfNorm_impl(
    MatrixType& H,
    chase::Base<typename MatrixType::value_type>* d_row_sums,
    chase::Base<typename MatrixType::value_type>* d_result,
    GpuCollectiveStream_t stream)
{
    using T = typename MatrixType::value_type;
    using kernels = gpu_device::kernels<GpuBackend>;

    kernels::matrix_row_abs_sums(H.l_data(), H.l_rows(), H.l_cols(), H.l_ld(),
                                 d_row_sums, stream);

    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_row_sums, d_row_sums, H.l_rows(), ncclSum,
        H.getMpiGrid()->get_nccl_row_comm(), &stream));

    kernels::template matrix_row_sums_max<T>(d_row_sums, H.l_rows(), d_result,
                                             stream);

    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        d_result, d_result, 1, ncclMax, H.getMpiGrid()->get_nccl_col_comm(),
        &stream));
}

} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
