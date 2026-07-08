// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "grid/gpu_collective_config.hpp"
#include "grid/nccl_utils.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"
#include "linalg/internal/typeTraits.hpp"
#include "linalg/matrix/matrix.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace gpu_xccl
{

/**
 * GPU-resident residuals for device collective distributed paths.
 *
 * Algorithm matches linalg/internal/gpu/dist_xccl/nccl/residuals.hpp (NVIDIA collective reference):
 * dedicated stream, async HEMM redistribute, GPU residual norms without sqrt,
 * collective row allreduce, GPU batched sqrt, async D2H, stream sync.
 */
template <typename GpuBackend, typename CollectiveKernelNamespace, typename BlasHandle,
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
    std::size_t offset, std::size_t subSize)
{
    using T = typename MatrixType::value_type;
    using RealT = chase::Base<T>;
    using kernels = gpu_device::kernels<GpuBackend>;

    auto stream = gpu_backend::create_stream<GpuBackend>();

    CollectiveKernelNamespace::MatrixMultiplyMultiVectorsAndRedistributeAsync(
        blas_handle, H, V1, W1, V2, W2, offset, subSize);

    kernels::residual_gpu(W1.l_rows(), subSize,
                          W1.l_data() + offset * W1.l_ld(), W1.l_ld(),
                          W2.l_data() + offset * W2.l_ld(), W2.l_ld(),
                          ritzv.data() + offset, resids.data() + offset, false,
                          stream);

    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        resids.data() + offset, resids.data() + offset, subSize, ncclSum,
        V1.getMpiGrid()->get_nccl_row_comm(), &stream));

    kernels::template sqrt<RealT>(resids.data() + offset,
                                    static_cast<int>(subSize), &stream);

    gpu_backend::memcpy_async_device_to_host<GpuBackend>(
        resids.cpu_data() + offset, resids.data() + offset, subSize, stream);

    gpu_backend::stream_synchronize<GpuBackend>(stream);
    gpu_backend::destroy_stream<GpuBackend>(stream);
}

} // namespace gpu_xccl
} // namespace internal
} // namespace linalg
} // namespace chase
