// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "../typeTraits.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda/residuals.cuh"
#include "linalg/internal/cuda/lanczos_kernels.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/nccl/nccl_kernels.hpp"
#include "mpi.h"

namespace chase
{
namespace linalg
{
namespace internal
{
template <typename MatrixType, typename InputMultiVectorType>
void cuda_nccl::residuals(
    cublasHandle_t cublas_handle, MatrixType& H, InputMultiVectorType& V1,
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

    // ========================================================================
    // GPU-RESIDENT OPTIMIZED RESIDUALS
    // Uses custom stream, async operations, and GPU sqrt
    // ========================================================================
    
    // Create dedicated stream for residuals computation
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Perform the distributed matrix-matrix multiplication
    chase::linalg::internal::cuda_nccl::
        MatrixMultiplyMultiVectorsAndRedistributeAsync(cublas_handle, H, V1, W1,
                                                       V2, W2, offset, subSize);
    
    // Compute residuals (squared norms) on GPU with custom stream
    // residuals[i] = ||W1[:,i] - ritzv[i] * W2[:,i]||² (without sqrt yet)
    chase::linalg::internal::cuda::residual_gpu(
        W1.l_rows(), subSize, 
        W1.l_data() + offset * W1.l_ld(), W1.l_ld(),
        W2.l_data() + offset * W2.l_ld(), W2.l_ld(), 
        ritzv.data() + offset,
        resids.data() + offset, 
        false,  // Don't compute sqrt yet (we'll do it after allreduce)
        stream);

    // NCCL Allreduce: sum squared residuals across row communicator (GPU-GPU)
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<RealT>(
        resids.data() + offset, resids.data() + offset, subSize, ncclSum,
        V1.getMpiGrid()->get_nccl_row_comm(), &stream));

    // Compute sqrt on GPU in parallel (batched kernel)
    using chase::linalg::internal::cuda::batchedSqrt;
    batchedSqrt(resids.data() + offset, subSize, &stream);

    // Async copy to CPU (only when needed for output)
    cudaMemcpyAsync(
        resids.cpu_data() + offset, resids.data() + offset,
        subSize * sizeof(RealT), cudaMemcpyDeviceToHost, stream);

    // Synchronize stream to ensure all operations complete
    cudaStreamSynchronize(stream);
    
    // Clean up
    cudaStreamDestroy(stream);
}

} // namespace internal
} // namespace linalg
} // namespace chase