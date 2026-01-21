// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
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

    // Perform the distributed matrix-matrix multiplication
    chase::linalg::internal::cuda_nccl::
        MatrixMultiplyMultiVectorsAndRedistributeAsync(cublas_handle, H, V1, W1,
                                                       V2, W2, offset, subSize);
    /*
            if constexpr (std::is_same<T, std::complex<float>>::value)
            {
                W1.enableDoublePrecision();
                W2.enableDoublePrecision();

                auto W1_d = W1.getDoublePrecisionMatrix();
                auto W2_d = W2.getDoublePrecisionMatrix();

                double *resids_d;
                double *ritzv_d;

                std::vector<double> resids_d_cpu(subSize);
                cudaMalloc((void**)&resids_d, sizeof(double) * subSize);
                cudaMalloc((void**)&ritzv_d, sizeof(double) * subSize);

                chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(ritzv.data()
       + offset, ritzv_d, subSize);
                chase::linalg::internal::cuda::convert_SP_TO_DP_GPU(resids.data()
       + offset, resids_d, subSize);

                chase::linalg::internal::cuda::residual_gpu(W1_d->l_rows(),
                                                            subSize,
                                                            W1_d->l_data() +
       offset * W1_d->l_ld(), W1_d->l_ld(), W2_d->l_data() + offset *
       W2_d->l_ld(), W2_d->l_ld(), ritzv_d, resids_d, false, (cudaStream_t)0);

                CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<double>(resids_d,
       resids_d, subSize, ncclSum, V1.getMpiGrid()->get_nccl_row_comm()));

                CHECK_CUDA_ERROR(cudaMemcpy(resids_d_cpu.data(), resids_d,
       subSize * sizeof(double), cudaMemcpyDeviceToHost));

                for (auto i = 0; i < subSize; ++i)
                {
                    resids.cpu_data()[i + offset] =
       float(std::sqrt(resids_d_cpu[i]));
                }

                W1.disableDoublePrecision();
                W2.disableDoublePrecision();
                cudaFree(resids_d);
                cudaFree(ritzv_d);

            }else*/
    {
        chase::linalg::internal::cuda::residual_gpu(
            W1.l_rows(), subSize, W1.l_data() + offset * W1.l_ld(), W1.l_ld(),
            W2.l_data() + offset * W2.l_ld(), W2.l_ld(), ritzv.data() + offset,
            resids.data() + offset, false, (cudaStream_t)0);

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<chase::Base<T>>(
            resids.data() + offset, resids.data() + offset, subSize, ncclSum,
            V1.getMpiGrid()->get_nccl_row_comm()));

        CHECK_CUDA_ERROR(cudaMemcpy(
            resids.cpu_data() + offset, resids.data() + offset,
            subSize * sizeof(chase::Base<T>), cudaMemcpyDeviceToHost));

        for (auto i = 0; i < subSize; ++i)
        {
            resids.cpu_data()[i + offset] =
                std::sqrt(resids.cpu_data()[i + offset]);
        }
    }
}

} // namespace internal
} // namespace linalg
} // namespace chase