#pragma once

#include "residuals.cuh"
#include "linalg/cublaspp/cublaspp.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/cuda/cuda_utils.hpp"
#include "Impl/cuda/nvtx.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    template<typename T>
    void residuals(cublasHandle_t cublas_handle, 
                   chase::matrix::MatrixGPU<T>& H,
                   chase::matrix::MatrixGPU<T>& V1,
                   chase::Base<T> *d_ritzv, 
                   chase::Base<T> *d_resids, 
                   std::size_t offset,
                   std::size_t subSize,                   
                   chase::matrix::MatrixGPU<T>* V2 = nullptr)
    {
        SCOPED_NVTX_RANGE();

        if (V2 == nullptr)
        {
            V2 = new chase::matrix::MatrixGPU<T>(V1.rows(), V1.cols());
        }

        T alpha = T(1.0);
        T beta = T(0.0);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                                                                cublas_handle, 
                                                                CUBLAS_OP_C, 
                                                                CUBLAS_OP_N, 
                                                                H.rows(), 
                                                                subSize, 
                                                                H.cols(), 
                                                                &alpha,
                                                                H.gpu_data(), 
                                                                H.gpu_ld(), 
                                                                V1.gpu_data() + offset * V1.gpu_ld(), 
                                                                V1.gpu_ld(), 
                                                                &beta, 
                                                                V2->gpu_data() + offset * V2->gpu_ld(),
                                                                V2->gpu_ld()));

        cudaStream_t usedStream;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &usedStream));

        residual_gpu(V2->rows(), 
                     subSize, 
                     V2->gpu_data() +  offset * V2->gpu_ld(), 
                     V2->gpu_ld(),
                     V1.gpu_data() + offset * V1.gpu_ld(),
                     V1.gpu_ld(), 
                     d_ritzv + offset,
                     d_resids + offset, 
                     true, usedStream);        
    }

}
}
}
}