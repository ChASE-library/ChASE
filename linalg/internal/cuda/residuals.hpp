#pragma once

#include "residuals.cuh"
#include "external/cublaspp/cublaspp.hpp"
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
                   chase::matrix::Matrix<T, chase::platform::GPU>& H,
                   chase::matrix::Matrix<T, chase::platform::GPU>& V1,
                   chase::Base<T> *d_ritzv, 
                   chase::Base<T> *d_resids, 
                   std::size_t offset,
                   std::size_t subSize,                   
                   chase::matrix::Matrix<T, chase::platform::GPU>* V2 = nullptr)
    {
        SCOPED_NVTX_RANGE();

        if (V2 == nullptr)
        {
            V2 = new chase::matrix::Matrix<T, chase::platform::GPU>(V1.rows(), V1.cols());
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
                                                                H.data(), 
                                                                H.ld(), 
                                                                V1.data() + offset * V1.ld(), 
                                                                V1.ld(), 
                                                                &beta, 
                                                                V2->data() + offset * V2->ld(),
                                                                V2->ld()));

        cudaStream_t usedStream;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &usedStream));

        residual_gpu(V2->rows(), 
                     subSize, 
                     V2->data() +  offset * V2->ld(), 
                     V2->ld(),
                     V1.data() + offset * V1.ld(),
                     V1.ld(), 
                     d_ritzv + offset,
                     d_resids + offset, 
                     true, usedStream);        
    }

}
}
}
}