#pragma once

#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/cuda/residuals.cuh"
#include "linalg/internal/nccl/nccl_kernels.hpp"
#include "../typeTraits.hpp"


namespace chase
{
namespace linalg
{
namespace internal
{
    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_nccl::residuals(cublasHandle_t cublas_handle,
                   MatrixType& H,
                   InputMultiVectorType& V1,
                   InputMultiVectorType& V2,
                   typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                   typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                   chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>, typename MatrixType::platform_type>& ritzv,
                   chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>, typename MatrixType::platform_type>& resids,
                   std::size_t offset,
                   std::size_t subSize)     
    {

        using T = typename MatrixType::value_type;

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
                        cublas_handle,
                        H, 
                        V1, 
                        W1, 
                        V2, 
                        W2,
                        offset,
                        subSize);
        
        chase::linalg::internal::cuda::residual_gpu(W1.l_rows(), 
                                                    subSize, 
                                                    W1.l_data() +  offset * W1.l_ld(), 
                                                    W1.l_ld(),
                                                    W2.l_data() + offset * W2.l_ld(),
                                                    W2.l_ld(), 
                                                    ritzv.data() + offset,
                                                    resids.data() + offset,
                                                    false, (cudaStream_t)0);  
        
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<chase::Base<T>>(resids.data() + offset,
                                                                    resids.data() + offset,
                                                                    subSize, 
                                                                    ncclSum, 
                                                                    V1.getMpiGrid()->get_nccl_row_comm()));

        CHECK_CUDA_ERROR(cudaMemcpy(resids.cpu_data() + offset, resids.data() + offset, subSize * sizeof(chase::Base<T>), cudaMemcpyDeviceToHost ));

        for (auto i = 0; i < subSize; ++i)
        {
            resids.cpu_data()[i + offset] = std::sqrt(resids.cpu_data()[i + offset]);
        }   
    }

}
}
}