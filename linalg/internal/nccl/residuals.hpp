#pragma once

#include "mpi.h"
#include "Impl/grid/mpiTypes.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/cuda/residuals.cuh"
#include "../typeTraits.hpp"


namespace chase
{
namespace linalg
{
namespace internal
{
namespace nccl
{

    template <typename MatrixType, typename InputMultiVectorType>
    void residuals(cublasHandle_t cublas_handle,
                   MatrixType& H,
                   InputMultiVectorType& V1,
                   InputMultiVectorType& V2,
                   typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                   typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                   chase::matrix::MatrixGPU<chase::Base<typename MatrixType::value_type>>& ritzv,
                   chase::matrix::MatrixGPU<chase::Base<typename MatrixType::value_type>>& resids,
                   std::size_t offset,
                   std::size_t subSize)     
    {

        using T = typename MatrixType::value_type;

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::nccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
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
                                                    ritzv.gpu_data() + offset,
                                                    resids.gpu_data() + offset,
                                                    false, (cudaStream_t)0);  
        
        CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<chase::Base<T>>(resids.gpu_data() + offset,
                                                                    resids.gpu_data() + offset,
                                                                    subSize, 
                                                                    ncclSum, 
                                                                    V1.getMpiGrid()->get_nccl_row_comm()));

        CHECK_CUDA_ERROR(cudaMemcpy(resids.cpu_data() + offset, resids.gpu_data() + offset, subSize * sizeof(chase::Base<T>), cudaMemcpyDeviceToHost ));

        for (auto i = 0; i < subSize; ++i)
        {
            resids.cpu_data()[i + offset] = std::sqrt(resids.cpu_data()[i + offset]);
        }   
    }

}
}
}
}