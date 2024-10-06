#pragma once

#include "mpi.h"
#include "Impl/grid/mpiTypes.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/cuda/residuals.cuh"


namespace chase
{
namespace linalg
{
namespace internal
{
namespace nccl
{

    template<typename T>
    void residuals(cublasHandle_t cublas_handle,
                   chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>& V1,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>& V2,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>& W1,
                   chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>& W2,
                   chase::matrix::MatrixGPU<chase::Base<T>>& ritzv, 
                   chase::matrix::MatrixGPU<chase::Base<T>>& resids, 
                   std::size_t offset,
                   std::size_t subSize)
    {

        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::nccl::BlockBlockMultiplyMultiVectorsAndRedistributeAsync<T>(
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