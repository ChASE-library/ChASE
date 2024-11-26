#pragma once

#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda_aware_mpi/hemm.hpp"
#include "linalg/internal/cuda/residuals.cuh"
#include "../typeTraits.hpp"


namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda_aware_mpi
{

    template <typename MatrixType, typename InputMultiVectorType>
    void residuals(cublasHandle_t cublas_handle,
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
        chase::linalg::internal::cuda_aware_mpi::MatrixMultiplyMultiVectorsAndRedistribute(
                        cublas_handle,
                        H, 
                        V1, 
                        W1, 
                        V2, 
                        W2,
                        offset,
                        subSize);

        cudaStream_t usedStream;
        CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &usedStream));

        chase::linalg::internal::cuda::residual_gpu(W1.l_rows(), 
                                                    subSize, 
                                                    W1.l_data() +  offset * W1.l_ld(), 
                                                    W1.l_ld(),
                                                    W2.l_data() + offset * W2.l_ld(),
                                                    W2.l_ld(), 
                                                    ritzv.data() + offset,
                                                    resids.data() + offset,
                                                    false, usedStream);  
        MPI_Allreduce(MPI_IN_PLACE,
                      resids.data() + offset,
                      subSize,
                      chase::mpi::getMPI_Type<chase::Base<T>>(),
                      MPI_SUM,
                      V1.getMpiGrid()->get_row_comm());

        CHECK_CUDA_ERROR(cudaMemcpy(resids.cpu_data() + offset, resids.data() + offset, subSize * sizeof(chase::Base<T>), cudaMemcpyDeviceToHost ));

        for (auto i = 0; i < subSize; ++i)
        {
            resids.cpu_data()[i + offset] = std::sqrt(resids.cpu_data()[i + offset]);
        }   
    }

}
}
}
}