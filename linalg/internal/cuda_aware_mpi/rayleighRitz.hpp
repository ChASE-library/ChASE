#pragma once

#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{

    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_mpi::rayleighRitz(cublasHandle_t cublas_handle, 
                      cusolverDnHandle_t cusolver_handle,
                      MatrixType& H,
                      InputMultiVectorType& V1,
                      InputMultiVectorType& V2,
                      typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                      typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,  
                      chase::distMatrix::RedundantMatrix<chase::Base<typename MatrixType::value_type>, chase::platform::GPU>& ritzv, 
                      std::size_t offset,
                      std::size_t subSize,
                      int* devInfo,
                      typename MatrixType::value_type *workspace,
                      int lwork_heevd,
                      chase::distMatrix::RedundantMatrix<typename MatrixType::value_type, chase::platform::GPU>* A                                         
                    )                
    {
        using T = typename MatrixType::value_type;

        std::cout <<"MPI Backend" << std::endl;

        std::unique_ptr<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>> A_ptr;
        std::size_t upperTriangularSize = std::size_t(subSize * (subSize + 1) / 2);

        if (A == nullptr) {
            // Allocate A if not provided
            A_ptr = std::make_unique<chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>>(subSize, subSize, V1.getMpiGrid_shared_ptr());
            A = A_ptr.get();
        }

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
        if(workspace == nullptr)
        {
            lwork_heevd = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                    cusolver_handle, 
                                    CUSOLVER_EIG_MODE_VECTOR, 
                                    CUBLAS_FILL_MODE_LOWER, 
                                    subSize, 
                                    A->l_data(), 
                                    subSize, 
                                    ritzv.l_data(), 
                                    &lwork_heevd));

            if(upperTriangularSize > lwork_heevd)
            {
                lwork_heevd = upperTriangularSize;
            }      

            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork_heevd));
            work_ptr.reset(workspace);
            workspace = work_ptr.get();            
        }
        // Perform the distributed matrix-matrix multiplication
        chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(
                        cublas_handle,
                        H, 
                        V1, 
                        W1, 
                        V2, 
                        W2,
                        offset,
                        subSize);

        T One = T(1.0);
        T Zero = T(0.0);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       subSize, 
                                       subSize, 
                                       W2.l_rows(),
                                       &One, 
                                       W2.l_data() + offset * W2.l_ld(),
                                       W2.l_ld(), 
                                       W1.l_data() + offset * W1.l_ld(), 
                                       W1.l_ld(),
                                       &Zero, 
                                       A->l_data(),
                                       subSize));


        chase::linalg::internal::cuda::extractUpperTriangular(A->l_data(), subSize, workspace, subSize);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        MPI_Allreduce(MPI_IN_PLACE, 
                      workspace, 
                      upperTriangularSize, 
                      chase::mpi::getMPI_Type<T>(),
                      MPI_SUM,
                      A->getMpiGrid()->get_row_comm());

        chase::linalg::internal::cuda::unpackUpperTriangular(workspace, subSize, A->l_data(), subSize);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd(
                                       cusolver_handle, 
                                       CUSOLVER_EIG_MODE_VECTOR, 
                                       CUBLAS_FILL_MODE_UPPER, 
                                       subSize,
                                       A->l_data(),
                                       subSize,
                                       ritzv.l_data() + offset,
                                       workspace, lwork_heevd, devInfo           
        ));    

        int info;
        CHECK_CUDA_ERROR(cudaMemcpy(&info, 
                                    devInfo, 
                                    1 * sizeof(int),
                                    cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            throw std::runtime_error("cusolver HEEVD failed in RayleighRitz");
        }

        CHECK_CUDA_ERROR(cudaMemcpy(ritzv.cpu_data() + offset, 
                                    ritzv.l_data() + offset, 
                                    subSize * sizeof(chase::Base<T>),
                                    cudaMemcpyDeviceToHost));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       V2.l_rows(),
                                       subSize, 
                                       subSize,
                                       &One, 
                                       V2.l_data() + offset * V2.l_ld(),
                                       V2.l_ld(), 
                                       A->l_data(),
                                       subSize, 
                                       &Zero,
                                       V1.l_data() + offset * V1.l_ld(),
                                       V1.l_ld()));
    }

}
}
}