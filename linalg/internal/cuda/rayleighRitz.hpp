#pragma once

#include "linalg/cublaspp/cublaspp.hpp"
#include "linalg/cusolverpp/cusolverpp.hpp"
#include "Impl/cuda/cuda_utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include "Impl/cuda/nvtx.hpp"

using namespace chase::linalg;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    template<typename T>
    void rayleighRitz(cublasHandle_t cublas_handle, 
                      cusolverDnHandle_t cusolver_handle,
                      chase::matrix::MatrixGPU<T>& H,
                      chase::matrix::MatrixGPU<T>& V1,
                      chase::matrix::MatrixGPU<T>& V2,
                      chase::matrix::MatrixGPU<chase::Base<T>>& ritzv,
                      std::size_t offset,
                      std::size_t subSize,
                      int* devInfo,
                      T *workspace = nullptr,
                      int lwork_heevd = 0,
                      chase::matrix::MatrixGPU<T> *A = nullptr)
    {
        SCOPED_NVTX_RANGE();

        if(A == nullptr)
        {
            A = new chase::matrix::MatrixGPU<T>(subSize, subSize);
        }

        if(workspace == nullptr)
        {
            lwork_heevd = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd_bufferSize(
                                    cusolver_handle, 
                                    CUSOLVER_EIG_MODE_VECTOR, 
                                    CUBLAS_FILL_MODE_LOWER, 
                                    subSize, 
                                    A->gpu_data(), 
                                    subSize, 
                                    ritzv.gpu_data(), 
                                    &lwork_heevd));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork_heevd));
        }

        T One = T(1.0);
        T Zero = T(0.0);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       H.rows(), 
                                       subSize, 
                                       H.cols(), 
                                       &One, 
                                       H.gpu_data(),
                                       H.gpu_ld(),
                                       V1.gpu_data() + offset * V1.gpu_ld(),
                                       V1.gpu_ld(),
                                       &Zero,
                                       V2.gpu_data() + offset * V2.gpu_ld(),
                                       V2.gpu_ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_C, 
                                       CUBLAS_OP_N, 
                                       subSize, 
                                       subSize, 
                                       V2.rows(),
                                       &One, 
                                       V2.gpu_data() + offset * V2.gpu_ld(),
                                       V2.gpu_ld(), 
                                       V1.gpu_data() + offset * V1.gpu_ld(),
                                       V1.gpu_ld(),
                                       &Zero, 
                                       A->gpu_data(),
                                       subSize));
        
        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTheevd(
                                       cusolver_handle, 
                                       CUSOLVER_EIG_MODE_VECTOR, 
                                       CUBLAS_FILL_MODE_LOWER, 
                                       subSize,
                                       A->gpu_data(),
                                       subSize,
                                       ritzv.gpu_data() + offset,
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
                                    ritzv.gpu_data() + offset, 
                                    subSize * sizeof(chase::Base<T>),
                                    cudaMemcpyDeviceToHost));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                       CUBLAS_OP_N, 
                                       CUBLAS_OP_N, 
                                       V1.rows(), 
                                       subSize, 
                                       subSize,
                                       &One, 
                                       V1.gpu_data() + offset * V1.gpu_ld(),
                                       V1.gpu_ld(), 
                                       A->gpu_data(),
                                       subSize, 
                                       &Zero,
                                       V2.gpu_data() + offset * V2.gpu_ld(),
                                       V2.gpu_ld()));


    }    
}
}
}
}