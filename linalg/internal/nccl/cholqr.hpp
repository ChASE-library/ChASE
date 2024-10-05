#pragma once

#include <limits>
#include <iomanip>
#include "mpi.h"
#include "Impl/grid/mpiTypes.hpp"
#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

#include "linalg/cublaspp/cublaspp.hpp"
#include "linalg/cusolverpp/cusolverpp.hpp"
#include "linalg/internal/cuda/absTrace.cuh"
#include "linalg/internal/cuda/shiftDiagonal.cuh"

using namespace chase::linalg::blaspp;
using namespace chase::linalg::lapackpp;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace nccl
{
    template<typename T>
    int cholQR1(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                std::size_t m, 
                std::size_t n, 
                T *V, 
                int ldv, 
                //MPI_Comm comm,
                ncclComm_t comm,
                T *workspace = nullptr,
                int lwork = 0,                
                T *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 1;

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
        if(A == nullptr)
        {
            CHECK_CUDA_ERROR(cudaMalloc(&A, n * n * sizeof(T))); 
            A_ptr.reset(A);
            A = A_ptr.get();          
        }

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                n, 
                                                                A, 
                                                                n,
                                                                &lwork));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
            work_ptr.reset(workspace);
            workspace = work_ptr.get();
        }

        cublasOperation_t transa;
        if constexpr (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value)
        {
            transa = CUBLAS_OP_C;
        }
        else
        {
            transa = CUBLAS_OP_T;
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                  CUBLAS_FILL_MODE_UPPER, 
                                                                  transa,
                                                                  n, 
                                                                  m, 
                                                                  &One, 
                                                                  V, 
                                                                  ldv, 
                                                                  &Zero, 
                                                                  A, 
                                                                  n));

        CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));

        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         n,
                                                                         A,
                                                                         n, 
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));                    
   
        if(info != 0)
        {
            return info;
        }
        else
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                    CUBLAS_SIDE_RIGHT, 
                                                                    CUBLAS_FILL_MODE_UPPER,
                                                                    CUBLAS_OP_N, 
                                                                    CUBLAS_DIAG_NON_UNIT, 
                                                                    m, 
                                                                    n,
                                                                    &one, 
                                                                    A, 
                                                                    n, 
                                                                    V, 
                                                                    ldv));
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 1" << std::endl;
#endif                
            return info;
        }   
        CHECK_CUDA_ERROR(cudaFree(devInfo));
    }


    template<typename T>
    int cholQR2(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                std::size_t m, 
                std::size_t n, 
                T *V, 
                int ldv, 
                ncclComm_t comm,
                T *workspace = nullptr,
                int lwork = 0,                
                T *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 0;

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
        if(A == nullptr)
        {
            CHECK_CUDA_ERROR(cudaMalloc(&A, n * n * sizeof(T))); 
            A_ptr.reset(A);
            A = A_ptr.get();          
        }

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                n, 
                                                                A, 
                                                                n,
                                                                &lwork));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
            work_ptr.reset(workspace);
            workspace = work_ptr.get();
        }

        cublasOperation_t transa;
        if constexpr (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value)
        {
            transa = CUBLAS_OP_C;
        }
        else
        {
            transa = CUBLAS_OP_T;
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                  CUBLAS_FILL_MODE_UPPER, 
                                                                  transa,
                                                                  n, 
                                                                  m, 
                                                                  &One, 
                                                                  V, 
                                                                  ldv, 
                                                                  &Zero, 
                                                                  A, 
                                                                  n));

        CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));

        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         n,
                                                                         A,
                                                                         n, 
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));                    
   
        if(info != 0)
        {
            return info;
        }
        else
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                    CUBLAS_SIDE_RIGHT, 
                                                                    CUBLAS_FILL_MODE_UPPER,
                                                                    CUBLAS_OP_N, 
                                                                    CUBLAS_DIAG_NON_UNIT, 
                                                                    m, 
                                                                    n,
                                                                    &one, 
                                                                    A, 
                                                                    n, 
                                                                    V, 
                                                                    ldv));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                    CUBLAS_FILL_MODE_UPPER, 
                                                                    transa,
                                                                    n, 
                                                                    m, 
                                                                    &One, 
                                                                    V, 
                                                                    ldv, 
                                                                    &Zero, 
                                                                    A, 
                                                                    n));

            CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));

            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                            CUBLAS_FILL_MODE_UPPER, 
                                                                            n,
                                                                            A,
                                                                            n, 
                                                                            workspace, 
                                                                            lwork, 
                                                                            devInfo));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                    CUBLAS_SIDE_RIGHT, 
                                                                    CUBLAS_FILL_MODE_UPPER,
                                                                    CUBLAS_OP_N, 
                                                                    CUBLAS_DIAG_NON_UNIT, 
                                                                    m, 
                                                                    n,
                                                                    &one, 
                                                                    A, 
                                                                    n, 
                                                                    V, 
                                                                    ldv));
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 2" << std::endl;
#endif                
            return info;
        }   
        CHECK_CUDA_ERROR(cudaFree(devInfo));
    }

    template<typename T>
    int shiftedcholQR2(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                std::size_t N, 
                std::size_t m, 
                std::size_t n, 
                T *V, 
                int ldv, 
                ncclComm_t comm,
                T *workspace = nullptr,
                int lwork = 0,                
                T *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);
        chase::Base<T> shift;

        int info = 1;

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
        if(A == nullptr)
        {
            CHECK_CUDA_ERROR(cudaMalloc(&A, n * n * sizeof(T))); 
            A_ptr.reset(A);
            A = A_ptr.get();          
        }

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> work_ptr = nullptr;
        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                n, 
                                                                A, 
                                                                n,
                                                                &lwork));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
            work_ptr.reset(workspace);
            workspace = work_ptr.get();
        }

        cublasOperation_t transa;
        if constexpr (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value)
        {
            transa = CUBLAS_OP_C;
        }
        else
        {
            transa = CUBLAS_OP_T;
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                  CUBLAS_FILL_MODE_UPPER, 
                                                                  transa,
                                                                  n, 
                                                                  m, 
                                                                  &One, 
                                                                  V, 
                                                                  ldv, 
                                                                  &Zero, 
                                                                  A, 
                                                                  n));

        CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));
        
        chase::Base<T> nrmf = 0.0;
        chase::Base<T> *d_nrmf;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nrmf, sizeof(chase::Base<T>)));
        chase::linalg::internal::cuda::absTrace_gpu(A, d_nrmf, n, n, (cudaStream_t)0);
        CHECK_CUDA_ERROR(cudaMemcpy(&nrmf, d_nrmf, sizeof(chase::Base<T>), cudaMemcpyDeviceToHost));
        shift = std::sqrt(N) * nrmf * std::numeric_limits<chase::Base<T>>::epsilon();

        chase::linalg::internal::cuda::chase_shift_matrix(A, n, n, shift, (cudaStream_t)0);

        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         n,
                                                                         A,
                                                                         n, 
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));                    
   
        if(info != 0)
        {
            std::cout << "1" << std::endl;
            return info;
        }
    
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                CUBLAS_SIDE_RIGHT, 
                                                                CUBLAS_FILL_MODE_UPPER,
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_DIAG_NON_UNIT, 
                                                                m, 
                                                                n,
                                                                &one, 
                                                                A, 
                                                                n, 
                                                                V, 
                                                                ldv));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                transa,
                                                                n, 
                                                                m, 
                                                                &One, 
                                                                V, 
                                                                ldv, 
                                                                &Zero, 
                                                                A, 
                                                                n));

        CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                        CUBLAS_FILL_MODE_UPPER, 
                                                                        n,
                                                                        A,
                                                                        n, 
                                                                        workspace, 
                                                                        lwork, 
                                                                        devInfo));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                CUBLAS_SIDE_RIGHT, 
                                                                CUBLAS_FILL_MODE_UPPER,
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_DIAG_NON_UNIT, 
                                                                m, 
                                                                n,
                                                                &one, 
                                                                A, 
                                                                n, 
                                                                V, 
                                                                ldv));
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                    CUBLAS_FILL_MODE_UPPER, 
                                                                    transa,
                                                                    n, 
                                                                    m, 
                                                                    &One, 
                                                                    V, 
                                                                    ldv, 
                                                                    &Zero, 
                                                                    A, 
                                                                    n));

        CHECK_NCCL_ERROR(chase::Impl::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                        CUBLAS_FILL_MODE_UPPER, 
                                                                        n,
                                                                        A,
                                                                        n, 
                                                                        workspace, 
                                                                        lwork, 
                                                                        devInfo));
        
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                CUBLAS_SIDE_RIGHT, 
                                                                CUBLAS_FILL_MODE_UPPER,
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_DIAG_NON_UNIT, 
                                                                m, 
                                                                n,
                                                                &one, 
                                                                A, 
                                                                n, 
                                                                V, 
                                                                ldv));
#ifdef CHASE_OUTPUT
        std::cout << std::setprecision(2) << "choldegree: 2, shift = " << shift << std::endl;
#endif                        
        CHECK_CUDA_ERROR(cudaFree(devInfo));
        
        CHECK_CUDA_ERROR(cudaFree(d_nrmf));

        return info;
       
    }

}
}
}
}