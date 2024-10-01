#pragma once

#include <limits>
#include <iomanip>
#include "linalg/cublaspp/cublaspp.hpp"
#include "linalg/cusolverpp/cusolverpp.hpp"
#include "Impl/cuda/cuda_utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/internal/cuda/absTrace.hpp"
#include "linalg/internal/cuda/shiftDiagonal.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    template<typename T>
    int cholQR1(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                chase::matrix::MatrixGPU<T>& V,
                T *workspace = nullptr,
                int lwork = 0,
                chase::matrix::MatrixGPU<T> *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 1;
        
        if(A == nullptr)
        {
            A = new chase::matrix::MatrixGPU<T>(V.cols(), V.cols());
        }

        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                V.cols(), 
                                                                A->gpu_data(), 
                                                                A->gpu_ld(),
                                                                &lwork));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
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
                                                                  V.cols(), 
                                                                  V.rows(), 
                                                                  &One, 
                                                                  V.gpu_data(), 
                                                                  V.gpu_ld(), 
                                                                  &Zero, 
                                                                  A->gpu_data(), 
                                                                  A->gpu_ld()));
        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         V.cols(),
                                                                         A->gpu_data(),
                                                                         V.cols(), 
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            return info;
        }else
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                    CUBLAS_SIDE_RIGHT, 
                                                                    CUBLAS_FILL_MODE_UPPER,
                                                                    CUBLAS_OP_N, 
                                                                    CUBLAS_DIAG_NON_UNIT, 
                                                                    V.rows(), 
                                                                    V.cols(),
                                                                    &one, 
                                                                    A->gpu_data(), 
                                                                    A->gpu_ld(), 
                                                                    V.gpu_data(), 
                                                                    V.gpu_ld()));
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 1" << std::endl;
#endif                
            return info;
        }
    }


    template<typename T>
    int cholQR2(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                chase::matrix::MatrixGPU<T>& V,
                T *workspace = nullptr,
                int lwork = 0,
                chase::matrix::MatrixGPU<T> *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 1;
        
        if(A == nullptr)
        {
            A = new chase::matrix::MatrixGPU<T>(V.cols(), V.cols());
        }

        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                V.cols(), 
                                                                A->gpu_data(), 
                                                                A->gpu_ld(),
                                                                &lwork));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
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
                                                                  V.cols(), 
                                                                  V.rows(), 
                                                                  &One, 
                                                                  V.gpu_data(), 
                                                                  V.gpu_ld(), 
                                                                  &Zero, 
                                                                  A->gpu_data(), 
                                                                  A->gpu_ld()));
        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         V.cols(),
                                                                         A->gpu_data(),
                                                                         V.cols(), 
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));

        if(info != 0)
        {
            return info;
        }else
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                    CUBLAS_SIDE_RIGHT, 
                                                                    CUBLAS_FILL_MODE_UPPER,
                                                                    CUBLAS_OP_N, 
                                                                    CUBLAS_DIAG_NON_UNIT, 
                                                                    V.rows(), 
                                                                    V.cols(),
                                                                    &one, 
                                                                    A->gpu_data(), 
                                                                    A->gpu_ld(), 
                                                                    V.gpu_data(), 
                                                                    V.gpu_ld()));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                    CUBLAS_FILL_MODE_UPPER, 
                                                                    transa,
                                                                    V.cols(), 
                                                                    V.rows(), 
                                                                    &One, 
                                                                    V.gpu_data(), 
                                                                    V.gpu_ld(), 
                                                                    &Zero, 
                                                                    A->gpu_data(), 
                                                                    A->gpu_ld()));  

            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                            CUBLAS_FILL_MODE_UPPER, 
                                                                            V.cols(),
                                                                            A->gpu_data(),
                                                                            V.cols(), 
                                                                            workspace, 
                                                                            lwork, 
                                                                            devInfo));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                    CUBLAS_SIDE_RIGHT, 
                                                                    CUBLAS_FILL_MODE_UPPER,
                                                                    CUBLAS_OP_N, 
                                                                    CUBLAS_DIAG_NON_UNIT, 
                                                                    V.rows(), 
                                                                    V.cols(),
                                                                    &one, 
                                                                    A->gpu_data(), 
                                                                    A->gpu_ld(), 
                                                                    V.gpu_data(), 
                                                                    V.gpu_ld()));                                                                         
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 2" << std::endl;
#endif                
            return info;
        }
    }

    template<typename T>
    int shiftedcholQR2(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                chase::matrix::MatrixGPU<T>& V,
                T *workspace = nullptr,
                int lwork = 0,
                chase::matrix::MatrixGPU<T> *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);

        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);
        chase::Base<T> shift;

        if(A == nullptr)
        {
            A = new chase::matrix::MatrixGPU<T>(V.cols(), V.cols());
        }

        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                V.cols(), 
                                                                A->gpu_data(), 
                                                                A->gpu_ld(),
                                                                &lwork));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        }

        cublasOperation_t transa;
        if (sizeof(T) == sizeof(Base<T>))
        {
            transa = CUBLAS_OP_T;
        }
        else
        {
            transa = CUBLAS_OP_C;
        }

        int info = 1;

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                  CUBLAS_FILL_MODE_UPPER, 
                                                                  transa,
                                                                  V.cols(), 
                                                                  V.rows(), 
                                                                  &One, 
                                                                  V.gpu_data(), 
                                                                  V.gpu_ld(), 
                                                                  &Zero, 
                                                                  A->gpu_data(), 
                                                                  A->gpu_ld()));

        chase::Base<T> nrmf = 0.0;
        chase::Base<T> *d_nrmf;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_nrmf, sizeof(chase::Base<T>)));
        chase::linalg::internal::cuda::absTrace(*A, d_nrmf);
        CHECK_CUDA_ERROR(cudaMemcpy(&nrmf, d_nrmf, sizeof(chase::Base<T>), cudaMemcpyDeviceToHost));
        shift = std::sqrt(V.rows()) * nrmf * std::numeric_limits<chase::Base<T>>::epsilon();

        chase::linalg::internal::cuda::shiftDiagonal(*A, shift);

        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         V.cols(),
                                                                         A->gpu_data(),
                                                                         V.cols(), 
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));

        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));   

        if(info != 0)
        {
            return info;
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                CUBLAS_SIDE_RIGHT, 
                                                                CUBLAS_FILL_MODE_UPPER,
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_DIAG_NON_UNIT, 
                                                                V.rows(), 
                                                                V.cols(),
                                                                &one, 
                                                                A->gpu_data(), 
                                                                A->gpu_ld(), 
                                                                V.gpu_data(), 
                                                                V.gpu_ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                transa,
                                                                V.cols(), 
                                                                V.rows(), 
                                                                &One, 
                                                                V.gpu_data(), 
                                                                V.gpu_ld(), 
                                                                &Zero, 
                                                                A->gpu_data(), 
                                                                A->gpu_ld()));  

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                        CUBLAS_FILL_MODE_UPPER, 
                                                                        V.cols(),
                                                                        A->gpu_data(),
                                                                        V.cols(), 
                                                                        workspace, 
                                                                        lwork, 
                                                                        devInfo));

        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));   

        if(info != 0)
        {
            return info;
        }

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                CUBLAS_SIDE_RIGHT, 
                                                                CUBLAS_FILL_MODE_UPPER,
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_DIAG_NON_UNIT, 
                                                                V.rows(), 
                                                                V.cols(),
                                                                &one, 
                                                                A->gpu_data(), 
                                                                A->gpu_ld(), 
                                                                V.gpu_data(), 
                                                                V.gpu_ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                transa,
                                                                V.cols(), 
                                                                V.rows(), 
                                                                &One, 
                                                                V.gpu_data(), 
                                                                V.gpu_ld(), 
                                                                &Zero, 
                                                                A->gpu_data(), 
                                                                A->gpu_ld()));  

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                        CUBLAS_FILL_MODE_UPPER, 
                                                                        V.cols(),
                                                                        A->gpu_data(),
                                                                        V.cols(), 
                                                                        workspace, 
                                                                        lwork, 
                                                                        devInfo));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                CUBLAS_SIDE_RIGHT, 
                                                                CUBLAS_FILL_MODE_UPPER,
                                                                CUBLAS_OP_N, 
                                                                CUBLAS_DIAG_NON_UNIT, 
                                                                V.rows(), 
                                                                V.cols(),
                                                                &one, 
                                                                A->gpu_data(), 
                                                                A->gpu_ld(), 
                                                                V.gpu_data(), 
                                                                V.gpu_ld()));    

#ifdef CHASE_OUTPUT
        std::cout << std::setprecision(2) << "choldegree: 2, shift = " << shift << std::endl;
#endif 
        return info;
    }

    template<typename T>
    void houseHoulderQR(cusolverDnHandle_t cusolver_handle,
                        chase::matrix::MatrixGPU<T>& V,
                        T *d_tau,
                        int* devInfo,
                        T *workspace = nullptr,
                        int lwork = 0)
    {
        if(workspace == nullptr)
        {
            int lwork_geqrf = 0;
            int lwork_orgqr = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
                                    cusolver_handle, 
                                    V.rows(),
                                    V.cols(),
                                    V.gpu_data(),
                                    V.gpu_ld(),  
                                    &lwork_geqrf));

            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
                                    cusolver_handle, 
                                    V.rows(),
                                    V.cols(),
                                    V.cols(),
                                    V.gpu_data(),
                                    V.gpu_ld(),
                                    d_tau,  
                                    &lwork_orgqr));

            lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        }

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf(cusolver_handle, 
                                                                         V.rows(), 
                                                                         V.cols(), 
                                                                         V.gpu_data(), 
                                                                         V.gpu_ld(), 
                                                                         d_tau,
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr(cusolver_handle, 
                                                                       V.rows(), 
                                                                       V.cols(), 
                                                                       V.cols(), 
                                                                       V.gpu_data(), 
                                                                       V.gpu_ld(),
                                                                       d_tau, 
                                                                       workspace, 
                                                                       lwork, 
                                                                       devInfo));
    }   
}
}
}
}