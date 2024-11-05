#pragma once

#include <limits>
#include <iomanip>
#include "mpi.h"
#include "grid/mpiTypes.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
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
        std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

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
            if(upperTriangularSize > lwork)
            {
                lwork = upperTriangularSize;
            }

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

        chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
        //CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(workspace, workspace, upperTriangularSize, ncclSum, comm));
        chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);

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
            CHECK_CUDA_ERROR(cudaFree(devInfo));
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

            int grank;
            MPI_Comm_rank(MPI_COMM_WORLD, &grank);
            if(grank == 0)
            {
                std::cout << "choldegree: 1" << std::endl;
            }
#endif
            CHECK_CUDA_ERROR(cudaFree(devInfo));
            return info;
        }   
        
    }


    template<typename InputMultiVectorType>
    int cholQR1(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                InputMultiVectorType& V, 
                typename InputMultiVectorType::value_type *workspace = nullptr,
                int lwork = 0,                
                typename InputMultiVectorType::value_type *A = nullptr)
    {
        using T = typename InputMultiVectorType::value_type;

        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 1;

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
        if(A == nullptr)
        {
            CHECK_CUDA_ERROR(cudaMalloc(&A, V.l_cols() * V.l_cols() * sizeof(T))); 
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
                                                                V.l_cols(), 
                                                                A, 
                                                                V.l_cols(),
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
                                                                  V.l_cols(), 
                                                                  V.l_rows(), 
                                                                  &One, 
                                                                  V.l_data(), 
                                                                  V.l_ld(), 
                                                                  &Zero, 
                                                                  A, 
                                                                  V.l_cols()));

        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(A, A, V.l_cols() * V.l_cols(), ncclSum, V.getMpiGrid()->get_nccl_col_comm()));

        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         V.l_cols(),
                                                                         A,
                                                                         V.l_cols(), 
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));
        CHECK_CUDA_ERROR(cudaMemcpy(&info, devInfo, 1 * sizeof(int), cudaMemcpyDeviceToHost));                    
   
        if(info != 0)
        {
            CHECK_CUDA_ERROR(cudaFree(devInfo));
            return info;
        }
        else
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTtrsm(cublas_handle, 
                                                                    CUBLAS_SIDE_RIGHT, 
                                                                    CUBLAS_FILL_MODE_UPPER,
                                                                    CUBLAS_OP_N, 
                                                                    CUBLAS_DIAG_NON_UNIT, 
                                                                    V.l_rows(), 
                                                                    V.l_cols(),
                                                                    &one, 
                                                                    A, 
                                                                    V.l_cols(), 
                                                                    V.l_data(), 
                                                                    V.l_ld()));
#ifdef CHASE_OUTPUT

            int grank;
            MPI_Comm_rank(MPI_COMM_WORLD, &grank);
            if(grank == 0)
            {
                std::cout << "choldegree: 1" << std::endl;
            }
#endif
            CHECK_CUDA_ERROR(cudaFree(devInfo));
            return info;
        }   
        
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
        std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

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
            if(upperTriangularSize > lwork)
            {
                lwork = upperTriangularSize;
            }
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

        chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(workspace, workspace, upperTriangularSize, ncclSum, comm));
        chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);

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
            CHECK_CUDA_ERROR(cudaFree(devInfo));
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

            chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
            CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(workspace, workspace, upperTriangularSize, ncclSum, comm));
            chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
            
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
            int grank;
            MPI_Comm_rank(MPI_COMM_WORLD, &grank);
            if(grank == 0)
            {
                std::cout << "choldegree: 2" << std::endl;
            }   
#endif                        
            CHECK_CUDA_ERROR(cudaFree(devInfo));
            return info;
        }   
    }


    template<typename InputMultiVectorType>
    int cholQR2(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                InputMultiVectorType& V, 
                typename InputMultiVectorType::value_type *workspace = nullptr,
                int lwork = 0,                
                typename InputMultiVectorType::value_type *A = nullptr)
    {
        using T = typename InputMultiVectorType::value_type;

        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 0;

        std::unique_ptr<T, chase::cuda::utils::CudaDeleter> A_ptr = nullptr;
        if(A == nullptr)
        {
            CHECK_CUDA_ERROR(cudaMalloc(&A, V.l_cols() * V.l_cols() * sizeof(T))); 
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
                                                                V.l_cols(), 
                                                                A, 
                                                                V.l_cols(),
                                                                &lwork));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
            work_ptr.reset(workspace);
            workspace = work_ptr.get();
        }

#ifdef ENABLE_MIXED_PRECISION
        if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value)
        {
            std::cout << "In cholqr2, the first cholqr using Single Precision" << std::endl;
            using singlePrecisionT = typename chase::ToSinglePrecisionTrait<T>::Type;
            V.enableSinglePrecision();
            auto V_sp = V.getSinglePrecisionMatrix();
            info = cholQR1<singlePrecisionT>(cublas_handle, cusolver_handle, *V_sp);
            V.disableSinglePrecision(true);
        }else
        {
            info = cholQR1<T>(cublas_handle, cusolver_handle, V, workspace, lwork, A);
        }      
#else
        info = cholQR1<T>(cublas_handle, cusolver_handle, V, workspace, lwork, A);
#endif
        if(info != 0)
        {
            return info;
        }

        info = cholQR1<T>(cublas_handle, cusolver_handle, V, workspace, lwork, A);

        return info;       
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
        std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

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
            if(upperTriangularSize > lwork)
            {
                lwork = upperTriangularSize;
            }

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

        chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(workspace, workspace, upperTriangularSize, ncclSum, comm));
        chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);
                
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

        //CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));
        chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(workspace, workspace, upperTriangularSize, ncclSum, comm));
        chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);

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

        //CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(A, A, n * n, ncclSum, comm));
        chase::linalg::internal::cuda::extractUpperTriangular(A, n, workspace, n);
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(workspace, workspace, upperTriangularSize, ncclSum, comm));
        chase::linalg::internal::cuda::unpackUpperTriangular(workspace, n, A, n);

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
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if(grank == 0)
        {
            std::cout << std::setprecision(2) << "choldegree: 2, shift = " << shift << std::endl;
        }
#endif                        
        CHECK_CUDA_ERROR(cudaFree(devInfo));
        
        CHECK_CUDA_ERROR(cudaFree(d_nrmf));

        return info;
       
    }

    template<typename T>
    int modifiedGramSchmidtCholQR(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                std::size_t m, 
                std::size_t n, 
                std::size_t locked,
                T *V, 
                std::size_t ldv, 
                ncclComm_t comm,
                T *workspace = nullptr,
                int lwork = 0,                
                T *A = nullptr)
    {
        T one = T(1.0);
        T negone = T(-1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);        
        int info = 1;

        int number_of_panels = 6;
        size_t panel_size = ceil((double)n / number_of_panels);
        size_t panel_size_rest;
        std::size_t upperTriangularSize = std::size_t(n * (n + 1) / 2);

        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));
        
        cublasOperation_t transa;
        if constexpr (std::is_same<T, std::complex<float>>::value || std::is_same<T, std::complex<double>>::value)
        {
            transa = CUBLAS_OP_C;
        }
        else
        {
            transa = CUBLAS_OP_T;
        }

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
            if(upperTriangularSize > lwork)
            {
                lwork = upperTriangularSize;
            }

            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
            work_ptr.reset(workspace);
            workspace = work_ptr.get();
        }

        if (locked < panel_size)
        {
            info = cholQR2(cublas_handle,
                           cusolver_handle,
                           m, 
                           panel_size, 
                           V, 
                           ldv, 
                           comm,
                           workspace,
                           lwork = 0,                
                           A);
            if(info != 0)
            {
                return info;
            }
        }else
        {
            panel_size = locked;
            number_of_panels = ceil((double)n / locked);
        }

        for(auto j = 1; j < number_of_panels; ++j )
        {
            panel_size_rest = (j == number_of_panels-1) ? n - (j) * panel_size: panel_size;   

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_C, 
                                        CUBLAS_OP_N, 
                                        panel_size_rest,
                                        n - j * panel_size, 
                                        m,
                                        &one, 
                                        V + (j - 1) * panel_size * ldv,
                                        ldv, 
                                        V + j * panel_size * ldv,
                                        ldv, 
                                        &zero,
                                        A,
                                        panel_size_rest));

            CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(A, A, (n - j * panel_size) * panel_size_rest, ncclSum, comm));


            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_N, 
                                        CUBLAS_OP_N, 
                                        m,
                                        n - j * panel_size, 
                                        panel_size_rest,
                                        &negone, 
                                        V + (j - 1) * panel_size * ldv,
                                        ldv, 
                                        A,
                                        panel_size_rest, 
                                        &one,
                                        V + j * panel_size * ldv,
                                        ldv));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                        CUBLAS_FILL_MODE_UPPER, 
                                                                        transa,
                                                                        panel_size_rest, 
                                                                        m, 
                                                                        &One, 
                                                                        V + j * panel_size * m, 
                                                                        ldv, 
                                                                        &Zero, 
                                                                        A, 
                                                                        panel_size_rest));     

            std::size_t upperTriangularSize = std::size_t(panel_size_rest * (panel_size_rest + 1) / 2);
            chase::linalg::internal::cuda::extractUpperTriangular(A, panel_size_rest, workspace, panel_size_rest);
            CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(workspace, workspace, upperTriangularSize, ncclSum, comm));
            chase::linalg::internal::cuda::unpackUpperTriangular(workspace, panel_size_rest, A, panel_size_rest);
            
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                            CUBLAS_FILL_MODE_UPPER, 
                                                                            panel_size_rest,
                                                                            A,
                                                                            panel_size_rest, 
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
                                                                        m, 
                                                                        panel_size_rest,
                                                                        &one, 
                                                                        A, 
                                                                        panel_size_rest, 
                                                                        V + j * panel_size * ldv, 
                                                                        ldv));                
            }

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_C, 
                                        CUBLAS_OP_N, 
                                        j * panel_size,
                                        panel_size_rest, 
                                        m,
                                        &one, 
                                        V,
                                        ldv, 
                                        V + j * panel_size * ldv,
                                        ldv, 
                                        &zero,
                                        A,
                                        j * panel_size));

            CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(A, A, (j * panel_size) * panel_size_rest, ncclSum, comm));


            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(cublas_handle, 
                                        CUBLAS_OP_N, 
                                        CUBLAS_OP_N, 
                                        m,
                                        panel_size_rest, 
                                        j * panel_size,
                                        &negone, 
                                        V,
                                        ldv, 
                                        A,
                                        j * panel_size, 
                                        &one,
                                        V + j * panel_size * ldv,
                                        ldv));

            info = cholQR1(cublas_handle,
                           cusolver_handle,
                           m, 
                           panel_size_rest, 
                           V + j * panel_size * ldv, 
                           ldv, 
                           comm,
                           workspace,
                           lwork = 0,                
                           A);                                                   
        }

#ifdef CHASE_OUTPUT
        int grank;
        MPI_Comm_rank(MPI_COMM_WORLD, &grank);
        if(grank == 0)
        {
            std::cout << "Use Modified Gram-Schmidt QR"<< std::endl;
        }
#endif  
        return info;

    }

    template<typename InputMultiVectorType>
    void houseHoulderQR(InputMultiVectorType& V)
    {
        using T = typename InputMultiVectorType::value_type;

#ifdef HAS_SCALAPACK
        V.allocate_cpu_data();//if not allocated
        V.D2H();
        std::size_t *desc = V.scalapack_descriptor_init();
        int one = 1;
        std::vector<T> tau(V.l_cols());

        chase::linalg::scalapackpp::t_pgeqrf(V.g_rows(), 
                                             V.g_cols(), 
                                             V.cpu_data(), 
                                             one, 
                                             one, 
                                             desc,
                                             tau.data());

        chase::linalg::scalapackpp::t_pgqr(V.g_rows(), 
                                           V.g_cols(), 
                                           V.g_cols(), 
                                           V.cpu_data(), 
                                           one, 
                                           one, 
                                           desc, 
                                           tau.data());
        V.H2D();

#else
        std::runtime_error("For ChASE-MPI, distributed Householder QR requires ScaLAPACK, which is not detected\n");
#endif
    }


}
}
}
}