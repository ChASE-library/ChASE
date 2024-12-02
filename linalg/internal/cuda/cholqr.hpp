#pragma once

#include <limits>
#include <iomanip>
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include "linalg/internal/cuda/absTrace.hpp"
#include "linalg/internal/cuda/shiftDiagonal.hpp"
#include "Impl/chase_gpu/nvtx.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{
    /**
    * @brief Performs a Cholesky-based QR factorization with a degree of 1.
    * 
    * This function computes the Cholesky decomposition of the matrix \( A \), 
    * and uses it for a QR factorization of a given matrix \( V \) on a GPU. 
    * The process involves matrix operations using cuBLAS and cuSolver.
    * 
    * @tparam T The data type of the matrix elements (e.g., float, double, or complex types).
    * @param cublas_handle The cuBLAS handle for managing operations.
    * @param cusolver_handle The cuSolver handle for solving linear systems and decompositions.
    * @param V The input/output matrix \( V \) to be factored (on GPU).
    * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
    * @param lwork The size of the workspace (default: 0).
    * @param A Optional output matrix for the Cholesky factor (default: nullptr). 
    *           If not provided, a new matrix is allocated.
    * @return int An error code (0 indicates success, non-zero indicates failure).
    */    
    template<typename T>
    int cholQR1(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                chase::matrix::Matrix<T, chase::platform::GPU>& V,
                T *workspace = nullptr,
                int lwork = 0,
                chase::matrix::Matrix<T, chase::platform::GPU> *A = nullptr)
    {
        SCOPED_NVTX_RANGE();

        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 1;
        
        if(A == nullptr)
        {
            A = new chase::matrix::Matrix<T, chase::platform::GPU>(V.cols(), V.cols());
        }

        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                V.cols(), 
                                                                A->data(), 
                                                                A->ld(),
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
                                                                  V.data(), 
                                                                  V.ld(), 
                                                                  &Zero, 
                                                                  A->data(), 
                                                                  A->ld()));
        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         V.cols(),
                                                                         A->data(),
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
                                                                    A->data(), 
                                                                    A->ld(), 
                                                                    V.data(), 
                                                                    V.ld()));
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 1" << std::endl;
#endif                
            return info;
        }
    }

    /**
    * @brief Performs a Cholesky-based QR factorization with a degree of 2.
    * 
    * This function computes the Cholesky decomposition of the matrix \( A \), 
    * and uses it for a QR factorization of a given matrix \( V \) on a GPU.
    * It involves several matrix operations using cuBLAS and cuSolver to perform the QR factorization.
    * 
    * @tparam T The data type of the matrix elements (e.g., float, double, or complex types).
    * @param cublas_handle The cuBLAS handle for managing operations.
    * @param cusolver_handle The cuSolver handle for solving linear systems and decompositions.
    * @param V The input/output matrix \( V \) to be factored (on GPU).
    * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
    * @param lwork The size of the workspace (default: 0).
    * @param A Optional output matrix for the Cholesky factor (default: nullptr). 
    *           If not provided, a new matrix is allocated.
    * @return int An error code (0 indicates success, non-zero indicates failure).
    */
    template<typename T>
    int cholQR2(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                chase::matrix::Matrix<T, chase::platform::GPU>& V,
                T *workspace = nullptr,
                int lwork = 0,
                chase::matrix::Matrix<T, chase::platform::GPU> *A = nullptr)
    {
        SCOPED_NVTX_RANGE();

        T one = T(1.0);
        T zero = T(0.0);
        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);

        int info = 1;
        
        if(A == nullptr)
        {
            A = new chase::matrix::Matrix<T, chase::platform::GPU>(V.cols(), V.cols());
        }

        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                V.cols(), 
                                                                A->data(), 
                                                                A->ld(),
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
                                                                  V.data(), 
                                                                  V.ld(), 
                                                                  &Zero, 
                                                                  A->data(), 
                                                                  A->ld()));
        int* devInfo;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                         CUBLAS_FILL_MODE_UPPER, 
                                                                         V.cols(),
                                                                         A->data(),
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
                                                                    A->data(), 
                                                                    A->ld(), 
                                                                    V.data(), 
                                                                    V.ld()));

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                    CUBLAS_FILL_MODE_UPPER, 
                                                                    transa,
                                                                    V.cols(), 
                                                                    V.rows(), 
                                                                    &One, 
                                                                    V.data(), 
                                                                    V.ld(), 
                                                                    &Zero, 
                                                                    A->data(), 
                                                                    A->ld()));  

            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                            CUBLAS_FILL_MODE_UPPER, 
                                                                            V.cols(),
                                                                            A->data(),
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
                                                                    A->data(), 
                                                                    A->ld(), 
                                                                    V.data(), 
                                                                    V.ld()));                                                                         
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 2" << std::endl;
#endif                
            return info;
        }
    }
    /**
    * @brief Performs a shifted Cholesky-based QR factorization with a degree of 2.
    * 
    * This function computes the shifted Cholesky decomposition for the matrix \( A \), 
    * and uses it to perform a QR factorization of the given matrix \( V \) on a GPU.
    * The function performs additional operations with shifted matrix values to enhance stability.
    * 
    * @tparam T The data type of the matrix elements (e.g., float, double, or complex types).
    * @param cublas_handle The cuBLAS handle for managing operations.
    * @param cusolver_handle The cuSolver handle for solving linear systems and decompositions.
    * @param V The input/output matrix \( V \) to be factored (on GPU).
    * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
    * @param lwork The size of the workspace (default: 0).
    * @param A Optional output matrix for the Cholesky factor (default: nullptr). 
    *           If not provided, a new matrix is allocated.
    * @return int An error code (0 indicates success, non-zero indicates failure).
    */
    template<typename T>
    int shiftedcholQR2(cublasHandle_t cublas_handle,
                cusolverDnHandle_t cusolver_handle,
                chase::matrix::Matrix<T, chase::platform::GPU>& V,
                T *workspace = nullptr,
                int lwork = 0,
                chase::matrix::Matrix<T, chase::platform::GPU> *A = nullptr)
    {
        SCOPED_NVTX_RANGE();

        T one = T(1.0);
        T zero = T(0.0);

        chase::Base<T> One = Base<T>(1.0);
        chase::Base<T> Zero = Base<T>(0.0);
        chase::Base<T> shift;

        if(A == nullptr)
        {
            A = new chase::matrix::Matrix<T, chase::platform::GPU>(V.cols(), V.cols());
        }

        if(workspace == nullptr)
        {
            lwork = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf_bufferSize(
                                                                cusolver_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                V.cols(), 
                                                                A->data(), 
                                                                A->ld(),
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
                                                                  V.data(), 
                                                                  V.ld(), 
                                                                  &Zero, 
                                                                  A->data(), 
                                                                  A->ld()));

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
                                                                         A->data(),
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
                                                                A->data(), 
                                                                A->ld(), 
                                                                V.data(), 
                                                                V.ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                transa,
                                                                V.cols(), 
                                                                V.rows(), 
                                                                &One, 
                                                                V.data(), 
                                                                V.ld(), 
                                                                &Zero, 
                                                                A->data(), 
                                                                A->ld()));  

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                        CUBLAS_FILL_MODE_UPPER, 
                                                                        V.cols(),
                                                                        A->data(),
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
                                                                A->data(), 
                                                                A->ld(), 
                                                                V.data(), 
                                                                V.ld()));

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTsyherk(cublas_handle, 
                                                                CUBLAS_FILL_MODE_UPPER, 
                                                                transa,
                                                                V.cols(), 
                                                                V.rows(), 
                                                                &One, 
                                                                V.data(), 
                                                                V.ld(), 
                                                                &Zero, 
                                                                A->data(), 
                                                                A->ld()));  

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTpotrf(cusolver_handle, 
                                                                        CUBLAS_FILL_MODE_UPPER, 
                                                                        V.cols(),
                                                                        A->data(),
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
                                                                A->data(), 
                                                                A->ld(), 
                                                                V.data(), 
                                                                V.ld()));    

#ifdef CHASE_OUTPUT
        std::cout << std::setprecision(2) << "choldegree: 2, shift = " << shift << std::endl;
#endif 
        return info;
    }

    /**
    * @brief Performs the Householder QR decomposition on a matrix \( V \) using cuSolver and cuBLAS.
    * 
    * This function computes the QR decomposition of a matrix \( V \) using the Householder transformation. 
    * It performs the required operations using cuSolver and cuBLAS on the GPU.
    * 
    * @tparam T The data type of the matrix elements (e.g., float, double, or complex types).
    * @param cusolver_handle The cuSolver handle for solving linear systems and decompositions.
    * @param V The input matrix \( V \) to be factored (on GPU).
    * @param d_tau Pointer to the array storing Householder reflectors (on GPU).
    * @param devInfo Pointer to an integer storing the result of the computation (on GPU).
    * @param workspace Pointer to a workspace for cuSolver (default: nullptr).
    * @param lwork The size of the workspace (default: 0).
    */
    template<typename T>
    void houseHoulderQR(cusolverDnHandle_t cusolver_handle,
                        chase::matrix::Matrix<T, chase::platform::GPU>& V,
                        T *d_tau,
                        int* devInfo,
                        T *workspace = nullptr,
                        int lwork = 0)
    {
        SCOPED_NVTX_RANGE();

        if(workspace == nullptr)
        {
            int lwork_geqrf = 0;
            int lwork_orgqr = 0;
            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf_bufferSize(
                                    cusolver_handle, 
                                    V.rows(),
                                    V.cols(),
                                    V.data(),
                                    V.ld(),  
                                    &lwork_geqrf));

            CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr_bufferSize(
                                    cusolver_handle, 
                                    V.rows(),
                                    V.cols(),
                                    V.cols(),
                                    V.data(),
                                    V.ld(),
                                    d_tau,  
                                    &lwork_orgqr));

            lwork = (lwork_geqrf > lwork_orgqr) ? lwork_geqrf : lwork_orgqr;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&workspace, sizeof(T) * lwork));
        }

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgeqrf(cusolver_handle, 
                                                                         V.rows(), 
                                                                         V.cols(), 
                                                                         V.data(), 
                                                                         V.ld(), 
                                                                         d_tau,
                                                                         workspace, 
                                                                         lwork, 
                                                                         devInfo));

        CHECK_CUSOLVER_ERROR(chase::linalg::cusolverpp::cusolverDnTgqr(cusolver_handle, 
                                                                       V.rows(), 
                                                                       V.cols(), 
                                                                       V.cols(), 
                                                                       V.data(), 
                                                                       V.ld(),
                                                                       d_tau, 
                                                                       workspace, 
                                                                       lwork, 
                                                                       devInfo));
    }   
}
}
}
}