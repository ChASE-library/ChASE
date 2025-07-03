// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

/**
 * \defgroup nccl_kernels chase::linalg::internal::nccl Namespace
 * \brief The `chase::linalg::internal::nccl` namespace contains 
 * kernels required by ChASE for the distributed-memory GPU using NCCL
 * for communications.
 * @{
 */
#include "external/cublaspp/cublaspp.hpp"
#include "external/cusolverpp/cusolverpp.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "../typeTraits.hpp"
/** @} */

namespace chase
{
namespace linalg
{
namespace internal
{
    struct cuda_nccl
    {
        template <typename T, typename MatrixType, typename InputMultiVectorType>
        static void MatrixMultiplyMultiVectors(cublasHandle_t cublas_handle, T* alpha,
                                            MatrixType& blockMatrix,
                                            InputMultiVectorType& input_multiVector,
                                            T* beta,
                                            typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                            std::size_t offset,
                                            std::size_t subSize);

        template <typename T, typename MatrixType, typename InputMultiVectorType>
        static void MatrixMultiplyMultiVectors(cublasHandle_t cublas_handle, T* alpha,
                                                MatrixType& blockMatrix,
                                                InputMultiVectorType& input_multiVector,
                                                T* beta,
                                                typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector);

        //this operation do: W1<-1.0 * H * V1, while redistribute V2 to W2
        template <typename MatrixType, typename InputMultiVectorType>    
        static void MatrixMultiplyMultiVectorsAndRedistributeAsync(cublasHandle_t cublas_handle,
                                            MatrixType& blockMatrix, 
                                            InputMultiVectorType& input_multiVector, 
                                            typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                            InputMultiVectorType& src_multiVector,   
                                            typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& target_multiVector,                                                                               
                                            std::size_t offset,
                                            std::size_t subSize);                                                                               

        template <typename MatrixType, typename InputMultiVectorType>    
        static void MatrixMultiplyMultiVectorsAndRedistributeAsync(cublasHandle_t cublas_handle,
                                            MatrixType& blockMatrix, 
                                            InputMultiVectorType& input_multiVector, 
                                            typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& result_multiVector,
                                            InputMultiVectorType& src_multiVector,   
                                            typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& target_multiVector);

        template <typename MatrixType, typename InputMultiVectorType>
        static void lanczos_dispatch(cublasHandle_t cublas_handle,
                    std::size_t M, 
                    std::size_t numvec,
                    MatrixType& H,
                    InputMultiVectorType& V,
                    chase::Base<typename MatrixType::value_type> *upperb,
                    chase::Base<typename MatrixType::value_type> *ritzv,
                    chase::Base<typename MatrixType::value_type> *Tau,
                    chase::Base<typename MatrixType::value_type> *ritzV);     

        template <typename MatrixType, typename InputMultiVectorType>
        static void lanczos_dispatch(cublasHandle_t cublas_handle,
                    std::size_t M,
                    MatrixType& H,
                    InputMultiVectorType& V,
                    chase::Base<typename MatrixType::value_type> *upperb); 

        template <typename MatrixType, typename InputMultiVectorType>
        static void lanczos(cublasHandle_t cublas_handle,
                    std::size_t M, 
                    std::size_t numvec,
                    MatrixType& H,
                    InputMultiVectorType& V,
                    chase::Base<typename MatrixType::value_type> *upperb,
                    chase::Base<typename MatrixType::value_type> *ritzv,
                    chase::Base<typename MatrixType::value_type> *Tau,
                    chase::Base<typename MatrixType::value_type> *ritzV);     

        template <typename MatrixType, typename InputMultiVectorType>
        static void lanczos(cublasHandle_t cublas_handle,
                    std::size_t M,
                    MatrixType& H,
                    InputMultiVectorType& V,
                    chase::Base<typename MatrixType::value_type> *upperb);  
        
	template <typename MatrixType, typename InputMultiVectorType>
        static void quasi_hermitian_lanczos(cublasHandle_t cublas_handle,
                    std::size_t M, 
                    std::size_t numvec,
                    MatrixType& H,
                    InputMultiVectorType& V,
                    chase::Base<typename MatrixType::value_type> *upperb,
                    chase::Base<typename MatrixType::value_type> *ritzv,
                    chase::Base<typename MatrixType::value_type> *Tau,
                    chase::Base<typename MatrixType::value_type> *ritzV);     

        template <typename MatrixType, typename InputMultiVectorType>
        static void quasi_hermitian_lanczos(cublasHandle_t cublas_handle,
                    std::size_t M,
                    MatrixType& H,
                    InputMultiVectorType& V,
                    chase::Base<typename MatrixType::value_type> *upperb);  


        template<typename T>
        static int cholQR1(cublasHandle_t cublas_handle,
                    cusolverDnHandle_t cusolver_handle,
                    std::size_t m, 
                    std::size_t n, 
                    T *V, 
                    int ldv, 
                    ncclComm_t comm,
                    T *workspace = nullptr,
                    int lwork = 0,                
                    T *A = nullptr);

        template<typename InputMultiVectorType>
        static int cholQR1(cublasHandle_t cublas_handle,
                    cusolverDnHandle_t cusolver_handle,
                    InputMultiVectorType& V, 
                    typename InputMultiVectorType::value_type *workspace = nullptr,
                    int lwork = 0,                
                    typename InputMultiVectorType::value_type *A = nullptr);
                    
        template<typename T>
        static int cholQR2(cublasHandle_t cublas_handle,
                    cusolverDnHandle_t cusolver_handle,
                    std::size_t m, 
                    std::size_t n, 
                    T *V, 
                    int ldv, 
                    ncclComm_t comm,
                    T *workspace = nullptr,
                    int lwork = 0,                
                    T *A = nullptr);

        template<typename InputMultiVectorType>
        static int cholQR2(cublasHandle_t cublas_handle,
                    cusolverDnHandle_t cusolver_handle,
                    InputMultiVectorType& V, 
                    typename InputMultiVectorType::value_type *workspace = nullptr,
                    int lwork = 0,                
                    typename InputMultiVectorType::value_type *A = nullptr);
                    
        template<typename T>
        static int shiftedcholQR2(cublasHandle_t cublas_handle,
                    cusolverDnHandle_t cusolver_handle,
                    std::size_t N, 
                    std::size_t m, 
                    std::size_t n, 
                    T *V, 
                    int ldv, 
                    ncclComm_t comm,
                    T *workspace = nullptr,
                    int lwork = 0,                
                    T *A = nullptr);

        template<typename T>
        static int modifiedGramSchmidtCholQR(cublasHandle_t cublas_handle,
                    cusolverDnHandle_t cusolver_handle,
                    std::size_t m, 
                    std::size_t n, 
                    std::size_t locked,
                    T *V, 
                    std::size_t ldv, 
                    ncclComm_t comm,
                    T *workspace = nullptr,
                    int lwork = 0,                
                    T *A = nullptr);

        template<typename InputMultiVectorType>
        static void houseHoulderQR(InputMultiVectorType& V);
        
        template<typename InputMultiVectorType>
        static chase::Base<typename InputMultiVectorType::value_type> computeConditionNumber(InputMultiVectorType& V);
        
        template <typename MatrixType, typename InputMultiVectorType>
        static void rayleighRitz(cublasHandle_t cublas_handle, 
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
                    typename MatrixType::value_type *workspace = nullptr,
                    int lwork_heevd = 0,
                    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type, chase::platform::GPU>* A = nullptr);                
        
	template <typename MatrixType, typename InputMultiVectorType>
        static void quasi_hermitian_rayleighRitz(cublasHandle_t cublas_handle, 
                    cusolverDnHandle_t cusolver_handle,
		    cusolverDnParams_t params,
                    MatrixType& H,
                    InputMultiVectorType& V1,
                    InputMultiVectorType& V2,
                    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,  
                    chase::distMatrix::RedundantMatrix<chase::Base<typename MatrixType::value_type>, chase::platform::GPU>& ritzv, 
                    std::size_t offset,
                    std::size_t subSize,
                    int* devInfo,
                    typename MatrixType::value_type *d_workspace = nullptr,
                    int d_lwork = 0,
                    typename MatrixType::value_type *h_workspace = nullptr,
                    int h_lwork = 0,
                    chase::distMatrix::RedundantMatrix<typename MatrixType::value_type, chase::platform::GPU>* A = nullptr);                

        template <typename MatrixType, typename InputMultiVectorType>
        static void residuals(cublasHandle_t cublas_handle,
                    MatrixType& H,
                    InputMultiVectorType& V1,
                    InputMultiVectorType& V2,
                    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W1,
                    typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type& W2,
                    chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>, typename MatrixType::platform_type>& ritzv,
                    chase::matrix::Matrix<chase::Base<typename MatrixType::value_type>, typename MatrixType::platform_type>& resids,
                    std::size_t offset,
                    std::size_t subSize);

        template<typename MatrixType>
        static bool checkSymmetryEasy(cublasHandle_t cublas_handle, MatrixType& H); 

        template<typename T>
        static void symOrHermMatrix(char uplo, chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H); 

        template<typename T>
        static void symOrHermMatrix(char uplo, chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H); 

        template<typename MatrixType>
        static void shiftDiagonal(MatrixType& H, 
                        std::size_t* d_off_m, 
                        std::size_t* d_off_n, 
                        std::size_t offsize, 
                        chase::Base<typename MatrixType::value_type> shift);
        
	template<typename T>
        static void flipLowerHalfMatrixSign(chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>& H);
	
	template<typename T>
        static void flipLowerHalfMatrixSign(chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>& H);

	template<typename InputMultiVectorType>
        static void flipLowerHalfMatrixSign(InputMultiVectorType& V);
	
	template<typename InputMultiVectorType>
        static void flipLowerHalfMatrixSign(InputMultiVectorType& V, 
			std::size_t offset, 
			std::size_t subSize);
                                                                    
    };

}
}
}

#include "linalg/internal/nccl/symOrHerm.hpp"
#include "linalg/internal/nccl/hemm.hpp"
#include "linalg/internal/nccl/lanczos.hpp"
#include "linalg/internal/nccl/quasi_hermitian_lanczos.hpp"
#include "linalg/internal/nccl/flipSign.hpp"
#include "linalg/internal/nccl/cholqr.hpp"
#include "linalg/internal/nccl/rayleighRitz.hpp"
#include "linalg/internal/nccl/quasi_hermitian_rayleighRitz.hpp"
#include "linalg/internal/nccl/residuals.hpp"
#include "linalg/internal/nccl/shiftDiagonal.hpp"
