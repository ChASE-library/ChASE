// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstring>
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/internal/nccl/nccl_kernels.hpp"
#include "../typeTraits.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{
    /**
    * @brief Executes the Lanczos algorithm to generate a tridiagonal matrix representation.
    * 
     * This function performs the Lanczos algorithm, which is used to estimate
     * the upper bound of spectra of symmetric/Hermitian matrix.
     * The algorithm is iteratively applied to the matrix H, where the input
     * matrix `H` is a square matrix of size `N x N`. The Lanczos algorithm
     * builds an orthonormal basis of the Krylov subspace, and the resulting 
     * tridiagonal matrix is diagonalized using the `t_stemr` function.
    * 
    * @tparam MatrixType Type of the input matrix, defining the value type and matrix operations.
    * @tparam InputMultiVectorType Type of the multi-vector for initial Lanczos basis vectors.
    * 
    * @param M Number of Lanczos iterations.
    * @param numvec The number of runs of Lanczos.
    * @param H The input matrix representing the system for which eigenvalues are sought.
    * @param V Initial Lanczos vectors; will be overwritten with orthonormalized basis vectors.
    * @param upperb Pointer to a variable that stores the computed upper bound for the largest eigenvalue.
    * @param ritzv Array storing the resulting Ritz values (eigenvalues).
    * @param Tau Array of values representing convergence estimates.
    * @param ritzV Vector storing the Ritz vectors associated with computed eigenvalues.
    * 
    * @throws std::runtime_error if the matrix `H` is not square or if `H` and `V` are not in the same MPI grid.
    */    
    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_nccl::pseudo_hermitian_lanczos(cublasHandle_t cublas_handle,
		 std::size_t M, 
                 std::size_t numvec,
                 MatrixType& H,
                 InputMultiVectorType& V,
                 chase::Base<typename MatrixType::value_type> *upperb,
                 chase::Base<typename MatrixType::value_type> *ritzv,
                 chase::Base<typename MatrixType::value_type> *Tau,
                 chase::Base<typename MatrixType::value_type> *ritzV)            
    {
        using T = typename MatrixType::value_type;
        using ResultMultiVectorType = typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

        if(H.g_cols() != H.g_rows())
        {
            std::runtime_error("Lanczos requires matrix to be squared");
        }

        if(H.getMpiGrid() != V.getMpiGrid())
        {
            std::runtime_error("Lanczos requires H and V in same MPI grid");
        }
        if(H.g_rows() != V.g_rows())
        {
            std::runtime_error("Lanczos H and V have same number of rows");
        }

        std::vector<chase::Base<T>> real_beta(numvec);
        std::vector<chase::Base<T>> real_alpha(numvec);
        
        std::vector<chase::Base<T>> d(M * numvec);
        std::vector<chase::Base<T>> e(M * numvec);

        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));
        
        T One = T(1.0);
        T Zero = T(0.0);

        std::size_t N = H.g_rows();

        auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
        auto v_1 = v_0.template clone<InputMultiVectorType>();
        auto v_2 = v_0.template clone<InputMultiVectorType>();
        auto Sv  = v_0.template clone<InputMultiVectorType>();
        auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

        chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec, V.l_data(), V.l_ld(), v_1.l_data(), v_1.l_ld());
	
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(cublas_handle,
								     &One, 
                                                                     H,
                                                                     v_1,
                                                                     &Zero,
                                                                     v_w);

	CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	v_w.redistributeImplAsync(&v_2);
	CHECK_CUDA_ERROR(cudaDeviceSynchronize());

	chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), numvec, v_2.l_data(), v_2.l_ld(), Sv.l_data(), Sv.l_ld());

	chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
	
	for(auto i = 0; i < numvec; i++)
        {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle,
                                                                       v_1.l_rows(),
                                                                       v_1.l_data() + i * v_1.l_ld(),
                                                                       1,
                                                                       Sv.l_data() + i * Sv.l_ld(),
                                                                       1,
                                                                       &beta[i]));
        }

	for(auto i = 0; i < numvec; i++)
	{
		real_beta[i] = std::real(beta[i]);
	}

        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        MPI_Allreduce(MPI_IN_PLACE, 
                    real_beta.data(), 
                    numvec, 
                    chase::mpi::getMPI_Type<chase::Base<T>>(),
                    MPI_SUM, 
                    H.getMpiGrid()->get_col_comm());

	for(auto i = 0; i < numvec; i++)
        {
	    beta[i] = One / std::sqrt(real_beta[i]);
        }
        
	for(auto i = 0; i < numvec; i++)
        {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                        v_1.l_rows(),
                                                                        &beta[i],
                                                                        v_1.l_data() + i * v_1.l_ld(),
                                                                        1));
        }
	
	for(auto i = 0; i < numvec; i++)
        {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                        v_2.l_rows(),
                                                                        &beta[i],
                                                                        v_2.l_data() + i * v_2.l_ld(),
                                                                        1));
        }

        for (std::size_t k = 0; k < M; k = k + 1)
        {
	    for(auto i = 0; i < numvec; i++){
                cudaMemcpy(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(), v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice);
            }

	    for(auto i = 0; i < numvec; i++)
            {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle,
                                                                       v_2.l_rows(),
                                                                       v_2.l_data() + i * v_2.l_ld(),
                                                                       1,
                                                                       Sv.l_data() + i * Sv.l_ld(),
                                                                       1,
                                                                       &alpha[i]));
            }

	    for(auto i = 0; i < numvec; i++)
	    {
		real_alpha[i] = std::real(alpha[i]);
	    }

	    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            MPI_Allreduce(MPI_IN_PLACE, 
                          real_alpha.data(), 
                          numvec, 
                    	  chase::mpi::getMPI_Type<chase::Base<T>>(),
                    	  MPI_SUM, 
                    	  H.getMpiGrid()->get_col_comm());
	
	    for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -real_alpha[i] * beta[i];  
	    }
            
	    for(auto i = 0; i < numvec; i++)
            {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle,
                                                                        v_1.l_rows(),
                                                                        &alpha[i],
                                                                        v_1.l_data() + i * v_1.l_ld(),
                                                                        1,
                                                                        v_2.l_data() + i * v_2.l_ld(),
                                                                        1));
            }

	    for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
	    }

            for(auto i = 0; i < numvec; i++)
            {
                d[k + M * i] = std::real(alpha[i]);
            }

	    if(k == M-1) break;
	    
	    for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -One / beta[i];
	    }
	    
	    for(auto i = 0; i < numvec; i++)
            {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle,
                                                                        v_0.l_rows(),
                                                                        &beta[i],
                                                                        v_0.l_data() + i * v_0.l_ld(),
                                                                        1,
                                                                        v_2.l_data() + i * v_2.l_ld(),
                                                                        1));
            }
	    
	    for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -beta[i];
	    }

            v_1.swap(v_0);
            v_1.swap(v_2);              
	    
	    chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(cublas_handle, 
			    						   &One, 
            	                                                           H,
                	                                                   v_1,
                        	                                           &Zero,
                                	                                   v_w);

	    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
	    v_w.redistributeImplAsync(&v_2);
	    
	    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), numvec, v_2.l_data(), v_2.l_ld(), Sv.l_data(), Sv.l_ld());

	    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
		
	    for(auto i = 0; i < numvec; i++)
            {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle,
                                                                       v_1.l_rows(),
                                                                       v_1.l_data() + i * v_1.l_ld(),
                                                                       1,
                                                                       Sv.l_data() + i * Sv.l_ld(),
                                                                       1,
                                                                       &beta[i]));
            }

	    for(auto i = 0; i < numvec; i++)
	    {
		real_beta[i] = std::real(beta[i]);
	    }

	    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            MPI_Allreduce(MPI_IN_PLACE,
  			  real_beta.data(), 
                    	  numvec, 
                    	  chase::mpi::getMPI_Type<chase::Base<T>>(),
                    	  MPI_SUM, 
                    	  H.getMpiGrid()->get_col_comm());

	    for(auto i = 0; i < numvec; i++)
            {
	        real_beta[i] = std::sqrt(real_beta[i]);
            }
	    
	    for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = real_beta[i];
            }
	
	    for(auto i = 0; i < numvec; i++)
            {
	        beta[i] = One / real_beta[i];
            }

	    for(auto i = 0; i < numvec; i++)
            {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                        v_1.l_rows(),
                                                                        &beta[i],
                                                                        v_1.l_data() + i * v_1.l_ld(),
                                                                        1));
            }
	
	    for(auto i = 0; i < numvec; i++)
            {
		CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                        v_2.l_rows(),
                                                                        &beta[i],
                                                                        v_2.l_data() + i * v_2.l_ld(),
                                                                        1));
            }
        }

	chase::linalg::internal::cuda::t_lacpy('A',
        		                       v_1.l_rows(),
                          		       numvec,
                                               v_1.l_data(),
                                               v_1.l_ld(),
                          		       V.l_data(),
                          		       V.l_ld());

        int notneeded_m;
        std::size_t vl = 0;
        std::size_t vu = 0;
        chase::Base<T> ul = 0;
        chase::Base<T> ll = 0;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);

        for(auto i = 0; i < numvec; i++)
        {

            lapackpp::t_stemr(LAPACK_COL_MAJOR, 'V', 'A', M, d.data() + i * M, e.data() + i * M, ul, ll, vl, vu,
                                &notneeded_m, ritzv + M * i, ritzV, M, M, isuppz.data(), &tryrac);

            for (std::size_t k = 0; k < M; ++k)
            {
                Tau[k + i * M] = std::abs(ritzV[k * M]) * std::abs(ritzV[k * M]);
            }
        }

        *upperb = ritzv[M-1];

    }

    /**
    * @brief Lanczos algorithm for eigenvalue computation (simplified version).
    * 
    * This version of the Lanczos algorithm is a simplified version that computes
    * only the upper bound of the eigenvalue spectrum and does not compute
    * eigenvectors. It operates similarly to the full Lanczos algorithm but
    * omits the eigenvector computation step.
    *
    * @tparam MatrixType Type of the input matrix, defining the value type and matrix operations.
    * @tparam InputMultiVectorType Type of the multi-vector for initial Lanczos basis vectors.
    * 
    * @param M Number of Lanczos iterations.
    * @param H The input matrix representing the system for which eigenvalues are sought.
    * @param V Initial Lanczos vector; will be overwritten with orthonormalized basis vectors.
    * @param upperb Pointer to a variable that stores the computed upper bound for the largest eigenvalue.
    * 
    * @throws std::runtime_error if the matrix `H` is not square or if `H` and `V` are not in the same MPI grid.
    */
    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_nccl::pseudo_hermitian_lanczos(cublasHandle_t cublas_handle,
		 std::size_t M, 
                 MatrixType& H,
                 InputMultiVectorType& V,
                 chase::Base<typename MatrixType::value_type>* upperb)
    {
        using T = typename MatrixType::value_type;
        using ResultMultiVectorType = typename ResultMultiVectorType<MatrixType, InputMultiVectorType>::type;

        if(H.g_cols() != H.g_rows())
        {
            std::runtime_error("Lanczos requires matrix to be squared");
        }

        if(H.getMpiGrid() != V.getMpiGrid())
        {
            std::runtime_error("Lanczos requires H and V in same MPI grid");
        }

        if(H.g_rows() != V.g_rows())
        {
            std::runtime_error("Lanczos H and V have same number of rows");
        }

        std::vector<chase::Base<T>> d(M);
        std::vector<chase::Base<T>> e(M);

        chase::Base<T> real_alpha;
        chase::Base<T> real_beta;

        T alpha = T(1.0);
        T beta  = T(0.0);
        T One   = T(1.0);
        T Zero  = T(0.0);
        
        std::size_t N = H.g_rows();

        auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
        auto v_1 = v_0.template clone<InputMultiVectorType>();
        auto v_2 = v_0.template clone<InputMultiVectorType>();
        auto Sv  = v_0.template clone<InputMultiVectorType>();
        auto v_w = V.template clone<ResultMultiVectorType>(N, 1);
        
	chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), 1, V.l_data(), V.l_ld(), v_1.l_data(), v_1.l_ld());
	
	chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(cublas_handle,
								     &One, 
                                                                     H,
                                                                     v_1,
                                                                     &Zero,
                                                                     v_w);

	v_w.redistributeImplAsync(&v_2);
        
	chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1, v_2.l_data(), v_2.l_ld(), Sv.l_data(), Sv.l_ld());
	
	chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
		
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle,
                                                                v_1.l_rows(),
                                                                v_1.l_data(),
                                                                1,
                                                                Sv.l_data(),
                                                                1,
                                                                &beta));
		
	real_beta = std::real(beta);

        MPI_Allreduce(MPI_IN_PLACE, 
                    &real_beta, 
                    1, 
                    chase::mpi::getMPI_Type<chase::Base<T>>(),
                    MPI_SUM, 
                    H.getMpiGrid()->get_col_comm());
    
	beta = One / std::sqrt(real_beta);
        	
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                v_1.l_rows(),
                                                                &beta,
                                                                v_1.l_data(),
                                                                1));
		
	CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                v_2.l_rows(),
                                                                &beta,
                                                                v_2.l_data(),
                                                                1));
        
	for (std::size_t k = 0; k < M; k = k + 1)
        {        	 
	    cudaMemcpy(V.l_data() + k * V.l_ld(), v_1.l_data(), v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice);
		
	    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle,
                                                                   v_2.l_rows(),
                                                                   v_2.l_data(),
                                                                   1,
                                                                   Sv.l_data(),
                                                                   1,
                                                                   &alpha));
	
   	    real_alpha = std::real(alpha);

            MPI_Allreduce(MPI_IN_PLACE, 
                          &real_alpha, 
                          1, 
                    	  chase::mpi::getMPI_Type<chase::Base<T>>(),
                    	  MPI_SUM, 
                    	  H.getMpiGrid()->get_col_comm());	
                
	    alpha = -real_alpha * beta;  
            
	    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle,
                                                                    v_1.l_rows(),
                                                                    &alpha,
                                                                    v_1.l_data(),
                                                                    1,
                                                                    v_2.l_data(),
                                                                    1));
                
	    alpha = -alpha;

	    d[k] = std::real(alpha);

	    if(k == M-1) break;
	        
	    beta = -One / beta;
	    	    	
	    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle,
                                                                    v_0.l_rows(),
                                                                    &beta,
                                                                    v_0.l_data(),
                                                                    1,
                                                                    v_2.l_data(),
                                                                    1));    
	    beta = -beta;

            v_1.swap(v_0);
            v_1.swap(v_2);              
	    
	    chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(cublas_handle,
			    						   &One, 
            	                                                           H,
                	                                                   v_1,
                        	                                           &Zero,
                                	                                   v_w);

	    v_w.redistributeImplAsync(&v_2);
	    
	    chase::linalg::internal::cuda::t_lacpy('A', Sv.l_rows(), 1, v_2.l_data(), v_2.l_ld(), Sv.l_data(), Sv.l_ld());

	    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(Sv);
			
	    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle,
                                                                   v_1.l_rows(),
                                                                   v_1.l_data(),
                                                                   1,
                                                                   Sv.l_data(),
                                                                   1,
                                                                   &beta));
	    real_beta = std::real(beta);
	    
            MPI_Allreduce(MPI_IN_PLACE,
  			  &real_beta, 
                    	  1, 
                    	  chase::mpi::getMPI_Type<chase::Base<T>>(),
                    	  MPI_SUM, 
                    	  H.getMpiGrid()->get_col_comm());
	    
	    real_beta = std::sqrt(real_beta);
	        
	    e[k] = real_beta;
	
	    beta = One / real_beta;
		
	    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                    v_1.l_rows(),
                                                                    &beta,
                                                                    v_1.l_data(),
                                                                    1));
		
	    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle,
                                                                    v_2.l_rows(),
                                                                    &beta,
                                                                    v_2.l_data(),
                                                                    1)); 
        }

        int notneeded_m;
        std::size_t vl = 0;
        std::size_t vu = 0;
        chase::Base<T> ul = 0;
        chase::Base<T> ll = 0;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);
        std::vector<chase::Base<T>> ritzv(M);

        lapackpp::t_stemr<chase::Base<T>>(LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
                         &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);

        *upperb = ritzv[M-1];
    }
}
}
}
