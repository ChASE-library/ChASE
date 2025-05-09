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
    * @brief Dispatch to the correct Lanczos procedure based on Matrix Type
    * 
     * This function dispatches to the correct Lanczos procedure based on the matrix type
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
    */    
    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_nccl::lanczos_dispatch(cublasHandle_t cublas_handle,
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
    
	if constexpr (std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
		      std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value)
	{
		cuda_nccl::quasi_hermitian_lanczos(cublas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
	}
	else
	{
		cuda_nccl::lanczos(cublas_handle, M, numvec, H, V, upperb, ritzv, Tau, ritzV);
	}
    }

    /**
    * @brief Dispatch to the correct Simplified Lanczos procedure based on Matrix Type
    * 
     * This function dispatches to the correct Simplified Lanczos procedure based on the matrix type
    * 
    * @tparam MatrixType Type of the input matrix, defining the value type and matrix operations.
    * @tparam InputMultiVectorType Type of the multi-vector for initial Lanczos basis vectors.
    * 
    * @param M Number of Lanczos iterations.
    * @param H The input matrix representing the system for which eigenvalues are sought.
    * @param V Initial Lanczos vectors; will be overwritten with orthonormalized basis vectors.
    * @param upperb Pointer to a variable that stores the computed upper bound for the largest eigenvalue.
    * 
    */    
    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_nccl::lanczos_dispatch(cublasHandle_t cublas_handle,
		 std::size_t M, 
                 MatrixType& H,
                 InputMultiVectorType& V,
                 chase::Base<typename MatrixType::value_type>* upperb)
    {
        using T = typename MatrixType::value_type;
	
	if constexpr (std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockBlockMatrix<T, chase::platform::GPU>>::value ||
		      std::is_same<MatrixType, chase::distMatrix::QuasiHermitianBlockCyclicMatrix<T, chase::platform::GPU>>::value)
	{
		cuda_nccl::quasi_hermitian_lanczos(cublas_handle, M, H, V, upperb);
	}
	else
	{
		cuda_nccl::lanczos(cublas_handle, M, H, V, upperb);
	}
    }

    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_nccl::lanczos(cublasHandle_t cublas_handle,
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

        std::vector<chase::Base<T>> r_beta(numvec);
        
        std::vector<chase::Base<T>> d(M * numvec);
        std::vector<chase::Base<T>> e(M * numvec);

        std::vector<chase::Base<T>> real_alpha(numvec);
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));
        
        T One = T(1.0);
        T Zero = T(0.0);
        
        std::size_t N = H.g_rows();

        auto v_0 = V.template clone<InputMultiVectorType>(N, numvec);
        auto v_1 = v_0.template clone<InputMultiVectorType>();
        auto v_2 = v_0.template clone<InputMultiVectorType>();
        auto v_w = V.template clone<ResultMultiVectorType>(N, numvec);

        chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), numvec, V.l_data(), V.l_ld(), v_1.l_data(), v_1.l_ld());
        for(auto i = 0; i < numvec; i++)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                    v_1.l_rows(), 
                                                                    v_1.l_data() + i * v_1.l_ld(),
                                                                    1, 
                                                                    &real_alpha[i]));
            real_alpha[i] = std::pow(real_alpha[i], 2);
        }
  
        MPI_Allreduce(MPI_IN_PLACE, 
                    real_alpha.data(), 
                    numvec, 
                    chase::mpi::getMPI_Type<chase::Base<T>>(),
                    MPI_SUM, 
                    H.getMpiGrid()->get_col_comm());

        for(auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = std::sqrt(real_alpha[i]);
            alpha[i] = T(1 / real_alpha[i]);
        }

        for(auto i = 0; i < numvec; i++)
        {
              CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                      v_1.l_rows(), 
                                                                      &alpha[i], 
                                                                      v_1.l_data() + i * v_1.l_ld(),
                                                                      1));  
        }        

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            for(auto i = 0; i < numvec; i++){
                cudaMemcpy(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(), v_1.l_rows() * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(cublas_handle,
                                                                         &One, 
                                                                         H,
                                                                         v_1,
                                                                         &Zero,
                                                                         v_w);
            v_w.redistributeImplAsync(&v_2);  

            for(auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle, 
                                                                             v_1.l_rows(), 
                                                                             v_1.l_data() + i * v_1.l_ld(), 
                                                                             1, 
                                                                             v_2.l_data() + i * v_2.l_ld(), 
                                                                             1, 
                                                                             &alpha[i])); 

                alpha[i] = -alpha[i];
            }             

            MPI_Allreduce(MPI_IN_PLACE, 
                          alpha.data(), 
                          numvec, 
                          chase::mpi::getMPI_Type<T>(), 
                          MPI_SUM,
                          H.getMpiGrid()->get_col_comm());

            for(auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle, 
                                                                              v_1.l_rows(), 
                                                                              &alpha[i], 
                                                                              v_1.l_data() + i * v_1.l_ld(), 
                                                                              1, 
                                                                              v_2.l_data() + i * v_2.l_ld(), 
                                                                              1));
                alpha[i] = -alpha[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                d[k + M * i] = std::real(alpha[i]);
            }
            if(k > 0){
                for(auto i = 0; i < numvec; i++)
                {
                    beta[i] = T(-r_beta[i]);
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
            }            
            
            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -beta[i];
            }

            for(auto i = 0; i < numvec; i++)
            {            
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                        v_2.l_rows(), 
                                                                        v_2.l_data() + i * v_2.l_ld(),
                                                                        1, 
                                                                        &r_beta[i]));
            
                r_beta[i] = std::pow(r_beta[i], 2);
            }

            MPI_Allreduce(MPI_IN_PLACE, 
                        r_beta.data(), 
                        numvec, 
                        chase::mpi::getMPI_Type<chase::Base<T>>(),
                        MPI_SUM, 
                        H.getMpiGrid()->get_col_comm());

            for(auto i = 0; i < numvec; i++)
            {   
                r_beta[i] = std::sqrt(r_beta[i]);
            }

            for(auto i = 0; i < numvec; i++)
            {               
                beta[i] = T(1 / r_beta[i]);
            }

            if (k == M - 1)
            {
                break;
            }
                
            for(auto i = 0; i < numvec; i++)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                        v_2.l_rows(), 
                                                                        &beta[i], 
                                                                        v_2.l_data() + i * v_2.l_ld(),
                                                                        1));              
            }

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = r_beta[i];
            }

            v_1.swap(v_0);
            v_1.swap(v_2); 
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

        chase::Base<T> max;
        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
                  std::abs(r_beta[0]);

        for(auto i = 1; i < numvec; i++)
        {
          max = std::max(std::abs(ritzv[i * M]), std::abs(ritzv[ (i + 1) * M - 1])) +
                  std::abs(r_beta[i]);
          *upperb = std::max(max, *upperb);        
        }      
    }

    template <typename MatrixType, typename InputMultiVectorType>
    void cuda_nccl::lanczos(cublasHandle_t cublas_handle,
                 std::size_t M,
                 MatrixType& H,
                 InputMultiVectorType& V,
                 chase::Base<typename MatrixType::value_type> *upperb)             
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
        chase::Base<T> r_beta;

        T alpha = T(1.0);
        T beta = T(0.0);
        T One = T(1.0);
        T Zero = T(0.0);
        
        std::size_t N = H.g_rows();

        auto v_0 = V.template clone<InputMultiVectorType>(N, 1);
        auto v_1 = v_0.template clone<InputMultiVectorType>();
        auto v_2 = v_0.template clone<InputMultiVectorType>();
        auto v_w = V.template clone<ResultMultiVectorType>(N, 1);

        chase::linalg::internal::cuda::t_lacpy('A', v_1.l_rows(), 1, V.l_data(), V.l_ld(), v_1.l_data(), v_1.l_ld());

        //real_alpha = chase::linalg::blaspp::t_norm_p2(v_1.l_rows(), v_1.l_data());
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                v_1.l_rows(), 
                                                                v_1.l_data(),
                                                                1, 
                                                                &real_alpha));
        real_alpha = std::pow(real_alpha, 2);
        
        MPI_Allreduce(MPI_IN_PLACE, 
                      &real_alpha, 
                      1, 
                      chase::mpi::getMPI_Type<chase::Base<T>>(),
                      MPI_SUM, 
                      H.getMpiGrid()->get_col_comm());

        real_alpha = std::sqrt(real_alpha);
        alpha = T(1 / real_alpha);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                      v_1.l_rows(), 
                                                                      &alpha, 
                                                                      v_1.l_data(),
                                                                      1));  
        for (std::size_t k = 0; k < M; k = k + 1)
        {
            chase::linalg::internal::cuda_nccl::MatrixMultiplyMultiVectors(cublas_handle,
                                                                         &One, 
                                                                         H,
                                                                         v_1,
                                                                         &Zero,
                                                                         v_w);

            v_w.redistributeImplAsync(&v_2);   

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(cublas_handle, 
                                                                            v_1.l_rows(), 
                                                                            v_1.l_data(), 
                                                                            1, 
                                                                            v_2.l_data(), 
                                                                            1, 
                                                                            &alpha));             
            alpha = -alpha;

            MPI_Allreduce(MPI_IN_PLACE, 
                          &alpha, 
                          1, 
                          chase::mpi::getMPI_Type<T>(), 
                          MPI_SUM,
                          H.getMpiGrid()->get_col_comm());

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle, 
                                                                            v_1.l_rows(), 
                                                                            &alpha, 
                                                                            v_1.l_data(), 
                                                                            1, 
                                                                            v_2.l_data(), 
                                                                            1));            
            alpha = -alpha;

            d[k] = std::real(alpha);
            
            if(k > 0){
                beta = T(-r_beta);
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTaxpy(cublas_handle, 
                                                                            v_0.l_rows(), 
                                                                            &beta, 
                                                                            v_0.l_data(), 
                                                                            1, 
                                                                            v_2.l_data(), 
                                                                            1));               
            }

            beta = -beta;

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTnrm2(cublas_handle, 
                                                                    v_2.l_rows(), 
                                                                    v_2.l_data(),
                                                                    1, 
                                                                    &r_beta));
        
            r_beta = std::pow(r_beta, 2);
            
            MPI_Allreduce(MPI_IN_PLACE, 
                        &r_beta, 
                        1, 
                        chase::mpi::getMPI_Type<chase::Base<T>>(),
                        MPI_SUM, 
                        H.getMpiGrid()->get_col_comm());
                        

            r_beta = std::sqrt(r_beta);

            beta = T(1 / r_beta);
            
            if (k == M - 1)
                break;

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 
                                                                    v_2.l_rows(), 
                                                                    &beta, 
                                                                    v_2.l_data(),
                                                                        1));  
            e[k] = r_beta;

            v_1.swap(v_0);
            v_1.swap(v_2);  
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

        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
                  std::abs(r_beta);
    }



}
}
}
