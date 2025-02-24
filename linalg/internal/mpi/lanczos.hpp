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
//#include "linalg/internal/mpi/hemm.hpp"
#include "linalg/internal/mpi/mpi_kernels.hpp"

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
    void cpu_mpi::lanczos(std::size_t M, 
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

        chase::linalg::lapackpp::t_lacpy('A', v_1.l_rows(), numvec, V.l_data(), V.l_ld(), v_1.l_data(), v_1.l_ld());

        for(auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = chase::linalg::blaspp::t_norm_p2(v_1.l_rows(), v_1.l_data() + i * v_1.l_ld());
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
            chase::linalg::blaspp::t_scal(v_1.l_rows(), 
                                          &alpha[i], 
                                          v_1.l_data() + i * v_1.l_ld(), 
                                          1);
        }

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            for(auto i = 0; i < numvec; i++){
                std::memcpy(V.l_data() + k * V.l_ld(), v_1.l_data() + i * v_1.l_ld(), v_1.l_rows() * sizeof(T));
            }

            chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&One, 
                                                                         H,
                                                                         v_1,
                                                                         &Zero,
                                                                         v_w);

            v_w.redistributeImpl(&v_2);   

            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = chase::linalg::blaspp::t_dot(v_1.l_rows(), 
                                                        v_1.l_data() + i * v_1.l_ld(), 
                                                        1, 
                                                        v_2.l_data() + i * v_2.l_ld(), 
                                                        1);
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
                chase::linalg::blaspp::t_axpy(v_1.l_rows(), 
                                              &alpha[i], 
                                              v_1.l_data() + i * v_1.l_ld(), 
                                              1, 
                                              v_2.l_data() + i * v_2.l_ld(), 
                                              1);
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
                    chase::linalg::blaspp::t_axpy(v_0.l_rows(), 
                                                  &beta[i], 
                                                  v_0.l_data() + i * v_0.l_ld(), 
                                                  1, 
                                                  v_2.l_data() + i * v_2.l_ld(), 
                                                  1);
                }                                
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -beta[i];
            }

            for(auto i = 0; i < numvec; i++)
            {            
                r_beta[i] = chase::linalg::blaspp::t_norm_p2(v_2.l_rows(), v_2.l_data() + i * v_2.l_ld());
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
                break;

            for(auto i = 0; i < numvec; i++)
            {   
                chase::linalg::blaspp::t_scal(v_2.l_rows(), 
                                             &beta[i], 
                                             v_2.l_data() + i * v_2.l_ld(), 
                                             1);
            }

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = r_beta[i];
            }

            v_1.swap(v_0);
            v_1.swap(v_2);              
        }

        lapackpp::t_lacpy('A', 
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

        Base<T> max;
        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
                  std::abs(r_beta[0]);

        for(auto i = 1; i < numvec; i++)
        {
          max = std::max(std::abs(ritzv[i * M]), std::abs(ritzv[ (i + 1) * M - 1])) +
                  std::abs(r_beta[i]);
          *upperb = std::max(max, *upperb);        
        }       
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
    void cpu_mpi::lanczos(std::size_t M, 
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

        chase::linalg::lapackpp::t_lacpy('A', v_1.l_rows(), 1, V.l_data(), V.l_ld(), v_1.l_data(), v_1.l_ld());

        real_alpha = chase::linalg::blaspp::t_norm_p2(v_1.l_rows(), v_1.l_data());

        MPI_Allreduce(MPI_IN_PLACE, 
                      &real_alpha, 
                      1, 
                      chase::mpi::getMPI_Type<chase::Base<T>>(),
                      MPI_SUM, 
                      H.getMpiGrid()->get_col_comm());

        real_alpha = std::sqrt(real_alpha);
        alpha = T(1 / real_alpha);

        chase::linalg::blaspp::t_scal(v_1.l_rows(), &alpha, v_1.l_data(), 1);

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&One, 
                                                                         H,
                                                                         v_1,
                                                                         &Zero,
                                                                         v_w);

            v_w.redistributeImpl(&v_2);   

            alpha = chase::linalg::blaspp::t_dot(v_1.l_rows(), v_1.l_data(), 1, v_2.l_data(), 1);
            alpha = -alpha;

            MPI_Allreduce(MPI_IN_PLACE, 
                          &alpha, 
                          1, 
                          chase::mpi::getMPI_Type<T>(), 
                          MPI_SUM,
                          H.getMpiGrid()->get_col_comm());

            chase::linalg::blaspp::t_axpy(v_1.l_rows(), &alpha, v_1.l_data(), 1, v_2.l_data(), 1);
            alpha = -alpha;

            d[k] = std::real(alpha);
            
            if(k > 0){
                beta = T(-r_beta);
                chase::linalg::blaspp::t_axpy(v_0.l_rows(), &beta, v_0.l_data(), 1, v_2.l_data(), 1);                                
            }

            beta = -beta;

            r_beta = chase::linalg::blaspp::t_norm_p2(v_2.l_rows(), v_2.l_data());

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

            chase::linalg::blaspp::t_scal(v_2.l_rows(), &beta, v_2.l_data(), 1);

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