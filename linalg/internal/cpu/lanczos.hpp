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

using namespace chase::linalg;

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cpu
{
    /**
     * @brief Lanczos algorithm for eigenvalue computation.
     *
     * This function performs the Lanczos algorithm, which is used to estimate
     * the upper bound of spectra of symmetric/Hermitian matrix.
     * The algorithm is iteratively applied to the matrix H, where the input
     * matrix `H` is a square matrix of size `N x N`. The Lanczos algorithm
     * builds an orthonormal basis of the Krylov subspace, and the resulting 
     * tridiagonal matrix is diagonalized using the `t_stemr` function.
     *
     * @tparam T The data type for the matrix elements (e.g., float, double).
     * @param M The number of Lanczos iterations.
     * @param numvec The number of runs of Lanczos.
     * @param N The size of the input matrix `H`.
     * @param H The input matrix for the Lanczos algorithm (of size `N x N`).
     * @param ldh The leading dimension of `H` (number of rows).
     * @param V The input matrix used for storing vectors (of size `N x numvec`).
     * @param ldv The leading dimension of `V` (number of rows).
     * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
     * @param ritzv A pointer to store the Ritz eigenvalues.
     * @param Tau A pointer to store the computed Tau values.
     * @param ritzV A pointer to store the Ritz eigenvectors.
     */    
    template<typename T>
    void lanczos(std::size_t M, std::size_t numvec, std::size_t N, T *H, std::size_t ldh, T *V, std::size_t ldv, 
                Base<T>* upperb, Base<T>* ritzv, Base<T>* Tau, Base<T>* ritzV)
    {
        T One = T(1.0);
        T Zero = T(0.0);
        std::vector<Base<T>> r_beta(numvec);
        
        std::vector<Base<T>> d(M * numvec);
        std::vector<Base<T>> e(M * numvec);

        std::vector<Base<T>> real_alpha(numvec);
        std::vector<T> alpha(numvec, T(1.0));
        std::vector<T> beta(numvec, T(0.0));

        std::vector<T> v_0(N * numvec, T(0.0));
        std::vector<T> v_1(N * numvec, T(0.0));
        std::vector<T> v_2(N * numvec, T(0.0));  

        lapackpp::t_lacpy('A', N, numvec, V, ldv, v_1.data(), N);

        for(auto i = 0; i < numvec; i++)
        {
            real_alpha[i] = blaspp::t_nrm2(N, v_1.data() + i * N, 1);
        }

        for(auto i = 0; i < numvec; i++)
        {
            alpha[i] = T(1 / real_alpha[i]);
        }

        for(auto i = 0; i < numvec; i++)
        {
            blaspp::t_scal(N, &alpha[i], v_1.data() + i * N, 1);
        }

        for (std::size_t k = 0; k < M; k = k + 1)
        {
            for(auto i = 0; i < numvec; i++){
                std::memcpy(V + k * ldv, v_1.data() + i * N, N * sizeof(T));
            }

            blaspp::t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, N,
                  numvec, N, &One, H, ldh,
                  v_1.data(), N, &Zero, v_2.data(), N);

            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = blaspp::t_dot(N, v_1.data() + i * N, 1, v_2.data() + i * N, 1);
            }

            for(auto i = 0; i < numvec; i++)
            {
                alpha[i] = -alpha[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                blaspp::t_axpy(N, &alpha[i], v_1.data() + i * N, 1, v_2.data() + i * N, 1);
            }
            
            for(auto i = 0; i < numvec; i++)
            {
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
                    blaspp::t_axpy(N, &beta[i], v_0.data() + i * N, 1, v_2.data() + i * N, 1);
                }                                
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = -beta[i];
            }

            for(auto i = 0; i < numvec; i++)
            {
                r_beta[i] = blaspp::t_nrm2(N, v_2.data() + i * N, 1);
            }

            for(auto i = 0; i < numvec; i++)
            {
                beta[i] = T(1 / r_beta[i]);
            }

            if (k == M - 1)
                break;

            for(auto i = 0; i < numvec; i++)
            {
                blaspp::t_scal(N, &beta[i], v_2.data() + i * N, 1);
            }

            for(auto i = 0; i < numvec; i++)
            {
                e[k + M * i] = r_beta[i];
            }

            v_1.swap(v_0);
            v_1.swap(v_2);                        
        }        

        lapackpp::t_lacpy('A', N, numvec, v_1.data(), N, V, ldv);

        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
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
     * @tparam T The data type for the matrix elements (e.g., float, double).
     * @param M The number of Lanczos iterations.
     * @param N The size of the input matrix `H`.
     * @param H The input matrix for the Lanczos algorithm (of size `N x N`).
     * @param ldh The leading dimension of `H` (number of rows).
     * @param V The input matrix used for storing vectors (of size `N x 1`).
     * @param ldv The leading dimension of `V` (number of rows).
     * @param upperb A pointer to the upper bound of the eigenvalue spectrum.
     */
    template<typename T>
    void lanczos(std::size_t M, std::size_t N, T *H, std::size_t ldh, T *V, std::size_t ldv, 
                Base<T>* upperb)
    {
        T One = T(1.0);
        T Zero = T(0.0);
        Base<T> r_beta;
        
        std::vector<Base<T>> d(M);
        std::vector<Base<T>> e(M);

        Base<T> real_alpha;
        T alpha = T(1.0);
        T beta = T(0.0);

        std::vector<T> v_0(N, T(0.0));
        std::vector<T> v_1(N, T(0.0));
        std::vector<T> v_2(N, T(0.0));  

        lapackpp::t_lacpy('A', N, 1, V, ldv, v_1.data(), N);

        real_alpha = blaspp::t_nrm2(N, v_1.data(), 1);
        alpha = T(1 / real_alpha);

        blaspp::t_scal(N, &alpha, v_1.data(), 1);
        
        for (std::size_t k = 0; k < M; k = k + 1)
        {
            blaspp::t_gemm<T>(CblasColMajor, CblasConjTrans, CblasNoTrans, N,
                  1, N, &One, H, ldh,
                  v_1.data(), N, &Zero, v_2.data(), N);

            alpha = blaspp::t_dot(N, v_1.data(), 1, v_2.data(), 1);
            alpha = -alpha;
            blaspp::t_axpy(N, &alpha, v_1.data(), 1, v_2.data(), 1);
            alpha = -alpha;

            d[k] = std::real(alpha);
            
            if(k > 0){
                beta = T(-r_beta);
                blaspp::t_axpy(N, &beta, v_0.data(), 1, v_2.data(), 1);                                
            }
            beta = -beta;

            r_beta = blaspp::t_nrm2(N, v_2.data(), 1);

            beta = T(1 / r_beta);
            
            if (k == M - 1)
                break;

            blaspp::t_scal(N, &beta, v_2.data(), 1);

            e[k] = r_beta;

            v_1.swap(v_0);
            v_1.swap(v_2);                        
        }        

        int notneeded_m;
        std::size_t vl, vu;
        Base<T> ul, ll;
        int tryrac = 0;
        std::vector<int> isuppz(2 * M);
        std::vector<Base<T>> ritzv(M);

        lapackpp::t_stemr<Base<T>>(LAPACK_COL_MAJOR, 'N', 'A', M, d.data(), e.data(), ul, ll, vl, vu,
                         &notneeded_m, ritzv.data(), NULL, M, M, isuppz.data(), &tryrac);

        *upperb = std::max(std::abs(ritzv[0]), std::abs(ritzv[M - 1])) +
                  std::abs(r_beta);

    }

}
}
}
}