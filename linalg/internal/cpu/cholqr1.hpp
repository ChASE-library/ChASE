#pragma once

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
     * \brief Performs Cholesky QR factorization (degree 1).
     * 
     * This function performs Cholesky QR factorization on the matrix V.
     * It computes \( A = V^T V \) and then solves \( A X = V \).
     * 
     * \param m The number of rows of matrix V.
     * \param n The number of columns of matrix V.
     * \param V The matrix on which the factorization is performed.
     * \param ldv The leading dimension of V.
     * \param A The output matrix that stores the result of the Cholesky factorization (optional, will be allocated if null).
     * \return 0 if successful, non-zero value otherwise.
     */    
    template<typename T>
    int cholQR1(std::size_t m, std::size_t n, T *V, int ldv, T *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
        info = lapackpp::t_potrf('U', n, A, n); 

        if(info != 0)
        {
            return info;
        }
        else
        {
            blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv); 
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 1" << std::endl;
#endif      
            return info;        
        }
    }

    /**
     * \brief Performs Cholesky QR factorization (degree 2).
     * 
     * This function performs Cholesky QR factorization on the matrix V.
     * It applies two iterations of Cholesky QR factorization.
     * 
     * \param m The number of rows of matrix V.
     * \param n The number of columns of matrix V.
     * \param V The matrix on which the factorization is performed.
     * \param ldv The leading dimension of V.
     * \param A The output matrix that stores the result of the Cholesky factorization (optional, will be allocated if null).
     * \return 0 if successful, non-zero value otherwise.
     */
    template<typename T>
    int cholQR2(std::size_t m, std::size_t n, T *V, int ldv, T *A = nullptr)
    {
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
        info = lapackpp::t_potrf('U', n, A, n); 
        if(info != 0)
        {
            return info;
        }
        else
        {
            blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv); 
            blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
            info = lapackpp::t_potrf('U', n, A, n); 
            blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv); 
#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 2" << std::endl;
#endif                    
            return info;              
        }
    }

    /**
     * \brief Performs Cholesky QR factorization with shifting (degree 2).
     * 
     * This function performs Cholesky QR factorization on the matrix V, with a shift applied to the matrix diagonal.
     * It applies two iterations of Cholesky QR factorization with a diagonal shift.
     * 
     * \param m The number of rows of matrix V.
     * \param n The number of columns of matrix V.
     * \param V The matrix on which the factorization is performed.
     * \param ldv The leading dimension of V.
     * \param A The output matrix that stores the result of the Cholesky factorization (optional, will be allocated if null).
     * \return 0 if successful, non-zero value otherwise.
     */
    template<typename T>
    int shiftedcholQR2(std::size_t m, std::size_t n, T *V, int ldv, T *A = nullptr)
    { 
        Base<T> shift;
        T one = T(1.0);
        T zero = T(0.0);
        int info = 1;
        
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
        Base<T> nrmf = 0.0;
        computeDiagonalAbsSum(n, n, A, n, &nrmf);
        shift = std::sqrt(m) * nrmf * std::numeric_limits<Base<T>>::epsilon();
        shiftMatrixDiagonal(n, n, A, n, (T)shift);
        info = lapackpp::t_potrf('U', n, A, n); 
        if(info != 0)
        {
            return info;
        }
        else
        {
            blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv); 
            blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
            info = lapackpp::t_potrf('U', n, A, n); 
            blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv); 
            blaspp::t_syherk('U', 'C', n, m, &one, V, ldv, &zero, A, n);
            info = lapackpp::t_potrf('U', n, A, n); 
            blaspp::t_trsm('R', 'U', 'N', 'N', m, n, &one, A, n, V, ldv);  

#ifdef CHASE_OUTPUT
            std::cout << "choldegree: 2, shift = " << shift << std::endl;
#endif
            return info;                        
        }

    }

    /**
     * \brief Performs Householder QR factorization.
     * 
     * This function computes the QR factorization of matrix V using the Householder transformation.
     * 
     * \param m The number of rows of matrix V.
     * \param n The number of columns of matrix V.
     * \param V The matrix on which the factorization is performed.
     * \param ldv The leading dimension of V.
     */
    template<typename T>
    void houseHoulderQR(std::size_t m, std::size_t n, T *V, std::size_t ldv)
    {
        std::unique_ptr<T[]> tau(new T[n]);

        chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, 
                                        m, 
                                        n, 
                                        V, 
                                        ldv, 
                                        tau.get());
        chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, 
                                        m, 
                                        n, 
                                        n, 
                                        V, 
                                        ldv, 
                                        tau.get());
    }   
}
} //end of namespace lapackpp
} //end of namespace linalg   
} //end of namespace chase