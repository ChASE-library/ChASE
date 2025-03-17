// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/matrix/matrix.hpp"

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
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a matrix.
    *
    * The Rayleigh-Ritz method computes an approximation to the eigenvalues and eigenvectors of a matrix
    * by projecting the matrix onto a subspace defined by a set of vectors (Q) and solving the eigenvalue
    * problem for the reduced matrix. The computed Ritz values are stored in the `ritzv` array, and the 
    * resulting eigenvectors are stored in `V`.
    *
    * @tparam T Data type for the matrix (e.g., float, double, etc.).
    * @param[in] N The number of rows of the matrix H.
    * @param[in] H The input matrix (N x N).
    * @param[in] ldh The leading dimension of the matrix H.
    * @param[in] n The number of vectors in Q (subspace size).
    * @param[in] Q The input matrix of size (N x n), whose columns are the basis vectors for the subspace.
    * @param[in] ldq The leading dimension of the matrix Q.
    * @param[out] V The output matrix (N x n), which will store the result of the projection.
    * @param[in] ldw The leading dimension of the matrix V.
    * @param[out] ritzv The array of Ritz values, which contains the eigenvalue approximations.
    * @param[in] A A temporary matrix used in intermediate calculations. If not provided, it is allocated internally.
    *
    * The procedure performs the following steps:
    * 1. Computes the matrix-vector multiplication: V = H * Q.
    * 2. Computes A = V' * Q, where V' is the conjugate transpose of V.
    * 3. Solves the eigenvalue problem for A using LAPACK's `heevd` function, computing the Ritz values in `ritzv`.
    * 4. Computes the final approximation to the eigenvectors by multiplying Q with the computed ritz vectors.
    */    
    template<typename T>
    void rayleighRitz(chase::matrix::Matrix<T> *H, std::size_t n, T *Q, std::size_t ldq, 
                    T * V, std::size_t ldv, Base<T> *ritzv, T *A = nullptr)
    {
	std::size_t N   = H->rows();	
	std::size_t ldh = H->ld();
        
	T One  = T(1.0);
        T Zero = T(0.0);
        
	std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N, n, N, &One,
               H->data(), ldh, Q, ldq, &Zero, V, ldv);

        // A <- V' * Q
        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, N,
               &One, V, ldv, Q, ldq, &Zero, A, n);

        lapackpp::t_heevd(LAPACK_COL_MAJOR, 'V', 'L', n, A, n, ritzv);

        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, n,
               &One, Q, ldq, A, n, &Zero, V, ldv);
    }
    
    /**
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a Quasi-Hermitian matrix.
    *
    * The Rayleigh-Ritz method computes an approximation to the eigenvalues and eigenvectors of a matrix
    * by projecting the matrix onto a subspace defined by a set of vectors (Q) and solving the eigenvalue
    * problem for the reduced matrix. The real parts of the computed Ritz values are stored in the `ritzv` array, and the 
    * resulting right eigenvectors are stored in `V`.
    *
    * @tparam T Data type for the matrix (e.g., float, double, etc.).
    * @param[in] H The Quasi-Hermitian input matrix (N x N).
    * @param[in] n The number of vectors in Q (subspace size).
    * @param[in] Q The input matrix of size (N x n), whose columns are the basis vectors for the subspace.
    * @param[in] ldq The leading dimension of the matrix Q.
    * @param[out] V The output matrix (N x n), which will store the result of the projection.
    * @param[in] ldv The leading dimension of the matrix V.
    * @param[out] ritzv The array of Ritz values, which contains the eigenvalue approximations.
    * @param[in] A A temporary matrix used in intermediate calculations. If not provided, it is allocated internally.
    *
    * The procedure performs the following steps:
    * 1. Computes the matrix-vector multiplication: V = H' * Q = S * H * S * Q.
    * 2. Computes A = V' * Q, where V' is the conjugate transpose of V.
    * 3. Solves the eigenvalue problem for A using LAPACK's `geev` function, computing the real part of Ritz values in `ritzv`.
    * 4. Computes the final approximation to the eigenvectors by multiplying Q with the computed ritz vectors.
    */    
    template<typename T>
    void rayleighRitz(chase::matrix::QuasiHermitianMatrix<T> *H, std::size_t n, T *Q, std::size_t ldq, 
                      T * V, std::size_t ldv, Base<T> *ritzv, T *A = nullptr)
    {

	std::size_t N   = H->rows();	
	std::size_t ldh = H->ld();
	std::size_t k   = N / 2;
	std::size_t lda = n+k;

	T alpha = T(2.0);
        T One   = T(1.0);
        T Zero  = T(0.0);

        std::unique_ptr<T[]> ptrA;
	//Allocate space for the rayleigh quotient
        if (A == nullptr)
        {
            ptrA = std::unique_ptr<T[]>{new T[lda * n]};
            A = ptrA.get();
        }

        //Allocate the space for halfQ transformations. Half size of an explicit dual basis
	T *halfQ = A + n; 

	//Alocate space for the ritz vectors, but we can probably reuse halfQ workspace. Double free?
	std::unique_ptr<T[]> ptrW = std::unique_ptr<T[]>{new T[n * n]};
        T* W = ptrW.get();
	
	//Allocate the space for scaling weights. Can be Base<T> since reals?
	std::vector<T> diag(n, T(0.0)); 

	//Allocate the space for the imaginary parts of ritz values
	std::vector<Base<T>> ritzvi(n, Base<T>(0.0)); 
 
	//Performs Q_2^T Q_2 for the construction of the dual basis, Q_2 is the lower part of Ql
	blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, k, &alpha,
		Q + k, ldq, Q + k, ldq, &Zero, A, lda);

       	//Compute the scaling weights such that diag = Ql^T Qr
	for(auto i = 0; i < n; i++)
	{
		diag[i] = One / (One-A[i*(lda+1)]);
	}	

       	//Matrix to compute the upper part of Ql
	for(auto i = 0; i < n; i++)
	{
		A[i*(lda+1)] = One;
	}

       	//Compute the upper part of Ql
	blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, n, n, &One,
		Q, ldq, A, lda, &Zero, halfQ, lda); //Performs Q_1

       	//Performs the multiplication of the first k cols of H with the upper part of Ql
        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, k, &One,
                H->data(), ldh, halfQ, lda, &Zero, V, ldv);
	
	//Flip the sign of the lower part to emulate the multiplication H' * Ql 
	alpha = -One;

       	//Matrix to compute the lower part of Ql
	for(auto i = 0; i < n; i++)
	{
		A[i*(lda+1)] = alpha;
	}

       	//Compute the lower part of Ql
	blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, k, n, n, &alpha,
		Q + k, ldq, A, lda, &Zero, halfQ, lda); //Performs Q_2

       	//Add to V the result of the multiplication of the last k cols of H with the lower part of Ql
        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, k, &One,
               	H->data() + k * N, ldh, halfQ, lda, &One, V, ldv);
	
	//Flip the sign of the lower part of V to emulate the multiplication H' * Ql 
	chase::linalg::internal::cpu::flipLowerHalfMatrixSign(N,n,V,ldv);
	
	//Last GEMM for the construction of the rayleigh Quotient : (H' * Ql)' * Qr
        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, N,
               &One, V, ldv, Q, ldq, &Zero, A, lda); 

	//Scale the rows because Ql' * Qr = diag =/= I
	for(auto i = 0; i < n; i++)
	{
		blaspp::t_scal(n, &diag[i], A + i, lda);
	}

	//Compute the eigenpairs of the non-hermitian rayleigh quotient
        lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', n, A, lda, ritzv, ritzvi.data(), W, n);

	//Project ritz vectors back to the initial space
        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, n,
               &One, Q, ldq, W, n, &Zero, V, ldv);

    }
}
}
}
}
