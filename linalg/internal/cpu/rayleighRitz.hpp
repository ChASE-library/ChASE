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
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a Pseudo-Hermitian matrix.
    *
    * The Rayleigh-Ritz method computes an approximation to the eigenvalues and eigenvectors of a matrix
    * by projecting the matrix onto a subspace defined by a set of vectors (Q) and solving the eigenvalue
    * problem for the reduced matrix. The real parts of the computed Ritz values are stored in the `ritzv` array, and the 
    * resulting right eigenvectors are stored in `V`.
    *
    * @tparam T Data type for the matrix (e.g., float, double, etc.).
    * @param[in] H The Pseudo-Hermitian input matrix (N x N).
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
    void rayleighRitz(chase::matrix::PseudoHermitianMatrix<T> *H, std::size_t n, T *Q, std::size_t ldq, 
                      T * V, std::size_t ldv, Base<T> *ritzv, T *A = nullptr)
    {

	std::size_t N   = H->rows();	
	std::size_t ldh = H->ld();
	std::size_t k   = N / 2;

        T One   = T(1.0);
        T Zero  = T(0.0);
        T NegOne = T(-1.0);
        T NegativeTwo = T(-2.0);

        std::unique_ptr<T[]> ptrA;
	//Allocate space for the rayleigh quotient
        if (A == nullptr)
        {
            ptrA = std::unique_ptr<T[]>{new T[3 * n * n]};
            A = ptrA.get();
        }

        T* M = A + n * n;
        T* W = A + 2 * n * n;

	//Allocate the space for scaling weights. Can be Base<T> since reals?
	std::vector<T> diag(n, T(0.0)); 

	//Allocate the space for the imaginary parts of ritz values
	std::vector<Base<T>> ritzvi(n, Base<T>(0.0)); 
 
	blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, N, &One,
		H->data(), ldh, Q, ldq, &Zero, V, ldv);  //T = AQr

        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, N, &One, Q, ldq, V, ldv, &Zero, W, n);  //A = Qr^* T

	//Performs Q_2^T Q_2 for the construction of the dual basis, Q_2 is the lower part of Q
	blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, k, &NegativeTwo,
		Q + k, ldq, Q + k, ldq, &Zero, M, n); //M = -2 Qr_2^* Qr_2

        // M = I - 2 Qr_2^* Qr_2
        for(auto i = 0; i < n; i++)
        {
            diag[i] = T(1.0) / (T(1.0) + M[i*(n+1)]);
        }

        for(auto i = 0; i < n; i++)
        {
            M[i*(n+1)] = T(0.0);
        }

        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, &NegOne,
                M, n, W, n, &Zero, A, n); //A = (Diag(M) - M) * A

	//Flip the sign of the lower part of V to emulate the multiplication H' * Ql 
	chase::linalg::internal::cpu::flipLowerHalfMatrixSign(N,n,V,ldv); //flip T

	//Last GEMM for the construction of the rayleigh Quotient : (H' * Ql)' * Qr
        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, N,
               &One, Q, ldq, V, ldv, &One, A, n); 
        
	//Scale the rows because Ql' * Qr = diag =/= I
	for(auto i = 0; i < n; i++)
	{
		blaspp::t_scal(n, &diag[i], A + i, n);
	}
        
	//Compute the eigenpairs of the non-hermitian rayleigh quotient
        lapackpp::t_geev(LAPACK_COL_MAJOR, 'V', n, A, n, ritzv, ritzvi.data(), W, n);

	//Sort indices based on ritz values
	std::vector<Base<T>> sorted_ritzv(ritzv, ritzv + n);
	std::vector<std::size_t> indices(n);
	std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, ..., n-1
	std::sort(indices.begin(), indices.end(), 
			[&sorted_ritzv](std::size_t i1, std::size_t i2) { return sorted_ritzv[i1] < sorted_ritzv[i2]; });

	// Create temporary storage for sorted eigenvalues and eigenvectors
        T *sorted_W = A;
	// Reorder eigenvalues and eigenvectors
	for (std::size_t i = 0; i < n; ++i) {
		ritzv[i] = sorted_ritzv[indices[i]];
	}

        for (std::size_t i = 0; i < n; ++i) {
            std::copy_n(W + indices[i] * n, n, sorted_W + i * n);
        }

	// Copy back to original arrays
	std::copy(sorted_W, sorted_W + n * n, W);
	
	//Project ritz vectors back to the initial space
        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, n,
               &One, Q, ldq, W, n, &Zero, V, ldv);

    }
    
    /**
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a Pseudo-Hermitian matrix.
    *
    * The Rayleigh-Ritz method computes an approximation to the eigenvalues and eigenvectors of a matrix
    * by projecting the matrix onto a subspace defined by a set of vectors (Q) and solving the eigenvalue
    * problem for the reduced matrix. The real parts of the computed Ritz values are stored in the `ritzv` array, and the 
    * resulting right eigenvectors are stored in `V`.
    *
    * @tparam T Data type for the matrix (e.g., float, double, etc.).
    * @param[in] H The Pseudo-Hermitian input matrix (N x N).
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
    void rayleighRitz_v2(chase::matrix::PseudoHermitianMatrix<T> *H, std::size_t n, T *Q, std::size_t ldq, 
                      T * V, std::size_t ldv, Base<T> *ritzv, T *A = nullptr)
    {

	std::size_t N   = H->rows();	
	std::size_t ldh = H->ld();
	std::size_t k   = N / 2;

        T One   = T(1.0);
        T Zero  = T(0.0);
        T NegativeTwo = T(-2.0);

        std::unique_ptr<T[]> ptrA;
	//Allocate space for the rayleigh quotient
        if (A == nullptr)
        {
            ptrA = std::unique_ptr<T[]>{new T[2 * n * n]};
            A = ptrA.get();
        }

        T* M = A + n * n;

	blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, N, &One,
		H->data(), ldh, Q, ldq, &Zero, V, ldv);  //T = AQ
	
	//Flip the sign of the lower part of T to emulate the multiplication SAQ
	chase::linalg::internal::cpu::flipLowerHalfMatrixSign(N,n,V,ldv); //flip T

        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, N, &One, Q, ldq, V, ldv, &Zero, A, n);  //A = Q' T

	//Factorize the HPD mat Q' SA Q
	lapackpp::t_potrf('L',n,A,n);
        
	//Compute the matrix M  = Q' S Q = I - 2* Q_2' Q_2 assuming Q'Q = I
	std::fill(M, M + n*n, 0);
	for(auto i = 0; i < n; i++)
        {
            M[i*(n+1)] = T(1.0);
        }

	//Performs Q_2^T Q_2, Q_2 is the lower part of Q
	blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, k, &NegativeTwo,
		Q + k, ldq, Q + k, ldq, &One, M, n); //M = I -2 Q_2^* Q_2

	//Create the inverse of the Hermitian Rayleigh Quotient by two successive trsm 
	blaspp::t_trsm('L','L','N','N',n,n,&One,A,n,M,n);

	blaspp::t_trsm('R','L','C','N',n,n,&One,A,n,M,n);
        
	//Compute the invtered ritz pairs of the Hermitian Rayleigh Quotient
	lapackpp::t_heevd(LAPACK_COL_MAJOR, 'V', 'L', n, M, n, ritzv);

	blaspp::t_trsm('L','L','C','N',n,n,&One,A,n,M,n);	

	//Invert the ritz values and normalize the vectors 	
	std::vector<T> norms(n);

	for(auto idx = 0; idx < n; idx++)
	{
		ritzv[idx] = 1.0 / ritzv[idx];
	}
	for(auto idx = 0; idx < n; idx++)
	{
		norms[idx] = T(1.0/blaspp::t_nrm2(n, M + idx * n, 1));
	}
	for(auto idx = 0; idx < n; idx++)
	{
		blaspp::t_scal(n, &norms[idx], M + idx * n, 1);
	}

	//Project ritz vectors back to the initial space
        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, n,
               &One, Q, ldq, M, n, &Zero, V, ldv);

    }
}
}
}
}
