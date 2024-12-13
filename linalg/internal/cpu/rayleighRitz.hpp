// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

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
    * @brief Perform the Rayleigh-Ritz procedure to compute eigenvalues and eigenvectors of a matrix.
    *
    * The Rayleigh-Ritz method computes an approximation to the eigenvalues and eigenvectors of a matrix
    * by projecting the matrix onto a subspace defined by a set of vectors (Q) and solving the eigenvalue
    * problem for the reduced matrix. The computed Ritz values are stored in the `ritzv` array, and the 
    * resulting eigenvectors are stored in `W`.
    *
    * @tparam T Data type for the matrix (e.g., float, double, etc.).
    * @param[in] N The number of rows of the matrix H.
    * @param[in] H The input matrix (N x N).
    * @param[in] ldh The leading dimension of the matrix H.
    * @param[in] n The number of vectors in Q (subspace size).
    * @param[in] Q The input matrix of size (N x n), whose columns are the basis vectors for the subspace.
    * @param[in] ldq The leading dimension of the matrix Q.
    * @param[out] W The output matrix (N x n), which will store the result of the projection.
    * @param[in] ldw The leading dimension of the matrix W.
    * @param[out] ritzv The array of Ritz values, which contains the eigenvalue approximations.
    * @param[in] A A temporary matrix used in intermediate calculations. If not provided, it is allocated internally.
    *
    * The procedure performs the following steps:
    * 1. Computes the matrix-vector multiplication: W = H * Q.
    * 2. Computes A = W' * Q, where W' is the conjugate transpose of W.
    * 3. Solves the eigenvalue problem for A using LAPACK's `heevd` function, computing the Ritz values in `ritzv`.
    * 4. Computes the final approximation to the eigenvectors by multiplying Q with the computed eigenvectors.
    */    
    template<typename T>
    void rayleighRitz(std::size_t N, T *H, std::size_t ldh, std::size_t n, T *Q, std::size_t ldq, 
                    T * W, std::size_t ldw, Base<T> *ritzv, T *A = nullptr)
    {
        std::unique_ptr<T[]> ptr;

        if (A == nullptr)
        {
            ptr = std::unique_ptr<T[]>{new T[n * n]};
            A = ptr.get();
        }

        T One = T(1.0);
        T Zero = T(0.0);

        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, N, n, N, &One,
               H, ldh, Q, ldq, &Zero, W, ldw);

        // A <- W' * V
        blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, n, n, N,
               &One, W, ldw, Q, ldq, &Zero, A, n);

        lapackpp::t_heevd(LAPACK_COL_MAJOR, 'V', 'L', n, A, n, ritzv);

        blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, n, n,
               &One, Q, ldq, A, n, &Zero, W, ldw);
    }
}
}
}
}