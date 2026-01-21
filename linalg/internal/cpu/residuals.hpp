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
 * @brief Compute the residuals of eigenvectors for a given matrix and
 * eigenvalues.
 *
 * This function computes the residuals of the eigenvectors, which measure how
 * well the eigenvectors satisfy the eigenvalue equation \( H \mathbf{v}_i =
 * \lambda_i \mathbf{v}_i \). The residual for each eigenvector \( \mathbf{v}_i
 * \) is defined as \( ||H \mathbf{v}_i - \lambda_i \mathbf{v}_i|| \), where \(
 * \lambda_i \) is the corresponding eigenvalue. The computed residuals are
 * stored in the `resids` array.
 *
 * @tparam T Data type for the matrix and vectors (e.g., float, double, etc.).
 * @param[in] N The number of rows and columns of the matrix H.
 * @param[in] H The input matrix (N x N).
 * @param[in] ldh The leading dimension of the matrix H.
 * @param[in] eigen_nb The number of eigenvalues and eigenvectors.
 * @param[in] evals The array of eigenvalues.
 * @param[in] evecs The matrix of eigenvectors (N x eigen_nb), where each column
 * is an eigenvector.
 * @param[in] ldv The leading dimension of the matrix evecs.
 * @param[out] resids The array that will store the computed residuals for each
 * eigenvector.
 * @param[in] V A temporary matrix used in intermediate calculations. If not
 * provided, it is allocated internally.
 *
 * The function performs the following steps:
 * 1. Computes the matrix-vector multiplication \( V = H \cdot E \), where E are
 * the eigenvectors.
 * 2. Subtracts the eigenvalue \( \lambda_i \) times the eigenvector from the
 * result.
 * 3. Computes the 2-norm of the residual for each eigenvector and stores it in
 * the `resids` array.
 */
template <typename T>
void residuals(std::size_t N, T* H, std::size_t ldh, std::size_t eigen_nb,
               Base<T>* evals, T* evecs, std::size_t ldv, Base<T>* resids,
               T* V = nullptr)
{
    std::unique_ptr<T[]> ptr;

    if (V == nullptr)
    {
        ptr = std::unique_ptr<T[]>{new T[N * eigen_nb]};
        V = ptr.get();
    }

    T alpha = T(1.0);
    T beta = T(0.0);

    blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, eigen_nb, N,
                   &alpha, H, ldh, evecs, ldv, &beta, V, N);

    for (std::size_t i = 0; i < eigen_nb; ++i)
    {
        beta = T(-evals[i]);
        blaspp::t_axpy(N, &beta, evecs + ldv * i, 1, V + N * i, 1);

        resids[i] = blaspp::t_nrm2(N, V + N * i, 1);
    }
}

} // namespace cpu
} // namespace internal
} // namespace linalg
} // namespace chase
