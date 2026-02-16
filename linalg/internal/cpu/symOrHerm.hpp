// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "external/blaspp/blaspp.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cpu
{
/**
 * @brief Checks if a matrix is symmetric using a randomized approach.
 *
 * This function checks the symmetry of a square matrix \( H \) by performing
 * two matrix-vector multiplications:
 * 1. It computes \( u = H \cdot v \), where \( v \) is a random vector.
 * 2. It computes \( uT = H^T \cdot v \), where \( H^T \) is the transpose of \(
 * H \). The matrix is considered symmetric if the vectors \( u \) and \( uT \)
 * are the same, i.e., \( u = uT \).
 *
 * This method is computationally efficient and uses random vectors to test
 * symmetry with high probability. However, it is not a guarantee for exact
 * symmetry due to numerical errors, but it can be a quick heuristic check.
 *
 * @tparam T Data type for the matrix (e.g., float, double).
 * @param[in] N The size of the matrix (N x N).
 * @param[in] H The matrix to be checked for symmetry (of size N x N).
 * @param[in] ldh The leading dimension of the matrix H.
 * @return `true` if the matrix is symmetric, `false` otherwise.
 */
template <typename T>
bool checkSymmetryEasy(std::size_t N, T* H, std::size_t ldh)
{
    std::vector<T> v(N);
    std::vector<T> u(N);
    std::vector<T> uT(N);

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;
    for (auto i = 0; i < N; i++)
    {
        v[i] = getRandomT<T>([&]() { return d(gen); });
    }

    T One = T(1.0);
    T Zero = T(0.0);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N,
                                  1, N, &One, H, ldh, v.data(), N, &Zero,
                                  u.data(), N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                                  N, 1, N, &One, H, ldh, v.data(), N, &Zero,
                                  uT.data(), N);

    using R = chase::Base<T>;
    const R eps = std::numeric_limits<R>::epsilon();
    const R tol = static_cast<R>(N) * R(10) * eps;

    for (auto i = 0; i < N; i++)
    {
        R diff = std::abs(u[i] - uT[i]);
        R scale = std::max(std::abs(u[i]), std::abs(uT[i])) + R(1);
        if (diff > tol * scale)
            return false;
    }

    return true;
}
/**
 * @brief Converts a matrix to its Hermitian or symmetric form based on the
 * given `uplo` argument.
 *
 * This function modifies the matrix \( H \) in-place such that it becomes
 * symmetric or Hermitian, depending on the value of the `uplo` parameter.
 * - If `uplo` is `'U'`, the function converts the upper triangular part of the
 * matrix to the Hermitian form, by setting the lower triangular part to the
 * conjugate transpose of the upper part.
 * - If `uplo` is `'L'`, the function converts the lower triangular part of the
 * matrix to the Hermitian form, by setting the upper triangular part to the
 * conjugate transpose of the lower part.
 *
 * The function assumes that the matrix is square (N x N) and modifies the
 * elements of the matrix in-place. The conjugation is done using the
 * `conjugate` function.
 *
 * @tparam T Data type for the matrix (e.g., float, double, std::complex).
 * @param[in] uplo A character indicating which part of the matrix to modify:
 * - `'U'` for the upper triangular part.
 * - `'L'` for the lower triangular part.
 * @param[in,out] N The size of the matrix (N x N). The matrix is modified
 * in-place.
 * @param[in,out] H The matrix to be modified. It is transformed into a
 * symmetric or Hermitian matrix.
 * @param[in] ldh The leading dimension of the matrix H.
 */
template <typename T>
void symOrHermMatrix(char uplo, std::size_t N, T* H, std::size_t ldh)
{

    if (uplo == 'U')
    {
        for (auto j = 0; j < N; j++)
        {
            for (auto i = 0; i < j; i++)
            {
                H[j + i * ldh] = conjugate(H[i + j * ldh]);
            }
        }
    }
    else
    {
        for (auto i = 0; i < N; i++)
        {
            for (auto j = 0; j < i; j++)
            {
                H[j + i * ldh] = conjugate(H[i + j * ldh]);
            }
        }
    }
}
} // namespace cpu
} // namespace internal
} // namespace linalg
} // namespace chase