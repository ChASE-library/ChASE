// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"
#include "external/blaspp/blaspp.hpp"
#include <algorithm>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cpu
{
/**
 * @brief Computes the sum of the absolute values of the diagonal elements of a
 * matrix.
 *
 * This function computes the sum of the absolute values of the diagonal
 * elements of the given matrix \( A \). It iterates over the diagonal elements
 * (i.e., elements where the row index equals the column index) and adds the
 * absolute value of each diagonal element to the `sum`.
 *
 * @tparam T Data type for the matrix elements (e.g., float, double).
 * @param[in] m The number of rows in the matrix \( A \).
 * @param[in] n The number of columns in the matrix \( A \).
 * @param[in] A The matrix of size \( m \times n \), where the diagonal elements
 * are summed.
 * @param[in] lda The leading dimension of the matrix \( A \), which is the
 * number of elements between the start of one column and the start of the next
 * column.
 * @param[out] sum The resulting sum of the absolute values of the diagonal
 * elements.
 */
template <typename T>
void computeDiagonalAbsSum(std::size_t m, std::size_t n, T* A, std::size_t lda,
                           Base<T>* sum)
{
    *sum = Base<T>(0.0);

    std::size_t cnt = std::min(m, n);

    for (auto i = 0; i < cnt; i++)
    {
        *sum += std::abs(A[i * lda + i]);
    }
}
/**
 * @brief Shifts the diagonal elements of a matrix by a given value.
 *
 * This function adds a specified shift value to the diagonal elements of the
 * given matrix \( A \). It modifies the matrix in place by adding the `shift`
 * to each diagonal element (i.e., elements where the row index equals the
 * column index).
 *
 * @tparam T Data type for the matrix elements (e.g., float, double).
 * @param[in] m The number of rows in the matrix \( A \).
 * @param[in] n The number of columns in the matrix \( A \).
 * @param[in,out] A The matrix of size \( m \times n \), whose diagonal elements
 * are shifted.
 * @param[in] lda The leading dimension of the matrix \( A \), which is the
 * number of elements between the start of one column and the start of the next
 * column.
 * @param[in] shift The value to be added to each diagonal element.
 */
template <typename T>
void shiftMatrixDiagonal(std::size_t m, std::size_t n, T* A, std::size_t lda,
                         T shift)
{
    std::size_t cnt = std::min(m, n);

    for (auto i = 0; i < cnt; i++)
    {
        A[i * lda + i] += shift;
    }
}

/**
 * @brief Flip the sign of the lower half part of the matrix
 *
 * This function toggles the sign of the lower half part of the matrix, i.e.,
 * the lower half part is multiplied by -1.0
 *
 * @tparam T Data type for the matrix elements (e.g., float, double).
 * @param[in] m The number of rows in the matrix \( A \).
 * @param[in] n The number of columns in the matrix \( A \).
 * @param[in,out] A The matrix of size \( m \times n \), whose diagonal elements
 * are shifted.
 * @param[in] lda The leading dimension of the matrix \( A \), which is the
 * number of elements between the start of one column and the start of the next
 * column.
 */
template <typename T>
void flipLowerHalfMatrixSign(std::size_t m, std::size_t n, T* A,
                             std::size_t lda)
{
    std::size_t half = m / 2;

    T alpha = -T(1.0);

    for (auto j = 0; j < n; j++)
    {
        chase::linalg::blaspp::t_scal(m - half, &alpha, A + half + j * lda, 1);
    }
}

/**
 * @brief Flip the sign of the right part of the matrix
 *
 * This function toggles the sign of the right part of the matrix, i.e., the
 * right part is multiplied by -1.0
 *
 * @tparam T Data type for the matrix elements (e.g., float, double).
 * @param[in] m The number of rows in the matrix \( A \).
 * @param[in] n The number of columns in the matrix \( A \).
 * @param[in,out] A The matrix of size \( m \times n \), whose diagonal elements
 * are shifted.
 * @param[in] lda The leading dimension of the matrix \( A \), which is the
 * number of elements between the start of one column and the start of the next
 * column.
 */
template <typename T>
void flipRightHalfMatrixSign(std::size_t m, std::size_t n, T* A,
                             std::size_t lda)
{
    std::size_t half = n / 2;

    T alpha = -T(1.0);

    chase::linalg::blaspp::t_scal(m * half, &alpha, A + half * lda, 1);
}

} // namespace cpu
} // namespace internal
} // namespace linalg
} // namespace chase
