#pragma once

#include "algorithm/types.hpp"
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
    * @brief Computes the sum of the absolute values of the diagonal elements of a matrix.
    *
    * This function computes the sum of the absolute values of the diagonal elements of the given matrix \( A \).
    * It iterates over the diagonal elements (i.e., elements where the row index equals the column index) and adds 
    * the absolute value of each diagonal element to the `sum`.
    *
    * @tparam T Data type for the matrix elements (e.g., float, double).
    * @param[in] m The number of rows in the matrix \( A \).
    * @param[in] n The number of columns in the matrix \( A \).
    * @param[in] A The matrix of size \( m \times n \), where the diagonal elements are summed.
    * @param[in] lda The leading dimension of the matrix \( A \), which is the number of elements between the start 
    * of one column and the start of the next column.
    * @param[out] sum The resulting sum of the absolute values of the diagonal elements.
    */    
    template<typename T>
    void computeDiagonalAbsSum(std::size_t m, std::size_t n, T *A, std::size_t lda, Base<T> *sum)
    {
        *sum = Base<T>(0.0);

        std::size_t cnt = std::min(m, n);

        for(auto i = 0; i < cnt; i++)
        {
            *sum += std::abs(A[i * lda + i]);
        }

    }
    /**
    * @brief Shifts the diagonal elements of a matrix by a given value.
    *
    * This function adds a specified shift value to the diagonal elements of the given matrix \( A \).
    * It modifies the matrix in place by adding the `shift` to each diagonal element (i.e., elements where the 
    * row index equals the column index).
    *
    * @tparam T Data type for the matrix elements (e.g., float, double).
    * @param[in] m The number of rows in the matrix \( A \).
    * @param[in] n The number of columns in the matrix \( A \).
    * @param[in,out] A The matrix of size \( m \times n \), whose diagonal elements are shifted.
    * @param[in] lda The leading dimension of the matrix \( A \), which is the number of elements between the start 
    * of one column and the start of the next column.
    * @param[in] shift The value to be added to each diagonal element.
    */
    template<typename T>
    void shiftMatrixDiagonal(std::size_t m, std::size_t n, T *A, std::size_t lda, T shift)
    {
        std::size_t cnt = std::min(m, n);

        for(auto i = 0; i < cnt; i++)
        {
            A[i * lda + i] += shift;
        }
    }

}
}
}
}