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