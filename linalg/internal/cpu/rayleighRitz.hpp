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