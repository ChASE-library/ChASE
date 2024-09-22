#pragma once

#include "linalg/blaspp/blaspp.hpp"
#include "linalg/lapackpp/lapackpp.hpp"
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
    void residuals(std::size_t N, T *H, std::size_t ldh, std::size_t eigen_nb, 
                    Base<T> *evals, T *evecs, std::size_t ldv, Base<T> *resids, T *V = nullptr)
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
               &alpha, H, ldh, evecs, ldv, &beta,
               V, N);

        for (std::size_t i = 0; i < eigen_nb; ++i)
        {
            beta = T(-evals[i]);
            blaspp::t_axpy(N, &beta, evecs + ldv * i, 1,
                   V + N * i, 1);

            resids[i] = blaspp::t_nrm2(N, V + N * i, 1);
        }
    }

}
}
}
}