#pragma once

#include <complex>
#include "algorithm/types.hpp"
#include "external/blaspp/blaspp.hpp"

namespace chase
{
namespace linalg
{
namespace lapackpp
{
template <typename T>
void t_lacpy(const char uplo, const std::size_t m, const std::size_t n,
             const T* a, const std::size_t lda, T* b, const std::size_t ldb);

template <typename T>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n, T* a,
                    std::size_t lda, T* tau);

template <typename T>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, T* a, std::size_t lda, const T* tau);

template <typename T>
int t_potrf(const char uplo, const std::size_t n, T* a, const std::size_t lda);

template <typename T>
std::size_t t_stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    T* d, T* e, T vl, T vu, std::size_t il, std::size_t iu,
                    int* m, T* w, T* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac);

template <typename T>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    T* a, std::size_t lda, Base<T>* w);

} //end of namespace lapackpp
} //end of namespace linalg   
} //end of namespace chase

#include "lapackpp.inc"