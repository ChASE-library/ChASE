#pragma once 

#include <complex>
#include "algorithm/types.hpp"

namespace chase
{
namespace linalg
{
namespace blaspp
{

#define CBLAS_LAYOUT int
#define CblasConjTrans 1
#define CblasTrans 2
#define CblasNoTrans 3
#define CblasColMajor 1
#define LAPACK_COL_MAJOR 1
#define CBLAS_TRANSPOSE int
#define lapack_logical int
#define CBLAS_UPLO int
#define CBLAS_SIDE int
#define CblasLeft 1
#define CblasLower 1

template <typename T>
Base<T> t_nrm2(const std::size_t n, const T* x, const std::size_t incx);

template <typename T>
T t_dot(const std::size_t n, const T* x, const std::size_t incx, const T* y,
        const std::size_t incy);
        
template <typename T>
void t_scal(const std::size_t n, const T* a, T* x, const std::size_t incx);

template <typename T>
void t_axpy(const std::size_t n, const T* a, const T* x, const std::size_t incx,
            T* y, const std::size_t incy);

template <typename T>
void t_syherk(const char uplo, const char trans, const std::size_t n,
              const std::size_t k, const T* alpha, T* a, const std::size_t lda,
              const T* beta, T* c, const std::size_t ldc);

template <typename T>
void t_trsm(const char side, const char uplo, const char trans, const char diag,
            const std::size_t m, const std::size_t n, const T* alpha,
            const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb);

template <typename T>
void t_gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
            const CBLAS_TRANSPOSE transb, const std::size_t m,
            const std::size_t n, const std::size_t k, const T* alpha,
            const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb, const T* beta, T* c, const std::size_t ldc);

} //end of namespace blaspp
} //end of namespace linalg   
} //end of namespace chase

#include "blaspp.inc"