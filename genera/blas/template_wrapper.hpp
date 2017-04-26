/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#ifndef CHASE_TEMPLATE_WRAPPER_H
#define CHASE_TEMPLATE_WRAPPER_H

#include <complex>
#define MKL_Complex16 std::complex<double>
#define MKL_Complex8 std::complex<float>
#include <mkl.h>

template <typename T>
Base<T> t_lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda);

template <typename T>
void t_gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
            const CBLAS_TRANSPOSE transb, const std::size_t m,
            const std::size_t n, const std::size_t k, const T* alpha,
            const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb, const T* beta, T* c, const std::size_t ldc);

template <typename T>
void t_hemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side,
            const CBLAS_UPLO uplo, const std::size_t m, const std::size_t n,
            const T* alpha, const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb, const T* beta, T* c, const std::size_t ldc);

template <typename T>
void t_axpy(const std::size_t n, const T* a, const T* x, const std::size_t incx,
            T* y, const std::size_t incy);

template <typename T>
Base<T> t_nrm2(const std::size_t n, const T* x, const std::size_t incx);

template <typename T>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n, T* a,
                    std::size_t lda, T* tau);

template <typename T>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, T* a, std::size_t lda, const T* tau);

template <typename T>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    T* a, std::size_t lda, Base<T>* w);

template <typename T>
void t_scal(const std::size_t n, const T* a, T* x, const std::size_t incx);

template <typename T>
void t_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
            const std::size_t m, const std::size_t n, const T* alpha,
            const T* a, const std::size_t lda, const T* x,
            const std::size_t incx, const T* beta, T* y,
            const std::size_t incy);

template <typename T>
std::size_t t_stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    T* d, T* e, T vl, T vu, std::size_t il, std::size_t iu,
                    int* m, T* w, T* z, std::size_t ldz, std::size_t nzc,
                    int* isuppz, lapack_logical* tryrac);

template <typename T>
void t_dot(const std::size_t n, const T* x, const std::size_t incx, const T* y,
           const std::size_t incy, T* dotc);

#include "template_wrapper_impl.hpp"

#endif
