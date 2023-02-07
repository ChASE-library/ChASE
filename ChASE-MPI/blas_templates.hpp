/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>

#include "algorithm/types.hpp"

namespace chase
{
namespace mpi
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
Base<T> t_sqrt_norm(T x);

template <typename T>
Base<T> t_lange(char norm, std::size_t m, std::size_t n, T* A, std::size_t lda);

template <typename T>
void t_gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
            const CBLAS_TRANSPOSE transb, const std::size_t m,
            const std::size_t n, const std::size_t k, const T* alpha,
            const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb, const T* beta, T* c, const std::size_t ldc);

template <typename T>
void t_syhemm(const char side, const char uplo, const std::size_t m,
              const std::size_t n, const T* alpha, T* a, const std::size_t lda,
              T* b, const std::size_t ldb, const T* beta, T* c,
              const std::size_t ldc);

template <typename T>
void t_syherk(const char uplo, const char trans, const std::size_t n,
              const std::size_t k, const T* alpha, T* a, const std::size_t lda,
              const T* beta, T* c, const std::size_t ldc);

template <typename T>
void t_axpy(const std::size_t n, const T* a, const T* x, const std::size_t incx,
            T* y, const std::size_t incy);

template <typename T>
Base<T> t_nrm2(const std::size_t n, const T* x, const std::size_t incx);

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
T t_dot(const std::size_t n, const T* x, const std::size_t incx, const T* y,
        const std::size_t incy);

template <typename T>
int t_potrf(const char uplo, const std::size_t n, T* a, const std::size_t lda);

template <typename T>
void t_trsm(const char side, const char uplo, const char trans, const char diag,
            const std::size_t m, const std::size_t n, const T* alpha,
            const T* a, const std::size_t lda, const T* b,
            const std::size_t ldb);

// scalapack
// BLACS
extern "C" void blacs_get_(int*, int*, int*);
extern "C" void blacs_pinfo_(int*, int*);
extern "C" void blacs_gridinit_(int*, char*, int*, int*);
extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
extern "C" void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*,
                          int*);
extern "C" void blacs_gridexit_(int*);
extern "C" void blacs_gridmap_(int*, int*, int*, int*, int*);

void t_descinit(std::size_t* desc, std::size_t* m, std::size_t* n,
                std::size_t* mb, std::size_t* nb, int* irsrc, int* icsrc,
                int* ictxt, std::size_t* lld, int* info);

#if defined(HAS_SCALAPACK)
// ScaLAPACK
extern "C" void pdgeqrf_(int*, int*, double*, int*, int*, int*, double*,
                         double*, int*, int*);
extern "C" void psgeqrf_(int*, int*, float*, int*, int*, int*, float*, float*,
                         int*, int*);
extern "C" void pcgeqrf_(int*, int*, std::complex<float>*, int*, int*, int*,
                         std::complex<float>*, std::complex<float>*, int*,
                         int*);
extern "C" void pzgeqrf_(int*, int*, std::complex<double>*, int*, int*, int*,
                         std::complex<double>*, std::complex<double>*, int*,
                         int*);

extern "C" void pdorgqr_(int*, int*, int*, double*, int*, int*, int*, double*,
                         double*, int*, int*);
extern "C" void psorgqr_(int*, int*, int*, float*, int*, int*, int*, float*,
                         float*, int*, int*);
extern "C" void pzungqr_(int*, int*, int*, std::complex<double>*, int*, int*,
                         int*, std::complex<double>*, std::complex<double>*,
                         int*, int*);
extern "C" void pcungqr_(int*, int*, int*, std::complex<float>*, int*, int*,
                         int*, std::complex<float>*, std::complex<float>*, int*,
                         int*);
extern "C" void pdgeadd_(const char *, int *, int *, double *, double *, int *, int *, int *, double *, double *, int *, int *, int *);
extern "C" void psgeadd_(const char *, int *, int *, float *, float *, int *, int *, int *, float *, float *, int *, int *, int *);
extern "C" void pzgeadd_(const char *, int *, int *, std::complex<double> *, std::complex<double>*, int *, int *, int *, std::complex<double> *, std::complex<double> *, int *, int *, int *);
extern "C" void pcgeadd_(const char *, int *, int *, std::complex<float> *, std::complex<float>*, int *, int *, int *, std::complex<float> *, std::complex<float> *, int *, int *, int *);
extern "C" void pdpotrf_(const char *, int *, double *, int *, int *, int *, int *);
extern "C" void pspotrf_(const char *, int *, float *, int *, int *, int *, int *);
extern "C" void pzpotrf_(const char *, int *, std::complex<double> *, int *, int *, int *, int *);
extern "C" void pcpotrf_(const char *, int *, std::complex<float> *, int *, int *, int *, int *);
extern "C" void pdsygst_(int *, const char *, int *, double *,  int *, int *, int *, double *, int *, int *, int *, double *, int *);
extern "C" void pssygst_(int *, const char *, int *, float *,  int *, int *, int *, float *, int *, int *, int *, float *, int *);
extern "C" void pzhegst_(int *, const char *, int *, std::complex<double> *,  int *, int *, int *, std::complex<double> *, int *, int *, int *, double *, int *);
extern "C" void pchegst_(int *, const char *, int *, std::complex<float> *,  int *, int *, int *, std::complex<float> *, int *, int *, int *, float *, int *);
extern "C" void pdtrtrs_(const char *, const char *, const char *, int *, int *, double *, int *, int *, int *, double *, int *, int *, int *, int *);
extern "C" void pstrtrs_(const char *, const char *, const char *, int *, int *, float *, int *, int *, int *, float *, int *, int *, int *, int *);
extern "C" void pctrtrs_(const char *, const char *, const char *, int *, int *, std::complex<float> *, int *, int *, int *, std::complex<float> *, int *, int *, int *, int *);
extern "C" void pztrtrs_(const char *, const char *, const char *, int *, int *, std::complex<double> *, int *, int *, int *, std::complex<double> *, int *, int *, int *, int *);

template <typename T>
void t_pgeqrf(std::size_t m, std::size_t n, T* A, int ia, int ja,
              std::size_t* desc_a, T* tau);
template <typename T>
void t_pgqr(std::size_t m, std::size_t n, std::size_t k, T* A, int ia, int ja,
            std::size_t* desc_a, T* tau);
template <typename T>
void t_geadd(const char trans, const std::size_t m, std::size_t n, T alpha,
             T* a, const std::size_t ia, const std::size_t ja,
             std::size_t* desc_a, T beta, T* c, const std::size_t ic,
             const std::size_t jc, std::size_t* desc_c);
template <typename T>
int t_ppotrf(const char uplo, const std::size_t n, T* a, const std::size_t ia,
              const std::size_t ja, std::size_t* desc_a);

template <typename T>
int t_psyhegst(const int ibtype, const char uplo, const std::size_t n, T* a,
                const std::size_t ia, const std::size_t ja, std::size_t* desc_a,
                T* b, const std::size_t ib, const std::size_t jb,
                std::size_t* desc_b, Base<T>* scale);
template <typename T>
int t_ptrtrs(const char uplo, const char trans, const char diag,
              const std::size_t n, const std::size_t nhs, T* a,
              const std::size_t ia, const std::size_t ja, std::size_t* desc_a,
              T* b, const std::size_t ib, const std::size_t jb,
              std::size_t* desc_b);
#endif

} // namespace mpi
} // namespace chase

#include "blas_templates.inc"
