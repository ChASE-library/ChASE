// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <assert.h>
#include <memory>
#include <vector>

#include "external/lapackpp/lapack_fortran.hpp"
#include "fortran_mangle.h"

namespace chase
{
namespace linalg
{
namespace lapackpp
{
template <>
void t_lacpy(const char uplo, const std::size_t m, const std::size_t n,
             const float* a, const std::size_t lda, float* b,
             const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    FC_GLOBAL(slacpy, SLACPY)(&uplo, &m_, &n_, a, &lda_, b, &ldb_);
}

template <>
void t_lacpy(const char uplo, const std::size_t m, const std::size_t n,
             const double* a, const std::size_t lda, double* b,
             const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    FC_GLOBAL(dlacpy, DLACPY)(&uplo, &m_, &n_, a, &lda_, b, &ldb_);
}

template <>
void t_lacpy(const char uplo, const std::size_t m, const std::size_t n,
             const std::complex<float>* a, const std::size_t lda,
             std::complex<float>* b, const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    FC_GLOBAL(clacpy, CLACPY)(&uplo, &m_, &n_, a, &lda_, b, &ldb_);
}

template <>
void t_lacpy(const char uplo, const std::size_t m, const std::size_t n,
             const std::complex<double>* a, const std::size_t lda,
             std::complex<double>* b, const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    FC_GLOBAL(zlacpy, ZLACPY)(&uplo, &m_, &n_, a, &lda_, b, &ldb_);
}

template <>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n, double* a,
                    std::size_t lda, double* tau)
{
    // return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
    using T = std::remove_reference<decltype((a[0]))>::type;

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;

    T* work;
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(dgeqrf, DGEQRF)(&m_, &n_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>((numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(dgeqrf, DGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}
template <>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n, float* a,
                    std::size_t lda, float* tau)
{
    // return LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau);
    using T = std::remove_reference<decltype((a[0]))>::type;

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;

    T* work;
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(sgeqrf, SGEQRF)(&m_, &n_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>((numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(sgeqrf, SGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n,
                    std::complex<double>* a, std::size_t lda,
                    std::complex<double>* tau)
{
    // return LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);

    using T = std::remove_reference<decltype((a[0]))>::type;

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;

    T* work;
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(zgeqrf, ZGEQRF)(&m_, &n_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(zgeqrf, ZGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n,
                    std::complex<float>* a, std::size_t lda,
                    std::complex<float>* tau)
{
    using T = std::remove_reference<decltype((a[0]))>::type;

    //  return LAPACKE_cgeqrf(matrix_layout, m, n, a, lda, tau);
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;

    T* work;
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(cgeqrf, CGEQRF)(&m_, &n_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(cgeqrf, CGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}

// Overload of ?gqr functions
template <>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, double* a, std::size_t lda, const double* tau)
{
    // return LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau);

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;

    double* work;
    double numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(dorgqr, DORGQR)
    (&m_, &n_, &k_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(numwork);
    auto ptr = std::vector<double>(numwork);
    work = ptr.data();

    FC_GLOBAL(dorgqr, DORGQR)
    (&m_, &n_, &k_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}
template <>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, float* a, std::size_t lda, const float* tau)
{
    // return LAPACKE_sorgqr(matrix_layout, m, n, k, a, lda, tau);

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;

    float* work;
    float numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(sorgqr, SORGQR)
    (&m_, &n_, &k_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(numwork);
    auto ptr = std::vector<float>(numwork);
    work = ptr.data();

    FC_GLOBAL(sorgqr, SORGQR)
    (&m_, &n_, &k_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, std::complex<double>* a, std::size_t lda,
                  const std::complex<double>* tau)
{
    //  return LAPACKE_zungqr(matrix_layout, m, n, k, a, lda, tau);

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;

    std::complex<double>* work;
    std::complex<double> numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(zungqr, ZUNGQR)
    (&m_, &n_, &k_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr = std::unique_ptr<std::complex<double>[]>{
        new std::complex<double>[lwork]};
    work = ptr.get();

    FC_GLOBAL(zungqr, ZUNGQR)
    (&m_, &n_, &k_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, std::complex<float>* a, std::size_t lda,
                  const std::complex<float>* tau)
{
    //  return LAPACKE_cungqr(matrix_layout, m, n, k, a, lda, tau);

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;

    std::complex<float>* work;
    std::complex<float> numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(cungqr, CUNGQR)
    (&m_, &n_, &k_, a, &lda_, tau, &numwork, &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr =
        std::unique_ptr<std::complex<float>[]>{new std::complex<float>[lwork]};
    work = ptr.get();

    FC_GLOBAL(cungqr, CUNGQR)
    (&m_, &n_, &k_, a, &lda_, tau, work, &lwork, &info);
    assert(info == 0);
    return info;
}

template <>
int t_potrf(const char uplo, const std::size_t n, float* a,
            const std::size_t lda)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt info;

    FC_GLOBAL(spotrf, SPOTRF)
    (&uplo, &n_, a, &lda_, &info);
    return info;
}

template <>
int t_potrf(const char uplo, const std::size_t n, double* a,
            const std::size_t lda)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt info;

    FC_GLOBAL(dpotrf, DPOTRF)
    (&uplo, &n_, a, &lda_, &info);

    return info;
}

template <>
int t_potrf(const char uplo, const std::size_t n, std::complex<float>* a,
            const std::size_t lda)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt info;

    FC_GLOBAL(cpotrf, CPOTRF)
    (&uplo, &n_, a, &lda_, &info);

    return info;
}

template <>
int t_potrf(const char uplo, const std::size_t n, std::complex<double>* a,
            const std::size_t lda)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt info;

    FC_GLOBAL(zpotrf, ZPOTRF)
    (&uplo, &n_, a, &lda_, &info);

    return info;
}

// Overload of ?stemr functions

template <>
std::size_t t_stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il,
                    std::size_t iu, int* m, double* w, double* z,
                    std::size_t ldz, std::size_t nzc, int* isuppz,
                    lapack_logical* tryrac)
{
    // return LAPACKE_dstemr(matrix_layout, jobz, range, n, d, e, vl, vu, il,
    // iu, m,
    //                       w, z, ldz, nzc, isuppz, tryrac);

    BlasInt n_ = n;
    BlasInt ldz_ = ldz;
    BlasInt il_ = il;
    BlasInt iu_ = iu;
    BlasInt nzc_ = nzc;
    BlasInt lwork, info, liwork;

    double* work;
    double numwork;
    BlasInt* iwork;
    BlasInt inumwork;

    lwork = -1;
    liwork = -1;

    FC_GLOBAL(dstemr, SSTEMR)
    (&jobz, &range, &n_, d, e, &vl, &vu, &il_, &iu_, m, w, z, &ldz_, &nzc_,
     isuppz, tryrac, &numwork, &lwork, &inumwork, &liwork, &info);

    lwork = static_cast<std::size_t>((numwork));
    auto ptr = std::unique_ptr<double[]>(new double[lwork]);
    work = ptr.get();

    liwork = static_cast<std::size_t>(inumwork);
    auto iptr = std::unique_ptr<BlasInt[]>{new BlasInt[liwork]};
    iwork = iptr.get();

    FC_GLOBAL(dstemr, DSTEMR)
    (&jobz, &range, &n_, d, e, &vl, &vu, &il_, &iu_, m, w, z, &ldz_, &nzc_,
     isuppz, tryrac, work, &lwork, iwork, &liwork, &info);

    assert(info == 0);
    return info;
}

template <>
std::size_t t_stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il,
                    std::size_t iu, int* m, float* w, float* z, std::size_t ldz,
                    std::size_t nzc, int* isuppz, lapack_logical* tryrac)
{
    // return LAPACKE_sstemr(matrix_layout, jobz, range, n, d, e, vl, vu, il,
    // iu, m,
    //                       w, z, ldz, nzc, isuppz, tryrac);

    BlasInt n_ = n;
    BlasInt ldz_ = ldz;
    BlasInt il_ = il;
    BlasInt iu_ = iu;
    BlasInt nzc_ = nzc;
    BlasInt lwork, info, liwork;

    float* work;
    float numwork;
    BlasInt* iwork;
    BlasInt inumwork;

    lwork = -1;
    liwork = -1;

    FC_GLOBAL(sstemr, SSTEMR)
    (&jobz, &range, &n_, d, e, &vl, &vu, &il_, &iu_, m, w, z, &ldz_, &nzc_,
     isuppz, tryrac, &numwork, &lwork, &inumwork, &liwork, &info);

    lwork = static_cast<std::size_t>((numwork));
    auto ptr = std::unique_ptr<float[]>{new float[lwork]};
    work = ptr.get();

    liwork = static_cast<std::size_t>(inumwork);
    auto iptr = std::unique_ptr<BlasInt[]>{new BlasInt[liwork]};
    iwork = iptr.get();

    FC_GLOBAL(sstemr, SSTEMR)
    (&jobz, &range, &n_, d, e, &vl, &vu, &il_, &iu_, m, w, z, &ldz_, &nzc_,
     isuppz, tryrac, work, &lwork, iwork, &liwork, &info);

    assert(info == 0);
    return info;
}

// Overload of ?heevd functions
template <>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    double* a, std::size_t lda, double* w)
{
    // return LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w);
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt lwork, info, liwork;

    double* work;
    double numwork;
    BlasInt* iwork;
    BlasInt inumwork;

    lwork = -1;
    liwork = -1;

    FC_GLOBAL(dsyevd, DSYEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, &numwork, &lwork, &inumwork, &liwork,
     &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(numwork);
    auto ptr = std::vector<double>(lwork);
    work = ptr.data();

    liwork = static_cast<std::size_t>(inumwork);
    auto iptr = std::vector<BlasInt>(liwork);
    iwork = iptr.data();

    FC_GLOBAL(dsyevd, DSYEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, work, &lwork, iwork, &liwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    float* a, std::size_t lda, float* w)
{
    // return LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, a, lda, w);

    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt lwork, info, liwork;

    float* work;
    float numwork;
    BlasInt* iwork;
    BlasInt inumwork;

    lwork = -1;
    liwork = -1;

    FC_GLOBAL(ssyevd, SSYEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, &numwork, &lwork, &inumwork, &liwork,
     &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(numwork);
    auto ptr = std::vector<float>(lwork);
    work = ptr.data();

    liwork = static_cast<std::size_t>(inumwork);
    auto iptr = std::vector<BlasInt>(liwork);
    iwork = iptr.data();

    FC_GLOBAL(ssyevd, SSYEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, work, &lwork, iwork, &liwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    std::complex<double>* a, std::size_t lda, double* w)
{
    // return LAPACKE_zheevd(matrix_layout, jobz, uplo, n, a, lda, w);

    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt lwork, info, lrwork, liwork;

    std::complex<double>* work;
    std::complex<double> numwork;
    double* rwork;
    double rnumwork;
    BlasInt* iwork;
    BlasInt inumwork;

    lwork = -1;
    lrwork = -1;
    liwork = -1;

    FC_GLOBAL(zheevd, ZHEEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, &numwork, &lwork, &rnumwork, &lrwork,
     &inumwork, &liwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr = std::unique_ptr<std::complex<double>[]>{
        new std::complex<double>[lwork]};
    work = ptr.get();

    lrwork = static_cast<std::size_t>(rnumwork);
    auto rptr = std::unique_ptr<double[]>{new double[lrwork]};
    rwork = rptr.get();

    liwork = static_cast<std::size_t>(inumwork);
    auto iptr = std::unique_ptr<BlasInt[]>{new BlasInt[liwork]};
    iwork = iptr.get();

    FC_GLOBAL(zheevd, ZHEEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, work, &lwork, rwork, &lrwork, iwork,
     &liwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    std::complex<float>* a, std::size_t lda, float* w)
{
    // return LAPACKE_cheevd(matrix_layout, jobz, uplo, n, a, lda, w);

    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt lwork, info, lrwork, liwork;

    std::complex<float>* work;
    std::complex<float> numwork;
    float* rwork;
    float rnumwork;
    BlasInt* iwork;
    BlasInt inumwork;

    lwork = -1;
    lrwork = -1;
    liwork = -1;

    FC_GLOBAL(cheevd, CHEEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, &numwork, &lwork, &rnumwork, &lrwork,
     &inumwork, &liwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr =
        std::unique_ptr<std::complex<float>[]>{new std::complex<float>[lwork]};
    work = ptr.get();

    lrwork = static_cast<std::size_t>(rnumwork);
    auto rptr = std::unique_ptr<float[]>{new float[lrwork]};
    rwork = rptr.get();

    liwork = static_cast<std::size_t>(inumwork);
    auto iptr = std::unique_ptr<BlasInt[]>{new BlasInt[liwork]};
    iwork = iptr.get();

    FC_GLOBAL(cheevd, CHEEVD)
    (&jobz, &uplo, &n_, a, &lda_, w, work, &lwork, rwork, &lrwork, iwork,
     &liwork, &info);
    assert(info == 0);
    return info;
}

template <>
std::size_t t_geev(int matrix_layout, char jobz, std::size_t n, float* a,
                   std::size_t lda, float* wr, float* wi, float* V,
                   std::size_t ldv)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldv_ = ldv;

    float* work;
    float numwork;

    BlasInt lwork = -1, info;

    char jobvl = 'N';
    BlasInt ldvl_ = 1;
    float* Vl = (float*)0;

    FC_GLOBAL(sgeev, SGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, wr, wi, Vl, &ldvl_, V, &ldv_, &numwork,
     &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(numwork);
    auto ptr = std::vector<float>(lwork);
    work = ptr.data();

    FC_GLOBAL(sgeev, SGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, wr, wi, Vl, &ldvl_, V, &ldv_, work, &lwork,
     &info);
    assert(info == 0);

    return info;
}

template <>
std::size_t t_geev(int matrix_layout, char jobz, std::size_t n, double* a,
                   std::size_t lda, double* wr, double* wi, double* V,
                   std::size_t ldv)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldv_ = ldv;

    double* work;
    double numwork;

    BlasInt lwork = -1, info;

    char jobvl = 'N';
    BlasInt ldvl_ = 1;
    double* Vl = (double*)0;

    FC_GLOBAL(dgeev, DGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, wr, wi, Vl, &ldvl_, V, &ldv_, &numwork,
     &lwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(numwork);
    auto ptr = std::vector<double>(lwork);
    work = ptr.data();

    FC_GLOBAL(dgeev, DGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, wr, wi, Vl, &ldvl_, V, &ldv_, work, &lwork,
     &info);
    assert(info == 0);

    return info;
}

template <>
std::size_t t_geev(int matrix_layout, char jobz, std::size_t n,
                   std::complex<float>* a, std::size_t lda, float* wr,
                   float* wi, std::complex<float>* V, std::size_t ldv)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldv_ = ldv;

    std::complex<float>* work;
    std::complex<float> numwork;

    BlasInt lwork = -1, info;

    char jobvl = 'N';
    BlasInt ldvl_ = 1;
    std::complex<float>* Vl = (std::complex<float>*)0;

    auto wptr =
        std::unique_ptr<std::complex<float>[]>{new std::complex<float>[n]};
    std::complex<float>* w = wptr.get();

    auto rptr = std::vector<float>(2 * n);
    float* rwork = rptr.data();

    FC_GLOBAL(cgeev, CGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, w, Vl, &ldvl_, V, &ldv_, &numwork, &lwork,
     rwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto numptr =
        std::unique_ptr<std::complex<float>[]>{new std::complex<float>[lwork]};
    work = numptr.get();

    FC_GLOBAL(cgeev, CGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, w, Vl, &ldvl_, V, &ldv_, work, &lwork, rwork,
     &info);
    assert(info == 0);

    for (auto i = 0; i < n; i++)
    {
        wr[i] = real(w[i]);
        wi[i] = imag(w[i]);
    }

    return info;
}

template <>
std::size_t t_geev(int matrix_layout, char jobz, std::size_t n,
                   std::complex<double>* a, std::size_t lda, double* wr,
                   double* wi, std::complex<double>* V, std::size_t ldv)
{
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldv_ = ldv;

    std::complex<double>* work;
    std::complex<double> numwork;

    BlasInt lwork = -1, info;

    char jobvl = 'N';
    BlasInt ldvl_ = 1;
    std::complex<double>* Vl = (std::complex<double>*)0;

    auto wptr =
        std::unique_ptr<std::complex<double>[]>{new std::complex<double>[n]};
    std::complex<double>* w = wptr.get();

    auto rptr = std::vector<double>(2 * n);
    double* rwork = rptr.data();

    FC_GLOBAL(zgeev, ZGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, w, Vl, &ldvl_, V, &ldv_, &numwork, &lwork,
     rwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto numptr = std::unique_ptr<std::complex<double>[]>{
        new std::complex<double>[lwork]};
    work = numptr.get();

    FC_GLOBAL(zgeev, ZGEEV)
    (&jobvl, &jobz, &n_, a, &lda_, w, Vl, &ldvl_, V, &ldv_, work, &lwork, rwork,
     &info);
    assert(info == 0);

    for (auto i = 0; i < n; i++)
    {
        wr[i] = real(w[i]);
        wi[i] = imag(w[i]);
    }

    return info;
}

template <>
void t_gesvd(const char jobu, const char jobvt, const std::size_t m,
             const std::size_t n, float* A, const std::size_t lda, float* S,
             float* U, const std::size_t ldu, float* Vt, const std::size_t ldvt)
{
    using T = std::remove_reference<decltype((A[0]))>::type;
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldu_ = ldu;
    BlasInt ldvt_ = ldvt;

    T* work;
    Base<T>* rwork = new Base<T>[5 * std::min(m, n)];
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(sgesvd, SGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, &numwork, &lwork, rwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>((numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(sgesvd, SGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, work, &lwork, rwork, &info);
    assert(info == 0);
}

template <>
void t_gesvd(const char jobu, const char jobvt, const std::size_t m,
             const std::size_t n, double* A, const std::size_t lda, double* S,
             double* U, const std::size_t ldu, double* Vt,
             const std::size_t ldvt)
{
    using T = std::remove_reference<decltype((A[0]))>::type;
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldu_ = ldu;
    BlasInt ldvt_ = ldvt;

    T* work;
    Base<T>* rwork = new Base<T>[5 * std::min(m, n)];
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(dgesvd, DGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, &numwork, &lwork, rwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>((numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(dgesvd, DGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, work, &lwork, rwork, &info);
    assert(info == 0);
}

template <>
void t_gesvd(const char jobu, const char jobvt, const std::size_t m,
             const std::size_t n, std::complex<float>* A, const std::size_t lda,
             float* S, std::complex<float>* U, const std::size_t ldu,
             std::complex<float>* Vt, const std::size_t ldvt)
{
    using T = std::remove_reference<decltype((A[0]))>::type;
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldu_ = ldu;
    BlasInt ldvt_ = ldvt;

    T* work;
    Base<T>* rwork = new Base<T>[5 * std::min(m, n)];
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(cgesvd, CGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, &numwork, &lwork, rwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(cgesvd, CGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, work, &lwork, rwork, &info);
    assert(info == 0);
}

template <>
void t_gesvd(const char jobu, const char jobvt, const std::size_t m,
             const std::size_t n, std::complex<double>* A,
             const std::size_t lda, double* S, std::complex<double>* U,
             const std::size_t ldu, std::complex<double>* Vt,
             const std::size_t ldvt)
{
    using T = std::remove_reference<decltype((A[0]))>::type;
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldu_ = ldu;
    BlasInt ldvt_ = ldvt;

    T* work;
    Base<T>* rwork = new Base<T>[5 * std::min(m, n)];
    T numwork;
    BlasInt lwork, info;

    lwork = -1;
    FC_GLOBAL(zgesvd, ZGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, &numwork, &lwork, rwork, &info);
    assert(info == 0);

    lwork = static_cast<std::size_t>(real(numwork));
    auto ptr = std::unique_ptr<T[]>{new T[lwork]};
    work = ptr.get();

    FC_GLOBAL(zgesvd, ZGESVD)(&jobu, &jobvt, &m_, &n_, A, &lda_, S, U, &ldu_,
                              Vt, &ldvt_, work, &lwork, rwork, &info);
    assert(info == 0);

    delete[] rwork;
}

} // end of namespace lapackpp
} // end of namespace linalg
} // end of namespace chase
