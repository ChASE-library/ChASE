// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <fortran_mangle.h>
#include "external/blaspp/blas_fortran.hpp"

namespace chase
{
namespace linalg
{
namespace lapackpp
{
extern "C"
{
    void FC_GLOBAL(slacpy, SLACPY)(const char* uplo, const BlasInt* m,
                                   const BlasInt* n, const float* a,
                                   const BlasInt* lda, float* b,
                                   const BlasInt* ldb);
    void FC_GLOBAL(dlacpy, DLACPY)(const char* uplo, const BlasInt* m,
                                   const BlasInt* n, const double* a,
                                   const BlasInt* lda, double* b,
                                   const BlasInt* ldb);
    void FC_GLOBAL(clacpy, CLACPY)(const char* uplo, const BlasInt* m,
                                   const BlasInt* n, const scomplex* a,
                                   const BlasInt* lda, scomplex* b,
                                   const BlasInt* ldb);
    void FC_GLOBAL(zlacpy, ZLACPY)(const char* uplo, const BlasInt* m,
                                   const BlasInt* n, const dcomplex* a,
                                   const BlasInt* lda, dcomplex* b,
                                   const BlasInt* ldb);

    // xGEQRF
    void FC_GLOBAL(sgeqrf, SGEQRF)(const BlasInt* m, const BlasInt* n, float* a,
                                   const BlasInt* lda, float* tau, float* work,
                                   const BlasInt* lwork, BlasInt* info);
    void FC_GLOBAL(dgeqrf, DGEQRF)(const BlasInt* m, const BlasInt* n,
                                   double* a, const BlasInt* lda, double* tau,
                                   double* work, const BlasInt* lwork,
                                   BlasInt* info);
    void FC_GLOBAL(cgeqrf, CGEQRF)(const BlasInt* m, const BlasInt* n,
                                   scomplex* a, const BlasInt* lda,
                                   scomplex* tau, scomplex* work,
                                   const BlasInt* lwork, BlasInt* info);
    void FC_GLOBAL(zgeqrf, ZGEQRF)(const BlasInt* m, const BlasInt* n,
                                   dcomplex* a, const BlasInt* lda,
                                   dcomplex* tau, dcomplex* work,
                                   const BlasInt* lwork, BlasInt* info);

    // xUNGQR / xGEQRF
    void FC_GLOBAL(sorgqr,
                   SORGQR)(const BlasInt* m, const BlasInt* n, const BlasInt* k,
                           float* a, const BlasInt* lda, const float* tau,
                           float* work, const BlasInt* lwork, BlasInt* info);
    void FC_GLOBAL(dorgqr,
                   DORGQR)(const BlasInt* m, const BlasInt* n, const BlasInt* k,
                           double* a, const BlasInt* lda, const double* tau,
                           double* work, const BlasInt* lwork, BlasInt* info);
    void FC_GLOBAL(cungqr,
                   CUNGQR)(const BlasInt* m, const BlasInt* n, const BlasInt* k,
                           scomplex* a, const BlasInt* lda, const scomplex* tau,
                           scomplex* work, const BlasInt* lwork, BlasInt* info);
    void FC_GLOBAL(zungqr,
                   ZUNGQR)(const BlasInt* m, const BlasInt* n, const BlasInt* k,
                           dcomplex* a, const BlasInt* lda, const dcomplex* tau,
                           dcomplex* work, const BlasInt* lwork, BlasInt* info);

    void FC_GLOBAL(spotrf, SPOTRF)(const char* uplo, const BlasInt* n, float* a,
                                   const BlasInt* lda, BlasInt* info);
    void FC_GLOBAL(dpotrf, DPOTRF)(const char* uplo, const BlasInt* n,
                                   double* a, const BlasInt* lda,
                                   BlasInt* info);
    void FC_GLOBAL(cpotrf, CPOTRF)(const char* uplo, const BlasInt* n,
                                   scomplex* a, const BlasInt* lda,
                                   BlasInt* info);
    void FC_GLOBAL(zpotrf, ZPOTRF)(const char* uplo, const BlasInt* n,
                                   dcomplex* a, const BlasInt* lda,
                                   BlasInt* info);                           

    // xSTEMR
    void FC_GLOBAL(sstemr,
                   SSTEMR)(const char* jobz, const char* range,
                           const BlasInt* n, float* d, float* e,
                           const float* vl, const float* vu, const BlasInt* il,
                           const BlasInt* iu, BlasInt* m, float* w, float* z,
                           const BlasInt* ldz, const BlasInt* nzc,
                           BlasInt* isuppz, BlasInt* tryrac, float* work,
                           const BlasInt* lwork, BlasInt* iwork,
                           const BlasInt* liwork, BlasInt* info);

    void FC_GLOBAL(dstemr,
                   DSTEMR)(const char* jobz, const char* range,
                           const BlasInt* n, double* d, double* e,
                           const double* vl, const double* vu,
                           const BlasInt* il, const BlasInt* iu, BlasInt* m,
                           double* w, double* z, const BlasInt* ldz,
                           const BlasInt* nzc, BlasInt* isuppz, BlasInt* tryrac,
                           double* work, const BlasInt* lwork, BlasInt* iwork,
                           const BlasInt* liwork, BlasInt* info);

    // xHEEVD / xSYEVD

    void FC_GLOBAL(ssyevd, SSYEVD)(const char* jobz, const char* uplo,
                                   const BlasInt* n, float* a,
                                   const BlasInt* lda, float* w, float* work,
                                   const BlasInt* lwork, BlasInt* iwork,
                                   const BlasInt* liwork, BlasInt* info);
    void FC_GLOBAL(dsyevd, DSYEVD)(const char* jobz, const char* uplo,
                                   const BlasInt* n, double* a,
                                   const BlasInt* lda, double* w, double* work,
                                   const BlasInt* lwork, BlasInt* iwork,
                                   const BlasInt* liwork, BlasInt* info);
    void FC_GLOBAL(cheevd, CHEEVD)(const char* jobz, const char* uplo,
                                   const BlasInt* n, scomplex* a,
                                   const BlasInt* lda, float* w, scomplex* work,
                                   const BlasInt* lwork, float* rwork,
                                   const BlasInt* lrwork, BlasInt* iwork,
                                   const BlasInt* liwork, BlasInt* info);
    void FC_GLOBAL(zheevd,
                   ZHEEVD)(const char* jobz, const char* uplo, const BlasInt* n,
                           dcomplex* a, const BlasInt* lda, double* w,
                           dcomplex* work, const BlasInt* lwork, double* rwork,
                           const BlasInt* lrwork, BlasInt* iwork,
                           const BlasInt* liwork, BlasInt* info);
    // xGEEV
    void FC_GLOBAL(sgeev,
                   SGEEV)(const char* jobz, const char* uplo, const BlasInt* n,
                           float* a, const BlasInt* lda, float* wr, float* wi,
			   float* vl, const BlasInt* ldvl, float* vr, const BlasInt* ldvr,
                           float* work, const BlasInt* lwork, BlasInt* info);
    void FC_GLOBAL(dgeev,
                   DGEEV)(const char* jobz, const char* uplo, const BlasInt* n,
                           double* a, const BlasInt* lda, double* wr, double* wi,
			   double* vl, const BlasInt* ldvl, double *vr, const BlasInt* ldvr,
                           double* work, const BlasInt* lwork, BlasInt* info);
    void FC_GLOBAL(cgeev,
                   CGEEV)(const char* jobz, const char* uplo, const BlasInt* n,
                           scomplex* a, const BlasInt* lda, scomplex* w,
			   scomplex *vl, const BlasInt* ldvl, scomplex *vr, const BlasInt* ldvr,
                           scomplex* work, const BlasInt* lwork, float* rwork, BlasInt* info);
    void FC_GLOBAL(zgeev,
                   ZGEEV)(const char* jobz, const char* uplo, const BlasInt* n,
                           dcomplex* a, const BlasInt* lda, dcomplex* w,
			   dcomplex *vl, const BlasInt* ldvl, dcomplex *vr, const BlasInt* ldvr,
                           dcomplex* work, const BlasInt* lwork, double* rwork, BlasInt* info);

    // xGESVD
    void FC_GLOBAL(sgesvd, SGESVD)(const char* jobu, const char* jobvt,
                                   const BlasInt* m, const BlasInt* n, float* A,
                                   const BlasInt* lda, float* S, float* U,
                                   const BlasInt* ldu, float* Vt,
                                   const BlasInt* ldvt, float* work,
                                   const BlasInt* lwork, float* rwork,
                                   BlasInt* info);
    void FC_GLOBAL(dgesvd, DGESVD)(const char* jobu, const char* jobvt,
                                   const BlasInt* m, const BlasInt* n,
                                   double* A, const BlasInt* lda, double* S,
                                   double* U, const BlasInt* ldu, double* Vt,
                                   const BlasInt* ldvt, double* work,
                                   const BlasInt* lwork, double* rwork,
                                   BlasInt* info);
    void FC_GLOBAL(cgesvd, CGESVD)(const char* jobu, const char* jobvt,
                                   const BlasInt* m, const BlasInt* n,
                                   scomplex* A, const BlasInt* lda, float* S,
                                   scomplex* U, const BlasInt* ldu,
                                   scomplex* Vt, const BlasInt* ldvt,
                                   scomplex* work, const BlasInt* lwork,
                                   float* rwork, BlasInt* info);
    void FC_GLOBAL(zgesvd, ZGESVD)(const char* jobu, const char* jobvt,
                                   const BlasInt* m, const BlasInt* n,
                                   dcomplex* A, const BlasInt* lda, double* S,
                                   dcomplex* U, const BlasInt* ldu,
                                   dcomplex* Vt, const BlasInt* ldvt,
                                   dcomplex* work, const BlasInt* lwork,
                                   double* rwork, BlasInt* info);

                           
                                                      
} // end of extern "C"
} //end of namespace lapackpp
} //end of namespace linalg   
} //end of namespace chase
