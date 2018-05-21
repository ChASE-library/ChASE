/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2018, Simulation Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany
// and
// Copyright (c) 2016-2018, Aachen Institute for Advanced Study in Computational
//   Engineering Science, RWTH Aachen University, Germany All rights reserved.
// License is 3-clause BSD:
// https://github.com/SimLabQuantumMaterials/ChASE/
#pragma once

#include <fortran_mangle.h>
#include <complex>

namespace chase {
namespace mpi {
using BlasInt = int;
using dcomplex = std::complex<double>;
using scomplex = std::complex<float>;

extern "C" {

////////////
// BLAS 1 //
////////////

// xSCAL
void FC_GLOBAL(sscal, SSCAL)(const BlasInt* n, const float* alpha, float* x,
                             const BlasInt* incx);
void FC_GLOBAL(dscal, DSCAL)(const BlasInt* n, const double* alpha, double* x,
                             const BlasInt* incx);
void FC_GLOBAL(cscal, CSCAL)(const BlasInt* n, const scomplex* alpha,
                             scomplex* x, const BlasInt* incx);
void FC_GLOBAL(zscal, ZSCAL)(const BlasInt* n, const dcomplex* alpha,
                             dcomplex* x, const BlasInt* incx);

// xAXPY
void FC_GLOBAL(saxpy, SAXPY)(const BlasInt* n, const float* alpha,
                             const float* x, const BlasInt* incx, float* y,
                             const BlasInt* incy);
void FC_GLOBAL(daxpy, DAXPY)(const BlasInt* n, const double* alpha,
                             const double* x, const BlasInt* incx, double* y,
                             const BlasInt* incy);
void FC_GLOBAL(caxpy, CAXPY)(const BlasInt* n, const scomplex* alpha,
                             const scomplex* x, const BlasInt* incx,
                             scomplex* y, const BlasInt* incy);
void FC_GLOBAL(zaxpy, ZAXPY)(const BlasInt* n, const dcomplex* alpha,
                             const dcomplex* x, const BlasInt* incx,
                             dcomplex* y, const BlasInt* incy);

// xDOT
/*
void FC_GLOBAL(sdot, SDOT)(const BlasInt* n, const float* x,
                           const BlasInt* incx, const float* y,
                           const BlasInt* incy, float* dot);
void FC_GLOBAL(ddot, DDOT)(const BlasInt* n, const double* x,
                           const BlasInt* incx, const double* y,
                           const BlasInt* incy, double* dot);
*/
float FC_GLOBAL(sdot, SDOT)(const BlasInt* n, const float* x,
                            const BlasInt* incx, const float* y,
                            const BlasInt* incy);
double FC_GLOBAL(ddot, DDOT)(const BlasInt* n, const double* x,
                             const BlasInt* incx, const double* y,
                             const BlasInt* incy);
#if defined(FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID)
void FC_GLOBAL(cdotc, CDOTC)(scomplex* result, const BlasInt* n,
                             const scomplex* x, const BlasInt* incx,
                             const scomplex* y, const BlasInt* incy);
void FC_GLOBAL(zdotc, ZDOTC)(dcomplex* result, const BlasInt* n,
                             const dcomplex* x, const BlasInt* incx,
                             const dcomplex* y, const BlasInt* incy);

#else
scomplex FC_GLOBAL(cdotc, CDOTC)(const BlasInt* n, const scomplex* x,
                                 const BlasInt* incx, const scomplex* y,
                                 const BlasInt* incy);
dcomplex FC_GLOBAL(zdotc, ZDOTC)(const BlasInt* n, const dcomplex* x,
                                 const BlasInt* incx, const dcomplex* y,
                                 const BlasInt* incy);
#endif
// xNRM2
float FC_GLOBAL(snrm2, SNRM2)(const BlasInt* n, const float* x,
                              const BlasInt* incx);
double FC_GLOBAL(dnrm2, DNRM2)(const BlasInt* n, const double* x,
                               const BlasInt* incx);
float FC_GLOBAL(scnrm2, SCNRM2)(const BlasInt* n, const scomplex* x,
                                const BlasInt* incx);
double FC_GLOBAL(dznrm2, DZNRM2)(const BlasInt* n, const dcomplex* x,
                                 const BlasInt* incx);

////////////
// BLAS 2 //
////////////

// xGEMV
void FC_GLOBAL(sgemv, SGEMV)(const char* trans, const BlasInt* m,
                             const BlasInt* n, const float* alpha,
                             const float* A, const BlasInt* ALDim,
                             const float* x, const BlasInt* incx,
                             const float* beta, float* y, const BlasInt* incy);
void FC_GLOBAL(dgemv, DGEMV)(const char* trans, const BlasInt* m,
                             const BlasInt* n, const double* alpha,
                             const double* A, const BlasInt* ALDim,
                             const double* x, const BlasInt* incx,
                             const double* beta, double* y,
                             const BlasInt* incy);
void FC_GLOBAL(cgemv, CGEMV)(const char* trans, const BlasInt* m,
                             const BlasInt* n, const scomplex* alpha,
                             const scomplex* A, const BlasInt* ALDim,
                             const scomplex* x, const BlasInt* incx,
                             const scomplex* beta, scomplex* y,
                             const BlasInt* incy);
void FC_GLOBAL(zgemv, ZGEMV)(const char* trans, const BlasInt* m,
                             const BlasInt* n, const dcomplex* alpha,
                             const dcomplex* A, const BlasInt* ALDim,
                             const dcomplex* x, const BlasInt* incx,
                             const dcomplex* beta, dcomplex* y,
                             const BlasInt* incy);

////////////
// Blas 3 //
////////////

// xGEMM
void FC_GLOBAL(sgemm, SGEMM)(const char* transA, const char* transB,
                             const BlasInt* m, const BlasInt* n,
                             const BlasInt* k, const float* alpha,
                             const float* A, const BlasInt* ALDim,
                             const float* B, const BlasInt* BLDim,
                             const float* beta, float* C, const BlasInt* CLDim);
void FC_GLOBAL(dgemm,
               DGEMM)(const char* transA, const char* transB, const BlasInt* m,
                      const BlasInt* n, const BlasInt* k, const double* alpha,
                      const double* A, const BlasInt* ALDim, const double* B,
                      const BlasInt* BLDim, const double* beta, double* C,
                      const BlasInt* CLDim);
void FC_GLOBAL(cgemm,
               CGEMM)(const char* transA, const char* transB, const BlasInt* m,
                      const BlasInt* n, const BlasInt* k, const scomplex* alpha,
                      const scomplex* A, const BlasInt* ALDim,
                      const scomplex* B, const BlasInt* BLDim,
                      const scomplex* beta, scomplex* C, const BlasInt* CLDim);
void FC_GLOBAL(zgemm,
               ZGEMM)(const char* transA, const char* transB, const BlasInt* m,
                      const BlasInt* n, const BlasInt* k, const dcomplex* alpha,
                      const dcomplex* A, const BlasInt* ALDim,
                      const dcomplex* B, const BlasInt* BLDim,
                      const dcomplex* beta, dcomplex* C, const BlasInt* CLDim);

// TODO
void FC_GLOBAL(zhemm, ZHEMM)(const char* side, const char* uplo,
                             const BlasInt* m, const BlasInt* n,
                             const dcomplex* alpha, const dcomplex* a,
                             const BlasInt* lda, const dcomplex* b,
                             const BlasInt* ldb, const dcomplex* beta,
                             dcomplex* c, const BlasInt* ldc);
////////////
// LAPACK //
////////////

// xGEQRF
void FC_GLOBAL(sgeqrf, SGEQRF)(const BlasInt* m, const BlasInt* n, float* a,
                               const BlasInt* lda, float* tau, float* work,
                               const BlasInt* lwork, BlasInt* info);
void FC_GLOBAL(dgeqrf, DGEQRF)(const BlasInt* m, const BlasInt* n, double* a,
                               const BlasInt* lda, double* tau, double* work,
                               const BlasInt* lwork, BlasInt* info);
void FC_GLOBAL(cgeqrf, CGEQRF)(const BlasInt* m, const BlasInt* n, scomplex* a,
                               const BlasInt* lda, scomplex* tau,
                               scomplex* work, const BlasInt* lwork,
                               BlasInt* info);
void FC_GLOBAL(zgeqrf, ZGEQRF)(const BlasInt* m, const BlasInt* n, dcomplex* a,
                               const BlasInt* lda, dcomplex* tau,
                               dcomplex* work, const BlasInt* lwork,
                               BlasInt* info);

// xUNGQR / xGEQRF
void FC_GLOBAL(sorgqr, SORGQR)(const BlasInt* m, const BlasInt* n,
                               const BlasInt* k, float* a, const BlasInt* lda,
                               const float* tau, float* work,
                               const BlasInt* lwork, BlasInt* info);
void FC_GLOBAL(dorgqr, DORGQR)(const BlasInt* m, const BlasInt* n,
                               const BlasInt* k, double* a, const BlasInt* lda,
                               const double* tau, double* work,
                               const BlasInt* lwork, BlasInt* info);
void FC_GLOBAL(cungqr,
               CUNGQR)(const BlasInt* m, const BlasInt* n, const BlasInt* k,
                       scomplex* a, const BlasInt* lda, const scomplex* tau,
                       scomplex* work, const BlasInt* lwork, BlasInt* info);
void FC_GLOBAL(zungqr,
               ZUNGQR)(const BlasInt* m, const BlasInt* n, const BlasInt* k,
                       dcomplex* a, const BlasInt* lda, const dcomplex* tau,
                       dcomplex* work, const BlasInt* lwork, BlasInt* info);

// xSTEMR
void FC_GLOBAL(sstemr,
               SSTEMR)(const char* jobz, const char* range, const BlasInt* n,
                       float* d, float* e, const float* vl, const float* vu,
                       const BlasInt* il, const BlasInt* iu, BlasInt* m,
                       float* w, float* z, const BlasInt* ldz,
                       const BlasInt* nzc, BlasInt* isuppz, BlasInt* tryrac,
                       float* work, const BlasInt* lwork, BlasInt* iwork,
                       const BlasInt* liwork, BlasInt* info);

void FC_GLOBAL(dstemr,
               DSTEMR)(const char* jobz, const char* range, const BlasInt* n,
                       double* d, double* e, const double* vl, const double* vu,
                       const BlasInt* il, const BlasInt* iu, BlasInt* m,
                       double* w, double* z, const BlasInt* ldz,
                       const BlasInt* nzc, BlasInt* isuppz, BlasInt* tryrac,
                       double* work, const BlasInt* lwork, BlasInt* iwork,
                       const BlasInt* liwork, BlasInt* info);

// xHEEVD / xSYEVD

void FC_GLOBAL(ssyevd, SSYEVD)(const char* jobz, const char* uplo,
                               const BlasInt* n, float* a, const BlasInt* lda,
                               float* w, float* work, const BlasInt* lwork,
                               BlasInt* iwork, const BlasInt* liwork,
                               BlasInt* info);
void FC_GLOBAL(dsyevd, DSYEVD)(const char* jobz, const char* uplo,
                               const BlasInt* n, double* a, const BlasInt* lda,
                               double* w, double* work, const BlasInt* lwork,
                               BlasInt* iwork, const BlasInt* liwork,
                               BlasInt* info);
void FC_GLOBAL(cheevd, CHEEVD)(const char* jobz, const char* uplo,
                               const BlasInt* n, scomplex* a,
                               const BlasInt* lda, float* w, scomplex* work,
                               const BlasInt* lwork, float* rwork,
                               const BlasInt* lrwork, BlasInt* iwork,
                               const BlasInt* liwork, BlasInt* info);
void FC_GLOBAL(zheevd, ZHEEVD)(const char* jobz, const char* uplo,
                               const BlasInt* n, dcomplex* a,
                               const BlasInt* lda, double* w, dcomplex* work,
                               const BlasInt* lwork, double* rwork,
                               const BlasInt* lrwork, BlasInt* iwork,
                               const BlasInt* liwork, BlasInt* info);

// xLANGE
float FC_GLOBAL(slange, SLANGE)(const char* norm, const BlasInt* m,
                                const BlasInt* n, const float* a,
                                const BlasInt* lda, float* work);
float FC_GLOBAL(dlange, DLANGE)(const char* norm, const BlasInt* m,
                                const BlasInt* n, const double* a,
                                const BlasInt* lda, double* work);
float FC_GLOBAL(clange, CLANGE)(const char* norm, const BlasInt* m,
                                const BlasInt* n, const scomplex* a,
                                const BlasInt* lda, float* work);
double FC_GLOBAL(zlange, ZLANGE)(const char* norm, const BlasInt* m,
                                 const BlasInt* n, const dcomplex* a,
                                 const BlasInt* lda, double* work);

}  // extern "C"
}  // namespace mpi
}  // namespace chase
