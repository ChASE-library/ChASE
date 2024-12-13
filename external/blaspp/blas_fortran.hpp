// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <fortran_mangle.h>

namespace chase
{
namespace linalg
{
using BlasInt = int;
using dcomplex = std::complex<double>;
using scomplex = std::complex<float>;

struct complex
{
    float r, i;
}; // c complex
struct zcomplex
{
    double r, i;
}; // c complex

namespace blaspp
{
extern "C"
{
    // xNRM2
    float FC_GLOBAL(snrm2, SNRM2)(const BlasInt* n, const float* x,
                                  const BlasInt* incx);
    double FC_GLOBAL(dnrm2, DNRM2)(const BlasInt* n, const double* x,
                                   const BlasInt* incx);
    float FC_GLOBAL(scnrm2, SCNRM2)(const BlasInt* n, const scomplex* x,
                                    const BlasInt* incx);
    double FC_GLOBAL(dznrm2, DZNRM2)(const BlasInt* n, const dcomplex* x,
                                     const BlasInt* incx);

    // xDOT
    float FC_GLOBAL(sdot, SDOT)(const BlasInt* n, const float* x,
                                const BlasInt* incx, const float* y,
                                const BlasInt* incy);

    double FC_GLOBAL(ddot, DDOT)(const BlasInt* n, const double* x,
                                 const BlasInt* incx, const double* y,
                                 const BlasInt* incy);

    complex FC_GLOBAL(cdotc, CDOTC)(const BlasInt* n, const scomplex* x,
                                    const BlasInt* incx, const scomplex* y,
                                    const BlasInt* incy);

    zcomplex FC_GLOBAL(zdotc, ZDOTC)(const BlasInt* n, const dcomplex* x,
                                     const BlasInt* incx, const dcomplex* y,
                                     const BlasInt* incy);
                                
    // xSCAL
    void FC_GLOBAL(sscal, SSCAL)(const BlasInt* n, const float* alpha, float* x,
                                 const BlasInt* incx);
    void FC_GLOBAL(dscal, DSCAL)(const BlasInt* n, const double* alpha,
                                 double* x, const BlasInt* incx);
    void FC_GLOBAL(cscal, CSCAL)(const BlasInt* n, const scomplex* alpha,
                                 scomplex* x, const BlasInt* incx);
    void FC_GLOBAL(zscal, ZSCAL)(const BlasInt* n, const dcomplex* alpha,
                                 dcomplex* x, const BlasInt* incx);

    // xAXPY
    void FC_GLOBAL(saxpy, SAXPY)(const BlasInt* n, const float* alpha,
                                 const float* x, const BlasInt* incx, float* y,
                                 const BlasInt* incy);
    void FC_GLOBAL(daxpy, DAXPY)(const BlasInt* n, const double* alpha,
                                 const double* x, const BlasInt* incx,
                                 double* y, const BlasInt* incy);
    void FC_GLOBAL(caxpy, CAXPY)(const BlasInt* n, const scomplex* alpha,
                                 const scomplex* x, const BlasInt* incx,
                                 scomplex* y, const BlasInt* incy);
    void FC_GLOBAL(zaxpy, ZAXPY)(const BlasInt* n, const dcomplex* alpha,
                                 const dcomplex* x, const BlasInt* incx,
                                 dcomplex* y, const BlasInt* incy);

    // xSYRK + xHERK
    void FC_GLOBAL(ssyrk, SSYRK)(const char* uplo, const char* transB,
                                 const BlasInt* n, const BlasInt* k,
                                 const float* alpha, float* A,
                                 const BlasInt* lda, const float* beta,
                                 float* C, const BlasInt* ldc);
    void FC_GLOBAL(dsyrk, DSYRK)(const char* uplo, const char* transB,
                                 const BlasInt* n, const BlasInt* k,
                                 const double* alpha, double* A,
                                 const BlasInt* lda, const double* beta,
                                 double* C, const BlasInt* ldc);
    void FC_GLOBAL(cherk, CHERK)(const char* uplo, const char* transB,
                                 const BlasInt* n, const BlasInt* k,
                                 const scomplex* alpha, scomplex* A,
                                 const BlasInt* lda, const scomplex* beta,
                                 scomplex* C, const BlasInt* ldc);
    void FC_GLOBAL(zherk, ZHERK)(const char* uplo, const char* transB,
                                 const BlasInt* n, const BlasInt* k,
                                 const dcomplex* alpha, dcomplex* A,
                                 const BlasInt* lda, const dcomplex* beta,
                                 dcomplex* C, const BlasInt* ldc);

    // xTRSM
    void FC_GLOBAL(strsm, STRSM)(const char* side, const char* uplo,
                                 const char* trans, const char* diag,
                                 const BlasInt* m, const BlasInt* n,
                                 const float* alpha, const float* a,
                                 const BlasInt* lda, const float* b,
                                 const BlasInt* ldb);
    void FC_GLOBAL(dtrsm, DTRSM)(const char* side, const char* uplo,
                                 const char* trans, const char* diag,
                                 const BlasInt* m, const BlasInt* n,
                                 const double* alpha, const double* a,
                                 const BlasInt* lda, const double* b,
                                 const BlasInt* ldb);
    void FC_GLOBAL(ctrsm, CTRSM)(const char* side, const char* uplo,
                                 const char* trans, const char* diag,
                                 const BlasInt* m, const BlasInt* n,
                                 const scomplex* alpha, const scomplex* a,
                                 const BlasInt* lda, const scomplex* b,
                                 const BlasInt* ldb);
    void FC_GLOBAL(ztrsm, ZTRSM)(const char* side, const char* uplo,
                                 const char* trans, const char* diag,
                                 const BlasInt* m, const BlasInt* n,
                                 const dcomplex* alpha, const dcomplex* a,
                                 const BlasInt* lda, const dcomplex* b,
                                 const BlasInt* ldb);

    // xGEMM
    void FC_GLOBAL(sgemm, SGEMM)(const char* transA, const char* transB,
                                 const BlasInt* m, const BlasInt* n,
                                 const BlasInt* k, const float* alpha,
                                 const float* A, const BlasInt* ALDim,
                                 const float* B, const BlasInt* BLDim,
                                 const float* beta, float* C,
                                 const BlasInt* CLDim);
    void FC_GLOBAL(dgemm, DGEMM)(const char* transA, const char* transB,
                                 const BlasInt* m, const BlasInt* n,
                                 const BlasInt* k, const double* alpha,
                                 const double* A, const BlasInt* ALDim,
                                 const double* B, const BlasInt* BLDim,
                                 const double* beta, double* C,
                                 const BlasInt* CLDim);
    void FC_GLOBAL(cgemm, CGEMM)(const char* transA, const char* transB,
                                 const BlasInt* m, const BlasInt* n,
                                 const BlasInt* k, const scomplex* alpha,
                                 const scomplex* A, const BlasInt* ALDim,
                                 const scomplex* B, const BlasInt* BLDim,
                                 const scomplex* beta, scomplex* C,
                                 const BlasInt* CLDim);
    void FC_GLOBAL(zgemm, ZGEMM)(const char* transA, const char* transB,
                                 const BlasInt* m, const BlasInt* n,
                                 const BlasInt* k, const dcomplex* alpha,
                                 const dcomplex* A, const BlasInt* ALDim,
                                 const dcomplex* B, const BlasInt* BLDim,
                                 const dcomplex* beta, dcomplex* C,
                                 const BlasInt* CLDim);                                 

} // end of extern "C"
} //end of namespace blaspp
} //end of namespace linalg   
} //end of namespace chase