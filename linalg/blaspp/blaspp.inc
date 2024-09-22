#pragma once 

#include <assert.h>
#include <memory>
#include <vector>

#include "linalg/blaspp/blas_fortran.hpp"
#include "fortran_mangle.h"

namespace chase
{
namespace linalg
{
namespace blaspp
{

// xNRM2
template <>
double t_nrm2(const std::size_t n, const double* x, const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;

    return FC_GLOBAL(dnrm2, DNRM2)(&n_, x, &incx_);
}
template <>
float t_nrm2(const std::size_t n, const float* x, const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;

    return FC_GLOBAL(snrm2, SNRM2)(&n_, x, &incx_);
}
template <>
double t_nrm2(const std::size_t n, const std::complex<double>* x,
              const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;

    return FC_GLOBAL(dznrm2, DZNRM2)(&n_, x, &incx_);
}
template <>
float t_nrm2(const std::size_t n, const std::complex<float>* x,
             const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;

    return FC_GLOBAL(scnrm2, SCNRM2)(&n_, x, &incx_);
}

template <>
float t_dot(const std::size_t n, const float* x, const std::size_t incx,
            const float* y, const std::size_t incy)
{

    BlasInt n_ = n;
    BlasInt incy_ = incy;
    BlasInt incx_ = incx;

    return FC_GLOBAL(sdot, SDOT)(&n_, x, &incx_, y, &incy_);
}

template <>
double t_dot(const std::size_t n, const double* x, const std::size_t incx,
             const double* y, const std::size_t incy)
{

    BlasInt n_ = n;
    BlasInt incy_ = incy;
    BlasInt incx_ = incx;

    return FC_GLOBAL(ddot, DDOT)(&n_, x, &incx_, y, &incy_);
}

// Dot is not worth the trouble to link to a blas function
template <typename T>
std::complex<T> t_dot(const std::size_t n, const std::complex<T>* x,
                      const std::size_t incx, const std::complex<T>* y,
                      const std::size_t incy) {
  std::complex<T> gamma;
  std::complex<T> alpha = 0;
  for (BlasInt i = 0; i < n; ++i) {
    alpha += std::conj(x[i * incx]) * y[i * incy];
  }

  return alpha;
}

// xSCAL
template <>
void t_scal(const std::size_t n, const float* a, float* x,
            const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;

    FC_GLOBAL(sscal, SSCAL)(&n_, a, x, &incx_);
}
template <>
void t_scal(const std::size_t n, const double* a, double* x,
            const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;

    FC_GLOBAL(dscal, DSCAL)(&n_, a, x, &incx_);
}
template <>
void t_scal(const std::size_t n, const std::complex<float>* a,
            std::complex<float>* x, const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;
    FC_GLOBAL(cscal, CSCAL)(&n_, a, x, &incx_);
}
template <>
void t_scal(const std::size_t n, const std::complex<double>* a,
            std::complex<double>* x, const std::size_t incx)
{
    BlasInt n_ = n;
    BlasInt incx_ = incx;

    FC_GLOBAL(zscal, ZSCAL)(&n_, a, x, &incx_);
}

// xAXPY
template <>
void t_axpy(const std::size_t n, const float* a, const float* x,
            const std::size_t incx, float* y, const std::size_t incy)
{
    BlasInt n_ = n;
    BlasInt incy_ = incy;
    BlasInt incx_ = incx;

    FC_GLOBAL(saxpy, SAXPY)(&n_, a, x, &incx_, y, &incy_);
}
template <>
void t_axpy(const std::size_t n, const double* a, const double* x,
            const std::size_t incx, double* y, const std::size_t incy)
{
    BlasInt n_ = n;
    BlasInt incy_ = incy;
    BlasInt incx_ = incx;

    FC_GLOBAL(daxpy, DAXPY)(&n_, a, x, &incx_, y, &incy_);
}
template <>
void t_axpy(const std::size_t n, const std::complex<float>* a,
            const std::complex<float>* x, const std::size_t incx,
            std::complex<float>* y, const std::size_t incy)
{
    BlasInt n_ = n;
    BlasInt incy_ = incy;
    BlasInt incx_ = incx;

    FC_GLOBAL(caxpy, CAXPY)(&n_, a, x, &incx_, y, &incy_);
}
template <>
void t_axpy(const std::size_t n, const std::complex<double>* a,
            const std::complex<double>* x, const std::size_t incx,
            std::complex<double>* y, const std::size_t incy)
{
    BlasInt n_ = n;
    BlasInt incy_ = incy;
    BlasInt incx_ = incx;

    FC_GLOBAL(zaxpy, ZAXPY)(&n_, a, x, &incx_, y, &incy_);
}

template <>
void t_syherk(const char uplo, const char trans, const std::size_t n,
              const std::size_t k, const float* alpha, float* a,
              const std::size_t lda, const float* beta, float* c,
              const std::size_t ldc)
{
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(ssyrk, SSYRK)
    (&uplo, &trans, &n_, &k_, alpha, a, &lda_, beta, c, &ldc_);
}

template <>
void t_syherk(const char uplo, const char trans, const std::size_t n,
              const std::size_t k, const double* alpha, double* a,
              const std::size_t lda, const double* beta, double* c,
              const std::size_t ldc)
{
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(dsyrk, DSYRK)
    (&uplo, &trans, &n_, &k_, alpha, a, &lda_, beta, c, &ldc_);
}

template <>
void t_syherk(const char uplo, const char trans, const std::size_t n,
              const std::size_t k, const std::complex<float>* alpha,
              std::complex<float>* a, const std::size_t lda,
              const std::complex<float>* beta, std::complex<float>* c,
              const std::size_t ldc)
{
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(cherk, CHERK)
    (&uplo, &trans, &n_, &k_, alpha, a, &lda_, beta, c, &ldc_);
}

template <>
void t_syherk(const char uplo, const char trans, const std::size_t n,
              const std::size_t k, const std::complex<double>* alpha,
              std::complex<double>* a, const std::size_t lda,
              const std::complex<double>* beta, std::complex<double>* c,
              const std::size_t ldc)
{
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(zherk, ZHERK)
    (&uplo, &trans, &n_, &k_, alpha, a, &lda_, beta, c, &ldc_);
}

template <>
void t_trsm(const char side, const char uplo, const char trans, const char diag,
            const std::size_t m, const std::size_t n, const float* alpha,
            const float* a, const std::size_t lda, const float* b,
            const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;

    FC_GLOBAL(strsm, STRSM)
    (&side, &uplo, &trans, &diag, &m_, &n_, alpha, a, &lda_, b, &ldb_);
}

template <>
void t_trsm(const char side, const char uplo, const char trans, const char diag,
            const std::size_t m, const std::size_t n, const double* alpha,
            const double* a, const std::size_t lda, const double* b,
            const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;

    FC_GLOBAL(dtrsm, DTRSM)
    (&side, &uplo, &trans, &diag, &m_, &n_, alpha, a, &lda_, b, &ldb_);
}

template <>
void t_trsm(const char side, const char uplo, const char trans, const char diag,
            const std::size_t m, const std::size_t n,
            const std::complex<float>* alpha, const std::complex<float>* a,
            const std::size_t lda, const std::complex<float>* b,
            const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;

    FC_GLOBAL(ctrsm, CTRSM)
    (&side, &uplo, &trans, &diag, &m_, &n_, alpha, a, &lda_, b, &ldb_);
}

template <>
void t_trsm(const char side, const char uplo, const char trans, const char diag,
            const std::size_t m, const std::size_t n,
            const std::complex<double>* alpha, const std::complex<double>* a,
            const std::size_t lda, const std::complex<double>* b,
            const std::size_t ldb)
{
    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;

    FC_GLOBAL(ztrsm, ZTRSM)
    (&side, &uplo, &trans, &diag, &m_, &n_, alpha, a, &lda_, b, &ldb_);
}

template <>
void t_gemm<double>(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
                    const CBLAS_TRANSPOSE transb, const std::size_t m,
                    const std::size_t n, const std::size_t k,
                    const double* alpha, const double* a, const std::size_t lda,
                    const double* b, const std::size_t ldb, const double* beta,
                    double* c, const std::size_t ldc)
{
    // cblas_dgemm(Layout, transa, transb, m, n, k, *alpha, a, lda, b, ldb,
    // *beta,c, ldc);

    char TA, TB;

    if (transa == CblasNoTrans)
        TA = 'N';
    if (transa == CblasTrans)
        TA = 'T';
    if (transa == CblasConjTrans)
        TA = 'C';

    if (transb == CblasNoTrans)
        TB = 'N';
    if (transb == CblasTrans)
        TB = 'T';
    if (transb == CblasConjTrans)
        TB = 'C';

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(dgemm, DGEMM)
    (&TA, &TB, &m_, &n_, &k_, alpha, a, &lda_, b, &ldb_, beta, c, &ldc_);
}

template <>
void t_gemm<float>(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
                   const CBLAS_TRANSPOSE transb, const std::size_t m,
                   const std::size_t n, const std::size_t k, const float* alpha,
                   const float* a, const std::size_t lda, const float* b,
                   const std::size_t ldb, const float* beta, float* c,
                   const std::size_t ldc)
{
    char TA, TB;

    if (transa == CblasNoTrans)
        TA = 'N';
    if (transa == CblasTrans)
        TA = 'T';
    if (transa == CblasConjTrans)
        TA = 'C';

    if (transb == CblasNoTrans)
        TB = 'N';
    if (transb == CblasTrans)
        TB = 'T';
    if (transb == CblasConjTrans)
        TB = 'C';

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(sgemm, SGEMM)
    (&TA, &TB, &m_, &n_, &k_, alpha, a, &lda_, b, &ldb_, beta, c, &ldc_);
}

template <>
void t_gemm<std::complex<double>>(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const std::size_t m, const std::size_t n,
    const std::size_t k, const std::complex<double>* alpha,
    const std::complex<double>* a, const std::size_t lda,
    const std::complex<double>* b, const std::size_t ldb,
    const std::complex<double>* beta, std::complex<double>* c,
    const std::size_t ldc)
{
    char TA, TB;

    if (transa == CblasNoTrans)
        TA = 'N';
    if (transa == CblasTrans)
        TA = 'T';
    if (transa == CblasConjTrans)
        TA = 'C';

    if (transb == CblasNoTrans)
        TB = 'N';
    if (transb == CblasTrans)
        TB = 'T';
    if (transb == CblasConjTrans)
        TB = 'C';

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(zgemm, ZGEMM)
    (&TA, &TB, &m_, &n_, &k_, alpha, a, &lda_, b, &ldb_, beta, c, &ldc_);
}

template <>
void t_gemm<std::complex<float>>(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const std::size_t m, const std::size_t n,
    const std::size_t k, const std::complex<float>* alpha,
    const std::complex<float>* a, const std::size_t lda,
    const std::complex<float>* b, const std::size_t ldb,
    const std::complex<float>* beta, std::complex<float>* c,
    const std::size_t ldc)
{
    char TA, TB;

    if (transa == CblasNoTrans)
        TA = 'N';
    if (transa == CblasTrans)
        TA = 'T';
    if (transa == CblasConjTrans)
        TA = 'C';

    if (transb == CblasNoTrans)
        TB = 'N';
    if (transb == CblasTrans)
        TB = 'T';
    if (transb == CblasConjTrans)
        TB = 'C';

    BlasInt m_ = m;
    BlasInt n_ = n;
    BlasInt k_ = k;
    BlasInt lda_ = lda;
    BlasInt ldb_ = ldb;
    BlasInt ldc_ = ldc;

    FC_GLOBAL(cgemm, CGEMM)
    (&TA, &TB, &m_, &n_, &k_, alpha, a, &lda_, b, &ldb_, beta, c, &ldc_);
}


} //end of namespace blaspp
} //end of namespace linalg   
} //end of namespace chase

#include "blaspp.inc"
