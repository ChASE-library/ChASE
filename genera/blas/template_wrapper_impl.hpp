/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#pragma once
#include "blas.h"

// xSCAL
template <>
void t_scal(const std::size_t n, const float* a, float* x,
            const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;

  FC_GLOBAL(sscal, SSCAL)(&n_, a, x, &incx_);
}
template <>
void t_scal(const std::size_t n, const double* a, double* x,
            const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;

  FC_GLOBAL(dscal, DSCAL)(&n_, a, x, &incx_);
}
template <>
void t_scal(const std::size_t n, const std::complex<float>* a,
            std::complex<float>* x, const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;
  FC_GLOBAL(cscal, CSCAL)(&n_, a, x, &incx_);
}
template <>
void t_scal(const std::size_t n, const std::complex<double>* a,
            std::complex<double>* x, const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;

  FC_GLOBAL(zscal, ZSCAL)(&n_, a, x, &incx_);
}

// xAXPY
template <>
void t_axpy(const std::size_t n, const float* a, const float* x,
            const std::size_t incx, float* y, const std::size_t incy) {
  BlasInt n_ = n;
  BlasInt incy_ = incy;
  BlasInt incx_ = incx;

  FC_GLOBAL(saxpy, SAXPY)(&n_, a, x, &incy_, y, &incy_);
}
template <>
void t_axpy(const std::size_t n, const double* a, const double* x,
            const std::size_t incx, double* y, const std::size_t incy) {
  BlasInt n_ = n;
  BlasInt incy_ = incy;
  BlasInt incx_ = incx;

  FC_GLOBAL(daxpy, DAXPY)(&n_, a, x, &incy_, y, &incy_);
}
template <>
void t_axpy(const std::size_t n, const std::complex<float>* a,
            const std::complex<float>* x, const std::size_t incx,
            std::complex<float>* y, const std::size_t incy) {
  BlasInt n_ = n;
  BlasInt incy_ = incy;
  BlasInt incx_ = incx;

  FC_GLOBAL(caxpy, CAXPY)(&n_, a, x, &incy_, y, &incy_);
}
template <>
void t_axpy(const std::size_t n, const std::complex<double>* a,
            const std::complex<double>* x, const std::size_t incx,
            std::complex<double>* y, const std::size_t incy) {
  BlasInt n_ = n;
  BlasInt incy_ = incy;
  BlasInt incx_ = incx;

  FC_GLOBAL(zaxpy, ZAXPY)(&n_, a, x, &incy_, y, &incy_);
}

// xDOT
/* TODO
template <>
float t_dot(const std::size_t n, const float* x, const std::size_t incx,
           const float* y, const std::size_t incy) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;
  BlasInt incy_ = incy;

  *dot = FC_GLOBAL(sdot, SDOT)(&n_, x, &incx_, y, &incy_);
}
template <>
void t_dot(const std::size_t n, const double* x, const std::size_t incx,
           const double* y, const std::size_t incy, double* dot) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;
  BlasInt incy_ = incy;

  *dot = FC_GLOBAL(ddot, DDOT)(&n_, x, &incx_, y, &incy_);
}
*/
template <>
scomplex t_dot(const std::size_t n, const std::complex<float>* x,
               const std::size_t incx, const std::complex<float>* y,
               const std::size_t incy) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;
  BlasInt incy_ = incy;
  scomplex result;

#if defined(FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID)
  FC_GLOBAL(cdotc, CDOTC)(&result, &n_, x, &incx_, y, &incy_);
#else
  result = FC_GLOBAL(cdotc, CDOTC)(&n_, x, &incx_, y, &incy_);
#endif
  return result;
}
template <>
dcomplex t_dot(const std::size_t n, const std::complex<double>* x,
               const std::size_t incx, const std::complex<double>* y,
               const std::size_t incy) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;
  BlasInt incy_ = incy;
  dcomplex result;

#if defined(FORTRAN_COMPLEX_FUNCTIONS_RETURN_VOID)
  FC_GLOBAL(zdotc, ZDOTC)(&result, &n_, x, &incx_, y, &incy_);
#else
  result = FC_GLOBAL(zdotc, ZDOTC)(&n_, x, &incx_, y, &incy_);
#endif

  return result;
}

/*
template <>
double
t_lange<double>(char norm, std::size_t m, std::size_t n, double* A,
                std::size_t lda)
{
  //return LAPACKE_dlange(LAPACK_COL_MAJOR, norm, m, n, A, lda);

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt lda_ = lda;

  FC_GLOBAL(dlange,DLANGE)(&norm, &m_, &n_, A, &lda_, nullptr);

}

template <>
float
t_lange(char norm, std::size_t m, std::size_t n, float* A, std::size_t lda)
{
  //return LAPACKE_slange(LAPACK_COL_MAJOR, norm, m, n, A, lda);

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt lda_ = lda;

  FC_GLOBAL(slange,SLANGE)(&norm, &m_, &n_, A, &lda_, nullptr);

}
*/
template <>
double t_lange(char norm, std::size_t m, std::size_t n, std::complex<double>* A,
               std::size_t lda) {
  // return LAPACKE_zlange(LAPACK_COL_MAJOR, norm, m, n, A, lda);

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt lda_ = lda;

  return FC_GLOBAL(zlange, ZLANGE)(&norm, &m_, &n_, A, &lda_, nullptr);
}

template <>
float t_lange(char norm, std::size_t m, std::size_t n, std::complex<float>* A,
              std::size_t lda) {
  //  return LAPACKE_clange(LAPACK_COL_MAJOR, norm, m, n, A, lda);

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt lda_ = lda;

  return FC_GLOBAL(clange, CLANGE)(&norm, &m_, &n_, A, &lda_, nullptr);
}

// Overload of ?gemm functions
/*
template <>
void
t_gemm<double>(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
               const CBLAS_TRANSPOSE transb, const std::size_t m,
               const std::size_t n, const std::size_t k, const double* alpha,
               const double* a, const std::size_t lda, const double* b,
               const std::size_t ldb, const double* beta, double* c,
               const std::size_t ldc)
{
  // cblas_dgemm(Layout, transa, transb, m, n, k, *alpha, a, lda, b, ldb, *beta,
c,
  //             ldc);


 char TA, TB;

  if( transa == CblasNoTrans )
    TA = 'N';
  if( transa == CblasTrans )
    TA = 'T';
  if( transa == CblasConjTrans )
    TA = 'C';

  if( transb == CblasNoTrans )
    TB = 'N';
  if( transb == CblasTrans )
    TB = 'T';
  if( transb == CblasConjTrans )
    TB = 'C';

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt k_ = k;
  BlasInt lda_ = lda;
  BlasInt ldb_ = ldb;
  BlasInt ldc_ = ldc;

  FC_GLOBAL(dgemm,DGEMM)(&TA, &TB, &m_, &n_, &k_, alpha, a, &lda_, b, &ldb_,
beta, c,
        &ldc_);

}

template <>
void
t_gemm<float>(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
              const CBLAS_TRANSPOSE transb, const std::size_t m,
              const std::size_t n, const std::size_t k, const float* alpha,
              const float* a, const std::size_t lda, const float* b,
              const std::size_t ldb, const float* beta, float* c,
              const std::size_t ldc)
{
  // cblas_sgemm(Layout, transa, transb, m, n, k, *alpha, a, lda, b, ldb, *beta,
c,
  //             ldc);

 char TA, TB;

  if( transa == CblasNoTrans )
    TA = 'N';
  if( transa == CblasTrans )
    TA = 'T';
  if( transa == CblasConjTrans )
    TA = 'C';

  if( transb == CblasNoTrans )
    TB = 'N';
  if( transb == CblasTrans )
    TB = 'T';
  if( transb == CblasConjTrans )
    TB = 'C';

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt k_ = k;
  BlasInt lda_ = lda;
  BlasInt ldb_ = ldb;
  BlasInt ldc_ = ldc;

  FC_GLOBAL(sgemm,SGEMM)(&TA, &TB, &m_, &n_, &k_, alpha, a, &lda_, b, &ldb_,
beta, c,
        &ldc_);

}
*/
template <>
void t_gemm<std::complex<double>>(
    const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, const std::size_t m, const std::size_t n,
    const std::size_t k, const std::complex<double>* alpha,
    const std::complex<double>* a, const std::size_t lda,
    const std::complex<double>* b, const std::size_t ldb,
    const std::complex<double>* beta, std::complex<double>* c,
    const std::size_t ldc) {
  char TA, TB;

  if (transa == CblasNoTrans) TA = 'N';
  if (transa == CblasTrans) TA = 'T';
  if (transa == CblasConjTrans) TA = 'C';

  if (transb == CblasNoTrans) TB = 'N';
  if (transb == CblasTrans) TB = 'T';
  if (transb == CblasConjTrans) TB = 'C';

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
    const std::size_t ldc) {
  char TA, TB;

  if (transa == CblasNoTrans) TA = 'N';
  if (transa == CblasTrans) TA = 'T';
  if (transa == CblasConjTrans) TA = 'C';

  if (transb == CblasNoTrans) TB = 'N';
  if (transb == CblasTrans) TB = 'T';
  if (transb == CblasConjTrans) TB = 'C';

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt k_ = k;
  BlasInt lda_ = lda;
  BlasInt ldb_ = ldb;
  BlasInt ldc_ = ldc;

  FC_GLOBAL(cgemm, CGEMM)
  (&TA, &TB, &m_, &n_, &k_, alpha, a, &lda_, b, &ldb_, beta, c, &ldc_);
}

// xNRM2
template <>
double t_nrm2(const std::size_t n, const double* x, const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;

  return FC_GLOBAL(dnrm2, DNRM2)(&n_, x, &incx_);
}
template <>
float t_nrm2(const std::size_t n, const float* x, const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;

  return FC_GLOBAL(snrm2, SNRM2)(&n_, x, &incx_);
}
template <>
double t_nrm2(const std::size_t n, const std::complex<double>* x,
              const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;

  return FC_GLOBAL(dznrm2, DZNRM2)(&n_, x, &incx_);
}
template <>
float t_nrm2(const std::size_t n, const std::complex<float>* x,
             const std::size_t incx) {
  BlasInt n_ = n;
  BlasInt incx_ = incx;

  return FC_GLOBAL(scnrm2, SCNRM2)(&n_, x, &incx_);
}

// Overload of ?geqrf functions
/*
template <>
std::size_t
t_geqrf(int matrix_layout, std::size_t m, std::size_t n, double* a,
        std::size_t lda, double* tau)
{
//return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
  using T = std::remove_reference<decltype((a[0]))>::type;

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt lda_ = lda;

  T* work;
  T numwork;
  BlasInt lwork, info;

  lwork = -1;
  FC_GLOBAL(dgeqrf,DGEQRF)(&m_, &n_, a, &lda_, tau, &numwork, &lwork, &info);
  assert(info == 0);

  lwork = static_cast<std::size_t>((numwork));
  auto ptr = std::unique_ptr<T[]> {
    new T[ lwork ]
  };
  work = ptr.get();

  FC_GLOBAL(dgeqrf,DGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork, &info);
  assert(info == 0);

}
template <>
std::size_t
t_geqrf(int matrix_layout, std::size_t m, std::size_t n, float* a,
        std::size_t lda, float* tau)
{
  //return LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau);
  using T = std::remove_reference<decltype((a[0]))>::type;

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt lda_ = lda;

  T* work;
  T numwork;
  BlasInt lwork, info;

  lwork = -1;
  FC_GLOBAL(sgeqrf,SGEQRF)(&m_, &n_, a, &lda_, tau, &numwork, &lwork, &info);
  assert(info == 0);

  lwork = static_cast<std::size_t>((numwork));
  auto ptr = std::unique_ptr<T[]> {
    new T[ lwork ]
  };
  work = ptr.get();

  FC_GLOBAL(sgeqrf,SGEQRF)(&m_, &n_, a, &lda_, tau, work, &lwork, &info);
  assert(info == 0);

}
*/
template <>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n,
                    std::complex<double>* a, std::size_t lda,
                    std::complex<double>* tau) {
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
}

template <>
std::size_t t_geqrf(int matrix_layout, std::size_t m, std::size_t n,
                    std::complex<float>* a, std::size_t lda,
                    std::complex<float>* tau) {
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
}
/*
// Overload of ?gqr functions
template <>
std::size_t
t_gqr(int matrix_layout, std::size_t m, std::size_t n, std::size_t k, double* a,
      std::size_t lda, const double* tau)
{
  return LAPACKE_dorgqr(matrix_layout, m, n, k, a, lda, tau);
}
template <>
std::size_t
t_gqr(int matrix_layout, std::size_t m, std::size_t n, std::size_t k, float* a,
      std::size_t lda, const float* tau)
{
  return LAPACKE_sorgqr(matrix_layout, m, n, k, a, lda, tau);
}
*/
template <>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, std::complex<double>* a, std::size_t lda,
                  const std::complex<double>* tau) {
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
  auto ptr = std::unique_ptr<std::complex<double>[]> {
    new std::complex<double>[ lwork ]
  };
  work = ptr.get();

  FC_GLOBAL(zungqr, ZUNGQR)(&m_, &n_, &k_, a, &lda_, tau, work, &lwork, &info);
  assert(info == 0);
}
template <>
std::size_t t_gqr(int matrix_layout, std::size_t m, std::size_t n,
                  std::size_t k, std::complex<float>* a, std::size_t lda,
                  const std::complex<float>* tau) {
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
  auto ptr = std::unique_ptr<std::complex<float>[]> {
    new std::complex<float>[ lwork ]
  };
  work = ptr.get();

  FC_GLOBAL(cungqr, CUNGQR)(&m_, &n_, &k_, a, &lda_, tau, work, &lwork, &info);
  assert(info == 0);
}
/*
// Overload of ?heevd functions
template <>
std::size_t
t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n, double* a,
        std::size_t lda, double* w)
{
  return LAPACKE_dsyevd(matrix_layout, jobz, uplo, n, a, lda, w);
}
template <>
std::size_t
t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n, float* a,
        std::size_t lda, float* w)
{
  return LAPACKE_ssyevd(matrix_layout, jobz, uplo, n, a, lda, w);
}
*/
template <>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    std::complex<double>* a, std::size_t lda, double* w) {
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

  FC_GLOBAL(zheevd, zHEEVD)
  (&jobz, &uplo, &n_, a, &lda_, w, &numwork, &lwork, &rnumwork, &lrwork,
   &inumwork, &liwork, &info);
  assert(info == 0);

  lwork = static_cast<std::size_t>(real(numwork));
  auto ptr = std::unique_ptr<std::complex<double>[]> {
    new std::complex<double>[ lwork ]
  };
  work = ptr.get();

  lrwork = static_cast<std::size_t>(rnumwork);
  auto rptr = std::unique_ptr<double[]>{new double[lrwork]};
  rwork = rptr.get();

  liwork = static_cast<std::size_t>(inumwork);
  auto iptr = std::unique_ptr<BlasInt[]>{new BlasInt[liwork]};
  iwork = iptr.get();

  FC_GLOBAL(zheevd, ZHEEVD)
  (&jobz, &uplo, &n_, a, &lda_, w, work, &lwork, rwork, &lrwork, iwork, &liwork,
   &info);
  assert(info == 0);
}
template <>
std::size_t t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
                    std::complex<float>* a, std::size_t lda, float* w) {
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
  auto ptr = std::unique_ptr<std::complex<float>[]> {
    new std::complex<float>[ lwork ]
  };
  work = ptr.get();

  lrwork = static_cast<std::size_t>(rnumwork);
  auto rptr = std::unique_ptr<float[]>{new float[lrwork]};
  rwork = rptr.get();

  liwork = static_cast<std::size_t>(inumwork);
  auto iptr = std::unique_ptr<BlasInt[]>{new BlasInt[liwork]};
  iwork = iptr.get();

  FC_GLOBAL(cheevd, CHEEVD)
  (&jobz, &uplo, &n_, a, &lda_, w, work, &lwork, rwork, &lrwork, iwork, &liwork,
   &info);
  assert(info == 0);
}

// Overload of ?gemv functions
/*
template <>
void
t_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
       const std::size_t m, const std::size_t n, const double* alpha,
       const double* a, const std::size_t lda, const double* x,
       const std::size_t incx, const double* beta, double* y,
       const std::size_t incy)
{
  cblas_dgemv(Layout, trans, m, n, *alpha, a, lda, x, incx, *beta, y, incy);
}
template <>
void
t_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
       const std::size_t m, const std::size_t n, const float* alpha,
       const float* a, const std::size_t lda, const float* x,
       const std::size_t incx, const float* beta, float* y,
       const std::size_t incy)
{
  cblas_sgemv(Layout, trans, m, n, *alpha, a, lda, x, incx, *beta, y, incy);
}
template <>
void
t_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
       const std::size_t m, const std::size_t n,
       const std::complex<double>* alpha, const std::complex<double>* a,
       const std::size_t lda, const std::complex<double>* x,
       const std::size_t incx, const std::complex<double>* beta,
       std::complex<double>* y, const std::size_t incy)
{
  cblas_zgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}
*/
template <>
void t_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
            const std::size_t m, const std::size_t n,
            const std::complex<float>* alpha, const std::complex<float>* a,
            const std::size_t lda, const std::complex<float>* x,
            const std::size_t incx, const std::complex<float>* beta,
            std::complex<float>* y, const std::size_t incy) {
  // cblas_cgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);

  char TA;

  if (trans == CblasNoTrans) TA = 'N';
  if (trans == CblasTrans) TA = 'T';
  if (trans == CblasConjTrans) TA = 'C';

  BlasInt m_ = m;
  BlasInt n_ = n;
  BlasInt lda_ = lda;
  BlasInt incx_ = incx;
  BlasInt incy_ = incy;

  FC_GLOBAL(cgemv, CGEMV)
  (&TA, &m_, &n_, alpha, a, &lda_, x, &incx_, beta, y, &incy_);
}

// Overload of ?stemr functions

template <>
std::size_t t_stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    double* d, double* e, double vl, double vu, std::size_t il,
                    std::size_t iu, int* m, double* w, double* z,
                    std::size_t ldz, std::size_t nzc, int* isuppz,
                    lapack_logical* tryrac) {
  // return LAPACKE_dstemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu,
  // m,
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
  auto ptr = std::unique_ptr<double[]>{new double[lwork]};
  work = ptr.get();

  liwork = static_cast<std::size_t>(inumwork);
  auto iptr = std::unique_ptr<BlasInt[]>{new BlasInt[liwork]};
  iwork = iptr.get();

  FC_GLOBAL(dstemr, DSTEMR)
  (&jobz, &range, &n_, d, e, &vl, &vu, &il_, &iu_, m, w, z, &ldz_, &nzc_,
   isuppz, tryrac, work, &lwork, iwork, &liwork, &info);
}
template <>
std::size_t t_stemr(int matrix_layout, char jobz, char range, std::size_t n,
                    float* d, float* e, float vl, float vu, std::size_t il,
                    std::size_t iu, int* m, float* w, float* z, std::size_t ldz,
                    std::size_t nzc, int* isuppz, lapack_logical* tryrac) {
  // return LAPACKE_sstemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu,
  // m,
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
}
