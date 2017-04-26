/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
#pragma once

template <>
double
t_lange<double>(char norm, std::size_t m, std::size_t n, double* A,
                std::size_t lda)
{
  return LAPACKE_dlange(LAPACK_COL_MAJOR, norm, m, n, A, lda);
}

template <>
float
t_lange(char norm, std::size_t m, std::size_t n, float* A, std::size_t lda)
{
  return LAPACKE_slange(LAPACK_COL_MAJOR, norm, m, n, A, lda);
}
template <>
double
t_lange(char norm, std::size_t m, std::size_t n, std::complex<double>* A,
        std::size_t lda)
{
  return LAPACKE_zlange(LAPACK_COL_MAJOR, norm, m, n, A, lda);
}

template <>
float
t_lange(char norm, std::size_t m, std::size_t n, std::complex<float>* A,
        std::size_t lda)
{
  return LAPACKE_clange(LAPACK_COL_MAJOR, norm, m, n, A, lda);
}

// Overload of ?gemm functions
template <>
void
t_gemm<double>(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
               const CBLAS_TRANSPOSE transb, const std::size_t m,
               const std::size_t n, const std::size_t k, const double* alpha,
               const double* a, const std::size_t lda, const double* b,
               const std::size_t ldb, const double* beta, double* c,
               const std::size_t ldc)
{
  cblas_dgemm(Layout, transa, transb, m, n, k, *alpha, a, lda, b, ldb, *beta, c,
              ldc);
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
  cblas_sgemm(Layout, transa, transb, m, n, k, *alpha, a, lda, b, ldb, *beta, c,
              ldc);
}
template <>
void
t_gemm<std::complex<double>>(
  const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa,
  const CBLAS_TRANSPOSE transb, const std::size_t m, const std::size_t n,
  const std::size_t k, const std::complex<double>* alpha,
  const std::complex<double>* a, const std::size_t lda,
  const std::complex<double>* b, const std::size_t ldb,
  const std::complex<double>* beta, std::complex<double>* c,
  const std::size_t ldc)
{
  cblas_zgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
              ldc);
}

template <>
void
t_gemm<std::complex<float>>(const CBLAS_LAYOUT Layout,
                            const CBLAS_TRANSPOSE transa,
                            const CBLAS_TRANSPOSE transb, const std::size_t m,
                            const std::size_t n, const std::size_t k,
                            const std::complex<float>* alpha,
                            const std::complex<float>* a, const std::size_t lda,
                            const std::complex<float>* b, const std::size_t ldb,
                            const std::complex<float>* beta,
                            std::complex<float>* c, const std::size_t ldc)
{
  cblas_cgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c,
              ldc);
}

template <>
void
t_hemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo,
       const std::size_t m, const std::size_t n, const float* alpha,
       const float* a, const std::size_t lda, const float* b,
       const std::size_t ldb, const float* beta, float* c,
       const std::size_t ldc)
{
  cblas_ssymm(Layout, side, uplo, m, n, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
template <>
void
t_hemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo,
       const std::size_t m, const std::size_t n, const double* alpha,
       const double* a, const std::size_t lda, const double* b,
       const std::size_t ldb, const double* beta, double* c,
       const std::size_t ldc)
{
  cblas_dsymm(Layout, side, uplo, m, n, *alpha, a, lda, b, ldb, *beta, c, ldc);
}
template <>
void
t_hemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo,
       const std::size_t m, const std::size_t n,
       const std::complex<float>* alpha, const std::complex<float>* a,
       const std::size_t lda, const std::complex<float>* b,
       const std::size_t ldb, const std::complex<float>* beta,
       std::complex<float>* c, const std::size_t ldc)
{
  cblas_chemm(Layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}
template <>
void
t_hemm(const CBLAS_LAYOUT Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo,
       const std::size_t m, const std::size_t n,
       const std::complex<double>* alpha, const std::complex<double>* a,
       const std::size_t lda, const std::complex<double>* b,
       const std::size_t ldb, const std::complex<double>* beta,
       std::complex<double>* c, const std::size_t ldc)
{
  cblas_zhemm(Layout, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

// Overload of ?axpy functions
template <>
void
t_axpy(const std::size_t n, const double* a, const double* x,
       const std::size_t incx, double* y, const std::size_t incy)
{
  cblas_daxpy(n, *a, x, incx, y, incy);
}
template <>
void
t_axpy(const std::size_t n, const float* a, const float* x,
       const std::size_t incx, float* y, const std::size_t incy)
{
  cblas_saxpy(n, *a, x, incx, y, incy);
}
template <>
void
t_axpy(const std::size_t n, const std::complex<double>* a,
       const std::complex<double>* x, const std::size_t incx,
       std::complex<double>* y, const std::size_t incy)
{
  cblas_zaxpy(n, a, x, incx, y, incy);
}
template <>
void
t_axpy(const std::size_t n, const std::complex<float>* a,
       const std::complex<float>* x, const std::size_t incx,
       std::complex<float>* y, const std::size_t incy)
{
  cblas_caxpy(n, a, x, incy, y, incy);
}

// Overload of ?nrm2 functions
template <>
double
t_nrm2(const std::size_t n, const double* x, const std::size_t incx)
{
  return cblas_dnrm2(n, x, incx);
}
template <>
float
t_nrm2(const std::size_t n, const float* x, const std::size_t incx)
{
  return cblas_snrm2(n, x, incx);
}
template <>
double
t_nrm2(const std::size_t n, const std::complex<double>* x,
       const std::size_t incx)
{
  return cblas_dznrm2(n, x, incx);
}
template <>
float
t_nrm2(const std::size_t n, const std::complex<float>* x,
       const std::size_t incx)
{
  return cblas_scnrm2(n, x, incx);
}

// Overload of ?geqrf functions
template <>
std::size_t
t_geqrf(int matrix_layout, std::size_t m, std::size_t n, double* a,
        std::size_t lda, double* tau)
{
  return LAPACKE_dgeqrf(matrix_layout, m, n, a, lda, tau);
}
template <>
std::size_t
t_geqrf(int matrix_layout, std::size_t m, std::size_t n, float* a,
        std::size_t lda, float* tau)
{
  return LAPACKE_sgeqrf(matrix_layout, m, n, a, lda, tau);
}
template <>
std::size_t
t_geqrf(int matrix_layout, std::size_t m, std::size_t n,
        std::complex<double>* a, std::size_t lda, std::complex<double>* tau)
{
  return LAPACKE_zgeqrf(matrix_layout, m, n, a, lda, tau);
}
template <>
std::size_t
t_geqrf(int matrix_layout, std::size_t m, std::size_t n, std::complex<float>* a,
        std::size_t lda, std::complex<float>* tau)
{
  return LAPACKE_cgeqrf(matrix_layout, m, n, a, lda, tau);
}

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
template <>
std::size_t
t_gqr(int matrix_layout, std::size_t m, std::size_t n, std::size_t k,
      std::complex<double>* a, std::size_t lda, const std::complex<double>* tau)
{
  return LAPACKE_zungqr(matrix_layout, m, n, k, a, lda, tau);
}
template <>
std::size_t
t_gqr(int matrix_layout, std::size_t m, std::size_t n, std::size_t k,
      std::complex<float>* a, std::size_t lda, const std::complex<float>* tau)
{
  return LAPACKE_cungqr(matrix_layout, m, n, k, a, lda, tau);
}

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
template <>
std::size_t
t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
        std::complex<double>* a, std::size_t lda, double* w)
{
  return LAPACKE_zheevd(matrix_layout, jobz, uplo, n, a, lda, w);
}
template <>
std::size_t
t_heevd(int matrix_layout, char jobz, char uplo, std::size_t n,
        std::complex<float>* a, std::size_t lda, float* w)
{
  return LAPACKE_cheevd(matrix_layout, jobz, uplo, n, a, lda, w);
}

// Overload of ?scal functions
template <>
void
t_scal(const std::size_t n, const double* a, double* x, const std::size_t incx)
{
  cblas_dscal(n, *a, x, incx);
}
template <>
void
t_scal(const std::size_t n, const float* a, float* x, const std::size_t incx)
{
  cblas_sscal(n, *a, x, incx);
}
template <>
void
t_scal(const std::size_t n, const std::complex<double>* a,
       std::complex<double>* x, const std::size_t incx)
{
  cblas_zscal(n, a, x, incx);
}
template <>
void
t_scal(const std::size_t n, const std::complex<float>* a,
       std::complex<float>* x, const std::size_t incx)
{
  cblas_cscal(n, a, x, incx);
}

// Overload of ?gemv functions
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
template <>
void
t_gemv(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans,
       const std::size_t m, const std::size_t n,
       const std::complex<float>* alpha, const std::complex<float>* a,
       const std::size_t lda, const std::complex<float>* x,
       const std::size_t incx, const std::complex<float>* beta,
       std::complex<float>* y, const std::size_t incy)
{
  cblas_cgemv(Layout, trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

// Overload of ?stemr functions
template <>
std::size_t
t_stemr(int matrix_layout, char jobz, char range, std::size_t n, double* d,
        double* e, double vl, double vu, std::size_t il, std::size_t iu, int* m,
        double* w, double* z, std::size_t ldz, std::size_t nzc, int* isuppz,
        lapack_logical* tryrac)
{
  return LAPACKE_dstemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m,
                        w, z, ldz, nzc, isuppz, tryrac);
}
template <>
std::size_t
t_stemr(int matrix_layout, char jobz, char range, std::size_t n, float* d,
        float* e, float vl, float vu, std::size_t il, std::size_t iu, int* m,
        float* w, float* z, std::size_t ldz, std::size_t nzc, int* isuppz,
        lapack_logical* tryrac)
{
  return LAPACKE_sstemr(matrix_layout, jobz, range, n, d, e, vl, vu, il, iu, m,
                        w, z, ldz, nzc, isuppz, tryrac);
}

// Overload of ?dot functions
template <>
void
t_dot(const std::size_t n, const double* x, const std::size_t incx,
      const double* y, const std::size_t incy, double* dotc)
{
  *dotc = cblas_ddot(n, x, incx, y, incy);
}
template <>
void
t_dot(const std::size_t n, const float* x, const std::size_t incx,
      const float* y, const std::size_t incy, float* dotc)
{
  *dotc = cblas_sdot(n, x, incx, y, incy);
}
template <>
void
t_dot(const std::size_t n, const std::complex<double>* x,
      const std::size_t incx, const std::complex<double>* y,
      const std::size_t incy, std::complex<double>* dotc)
{
  cblas_zdotc_sub(n, x, incx, y, incy, dotc);
}
template <>
void
t_dot(const std::size_t n, const std::complex<float>* x, const std::size_t incx,
      const std::complex<float>* y, const std::size_t incy,
      std::complex<float>* dotc)
{
  cblas_cdotc_sub(n, x, incx, y, incy, dotc);
}
