/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2023, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           int m, int n,
                           const float* alpha, const float* A, int lda,
                           const float* x, int incx, const float* beta, float* y,
                           int incy)
{
    return cublasSgemv(handle, transa, m, n, alpha, A, lda, x, incx,
                       beta, y, incy);
}

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           int m, int n,
                           const double* alpha, const double* A, int lda,
                           const double* x, int incx, const double* beta,
                           double* y, int incy)
{
    return cublasDgemv(handle, transa, m, n, alpha, A, lda, x, incx,
                       beta, y, incy);
}

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           int m, int n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, int lda,
                           const std::complex<float>* x, int incx,
                           const std::complex<float>* beta,
                           std::complex<float>* y, int incy)
{
    return cublasCgemv(handle, transa, m, n,
                       reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(A), lda,
                       reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<const cuComplex*>(beta),
                       reinterpret_cast<cuComplex*>(y), incy);
}

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           int m, int n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, int lda,
                           const std::complex<double>* x, int incx,
                           const std::complex<double>* beta,
                           std::complex<double>* y, int incy)
{
    return cublasZgemv(handle, transa, m, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(A), lda,
                       reinterpret_cast<const cuDoubleComplex*>(x), incx,
                       reinterpret_cast<const cuDoubleComplex*>(beta),
                       reinterpret_cast<cuDoubleComplex*>(y), incy);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* B, int ldb, const float* beta, float* C,
                           int ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double* alpha, const double* A, int lda,
                           const double* B, int ldb, const double* beta,
                           double* C, int ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, int lda,
                           const std::complex<float>* B, int ldb,
                           const std::complex<float>* beta,
                           std::complex<float>* C, int ldc)
{
    return cublasCgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(A), lda,
                       reinterpret_cast<const cuComplex*>(B), ldb,
                       reinterpret_cast<const cuComplex*>(beta),
                       reinterpret_cast<cuComplex*>(C), ldc);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, int lda,
                           const std::complex<double>* B, int ldb,
                           const std::complex<double>* beta,
                           std::complex<double>* C, int ldc)
{
    return cublasZgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(A), lda,
                       reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                       reinterpret_cast<const cuDoubleComplex*>(beta),
                       reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, float* A, int lda,
                                             int* Lwork)
{

    return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, double* A, int lda,
                                             int* Lwork)
{

    return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, std::complex<float>* A,
                                             int lda, int* Lwork)
{

    return cusolverDnCgeqrf_bufferSize(
        handle, m, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, int m,
                                             int n, std::complex<double>* A,
                                             int lda, int* Lwork)
{

    return cusolverDnZgeqrf_bufferSize(
        handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  float* A, int lda, float* TAU,
                                  float* Workspace, int Lwork, int* devInfo)
{
    return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  double* A, int lda, double* TAU,
                                  double* Workspace, int Lwork, int* devInfo)
{
    return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  std::complex<float>* A, int lda,
                                  std::complex<float>* TAU,
                                  std::complex<float>* Workspace, int Lwork,
                                  int* devInfo)
{
    return cusolverDnCgeqrf(handle, m, n, reinterpret_cast<cuComplex*>(A), lda,
                            reinterpret_cast<cuComplex*>(TAU),
                            reinterpret_cast<cuComplex*>(Workspace), Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  std::complex<double>* A, int lda,
                                  std::complex<double>* TAU,
                                  std::complex<double>* Workspace, int Lwork,
                                  int* devInfo)
{
    return cusolverDnZgeqrf(handle, m, n, reinterpret_cast<cuDoubleComplex*>(A),
                            lda, reinterpret_cast<cuDoubleComplex*>(TAU),
                            reinterpret_cast<cuDoubleComplex*>(Workspace),
                            Lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k, float* A, int lda,
                                           float* tau, int* lwork)
{

    return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k, double* A, int lda,
                                           double* tau, int* lwork)
{

    return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k, std::complex<float>* A,
                                           int lda, std::complex<float>* tau,
                                           int* lwork)
{

    return cusolverDnCungqr_bufferSize(
        handle, m, n, k, reinterpret_cast<cuComplex*>(A), lda,
        reinterpret_cast<cuComplex*>(tau), lwork);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, int m,
                                           int n, int k,
                                           std::complex<double>* A, int lda,
                                           std::complex<double>* tau,
                                           int* lwork)
{

    return cusolverDnZungqr_bufferSize(
        handle, m, n, k, reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(tau), lwork);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                float* A, int lda, float* tau, float* work,
                                int lwork, int* devInfo)
{

    return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                double* A, int lda, double* tau, double* work,
                                int lwork, int* devInfo)
{

    return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                std::complex<float>* A, int lda,
                                std::complex<float>* tau,
                                std::complex<float>* work, int lwork,
                                int* devInfo)
{

    return cusolverDnCungqr(handle, m, n, k, reinterpret_cast<cuComplex*>(A),
                            lda, reinterpret_cast<cuComplex*>(tau),
                            reinterpret_cast<cuComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, int m, int n, int k,
                                std::complex<double>* A, int lda,
                                std::complex<double>* tau,
                                std::complex<double>* work, int lwork,
                                int* devInfo)
{

    return cusolverDnZungqr(
        handle, m, n, k, reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(tau),
        reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             float* A, int lda, float* W,
                                             int* lwork)
{

    return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             double* A, int lda, double* W,
                                             int* lwork)
{

    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             std::complex<float>* A, int lda,
                                             float* W, int* lwork)
{

    return cusolverDnCheevd_bufferSize(
        handle, jobz, uplo, n, reinterpret_cast<cuComplex*>(A), lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int n,
                                             std::complex<double>* A, int lda,
                                             double* W, int* lwork)
{

    return cusolverDnZheevd_bufferSize(handle, jobz, uplo, n,
                                       reinterpret_cast<cuDoubleComplex*>(A),
                                       lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, float* A, int lda, float* W,
                                  float* work, int lwork, int* devInfo)
{

    return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, double* A, int lda, double* W,
                                  double* work, int lwork, int* devInfo)
{
    return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, std::complex<float>* A, int lda,
                                  float* W, std::complex<float>* work,
                                  int lwork, int* devInfo)
{
    return cusolverDnCheevd(handle, jobz, uplo, n,
                            reinterpret_cast<cuComplex*>(A), lda, W,
                            reinterpret_cast<cuComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, std::complex<double>* A, int lda,
                                  double* W, std::complex<double>* work,
                                  int lwork, int* devInfo)
{
    return cusolverDnZheevd(
        handle, jobz, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, W,
        reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             float* A, int lda, int* Lwork)
{

    return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             double* A, int lda, int* Lwork)
{

    return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             std::complex<float>* A, int lda,
                                             int* Lwork)
{

    return cusolverDnCpotrf_bufferSize(
        handle, uplo, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, int n,
                                             std::complex<double>* A, int lda,
                                             int* Lwork)
{

    return cusolverDnZpotrf_bufferSize(
        handle, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, float* A,
                                  int lda, float* Workspace, int Lwork,
                                  int* devInfo)
{

    return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n, double* A,
                                  int lda, double* Workspace, int Lwork,
                                  int* devInfo)
{

    return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  std::complex<float>* A, int lda,
                                  std::complex<float>* Workspace, int Lwork,
                                  int* devInfo)
{

    return cusolverDnCpotrf(handle, uplo, n, reinterpret_cast<cuComplex*>(A),
                            lda, reinterpret_cast<cuComplex*>(Workspace), Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, int n,
                                  std::complex<double>* A, int lda,
                                  std::complex<double>* Workspace, int Lwork,
                                  int* devInfo)
{

    return cusolverDnZpotrf(
        handle, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(Workspace), Lwork, devInfo);
}

////
cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const float* alpha, const float* A, int lda,
                             const float* beta, float* C, int ldc)
{

    return cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const double* alpha, const double* A, int lda,
                             const double* beta, double* C, int ldc)
{

    return cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const float* alpha, const std::complex<float>* A,
                             int lda, const float* beta, std::complex<float>* C,
                             int ldc)
{

    return cublasCherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const cuComplex*>(A), lda, beta,
                       reinterpret_cast<cuComplex*>(C), ldc);
}

cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const double* alpha, const std::complex<double>* A,
                             int lda, const double* beta,
                             std::complex<double>* C, int ldc)
{

    return cublasZherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const cuDoubleComplex*>(A), lda, beta,
                       reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

/////
cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const float* alpha, const float* A, int lda,
                           float* B, int ldb)
{

    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const double* alpha, const double* A, int lda,
                           double* B, int ldb)
{

    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, int lda,
                           std::complex<float>* B, int ldb)
{

    return cublasCtrsm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(A), lda,
                       reinterpret_cast<cuComplex*>(B), ldb);
}

cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, int lda,
                           std::complex<double>* B, int ldb)
{

    return cublasZtrsm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(A), lda,
                       reinterpret_cast<cuDoubleComplex*>(B), ldb);
}

#if 1
cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const float* alpha,
                           const float* x, int incx, float* y, int incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const double* alpha,
                           const double* x, int incx, double* y, int incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* x, int incx,
                           std::complex<float>* y, int incy)
{
    return cublasCaxpy(handle, n, reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<cuComplex*>(y), incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* x, int incx,
                           std::complex<double>* y, int incy)
{
    return cublasZaxpy(handle, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(x), incx,
                       reinterpret_cast<cuDoubleComplex*>(y), incy);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, int n, const float* x,
                           int incx, float* result)
{
    return cublasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, int n, const double* x,
                           int incx, double* result)
{
    return cublasDnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, int n,
                           const std::complex<float>* x, int incx,
                           float* result)
{
    return cublasScnrm2(handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                        result);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, int n,
                           const std::complex<double>* x, int incx,
                           double* result)
{
    return cublasDznrm2(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                        incx, result);
}

cublasStatus_t cublasTdot(cublasHandle_t handle, int n, const float* x,
                          int incx, const float* y, int incy, float* result)
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTdot(cublasHandle_t handle, int n, const double* x,
                          int incx, const double* y, int incy, double* result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTdot(cublasHandle_t handle, int n,
                          const std::complex<float>* x, int incx,
                          const std::complex<float>* y, int incy,
                          std::complex<float>* result)
{
    return cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<const cuComplex*>(y), incy,
                       reinterpret_cast<cuComplex*>(result));
}

cublasStatus_t cublasTdot(cublasHandle_t handle, int n,
                          const std::complex<double>* x, int incx,
                          const std::complex<double>* y, int incy,
                          std::complex<double>* result)
{
    return cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                       incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                       reinterpret_cast<cuDoubleComplex*>(result));
}

cublasStatus_t cublasTscal(cublasHandle_t handle, int n, const float* alpha,
                           float* x, int incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasTscal(cublasHandle_t handle, int n, const double* alpha,
                           double* x, int incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasTscal(cublasHandle_t handle, int n,
                           const std::complex<float>* alpha,
                           std::complex<float>* x, int incx)
{
    return cublasCscal(handle, n, reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<cuComplex*>(x), incx);
}

cublasStatus_t cublasTscal(cublasHandle_t handle, int n,
                           const std::complex<double>* alpha,
                           std::complex<double>* x, int incx)
{
    return cublasZscal(handle, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<cuDoubleComplex*>(x), incx);
}

#endif

static const char* cublasGetErrorString(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}

#define cuda_exec(func_call)                                                   \
    do                                                                         \
    {                                                                          \
        cudaError_t error = (func_call);                                       \
                                                                               \
        if (error != cudaSuccess)                                              \
        {                                                                      \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                    cudaGetErrorString(error));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define cublas_exec(func_call)                                                 \
    do                                                                         \
    {                                                                          \
        cublasStatus_t error = (func_call);                                    \
                                                                               \
        if (error != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                      \
            fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,                 \
                    cublasGetErrorString(error));                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
