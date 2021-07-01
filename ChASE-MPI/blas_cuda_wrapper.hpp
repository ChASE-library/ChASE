/* -*- Mode: C++; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
// This file is a part of ChASE.
// Copyright (c) 2015-2021, Simulation and Data Laboratory Quantum Materials, 
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float* alpha, const float* A, int lda,
                           const float* B, int ldb, const float* beta, float* C,
                           int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double* alpha, const double* A, int lda,
                           const double* B, int ldb, const double* beta,
                           double* C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, int lda,
                           const std::complex<float>* B, int ldb,
                           const std::complex<float>* beta,
                           std::complex<float>* C, int ldc) {
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
                           std::complex<double>* C, int ldc) {
  return cublasZgemm(handle, transa, transb, m, n, k,
                     reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(A), lda,
                     reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                     reinterpret_cast<const cuDoubleComplex*>(beta),
                     reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

cusolverStatus_t
cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      float *A,
                      int lda,
                      int *Lwork ){


    return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t
cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      double *A,
                      int lda,
                      int *Lwork ){


    return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t
cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      std::complex<float> *A,
                      int lda,
                      int *Lwork ){


    return cusolverDnCgeqrf_bufferSize(handle, m, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
}

cusolverStatus_t
cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle,
                      int m,
                      int n,
                      std::complex<double> *A,
                      int lda,
                      int *Lwork ){


    return cusolverDnZgeqrf_bufferSize(handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
}



cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n, float *A,
				  int lda, float *TAU, float *Workspace, int Lwork, int *devInfo){
  return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n, double *A,
                                  int lda, double *TAU, double *Workspace, int Lwork, int *devInfo){
  return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n, std::complex<float> *A,
                                  int lda, std::complex<float> *TAU, std::complex<float> *Workspace, int Lwork, int *devInfo){
  return cusolverDnCgeqrf(handle, m, n,  reinterpret_cast<cuComplex*>(A), 
			  lda,  reinterpret_cast<cuComplex*>(TAU),  
			  reinterpret_cast<cuComplex*>(Workspace), Lwork, devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, int m, int n, std::complex<double> *A,
                                  int lda, std::complex<double> *TAU, std::complex<double> *Workspace, int Lwork, int *devInfo){
  return cusolverDnZgeqrf(handle, m, n,  reinterpret_cast<cuDoubleComplex*>(A),
                          lda,  reinterpret_cast<cuDoubleComplex*>(TAU),
                          reinterpret_cast<cuDoubleComplex*>(Workspace), Lwork, devInfo);
}

cusolverStatus_t
cusolverDnTgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    float *A,
    int lda,
    float *tau,
    int *lwork){

    return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);

}

cusolverStatus_t
cusolverDnTgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    double *A,
    int lda,
    double *tau,
    int *lwork){

    return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);

}


cusolverStatus_t
cusolverDnTgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    std::complex<float> *A,
    int lda,
    std::complex<float> *tau,
    int *lwork){

    return cusolverDnCungqr_bufferSize(handle, m, n, k, reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(tau), lwork);

}

cusolverStatus_t
cusolverDnTgqr_bufferSize(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    std::complex<double> *A,
    int lda,
    std::complex<double> *tau,
    int *lwork){

    return cusolverDnZungqr_bufferSize(handle, m, n, k, reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(tau), lwork);

}


cusolverStatus_t
cusolverDnTgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    float *A,
    int lda,
    float *tau,
    float *work,
    int lwork,
    int *devInfo){

    return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

cusolverStatus_t
cusolverDnTgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    double *A,
    int lda,
    double *tau,
    double *work,
    int lwork,
    int *devInfo){

    return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

cusolverStatus_t
cusolverDnTgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    std::complex<float> *A,
    int lda,
    std::complex<float> *tau,
    std::complex<float> *work,
    int lwork,
    int *devInfo){

    return cusolverDnCungqr(handle, m, n, k, reinterpret_cast<cuComplex*>(A), lda, reinterpret_cast<cuComplex*>(tau), reinterpret_cast<cuComplex*>(work), lwork, devInfo);
}


cusolverStatus_t
cusolverDnTgqr(
    cusolverDnHandle_t handle,
    int m,
    int n,
    int k,
    std::complex<double> *A,
    int lda,
    std::complex<double> *tau,
    std::complex<double> *work,
    int lwork,
    int *devInfo){

    return cusolverDnZungqr(handle, m, n, k, reinterpret_cast<cuDoubleComplex*>(A), lda, reinterpret_cast<cuDoubleComplex*>(tau), reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
}


cusolverStatus_t
cusolverDnTheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *W,
    int *lwork)
{

    return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t
cusolverDnTheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W,
    int *lwork)
{

    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}


cusolverStatus_t
cusolverDnTheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    std::complex<float> *A,
    int lda,
    float *W,
    int *lwork)
{

    return cusolverDnCheevd_bufferSize(handle, jobz, uplo, n, reinterpret_cast<cuComplex*>(A), lda, W, lwork);
}


cusolverStatus_t
cusolverDnTheevd_bufferSize(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    std::complex<double> *A,
    int lda,
    double *W,
    int *lwork)
{

    return cusolverDnZheevd_bufferSize(handle, jobz, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, W, lwork);
}



cusolverStatus_t
cusolverDnTheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    float *A,
    int lda,
    float *W,
    float *work,
    int lwork,
    int *devInfo)
{

    return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}

cusolverStatus_t
cusolverDnTheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    double *A,
    int lda,
    double *W,
    double *work,
    int lwork,
    int *devInfo)
{
    return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}

cusolverStatus_t
cusolverDnTheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    std::complex<float> *A,
    int lda,
    float *W,
    std::complex<float> *work,
    int lwork,
    int *devInfo)
{
    return cusolverDnCheevd(handle, jobz, uplo, n, reinterpret_cast<cuComplex*>(A), lda, W, reinterpret_cast<cuComplex*>(work), lwork, devInfo);
}

cusolverStatus_t
cusolverDnTheevd(
    cusolverDnHandle_t handle,
    cusolverEigMode_t jobz,
    cublasFillMode_t uplo,
    int n,
    std::complex<double> *A,
    int lda,
    double *W,
    std::complex<double> *work,
    int lwork,
    int *devInfo)
{
    return cusolverDnZheevd(handle, jobz, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, W, reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
}

#if 1
cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const float* alpha,
                           const float* x, int incx, float* y, int incy) {
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n, const double* alpha,
                           const double* x, int incx, double* y, int incy) {
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* x, int incx,
                           std::complex<float>* y, int incy) {
  return cublasCaxpy(handle, n, reinterpret_cast<const cuComplex*>(alpha),
                     reinterpret_cast<const cuComplex*>(x), incx,
                     reinterpret_cast<cuComplex*>(y), incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, int n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* x, int incx,
                           std::complex<double>* y, int incy) {
  return cublasZaxpy(handle, n, reinterpret_cast<const cuDoubleComplex*>(alpha),
                     reinterpret_cast<const cuDoubleComplex*>(x), incx,
                     reinterpret_cast<cuDoubleComplex*>(y), incy);
}
#endif
/*
void shiftMatrixGPU(float* A, int lda, int n, float shift, int offset,
                    cudaStream_t stream);
void shiftMatrixGPU(double* A, int lda, int n, double shift, int offset,
                    cudaStream_t stream);
void shiftMatrixGPU(std::complex<float>* A, int lda, int n,
                    std::complex<float> shift, int offset, cudaStream_t stream);
void shiftMatrixGPU(std::complex<double>* A, int lda, int n,
                    std::complex<double> shift, int offset,
                    cudaStream_t stream);
*/

#define cuda_exec(func_call)                             \
  do {                                                   \
    cudaError_t error = (func_call);                     \
                                                         \
    if (error != cudaSuccess) {                          \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error));                \
      exit(EXIT_FAILURE);                                \
    }                                                    \
  } while (0)
