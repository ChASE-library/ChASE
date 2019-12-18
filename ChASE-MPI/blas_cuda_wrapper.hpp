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

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
