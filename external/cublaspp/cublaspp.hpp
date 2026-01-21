// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <iostream>  // For std::cerr, std::endl

#include "algorithm/types.hpp"
#include <complex>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macro to check for cuBLAS errors
#define CHECK_CUBLAS_ERROR(val) checkCublas((val), #val, __FILE__, __LINE__)

// Function to check cuBLAS errors
void checkCublas(cublasStatus_t status, const char* const func,
                 const char* const file, const std::size_t line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Runtime Error at: " << file << ":" << line
                  << std::endl;
        switch (status)
        {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED: The cuBLAS "
                             "library was not initialized."
                          << std::endl;
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                std::cerr
                    << "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed."
                    << std::endl;
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                std::cerr << "CUBLAS_STATUS_INVALID_VALUE: An invalid value "
                             "was encountered."
                          << std::endl;
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH: The device "
                             "architecture is not supported."
                          << std::endl;
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                std::cerr
                    << "CUBLAS_STATUS_EXECUTION_FAILED: The execution failed."
                    << std::endl;
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR: An internal error "
                             "occurred."
                          << std::endl;
                break;
            default:
                std::cerr << "Unknown cuBLAS error." << std::endl;
                break;
        }
        std::cerr << "Function: " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @page cublaspp_namespace chase::linalg::cublaspp Namespace
 * @brief A templated C++ interface to the cuBLAS library.
 *
 * The `chase::linalg::cublaspp` namespace contains templated functions that
 * interface with the cuBLAS library to provide efficient linear algebra
 * routines on GPU. The functions are templated to work with different data
 * types such as `float`, `double`, and `std::complex`. These functions allow
 * for easy and efficient numerical computations in scientific computing.
 */

/**
 * @defgroup cuBlasFunctions cuBLAS Routines
 * @brief Template functions that interface with cuBLAS routines.
 *
 * These functions provide common linear algebra operations such as vector
 * norms, dot products, matrix multiplications, etc. They allow for operations
 * on both real and complex numbers.
 */

namespace chase
{
namespace linalg
{
namespace cublaspp
{
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTaxpy(cublasHandle_t handle, std::size_t n,
                           const float* alpha, const float* x, std::size_t incx,
                           float* y, std::size_t incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTaxpy(cublasHandle_t handle, std::size_t n,
                           const double* alpha, const double* x,
                           std::size_t incx, double* y, std::size_t incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTaxpy(cublasHandle_t handle, std::size_t n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* x, std::size_t incx,
                           std::complex<float>* y, std::size_t incy)
{
    return cublasCaxpy(handle, n, reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<cuComplex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTaxpy(cublasHandle_t handle, std::size_t n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* x, std::size_t incx,
                           std::complex<double>* y, std::size_t incy)
{
    return cublasZaxpy(handle, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(x), incx,
                       reinterpret_cast<cuDoubleComplex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n, const float* x,
                           std::size_t incx, float* result)
{
    return cublasSnrm2(handle, n, x, incx, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n,
                           const double* x, std::size_t incx, double* result)
{
    return cublasDnrm2(handle, n, x, incx, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n,
                           const std::complex<float>* x, std::size_t incx,
                           float* result)
{
    return cublasScnrm2(handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                        result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n,
                           const std::complex<double>* x, std::size_t incx,
                           double* result)
{
    return cublasDznrm2(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                        incx, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n, const float* x,
                          std::size_t incx, const float* y, std::size_t incy,
                          float* result)
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n, const double* x,
                          std::size_t incx, const double* y, std::size_t incy,
                          double* result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n,
                          const std::complex<float>* x, std::size_t incx,
                          const std::complex<float>* y, std::size_t incy,
                          std::complex<float>* result)
{
    return cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<const cuComplex*>(y), incy,
                       reinterpret_cast<cuComplex*>(result));
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n,
                          const std::complex<double>* x, std::size_t incx,
                          const std::complex<double>* y, std::size_t incy,
                          std::complex<double>* result)
{
    return cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                       incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                       reinterpret_cast<cuDoubleComplex*>(result));
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n,
                           const float* alpha, float* x, std::size_t incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n,
                           const double* alpha, double* x, std::size_t incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n,
                           const std::complex<float>* alpha,
                           std::complex<float>* x, std::size_t incx)
{
    return cublasCscal(handle, n, reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<cuComplex*>(x), incx);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle cuBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n,
                           const std::complex<double>* alpha,
                           std::complex<double>* x, std::size_t incx)
{
    return cublasZscal(handle, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<cuDoubleComplex*>(x), incx);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A.
 * @param alpha Scaling factor for A*x.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param beta Scaling factor for y.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n, const float* alpha,
                           const float* A, std::size_t lda, const float* x,
                           std::size_t incx, const float* beta, float* y,
                           std::size_t incy)
{
    return cublasSgemv(handle, transa, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A.
 * @param alpha Scaling factor for A*x.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param beta Scaling factor for y.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n, const double* alpha,
                           const double* A, std::size_t lda, const double* x,
                           std::size_t incx, const double* beta, double* y,
                           std::size_t incy)
{
    return cublasDgemv(handle, transa, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A.
 * @param alpha Scaling factor for A*x.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param beta Scaling factor for y.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, std::size_t lda,
                           const std::complex<float>* x, std::size_t incx,
                           const std::complex<float>* beta,
                           std::complex<float>* y, std::size_t incy)
{
    return cublasCgemv(handle, transa, m, n,
                       reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(A), lda,
                       reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<const cuComplex*>(beta),
                       reinterpret_cast<cuComplex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix A.
 * @param alpha Scaling factor for A*x.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param beta Scaling factor for y.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, std::size_t lda,
                           const std::complex<double>* x, std::size_t incx,
                           const std::complex<double>* beta,
                           std::complex<double>* y, std::size_t incy)
{
    return cublasZgemv(handle, transa, m, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(A), lda,
                       reinterpret_cast<const cuDoubleComplex*>(x), incx,
                       reinterpret_cast<const cuDoubleComplex*>(beta),
                       reinterpret_cast<cuDoubleComplex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param transb Operation on matrix B.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix B.
 * @param k Number of columns in A and rows in B.
 * @param alpha Scaling factor for A*B.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param B Pointer to matrix B.
 * @param ldb Leading dimension of B.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m,
                           std::size_t n, std::size_t k, const float* alpha,
                           const float* A, std::size_t lda, const float* B,
                           std::size_t ldb, const float* beta, float* C,
                           std::size_t ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param transb Operation on matrix B.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix B.
 * @param k Number of columns in A and rows in B.
 * @param alpha Scaling factor for A*B.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param B Pointer to matrix B.
 * @param ldb Leading dimension of B.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m,
                           std::size_t n, std::size_t k, const double* alpha,
                           const double* A, std::size_t lda, const double* B,
                           std::size_t ldb, const double* beta, double* C,
                           std::size_t ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param transb Operation on matrix B.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix B.
 * @param k Number of columns in A and rows in B.
 * @param alpha Scaling factor for A*B.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param B Pointer to matrix B.
 * @param ldb Leading dimension of B.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m,
                           std::size_t n, std::size_t k,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, std::size_t lda,
                           const std::complex<float>* B, std::size_t ldb,
                           const std::complex<float>* beta,
                           std::complex<float>* C, std::size_t ldc)
{
    return cublasCgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(A), lda,
                       reinterpret_cast<const cuComplex*>(B), ldb,
                       reinterpret_cast<const cuComplex*>(beta),
                       reinterpret_cast<cuComplex*>(C), ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle cuBLAS handle.
 * @param transa Operation on matrix A.
 * @param transb Operation on matrix B.
 * @param m Number of rows in matrix A.
 * @param n Number of columns in matrix B.
 * @param k Number of columns in A and rows in B.
 * @param alpha Scaling factor for A*B.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param B Pointer to matrix B.
 * @param ldb Leading dimension of B.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m,
                           std::size_t n, std::size_t k,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, std::size_t lda,
                           const std::complex<double>* B, std::size_t ldb,
                           const std::complex<double>* beta,
                           std::complex<double>* C, std::size_t ldc)
{
    return cublasZgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(A), lda,
                       reinterpret_cast<const cuDoubleComplex*>(B), ldb,
                       reinterpret_cast<const cuDoubleComplex*>(beta),
                       reinterpret_cast<cuDoubleComplex*>(C), ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle cuBLAS handle.
 * @param uplo Specifies whether the upper or lower triangular part of C is
 * used.
 * @param trans Operation on matrix A.
 * @param n Number of rows and columns in matrix C.
 * @param k Number of columns of A if trans is non-transpose, rows otherwise.
 * @param alpha Scaling factor for A * A^H.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output Hermitian matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n,
                             std::size_t k, const float* alpha, const float* A,
                             std::size_t lda, const float* beta, float* C,
                             std::size_t ldc)
{

    return cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle cuBLAS handle.
 * @param uplo Specifies whether the upper or lower triangular part of C is
 * used.
 * @param trans Operation on matrix A.
 * @param n Number of rows and columns in matrix C.
 * @param k Number of columns of A if trans is non-transpose, rows otherwise.
 * @param alpha Scaling factor for A * A^H.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output Hermitian matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n,
                             std::size_t k, const double* alpha,
                             const double* A, std::size_t lda,
                             const double* beta, double* C, std::size_t ldc)
{

    return cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle cuBLAS handle.
 * @param uplo Specifies whether the upper or lower triangular part of C is
 * used.
 * @param trans Operation on matrix A.
 * @param n Number of rows and columns in matrix C.
 * @param k Number of columns of A if trans is non-transpose, rows otherwise.
 * @param alpha Scaling factor for A * A^H.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output Hermitian matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n,
                             std::size_t k, const float* alpha,
                             const std::complex<float>* A, std::size_t lda,
                             const float* beta, std::complex<float>* C,
                             std::size_t ldc)
{

    return cublasCherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const cuComplex*>(A), lda, beta,
                       reinterpret_cast<cuComplex*>(C), ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle cuBLAS handle.
 * @param uplo Specifies whether the upper or lower triangular part of C is
 * used.
 * @param trans Operation on matrix A.
 * @param n Number of rows and columns in matrix C.
 * @param k Number of columns of A if trans is non-transpose, rows otherwise.
 * @param alpha Scaling factor for A * A^H.
 * @param A Pointer to matrix A.
 * @param lda Leading dimension of A.
 * @param beta Scaling factor for C.
 * @param C Pointer to input/output Hermitian matrix C.
 * @param ldc Leading dimension of C.
 * @return Status of the operation.
 */
cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n,
                             std::size_t k, const double* alpha,
                             const std::complex<double>* A, std::size_t lda,
                             const double* beta, std::complex<double>* C,
                             std::size_t ldc)
{

    return cublasZherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const cuDoubleComplex*>(A), lda, beta,
                       reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using cuBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The cuBLAS library handle, used to manage the cuBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `CUBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `CUBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `CUBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `CUBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `CUBLAS_OP_N`: No transpose
 *                  - `CUBLAS_OP_T`: Transpose
 *                  - `CUBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `CUBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `CUBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `CUBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `CUBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `cublasStatus_t`, indicating success or the type of
 * error encountered.
 */
cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, std::size_t m, std::size_t n,
                           const float* alpha, const float* A, std::size_t lda,
                           float* B, std::size_t ldb)
{

    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using cuBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The cuBLAS library handle, used to manage the cuBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `CUBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `CUBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `CUBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `CUBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `CUBLAS_OP_N`: No transpose
 *                  - `CUBLAS_OP_T`: Transpose
 *                  - `CUBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `CUBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `CUBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `CUBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `CUBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `cublasStatus_t`, indicating success or the type of
 * error encountered.
 */
cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, std::size_t m, std::size_t n,
                           const double* alpha, const double* A,
                           std::size_t lda, double* B, std::size_t ldb)
{

    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using cuBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The cuBLAS library handle, used to manage the cuBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `CUBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `CUBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `CUBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `CUBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `CUBLAS_OP_N`: No transpose
 *                  - `CUBLAS_OP_T`: Transpose
 *                  - `CUBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `CUBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `CUBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `CUBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `CUBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `cublasStatus_t`, indicating success or the type of
 * error encountered.
 */
cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, std::size_t m, std::size_t n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, std::size_t lda,
                           std::complex<float>* B, std::size_t ldb)
{

    return cublasCtrsm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(A), lda,
                       reinterpret_cast<cuComplex*>(B), ldb);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using cuBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The cuBLAS library handle, used to manage the cuBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `CUBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `CUBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `CUBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `CUBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `CUBLAS_OP_N`: No transpose
 *                  - `CUBLAS_OP_T`: Transpose
 *                  - `CUBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `CUBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `CUBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `CUBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `CUBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `cublasStatus_t`, indicating success or the type of
 * error encountered.
 */
cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, std::size_t m, std::size_t n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, std::size_t lda,
                           std::complex<double>* B, std::size_t ldb)
{

    return cublasZtrsm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<const cuDoubleComplex*>(A), lda,
                       reinterpret_cast<cuDoubleComplex*>(B), ldb);
}

/**
 * @ingroup cuBlasFunctions
 * @brief Performs an extended matrix-matrix product: C = alpha * A * B + beta *
 * C Template specialization for float type
 */
template <cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F>
cublasStatus_t cublasTgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, std::size_t m,
                             std::size_t n, std::size_t k, const float* alpha,
                             const float* A, std::size_t lda, const float* B,
                             std::size_t ldb, const float* beta, float* C,
                             std::size_t ldc)
{
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_32F,
                        lda, B, CUDA_R_32F, ldb, beta, C, CUDA_R_32F, ldc,
                        ComputeType, CUBLAS_GEMM_DEFAULT);
}

/**
 * @brief Template specialization for double type
 */
template <cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F_FAST_TF32>
cublasStatus_t cublasTgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, std::size_t m,
                             std::size_t n, std::size_t k, const double* alpha,
                             const double* A, std::size_t lda, const double* B,
                             std::size_t ldb, const double* beta, double* C,
                             std::size_t ldc)
{
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_R_64F,
                        lda, B, CUDA_R_64F, ldb, beta, C, CUDA_R_64F, ldc,
                        ComputeType, CUBLAS_GEMM_DEFAULT);
}

/**
 * @brief Template specialization for complex<float> type
 */
template <cublasComputeType_t ComputeType = CUBLAS_COMPUTE_32F_FAST_TF32>
cublasStatus_t cublasTgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, std::size_t m,
                             std::size_t n, std::size_t k,
                             const std::complex<float>* alpha,
                             const std::complex<float>* A, std::size_t lda,
                             const std::complex<float>* B, std::size_t ldb,
                             const std::complex<float>* beta,
                             std::complex<float>* C, std::size_t ldc)
{
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_C_32F,
                        lda, B, CUDA_C_32F, ldb, beta, C, CUDA_C_32F, ldc,
                        ComputeType, CUBLAS_GEMM_DEFAULT);
}

/**
 * @brief Template specialization for complex<double> type
 */
template <cublasComputeType_t ComputeType = CUBLAS_COMPUTE_64F>
cublasStatus_t cublasTgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, std::size_t m,
                             std::size_t n, std::size_t k,
                             const std::complex<double>* alpha,
                             const std::complex<double>* A, std::size_t lda,
                             const std::complex<double>* B, std::size_t ldb,
                             const std::complex<double>* beta,
                             std::complex<double>* C, std::size_t ldc)
{
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, CUDA_C_64F,
                        lda, B, CUDA_C_64F, ldb, beta, C, CUDA_C_64F, ldc,
                        ComputeType, CUBLAS_GEMM_DEFAULT);
}

} // namespace cublaspp
} // namespace linalg
} // namespace chase