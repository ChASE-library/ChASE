// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <iostream>  // For std::cerr, std::endl
#include <cstdlib>

#include "algorithm/types.hpp"
#include <complex>
#include <rocblas/rocblas.h>

#include <hip/hip_runtime.h>

// Macro to check for rocBLAS errors
#define CHECK_ROCBLAS_ERROR(val) checkRocblas((val), #val, __FILE__, __LINE__)

// Function to check rocBLAS errors
void checkRocblas(rocblas_status status, const char* const func,
                 const char* const file, const std::size_t line)
{
    if (status != rocblas_status_success)
    {
        std::cerr << "rocBLAS Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << "rocblas_status = " << static_cast<int>(status);
        switch (status)
        {
            case rocblas_status_invalid_handle:
                std::cerr << " (invalid_handle)";
                break;
            case rocblas_status_not_implemented:
                std::cerr << " (not_implemented)";
                break;
            case rocblas_status_invalid_pointer:
                std::cerr << " (invalid_pointer)";
                break;
            case rocblas_status_invalid_size:
                std::cerr << " (invalid_size)";
                break;
            case rocblas_status_memory_error:
                std::cerr << " (memory_error)";
                break;
            case rocblas_status_internal_error:
                std::cerr << " (internal_error)";
                break;
            case rocblas_status_perf_degraded:
                std::cerr << " (perf_degraded)";
                break;
            case rocblas_status_invalid_value:
                std::cerr << " (invalid_value)";
                break;
            case rocblas_status_arch_mismatch:
                std::cerr << " (arch_mismatch)";
                break;
            default:
                break;
        }
        std::cerr << std::endl;
        std::cerr << "Function: " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

/**
 * @page rocblaspp_namespace chase::linalg::rocblaspp Namespace
 * @brief A templated C++ interface to the rocBLAS library.
 *
 * The `chase::linalg::rocblaspp` namespace contains templated functions that
 * interface with the rocBLAS library to provide efficient linear algebra
 * routines on GPU. The functions are templated to work with different data
 * types such as `float`, `double`, and `std::complex`. These functions allow
 * for easy and efficient numerical computations in scientific computing.
 */

/**
 * @defgroup cuBlasFunctions rocBLAS Routines
 * @brief Template functions that interface with rocBLAS routines.
 *
 * These functions provide common linear algebra operations such as vector
 * norms, dot products, matrix multiplications, etc. They allow for operations
 * on both real and complex numbers.
 */

namespace chase
{
namespace linalg
{
namespace rocblaspp
{
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTaxpy(rocblas_handle handle, std::size_t n,
                           const float* alpha, const float* x, std::size_t incx,
                           float* y, std::size_t incy)
{
    return rocblas_saxpy(handle, n, alpha, x, incx, y, incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTaxpy(rocblas_handle handle, std::size_t n,
                           const double* alpha, const double* x,
                           std::size_t incx, double* y, std::size_t incy)
{
    return rocblas_daxpy(handle, n, alpha, x, incx, y, incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTaxpy(rocblas_handle handle, std::size_t n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* x, std::size_t incx,
                           std::complex<float>* y, std::size_t incy)
{
    return rocblas_caxpy(handle, n, reinterpret_cast<const rocblas_float_complex*>(alpha),
                       reinterpret_cast<const rocblas_float_complex*>(x), incx,
                       reinterpret_cast<rocblas_float_complex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs the AXPY operation: y = alpha * x + y.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param alpha Scaling factor.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTaxpy(rocblas_handle handle, std::size_t n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* x, std::size_t incx,
                           std::complex<double>* y, std::size_t incy)
{
    return rocblas_zaxpy(handle, n,
                       reinterpret_cast<const rocblas_double_complex*>(alpha),
                       reinterpret_cast<const rocblas_double_complex*>(x), incx,
                       reinterpret_cast<rocblas_double_complex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
rocblas_status rocblasTnrm2(rocblas_handle handle, std::size_t n, const float* x,
                           std::size_t incx, float* result)
{
    return rocblas_snrm2(handle, n, x, incx, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
rocblas_status rocblasTnrm2(rocblas_handle handle, std::size_t n,
                           const double* x, std::size_t incx, double* result)
{
    return rocblas_dnrm2(handle, n, x, incx, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
rocblas_status rocblasTnrm2(rocblas_handle handle, std::size_t n,
                           const std::complex<float>* x, std::size_t incx,
                           float* result)
{
    return rocblas_scnrm2(handle, n, reinterpret_cast<const rocblas_float_complex*>(x), incx,
                        result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the Euclidean norm of a vector.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param result Pointer to the result norm.
 * @return Status of the operation.
 */
rocblas_status rocblasTnrm2(rocblas_handle handle, std::size_t n,
                           const std::complex<double>* x, std::size_t incx,
                           double* result)
{
    return rocblas_dznrm2(handle, n, reinterpret_cast<const rocblas_double_complex*>(x),
                        incx, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
rocblas_status rocblasTdot(rocblas_handle handle, std::size_t n, const float* x,
                          std::size_t incx, const float* y, std::size_t incy,
                          float* result)
{
    return rocblas_sdot(handle, n, x, incx, y, incy, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
rocblas_status rocblasTdot(rocblas_handle handle, std::size_t n, const double* x,
                          std::size_t incx, const double* y, std::size_t incy,
                          double* result)
{
    return rocblas_ddot(handle, n, x, incx, y, incy, result);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
rocblas_status rocblasTdot(rocblas_handle handle, std::size_t n,
                          const std::complex<float>* x, std::size_t incx,
                          const std::complex<float>* y, std::size_t incy,
                          std::complex<float>* result)
{
    return rocblas_cdotc(handle, n, reinterpret_cast<const rocblas_float_complex*>(x), incx,
                       reinterpret_cast<const rocblas_float_complex*>(y), incy,
                       reinterpret_cast<rocblas_float_complex*>(result));
}
/**
 * @ingroup cuBlasFunctions
 * @brief Computes the dot product of two vectors.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input vector y.
 * @param incy Increment for y elements.
 * @param result Pointer to the result dot product.
 * @return Status of the operation.
 */
rocblas_status rocblasTdot(rocblas_handle handle, std::size_t n,
                          const std::complex<double>* x, std::size_t incx,
                          const std::complex<double>* y, std::size_t incy,
                          std::complex<double>* result)
{
    return rocblas_zdotc(handle, n, reinterpret_cast<const rocblas_double_complex*>(x),
                       incx, reinterpret_cast<const rocblas_double_complex*>(y), incy,
                       reinterpret_cast<rocblas_double_complex*>(result));
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTscal(rocblas_handle handle, std::size_t n,
                           const float* alpha, float* x, std::size_t incx)
{
    return rocblas_sscal(handle, n, alpha, x, incx);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTscal(rocblas_handle handle, std::size_t n,
                           const double* alpha, double* x, std::size_t incx)
{
    return rocblas_dscal(handle, n, alpha, x, incx);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTscal(rocblas_handle handle, std::size_t n,
                           const std::complex<float>* alpha,
                           std::complex<float>* x, std::size_t incx)
{
    return rocblas_cscal(handle, n, reinterpret_cast<const rocblas_float_complex*>(alpha),
                       reinterpret_cast<rocblas_float_complex*>(x), incx);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Scales a vector by a scalar.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vector x.
 * @param alpha Scaling factor.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTscal(rocblas_handle handle, std::size_t n,
                           const std::complex<double>* alpha,
                           std::complex<double>* x, std::size_t incx)
{
    return rocblas_zscal(handle, n,
                       reinterpret_cast<const rocblas_double_complex*>(alpha),
                       reinterpret_cast<rocblas_double_complex*>(x), incx);
}

/**
 * @ingroup cuBlasFunctions
 * @brief Swaps two vectors element-wise.
 * @param handle rocBLAS handle.
 * @param n Number of elements in vectors x and y.
 * @param x Pointer to input/output vector x.
 * @param incx Increment for x elements.
 * @param y Pointer to input/output vector y.
 * @param incy Increment for y elements.
 * @return Status of the operation.
 */
rocblas_status rocblasTswap(rocblas_handle handle, std::size_t n, float* x,
                           std::size_t incx, float* y, std::size_t incy)
{
    return rocblas_sswap(handle, n, x, incx, y, incy);
}

rocblas_status rocblasTswap(rocblas_handle handle, std::size_t n, double* x,
                           std::size_t incx, double* y, std::size_t incy)
{
    return rocblas_dswap(handle, n, x, incx, y, incy);
}

rocblas_status rocblasTswap(rocblas_handle handle, std::size_t n,
                           std::complex<float>* x, std::size_t incx,
                           std::complex<float>* y, std::size_t incy)
{
    return rocblas_cswap(handle, n, reinterpret_cast<rocblas_float_complex*>(x), incx,
                       reinterpret_cast<rocblas_float_complex*>(y), incy);
}

rocblas_status rocblasTswap(rocblas_handle handle, std::size_t n,
                           std::complex<double>* x, std::size_t incx,
                           std::complex<double>* y, std::size_t incy)
{
    return rocblas_zswap(handle, n, reinterpret_cast<rocblas_double_complex*>(x), incx,
                       reinterpret_cast<rocblas_double_complex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemv(rocblas_handle handle, rocblas_operation transa,
                           std::size_t m, std::size_t n, const float* alpha,
                           const float* A, std::size_t lda, const float* x,
                           std::size_t incx, const float* beta, float* y,
                           std::size_t incy)
{
    return rocblas_sgemv(handle, transa, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemv(rocblas_handle handle, rocblas_operation transa,
                           std::size_t m, std::size_t n, const double* alpha,
                           const double* A, std::size_t lda, const double* x,
                           std::size_t incx, const double* beta, double* y,
                           std::size_t incy)
{
    return rocblas_dgemv(handle, transa, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemv(rocblas_handle handle, rocblas_operation transa,
                           std::size_t m, std::size_t n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, std::size_t lda,
                           const std::complex<float>* x, std::size_t incx,
                           const std::complex<float>* beta,
                           std::complex<float>* y, std::size_t incy)
{
    return rocblas_cgemv(handle, transa, m, n,
                       reinterpret_cast<const rocblas_float_complex*>(alpha),
                       reinterpret_cast<const rocblas_float_complex*>(A), lda,
                       reinterpret_cast<const rocblas_float_complex*>(x), incx,
                       reinterpret_cast<const rocblas_float_complex*>(beta),
                       reinterpret_cast<rocblas_float_complex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-vector product: y = alpha * A * x + beta * y.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemv(rocblas_handle handle, rocblas_operation transa,
                           std::size_t m, std::size_t n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, std::size_t lda,
                           const std::complex<double>* x, std::size_t incx,
                           const std::complex<double>* beta,
                           std::complex<double>* y, std::size_t incy)
{
    return rocblas_zgemv(handle, transa, m, n,
                       reinterpret_cast<const rocblas_double_complex*>(alpha),
                       reinterpret_cast<const rocblas_double_complex*>(A), lda,
                       reinterpret_cast<const rocblas_double_complex*>(x), incx,
                       reinterpret_cast<const rocblas_double_complex*>(beta),
                       reinterpret_cast<rocblas_double_complex*>(y), incy);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemm(rocblas_handle handle, rocblas_operation transa,
                           rocblas_operation transb, std::size_t m,
                           std::size_t n, std::size_t k, const float* alpha,
                           const float* A, std::size_t lda, const float* B,
                           std::size_t ldb, const float* beta, float* C,
                           std::size_t ldc)
{
    return rocblas_sgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemm(rocblas_handle handle, rocblas_operation transa,
                           rocblas_operation transb, std::size_t m,
                           std::size_t n, std::size_t k, const double* alpha,
                           const double* A, std::size_t lda, const double* B,
                           std::size_t ldb, const double* beta, double* C,
                           std::size_t ldc)
{
    return rocblas_dgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemm(rocblas_handle handle, rocblas_operation transa,
                           rocblas_operation transb, std::size_t m,
                           std::size_t n, std::size_t k,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, std::size_t lda,
                           const std::complex<float>* B, std::size_t ldb,
                           const std::complex<float>* beta,
                           std::complex<float>* C, std::size_t ldc)
{
    return rocblas_cgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const rocblas_float_complex*>(alpha),
                       reinterpret_cast<const rocblas_float_complex*>(A), lda,
                       reinterpret_cast<const rocblas_float_complex*>(B), ldb,
                       reinterpret_cast<const rocblas_float_complex*>(beta),
                       reinterpret_cast<rocblas_float_complex*>(C), ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a matrix-matrix product: C = alpha * A * B + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTgemm(rocblas_handle handle, rocblas_operation transa,
                           rocblas_operation transb, std::size_t m,
                           std::size_t n, std::size_t k,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, std::size_t lda,
                           const std::complex<double>* B, std::size_t ldb,
                           const std::complex<double>* beta,
                           std::complex<double>* C, std::size_t ldc)
{
    return rocblas_zgemm(handle, transa, transb, m, n, k,
                       reinterpret_cast<const rocblas_double_complex*>(alpha),
                       reinterpret_cast<const rocblas_double_complex*>(A), lda,
                       reinterpret_cast<const rocblas_double_complex*>(B), ldb,
                       reinterpret_cast<const rocblas_double_complex*>(beta),
                       reinterpret_cast<rocblas_double_complex*>(C), ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTsyherk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, std::size_t n,
                             std::size_t k, const float* alpha, const float* A,
                             std::size_t lda, const float* beta, float* C,
                             std::size_t ldc)
{

    return rocblas_ssyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTsyherk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, std::size_t n,
                             std::size_t k, const double* alpha,
                             const double* A, std::size_t lda,
                             const double* beta, double* C, std::size_t ldc)
{

    return rocblas_dsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTsyherk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, std::size_t n,
                             std::size_t k, const float* alpha,
                             const std::complex<float>* A, std::size_t lda,
                             const float* beta, std::complex<float>* C,
                             std::size_t ldc)
{

    return rocblas_cherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const rocblas_float_complex*>(A), lda, beta,
                       reinterpret_cast<rocblas_float_complex*>(C), ldc);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Performs a Hermitian rank-k update: C = alpha * A * A^H + beta * C.
 * @param handle rocBLAS handle.
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
rocblas_status rocblasTsyherk(rocblas_handle handle, rocblas_fill uplo,
                             rocblas_operation trans, std::size_t n,
                             std::size_t k, const double* alpha,
                             const std::complex<double>* A, std::size_t lda,
                             const double* beta, std::complex<double>* C,
                             std::size_t ldc)
{

    return rocblas_zherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const rocblas_double_complex*>(A), lda, beta,
                       reinterpret_cast<rocblas_double_complex*>(C), ldc);
}

/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using rocBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The rocBLAS library handle, used to manage the rocBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `ROCBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `ROCBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `ROCBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `ROCBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `ROCBLAS_OP_N`: No transpose
 *                  - `ROCBLAS_OP_T`: Transpose
 *                  - `ROCBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `ROCBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `ROCBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `ROCBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `ROCBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `rocblas_status`, indicating success or the type of
 * error encountered.
 */
rocblas_status rocblasTtrsm(rocblas_handle handle, rocblas_side side,
                           rocblas_fill uplo, rocblas_operation trans,
                           rocblas_diagonal diag, std::size_t m, std::size_t n,
                           const float* alpha, const float* A, std::size_t lda,
                           float* B, std::size_t ldb)
{

    return rocblas_strsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using rocBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The rocBLAS library handle, used to manage the rocBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `ROCBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `ROCBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `ROCBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `ROCBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `ROCBLAS_OP_N`: No transpose
 *                  - `ROCBLAS_OP_T`: Transpose
 *                  - `ROCBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `ROCBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `ROCBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `ROCBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `ROCBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `rocblas_status`, indicating success or the type of
 * error encountered.
 */
rocblas_status rocblasTtrsm(rocblas_handle handle, rocblas_side side,
                           rocblas_fill uplo, rocblas_operation trans,
                           rocblas_diagonal diag, std::size_t m, std::size_t n,
                           const double* alpha, const double* A,
                           std::size_t lda, double* B, std::size_t ldb)
{

    return rocblas_dtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using rocBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The rocBLAS library handle, used to manage the rocBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `ROCBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `ROCBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `ROCBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `ROCBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `ROCBLAS_OP_N`: No transpose
 *                  - `ROCBLAS_OP_T`: Transpose
 *                  - `ROCBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `ROCBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `ROCBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `ROCBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `ROCBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `rocblas_status`, indicating success or the type of
 * error encountered.
 */
rocblas_status rocblasTtrsm(rocblas_handle handle, rocblas_side side,
                           rocblas_fill uplo, rocblas_operation trans,
                           rocblas_diagonal diag, std::size_t m, std::size_t n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* A, std::size_t lda,
                           std::complex<float>* B, std::size_t ldb)
{

    return rocblas_ctrsm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const rocblas_float_complex*>(alpha),
                       reinterpret_cast<const rocblas_float_complex*>(A), lda,
                       reinterpret_cast<rocblas_float_complex*>(B), ldb);
}
/**
 * @ingroup cuBlasFunctions
 * @brief Solves a triangular matrix equation using rocBLAS for single precision.
 *
 * This function performs a triangular solve (TRSM) operation, solving for
 * matrix X in one of the following matrix equations:
 * - `op(A) * X = alpha * B`
 * - `X * op(A) = alpha * B`
 * where `op(A)` is either `A`, `A^T` (transpose of A), or `A^H` (conjugate
 * transpose of A).
 *
 * @param handle    The rocBLAS library handle, used to manage the rocBLAS
 * context.
 * @param side      Specifies whether A appears on the left or right side of X.
 *                  Use `ROCBLAS_SIDE_LEFT` to solve `op(A) * X = alpha * B` or
 *                  `ROCBLAS_SIDE_RIGHT` to solve `X * op(A) = alpha * B`.
 * @param uplo      Specifies whether the matrix A is lower or upper triangular.
 *                  Use `ROCBLAS_FILL_MODE_LOWER` if A is lower triangular, or
 *                  `ROCBLAS_FILL_MODE_UPPER` if A is upper triangular.
 * @param trans     Indicates the operation to perform on A. Possible values
 * are:
 *                  - `ROCBLAS_OP_N`: No transpose
 *                  - `ROCBLAS_OP_T`: Transpose
 *                  - `ROCBLAS_OP_C`: Conjugate transpose
 * @param diag      Specifies whether the diagonal elements of A are assumed to
 * be unit (1) or not. Use `ROCBLAS_DIAG_UNIT` if A has a unit diagonal, or
 * `ROCBLAS_DIAG_NON_UNIT` if it does not.
 * @param m         Number of rows of matrix B.
 * @param n         Number of columns of matrix B.
 * @param alpha     Pointer to a scalar value that multiplies B.
 * @param A         Pointer to the triangular matrix A. If `side` is set to
 *                  `ROCBLAS_SIDE_LEFT`, the dimensions of A are `lda x m`.
 *                  If `side` is set to `ROCBLAS_SIDE_RIGHT`, the dimensions of A
 *                  are `lda x n`.
 * @param lda       Leading dimension of matrix A.
 * @param B         Pointer to matrix B, which will be overwritten with the
 * solution matrix X.
 * @param ldb       Leading dimension of matrix B.
 *
 * @return          Returns `rocblas_status`, indicating success or the type of
 * error encountered.
 */
rocblas_status rocblasTtrsm(rocblas_handle handle, rocblas_side side,
                           rocblas_fill uplo, rocblas_operation trans,
                           rocblas_diagonal diag, std::size_t m, std::size_t n,
                           const std::complex<double>* alpha,
                           const std::complex<double>* A, std::size_t lda,
                           std::complex<double>* B, std::size_t ldb)
{

    return rocblas_ztrsm(handle, side, uplo, trans, diag, m, n,
                       reinterpret_cast<const rocblas_double_complex*>(alpha),
                       reinterpret_cast<const rocblas_double_complex*>(A), lda,
                       reinterpret_cast<rocblas_double_complex*>(B), ldb);
}

/**
 * @ingroup cuBlasFunctions
 * @brief Performs an extended matrix-matrix product: C = alpha * A * B + beta *
 * C Template specialization for float type
 */
template <rocblas_datatype ComputeType = rocblas_datatype_f32_r>
rocblas_status rocblasTgemmEx(rocblas_handle handle, rocblas_operation transa,
                             rocblas_operation transb, std::size_t m,
                             std::size_t n, std::size_t k, const float* alpha,
                             const float* A, std::size_t lda, const float* B,
                             std::size_t ldb, const float* beta, float* C,
                             std::size_t ldc)
{
    return rocblas_gemm_ex(handle, transa, transb, m, n, k, alpha, A,
                           rocblas_datatype_f32_r, lda, B, rocblas_datatype_f32_r,
                           ldb, beta, C, rocblas_datatype_f32_r, ldc, C,
                           rocblas_datatype_f32_r, ldc, ComputeType,
                           rocblas_gemm_algo_standard, 0, 0);
}

/**
 * @brief Template specialization for double type
 */
template <rocblas_datatype ComputeType = rocblas_datatype_f64_r>
rocblas_status rocblasTgemmEx(rocblas_handle handle, rocblas_operation transa,
                             rocblas_operation transb, std::size_t m,
                             std::size_t n, std::size_t k, const double* alpha,
                             const double* A, std::size_t lda, const double* B,
                             std::size_t ldb, const double* beta, double* C,
                             std::size_t ldc)
{
    return rocblas_gemm_ex(handle, transa, transb, m, n, k, alpha, A,
                           rocblas_datatype_f64_r, lda, B, rocblas_datatype_f64_r,
                           ldb, beta, C, rocblas_datatype_f64_r, ldc, C,
                           rocblas_datatype_f64_r, ldc, ComputeType,
                           rocblas_gemm_algo_standard, 0, 0);
}

/**
 * @brief Template specialization for complex<float> type
 */
template <rocblas_datatype ComputeType = rocblas_datatype_f32_c>
rocblas_status rocblasTgemmEx(rocblas_handle handle, rocblas_operation transa,
                             rocblas_operation transb, std::size_t m,
                             std::size_t n, std::size_t k,
                             const std::complex<float>* alpha,
                             const std::complex<float>* A, std::size_t lda,
                             const std::complex<float>* B, std::size_t ldb,
                             const std::complex<float>* beta,
                             std::complex<float>* C, std::size_t ldc)
{
    return rocblas_gemm_ex(handle, transa, transb, m, n, k, alpha, A,
                           rocblas_datatype_f32_c, lda, B, rocblas_datatype_f32_c,
                           ldb, beta, C, rocblas_datatype_f32_c, ldc, C,
                           rocblas_datatype_f32_c, ldc, ComputeType,
                           rocblas_gemm_algo_standard, 0, 0);
}

/**
 * @brief Template specialization for complex<double> type
 */
template <rocblas_datatype ComputeType = rocblas_datatype_f64_c>
rocblas_status rocblasTgemmEx(rocblas_handle handle, rocblas_operation transa,
                             rocblas_operation transb, std::size_t m,
                             std::size_t n, std::size_t k,
                             const std::complex<double>* alpha,
                             const std::complex<double>* A, std::size_t lda,
                             const std::complex<double>* B, std::size_t ldb,
                             const std::complex<double>* beta,
                             std::complex<double>* C, std::size_t ldc)
{
    return rocblas_gemm_ex(handle, transa, transb, m, n, k, alpha, A,
                           rocblas_datatype_f64_c, lda, B, rocblas_datatype_f64_c,
                           ldb, beta, C, rocblas_datatype_f64_c, ldc, C,
                           rocblas_datatype_f64_c, ldc, ComputeType,
                           rocblas_gemm_algo_standard, 0, 0);
}

} // namespace rocblaspp
} // namespace linalg
} // namespace chase