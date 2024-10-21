#pragma once 

#include <complex>
#include "algorithm/types.hpp"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Macro to check for cuBLAS errors
#define CHECK_CUBLAS_ERROR(val) checkCublas((val), #val, __FILE__, __LINE__)

// Function to check cuBLAS errors
void checkCublas(cublasStatus_t status, const char* const func,
                 const char* const file, const std::size_t line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Runtime Error at: " << file << ":" << line << std::endl;
        switch (status) {
            case CUBLAS_STATUS_NOT_INITIALIZED:
                std::cerr << "CUBLAS_STATUS_NOT_INITIALIZED: The cuBLAS library was not initialized." << std::endl;
                break;
            case CUBLAS_STATUS_ALLOC_FAILED:
                std::cerr << "CUBLAS_STATUS_ALLOC_FAILED: Resource allocation failed." << std::endl;
                break;
            case CUBLAS_STATUS_INVALID_VALUE:
                std::cerr << "CUBLAS_STATUS_INVALID_VALUE: An invalid value was encountered." << std::endl;
                break;
            case CUBLAS_STATUS_ARCH_MISMATCH:
                std::cerr << "CUBLAS_STATUS_ARCH_MISMATCH: The device architecture is not supported." << std::endl;
                break;
            case CUBLAS_STATUS_EXECUTION_FAILED:
                std::cerr << "CUBLAS_STATUS_EXECUTION_FAILED: The execution failed." << std::endl;
                break;
            case CUBLAS_STATUS_INTERNAL_ERROR:
                std::cerr << "CUBLAS_STATUS_INTERNAL_ERROR: An internal error occurred." << std::endl;
                break;
            default:
                std::cerr << "Unknown cuBLAS error." << std::endl;
                break;
        }
        std::cerr << "Function: " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
   
namespace chase
{
namespace linalg
{
namespace cublaspp
{
cublasStatus_t cublasTaxpy(cublasHandle_t handle, std::size_t n, const float* alpha,
                           const float* x, std::size_t incx, float* y, std::size_t incy)
{
    return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, std::size_t n, const double* alpha,
                           const double* x, std::size_t incx, double* y, std::size_t incy)
{
    return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasTaxpy(cublasHandle_t handle, std::size_t n,
                           const std::complex<float>* alpha,
                           const std::complex<float>* x, std::size_t incx,
                           std::complex<float>* y, std::size_t incy)
{
    return cublasCaxpy(handle, n, reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<cuComplex*>(y), incy);
}

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

cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n, const float* x,
                           std::size_t incx, float* result)
{
    return cublasSnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n, const double* x,
                           std::size_t incx, double* result)
{
    return cublasDnrm2(handle, n, x, incx, result);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n,
                           const std::complex<float>* x, std::size_t incx,
                           float* result)
{
    return cublasScnrm2(handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                        result);
}

cublasStatus_t cublasTnrm2(cublasHandle_t handle, std::size_t n,
                           const std::complex<double>* x, std::size_t incx,
                           double* result)
{
    return cublasDznrm2(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                        incx, result);
}

cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n, const float* x,
                          std::size_t incx, const float* y, std::size_t incy, float* result)
{
    return cublasSdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n, const double* x,
                          std::size_t incx, const double* y, std::size_t incy, double* result)
{
    return cublasDdot(handle, n, x, incx, y, incy, result);
}

cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n,
                          const std::complex<float>* x, std::size_t incx,
                          const std::complex<float>* y, std::size_t incy,
                          std::complex<float>* result)
{
    return cublasCdotc(handle, n, reinterpret_cast<const cuComplex*>(x), incx,
                       reinterpret_cast<const cuComplex*>(y), incy,
                       reinterpret_cast<cuComplex*>(result));
}

cublasStatus_t cublasTdot(cublasHandle_t handle, std::size_t n,
                          const std::complex<double>* x, std::size_t incx,
                          const std::complex<double>* y, std::size_t incy,
                          std::complex<double>* result)
{
    return cublasZdotc(handle, n, reinterpret_cast<const cuDoubleComplex*>(x),
                       incx, reinterpret_cast<const cuDoubleComplex*>(y), incy,
                       reinterpret_cast<cuDoubleComplex*>(result));
}

cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n, const float* alpha,
                           float* x, std::size_t incx)
{
    return cublasSscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n, const double* alpha,
                           double* x, std::size_t incx)
{
    return cublasDscal(handle, n, alpha, x, incx);
}

cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n,
                           const std::complex<float>* alpha,
                           std::complex<float>* x, std::size_t incx)
{
    return cublasCscal(handle, n, reinterpret_cast<const cuComplex*>(alpha),
                       reinterpret_cast<cuComplex*>(x), incx);
}

cublasStatus_t cublasTscal(cublasHandle_t handle, std::size_t n,
                           const std::complex<double>* alpha,
                           std::complex<double>* x, std::size_t incx)
{
    return cublasZscal(handle, n,
                       reinterpret_cast<const cuDoubleComplex*>(alpha),
                       reinterpret_cast<cuDoubleComplex*>(x), incx);
}

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n, const float* alpha, const float* A,
                           std::size_t lda, const float* x, std::size_t incx, const float* beta,
                           float* y, std::size_t incy)
{
    return cublasSgemv(handle, transa, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n, const double* alpha, const double* A,
                           std::size_t lda, const double* x, std::size_t incx,
                           const double* beta, double* y, std::size_t incy)
{
    return cublasDgemv(handle, transa, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n, const std::complex<float>* alpha,
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

cublasStatus_t cublasTgemv(cublasHandle_t handle, cublasOperation_t transa,
                           std::size_t m, std::size_t n, const std::complex<double>* alpha,
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

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m, std::size_t n, std::size_t k,
                           const float* alpha, const float* A, std::size_t lda,
                           const float* B, std::size_t ldb, const float* beta, float* C,
                           std::size_t ldc)
{
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m, std::size_t n, std::size_t k,
                           const double* alpha, const double* A, std::size_t lda,
                           const double* B, std::size_t ldb, const double* beta,
                           double* C, std::size_t ldc)
{
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m, std::size_t n, std::size_t k,
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

cublasStatus_t cublasTgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, std::size_t m, std::size_t n, std::size_t k,
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

cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n, std::size_t k,
                             const float* alpha, const float* A, std::size_t lda,
                             const float* beta, float* C, std::size_t ldc)
{

    return cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n, std::size_t k,
                             const double* alpha, const double* A, std::size_t lda,
                             const double* beta, double* C, std::size_t ldc)
{

    return cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n, std::size_t k,
                             const float* alpha, const std::complex<float>* A,
                             std::size_t lda, const float* beta, std::complex<float>* C,
                             std::size_t ldc)
{

    return cublasCherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const cuComplex*>(A), lda, beta,
                       reinterpret_cast<cuComplex*>(C), ldc);
}

cublasStatus_t cublasTsyherk(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, std::size_t n, std::size_t k,
                             const double* alpha, const std::complex<double>* A,
                             std::size_t lda, const double* beta,
                             std::complex<double>* C, std::size_t ldc)
{

    return cublasZherk(handle, uplo, trans, n, k, alpha,
                       reinterpret_cast<const cuDoubleComplex*>(A), lda, beta,
                       reinterpret_cast<cuDoubleComplex*>(C), ldc);
}

/////
cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, std::size_t m, std::size_t n,
                           const float* alpha, const float* A, std::size_t lda,
                           float* B, std::size_t ldb)
{

    return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

cublasStatus_t cublasTtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, std::size_t m, std::size_t n,
                           const double* alpha, const double* A, std::size_t lda,
                           double* B, std::size_t ldb)
{

    return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                       ldb);
}

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


}
}
}