#pragma once 

#include <complex>
#include "algorithm/types.hpp"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "Impl/cuda/nvtx.hpp"
// Macro to check for cuSOLVER errors
#define CHECK_CUSOLVER_ERROR(val) checkCusolver((val), #val, __FILE__, __LINE__)

// Function to check cuSOLVER errors
void checkCusolver(cusolverStatus_t status, const char* const func,
                   const char* const file, const std::size_t line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER Runtime Error at: " << file << ":" << line << std::endl;
        switch (status) {
            case CUSOLVER_STATUS_NOT_INITIALIZED:
                std::cerr << "CUSOLVER_STATUS_NOT_INITIALIZED: The cuSOLVER library was not initialized." << std::endl;
                break;
            case CUSOLVER_STATUS_ALLOC_FAILED:
                std::cerr << "CUSOLVER_STATUS_ALLOC_FAILED: Resource allocation failed." << std::endl;
                break;
            case CUSOLVER_STATUS_INVALID_VALUE:
                std::cerr << "CUSOLVER_STATUS_INVALID_VALUE: An invalid value was encountered." << std::endl;
                break;
            case CUSOLVER_STATUS_ARCH_MISMATCH:
                std::cerr << "CUSOLVER_STATUS_ARCH_MISMATCH: The device architecture is not supported." << std::endl;
                break;
            case CUSOLVER_STATUS_EXECUTION_FAILED:
                std::cerr << "CUSOLVER_STATUS_EXECUTION_FAILED: The execution failed." << std::endl;
                break;
            case CUSOLVER_STATUS_INTERNAL_ERROR:
                std::cerr << "CUSOLVER_STATUS_INTERNAL_ERROR: An internal error occurred." << std::endl;
                break;
            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                std::cerr << "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: The matrix type is not supported." << std::endl;
                break;
            default:
                std::cerr << "Unknown cuSOLVER error." << std::endl;
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
namespace cusolverpp
{

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, float* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, double* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, std::complex<float>* A,
                                             std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCgeqrf_bufferSize(
        handle, m, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, std::complex<double>* A,
                                             std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZgeqrf_bufferSize(
        handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, std::size_t m, std::size_t n,
                                  float* A, std::size_t lda, float* TAU,
                                  float* Workspace, std::size_t Lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, std::size_t m, std::size_t n,
                                  double* A, std::size_t lda, double* TAU,
                                  double* Workspace, std::size_t Lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, std::size_t m, std::size_t n,
                                  std::complex<float>* A, std::size_t lda,
                                  std::complex<float>* TAU,
                                  std::complex<float>* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCgeqrf(handle, m, n, reinterpret_cast<cuComplex*>(A), lda,
                            reinterpret_cast<cuComplex*>(TAU),
                            reinterpret_cast<cuComplex*>(Workspace), Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, std::size_t m, std::size_t n,
                                  std::complex<double>* A, std::size_t lda,
                                  std::complex<double>* TAU,
                                  std::complex<double>* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZgeqrf(handle, m, n, reinterpret_cast<cuDoubleComplex*>(A),
                            lda, reinterpret_cast<cuDoubleComplex*>(TAU),
                            reinterpret_cast<cuDoubleComplex*>(Workspace),
                            Lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                           std::size_t n, std::size_t k, float* A, std::size_t lda,
                                           float* tau, int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                           std::size_t n, std::size_t k, double* A, std::size_t lda,
                                           double* tau, int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                           std::size_t n, std::size_t k, std::complex<float>* A,
                                           std::size_t lda, std::complex<float>* tau,
                                           int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCungqr_bufferSize(
        handle, m, n, k, reinterpret_cast<cuComplex*>(A), lda,
        reinterpret_cast<cuComplex*>(tau), lwork);
}

cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                           std::size_t n, std::size_t k,
                                           std::complex<double>* A, std::size_t lda,
                                           std::complex<double>* tau,
                                           int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZungqr_bufferSize(
        handle, m, n, k, reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(tau), lwork);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, std::size_t m, std::size_t n, std::size_t k,
                                float* A, std::size_t lda, float* tau, float* work,
                                std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, std::size_t m, std::size_t n, std::size_t k,
                                double* A, std::size_t lda, double* tau, double* work,
                                std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, std::size_t m, std::size_t n, std::size_t k,
                                std::complex<float>* A, std::size_t lda,
                                std::complex<float>* tau,
                                std::complex<float>* work, std::size_t lwork,
                                int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCungqr(handle, m, n, k, reinterpret_cast<cuComplex*>(A),
                            lda, reinterpret_cast<cuComplex*>(tau),
                            reinterpret_cast<cuComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, std::size_t m, std::size_t n, std::size_t k,
                                std::complex<double>* A, std::size_t lda,
                                std::complex<double>* tau,
                                std::complex<double>* work, std::size_t lwork,
                                int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZungqr(
        handle, m, n, k, reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(tau),
        reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, std::size_t n,
                                             float* A, std::size_t lda, float* W,
                                             int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, std::size_t n,
                                             double* A, std::size_t lda, double* W,
                                             int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, std::size_t n,
                                             std::complex<float>* A, std::size_t lda,
                                             float* W, int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCheevd_bufferSize(
        handle, jobz, uplo, n, reinterpret_cast<cuComplex*>(A), lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, std::size_t n,
                                             std::complex<double>* A, std::size_t lda,
                                             double* W, int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZheevd_bufferSize(handle, jobz, uplo, n,
                                       reinterpret_cast<cuDoubleComplex*>(A),
                                       lda, W, lwork);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  std::size_t n, float* A, std::size_t lda, float* W,
                                  float* work, std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  std::size_t n, double* A, std::size_t lda, double* W,
                                  double* work, std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  std::size_t n, std::complex<float>* A, std::size_t lda,
                                  float* W, std::complex<float>* work,
                                  std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCheevd(handle, jobz, uplo, n,
                            reinterpret_cast<cuComplex*>(A), lda, W,
                            reinterpret_cast<cuComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  std::size_t n, std::complex<double>* A, std::size_t lda,
                                  double* W, std::complex<double>* work,
                                  std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZheevd(
        handle, jobz, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, W,
        reinterpret_cast<cuDoubleComplex*>(work), lwork, devInfo);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             float* A, std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             double* A, std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             std::complex<float>* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCpotrf_bufferSize(
        handle, uplo, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             std::complex<double>* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZpotrf_bufferSize(
        handle, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, std::size_t n, float* A,
                                  std::size_t lda, float* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, std::size_t n, double* A,
                                  std::size_t lda, double* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, std::size_t n,
                                  std::complex<float>* A, std::size_t lda,
                                  std::complex<float>* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCpotrf(handle, uplo, n, reinterpret_cast<cuComplex*>(A),
                            lda, reinterpret_cast<cuComplex*>(Workspace), Lwork,
                            devInfo);
}

cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, std::size_t n,
                                  std::complex<double>* A, std::size_t lda,
                                  std::complex<double>* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZpotrf(
        handle, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda,
        reinterpret_cast<cuDoubleComplex*>(Workspace), Lwork, devInfo);
}

}
}
}