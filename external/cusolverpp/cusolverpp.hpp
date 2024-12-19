// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once 

#include <complex>
#include "algorithm/types.hpp"
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "Impl/chase_gpu/nvtx.hpp"
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

/**
 * @page cusolverpp_namespace chase::linalg::cusolverpp Namespace
 * @brief The namespace `chase::linalg::cusolverpp` contains a templated C++ interface to the cuSOLVER library.
 *
 * This namespace contains templated functions that interface with the cuSOLVER library to provide efficient linear algebra and decomposition routines on GPUs.
 * The functions are templated to work with data types such as `float`, `double`, and `std::complex`, enabling efficient numerical computations in scientific applications.
 */

/**
 * @defgroup cuSolverFunctions cuSOLVER Routines
 * @brief Template functions that interface with cuSOLVER routines.
 * 
 * These functions provide common matrix decomposition and eigenvalue routines such as QR factorizations, Cholesky factorizations, and eigenvalue decompositions.
 * They support operations on both real and complex matrices.
 */

namespace chase
{
namespace linalg
{
namespace cusolverpp
{
/**
 * @ingroup cuSolverFunctions
 * @brief Computes the workspace size for QR factorization of a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param Lwork Pointer to the workspace size (output)
 * @return cusolverStatus_t Status of the cuSolver routine
 */
cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, float* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
/**
 * @ingroup cuSolverFunctions
 * @brief Computes the workspace size for QR factorization of a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param Lwork Pointer to the workspace size (output)
 * @return cusolverStatus_t Status of the cuSolver routine
 */
cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, double* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
/**
 * @ingroup cuSolverFunctions
 * @brief Computes the workspace size for QR factorization of a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param Lwork Pointer to the workspace size (output)
 * @return cusolverStatus_t Status of the cuSolver routine
 */
cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, std::complex<float>* A,
                                             std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCgeqrf_bufferSize(
        handle, m, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
}
/**
 * @ingroup cuSolverFunctions
 * @brief Computes the workspace size for QR factorization of a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param Lwork Pointer to the workspace size (output)
 * @return cusolverStatus_t Status of the cuSolver routine
 */
cusolverStatus_t cusolverDnTgeqrf_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                             std::size_t n, std::complex<double>* A,
                                             std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZgeqrf_bufferSize(
        handle, m, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
}
/**
 * @ingroup cuSolverFunctions
 * @brief Performs QR factorization on a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param TAU Pointer to the output vector of scalar factors of the elementary reflectors
 * @param Workspace Pointer to the workspace array
 * @param Lwork Size of the workspace array
 * @param devInfo Pointer to the device-side status code
 * @return cusolverStatus_t Status of the cuSolver routine
 */
cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, std::size_t m, std::size_t n,
                                  float* A, std::size_t lda, float* TAU,
                                  float* Workspace, std::size_t Lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork,
                            devInfo);
}
/**
 * @ingroup cuSolverFunctions
 * @brief Performs QR factorization on a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param TAU Pointer to the output vector of scalar factors of the elementary reflectors
 * @param Workspace Pointer to the workspace array
 * @param Lwork Size of the workspace array
 * @param devInfo Pointer to the device-side status code
 * @return cusolverStatus_t Status of the cuSolver routine
 */
cusolverStatus_t cusolverDnTgeqrf(cusolverDnHandle_t handle, std::size_t m, std::size_t n,
                                  double* A, std::size_t lda, double* TAU,
                                  double* Workspace, std::size_t Lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork,
                            devInfo);
}
/**
 * @ingroup cuSolverFunctions
 * @brief Performs QR factorization on a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param TAU Pointer to the output vector of scalar factors of the elementary reflectors
 * @param Workspace Pointer to the workspace array
 * @param Lwork Size of the workspace array
 * @param devInfo Pointer to the device-side status code
 * @return cusolverStatus_t Status of the cuSolver routine
 */
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
/**
 * @ingroup cuSolverFunctions
 * @brief Performs QR factorization on a general m-by-n matrix (single-precision).
 * 
 * @param handle cuSolver handle
 * @param m Number of rows of the matrix A
 * @param n Number of columns of the matrix A
 * @param A Pointer to the matrix elements
 * @param lda Leading dimension of the matrix A
 * @param TAU Pointer to the output vector of scalar factors of the elementary reflectors
 * @param Workspace Pointer to the workspace array
 * @param Lwork Size of the workspace array
 * @param devInfo Pointer to the device-side status code
 * @return cusolverStatus_t Status of the cuSolver routine
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for generating the unitary matrix Q 
 *        from the QR factorization of a matrix with single-precision floating-point values.
 * 
 * This function calculates the workspace size required for the `cusolverDnSorgqr` 
 * function, which generates the unitary matrix Q from the QR decomposition.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in] Pointer to the matrix A, the input matrix.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param lwork [out] Pointer to an integer where the size of the workspace is returned.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                           std::size_t n, std::size_t k, float* A, std::size_t lda,
                                           float* tau, int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for generating the unitary matrix Q 
 *        from the QR factorization of a matrix with single-precision floating-point values.
 * 
 * This function calculates the workspace size required for the `cusolverDnSorgqr` 
 * function, which generates the unitary matrix Q from the QR decomposition.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in] Pointer to the matrix A, the input matrix.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param lwork [out] Pointer to an integer where the size of the workspace is returned.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
cusolverStatus_t cusolverDnTgqr_bufferSize(cusolverDnHandle_t handle, std::size_t m,
                                           std::size_t n, std::size_t k, double* A, std::size_t lda,
                                           double* tau, int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for generating the unitary matrix Q 
 *        from the QR factorization of a matrix with single-precision floating-point values.
 * 
 * This function calculates the workspace size required for the `cusolverDnSorgqr` 
 * function, which generates the unitary matrix Q from the QR decomposition.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in] Pointer to the matrix A, the input matrix.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param lwork [out] Pointer to an integer where the size of the workspace is returned.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for generating the unitary matrix Q 
 *        from the QR factorization of a matrix with single-precision floating-point values.
 * 
 * This function calculates the workspace size required for the `cusolverDnSorgqr` 
 * function, which generates the unitary matrix Q from the QR decomposition.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in] Pointer to the matrix A, the input matrix.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param lwork [out] Pointer to an integer where the size of the workspace is returned.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Generates the unitary matrix Q from the QR factorization with single-precision floating-point values.
 * 
 * This function uses the `cusolverDnSorgqr` function to generate the unitary matrix Q from 
 * the QR decomposition of matrix A.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in,out] Pointer to the matrix A, which will be overwritten with the matrix Q.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param work [in] Pointer to the workspace array.
 * @param lwork [in] The size of the workspace.
 * @param devInfo [out] Pointer to an integer that returns the status of the operation.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, std::size_t m, std::size_t n, std::size_t k,
                                float* A, std::size_t lda, float* tau, float* work,
                                std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Generates the unitary matrix Q from the QR factorization with single-precision floating-point values.
 * 
 * This function uses the `cusolverDnSorgqr` function to generate the unitary matrix Q from 
 * the QR decomposition of matrix A.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in,out] Pointer to the matrix A, which will be overwritten with the matrix Q.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param work [in] Pointer to the workspace array.
 * @param lwork [in] The size of the workspace.
 * @param devInfo [out] Pointer to an integer that returns the status of the operation.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
cusolverStatus_t cusolverDnTgqr(cusolverDnHandle_t handle, std::size_t m, std::size_t n, std::size_t k,
                                double* A, std::size_t lda, double* tau, double* work,
                                std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Generates the unitary matrix Q from the QR factorization with single-precision floating-point values.
 * 
 * This function uses the `cusolverDnSorgqr` function to generate the unitary matrix Q from 
 * the QR decomposition of matrix A.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in,out] Pointer to the matrix A, which will be overwritten with the matrix Q.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param work [in] Pointer to the workspace array.
 * @param lwork [in] The size of the workspace.
 * @param devInfo [out] Pointer to an integer that returns the status of the operation.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Generates the unitary matrix Q from the QR factorization with single-precision floating-point values.
 * 
 * This function uses the `cusolverDnSorgqr` function to generate the unitary matrix Q from 
 * the QR decomposition of matrix A.
 * 
 * @param handle [in] cuSOLVER handle to the context.
 * @param m [in] The number of rows in the matrix A.
 * @param n [in] The number of columns in the matrix A.
 * @param k [in] The number of elementary reflectors, or the rank of the matrix A.
 * @param A [in,out] Pointer to the matrix A, which will be overwritten with the matrix Q.
 * @param lda [in] Leading dimension of the matrix A.
 * @param tau [in] Array of size k containing the scalar factors of the elementary reflectors.
 * @param work [in] Pointer to the workspace array.
 * @param lwork [in] The size of the workspace.
 * @param devInfo [out] Pointer to an integer that returns the status of the operation.
 * 
 * @return cusolverStatus_t Returns the status of the operation.
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Get the buffer size for real double precision eigenvalue/eigenvector decomposition.
 *
 * This function queries the required buffer size for performing an eigenvalue decomposition of a
 * real symmetric matrix A, storing the eigenvalues in W, and the eigenvectors in Q (if requested),
 * using the cuSolver library with double precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param lwork [out] Pointer to the integer value storing the required buffer size.
 * @return cusolverStatus_t status of the function call.
 */
cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, std::size_t n,
                                             float* A, std::size_t lda, float* W,
                                             int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Get the buffer size for real double precision eigenvalue/eigenvector decomposition.
 *
 * This function queries the required buffer size for performing an eigenvalue decomposition of a
 * real symmetric matrix A, storing the eigenvalues in W, and the eigenvectors in Q (if requested),
 * using the cuSolver library with double precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param lwork [out] Pointer to the integer value storing the required buffer size.
 * @return cusolverStatus_t status of the function call.
 */
cusolverStatus_t cusolverDnTheevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, std::size_t n,
                                             double* A, std::size_t lda, double* W,
                                             int* lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Get the buffer size for real double precision eigenvalue/eigenvector decomposition.
 *
 * This function queries the required buffer size for performing an eigenvalue decomposition of a
 * real symmetric matrix A, storing the eigenvalues in W, and the eigenvectors in Q (if requested),
 * using the cuSolver library with double precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param lwork [out] Pointer to the integer value storing the required buffer size.
 * @return cusolverStatus_t status of the function call.
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Get the buffer size for real double precision eigenvalue/eigenvector decomposition.
 *
 * This function queries the required buffer size for performing an eigenvalue decomposition of a
 * real symmetric matrix A, storing the eigenvalues in W, and the eigenvectors in Q (if requested),
 * using the cuSolver library with double precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param lwork [out] Pointer to the integer value storing the required buffer size.
 * @return cusolverStatus_t status of the function call.
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Compute the eigenvalues and eigenvectors of a real single precision symmetric matrix.
 *
 * This function performs eigenvalue decomposition of a symmetric matrix A, computing the eigenvalues
 * in W and the eigenvectors in Q (if requested), using the cuSolver library with single precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in/out] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param work [out] Pointer to the workspace array.
 * @param lwork [in] Size of the workspace array.
 * @param devInfo [out] Pointer to the device info integer for status.
 * @return cusolverStatus_t status of the function call.
 */
cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  std::size_t n, float* A, std::size_t lda, float* W,
                                  float* work, std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                            devInfo);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Compute the eigenvalues and eigenvectors of a real single precision symmetric matrix.
 *
 * This function performs eigenvalue decomposition of a symmetric matrix A, computing the eigenvalues
 * in W and the eigenvectors in Q (if requested), using the cuSolver library with single precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in/out] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param work [out] Pointer to the workspace array.
 * @param lwork [in] Size of the workspace array.
 * @param devInfo [out] Pointer to the device info integer for status.
 * @return cusolverStatus_t status of the function call.
 */
cusolverStatus_t cusolverDnTheevd(cusolverDnHandle_t handle,
                                  cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  std::size_t n, double* A, std::size_t lda, double* W,
                                  double* work, std::size_t lwork, int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork,
                            devInfo);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Compute the eigenvalues and eigenvectors of a real single precision symmetric matrix.
 *
 * This function performs eigenvalue decomposition of a symmetric matrix A, computing the eigenvalues
 * in W and the eigenvectors in Q (if requested), using the cuSolver library with single precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in/out] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param work [out] Pointer to the workspace array.
 * @param lwork [in] Size of the workspace array.
 * @param devInfo [out] Pointer to the device info integer for status.
 * @return cusolverStatus_t status of the function call.
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Compute the eigenvalues and eigenvectors of a real single precision symmetric matrix.
 *
 * This function performs eigenvalue decomposition of a symmetric matrix A, computing the eigenvalues
 * in W and the eigenvectors in Q (if requested), using the cuSolver library with single precision.
 *
 * @param handle [in] cuSolver handle.
 * @param jobz [in] Eigenvalue problem mode (whether to compute eigenvectors or not).
 * @param uplo [in] Specifies whether the matrix is upper or lower triangular.
 * @param n [in] The order of the matrix A (number of rows/columns).
 * @param A [in/out] Pointer to the matrix A.
 * @param lda [in] Leading dimension of matrix A.
 * @param W [out] Pointer to the array for storing eigenvalues.
 * @param work [out] Pointer to the workspace array.
 * @param lwork [in] Size of the workspace array.
 * @param devInfo [out] Pointer to the device info integer for status.
 * @return cusolverStatus_t status of the function call.
 */
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
/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf_bufferSize`.
 * It computes the size of the buffer needed to perform a Cholesky factorization for
 * a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param Lwork The size of the work array needed for the factorization
 * 
 * @return cusolverStatus_t Status of the operation
 */
cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             float* A, std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf_bufferSize`.
 * It computes the size of the buffer needed to perform a Cholesky factorization for
 * a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param Lwork The size of the work array needed for the factorization
 * 
 * @return cusolverStatus_t Status of the operation
 */
cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             double* A, std::size_t lda, int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf_bufferSize`.
 * It computes the size of the buffer needed to perform a Cholesky factorization for
 * a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param Lwork The size of the work array needed for the factorization
 * 
 * @return cusolverStatus_t Status of the operation
 */
cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             std::complex<float>* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnCpotrf_bufferSize(
        handle, uplo, n, reinterpret_cast<cuComplex*>(A), lda, Lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Computes the optimal buffer size for the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf_bufferSize`.
 * It computes the size of the buffer needed to perform a Cholesky factorization for
 * a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param Lwork The size of the work array needed for the factorization
 * 
 * @return cusolverStatus_t Status of the operation
 */
cusolverStatus_t cusolverDnTpotrf_bufferSize(cusolverDnHandle_t handle,
                                             cublasFillMode_t uplo, std::size_t n,
                                             std::complex<double>* A, std::size_t lda,
                                             int* Lwork)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnZpotrf_bufferSize(
        handle, uplo, n, reinterpret_cast<cuDoubleComplex*>(A), lda, Lwork);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Performs the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf`.
 * It performs the Cholesky factorization of a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param devInfo Output variable to indicate the status of the factorization
 *
 * @return cusolverStatus_t Status of the operation
 */
cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, std::size_t n, float* A,
                                  std::size_t lda, float* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Performs the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf`.
 * It performs the Cholesky factorization of a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param devInfo Output variable to indicate the status of the factorization
 *
 * @return cusolverStatus_t Status of the operation
 */
cusolverStatus_t cusolverDnTpotrf(cusolverDnHandle_t handle,
                                  cublasFillMode_t uplo, std::size_t n, double* A,
                                  std::size_t lda, double* Workspace, std::size_t Lwork,
                                  int* devInfo)
{
    SCOPED_NVTX_RANGE();
    return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

/**
 * @ingroup cuSolverFunctions
 * @brief Performs the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf`.
 * It performs the Cholesky factorization of a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param devInfo Output variable to indicate the status of the factorization
 *
 * @return cusolverStatus_t Status of the operation
 */
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

/**
 * @ingroup cuSolverFunctions
 * @brief Performs the Cholesky factorization of a matrix.
 *
 * This function is a wrapper around the cuSolver function `cusolverDnSpotrf`.
 * It performs the Cholesky factorization of a matrix with real single-precision elements.
 *
 * @param handle cuSolver handle
 * @param uplo Specifies whether the upper or lower triangular part of the matrix is stored
 * @param n The order of the matrix A (number of rows/columns)
 * @param A Matrix A
 * @param lda Leading dimension of A
 * @param devInfo Output variable to indicate the status of the factorization
 *
 * @return cusolverStatus_t Status of the operation
 */
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