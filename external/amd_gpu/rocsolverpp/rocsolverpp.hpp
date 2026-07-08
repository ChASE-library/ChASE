// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "algorithm/types.hpp"

#include <complex>
#include <cstdlib>
#include <iostream>

#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

// Macro to check for rocSOLVER errors
#define CHECK_ROCSOLVER_ERROR(val) checkRocsolver((val), #val, __FILE__, __LINE__)

inline void checkRocsolver(rocblas_status status, const char* const func,
                             const char* const file, std::size_t line)
{
    if (status != rocblas_status_success)
    {
        std::cerr << "rocSOLVER Runtime Error at: " << file << ":" << line
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
 * @page rocsolverpp_namespace chase::linalg::rocsolverpp Namespace
 * @brief Templated C++ dispatch to rocSOLVER (HIP analogue of cusolverpp).
 *
 * rocSOLVER does not expose cuSOLVER-style `*_bufferSize` APIs for
 * caller-owned device workspace. Workspace sizing and binding live in
 * `gpu_backend::solver<hip_backend>` (rocBLAS device-memory query +
 * `rocblas_set_workspace`). This header provides typed routine dispatch only.
 */

namespace chase
{
namespace linalg
{
namespace rocsolverpp
{

template <typename T>
rocblas_status rocsolverTheevd(rocblas_handle handle, rocblas_evect evect,
                               rocblas_fill uplo, rocblas_int n, T* A,
                               rocblas_int lda, chase::Base<T>* D,
                               chase::Base<T>* E, rocblas_int* info);

template <>
inline rocblas_status rocsolverTheevd<float>(rocblas_handle handle,
                                             rocblas_evect evect,
                                             rocblas_fill uplo, rocblas_int n,
                                             float* A, rocblas_int lda, float* D,
                                             float* E, rocblas_int* info)
{
    return rocsolver_ssyevd(handle, evect, uplo, n, A, lda, D, E, info);
}

template <>
inline rocblas_status rocsolverTheevd<double>(rocblas_handle handle,
                                              rocblas_evect evect,
                                              rocblas_fill uplo, rocblas_int n,
                                              double* A, rocblas_int lda,
                                              double* D, double* E,
                                              rocblas_int* info)
{
    return rocsolver_dsyevd(handle, evect, uplo, n, A, lda, D, E, info);
}

template <>
inline rocblas_status rocsolverTheevd<std::complex<float>>(
    rocblas_handle handle, rocblas_evect evect, rocblas_fill uplo,
    rocblas_int n, std::complex<float>* A, rocblas_int lda, float* D,
    float* E, rocblas_int* info)
{
    return rocsolver_cheevd(handle, evect, uplo, n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            D, E, info);
}

template <>
inline rocblas_status rocsolverTheevd<std::complex<double>>(
    rocblas_handle handle, rocblas_evect evect, rocblas_fill uplo,
    rocblas_int n, std::complex<double>* A, rocblas_int lda, double* D,
    double* E, rocblas_int* info)
{
    return rocsolver_zheevd(handle, evect, uplo, n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            D, E, info);
}

template <typename T>
rocblas_status rocsolverTpotrf(rocblas_handle handle, rocblas_fill uplo,
                               rocblas_int n, T* A, rocblas_int lda,
                               rocblas_int* info);

template <>
inline rocblas_status rocsolverTpotrf<float>(rocblas_handle handle,
                                             rocblas_fill uplo, rocblas_int n,
                                             float* A, rocblas_int lda,
                                             rocblas_int* info)
{
    return rocsolver_spotrf(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolverTpotrf<double>(rocblas_handle handle,
                                              rocblas_fill uplo, rocblas_int n,
                                              double* A, rocblas_int lda,
                                              rocblas_int* info)
{
    return rocsolver_dpotrf(handle, uplo, n, A, lda, info);
}

template <>
inline rocblas_status rocsolverTpotrf<std::complex<float>>(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    std::complex<float>* A, rocblas_int lda, rocblas_int* info)
{
    return rocsolver_cpotrf(handle, uplo, n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            info);
}

template <>
inline rocblas_status rocsolverTpotrf<std::complex<double>>(
    rocblas_handle handle, rocblas_fill uplo, rocblas_int n,
    std::complex<double>* A, rocblas_int lda, rocblas_int* info)
{
    return rocsolver_zpotrf(handle, uplo, n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            info);
}

template <typename T>
rocblas_status rocsolverTgeqrf(rocblas_handle handle, rocblas_int m,
                               rocblas_int n, T* A, rocblas_int lda, T* tau);

template <>
inline rocblas_status rocsolverTgeqrf<float>(rocblas_handle handle,
                                             rocblas_int m, rocblas_int n,
                                             float* A, rocblas_int lda,
                                             float* tau)
{
    return rocsolver_sgeqrf(handle, m, n, A, lda, tau);
}

template <>
inline rocblas_status rocsolverTgeqrf<double>(rocblas_handle handle,
                                              rocblas_int m, rocblas_int n,
                                              double* A, rocblas_int lda,
                                              double* tau)
{
    return rocsolver_dgeqrf(handle, m, n, A, lda, tau);
}

template <>
inline rocblas_status rocsolverTgeqrf<std::complex<float>>(
    rocblas_handle handle, rocblas_int m, rocblas_int n,
    std::complex<float>* A, rocblas_int lda, std::complex<float>* tau)
{
    return rocsolver_cgeqrf(handle, m, n,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            reinterpret_cast<rocblas_float_complex*>(tau));
}

template <>
inline rocblas_status rocsolverTgeqrf<std::complex<double>>(
    rocblas_handle handle, rocblas_int m, rocblas_int n,
    std::complex<double>* A, rocblas_int lda, std::complex<double>* tau)
{
    return rocsolver_zgeqrf(handle, m, n,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            reinterpret_cast<rocblas_double_complex*>(tau));
}

template <typename T>
rocblas_status rocsolverTorgqr(rocblas_handle handle, rocblas_int m,
                               rocblas_int n, rocblas_int k, T* A,
                               rocblas_int lda, T* tau);

template <>
inline rocblas_status rocsolverTorgqr<float>(rocblas_handle handle,
                                             rocblas_int m, rocblas_int n,
                                             rocblas_int k, float* A,
                                             rocblas_int lda, float* tau)
{
    return rocsolver_sorgqr(handle, m, n, k, A, lda, tau);
}

template <>
inline rocblas_status rocsolverTorgqr<double>(rocblas_handle handle,
                                              rocblas_int m, rocblas_int n,
                                              rocblas_int k, double* A,
                                              rocblas_int lda, double* tau)
{
    return rocsolver_dorgqr(handle, m, n, k, A, lda, tau);
}

template <>
inline rocblas_status rocsolverTorgqr<std::complex<float>>(
    rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k,
    std::complex<float>* A, rocblas_int lda, std::complex<float>* tau)
{
    return rocsolver_cungqr(handle, m, n, k,
                            reinterpret_cast<rocblas_float_complex*>(A), lda,
                            reinterpret_cast<rocblas_float_complex*>(tau));
}

template <>
inline rocblas_status rocsolverTorgqr<std::complex<double>>(
    rocblas_handle handle, rocblas_int m, rocblas_int n, rocblas_int k,
    std::complex<double>* A, rocblas_int lda, std::complex<double>* tau)
{
    return rocsolver_zungqr(handle, m, n, k,
                            reinterpret_cast<rocblas_double_complex*>(A), lda,
                            reinterpret_cast<rocblas_double_complex*>(tau));
}

} // namespace rocsolverpp
} // namespace linalg
} // namespace chase
