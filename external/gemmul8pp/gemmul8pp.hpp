#pragma once

#include <complex>
#include <cstddef>

#include <cublas_v2.h>

// Declarations only: implementations live in gemmul8.cu (compiled by NVCC).

#ifdef CHASE_ENABLE_OZAKI

namespace chase {
namespace linalg {
namespace gemmul8pp {

template <typename T>
std::size_t workSize(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA = false,
    bool enable_skip_scalB = false,
    std::size_t* workSizeA = nullptr,
    std::size_t* workSizeB = nullptr);

cublasStatus_t gemmul8Tgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const float* alpha,
    const float* A,
    std::size_t lda,
    const float* B,
    std::size_t ldb,
    const float* beta,
    float* C,
    std::size_t ldc,
    unsigned num_moduli,
    bool fastmode,
    void* work,
    void* workA = nullptr,
    void* workB = nullptr,
    bool enable_skip_scalA = false,
    bool enable_skip_scalB = false,
    bool skip_scalA = false,
    bool skip_scalB = false);

cublasStatus_t gemmul8Tgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const double* alpha,
    const double* A,
    std::size_t lda,
    const double* B,
    std::size_t ldb,
    const double* beta,
    double* C,
    std::size_t ldc,
    unsigned num_moduli,
    bool fastmode,
    void* work,
    void* workA = nullptr,
    void* workB = nullptr,
    bool enable_skip_scalA = false,
    bool enable_skip_scalB = false,
    bool skip_scalA = false,
    bool skip_scalB = false);

cublasStatus_t gemmul8Tgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::complex<float>* alpha,
    const std::complex<float>* A,
    std::size_t lda,
    const std::complex<float>* B,
    std::size_t ldb,
    const std::complex<float>* beta,
    std::complex<float>* C,
    std::size_t ldc,
    unsigned num_moduli,
    bool fastmode,
    void* work,
    void* workA = nullptr,
    void* workB = nullptr,
    bool enable_skip_scalA = false,
    bool enable_skip_scalB = false,
    bool skip_scalA = false,
    bool skip_scalB = false);

cublasStatus_t gemmul8Tgemm(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const std::complex<double>* alpha,
    const std::complex<double>* A,
    std::size_t lda,
    const std::complex<double>* B,
    std::size_t ldb,
    const std::complex<double>* beta,
    std::complex<double>* C,
    std::size_t ldc,
    unsigned num_moduli,
    bool fastmode,
    void* work,
    void* workA = nullptr,
    void* workB = nullptr,
    bool enable_skip_scalA = false,
    bool enable_skip_scalB = false,
    bool skip_scalA = false,
    bool skip_scalB = false);

} // namespace gemmul8pp
} // namespace linalg
} // namespace chase

#endif // CHASE_ENABLE_OZAKI
