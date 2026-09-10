// Ozaki Scheme II (GEMMul8): implementations compiled by NVCC only.

#include <gemmul8.hpp>

#include <complex>
#include <cstddef>
#include <type_traits>

#include <cublas_v2.h>
#include <cuComplex.h>
#include <iostream>

namespace chase {
namespace linalg {
namespace gemmul8pp {

template <typename T>
std::size_t workSize(
    std::size_t m,
    std::size_t n,
    std::size_t k,
    unsigned num_moduli,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    std::size_t* workSizeA,
    std::size_t* workSizeB) {
  constexpr bool is_complex_v =
      std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;
  return gemmul8::workSize<is_complex_v, gemmul8::Backend::INT8>(
      m, n, k, num_moduli,
      enable_skip_scalA, enable_skip_scalB,
      workSizeA, workSizeB);
}

template std::size_t workSize<float>(
    std::size_t, std::size_t, std::size_t, unsigned,
    bool, bool, std::size_t*, std::size_t*);
template std::size_t workSize<double>(
    std::size_t, std::size_t, std::size_t, unsigned,
    bool, bool, std::size_t*, std::size_t*);
template std::size_t workSize<std::complex<float>>(
    std::size_t, std::size_t, std::size_t, unsigned,
    bool, bool, std::size_t*, std::size_t*);
template std::size_t workSize<std::complex<double>>(
    std::size_t, std::size_t, std::size_t, unsigned,
    bool, bool, std::size_t*, std::size_t*);

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
    void* workA,
    void* workB,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    bool skip_scalA,
    bool skip_scalB) {
  (void)gemmul8::gemm<float, gemmul8::Backend::INT8>(
      handle, transa, transb, m, n, k,
      alpha, A, lda, B, ldb, beta, C, ldc,
      num_moduli, fastmode, work, workA, workB,
      enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB);
  return CUBLAS_STATUS_SUCCESS;
}

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
    void* workA,
    void* workB,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    bool skip_scalA,
    bool skip_scalB) {
  (void)gemmul8::gemm<double, gemmul8::Backend::INT8>(
      handle, transa, transb, m, n, k,
      alpha, A, lda, B, ldb, beta, C, ldc,
      num_moduli, fastmode, work, workA, workB,
      enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB);
  return CUBLAS_STATUS_SUCCESS;
}

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
    void* workA,
    void* workB,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    bool skip_scalA,
    bool skip_scalB) {
  const auto* alpha_c = reinterpret_cast<const cuComplex*>(alpha);
  const auto* beta_c  = reinterpret_cast<const cuComplex*>(beta);
  const auto* A_c     = reinterpret_cast<const cuComplex*>(A);
  const auto* B_c     = reinterpret_cast<const cuComplex*>(B);
  auto* C_c           = reinterpret_cast<cuComplex*>(C);
  (void)gemmul8::gemm<cuComplex, gemmul8::Backend::INT8>(
      handle, transa, transb, m, n, k,
      alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc,
      num_moduli, fastmode, work, workA, workB,
      enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB);
  return CUBLAS_STATUS_SUCCESS;
}

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
    void* workA,
    void* workB,
    bool enable_skip_scalA,
    bool enable_skip_scalB,
    bool skip_scalA,
    bool skip_scalB) {
  const auto* alpha_c = reinterpret_cast<const cuDoubleComplex*>(alpha);
  const auto* beta_c  = reinterpret_cast<const cuDoubleComplex*>(beta);
  const auto* A_c     = reinterpret_cast<const cuDoubleComplex*>(A);
  const auto* B_c     = reinterpret_cast<const cuDoubleComplex*>(B);
  auto* C_c           = reinterpret_cast<cuDoubleComplex*>(C);

  (void)gemmul8::gemm<cuDoubleComplex, gemmul8::Backend::INT8>(
      handle, transa, transb, m, n, k,
      alpha_c, A_c, lda, B_c, ldb, beta_c, C_c, ldc,
      num_moduli, fastmode, work, workA, workB,
      enable_skip_scalA, enable_skip_scalB, skip_scalA, skip_scalB);
  return CUBLAS_STATUS_SUCCESS;
}

} // namespace gemmul8pp
} // namespace linalg
} // namespace chase
