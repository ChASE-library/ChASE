#pragma once

// Ozaki-II Alg. 2 helpers: μ'/ν', INT8 packs Ā/B̄, INT32 max of C'',
// ||2^{α'}||_2 and max 2^{β'} for the (1,2)-norm error bound.
// Implementations in utilities.cu (NVCC).

#include <complex>
#include <cstddef>
#include <cstdint>

#include <cublas_v2.h>

#ifdef CHASE_ENABLE_OZAKI

namespace chase {
namespace linalg {
namespace gemmul8pp {

inline constexpr int kMaxUFP = 5; // μ' = 5 - floor(log2 α)

// --- scaling factors ---
// Complex α = max(|Re|,|Im|) over the row/col.
// inv_e scales |A| as |A|/|e| for Ā without rewriting A (default: 1).
template <typename T>
void compute_row_sft(T const* A, std::size_t lda, std::size_t m, std::size_t k,
                     std::int16_t* sft, double inv_e, cudaStream_t stream);

template <typename T>
void compute_col_sft(T const* V, std::size_t ldv, std::size_t k, std::size_t n,
                     std::int16_t* sft, cudaStream_t stream);

// --- INT8 packs (given sft) ---
// Real: A_lo / B_lo are int8, column-major.
// Complex: separate Re/Im int8 planes (same lda_lo / ldb_lo).
// inv_e: pack |A|/|e| without rewriting A (default: 1).
template <typename T>
void pack_A_lo(T const* A, std::size_t lda, std::size_t m, std::size_t k,
               std::int16_t const* sftA, std::int8_t* A_lo, std::size_t lda_lo,
               double inv_e, cudaStream_t stream);

template <typename T>
void pack_A_lo(T const* A, std::size_t lda, std::size_t m, std::size_t k,
               std::int16_t const* sftA, std::int8_t* A_lo_re,
               std::int8_t* A_lo_im, std::size_t lda_lo, double inv_e,
               cudaStream_t stream);

template <typename T>
void pack_B_lo(T const* B, std::size_t ldb, std::size_t k, std::size_t n,
               std::int16_t const* sftB, std::int8_t* B_lo, std::size_t ldb_lo,
               cudaStream_t stream);

template <typename T>
void pack_B_lo(T const* B, std::size_t ldb, std::size_t k, std::size_t n,
               std::int16_t const* sftB, std::int8_t* B_lo_re,
               std::int8_t* B_lo_im, std::size_t ldb_lo, cudaStream_t stream);

// --- INT32 max of C'' (Alg. 2 e', f') ---
// Real: one C plane. Complex: two planes; hi = max(C1+C2, C2).
void compute_row_max_C(std::int32_t const* C, std::size_t ldc, std::size_t m,
                       std::size_t n, std::int32_t* amax_row,
                       cudaStream_t stream);

void compute_col_max_C(std::int32_t const* C, std::size_t ldc, std::size_t m,
                       std::size_t n, std::int32_t* amax_col,
                       cudaStream_t stream);

void compute_row_max_C(std::int32_t const* C1, std::int32_t const* C2,
                       std::size_t ldc, std::size_t m, std::size_t n,
                       std::int32_t* amax_row, cudaStream_t stream);

void compute_col_max_C(std::int32_t const* C1, std::int32_t const* C2,
                       std::size_t ldc, std::size_t m, std::size_t n,
                       std::int32_t* amax_col, cudaStream_t stream);

// --- Alg. 2 primes: 2^{α'} with α' = α + e'/2, α = kMaxUFP - sft ---
// Fused kernel writes 2^{α'_i} (always > 0) so cuBLAS iamax is valid.
// Same for β' via sftB / amax_col. Real and complex share this path.
// stream is the cuBLAS handle stream so the following reduction sees this write.
void compute_ozaki_two_pow_primes(std::int16_t const* sft,
                                  std::int32_t const* amax_C, std::size_t n,
                                  double* two_pow, cudaStream_t stream);

// Fill two_pow, then cublasIdamax → max 2^{α'} (or max 2^{β'}).
double max_two_pow_ozaki_prime(cublasHandle_t handle, std::int16_t const* sft,
                               std::int32_t const* amax_C, std::size_t n,
                               double* d_two_pow);

// Fill two_pow, then cublasDnrm2 → ||2^{α'}||_2 (or ||2^{β'}||_2).
double nrm2_two_pow_ozaki_prime(cublasHandle_t handle, std::int16_t const* sft,
                                std::int32_t const* amax_C, std::size_t n,
                                double* d_two_pow);

// Bytes required inside `work` for ozaki_compute_scaling_factors_error_bound.
template <typename T>
std::size_t ozaki_scaling_factors_error_bound_work_bytes(std::size_t m,
                                                         std::size_t n,
                                                         std::size_t k);

// Alg. 2 probe on Ā = A/e (via inv_e) and B → ||2^{α'}||_2, max 2^{β'}.
// Carves scratch from `work` (lda_lo = ldb_lo = k, ldc = m).
template <typename T>
void ozaki_compute_scaling_factors_error_bound(
    cublasHandle_t handle, T const* A, std::size_t lda, std::size_t m,
    std::size_t k, T const* B, std::size_t ldb, std::size_t n, void* work,
    std::size_t work_bytes, double* h_nrm2_a, double* h_max2_b,
    double inv_e = 1.0);

} // namespace gemmul8pp
} // namespace linalg
} // namespace chase

#endif // CHASE_ENABLE_OZAKI
