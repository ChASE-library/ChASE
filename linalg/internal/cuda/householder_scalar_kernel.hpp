// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda
{

// Implemented in householder_scalar_kernel.cu (compiled with nvcc).
// Fused Householder scalars: x0, nrm_sq -> tau, inv_denom, neg_beta, denom_bcast, saved_rkk.
template <typename T, typename RealT>
void run_householder_scalar_kernel(cudaStream_t stream, int pivot_here,
                                  const T* d_x0, const RealT* d_nrm_sq, T* d_tau,
                                  T* d_inv_denom, T* d_neg_beta, T* d_denom_bcast,
                                  T* d_saved_rkk);

// After Allreduce(denom_bcast): d_inv_denom = 1/d_denom_bcast (or 0 if zero).
template <typename T>
void run_inv_denom_from_denom_bcast(cudaStream_t stream,
                                    const T* d_denom_bcast, T* d_inv_denom);

// If denom_bcast != 0: set inv_denom = 1/denom_bcast and scale d_V_col[0..n-1].
// No-op when denom_bcast == 0. Avoids D2H for panel scal phase.
template <typename T>
void run_nonpivot_scal_if_denom_nonzero(cudaStream_t stream,
                                       const T* d_denom_bcast, T* d_inv_denom,
                                       T* d_V_col, int n);

// Scale V_col with inv_denom only when tau != 0. On pivot row write neg_beta.
template <typename T>
void run_guarded_scaling(cudaStream_t stream, int n, const T* d_tau,
                         const T* d_inv_denom, T* d_V_col, bool pivot_here,
                         int pivot_loc, const T* d_neg_beta);

// Batch save/restore upper-triangular panel entries for blocked Householder QR.
template <typename T>
void run_batch_save_restore_upper_triangular(cudaStream_t stream, T* d_V,
                                             T* d_saved, std::size_t ldv,
                                             std::size_t jb, std::size_t k,
                                             std::size_t g_off,
                                             std::size_t l_rows,
                                             const T* d_one, const T* d_zero,
                                             bool save_mode);

// Initialize the distributed local tile as the identity matrix.
template <typename T>
void run_init_identity_distributed(cudaStream_t stream, T* d_V, std::size_t ldv,
                                   std::size_t n, std::size_t g_off,
                                   std::size_t l_rows);

// Compact WY T-block from S = V^H V and tau (GPU-resident, no per-column gemv).
// Tb: output nb x nb column-major; d_S: jb x jb column-major (overwritten temp);
// d_tau: length jb (panel taus).
template <typename T>
void run_compute_T_block(cudaStream_t stream, T* Tb, T* d_S, const T* d_tau,
                        int jb, int nb);

// In-place WY cleaning for one panel column: zero rows with global index <
// pivot; at pivot copy *d_saved_rkk to d_r_diag_out and set v to 1 (if
// d_r_diag_out == nullptr, skip peel). d_V_col0 = V(0, col).
template <typename T>
void run_split_and_pad_v_column(cudaStream_t stream, T* d_V_col0,
                                const std::uint64_t* d_row_global, int l_rows,
                                std::uint64_t pivot_global, int pivot_here,
                                int pivot_loc, const T* d_saved_rkk,
                                T* d_r_diag_out);

// Before panel factor column loop: for V(:, k..k+jb-1), zero local rows with
// global row < k only (no R peel, pivot unchanged). d_V_panel = V + k*ldv.
template <typename T>
void run_panel_pre_clean(cudaStream_t stream, T* d_V_panel, std::size_t ldv,
                         const std::uint64_t* d_row_global, int l_rows,
                         std::size_t k_col0, int jb_cols);

} // namespace cuda
} // namespace internal
} // namespace linalg
} // namespace chase
