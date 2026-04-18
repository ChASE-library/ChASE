// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <complex>
#include <cuda_runtime.h>
#include <cuComplex.h>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace cuda_nccl_ph_diag
{

/** Local sum of |B_ij|^2 over the tile intersecting the global upper-right block.
 *  All overloads expect \p dA in LAPACK/BLAS **column-major** order:
 *  local row \c i, local column \c j → <tt>dA[i + j*ld]</tt> with \c ld >= lrows. */
void chase_ph_diag_B_fro_sq_float(const float* dA, std::size_t ld,
                                   std::size_t lrows, std::size_t lcols,
                                   std::size_t g_row_off, std::size_t g_col_off,
                                   std::size_t half_m, std::size_t N,
                                   cudaStream_t stream, double* h_sum_sq);
void chase_ph_diag_B_fro_sq_double(const double* dA, std::size_t ld,
                                   std::size_t lrows, std::size_t lcols,
                                   std::size_t g_row_off, std::size_t g_col_off,
                                   std::size_t half_m, std::size_t N,
                                   cudaStream_t stream, double* h_sum_sq);
void chase_ph_diag_B_fro_sq_complex_float(
    const std::complex<float>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t half_m, std::size_t N, cudaStream_t stream, double* h_sum_sq);
void chase_ph_diag_B_fro_sq_complex_double(
    const std::complex<double>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t half_m, std::size_t N, cudaStream_t stream, double* h_sum_sq);

/** Local min of |A_{kk}| over entries of the local tile on the global diagonal
 *  (gi==gj). \p dA column-major: \c dA[i+j*ld]. Writes \c +inf if none. */
void chase_ph_diag_min_abs_diag_float(const float* dA, std::size_t ld,
                                      std::size_t lrows, std::size_t lcols,
                                      std::size_t g_row_off, std::size_t g_col_off,
                                      std::size_t N, cudaStream_t stream,
                                      double* h_min_abs);
void chase_ph_diag_min_abs_diag_double(const double* dA, std::size_t ld,
                                       std::size_t lrows, std::size_t lcols,
                                       std::size_t g_row_off, std::size_t g_col_off,
                                       std::size_t N, cudaStream_t stream,
                                       double* h_min_abs);
void chase_ph_diag_min_abs_diag_complex_float(
    const std::complex<float>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t N, cudaStream_t stream, double* h_min_abs);
void chase_ph_diag_min_abs_diag_complex_double(
    const std::complex<double>* dA, std::size_t ld, std::size_t lrows,
    std::size_t lcols, std::size_t g_row_off, std::size_t g_col_off,
    std::size_t N, cudaStream_t stream, double* h_min_abs);

} // namespace cuda_nccl_ph_diag
} // namespace internal
} // namespace linalg
} // namespace chase
