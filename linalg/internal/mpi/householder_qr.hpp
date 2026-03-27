// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "mpi.h"

#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"

namespace chase
{
namespace linalg
{
namespace internal
{

namespace
{
/** For each global row r < tab_len, out[r] is segment index s with r in that
 *  segment; unowned rows stay nseg (sentinel). */
inline void fill_pivot_row_seg_index(
    std::vector<std::size_t>&       out,
    std::size_t                     tab_len,
    std::size_t                     m_global,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t                     nseg)
{
    std::fill(out.begin(), out.end(), nseg);
    for (std::size_t s = 0; s < nseg; ++s)
    {
        const std::size_t g0 = seg_global_offs[s];
        const std::size_t g1 = std::min(g0 + seg_lens[s], m_global);
        if (g0 >= tab_len)
            continue;
        const std::size_t r_lo = g0;
        const std::size_t r_hi = std::min(g1, tab_len);
        for (std::size_t r = r_lo; r < r_hi; ++r)
            out[r] = s;
    }
}
} // namespace

//==============================================================================
// cpu_distributed_houseQR_panel_factor_1d_block
//
// Factor columns [k, k+jb) of V in place (Householder panel). Writes tau to
// tau_vec[k .. k+jb). Does NOT do form-T or trailing update. w is workspace
// (size >= jb). Same role as NCCL distributed_houseQR_panel_factor for swapping strategy.
//
// ChASE QR is tall-skinny: global rows m_global >= n (see formQ entry points
// that enforce this); pivot row index stays below n <= m_global.
//==============================================================================
template <typename T>
void cpu_mpi::cpu_distributed_houseQR_panel_factor_1d_block(
    std::size_t n, std::size_t l_rows, std::size_t g_off, std::size_t ldv,
    T* V, std::size_t k, std::size_t jb,
    std::vector<T>& tau_vec, MPI_Comm mpi_comm, std::vector<T>& w)
{
    using RealT = chase::Base<T>;
    const MPI_Datatype mpi_real_t = chase::mpi::getMPI_Type<RealT>();
    const MPI_Datatype mpi_T_t    = chase::mpi::getMPI_Type<T>();
    const T one  = T(1);
    const T zero = T(0);
    const int cblas_conj =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CblasConjTrans : CblasTrans;

    for (std::size_t jj = 0; jj < jb; ++jj)
    {
        const std::size_t col       = k + jj;
        const bool        pivot_here = (g_off <= col && col < g_off + l_rows);
        const std::size_t pivot_loc  = pivot_here ? (col - g_off) : 0;

        std::size_t rs, vr;
        if (col < g_off)             { rs = 0;           vr = l_rows; }
        else if (col < g_off+l_rows) { rs = col - g_off; vr = l_rows - rs; }
        else                         { rs = 0;           vr = 0; }

        RealT loc_nrm = (vr > 0)
            ? chase::linalg::blaspp::t_nrm2(vr, V + rs + col * ldv, 1)
            : RealT(0);
        RealT global_nrm_sq = loc_nrm * loc_nrm;
        MPI_Allreduce(MPI_IN_PLACE, &global_nrm_sq, 1, mpi_real_t, MPI_SUM, mpi_comm);

        if (global_nrm_sq <= RealT(0))
            continue;

        RealT global_nrm = std::sqrt(global_nrm_sq);

        T tau_col = zero;
        if (pivot_here)
        {
            T x0 = V[pivot_loc + col * ldv];
            T beta_T;
            if constexpr (std::is_same<T, float>::value ||
                          std::is_same<T, double>::value)
                beta_T = (x0 >= T(0)) ? T(global_nrm) : T(-global_nrm);
            else
            {
                const RealT ax0 = std::abs(x0);
                T sign_x0 = (ax0 == RealT(0)) ? T(1) : (x0 / ax0);
                beta_T = sign_x0 * T(global_nrm);
            }
            T denom = x0 + beta_T;
            if (denom != zero)
            {
                tau_col = denom / beta_T;
                if (vr > 1)
                {
                    T inv_denom = one / denom;
                    chase::linalg::blaspp::t_scal(
                        vr - 1, &inv_denom, V + (pivot_loc + 1) + col * ldv, 1);
                }
                V[pivot_loc + col * ldv] = -beta_T;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &tau_col, 1, mpi_T_t, MPI_SUM, mpi_comm);
        tau_vec[col] = tau_col;

        if (tau_col == zero)
            continue;

        T denom_bcast = zero;
        if (pivot_here)
        {
            T beta_T = -(V[pivot_loc + col * ldv]);
            denom_bcast = tau_col * beta_T;
        }
        MPI_Allreduce(MPI_IN_PLACE, &denom_bcast, 1, mpi_T_t, MPI_SUM, mpi_comm);
        if (!pivot_here && vr > 0 && denom_bcast != zero)
        {
            T inv_denom = one / denom_bcast;
            chase::linalg::blaspp::t_scal(vr, &inv_denom, V + rs + col * ldv, 1);
        }

        T saved_rkk = zero;
        if (pivot_here)
        {
            saved_rkk = V[pivot_loc + col * ldv];
            V[pivot_loc + col * ldv] = one;
        }

        const std::size_t n_panel_trail = (k + jb) - col - 1;
        if (n_panel_trail > 0)
        {
            std::fill(w.begin(), w.begin() + n_panel_trail, zero);
            if (vr > 0)
            {
                chase::linalg::blaspp::t_gemm(
                    CblasColMajor, cblas_conj, CblasNoTrans,
                    n_panel_trail, 1, vr,
                    &one,
                    V + rs + (col + 1) * ldv, ldv,
                    V + rs + col * ldv,        ldv,
                    &zero,
                    w.data(), n_panel_trail);
            }
            MPI_Allreduce(MPI_IN_PLACE, w.data(),
                          static_cast<int>(n_panel_trail), mpi_T_t, MPI_SUM, mpi_comm);
            if (vr > 0)
            {
                T minus_tau = -tau_col;
                chase::linalg::blaspp::t_gemm(
                    CblasColMajor, CblasNoTrans, cblas_conj,
                    vr, n_panel_trail, 1,
                    &minus_tau,
                    V + rs + col * ldv,        ldv,
                    w.data(),                  n_panel_trail,
                    &one,
                    V + rs + (col + 1) * ldv,  ldv);
            }
        }

        if (pivot_here)
            V[pivot_loc + col * ldv] = saved_rkk;
    }
}

//==============================================================================
// cpu_distributed_houseQR_formQ_1d_block
//
// Self-contained distributed-memory Householder QR on CPU using only BLAS
// (blaspp) and MPI.  The matrix V (m_global × n) is 1-D row-distributed: this
// rank owns rows [g_off, g_off + l_rows), stored column-major with leading
// dimension ldv.  On exit V holds the first n columns of Q.
//
// Precondition: m_global >= n (ChASE tall-skinny QR only).
//==============================================================================
template <typename T>
void cpu_mpi::cpu_distributed_houseQR_formQ_1d_block(std::size_t m_global,
                                            std::size_t n,
                                            std::size_t l_rows,
                                            std::size_t g_off,
                                            std::size_t ldv,
                                            T*          V,
                                            MPI_Comm    mpi_comm)
{
    using RealT = chase::Base<T>;

    if (n == 0)
        return;
    if (n > m_global)
        throw std::runtime_error(
            "cpu_distributed_houseQR_formQ_1d_block requires m_global >= n "
            "(ChASE tall-skinny QR)");
    if (l_rows != 0 &&
        n > std::numeric_limits<std::size_t>::max() / l_rows)
        throw std::runtime_error(
            "cpu_distributed_houseQR_formQ_1d_block: l_rows * n exceeds size_t");

    const MPI_Datatype mpi_T_t = chase::mpi::getMPI_Type<T>();
    const T one  = T(1);
    const T zero = T(0);
    const int cblas_conj =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CblasConjTrans : CblasTrans;

    std::vector<T> tau_vec(n, zero);
    std::vector<T> w(n);

    cpu_mpi::cpu_distributed_houseQR_panel_factor_1d_block<T>(
        n, l_rows, g_off, ldv, V, 0, n, tau_vec, mpi_comm, w);

    std::vector<T> VH(l_rows * n);
    chase::linalg::lapackpp::t_lacpy('A', l_rows, n, V, ldv, VH.data(), l_rows);

    std::memset(V, 0, ldv * n * sizeof(T));
    for (std::size_t c = 0; c < n; ++c)
    {
        if (g_off <= c && c < g_off + l_rows)
            V[(c - g_off) + c * ldv] = one;
    }

    for (std::size_t jj = 0; jj < n; ++jj)
    {
        const std::size_t j = n - 1 - jj;
        T tau_j = tau_vec[j];
        if (tau_j == zero)
            continue;

        const std::size_t n_cols = n - j;

        std::size_t rs, vr;
        if (j < g_off)             { rs = 0;          vr = l_rows; }
        else if (j < g_off+l_rows){ rs = j - g_off;  vr = l_rows - rs; }
        else                      { rs = 0;          vr = 0; }

        bool pivot_here = (g_off <= j && j < g_off + l_rows);
        T saved_diag = zero;
        if (pivot_here)
        {
            std::size_t ploc = j - g_off;
            saved_diag = VH[ploc + j * l_rows];
            VH[ploc + j * l_rows] = one;
        }

        std::fill(w.begin(), w.begin() + n_cols, zero);
        if (vr > 0)
        {
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, cblas_conj, CblasNoTrans,
                n_cols, 1, vr,
                &one,
                V + rs + j * ldv,         ldv,
                VH.data() + rs + j * l_rows, l_rows,
                &zero,
                w.data(), n_cols);
        }
        MPI_Allreduce(MPI_IN_PLACE, w.data(),
                      static_cast<int>(n_cols), mpi_T_t, MPI_SUM, mpi_comm);

        if (vr > 0)
        {
            T minus_tau = -tau_j;
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, CblasNoTrans, cblas_conj,
                vr, n_cols, 1,
                &minus_tau,
                VH.data() + rs + j * l_rows, l_rows,
                w.data(),                     n_cols,
                &one,
                V + rs + j * ldv,             ldv);
        }

        if (pivot_here)
            VH[(j - g_off) + j * l_rows] = saved_diag;
    }
}

//==============================================================================
// cpu_distributed_houseQR_panel_factor_block_cyclic_1d
//
// Same numerical panel as cpu_distributed_houseQR_panel_factor_1d_block, but
// local rows are given by contiguous segments (global start, local offset, len)
// as in DistMultiVectorBlockCyclic1D::m_contiguous_* . m_global rows total.
//
// Precondition: m_global >= n (ChASE tall-skinny QR only).
//==============================================================================
template <typename T>
void cpu_mpi::cpu_distributed_houseQR_panel_factor_block_cyclic_1d(
    std::size_t n,
    std::size_t m_global,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t ldv,
    T*          V,
    std::size_t k,
    std::size_t jb,
    std::vector<T>& tau_vec,
    MPI_Comm    mpi_comm,
    std::vector<T>& w)
{
    using RealT = chase::Base<T>;
    if (n > m_global)
        throw std::runtime_error(
            "cpu_distributed_houseQR_panel_factor_block_cyclic_1d requires "
            "m_global >= n (ChASE tall-skinny QR)");

    const std::size_t nseg = seg_global_offs.size();
    if (nseg != seg_local_offs.size() || nseg != seg_lens.size())
        throw std::runtime_error(
            "cpu_distributed_houseQR_panel_factor_block_cyclic_1d: segment vector "
            "sizes must match");

    const MPI_Datatype mpi_real_t = chase::mpi::getMPI_Type<RealT>();
    const MPI_Datatype mpi_T_t    = chase::mpi::getMPI_Type<T>();
    const T one  = T(1);
    const T zero = T(0);
    const int cblas_conj =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CblasConjTrans : CblasTrans;

    const std::size_t pivot_tab_len = std::max(n, k + jb);
    std::vector<std::size_t> row_seg(pivot_tab_len, nseg);
    fill_pivot_row_seg_index(row_seg, pivot_tab_len, m_global, seg_global_offs,
                             seg_lens, nseg);

    for (std::size_t jj = 0; jj < jb; ++jj)
    {
        const std::size_t col = k + jj;
        if (col >= m_global)
            continue;

        bool          pivot_here = false;
        std::size_t pivot_loc  = 0;
        if (col < pivot_tab_len)
        {
            const std::size_t s = row_seg[col];
            if (s < nseg)
            {
                pivot_here = true;
                pivot_loc =
                    seg_local_offs[s] + (col - seg_global_offs[s]);
            }
        }
        else
        {
            for (std::size_t s = 0; s < nseg; ++s)
            {
                const std::size_t g0 = seg_global_offs[s];
                const std::size_t g1 = g0 + seg_lens[s];
                if (col >= g0 && col < g1)
                {
                    pivot_here = true;
                    pivot_loc  = seg_local_offs[s] + (col - g0);
                    break;
                }
            }
        }

        RealT loc_nrm_sq = RealT(0);
        for (std::size_t s = 0; s < nseg; ++s)
        {
            const std::size_t g0 = seg_global_offs[s];
            const std::size_t g1 = g0 + seg_lens[s];
            const std::size_t row_lo = std::max(g0, col);
            const std::size_t row_hi = std::min(g1, m_global);
            if (row_lo < row_hi)
            {
                const std::size_t loc_b = seg_local_offs[s] + (row_lo - g0);
                const std::size_t vr    = row_hi - row_lo;
                RealT seg_nrm =
                    chase::linalg::blaspp::t_nrm2(vr, V + loc_b + col * ldv, 1);
                loc_nrm_sq += seg_nrm * seg_nrm;
            }
        }
        RealT global_nrm_sq = loc_nrm_sq;
        MPI_Allreduce(MPI_IN_PLACE, &global_nrm_sq, 1, mpi_real_t, MPI_SUM, mpi_comm);

        if (global_nrm_sq <= RealT(0))
            continue;

        RealT global_nrm = std::sqrt(global_nrm_sq);

        T tau_col = zero;
        if (pivot_here)
        {
            T x0 = V[pivot_loc + col * ldv];
            T beta_T;
            if constexpr (std::is_same<T, float>::value ||
                          std::is_same<T, double>::value)
                beta_T = (x0 >= T(0)) ? T(global_nrm) : T(-global_nrm);
            else
            {
                const RealT ax0 = std::abs(x0);
                T sign_x0       = (ax0 == RealT(0)) ? T(1) : (x0 / ax0);
                beta_T          = sign_x0 * T(global_nrm);
            }
            T denom = x0 + beta_T;
            if (denom != zero)
            {
                const T inv_denom = one / denom;
                for (std::size_t s = 0; s < nseg; ++s)
                {
                    const std::size_t g0 = seg_global_offs[s];
                    const std::size_t g1 = g0 + seg_lens[s];
                    const std::size_t row_lo = std::max(g0, col);
                    const std::size_t row_hi = std::min(g1, m_global);
                    if (row_lo >= row_hi)
                        continue;
                    if (col >= g0 && col < g1)
                    {
                        const std::size_t ploc   = seg_local_offs[s] + (col - g0);
                        const std::size_t n_below = row_hi - (col + 1);
                        if (n_below > 0)
                        {
                            chase::linalg::blaspp::t_scal(
                                n_below, &inv_denom, V + (ploc + 1) + col * ldv, 1);
                        }
                    }
                    else
                    {
                        const std::size_t loc_b = seg_local_offs[s] + (row_lo - g0);
                        const std::size_t vr    = row_hi - row_lo;
                        chase::linalg::blaspp::t_scal(vr, &inv_denom, V + loc_b + col * ldv, 1);
                    }
                }
                tau_col = denom / beta_T;
                V[pivot_loc + col * ldv] = -beta_T;
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &tau_col, 1, mpi_T_t, MPI_SUM, mpi_comm);
        tau_vec[col] = tau_col;

        if (tau_col == zero)
            continue;

        T denom_bcast = zero;
        if (pivot_here)
        {
            const T beta_T = -(V[pivot_loc + col * ldv]);
            denom_bcast    = tau_col * beta_T;
        }
        MPI_Allreduce(MPI_IN_PLACE, &denom_bcast, 1, mpi_T_t, MPI_SUM, mpi_comm);
        if (!pivot_here && denom_bcast != zero)
        {
            const T inv_denom = one / denom_bcast;
            for (std::size_t s = 0; s < nseg; ++s)
            {
                const std::size_t g0 = seg_global_offs[s];
                const std::size_t g1 = g0 + seg_lens[s];
                const std::size_t row_lo = std::max(g0, col);
                const std::size_t row_hi = std::min(g1, m_global);
                if (row_lo >= row_hi)
                    continue;
                const std::size_t loc_b = seg_local_offs[s] + (row_lo - g0);
                const std::size_t vr    = row_hi - row_lo;
                chase::linalg::blaspp::t_scal(vr, &inv_denom, V + loc_b + col * ldv, 1);
            }
        }

        T saved_rkk = zero;
        if (pivot_here)
        {
            saved_rkk                  = V[pivot_loc + col * ldv];
            V[pivot_loc + col * ldv] = one;
        }

        const std::size_t n_panel_trail = (k + jb) - col - 1;
        if (n_panel_trail > 0)
        {
            std::fill(w.begin(), w.begin() + n_panel_trail, zero);
            bool first_w = true;
            for (std::size_t s = 0; s < nseg; ++s)
            {
                const std::size_t g0 = seg_global_offs[s];
                const std::size_t g1 = g0 + seg_lens[s];
                const std::size_t row_lo = std::max(g0, col);
                const std::size_t row_hi = std::min(g1, m_global);
                if (row_lo >= row_hi)
                    continue;
                const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
                const std::size_t vr_seg = row_hi - row_lo;
                const T           beta_w = first_w ? zero : one;
                first_w                    = false;
                chase::linalg::blaspp::t_gemm(
                    CblasColMajor, cblas_conj, CblasNoTrans,
                    n_panel_trail, 1, vr_seg,
                    &one,
                    V + rs_seg + (col + 1) * ldv, ldv,
                    V + rs_seg + col * ldv, ldv,
                    &beta_w,
                    w.data(), n_panel_trail);
            }
            MPI_Allreduce(MPI_IN_PLACE, w.data(),
                          static_cast<int>(n_panel_trail), mpi_T_t, MPI_SUM, mpi_comm);
            const T minus_tau = -tau_col;
            for (std::size_t s = 0; s < nseg; ++s)
            {
                const std::size_t g0 = seg_global_offs[s];
                const std::size_t g1 = g0 + seg_lens[s];
                const std::size_t row_lo = std::max(g0, col);
                const std::size_t row_hi = std::min(g1, m_global);
                if (row_lo >= row_hi)
                    continue;
                const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
                const std::size_t vr_seg = row_hi - row_lo;
                chase::linalg::blaspp::t_gemm(
                    CblasColMajor, CblasNoTrans, cblas_conj,
                    vr_seg, n_panel_trail, 1,
                    &minus_tau,
                    V + rs_seg + col * ldv, ldv,
                    w.data(), n_panel_trail,
                    &one,
                    V + rs_seg + (col + 1) * ldv, ldv);
            }
        }

        if (pivot_here)
            V[pivot_loc + col * ldv] = saved_rkk;
    }
}

//==============================================================================
// cpu_distributed_houseQR_formQ_block_cyclic_1d
//
// Unblocked distributed Householder QR + form Q for 1-D block-cyclic row layout
// described by segment metadata (same as panel_factor_block_cyclic_1d).
//
// Precondition: m_global >= n (ChASE tall-skinny QR only).
//==============================================================================
template <typename T>
void cpu_mpi::cpu_distributed_houseQR_formQ_block_cyclic_1d(
    std::size_t m_global,
    std::size_t n,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t ldv,
    T*          V,
    MPI_Comm    mpi_comm)
{
    if (n == 0)
        return;
    if (n > m_global)
        throw std::runtime_error(
            "cpu_distributed_houseQR_formQ_block_cyclic_1d requires m_global >= n "
            "(ChASE tall-skinny QR)");

    const std::size_t nseg = seg_global_offs.size();
    if (nseg != seg_local_offs.size() || nseg != seg_lens.size())
        throw std::runtime_error(
            "cpu_distributed_houseQR_formQ_block_cyclic_1d: segment vector sizes "
            "must match");

    const std::size_t l_rows =
        nseg == 0 ? 0 : seg_local_offs.back() + seg_lens.back();

    if (l_rows != 0 &&
        n > std::numeric_limits<std::size_t>::max() / l_rows)
        throw std::runtime_error(
            "cpu_distributed_houseQR_formQ_block_cyclic_1d: l_rows * n exceeds "
            "size_t");

    const MPI_Datatype mpi_T_t = chase::mpi::getMPI_Type<T>();
    const T one  = T(1);
    const T zero = T(0);
    const int cblas_conj =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CblasConjTrans : CblasTrans;

    std::vector<std::size_t> row_seg(n, nseg);
    fill_pivot_row_seg_index(row_seg, n, m_global, seg_global_offs, seg_lens,
                             nseg);

    std::vector<T> tau_vec(n, zero);
    std::vector<T> w(n);

    cpu_mpi::cpu_distributed_houseQR_panel_factor_block_cyclic_1d<T>(
        n, m_global, seg_global_offs, seg_local_offs, seg_lens, ldv, V, 0, n,
        tau_vec, mpi_comm, w);

    std::vector<T> VH(l_rows * n);
    chase::linalg::lapackpp::t_lacpy('A', l_rows, n, V, ldv, VH.data(), l_rows);

    std::memset(V, 0, ldv * n * sizeof(T));
    for (std::size_t c = 0; c < n; ++c)
    {
        const std::size_t s = row_seg[c];
        if (s < nseg)
        {
            const std::size_t ll =
                seg_local_offs[s] + (c - seg_global_offs[s]);
            V[ll + c * ldv] = one;
        }
    }

    for (std::size_t jj = 0; jj < n; ++jj)
    {
        const std::size_t j = n - 1 - jj;
        T tau_j = tau_vec[j];
        if (tau_j == zero)
            continue;

        const std::size_t n_cols = n - j;

        bool          pivot_here = false;
        std::size_t pivot_ploc   = 0;
        const std::size_t sj     = row_seg[j];
        if (sj < nseg)
        {
            pivot_here = true;
            pivot_ploc =
                seg_local_offs[sj] + (j - seg_global_offs[sj]);
        }

        T saved_diag = zero;
        if (pivot_here)
        {
            saved_diag                     = VH[pivot_ploc + j * l_rows];
            VH[pivot_ploc + j * l_rows] = one;
        }

        std::fill(w.begin(), w.begin() + n_cols, zero);
        bool first_w = true;
        for (std::size_t s = 0; s < nseg; ++s)
        {
            const std::size_t g0 = seg_global_offs[s];
            const std::size_t g1 = g0 + seg_lens[s];
            const std::size_t row_lo = std::max(g0, j);
            const std::size_t row_hi = std::min(g1, m_global);
            if (row_lo >= row_hi)
                continue;
            const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
            const std::size_t vr_seg = row_hi - row_lo;
            const T           beta_w = first_w ? zero : one;
            first_w                    = false;
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, cblas_conj, CblasNoTrans,
                n_cols, 1, vr_seg,
                &one,
                V + rs_seg + j * ldv, ldv,
                VH.data() + rs_seg + j * l_rows, l_rows,
                &beta_w,
                w.data(), n_cols);
        }
        MPI_Allreduce(MPI_IN_PLACE, w.data(),
                      static_cast<int>(n_cols), mpi_T_t, MPI_SUM, mpi_comm);

        const T minus_tau = -tau_j;
        for (std::size_t s = 0; s < nseg; ++s)
        {
            const std::size_t g0 = seg_global_offs[s];
            const std::size_t g1 = g0 + seg_lens[s];
            const std::size_t row_lo = std::max(g0, j);
            const std::size_t row_hi = std::min(g1, m_global);
            if (row_lo >= row_hi)
                continue;
            const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
            const std::size_t vr_seg = row_hi - row_lo;
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, CblasNoTrans, cblas_conj,
                vr_seg, n_cols, 1,
                &minus_tau,
                VH.data() + rs_seg + j * l_rows, l_rows,
                w.data(), n_cols,
                &one,
                V + rs_seg + j * ldv, ldv);
        }

        if (pivot_here)
            VH[pivot_ploc + j * l_rows] = saved_diag;
    }
}

template <typename InputMultiVectorType>
void cpu_mpi::cpu_distributed_houseQR_formQ(InputMultiVectorType& V)
{
    using Vt = typename InputMultiVectorType::value_type;
    if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<
                      InputMultiVectorType>::value)
    {
        const auto seg_g = V.m_contiguous_global_offs();
        const auto seg_l = V.m_contiguous_local_offs();
        const auto seg_n = V.m_contiguous_lens();
        cpu_mpi::cpu_distributed_houseQR_formQ_block_cyclic_1d<Vt>(
            V.g_rows(), V.g_cols(), seg_g, seg_l, seg_n, V.l_ld(), V.l_data(),
            V.getMpiGrid()->get_col_comm());
    }
    else
    {
        cpu_mpi::cpu_distributed_houseQR_formQ_1d_block<Vt>(
            V.g_rows(), V.g_cols(), V.l_rows(), V.g_off(), V.l_ld(), V.l_data(),
            V.getMpiGrid()->get_col_comm());
    }
}

//==============================================================================
// cpu_distributed_blocked_houseQR_formQ_1d_block
//
// Blocked distributed Householder QR + form Q (compact WY).  nb = panel size.
// On exit V holds the first n columns of Q.
//
// Precondition: m_global >= n (ChASE tall-skinny QR only).
//==============================================================================
template <typename T>
void cpu_mpi::cpu_distributed_blocked_houseQR_formQ_1d_block(std::size_t m_global,
                                                    std::size_t n,
                                                    std::size_t l_rows,
                                                    std::size_t g_off,
                                                    std::size_t ldv,
                                                    T*          V,
                                                    MPI_Comm    mpi_comm,
                                                    std::size_t nb)
{
    using RealT = chase::Base<T>;

    if (n == 0)
        return;
    if (n > m_global)
        throw std::runtime_error(
            "cpu_distributed_blocked_houseQR_formQ_1d_block requires m_global >= n "
            "(ChASE tall-skinny QR)");
    if (l_rows != 0 &&
        n > std::numeric_limits<std::size_t>::max() / l_rows)
        throw std::runtime_error(
            "cpu_distributed_blocked_houseQR_formQ_1d_block: l_rows * n exceeds "
            "size_t");

    if (nb == 0 || nb >= n)
    {
        cpu_mpi::cpu_distributed_houseQR_formQ_1d_block<T>(
            m_global, n, l_rows, g_off, ldv, V, mpi_comm);
        return;
    }

    const MPI_Datatype mpi_T_t = chase::mpi::getMPI_Type<T>();

    const T one  = T(1);
    const T zero = T(0);
    const int cblas_conj =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CblasConjTrans : CblasTrans;

    const std::size_t num_blocks = (n + nb - 1) / nb;

    std::vector<T>              tau_vec(n, zero);
    std::vector<std::vector<T>> T_blocks(num_blocks);
    std::vector<T> w(n);

    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        const std::size_t k  = b * nb;
        const std::size_t jb = std::min(nb, n - k);
        T_blocks[b].assign(jb * jb, zero);
        T* Tb = T_blocks[b].data();

        cpu_mpi::cpu_distributed_houseQR_panel_factor_1d_block<T>(
            n, l_rows, g_off, ldv, V, k, jb, tau_vec, mpi_comm, w);

        std::vector<T> t_col(nb);
        std::vector<T> tmp(nb);

        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            Tb[jj + jj * jb] = tau_vec[k + jj];
            if (jj == 0 || tau_vec[k + jj] == zero)
                continue;

            const std::size_t col       = k + jj;
            const bool        pivot_jj  = (g_off <= col && col < g_off + l_rows);

            std::size_t rs2, vr2;
            if (col < g_off)             { rs2 = 0;           vr2 = l_rows; }
            else if (col < g_off+l_rows) { rs2 = col - g_off; vr2 = l_rows - rs2; }
            else                         { rs2 = 0;           vr2 = 0; }

            std::fill(t_col.begin(), t_col.begin() + jj, zero);

            T saved_v_jj = zero;
            if (pivot_jj)
            {
                saved_v_jj = V[rs2 + col * ldv];
                V[rs2 + col * ldv] = one;
            }

            if (vr2 > 0)
            {
                chase::linalg::blaspp::t_gemm(
                    CblasColMajor, cblas_conj, CblasNoTrans,
                    jj, 1, vr2,
                    &one,
                    V + rs2 + k * ldv,   ldv,
                    V + rs2 + col * ldv, ldv,
                    &zero,
                    t_col.data(), jj);
            }
            MPI_Allreduce(MPI_IN_PLACE, t_col.data(),
                          static_cast<int>(jj), mpi_T_t, MPI_SUM, mpi_comm);

            if (pivot_jj)
                V[rs2 + col * ldv] = saved_v_jj;

            std::fill(tmp.begin(), tmp.begin() + jj, zero);
            for (std::size_t ii = 0; ii < jj; ++ii)
                for (std::size_t ll = ii; ll < jj; ++ll)
                    tmp[ii] += Tb[ii + ll * jb] * t_col[ll];

            T neg_tau = -tau_vec[col];
            for (std::size_t ii = 0; ii < jj; ++ii)
                Tb[ii + jj * jb] = neg_tau * tmp[ii];
        }

        const std::size_t n_trail = n - k - jb;
        if (n_trail == 0)
            continue;

        std::size_t rs_k, vr_k;
        if (k < g_off)             { rs_k = 0;         vr_k = l_rows; }
        else if (k < g_off+l_rows) { rs_k = k - g_off; vr_k = l_rows - rs_k; }
        else                       { rs_k = 0;         vr_k = 0; }

        std::vector<T> saved_block(jb * jb, zero);
        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (g_off <= grow && grow < g_off + l_rows)
                {
                    const std::size_t lrow = grow - g_off;
                    saved_block[ii + jj * jb] = V[lrow + col * ldv];
                    V[lrow + col * ldv] = (ii == jj) ? one : zero;
                }
            }
        }

        std::vector<T> W (jb * n_trail, zero);
        std::vector<T> TW(jb * n_trail, zero);

        if (vr_k > 0)
        {
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, cblas_conj, CblasNoTrans,
                jb, n_trail, vr_k,
                &one,
                V + rs_k + k * ldv,        ldv,
                V + rs_k + (k + jb) * ldv, ldv,
                &zero,
                W.data(), jb);
        }
        MPI_Allreduce(MPI_IN_PLACE, W.data(),
                      static_cast<int>(jb * n_trail), mpi_T_t, MPI_SUM, mpi_comm);

        chase::linalg::blaspp::t_gemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            jb, n_trail, jb,
            &one,
            Tb, jb,
            W.data(), jb,
            &zero,
            TW.data(), jb);

        if (vr_k > 0)
        {
            T minus_one = T(-1);
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                vr_k, n_trail, jb,
                &minus_one,
                V + rs_k + k * ldv,        ldv,
                TW.data(), jb,
                &one,
                V + rs_k + (k + jb) * ldv, ldv);
        }

        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (g_off <= grow && grow < g_off + l_rows)
                    V[(grow - g_off) + col * ldv] = saved_block[ii + jj * jb];
            }
        }
    }

    std::vector<T> VH(l_rows * n);
    chase::linalg::lapackpp::t_lacpy('A', l_rows, n, V, ldv, VH.data(), l_rows);

    std::memset(V, 0, ldv * n * sizeof(T));
    for (std::size_t c = 0; c < n; ++c)
    {
        if (g_off <= c && c < g_off + l_rows)
            V[(c - g_off) + c * ldv] = one;
    }

    for (std::size_t bb = 0; bb < num_blocks; ++bb)
    {
        const std::size_t b      = num_blocks - 1 - bb;
        const std::size_t k      = b * nb;
        const std::size_t jb     = std::min(nb, n - k);
        const std::size_t n_cols = n - k;
        T* Tb = T_blocks[b].data();

        std::size_t rs_k, vr_k;
        if (k < g_off)             { rs_k = 0;         vr_k = l_rows; }
        else if (k < g_off+l_rows) { rs_k = k - g_off; vr_k = l_rows - rs_k; }
        else                       { rs_k = 0;         vr_k = 0; }

        std::vector<T> saved_block(jb * jb, zero);
        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (g_off <= grow && grow < g_off + l_rows)
                {
                    const std::size_t lrow = grow - g_off;
                    saved_block[ii + jj * jb] = VH[lrow + col * l_rows];
                    VH[lrow + col * l_rows] = (ii == jj) ? one : zero;
                }
            }
        }

        std::vector<T> W (jb * n_cols, zero);
        std::vector<T> TW(jb * n_cols, zero);

        if (vr_k > 0)
        {
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, cblas_conj, CblasNoTrans,
                jb, n_cols, vr_k,
                &one,
                VH.data() + rs_k + k * l_rows, l_rows,
                V  + rs_k + k * ldv,            ldv,
                &zero,
                W.data(), jb);
        }
        MPI_Allreduce(MPI_IN_PLACE, W.data(),
                      static_cast<int>(jb * n_cols), mpi_T_t, MPI_SUM, mpi_comm);

        chase::linalg::blaspp::t_gemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            jb, n_cols, jb,
            &one,
            Tb, jb,
            W.data(), jb,
            &zero,
            TW.data(), jb);

        if (vr_k > 0)
        {
            T minus_one = T(-1);
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                vr_k, n_cols, jb,
                &minus_one,
                VH.data() + rs_k + k * l_rows, l_rows,
                TW.data(), jb,
                &one,
                V + rs_k + k * ldv, ldv);
        }

        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (g_off <= grow && grow < g_off + l_rows)
                    VH[(grow - g_off) + col * l_rows] = saved_block[ii + jj * jb];
            }
        }
    }
}

//==============================================================================
// cpu_distributed_blocked_houseQR_formQ_block_cyclic_1d
//
// Blocked compact-WY Householder QR + form Q; same segment layout as
// panel_factor_block_cyclic_1d. Delegates to formQ_block_cyclic_1d when
// nb == 0 or nb >= n.
//
// Precondition: m_global >= n (ChASE tall-skinny QR only).
//==============================================================================
template <typename T>
void cpu_mpi::cpu_distributed_blocked_houseQR_formQ_block_cyclic_1d(
    std::size_t m_global,
    std::size_t n,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t ldv,
    T*          V,
    MPI_Comm    mpi_comm,
    std::size_t nb)
{
    if (n == 0)
        return;
    if (n > m_global)
        throw std::runtime_error(
            "cpu_distributed_blocked_houseQR_formQ_block_cyclic_1d requires "
            "m_global >= n (ChASE tall-skinny QR)");

    const std::size_t nseg = seg_global_offs.size();
    if (nseg != seg_local_offs.size() || nseg != seg_lens.size())
        throw std::runtime_error(
            "cpu_distributed_blocked_houseQR_formQ_block_cyclic_1d: segment vector "
            "sizes must match");

    const std::size_t l_rows =
        nseg == 0 ? 0 : seg_local_offs.back() + seg_lens.back();

    if (l_rows != 0 &&
        n > std::numeric_limits<std::size_t>::max() / l_rows)
        throw std::runtime_error(
            "cpu_distributed_blocked_houseQR_formQ_block_cyclic_1d: l_rows * n "
            "exceeds size_t");

    if (nb == 0 || nb >= n)
    {
        cpu_mpi::cpu_distributed_houseQR_formQ_block_cyclic_1d<T>(
            m_global, n, seg_global_offs, seg_local_offs, seg_lens, ldv, V,
            mpi_comm);
        return;
    }

    std::vector<std::size_t> row_seg(n, nseg);
    fill_pivot_row_seg_index(row_seg, n, m_global, seg_global_offs, seg_lens,
                             nseg);

    const MPI_Datatype mpi_T_t = chase::mpi::getMPI_Type<T>();

    const T one  = T(1);
    const T zero = T(0);
    const int cblas_conj =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CblasConjTrans : CblasTrans;

    const std::size_t num_blocks = (n + nb - 1) / nb;

    std::vector<T>              tau_vec(n, zero);
    std::vector<std::vector<T>> T_blocks(num_blocks);
    std::vector<T>              w(n);

    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        const std::size_t k  = b * nb;
        const std::size_t jb = std::min(nb, n - k);
        T_blocks[b].assign(jb * jb, zero);
        T* Tb = T_blocks[b].data();

        cpu_mpi::cpu_distributed_houseQR_panel_factor_block_cyclic_1d<T>(
            n, m_global, seg_global_offs, seg_local_offs, seg_lens, ldv, V, k,
            jb, tau_vec, mpi_comm, w);

        std::vector<T> t_col(nb);
        std::vector<T> tmp(nb);

        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            Tb[jj + jj * jb] = tau_vec[k + jj];
            if (jj == 0 || tau_vec[k + jj] == zero)
                continue;

            const std::size_t col      = k + jj;
            const std::size_t sj     = row_seg[col];
            const bool        pivot_jj = (sj < nseg);

            std::size_t rs_piv = 0;
            if (pivot_jj)
                rs_piv = seg_local_offs[sj] + (col - seg_global_offs[sj]);

            std::fill(t_col.begin(), t_col.begin() + jj, zero);

            T saved_v_jj = zero;
            if (pivot_jj)
            {
                saved_v_jj                = V[rs_piv + col * ldv];
                V[rs_piv + col * ldv] = one;
            }

            bool first_tc = true;
            for (std::size_t s = 0; s < nseg; ++s)
            {
                const std::size_t g0 = seg_global_offs[s];
                const std::size_t g1 = g0 + seg_lens[s];
                const std::size_t row_lo = std::max(g0, col);
                const std::size_t row_hi = std::min(g1, m_global);
                if (row_lo >= row_hi)
                    continue;
                const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
                const std::size_t vr_seg = row_hi - row_lo;
                const T           beta_tc = first_tc ? zero : one;
                first_tc                  = false;
                chase::linalg::blaspp::t_gemm(
                    CblasColMajor, cblas_conj, CblasNoTrans,
                    jj, 1, vr_seg,
                    &one,
                    V + rs_seg + k * ldv, ldv,
                    V + rs_seg + col * ldv, ldv,
                    &beta_tc,
                    t_col.data(), jj);
            }
            MPI_Allreduce(MPI_IN_PLACE, t_col.data(),
                          static_cast<int>(jj), mpi_T_t, MPI_SUM, mpi_comm);

            if (pivot_jj)
                V[rs_piv + col * ldv] = saved_v_jj;

            std::fill(tmp.begin(), tmp.begin() + jj, zero);
            for (std::size_t ii = 0; ii < jj; ++ii)
                for (std::size_t ll = ii; ll < jj; ++ll)
                    tmp[ii] += Tb[ii + ll * jb] * t_col[ll];

            const T neg_tau = -tau_vec[col];
            for (std::size_t ii = 0; ii < jj; ++ii)
                Tb[ii + jj * jb] = neg_tau * tmp[ii];
        }

        const std::size_t n_trail = n - k - jb;
        if (n_trail == 0)
            continue;

        std::vector<T> saved_block(jb * jb, zero);
        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (grow < n)
                {
                    const std::size_t sg = row_seg[grow];
                    if (sg < nseg)
                    {
                        const std::size_t lrow =
                            seg_local_offs[sg] + (grow - seg_global_offs[sg]);
                        saved_block[ii + jj * jb] = V[lrow + col * ldv];
                        V[lrow + col * ldv] = (ii == jj) ? one : zero;
                    }
                }
            }
        }

        std::vector<T> W(jb * n_trail, zero);
        std::vector<T> TW(jb * n_trail, zero);

        bool first_W = true;
        for (std::size_t s = 0; s < nseg; ++s)
        {
            const std::size_t g0 = seg_global_offs[s];
            const std::size_t g1 = g0 + seg_lens[s];
            const std::size_t row_lo = std::max(g0, k);
            const std::size_t row_hi = std::min(g1, m_global);
            if (row_lo >= row_hi)
                continue;
            const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
            const std::size_t vr_seg = row_hi - row_lo;
            const T           beta_W = first_W ? zero : one;
            first_W                    = false;
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, cblas_conj, CblasNoTrans,
                jb, n_trail, vr_seg,
                &one,
                V + rs_seg + k * ldv, ldv,
                V + rs_seg + (k + jb) * ldv, ldv,
                &beta_W,
                W.data(), jb);
        }
        MPI_Allreduce(MPI_IN_PLACE, W.data(),
                      static_cast<int>(jb * n_trail), mpi_T_t, MPI_SUM,
                      mpi_comm);

        chase::linalg::blaspp::t_gemm(
            CblasColMajor, CblasConjTrans, CblasNoTrans,
            jb, n_trail, jb,
            &one,
            Tb, jb,
            W.data(), jb,
            &zero,
            TW.data(), jb);

        const T minus_one = T(-1);
        for (std::size_t s = 0; s < nseg; ++s)
        {
            const std::size_t g0 = seg_global_offs[s];
            const std::size_t g1 = g0 + seg_lens[s];
            const std::size_t row_lo = std::max(g0, k);
            const std::size_t row_hi = std::min(g1, m_global);
            if (row_lo >= row_hi)
                continue;
            const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
            const std::size_t vr_seg = row_hi - row_lo;
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                vr_seg, n_trail, jb,
                &minus_one,
                V + rs_seg + k * ldv, ldv,
                TW.data(), jb,
                &one,
                V + rs_seg + (k + jb) * ldv, ldv);
        }

        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (grow < n)
                {
                    const std::size_t sg = row_seg[grow];
                    if (sg < nseg)
                    {
                        const std::size_t lrow =
                            seg_local_offs[sg] + (grow - seg_global_offs[sg]);
                        V[lrow + col * ldv] = saved_block[ii + jj * jb];
                    }
                }
            }
        }
    }

    std::vector<T> VH(l_rows * n);
    chase::linalg::lapackpp::t_lacpy('A', l_rows, n, V, ldv, VH.data(), l_rows);

    std::memset(V, 0, ldv * n * sizeof(T));
    for (std::size_t c = 0; c < n; ++c)
    {
        const std::size_t s = row_seg[c];
        if (s < nseg)
        {
            const std::size_t ll =
                seg_local_offs[s] + (c - seg_global_offs[s]);
            V[ll + c * ldv] = one;
        }
    }

    for (std::size_t bb = 0; bb < num_blocks; ++bb)
    {
        const std::size_t b      = num_blocks - 1 - bb;
        const std::size_t k      = b * nb;
        const std::size_t jb     = std::min(nb, n - k);
        const std::size_t n_cols = n - k;
        T* Tb = T_blocks[b].data();

        std::vector<T> saved_block(jb * jb, zero);
        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (grow < n)
                {
                    const std::size_t sg = row_seg[grow];
                    if (sg < nseg)
                    {
                        const std::size_t lrow =
                            seg_local_offs[sg] + (grow - seg_global_offs[sg]);
                        saved_block[ii + jj * jb] = VH[lrow + col * l_rows];
                        VH[lrow + col * l_rows] = (ii == jj) ? one : zero;
                    }
                }
            }
        }

        std::vector<T> W(jb * n_cols, zero);
        std::vector<T> TW(jb * n_cols, zero);

        bool first_bkw = true;
        for (std::size_t s = 0; s < nseg; ++s)
        {
            const std::size_t g0 = seg_global_offs[s];
            const std::size_t g1 = g0 + seg_lens[s];
            const std::size_t row_lo = std::max(g0, k);
            const std::size_t row_hi = std::min(g1, m_global);
            if (row_lo >= row_hi)
                continue;
            const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
            const std::size_t vr_seg = row_hi - row_lo;
            const T           beta_b = first_bkw ? zero : one;
            first_bkw                  = false;
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, cblas_conj, CblasNoTrans,
                jb, n_cols, vr_seg,
                &one,
                VH.data() + rs_seg + k * l_rows, l_rows,
                V + rs_seg + k * ldv, ldv,
                &beta_b,
                W.data(), jb);
        }
        MPI_Allreduce(MPI_IN_PLACE, W.data(),
                      static_cast<int>(jb * n_cols), mpi_T_t, MPI_SUM, mpi_comm);

        chase::linalg::blaspp::t_gemm(
            CblasColMajor, CblasNoTrans, CblasNoTrans,
            jb, n_cols, jb,
            &one,
            Tb, jb,
            W.data(), jb,
            &zero,
            TW.data(), jb);

        const T minus_one_b = T(-1);
        for (std::size_t s = 0; s < nseg; ++s)
        {
            const std::size_t g0 = seg_global_offs[s];
            const std::size_t g1 = g0 + seg_lens[s];
            const std::size_t row_lo = std::max(g0, k);
            const std::size_t row_hi = std::min(g1, m_global);
            if (row_lo >= row_hi)
                continue;
            const std::size_t rs_seg = seg_local_offs[s] + (row_lo - g0);
            const std::size_t vr_seg = row_hi - row_lo;
            chase::linalg::blaspp::t_gemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                vr_seg, n_cols, jb,
                &minus_one_b,
                VH.data() + rs_seg + k * l_rows, l_rows,
                TW.data(), jb,
                &one,
                V + rs_seg + k * ldv, ldv);
        }

        for (std::size_t jj = 0; jj < jb; ++jj)
        {
            const std::size_t col = k + jj;
            for (std::size_t ii = 0; ii <= jj; ++ii)
            {
                const std::size_t grow = k + ii;
                if (grow < n)
                {
                    const std::size_t sg = row_seg[grow];
                    if (sg < nseg)
                    {
                        const std::size_t lrow =
                            seg_local_offs[sg] + (grow - seg_global_offs[sg]);
                        VH[lrow + col * l_rows] = saved_block[ii + jj * jb];
                    }
                }
            }
        }
    }
}

template <typename InputMultiVectorType>
void cpu_mpi::cpu_distributed_blocked_houseQR_formQ(InputMultiVectorType& V,
                                                    std::size_t nb)
{
    using Vt = typename InputMultiVectorType::value_type;
    if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<
                      InputMultiVectorType>::value)
    {
        const auto seg_g = V.m_contiguous_global_offs();
        const auto seg_l = V.m_contiguous_local_offs();
        const auto seg_n = V.m_contiguous_lens();
        cpu_mpi::cpu_distributed_blocked_houseQR_formQ_block_cyclic_1d<Vt>(
            V.g_rows(), V.g_cols(), seg_g, seg_l, seg_n, V.l_ld(), V.l_data(),
            V.getMpiGrid()->get_col_comm(), nb);
    }
    else
    {
        cpu_mpi::cpu_distributed_blocked_houseQR_formQ_1d_block<Vt>(
            V.g_rows(), V.g_cols(), V.l_rows(), V.g_off(), V.l_ld(), V.l_data(),
            V.getMpiGrid()->get_col_comm(), nb);
    }
}

} // namespace internal
} // namespace linalg
} // namespace chase
