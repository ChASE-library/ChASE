// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "algorithm/logger.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

#include "external/cublaspp/cublaspp.hpp"
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/internal/cuda/householder_scalar_kernel.hpp"
#include "grid/nccl_utils.hpp"
#include "external/blaspp/blaspp.hpp"
#include "external/lapackpp/lapackpp.hpp"

#ifndef STRICT_ORTHO_TBUILD
#define STRICT_ORTHO_TBUILD 0
#endif

#ifndef STRICT_TAU_BCAST
#define STRICT_TAU_BCAST 0
#endif

#ifndef STRICT_TIMING
#define STRICT_TIMING 0
#endif

namespace chase
{
namespace linalg
{
namespace internal
{
namespace
{
inline void nccl_fill_pivot_row_seg_index(
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

inline void nccl_fill_pivot_row_local_index(
    std::vector<std::size_t>&              row_local,
    const std::vector<std::size_t>&        row_seg,
    std::size_t                            tab_len,
    const std::vector<std::size_t>&        seg_global_offs,
    const std::vector<std::size_t>&        seg_local_offs,
    std::size_t                            nseg)
{
    for (std::size_t r = 0; r < tab_len; ++r)
    {
        const std::size_t s = row_seg[r];
        if (s < nseg)
            row_local[r] = seg_local_offs[s] + (r - seg_global_offs[s]);
        else
            row_local[r] = 0;
    }
}

inline bool nccl_locate_global_row_local_index(
    std::size_t                     g_row,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t&                    local_idx)
{
    const std::size_t nseg = seg_global_offs.size();
    for (std::size_t s = 0; s < nseg; ++s)
    {
        const std::size_t g0 = seg_global_offs[s];
        const std::size_t g1 = g0 + seg_lens[s];
        if (g0 <= g_row && g_row < g1)
        {
            local_idx = seg_local_offs[s] + (g_row - g0);
            return true;
        }
    }
    local_idx = 0;
    return false;
}

inline std::size_t nccl_first_active_local_index(
    std::size_t                     col,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t                     l_rows)
{
    const std::size_t nseg = seg_global_offs.size();
    for (std::size_t s = 0; s < nseg; ++s)
    {
        const std::size_t g0 = seg_global_offs[s];
        const std::size_t g1 = g0 + seg_lens[s];
        if (col < g0)
            return seg_local_offs[s];
        if (g0 <= col && col < g1)
            return seg_local_offs[s] + (col - g0);
    }
    return l_rows;
}

template <typename T>
inline void nccl_split_sync_fix_allreduce(
    cudaStream_t stream, const T* d_in, T* d_out, std::size_t count,
    ncclComm_t nccl_col_comm, double* d_hi, double* d_lo,
    cublasHandle_t cublas_handle, int rank,
    std::size_t audit_block_id, const char* audit_phase)
{
    const bool trace_on =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Trace);
    const std::size_t scalar_count =
        chase::linalg::internal::cuda::split_sync_scalar_count<T>(count);

    chase::linalg::internal::cuda::run_split_to_hilo<T>(stream, d_in, count,
                                                        d_hi, d_lo);
    chase::nccl::ncclAllReduceWrapper<double>(d_hi, d_hi, scalar_count, ncclSum,
                                              nccl_col_comm, &stream);
    chase::nccl::ncclAllReduceWrapper<double>(d_lo, d_lo, scalar_count, ncclSum,
                                              nccl_col_comm, &stream);
    chase::linalg::internal::cuda::run_renorm_hilo(stream, d_hi, d_lo,
                                                   scalar_count);
    chase::linalg::internal::cuda::run_merge_hilo_to_out<T>(
        stream, d_hi, d_lo, d_out, count);

    if (trace_on && scalar_count > 0 && rank == 0)
    {
        cublasPointerMode_t prev_mode;
        CHECK_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &prev_mode));
        CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

        double n_hi = 0.0, n_lo = 0.0;
        CHECK_CUBLAS_ERROR(cublasDnrm2(cublas_handle, static_cast<int>(scalar_count), d_hi, 1, &n_hi));
        CHECK_CUBLAS_ERROR(cublasDnrm2(cublas_handle, static_cast<int>(scalar_count), d_lo, 1, &n_lo));
        CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, prev_mode));

        const double denom = std::max(n_hi, std::numeric_limits<double>::min());
        const double error_ratio = n_lo / denom;
        /*std::ostringstream oss;
        oss << std::scientific << std::setprecision(6)
            << "[TRACE] HIPREC active (Split-Sync-Fix): phase=" << audit_phase
            << " [b=" << audit_block_id << "]"
            << " scalar_count=" << scalar_count
            << "  ||hi||_2=" << n_hi
            << "  ||lo||_2=" << n_lo
            << "  Error_Ratio=" << error_ratio << "\n";
        if (error_ratio < 1e-12)
        {
            oss << "[ADVISE] Numerical drift is well-suppressed. Suggest Ozaki MINIMAL split (d=2) to harvest H100 throughput.\n";
        }
        else
        {
            oss << "[ADVISE] Significant precision loss detected in reduction! Suggest Ozaki MORE splits (d>=4) to widen dynamic range.\n";
        }
        chase::GetLogger().Log(chase::LogLevel::Trace, "linalg", oss.str(), rank);
        */
    }
}

template <typename T>
inline double nccl_abs_value(const T& x)
{
    return std::abs(x);
}

template <>
inline double nccl_abs_value<std::complex<float>>(const std::complex<float>& x)
{
    return static_cast<double>(std::abs(x));
}

template <>
inline double nccl_abs_value<std::complex<double>>(const std::complex<double>& x)
{
    return std::abs(x);
}

inline bool nccl_should_validate_orthogonality()
{
    const char* env = std::getenv("CHASE_QR_CHECK_ORTHO");
    if (env == nullptr)
        return false;
    const std::string v(env);
    return (v == "1" || v == "true" || v == "TRUE" || v == "on" ||
            v == "ON");
}

inline std::size_t nccl_qr_sub_nb_env(const cuda_nccl::HouseQRTuning* tuning = nullptr,
                                      std::size_t fallback = 8)
{
    if (tuning != nullptr && tuning->panel_sub_nb > 0)
        return static_cast<std::size_t>(tuning->panel_sub_nb);
    const char* env = std::getenv("CHASE_QR_SUB_NB");
    if (env == nullptr) return fallback;
    const int v = std::atoi(env);
    if (v <= 0) return fallback;
    return static_cast<std::size_t>(v);
}

inline int nccl_qr_formq_chunks(const cuda_nccl::HouseQRTuning* tuning = nullptr,
                                int fallback = 1)
{
    if (tuning != nullptr && tuning->formq_chunks > 0)
        return tuning->formq_chunks;
    if (const char* e = std::getenv("CHASE_FORMQ_CHUNKS"))
    {
        const int v = std::atoi(e);
        if (v > 1) return v;
    }
    return fallback;
}

inline bool nccl_qr_timing_blocking(const cuda_nccl::HouseQRTuning* tuning = nullptr)
{
    if (tuning != nullptr && tuning->timing_blocking >= 0)
        return tuning->timing_blocking != 0;
    const char* e = std::getenv("CHASE_QR_TIMING_BLOCKING");
    if (e == nullptr) return false;
    const std::string v(e);
    return (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON");
}

inline bool nccl_qr_panel_hiprec(const cuda_nccl::HouseQRTuning* tuning = nullptr)
{
    if (tuning != nullptr && tuning->panel_hiprec >= 0)
        return tuning->panel_hiprec != 0;
#if CHASE_PANEL_HIPREC
    return true;
#else
    return false;
#endif
}

inline std::size_t nccl_qr_block_nb(const cuda_nccl::HouseQRTuning* tuning = nullptr,
                                    std::size_t fallback_nb = 32)
{
    if (tuning != nullptr && tuning->outer_block_nb > 0)
        return static_cast<std::size_t>(tuning->outer_block_nb);
    if (const char* e = std::getenv("CHASE_QR_OUTER_BLOCK_NB"))
    {
        const int v = std::atoi(e);
        if (v > 0) return static_cast<std::size_t>(v);
    }
    return fallback_nb;
}

template <typename T>
inline double nccl_orthogonality_tol()
{
    using RealT = chase::Base<T>;
    if constexpr (std::is_same<RealT, float>::value)
        return 1e-5;
    else
        return 1e-12;
}

template <typename T>
inline void nccl_validate_orthogonality(cublasHandle_t cublas_handle,
                                        T* Q,
                                        std::size_t l_rows,
                                        std::size_t n,
                                        std::size_t ldq,
                                        ncclComm_t nccl_col_comm,
                                        int rank)
{
    if (n == 0) return;

    cublasPointerMode_t old_mode;
    CHECK_CUBLAS_ERROR(cublasGetPointerMode(cublas_handle, &old_mode));
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));

    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CUBLAS_OP_C : CUBLAS_OP_T;

    T* d_qtq = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_qtq, n * n * sizeof(T)));
    
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_qtq, 0, n * n * sizeof(T), stream));

    const T one = T(1);
    const T zero = T(0);


    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
        cublas_handle, cublas_op_c, CUBLAS_OP_N, 
        n, n, l_rows, 
        &one, Q, ldq, Q, ldq, 
        &zero, d_qtq, n));

    /*
    T h_diag_part;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(&h_diag_part, d_qtq, sizeof(T), 
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    printf("[DEBUG] Rank %d local diag[0,0] contribution: %f\n", 
            rank, std::abs(h_diag_part));
    */

    chase::nccl::ncclAllReduceWrapper<T>(d_qtq, d_qtq, n * n, ncclSum,
                                         nccl_col_comm, &stream);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    if (rank == 0)
    {
        std::vector<T> h_qtq(n * n);
        CHECK_CUDA_ERROR(cudaMemcpy(h_qtq.data(), d_qtq, n * n * sizeof(T),
                                    cudaMemcpyDeviceToHost));
        
        double inf_norm = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            double row_sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
            {
                T v = h_qtq[i + j * n];
                if (i == j)
                    v -= T(1); 
                row_sum += std::abs(v);
            }
            inf_norm = std::max(inf_norm, row_sum);
        }

        std::ostringstream oss;
        oss << std::scientific << std::setprecision(6)
            << "[ORTHO] ||Q^H Q - I||_inf = " << inf_norm 
            << " (ncols=" << n << ", l_rows=" << l_rows << ")\n";
        
        if (inf_norm > nccl_orthogonality_tol<T>())
            oss << "[ORTHO][WARN] Orthogonality drift above tolerance. "
                << "Check index mapping or communication sync.\n";
        
        chase::GetLogger().Log(chase::LogLevel::Info, "linalg", oss.str(), rank);
    }

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, old_mode));
    CHECK_CUDA_ERROR(cudaFree(d_qtq));
}

template <typename T>
inline void nccl_trace_tmatrix_condition(
    const T* Tb, int jb, int nb, int rank, std::size_t audit_block_id)
{
    const bool trace_on =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Trace);
    if (!trace_on || rank != 0 || jb <= 0)
        return;

    double diag_max = 0.0;
    double diag_min = std::numeric_limits<double>::max();
    for (int i = 0; i < jb; ++i)
    {
        T host_diag{};
        CHECK_CUDA_ERROR(cudaMemcpy(&host_diag, Tb + i + i * nb, sizeof(T),
                                    cudaMemcpyDeviceToHost));
        const double a = nccl_abs_value(host_diag);
        diag_max = std::max(diag_max, a);
        if (a > 0.0)
            diag_min = std::min(diag_min, a);
    }
    const double safe_min = (diag_min == std::numeric_limits<double>::max())
                                ? std::numeric_limits<double>::min()
                                : std::max(diag_min, std::numeric_limits<double>::min());
    const double kappa_t = diag_max / safe_min;

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6)
        << "[TRACE] HIPREC active (T-build): [b=" << audit_block_id << "]"
        << " kappa(T)_diag_est=" << kappa_t
        << "  diag_max=" << diag_max
        << "  diag_min=" << safe_min << "\n";
    if (kappa_t > 1e10)
    {
        oss << "[ADVISE] Extremely ill-conditioned block! Heavy Ozaki splitting recommended for trailing update.\n";
    }
    chase::GetLogger().Log(chase::LogLevel::Trace, "linalg", oss.str(), rank);
}

inline void nccl_strict_stream_barrier(cudaStream_t stream)
{
#if STRICT_ORTHO_TBUILD
    cudaEvent_t ev = nullptr;
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
    CHECK_CUDA_ERROR(cudaEventRecord(ev, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(ev));
    CHECK_CUDA_ERROR(cudaEventDestroy(ev));
#else
    (void)stream;
#endif
}

} // namespace


//==============================================================================
// distributed_houseQR_panel_factor (NCCL)
//
// Factor columns [k, k+jb) of V in place (Householder panel). Writes tau to
// d_tau[k .. k+jb). Does NOT do form-T or trailing update. Caller provides
// workspace (d_w must have at least jb elements). Use this so panel strategy
// can be swapped (e.g. different algorithm per block).
//==============================================================================
template <typename T>
void cuda_nccl::distributed_houseQR_panel_factor(
    std::size_t n, std::size_t l_rows, std::size_t g_off, std::size_t ldv,
    T* V, std::size_t k, std::size_t jb, T* d_tau,
    cublasHandle_t cublas_handle,
    chase::Base<T>* d_real_scalar, T* d_T_scalar,
    T* d_one, T* d_zero, T* d_minus_one, T* d_panel_scalars, T* d_w,
    ncclComm_t nccl_col_comm,
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming* panel_timing)
{
    // Optional timing: when out_t_norm_ms != nullptr, record per-phase CUDA events and fill all five outputs (ms).
    using RealT = chase::Base<T>;
    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CUBLAS_OP_C : CUBLAS_OP_T;

    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));
    int col_rank = 0;
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &col_rank));
#if STRICT_TAU_BCAST
    int* d_owner_vote = nullptr;
    int* d_owner_sel = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_owner_vote, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_owner_sel, sizeof(int)));
#endif

    const bool enable_timing =
        (panel_timing != nullptr) &&
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);
    const bool do_timing = enable_timing;
    const std::size_t max_jb_timing = 128u;
    const std::size_t jb_ev = do_timing ? std::min(jb, max_jb_timing) : 0u;
    std::vector<cudaEvent_t> ev_start, ev_end;
    if (do_timing && jb_ev > 0)
    {
        ev_start.resize(jb_ev * 5u);
        ev_end.resize(jb_ev * 5u);
        for (std::size_t i = 0; i < jb_ev * 5u; ++i)
        {
            CHECK_CUDA_ERROR(cudaEventCreate(&ev_start[i]));
            CHECK_CUDA_ERROR(cudaEventCreate(&ev_end[i]));
        }
        panel_timing->norm_ms          = 0.f;
        panel_timing->scalar_kernel_ms = 0.f;
        panel_timing->allreduce_tau_ms = 0.f;
        panel_timing->scal_ms          = 0.f;
        panel_timing->trail_ms         = 0.f;
        panel_timing->panel_total_ms   = 0.f;
    }

    cudaEvent_t ev_panel_wall_start = nullptr, ev_panel_wall_end = nullptr;
    if (do_timing)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_panel_wall_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_panel_wall_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_panel_wall_start, stream));
    }

    for (std::size_t jj = 0; jj < jb; ++jj)
    {
        const std::size_t col       = k + jj;
        const bool        pivot_here = (g_off <= col && col < g_off + l_rows);
        const std::size_t pivot_loc  = pivot_here ? (col - g_off) : 0;

        // Audit path: when pivot is local, operate on the local tail
        // [pivot_loc, l_rows) explicitly (equivalent to logical row filtering).
        std::size_t rs = 0, vr = 0;
        if (col < g_off)
        {
            rs = 0;
            vr = l_rows;
        }
        else if (col < g_off + l_rows)
        {
            rs = pivot_loc;
            vr = l_rows - pivot_loc;
        }
        else
        {
            rs = 0;
            vr = 0;
        }

        // Phase 0: norm (dot + Allreduce nrm_sq)
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_start[jj * 5u + 0], stream));
        if (vr > 0)
        {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value)
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                    cublas_handle, vr, V + rs + col * ldv, 1,
                    V + rs + col * ldv, 1, d_real_scalar));
            else
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                    cublas_handle, vr, V + rs + col * ldv, 1,
                    V + rs + col * ldv, 1, d_T_scalar));
                chase::linalg::internal::cuda::run_extract_real_part_from_scalar<T, RealT>(
                    stream, d_T_scalar, d_real_scalar);
            }
        }
        else
        {
            CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_scalar, 0, sizeof(RealT), stream)); 
        }
        chase::nccl::ncclAllReduceWrapper<RealT>(d_real_scalar, d_real_scalar, 1, ncclSum, nccl_col_comm, &stream);

        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_end[jj * 5u + 0], stream));

        // Phase 1: scalar kernel
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_start[jj * 5u + 1], stream));
        T* d_x0_col = pivot_here ? (V + pivot_loc + col * ldv) : d_zero;
        T* d_inv_denom   = d_panel_scalars + 0;
        T* d_neg_beta   = d_panel_scalars + 1;
        T* d_denom_bcast= d_panel_scalars + 2;
        T* d_saved_rkk  = d_panel_scalars + 3;
        chase::linalg::internal::cuda::run_householder_scalar_kernel<T, RealT>(
            stream, pivot_here ? 1 : 0, d_x0_col, d_real_scalar,
            d_tau + col, d_inv_denom, d_neg_beta, d_denom_bcast, d_saved_rkk);
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_end[jj * 5u + 1], stream));

        // Phase 2: scalar exchange (strict mode uses owner broadcast for bitwise consistency)
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_start[jj * 5u + 2], stream));
#if STRICT_TAU_BCAST
        const int h_vote = pivot_here ? (col_rank + 1) : 0;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_owner_vote, &h_vote, sizeof(int),
                                         cudaMemcpyHostToDevice, stream));
        CHECK_NCCL_ERROR(ncclAllReduce(d_owner_vote, d_owner_sel, 1, ncclInt32,
                                       ncclMax, nccl_col_comm, stream));
        int h_owner_plus = 0;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(&h_owner_plus, d_owner_sel, sizeof(int),
                                         cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        const int owner_rank = (h_owner_plus > 0) ? (h_owner_plus - 1) : 0;
        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper<T>(
            d_tau + col, 1, owner_rank, nccl_col_comm, &stream));
        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper<T>(
            d_denom_bcast, 1, owner_rank, nccl_col_comm, &stream));
#else
        chase::nccl::ncclAllReduceWrapper<T>(d_tau + col, d_T_scalar, 1, ncclSum, nccl_col_comm, &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_tau + col, d_T_scalar, sizeof(T), cudaMemcpyDeviceToDevice, stream));
        chase::nccl::ncclAllReduceWrapper<T>(d_denom_bcast, d_denom_bcast, 1, ncclSum, nccl_col_comm, &stream);
#endif
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_end[jj * 5u + 2], stream));

        // Phase 3: guarded scal (inv_denom, scale, set pivot 1)
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_start[jj * 5u + 3], stream));
        chase::linalg::internal::cuda::run_inv_denom_from_denom_bcast<T>(
            stream, d_denom_bcast, d_inv_denom);
        if (vr > 0)
        {
            const int pivot_loc_rel = 0;
            chase::linalg::internal::cuda::run_guarded_scaling<T>(
                stream,
                static_cast<int>(vr), d_tau + col, d_inv_denom, V + rs + col * ldv,
                pivot_here, pivot_loc_rel, d_neg_beta);
        }

        if (pivot_here)
            CHECK_CUDA_ERROR(cudaMemcpyAsync(V + pivot_loc + col * ldv, d_one, sizeof(T), cudaMemcpyDeviceToDevice, stream));
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_end[jj * 5u + 3], stream));

        // Phase 4: trail (memset, gemm w, Allreduce w, rank-1 update, restore pivot)
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_start[jj * 5u + 4], stream));
        const std::size_t n_panel_trail = (k + jb) - col - 1;
        if (n_panel_trail > 0)
        {
            CHECK_CUDA_ERROR(cudaMemsetAsync(d_w, 0, n_panel_trail * sizeof(T), stream));
            if (vr > 0)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    n_panel_trail, 1, vr, d_one,
                    V + rs + (col + 1) * ldv, ldv,
                    V + rs + col * ldv, ldv,
                    d_zero, d_w, n_panel_trail));
            }
            chase::nccl::ncclAllReduceWrapper<T>(d_w, d_w, n_panel_trail, ncclSum, nccl_col_comm, &stream);
            if (vr > 0)
            {
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T_scalar, d_tau + col, sizeof(T), cudaMemcpyDeviceToDevice, stream));
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                    cublas_handle, 1, d_minus_one, d_T_scalar, 1));
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, cublas_op_c,
                    vr, n_panel_trail, 1, d_T_scalar,
                    V + rs + col * ldv, ldv, d_w, n_panel_trail, d_one,
                    V + rs + (col + 1) * ldv, ldv));
            }
        }

        if (pivot_here)
            CHECK_CUDA_ERROR(cudaMemcpyAsync(V + pivot_loc + col * ldv, d_saved_rkk, sizeof(T), cudaMemcpyDeviceToDevice, stream));
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_end[jj * 5u + 4], stream));
    }

    if (do_timing)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_panel_wall_end, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_panel_wall_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&panel_timing->panel_total_ms,
            ev_panel_wall_start, ev_panel_wall_end));
        cudaEventDestroy(ev_panel_wall_start);
        cudaEventDestroy(ev_panel_wall_end);
        ev_panel_wall_start = ev_panel_wall_end = nullptr;
    }

    if (do_timing && jb_ev > 0)
    {
        for (std::size_t p = 0; p < 5u; ++p)
        {
            float sum_ms = 0.f;
            for (std::size_t jj = 0; jj < jb_ev; ++jj)
            {
                float t_ms = 0.f;
                CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_ms, ev_start[jj * 5u + p], ev_end[jj * 5u + p]));
                sum_ms += t_ms;
            }
            if (p == 0) panel_timing->norm_ms = sum_ms;
            else if (p == 1) panel_timing->scalar_kernel_ms = sum_ms;
            else if (p == 2) panel_timing->allreduce_tau_ms = sum_ms;
            else if (p == 3) panel_timing->scal_ms = sum_ms;
            else panel_timing->trail_ms = sum_ms;
        }
        for (std::size_t i = 0; i < ev_start.size(); ++i)
        {
            cudaEventDestroy(ev_start[i]);
            cudaEventDestroy(ev_end[i]);
        }
    }
#if STRICT_TAU_BCAST
    cudaFree(d_owner_vote);
    cudaFree(d_owner_sel);
#endif
}

//==============================================================================
// distributed_houseQR_panel_factor_block_cyclic_1d_columns (NCCL)
//
// Column range [jj_begin, jj_end) within panel starting at global column k.
// jb_panel_total is the full panel width used only for Phase 4 trailing extent:
// n_panel_trail = (k + jb_panel_total) - col - 1.
// Caller must run pre_clean and validation.
//==============================================================================
template <typename T>
void cuda_nccl::distributed_houseQR_panel_factor_block_cyclic_1d_columns(
    std::size_t n, std::size_t m_global,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t ldv, T* V, std::size_t k, std::size_t jj_begin,
    std::size_t jj_end, std::size_t jb_panel_total, std::size_t nb_dist,
    T* d_tau, cublasHandle_t cublas_handle,
    chase::Base<T>* d_real_scalar, T* d_T_scalar, T* d_one, T* d_zero,
    T* d_minus_one, T* d_panel_scalars, T* d_w, ncclComm_t nccl_col_comm,
    int col_rank, int col_size, std::size_t l_rows,
    const std::uint64_t* d_row_global, T* d_r_diag,
    const HouseQRTuning* tuning,
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming* panel_timing,
    T* d_sub_workspace, std::size_t d_sub_workspace_elems)
{
    (void)n;
    (void)seg_global_offs;
    (void)seg_local_offs;
    (void)seg_lens;
    (void)panel_timing;

    using RealT = chase::Base<T>;
    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CUBLAS_OP_C
            : CUBLAS_OP_T;
    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));

    const bool trace_audit =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Trace);
    const std::size_t nb_bc = nb_dist > 0 ? nb_dist : 1;

    const std::size_t span = jj_end - jj_begin;
    const std::size_t sub_nb = nccl_qr_sub_nb_env(tuning, 8);
    const std::size_t max_remain_cols = span;
    constexpr std::size_t kInvalidCol =
        std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> bc_col(span, kInvalidCol);
    std::vector<int> bc_owner(span);
    std::vector<unsigned char> bc_pivot_here(span);
    std::vector<std::size_t> bc_pivot_loc(span);
    std::vector<std::size_t> bc_active_start(span);
    std::vector<std::size_t> bc_vr(span);
    T* d_comm_pack = nullptr;   // [tau, denom]
    T* d_sub_s = nullptr;       // sub_nb x sub_nb
    T* d_sub_t = nullptr;       // sub_nb x sub_nb
    T* d_sub_w = nullptr;       // sub_nb x max_remain_cols
    T* d_sub_tw = nullptr;      // sub_nb x max_remain_cols
    if (span > 0)
    {
        const std::size_t need_elems =
            2 + 2 * sub_nb * sub_nb + 2 * sub_nb * max_remain_cols;
        if (d_sub_workspace == nullptr || d_sub_workspace_elems < need_elems)
            throw std::runtime_error(
                "distributed_houseQR_panel_factor_block_cyclic_1d_columns: insufficient sub-workspace");
        d_comm_pack = d_sub_workspace;
        d_sub_s = d_comm_pack + 2;
        d_sub_t = d_sub_s + sub_nb * sub_nb;
        d_sub_w = d_sub_t + sub_nb * sub_nb;
        d_sub_tw = d_sub_w + sub_nb * max_remain_cols;
    }

    for (std::size_t idx = 0; idx < span; ++idx)
    {
        const std::size_t jj = jj_begin + idx;
        const std::size_t col = k + jj;
        if (col >= m_global)
            continue;

        const std::size_t block_id = col / nb_bc;
        const int         owner_rank =
            static_cast<int>(block_id % static_cast<std::size_t>(col_size));
        bool        pivot_here = (owner_rank == col_rank);
        std::size_t pivot_loc  = 0;
        if (pivot_here)
        {
            const std::size_t local_block_id =
                block_id / static_cast<std::size_t>(col_size);
            pivot_loc = local_block_id * nb_bc + (col % nb_bc);
            if (pivot_loc >= l_rows)
            {
                pivot_here = false;
                pivot_loc  = 0;
            }
        }

        const std::size_t b0 = block_id;
        const int         rem =
            static_cast<int>(b0 % static_cast<std::size_t>(col_size));
        const int shift = (col_rank - rem + col_size) % col_size;
        const std::size_t first_local_block =
            b0 + static_cast<std::size_t>(shift);
        std::size_t active_row_start = 0;
        if (pivot_here)
        {
            const std::size_t local_block_id =
                b0 / static_cast<std::size_t>(col_size);
            active_row_start = local_block_id * nb_bc + (col % nb_bc);
        }
        else
        {
            const std::size_t local_block_id =
                first_local_block / static_cast<std::size_t>(col_size);
            active_row_start = local_block_id * nb_bc;
        }
        if (active_row_start >= l_rows)
            active_row_start = l_rows;
        const std::size_t vr =
            active_row_start < l_rows ? (l_rows - active_row_start) : 0;

        bc_col[idx]            = col;
        bc_owner[idx]          = owner_rank;
        bc_pivot_here[idx]     = pivot_here ? 1 : 0;
        bc_pivot_loc[idx]      = pivot_loc;
        bc_active_start[idx]   = active_row_start;
        bc_vr[idx]             = vr;
    }

    for (std::size_t idx = 0; idx < span; ++idx)
    {
        const std::size_t col = bc_col[idx];
        if (col == kInvalidCol)
            continue;

        const std::size_t jj             = jj_begin + idx;
        const int         owner_rank   = bc_owner[idx];
        const bool        pivot_here   = bc_pivot_here[idx] != 0;
        const std::size_t pivot_loc    = bc_pivot_loc[idx];
        const std::size_t active_row_start = bc_active_start[idx];
        const std::size_t vr           = bc_vr[idx];
        T* const          v_col_active = V + active_row_start + col * ldv;

        // NOTE [zero-size local work in block-cyclic panel]:
        // vr can be 0 on ranks that have no active rows for this column.
        // We intentionally still issue the local BLAS call and rely on cuBLAS
        // zero-size semantics instead of pre-zeroing outputs via cudaMemsetAsync.
        // This keeps the code path compact while preserving collective ordering:
        // all ranks still execute the same NCCL collectives in the same order.
        // Tradeoff: explicit memset would be more defensive against stale-buffer
        // risks on edge backends; current path prioritizes simplicity/perf.
        if constexpr (std::is_same<T, float>::value ||
                      std::is_same<T, double>::value)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, vr, v_col_active, 1, v_col_active, 1,
                d_real_scalar));
        }
        else
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                cublas_handle, vr, v_col_active, 1, v_col_active, 1,
                d_T_scalar));
            chase::linalg::internal::cuda::run_extract_real_part_from_scalar<T, RealT>(
                stream, d_T_scalar, d_real_scalar);
        }
        chase::nccl::ncclAllReduceWrapper<RealT>(
            d_real_scalar, d_real_scalar, 1, ncclSum, nccl_col_comm, &stream);

        T* d_x0_col      = pivot_here ? (V + pivot_loc + col * ldv) : d_zero;
        T* d_inv_denom   = d_panel_scalars + 0;
        T* d_neg_beta    = d_panel_scalars + 1;
        T* d_denom_bcast = d_comm_pack + 1;
        chase::linalg::internal::cuda::run_householder_scalar_kernel<T, RealT>(
            stream, pivot_here ? 1 : 0, d_x0_col, d_real_scalar, d_comm_pack + 0,
            d_inv_denom, d_neg_beta, d_denom_bcast, d_panel_scalars + 3);

        // Stage-1 fusion: synchronize [tau, denom] in one collective.
#if STRICT_TAU_BCAST
        CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper<T>(
            d_comm_pack, 2, owner_rank, nccl_col_comm, &stream));
#else
        CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
            d_comm_pack, d_comm_pack, 2, ncclSum, nccl_col_comm, &stream));
#endif
        chase::linalg::internal::cuda::run_copy_scalar_kernel<T>(
            stream, d_comm_pack + 0, d_tau + col);

        chase::linalg::internal::cuda::run_fused_householder_finish_kernel<T>(
            stream, pivot_here ? 1 : 0, V + col * ldv, static_cast<int>(ldv),
            d_row_global, static_cast<int>(l_rows), static_cast<std::uint64_t>(col),
            static_cast<int>(pivot_loc), static_cast<int>(active_row_start),
            d_denom_bcast, d_inv_denom, d_panel_scalars + 3,
            d_r_diag ? (d_r_diag + col) : nullptr);

        // Inside sub-block: only update remaining columns within current sub-block.
        const std::size_t sub_begin = (idx / sub_nb) * sub_nb;
        const std::size_t sub_end = std::min(sub_begin + sub_nb, span);
        const std::size_t local_trail = sub_end - (idx + 1);
        if (local_trail > 0)
        {
            // NOTE [local_trail + vr==0]:
            // For ranks with vr==0, this GEMM has K=0 and contributes nothing.
            // We intentionally rely on cuBLAS zero-size behavior and then still
            // run NCCL AllReduce on d_w so collective sequence stays identical
            // across ranks. Avoid guarding this allreduce on vr.
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, cublas_op_c, CUBLAS_OP_N, local_trail, 1, vr,
                d_one, V + active_row_start + (col + 1) * ldv, ldv,
                v_col_active, ldv, d_zero, d_w, local_trail));
            chase::nccl::ncclAllReduceWrapper<T>(
                d_w, d_w, local_trail, ncclSum, nccl_col_comm, &stream);
            chase::linalg::internal::cuda::run_copy_scalar_kernel<T>(
                stream, d_tau + col, d_T_scalar);
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, 1, d_minus_one, d_T_scalar, 1));
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, CUBLAS_OP_N, cublas_op_c, vr, local_trail, 1,
                d_T_scalar, v_col_active, ldv, d_w, local_trail, d_one,
                V + active_row_start + (col + 1) * ldv, ldv));
        }

        // Sub-block barrier: one BLAS-3 WY update for remaining panel columns.
        if ((idx + 1) == sub_end)
        {
            const std::size_t swidth = sub_end - sub_begin;
            const std::size_t sub_first_col = k + (jj_begin + sub_begin);
            const std::size_t after_sub_col = sub_first_col + swidth;
            const std::size_t remain_cols = (k + jb_panel_total) - after_sub_col;
            if (remain_cols > 0)
            {
                // NOTE [sub-WY with l_rows==0]:
                // l_rows can be 0 on some ranks in pathological layouts.
                // Here we keep unconditional local GEMM/collective structure and
                // rely on zero-size GEMM semantics for local no-op contribution.
                // This is safe for collective ordering (global guard is remain_cols)
                // but less explicit than a memset-to-zero fallback.
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    swidth, swidth, l_rows, d_one,
                    V + sub_first_col * ldv, ldv,
                    V + sub_first_col * ldv, ldv, d_zero, d_sub_s, swidth));
                chase::nccl::ncclAllReduceWrapper<T>(
                    d_sub_s, d_sub_s, swidth * swidth, ncclSum, nccl_col_comm, &stream);
                chase::linalg::internal::cuda::run_compute_T_block<T>(
                    stream, d_sub_t, d_sub_s, d_tau + sub_first_col,
                    static_cast<int>(swidth), static_cast<int>(swidth));

                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    swidth, remain_cols, l_rows, d_one,
                    V + sub_first_col * ldv, ldv,
                    V + after_sub_col * ldv, ldv, d_zero, d_sub_w, swidth));
                chase::nccl::ncclAllReduceWrapper<T>(
                    d_sub_w, d_sub_w, swidth * remain_cols, ncclSum, nccl_col_comm, &stream);

                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                    swidth, remain_cols, swidth, d_one,
                    d_sub_t, swidth, d_sub_w, swidth, d_zero, d_sub_tw, swidth));
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    l_rows, remain_cols, swidth, d_minus_one,
                    V + sub_first_col * ldv, ldv, d_sub_tw, swidth, d_one,
                    V + after_sub_col * ldv, ldv));
            }
        }

        /*if (trace_audit && pivot_here && d_r_diag != nullptr)
        {
            std::ostringstream oss;
            oss << std::scientific << std::setprecision(12)
                << "[subspace-audit] jj=" << jj << " col=" << col
                << " owner_rank=" << owner_rank << " col_rank=" << col_rank
                << " pivot_loc=" << pivot_loc
                << " active_row_start=" << active_row_start
                << " active_rows=" << vr
                << " (gpu norm/tau/r_diag not read; no sync)\n";
            chase::GetLogger().Log(chase::LogLevel::Trace, "linalg", oss.str(),
                                   col_rank);
        }
        */
    }
}

//==============================================================================
// distributed_houseQR_panel_factor_block_cyclic_1d (NCCL)
//
// Same numerical panel as distributed_houseQR_panel_factor for 1-D block-cyclic
// rows described by contiguous segments. After each column's intra-panel update,
// run_split_and_pad_v_column enforces WY-clean V (zero global rows above pivot,
// diagonal 1, R_kk peeled to d_r_diag when non-null) so trailing GEMMs need no
// save/restore and may use one cuBLAS call over full local height.
// Uses column-wise panel factorization only (non-recursive).
//==============================================================================
template <typename T>
void cuda_nccl::distributed_houseQR_panel_factor_block_cyclic_1d(
    std::size_t n, std::size_t m_global,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t ldv, T* V, std::size_t k, std::size_t jb, std::size_t nb_dist,
    T* d_tau,
    cublasHandle_t cublas_handle,
    chase::Base<T>* d_real_scalar, T* d_T_scalar,
    T* d_one, T* d_zero, T* d_minus_one, T* d_panel_scalars, T* d_w,
    ncclComm_t nccl_col_comm,
    std::size_t l_rows, const std::uint64_t* d_row_global, T* d_r_diag,
    const HouseQRTuning* tuning,
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming* panel_timing)
{
    const std::size_t nseg = seg_global_offs.size();
    if (nseg != seg_local_offs.size() || nseg != seg_lens.size())
        throw std::runtime_error(
            "distributed_houseQR_panel_factor_block_cyclic_1d: segment vector sizes must match");
    const std::size_t l_rows_expect =
        nseg == 0 ? 0 : (seg_local_offs.back() + seg_lens.back());
    if (l_rows != l_rows_expect)
        throw std::runtime_error(
            "distributed_houseQR_panel_factor_block_cyclic_1d: l_rows mismatch");
    if (l_rows > 0 && d_row_global == nullptr)
        throw std::runtime_error(
            "distributed_houseQR_panel_factor_block_cyclic_1d: d_row_global is null");

    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));

    const bool do_panel_wall =
        (panel_timing != nullptr) &&
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);
    int col_rank = 0, col_size = 1;
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &col_rank));
    CHECK_NCCL_ERROR(ncclCommCount(nccl_col_comm, &col_size));
    if (panel_timing != nullptr)
    {
        panel_timing->norm_ms = panel_timing->scalar_kernel_ms = 0.f;
        panel_timing->allreduce_tau_ms = panel_timing->scal_ms = panel_timing->trail_ms = 0.f;
        panel_timing->panel_total_ms = 0.f;
    }

    cudaEvent_t ev_bc_panel_start = nullptr, ev_bc_panel_end = nullptr;
    if (do_panel_wall)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_bc_panel_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_bc_panel_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_bc_panel_start, stream));
    }

    if (k < m_global && jb > 0 && l_rows > 0)
    {
        const std::size_t jb_cols =
            std::min(jb, static_cast<std::size_t>(m_global - k));
        const int jb_i =
            jb_cols > static_cast<std::size_t>(INT_MAX)
                ? INT_MAX
                : static_cast<int>(jb_cols);
        chase::linalg::internal::cuda::run_panel_pre_clean<T>(
            stream, V + k * ldv, ldv, d_row_global, static_cast<int>(l_rows), k,
            jb_i);
    }

    if (jb > 0)
    {
        const std::size_t sub_nb = nccl_qr_sub_nb_env(tuning, 8);
        const std::size_t sub_workspace_elems =
            2 + 2 * sub_nb * sub_nb + 2 * sub_nb * jb;
        T* d_sub_workspace = nullptr;
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sub_workspace,
                                    sub_workspace_elems * sizeof(T)));
        distributed_houseQR_panel_factor_block_cyclic_1d_columns<T>(
            n, m_global, seg_global_offs, seg_local_offs, seg_lens, ldv, V, k,
            0, jb, jb, nb_dist, d_tau, cublas_handle, d_real_scalar, d_T_scalar,
            d_one, d_zero, d_minus_one, d_panel_scalars, d_w, nccl_col_comm,
            col_rank, col_size, l_rows, d_row_global, d_r_diag, tuning,
            panel_timing, d_sub_workspace, sub_workspace_elems);
        CHECK_CUDA_ERROR(cudaFree(d_sub_workspace));
    }
    // jb == 0 is an intentional no-op.

    if (do_panel_wall)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_bc_panel_end, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_bc_panel_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&panel_timing->panel_total_ms,
            ev_bc_panel_start, ev_bc_panel_end));
        cudaEventDestroy(ev_bc_panel_start);
        cudaEventDestroy(ev_bc_panel_end);
    }
}

//==============================================================================
// distributed_houseQR_formQ (NCCL — V on device, NCCL collectives only)
//
// Self-contained distributed Householder QR. V is device pointer; on exit holds
// the first n columns of Q. Uses NCCL for all reduce operations.
//==============================================================================
template <typename T>
void cuda_nccl::distributed_houseQR_formQ(std::size_t m_global,
                                             std::size_t n,
                                             std::size_t l_rows,
                                             std::size_t g_off,
                                             std::size_t ldv,
                                             T*          V,
                                             MPI_Comm    mpi_comm,
                                             cublasHandle_t cublas_handle,
                                             T* d_workspace,
                                             std::size_t lwork_elems,
                                             ncclComm_t nccl_col_comm,
                                             const HouseQRTuning* tuning)
{
    using RealT = chase::Base<T>;

    if (n == 0)
        return;

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));

    int rank = 0;
    MPI_Comm_rank(mpi_comm, &rank);

    const bool enable_timing =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);
    const bool timing_blocking = nccl_qr_timing_blocking(tuning);

    cudaEvent_t ev_start = nullptr, ev_panel_end = nullptr, ev_formQ_end = nullptr;
    if (enable_timing)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_panel_end));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_formQ_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_start, stream));
    }

    RealT* d_real_scalar = nullptr;
    T* d_T_scalar       = nullptr;
    T* d_one            = nullptr;
    T* d_zero           = nullptr;
    T* d_minus_one      = nullptr;
    T* d_panel_scalars  = nullptr; // [inv_denom, neg_beta, denom_bcast, saved_rkk]
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar, sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_one, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_zero, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_minus_one, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars, 4 * sizeof(T)));
    const T one  = T(1);
    const T zero = T(0);
    const T minus_one = T(-1);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_one, &one, sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_zero, &zero, sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_minus_one, &minus_one, sizeof(T), cudaMemcpyHostToDevice, stream));

    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CUBLAS_OP_C : CUBLAS_OP_T;

    const std::size_t need_elems = 7 + 2 * n + l_rows * n; // 3 + 4 panel scalars + 2n + l_rows*n
    T* d_V   = V;
    T* d_w   = nullptr;
    T* d_tau = nullptr;
    T* d_VH  = nullptr;
    bool alloc_own = (d_workspace == nullptr || lwork_elems < need_elems);
    if (alloc_own)
    {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w, n * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tau, n * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_VH, l_rows * n * sizeof(T)));
    }
    else
    {
        d_one          = d_workspace;
        d_zero         = d_workspace + 1;
        d_minus_one    = d_workspace + 2;
        d_panel_scalars= d_workspace + 3;
        d_w            = d_workspace + 7;
        d_tau          = d_workspace + 7 + n;
        d_VH           = d_workspace + 7 + 2 * n;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_one, &one, sizeof(T), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_zero, &zero, sizeof(T), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_minus_one, &minus_one, sizeof(T), cudaMemcpyHostToDevice, stream));
    }

    cuda_nccl::distributed_houseQR_panel_factor<T>(
        n, l_rows, g_off, ldv, d_V, 0, n, d_tau,
        cublas_handle,
        d_real_scalar, d_T_scalar, d_one, d_zero, d_minus_one, d_panel_scalars, d_w,
        nccl_col_comm,
        nullptr);

    if (enable_timing)
        CHECK_CUDA_ERROR(cudaEventRecord(ev_panel_end, stream));

    chase::linalg::internal::cuda::t_lacpy('A', l_rows, n, d_V, ldv, d_VH, l_rows, &stream);

    CHECK_CUDA_ERROR(cudaMemsetAsync(d_V, 0, ldv * n * sizeof(T), stream));
    for (std::size_t c = 0; c < n; ++c)
    {
        if (g_off <= c && c < g_off + l_rows)
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_V + (c - g_off) + c * ldv, d_one,
                                        sizeof(T), cudaMemcpyDeviceToDevice, stream));
    }

    T* d_saved_diag_slot = d_panel_scalars + 3;

    for (std::size_t jj = 0; jj < n; ++jj)
    {
        const std::size_t j = n - 1 - jj;
        const std::size_t n_cols = n - j;

        std::size_t rs, vr;
        if (j < g_off)             { rs = 0;          vr = l_rows; }
        else if (j < g_off+l_rows) { rs = j - g_off;  vr = l_rows - rs; }
        else                       { rs = 0;          vr = 0; }

        bool pivot_here = (g_off <= j && j < g_off + l_rows);
        if (pivot_here)
        {
            std::size_t ploc = j - g_off;
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_saved_diag_slot, d_VH + ploc + j * l_rows,
                sizeof(T), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_VH + ploc + j * l_rows, d_one,
                sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_w, 0, n_cols * sizeof(T), stream));
        if (vr > 0)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, cublas_op_c, CUBLAS_OP_N,
                n_cols, 1, vr, d_one,
                d_V + rs + j * ldv, ldv,
                d_VH + rs + j * l_rows, l_rows,
                d_zero, d_w, n_cols));
        }
        chase::nccl::ncclAllReduceWrapper<T>(
            d_w, d_w, n_cols, ncclSum, nccl_col_comm, &stream);

        if (vr > 0)
        {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T_scalar, d_tau + j, sizeof(T), cudaMemcpyDeviceToDevice, stream));
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                cublas_handle, 1, d_minus_one, d_T_scalar, 1));
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, CUBLAS_OP_N, cublas_op_c,
                vr, n_cols, 1, d_T_scalar,
                d_VH + rs + j * l_rows, l_rows,
                d_w, n_cols, d_one,
                d_V + rs + j * ldv, ldv));
        }

        if (pivot_here){
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_VH + (j - g_off) + j * l_rows, d_saved_diag_slot,
            sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
    }

    if (enable_timing)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_formQ_end, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_formQ_end));
        float t_panel_ms = 0.f, t_formQ_ms = 0.f, t_total_ms = 0.f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_panel_ms, ev_start, ev_panel_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_formQ_ms, ev_panel_end, ev_formQ_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_total_ms, ev_start, ev_formQ_end));
        if (rank == 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "[Householder QR unblocked] Total: " << (t_total_ms / 1000.0) << " s  "
                << "Panel: " << (t_panel_ms / 1000.0) << " s  "
                << "FormQ: " << (t_formQ_ms / 1000.0) << " s  "
                << "(n=" << n << " l_rows=" << l_rows << ")\n";
            chase::GetLogger().Log(chase::LogLevel::Trace, "linalg", oss.str(), rank);
        }
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_panel_end);
        cudaEventDestroy(ev_formQ_end);
    }

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));

    cudaFree(d_real_scalar);
    cudaFree(d_T_scalar);
    if (alloc_own)
    {
        cudaFree(d_one);
        cudaFree(d_zero);
        cudaFree(d_minus_one);
        cudaFree(d_panel_scalars);
        cudaFree(d_w);
        cudaFree(d_tau);
        cudaFree(d_VH);
    }
}

//==============================================================================
// houseQR_panel_factor_1d_columns_impl
//
// Block (contiguous 1D) analog of
// distributed_houseQR_panel_factor_block_cyclic_1d_columns.
// Rank owns global rows [g_off, g_off+l_rows) contiguously; d_row_global[i] =
// g_off+i (linear map built by caller).  Uses:
//   - fused [tau, denom] NCCL collective (1 AllReduce of 2 elements)
//   - run_fused_householder_finish_kernel for physical-clean WY representation
//   - sub_nb=8 intra-sub-block rank-1 updates + BLAS-3 WY at sub-block boundary
// After each column the panel V[:,col] is physically clean (zeros above global
// pivot, pivot=1), enabling full-height GEMMs in the caller without save/restore.
//==============================================================================
template <typename T>
static void houseQR_panel_factor_1d_columns_impl(
    std::size_t m_global, std::size_t g_off, std::size_t l_rows,
    std::size_t ldv, T* V, std::size_t k,
    std::size_t jj_begin, std::size_t jj_end, std::size_t jb_panel_total,
    T* d_tau, cublasHandle_t cublas_handle,
    chase::Base<T>* d_real_scalar, T* d_T_scalar,
    T* d_one, T* d_zero, T* d_minus_one, T* d_panel_scalars, T* d_w,
    ncclComm_t nccl_col_comm,
    const std::uint64_t* d_row_global, T* d_r_diag,
    T* d_sub_workspace, std::size_t d_sub_workspace_elems,
    double* d_split_hi, double* d_split_lo, int rank, std::size_t block_id,
    const cuda_nccl::HouseQRTuning* tuning)
{
    using RealT = chase::Base<T>;
    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CUBLAS_OP_C : CUBLAS_OP_T;
    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));
    const bool qr_hang_trace = []() {
        const char* e = std::getenv("CHASE_QR_HANG_TRACE");
        if (e == nullptr) return false;
        const std::string v(e);
        return (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON");
    }();
    int nccl_rank = -1;
    if (qr_hang_trace)
        CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &nccl_rank));
    auto panel_trace = [&](const std::string& msg) {
        if (!qr_hang_trace) return;
        std::cout << "[QR-HANG-TRACE][panel-1d-cols][rank=" << nccl_rank << "] "
                  << "k=" << k << " jj_range=[" << jj_begin << "," << jj_end << ") "
                  << msg << std::endl;
    };

    const std::size_t span = jj_end - jj_begin;
    const std::size_t sub_nb = nccl_qr_sub_nb_env(tuning, 8);
    const bool use_panel_hiprec = nccl_qr_panel_hiprec(tuning);
    const std::size_t max_remain_cols = span;
    T* d_comm_pack = nullptr;  // [tau, denom] — 2 elements
    T* d_sub_s     = nullptr;  // sub_nb x sub_nb
    T* d_sub_t     = nullptr;  // sub_nb x sub_nb
    T* d_sub_w     = nullptr;  // sub_nb x max_remain_cols
    T* d_sub_tw    = nullptr;  // sub_nb x max_remain_cols
    if (span > 0)
    {
        const std::size_t need_elems =
            2 + 2 * sub_nb * sub_nb + 2 * sub_nb * max_remain_cols;
        if (d_sub_workspace == nullptr || d_sub_workspace_elems < need_elems)
            throw std::runtime_error(
                "houseQR_panel_factor_1d_columns_impl: insufficient sub-workspace");
        d_comm_pack = d_sub_workspace;
        d_sub_s     = d_comm_pack + 2;
        d_sub_t     = d_sub_s  + sub_nb * sub_nb;
        d_sub_w     = d_sub_t  + sub_nb * sub_nb;
        d_sub_tw    = d_sub_w  + sub_nb * max_remain_cols;
    }

    for (std::size_t idx = 0; idx < span; ++idx)
    {
        const std::size_t jj  = jj_begin + idx;
        const std::size_t col = k + jj;
        if (col >= m_global) continue;
        panel_trace("col=" + std::to_string(col) + " idx=" +
                    std::to_string(idx) + " SEQ-BEGIN");

        // Block layout: pivot is at local row col-g_off on the rank that owns it.
        const bool        pivot_here = (g_off <= col && col < g_off + l_rows);
        const std::size_t pivot_loc  = pivot_here ? (col - g_off) : 0;

        // Active rows (global index >= col):
        //   col < g_off          -> all l_rows rows active (rank is below pivot)
        //   g_off <= col < g_off+l_rows -> rows [col-g_off, l_rows)
        //   col >= g_off+l_rows  -> no active rows (rank is above pivot)
        std::size_t active_row_start = 0;
        if (col < g_off)
            active_row_start = 0;
        else if (col < g_off + l_rows)
            active_row_start = col - g_off;
        else
            active_row_start = l_rows;
        const std::size_t vr         = active_row_start < l_rows
                                           ? (l_rows - active_row_start) : 0;
        T* const          v_col_active = V + active_row_start + col * ldv;

        // Phase 0: local dot on active rows + AllReduce norm^2
        if (vr > 0)
        {
            if constexpr (std::is_same<T, float>::value ||
                          std::is_same<T, double>::value)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                    cublas_handle, vr, v_col_active, 1, v_col_active, 1,
                    d_real_scalar));
            }
            else
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTdot(
                    cublas_handle, vr, v_col_active, 1, v_col_active, 1,
                    d_T_scalar));
                chase::linalg::internal::cuda::run_extract_real_part_from_scalar<T, RealT>(
                    stream, d_T_scalar, d_real_scalar);
            }
        }
        else
        {
            CHECK_CUDA_ERROR(cudaMemsetAsync(d_real_scalar, 0, sizeof(RealT), stream));
        }
        panel_trace("col=" + std::to_string(col) + " SEQ-NORM enter");
        chase::nccl::ncclAllReduceWrapper<RealT>(
            d_real_scalar, d_real_scalar, 1, ncclSum, nccl_col_comm, &stream);
        panel_trace("col=" + std::to_string(col) + " SEQ-NORM return");

        // Phase 1: scalar kernel -> tau (in d_comm_pack[0]), denom (in d_comm_pack[1])
        T* d_x0_col      = pivot_here ? (V + pivot_loc + col * ldv) : d_zero;
        T* d_inv_denom   = d_panel_scalars + 0;
        T* d_neg_beta    = d_panel_scalars + 1;
        T* d_denom_bcast = d_comm_pack + 1;
        chase::linalg::internal::cuda::run_householder_scalar_kernel<T, RealT>(
            stream, pivot_here ? 1 : 0, d_x0_col, d_real_scalar,
            d_comm_pack + 0, d_inv_denom, d_neg_beta, d_denom_bcast,
            d_panel_scalars + 3);

        // Phase 2: fused [tau, denom] collective — one AllReduce of 2 elements.
#if STRICT_TAU_BCAST
        {
            // Determine owner via vote (block layout: one rank owns pivot).
            int col_rank_local = 0, col_size_local = 1;
            CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &col_rank_local));
            CHECK_NCCL_ERROR(ncclCommCount(nccl_col_comm, &col_size_local));
            const int h_vote = pivot_here ? (col_rank_local + 1) : 0;
            int* d_owner_vote = nullptr;
            int* d_owner_sel  = nullptr;
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_owner_vote, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_owner_sel,  sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyAsync(d_owner_vote, &h_vote, sizeof(int),
                                             cudaMemcpyHostToDevice, stream));
            CHECK_NCCL_ERROR(ncclAllReduce(d_owner_vote, d_owner_sel, 1,
                                           ncclInt32, ncclMax, nccl_col_comm, stream));
            int h_owner_plus = 0;
            CHECK_CUDA_ERROR(cudaMemcpyAsync(&h_owner_plus, d_owner_sel, sizeof(int),
                                             cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
            const int owner_rank = (h_owner_plus > 0) ? (h_owner_plus - 1) : 0;
            CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper<T>(
                d_comm_pack, 2, owner_rank, nccl_col_comm, &stream));
            cudaFree(d_owner_vote);
            cudaFree(d_owner_sel);
        }
#else
        panel_trace("col=" + std::to_string(col) + " SEQ-TAU enter");
        // Optional high-precision panel reduction for [tau, denom].
        if (use_panel_hiprec)
        {
            nccl_split_sync_fix_allreduce<T>(
                stream, d_comm_pack, d_comm_pack, 2, nccl_col_comm,
                d_split_hi, d_split_lo, cublas_handle, rank, block_id, "panel_tau_denom");
        }
        else
        {
            CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper<T>(
                d_comm_pack, d_comm_pack, 2, ncclSum, nccl_col_comm, &stream));
        }
        panel_trace("col=" + std::to_string(col) + " SEQ-TAU return");
#endif
        chase::linalg::internal::cuda::run_copy_scalar_kernel<T>(
            stream, d_comm_pack + 0, d_tau + col);

        // Phase 3: fused finish — scale v, set pivot=1, optionally peel Rkk.
        // After this, V[:,col] is physically clean: zeros for global rows < col,
        // 1 at pivot, reflector tail below pivot.
        chase::linalg::internal::cuda::run_fused_householder_finish_kernel<T>(
            stream, pivot_here ? 1 : 0, V + col * ldv, static_cast<int>(ldv),
            d_row_global, static_cast<int>(l_rows),
            static_cast<std::uint64_t>(col),
            static_cast<int>(pivot_loc), static_cast<int>(active_row_start),
            d_denom_bcast, d_inv_denom, d_panel_scalars + 3,
            d_r_diag ? (d_r_diag + col) : nullptr);

        // Phase 4: intra-sub-block rank-1 trailing update (active rows only).
        const std::size_t sub_begin   = (idx / sub_nb) * sub_nb;
        const std::size_t sub_end     = std::min(sub_begin + sub_nb, span);
        const std::size_t local_trail = sub_end - (idx + 1);
        if (local_trail > 0)
        {
            if (vr > 0)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    local_trail, 1, vr, d_one,
                    V + active_row_start + (col + 1) * ldv, ldv,
                    v_col_active, ldv, d_zero, d_w, local_trail));
            }
            else
            {
                CHECK_CUDA_ERROR(cudaMemsetAsync(
                    d_w, 0, local_trail * sizeof(T), stream));
            }
            panel_trace("col=" + std::to_string(col) + " SEQ-R1 enter"
                        + " local_trail=" + std::to_string(local_trail));
            // Optional high-precision panel reduction for rank-1 trailing coefficients.
            if (use_panel_hiprec)
            {
                nccl_split_sync_fix_allreduce<T>(
                    stream, d_w, d_w, local_trail, nccl_col_comm,
                    d_split_hi, d_split_lo, cublas_handle, rank, block_id, "panel_rank1");
            }
            else
            {
                chase::nccl::ncclAllReduceWrapper<T>(
                    d_w, d_w, local_trail, ncclSum, nccl_col_comm, &stream);
            }
            panel_trace("col=" + std::to_string(col) + " SEQ-R1 return");
            if (vr > 0)
            {
                chase::linalg::internal::cuda::run_copy_scalar_kernel<T>(
                    stream, d_tau + col, d_T_scalar);
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(
                    cublas_handle, 1, d_minus_one, d_T_scalar, 1));
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, cublas_op_c,
                    vr, local_trail, 1, d_T_scalar,
                    v_col_active, ldv, d_w, local_trail, d_one,
                    V + active_row_start + (col + 1) * ldv, ldv));
            }
        }

        // Phase 5: sub-block WY trailing update at sub-block boundary.
        // V is physically clean after fused_finish, so full l_rows GEMM is correct.
        if ((idx + 1) == sub_end)
        {
            const std::size_t swidth        = sub_end - sub_begin;
            const std::size_t sub_first_col = k + (jj_begin + sub_begin);
            const std::size_t after_sub_col = sub_first_col + swidth;
            const std::size_t remain_cols   = (k + jb_panel_total) - after_sub_col;
            if (remain_cols > 0)
            {
                // S = V[:,sub]^H * V[:,sub]  (full l_rows — physically clean)
                if (l_rows > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, cublas_op_c, CUBLAS_OP_N,
                        swidth, swidth, l_rows, d_one,
                        V + sub_first_col * ldv, ldv,
                        V + sub_first_col * ldv, ldv,
                        d_zero, d_sub_s, swidth));
                }
                else
                {
                    CHECK_CUDA_ERROR(cudaMemsetAsync(
                        d_sub_s, 0, swidth * swidth * sizeof(T), stream));
                }
                panel_trace("col=" + std::to_string(col) + " SEQ-S enter"
                            + " swidth=" + std::to_string(swidth));
                // Optional high-precision reduction for sub-block Gram matrix S.
                if (use_panel_hiprec)
                {
                    nccl_split_sync_fix_allreduce<T>(
                        stream, d_sub_s, d_sub_s, swidth * swidth, nccl_col_comm,
                        d_split_hi, d_split_lo, cublas_handle, rank, block_id, "panel_subS");
                }
                else
                {
                    chase::nccl::ncclAllReduceWrapper<T>(
                        d_sub_s, d_sub_s, swidth * swidth, ncclSum, nccl_col_comm, &stream);
                }
                panel_trace("col=" + std::to_string(col) + " SEQ-S return");
                chase::linalg::internal::cuda::run_compute_T_block<T>(
                    stream, d_sub_t, d_sub_s, d_tau + sub_first_col,
                    static_cast<int>(swidth), static_cast<int>(swidth));

                // W = V[:,sub]^H * V[:,remain]  (full l_rows)
                if (l_rows > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, cublas_op_c, CUBLAS_OP_N,
                        swidth, remain_cols, l_rows, d_one,
                        V + sub_first_col * ldv, ldv,
                        V + after_sub_col * ldv, ldv,
                        d_zero, d_sub_w, swidth));
                }
                else
                {
                    CHECK_CUDA_ERROR(cudaMemsetAsync(
                        d_sub_w, 0, swidth * remain_cols * sizeof(T), stream));
                }
                panel_trace("col=" + std::to_string(col) + " SEQ-W enter"
                            + " remain_cols=" + std::to_string(remain_cols));
                // Optional high-precision reduction for sub-block W.
                if (use_panel_hiprec)
                {
                    nccl_split_sync_fix_allreduce<T>(
                        stream, d_sub_w, d_sub_w, swidth * remain_cols, nccl_col_comm,
                        d_split_hi, d_split_lo, cublas_handle, rank, block_id, "panel_subW");
                }
                else
                {
                    chase::nccl::ncclAllReduceWrapper<T>(
                        d_sub_w, d_sub_w, swidth * remain_cols, ncclSum,
                        nccl_col_comm, &stream);
                }
                panel_trace("col=" + std::to_string(col) + " SEQ-W return");

                // TW = T^C * W
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                    swidth, remain_cols, swidth, d_one,
                    d_sub_t, swidth, d_sub_w, swidth,
                    d_zero, d_sub_tw, swidth));
                // V[:,remain] -= V[:,sub] * TW  (full l_rows)
                if (l_rows > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        l_rows, remain_cols, swidth, d_minus_one,
                        V + sub_first_col * ldv, ldv,
                        d_sub_tw, swidth, d_one,
                        V + after_sub_col * ldv, ldv));
                }
            }
        }
        panel_trace("col=" + std::to_string(col) + " idx=" +
                    std::to_string(idx) + " SEQ-END");
    }
}

//==============================================================================
// houseQR_panel_factor_1d_impl
//
// Block (contiguous 1D) analog of
// distributed_houseQR_panel_factor_block_cyclic_1d.
// Runs run_panel_pre_clean (zeros subdiagonal of panel columns using d_row_global),
// then delegates to houseQR_panel_factor_1d_columns_impl.
//==============================================================================
template <typename T>
static void houseQR_panel_factor_1d_impl(
    std::size_t m_global, std::size_t g_off, std::size_t l_rows,
    std::size_t ldv, T* V, std::size_t k, std::size_t jb,
    T* d_tau, cublasHandle_t cublas_handle,
    chase::Base<T>* d_real_scalar, T* d_T_scalar,
    T* d_one, T* d_zero, T* d_minus_one, T* d_panel_scalars, T* d_w,
    ncclComm_t nccl_col_comm,
    const std::uint64_t* d_row_global, T* d_r_diag,
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming* panel_timing,
    double* d_split_hi, double* d_split_lo, int rank, std::size_t block_id,
    const cuda_nccl::HouseQRTuning* tuning)
{
    if (l_rows > 0 && d_row_global == nullptr)
        throw std::runtime_error(
            "houseQR_panel_factor_1d_impl: d_row_global is null");

    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));
    const bool qr_hang_trace = []() {
        const char* e = std::getenv("CHASE_QR_HANG_TRACE");
        if (e == nullptr) return false;
        const std::string v(e);
        return (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON");
    }();
    int nccl_rank = -1;
    if (qr_hang_trace)
        CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &nccl_rank));
    auto panel_trace = [&](const std::string& msg) {
        if (!qr_hang_trace) return;
        std::cout << "[QR-HANG-TRACE][panel-1d][rank=" << nccl_rank << "] "
                  << "k=" << k << " jb=" << jb << " " << msg << std::endl;
    };
    panel_trace("enter");

    const bool do_panel_wall =
        (panel_timing != nullptr) &&
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);
    if (panel_timing != nullptr)
    {
        panel_timing->norm_ms = panel_timing->scalar_kernel_ms = 0.f;
        panel_timing->allreduce_tau_ms = panel_timing->scal_ms =
            panel_timing->trail_ms = 0.f;
        panel_timing->panel_total_ms = 0.f;
    }

    cudaEvent_t ev_start = nullptr, ev_end = nullptr;
    if (do_panel_wall)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_start, stream));
    }

    // Zero rows with global index < k for panel columns [k, k+jb).
    if (k < m_global && jb > 0 && l_rows > 0)
    {
        const std::size_t jb_cols = std::min(jb, m_global - k);
        panel_trace("pre_clean enter jb_cols=" + std::to_string(jb_cols));
        chase::linalg::internal::cuda::run_panel_pre_clean<T>(
            stream, V + k * ldv, ldv, d_row_global,
            static_cast<int>(l_rows), k, static_cast<int>(jb_cols));
        panel_trace("pre_clean launch return");
    }

    if (jb > 0)
    {
        const std::size_t sub_nb = nccl_qr_sub_nb_env(tuning, 8);
        const std::size_t sub_workspace_elems =
            2 + 2 * sub_nb * sub_nb + 2 * sub_nb * jb;
        T* d_sub_workspace = nullptr;
        panel_trace("cudaMalloc sub_workspace enter elems=" +
                    std::to_string(sub_workspace_elems));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sub_workspace,
                                    sub_workspace_elems * sizeof(T)));
        panel_trace("cudaMalloc sub_workspace return");
        panel_trace("columns_impl enter");
        houseQR_panel_factor_1d_columns_impl<T>(
            m_global, g_off, l_rows, ldv, V, k, 0, jb, jb,
            d_tau, cublas_handle,
            d_real_scalar, d_T_scalar,
            d_one, d_zero, d_minus_one, d_panel_scalars, d_w,
            nccl_col_comm, d_row_global, d_r_diag,
            d_sub_workspace, sub_workspace_elems,
            d_split_hi, d_split_lo, rank, block_id, tuning);
        panel_trace("columns_impl return");
        panel_trace("cudaFree sub_workspace enter");
        CHECK_CUDA_ERROR(cudaFree(d_sub_workspace));
        panel_trace("cudaFree sub_workspace return");
    }

    if (do_panel_wall)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_end, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&panel_timing->panel_total_ms,
                                              ev_start, ev_end));
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
    }
    panel_trace("return");
}

//==============================================================================
// distributed_blocked_houseQR_formQ (NCCL — V on device, NCCL collectives only)
//
// Block (contiguous 1D) path — fully optimized to match block-cyclic path:
//   - Physical-clean panel (houseQR_panel_factor_1d_impl)
//   - Full l_rows GEMMs in T-build and trailing update (no save/restore)
//   - Chunked backward formQ with ping-pong W/TW buffers
//   - Granular phase timing matching block-cyclic output
//==============================================================================
template <typename T>
void cuda_nccl::distributed_blocked_houseQR_formQ(std::size_t m_global,
                                                     std::size_t n,
                                                     std::size_t l_rows,
                                                     std::size_t g_off,
                                                     std::size_t ldv,
                                                     T*          V,
                                                     MPI_Comm    mpi_comm,
                                                     cublasHandle_t cublas_handle,
                                                     T* d_workspace,
                                                     std::size_t lwork_elems,
                                                     ncclComm_t nccl_col_comm,
                                                     const HouseQRTuning* tuning)
{
    using RealT = chase::Base<T>;
    const std::size_t nb = nccl_qr_block_nb(tuning);

    if (n == 0)
        return;

    if (nb == 0 || nb >= n)
    {
        cuda_nccl::distributed_houseQR_formQ<T>(
            m_global, n, l_rows, g_off, ldv, V, mpi_comm,
            cublas_handle, d_workspace, lwork_elems, nccl_col_comm, tuning);
        return;
    }

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    cudaStream_t stream_compute = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_compute));

    int leastPriority = 0, greatestPriority = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    cudaStream_t stream_panel = nullptr;
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream_panel, cudaStreamNonBlocking,
                                                   greatestPriority));
    cublasHandle_t cublas_panel = nullptr;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_panel));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_panel, stream_panel));
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_panel, CUBLAS_POINTER_MODE_DEVICE));

    const std::size_t num_blocks = (n + nb - 1) / nb;
    std::vector<cudaEvent_t> event_cols_ready(num_blocks, nullptr);
    std::vector<cudaEvent_t> event_panel_done(num_blocks, nullptr);
    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event_cols_ready[b], cudaEventDisableTiming));
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event_panel_done[b], cudaEventDisableTiming));
    }

    int rank = 0;
    MPI_Comm_rank(mpi_comm, &rank);

    const bool enable_timing =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);
    // When false (the default), time_scope never blocks the host on per-phase
    // CUDA events so the look-ahead overlap is preserved in production.
    // Set CHASE_QR_TIMING_BLOCKING=1 to get accurate per-phase GPU times at
    // the cost of serialising the look-ahead (same semantics as block-cyclic).
    const bool timing_blocking = nccl_qr_timing_blocking(tuning);
    // Hang-trace switch: CHASE_QR_HANG_TRACE=1 enables fine-grained checkpoint logs.
    const bool qr_hang_trace = []() {
        const char* e = std::getenv("CHASE_QR_HANG_TRACE");
        if (e == nullptr) return false;
        const std::string v(e);
        return (v == "1" || v == "true" || v == "TRUE" || v == "on" || v == "ON");
    }();

    const T one = T(1), zero = T(0), minus_one = T(-1);
    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CUBLAS_OP_C : CUBLAS_OP_T;

    // Chunked backward application (CHASE_FORMQ_CHUNKS env var).
    const int formq_chunks = nccl_qr_formq_chunks(tuning, 1);
    auto qr_trace = [&](const std::string& msg) {
        if (!qr_hang_trace || rank != 0) return;
        std::ostringstream oss;
        oss << "[QR-HANG-TRACE][blocked-1d][rank=" << rank << "] "
            << msg << '\n';
        chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), rank);
    };

    // -----------------------------------------------------------------------
    // Allocate device buffers (always cudaMalloc — no workspace sharing).
    // -----------------------------------------------------------------------
    const std::size_t need_Tall = num_blocks * nb * nb;
    const std::size_t need_W    = nb * n;
    const std::size_t split_need =
        need_W * chase::nccl::DataCountMultiplier<T>::value;

    T* d_one = nullptr, *d_zero = nullptr, *d_minus_one = nullptr;
    T* d_panel_scalars[2] = {nullptr, nullptr};
    T* d_w_panel[2]       = {nullptr, nullptr};
    RealT* d_real_scalar[2] = {nullptr, nullptr};
    T*     d_T_scalar[2]    = {nullptr, nullptr};
    T* d_tau       = nullptr;
    T* d_VH        = nullptr;
    T* d_T_blocks  = nullptr;
    T* d_r_diag    = nullptr;
    T* d_W_buf[2]  = {nullptr, nullptr};
    T* d_TW_buf[2] = {nullptr, nullptr};
    double* d_split_hi = nullptr;
    double* d_split_lo = nullptr;
    std::uint64_t* d_row_global = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_one,       sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_zero,      sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_minus_one, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars[0], 4 * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars[1], 4 * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w_panel[0],   nb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w_panel[1],   nb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar[0], sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar[1], sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar[0],    sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar[1],    sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tau,      n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_VH,       l_rows * n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_blocks, need_Tall * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_r_diag,   n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_W_buf[0],  need_W * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_W_buf[1],  need_W * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_TW_buf[0], need_W * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_TW_buf[1], need_W * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_split_hi,  split_need * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_split_lo,  split_need * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_one,       &one,       sizeof(T), cudaMemcpyHostToDevice, stream_compute));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_zero,      &zero,      sizeof(T), cudaMemcpyHostToDevice, stream_compute));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_minus_one, &minus_one, sizeof(T), cudaMemcpyHostToDevice, stream_compute));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_T_blocks, 0, need_Tall * sizeof(T), stream_compute));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_r_diag,   0, n * sizeof(T),         stream_compute));

    // Build d_row_global: contiguous block — row i maps to global row g_off+i.
    if (l_rows > 0)
    {
        std::vector<std::uint64_t> h_rg(l_rows);
        for (std::size_t i = 0; i < l_rows; ++i)
            h_rg[i] = static_cast<std::uint64_t>(g_off + i);
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_row_global, l_rows * sizeof(std::uint64_t)));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_row_global, h_rg.data(),
            l_rows * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream_compute));
    }

    // Timing accumulators (matching block-cyclic granularity).
    float t_panel_ms = 0.f, t_tbuild_ms = 0.f, t_trail_ms = 0.f;
    float t_initq_ms = 0.f;
    float t_vhq_gemm_ms = 0.f, t_vhq_nccl_ms = 0.f;
    float t_apply_t_ms  = 0.f, t_rankk_ms    = 0.f;
    cudaEvent_t ev_a = nullptr, ev_b_t = nullptr;
    cudaEvent_t ev_total_start = nullptr, ev_total_end = nullptr;
    if (enable_timing)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_a));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_b_t));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_total_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_total_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_total_start, stream_compute));
    }

    // Lightweight scope timer — matches block-cyclic semantics exactly:
    //   !enable_timing or !timing_blocking → call fn() without any host stall
    //                                         (look-ahead overlap preserved).
    //   timing_blocking=true               → record CUDA events on stream_compute,
    //                                         sync after fn(), accumulate elapsed ms.
    auto time_scope = [&](float& acc_ms, auto&& fn)
    {
        if (!enable_timing)
        {
            fn();
#if STRICT_TIMING
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_compute));
#endif
            return;
        }
        if (!timing_blocking)
        {
            fn();
#if STRICT_TIMING
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_compute));
#endif
            return;
        }
        CHECK_CUDA_ERROR(cudaEventRecord(ev_a, stream_compute));
        fn();
#if STRICT_TIMING
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_compute));
#endif
        CHECK_CUDA_ERROR(cudaEventRecord(ev_b_t, stream_compute));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_b_t));
        float ms = 0.f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, ev_a, ev_b_t));
        acc_ms += ms;
    };

    // -----------------------------------------------------------------------
    // Prime pipeline: launch block-0 panel factor on stream_panel.
    // -----------------------------------------------------------------------
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming panel_blk{};
    float sum_panel_wall_ms = 0.f;

    CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[0], stream_compute));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[0], 0));
    qr_trace("prime: launch panel b=0 (enter)");
    {
        const std::size_t jb0 = std::min(nb, n);
        houseQR_panel_factor_1d_impl<T>(
            m_global, g_off, l_rows, ldv, V, 0, jb0, d_tau, cublas_panel,
            d_real_scalar[0], d_T_scalar[0], d_one, d_zero, d_minus_one,
            d_panel_scalars[0], d_w_panel[0], nccl_col_comm,
            d_row_global, d_r_diag,
            enable_timing ? &panel_blk : nullptr,
            d_split_hi, d_split_lo, rank, 0, tuning);
        qr_trace("prime: launch panel b=0 (return)");
        CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[0], stream_panel));
        qr_trace("prime: panel_done[0] recorded");
        if (enable_timing) sum_panel_wall_ms += panel_blk.panel_total_ms;
    }

    // -----------------------------------------------------------------------
    // Main forward look-ahead loop.
    // -----------------------------------------------------------------------
    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        const std::size_t k    = b * nb;
        const std::size_t jb   = std::min(nb, n - k);
        T* Tb                  = d_T_blocks + b * nb * nb;
        const int         next = static_cast<int>(1 - b % 2);
        T* d_W  = d_W_buf[0];
        T* d_TW = d_TW_buf[0];

        qr_trace("b=" + std::to_string(b) + " wait panel_done");
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_compute, event_panel_done[b], 0));
        qr_trace("b=" + std::to_string(b) + " panel_done reached");
        if (enable_timing) t_panel_ms += panel_blk.panel_total_ms;

        // T-build: V is physically clean after new panel — full l_rows GEMM, no save/restore.
        time_scope(t_tbuild_ms, [&]()
        {
            qr_trace("b=" + std::to_string(b) + " T-build begin");
            if (l_rows > 0)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    jb, jb, l_rows, d_one,
                    V + k * ldv, ldv,
                    V + k * ldv, ldv,
                    d_zero, d_TW, jb));
            }
            else
            {
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_TW, 0, jb * jb * sizeof(T), stream_compute));
            }
#if CHASE_PANEL_HIPREC
            nccl_split_sync_fix_allreduce<T>(
                stream_compute, d_TW, d_TW, jb * jb, nccl_col_comm,
                d_split_hi, d_split_lo, cublas_handle, rank, b, "tbuild");
#else
            chase::nccl::ncclAllReduceWrapper<T>(d_TW, d_TW, jb * jb, ncclSum,
                                                 nccl_col_comm, &stream_compute);
#endif
            qr_trace("b=" + std::to_string(b) + " T-build allreduce done");
            nccl_strict_stream_barrier(stream_compute);
            chase::linalg::internal::cuda::run_compute_T_block<T>(
                stream_compute, Tb, d_TW, d_tau + k,
                static_cast<int>(jb), static_cast<int>(nb));
            qr_trace("b=" + std::to_string(b) + " T-build done");
#if CHASE_PANEL_HIPREC
            nccl_trace_tmatrix_condition<T>(Tb, static_cast<int>(jb),
                                            static_cast<int>(nb), rank, b);
#endif
        });

        // Trailing update: full l_rows GEMMs (no save/restore).
        const std::size_t n_trail = n - k - jb;
        if (n_trail > 0)
        {
            time_scope(t_trail_ms, [&]()
            {
                qr_trace("b=" + std::to_string(b) + " trail begin");
                const std::size_t jb_next = std::min(nb, n_trail);
                const std::size_t n_rest  = n_trail - jb_next;

                // Phase A: update next-block columns (enables look-ahead).
                if (l_rows > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, cublas_op_c, CUBLAS_OP_N,
                        jb, jb_next, l_rows, d_one,
                        V + k * ldv, ldv,
                        V + (k + jb) * ldv, ldv,
                        d_zero, d_W, jb));
                }
                else
                {
                    CHECK_CUDA_ERROR(cudaMemsetAsync(d_W, 0, jb * jb_next * sizeof(T), stream_compute));
                }
#if CHASE_PANEL_HIPREC
                nccl_split_sync_fix_allreduce<T>(
                    stream_compute, d_W, d_W, jb * jb_next, nccl_col_comm,
                    d_split_hi, d_split_lo, cublas_handle, rank, b, "trail_next");
#else
                chase::nccl::ncclAllReduceWrapper<T>(d_W, d_W, jb * jb_next, ncclSum,
                                                     nccl_col_comm, &stream_compute);
#endif
                qr_trace("b=" + std::to_string(b) + " trail-next allreduce done");
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                    jb, jb_next, jb, d_one, Tb, nb, d_W, jb, d_zero, d_TW, jb));
                if (l_rows > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        l_rows, jb_next, jb, d_minus_one,
                        V + k * ldv, ldv, d_TW, jb, d_one,
                        V + (k + jb) * ldv, ldv));
                }

                // Signal look-ahead: next panel can start.
                if (b + 1 < num_blocks)
                {
                    CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[b + 1], stream_compute));
                    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[b + 1], 0));
                    const std::size_t k_next  = k + jb;
                    const std::size_t jb_fact = std::min(nb, n - k_next);
                    qr_trace("b=" + std::to_string(b) + " launch next panel b=" +
                             std::to_string(b + 1) + " (enter)");
                    qr_trace("b=" + std::to_string(b) + " panel args: k_next=" +
                             std::to_string(k_next) + " jb_fact=" +
                             std::to_string(jb_fact));
                    houseQR_panel_factor_1d_impl<T>(
                        m_global, g_off, l_rows, ldv, V, k_next, jb_fact,
                        d_tau, cublas_panel,
                        d_real_scalar[next], d_T_scalar[next], d_one, d_zero, d_minus_one,
                        d_panel_scalars[next], d_w_panel[next], nccl_col_comm,
                        d_row_global, d_r_diag,
                        enable_timing ? &panel_blk : nullptr,
                        d_split_hi, d_split_lo, rank, b + 1, tuning);
                    qr_trace("b=" + std::to_string(b) + " launch next panel b=" +
                             std::to_string(b + 1) + " (return)");
                    CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[b + 1], stream_panel));
                    qr_trace("b=" + std::to_string(b) + " panel_done[" +
                             std::to_string(b + 1) + "] recorded");
                    if (enable_timing) sum_panel_wall_ms += panel_blk.panel_total_ms;
                }

                // Phase B: rest of trailing (concurrent with next panel).
                if (n_rest > 0)
                {
                    if (l_rows > 0)
                    {
                        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                            cublas_handle, cublas_op_c, CUBLAS_OP_N,
                            jb, n_rest, l_rows, d_one,
                            V + k * ldv, ldv,
                            V + (k + jb + jb_next) * ldv, ldv,
                            d_zero, d_W, jb));
                    }
                    else
                    {
                        CHECK_CUDA_ERROR(cudaMemsetAsync(d_W, 0, jb * n_rest * sizeof(T), stream_compute));
                    }
#if CHASE_PANEL_HIPREC
                    nccl_split_sync_fix_allreduce<T>(
                        stream_compute, d_W, d_W, jb * n_rest, nccl_col_comm,
                        d_split_hi, d_split_lo, cublas_handle, rank, b, "trail_rest");
#else
                    chase::nccl::ncclAllReduceWrapper<T>(d_W, d_W, jb * n_rest, ncclSum,
                                                         nccl_col_comm, &stream_compute);
#endif
                    qr_trace("b=" + std::to_string(b) + " trail-rest allreduce done");
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                        jb, n_rest, jb, d_one, Tb, nb, d_W, jb, d_zero, d_TW, jb));
                    if (l_rows > 0)
                    {
                        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            l_rows, n_rest, jb, d_minus_one,
                            V + k * ldv, ldv, d_TW, jb, d_one,
                            V + (k + jb + jb_next) * ldv, ldv));
                    }
                }
                qr_trace("b=" + std::to_string(b) + " trail done");
            });
        }
        else if (b + 1 < num_blocks)
        {
            // No trailing — but still need to launch next panel.
            const std::size_t k_next  = k + jb;
            const std::size_t jb_fact = std::min(nb, n - k_next);
            CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[b + 1], stream_compute));
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[b + 1], 0));
            qr_trace("b=" + std::to_string(b) + " no-trail next panel b=" +
                     std::to_string(b + 1) + " (enter)"
                     + " k_next=" + std::to_string(k_next)
                     + " jb_fact=" + std::to_string(jb_fact));
            houseQR_panel_factor_1d_impl<T>(
                m_global, g_off, l_rows, ldv, V, k_next, jb_fact,
                d_tau, cublas_panel,
                d_real_scalar[next], d_T_scalar[next], d_one, d_zero, d_minus_one,
                d_panel_scalars[next], d_w_panel[next], nccl_col_comm,
                d_row_global, d_r_diag,
                enable_timing ? &panel_blk : nullptr,
                d_split_hi, d_split_lo, rank, b + 1, tuning);
            qr_trace("b=" + std::to_string(b) + " no-trail next panel b=" +
                     std::to_string(b + 1) + " (return)");
            CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[b + 1], stream_panel));
            qr_trace("b=" + std::to_string(b) + " panel_done[" +
                     std::to_string(b + 1) + "] recorded");
            if (enable_timing) sum_panel_wall_ms += panel_blk.panel_total_ms;
        }
    }

    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_compute, event_panel_done[num_blocks - 1], 0));
    qr_trace("forward phase complete");

    // -----------------------------------------------------------------------
    // Initialise Q: copy V -> VH, zero V, set diagonal pivots.
    // For contiguous block layout: column c has its pivot at local row c-g_off
    // iff g_off <= c < g_off+l_rows.
    // -----------------------------------------------------------------------
    time_scope(t_initq_ms, [&]()
    {
        chase::linalg::internal::cuda::t_lacpy('A', l_rows, n, V, ldv, d_VH, l_rows,
                                               &stream_compute);
        CHECK_CUDA_ERROR(cudaMemsetAsync(V, 0, ldv * n * sizeof(T), stream_compute));
        for (std::size_t c = 0; c < n; ++c)
        {
            if (g_off <= c && c < g_off + l_rows)
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    V + (c - g_off) + c * ldv, d_one, sizeof(T),
                    cudaMemcpyDeviceToDevice, stream_compute));
        }
    });

    // -----------------------------------------------------------------------
    // Backward blocked application: chunked + ping-pong W/TW buffers.
    // All GEMMs use full l_rows (VH is physically clean after the forward pass).
    // -----------------------------------------------------------------------
    for (std::size_t bb = 0; bb < num_blocks; ++bb)
    {
        const std::size_t b      = num_blocks - 1 - bb;
        const std::size_t k      = b * nb;
        const std::size_t jb     = std::min(nb, n - k);
        const std::size_t n_cols = n - k;
        T* Tb = d_T_blocks + b * nb * nb;

        const int bw     = static_cast<int>(bb & 1u);
        T* d_W_curr      = d_W_buf[bw];
        T* d_TW_curr     = d_TW_buf[bw];

        const std::size_t chunks =
            std::max<std::size_t>(1,
                std::min<std::size_t>(static_cast<std::size_t>(formq_chunks), n_cols));

        qr_trace("backward block b=" + std::to_string(b) + " begin");
        for (std::size_t ch = 0; ch < chunks; ++ch)
        {
            const std::size_t c0 = (ch * n_cols) / chunks;
            const std::size_t c1 = ((ch + 1) * n_cols) / chunks;
            const std::size_t cw = c1 - c0;
            if (cw == 0) continue;

            // W = VH[:,k..k+jb-1]^H * Q[:,k+c0..k+c1-1]  (full l_rows)
            time_scope(t_vhq_gemm_ms, [&]()
            {
                if (l_rows > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, cublas_op_c, CUBLAS_OP_N,
                        jb, cw, l_rows, d_one,
                        d_VH + k * l_rows, l_rows,
                        V + (k + c0) * ldv, ldv,
                        d_zero, d_W_curr, jb));
                }
                else
                {
                    CHECK_CUDA_ERROR(cudaMemsetAsync(d_W_curr, 0, jb * cw * sizeof(T), stream_compute));
                }
            });

            time_scope(t_vhq_nccl_ms, [&]()
            {
#if CHASE_PANEL_HIPREC
                nccl_split_sync_fix_allreduce<T>(
                    stream_compute, d_W_curr, d_W_curr, jb * cw, nccl_col_comm,
                    d_split_hi, d_split_lo, cublas_handle, rank, b, "formQ_chunk");
#else
                chase::nccl::ncclAllReduceWrapper<T>(
                    d_W_curr, d_W_curr, jb * cw, ncclSum, nccl_col_comm, &stream_compute);
#endif
            });
            if (ch == 0)
                qr_trace("backward block b=" + std::to_string(b) + " first chunk allreduce done");

            // TW = T * W
            time_scope(t_apply_t_ms, [&]()
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    jb, cw, jb, d_one,
                    Tb, nb, d_W_curr, jb, d_zero, d_TW_curr, jb));
            });

            // Q[:,k+c0..] -= VH[:,k..] * TW  (full l_rows)
            time_scope(t_rankk_ms, [&]()
            {
                if (l_rows > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        l_rows, cw, jb, d_minus_one,
                        d_VH + k * l_rows, l_rows,
                        d_TW_curr, jb, d_one,
                        V + (k + c0) * ldv, ldv));
                }
            });
        }
        qr_trace("backward block b=" + std::to_string(b) + " done");
    }

    const float t_formq_ms =
        t_vhq_gemm_ms + t_vhq_nccl_ms + t_apply_t_ms + t_rankk_ms;

    if (enable_timing)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_total_end, stream_compute));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_total_end));
        float t_total_ms = 0.f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_total_ms, ev_total_start, ev_total_end));
        if (rank == 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "[Householder QR blocked 1d] Total: " << (t_total_ms / 1000.0) << " s"
                << "  (n=" << n << " nb=" << nb << " l_rows=" << l_rows << ")\n"
                << "  Breakdown: panel=" << (t_panel_ms / 1000.0)
                << " s  T_build=" << (t_tbuild_ms / 1000.0)
                << " s  trailing=" << (t_trail_ms / 1000.0)
                << " s  initQ=" << (t_initq_ms / 1000.0)
                << " s  formQ=" << (t_formq_ms / 1000.0) << " s\n"
                << "  Backward breakdown:"
                << "  vhq_gemm=" << (t_vhq_gemm_ms / 1000.0)
                << " s  vhq_nccl=" << (t_vhq_nccl_ms / 1000.0)
                << " s  apply_T=" << (t_apply_t_ms / 1000.0)
                << " s  rank_k=" << (t_rankk_ms / 1000.0)
                << " s  chunks=" << formq_chunks << "\n"
                << "  Panel CUDA wall (sum): " << sum_panel_wall_ms << " ms\n";
            chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), rank);
        }
        cudaEventDestroy(ev_a);
        cudaEventDestroy(ev_b_t);
        cudaEventDestroy(ev_total_start);
        cudaEventDestroy(ev_total_end);
    }

    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        cudaEventDestroy(event_cols_ready[b]);
        cudaEventDestroy(event_panel_done[b]);
    }
    cublasDestroy(cublas_panel);
    cudaStreamDestroy(stream_panel);

    cudaFree(d_one);      cudaFree(d_zero);     cudaFree(d_minus_one);
    cudaFree(d_panel_scalars[0]); cudaFree(d_panel_scalars[1]);
    cudaFree(d_w_panel[0]);       cudaFree(d_w_panel[1]);
    cudaFree(d_real_scalar[0]);   cudaFree(d_real_scalar[1]);
    cudaFree(d_T_scalar[0]);      cudaFree(d_T_scalar[1]);
    cudaFree(d_tau);    cudaFree(d_VH);     cudaFree(d_T_blocks); cudaFree(d_r_diag);
    cudaFree(d_W_buf[0]); cudaFree(d_W_buf[1]);
    cudaFree(d_TW_buf[0]); cudaFree(d_TW_buf[1]);
    cudaFree(d_split_hi); cudaFree(d_split_lo);
    if (d_row_global) cudaFree(d_row_global);

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
    qr_trace("exit distributed_blocked_houseQR_formQ");
}

//==============================================================================
// distributed_houseQR_formQ_block_cyclic_1d (NCCL)
//
// Block-cyclic 1D path: physical cleaning via split-and-pad (no panel
// save/restore), enabling direct full-height GEMM updates.
//==============================================================================
template <typename T>
void cuda_nccl::distributed_houseQR_formQ_block_cyclic_1d(
    std::size_t m_global, std::size_t n,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t ldv, T* V, std::size_t nb_dist, MPI_Comm mpi_comm,
    cublasHandle_t cublas_handle, T* d_workspace, std::size_t lwork_elems,
    ncclComm_t nccl_col_comm,
    const HouseQRTuning* tuning)
{
    const std::size_t nseg = seg_global_offs.size();
    if (nseg != seg_local_offs.size() || nseg != seg_lens.size())
        throw std::runtime_error(
            "distributed_houseQR_formQ_block_cyclic_1d: segment vector sizes must match");
    const std::size_t l_rows = nseg == 0 ? 0 : seg_local_offs.back() + seg_lens.back();

    // Reuse existing blocked core with nb=0 fallback route by running panel block size n.
    // This keeps API parity while block-cyclic path below remains explicit.
    if (n == 0) return;

    const T one = T(1);
    const T zero = T(0);
    const T minus_one = T(-1);
    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value) ? CUBLAS_OP_C : CUBLAS_OP_T;
    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    int qr_rank = 0;
    MPI_Comm_rank(mpi_comm, &qr_rank);
    const bool qr_debug_timing =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming panel_prof;
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming* panel_prof_ptr =
        qr_debug_timing ? &panel_prof : nullptr;

    std::vector<std::size_t> row_seg(n, nseg), row_local(n, 0);
    nccl_fill_pivot_row_seg_index(row_seg, n, m_global, seg_global_offs, seg_lens, nseg);
    nccl_fill_pivot_row_local_index(row_local, row_seg, n, seg_global_offs, seg_local_offs, nseg);

    chase::Base<T>* d_real_scalar = nullptr;
    T *d_T_scalar = nullptr, *d_one = nullptr, *d_zero = nullptr, *d_minus_one = nullptr;
    T *d_panel_scalars = nullptr, *d_w = nullptr, *d_tau = nullptr, *d_VH = nullptr;
    double* d_split_hi = nullptr;
    double* d_split_lo = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar, sizeof(chase::Base<T>)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_one, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_zero, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_minus_one, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars, 4 * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tau, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_VH, l_rows * n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_split_hi,
        n * chase::nccl::DataCountMultiplier<T>::value * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_split_lo,
        n * chase::nccl::DataCountMultiplier<T>::value * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_one, &one, sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_zero, &zero, sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_minus_one, &minus_one, sizeof(T), cudaMemcpyHostToDevice, stream));

    std::vector<std::uint64_t> h_row_global(l_rows);
    for (std::size_t s = 0; s < nseg; ++s)
        for (std::size_t r = 0; r < seg_lens[s]; ++r)
            h_row_global[seg_local_offs[s] + r] =
                static_cast<std::uint64_t>(seg_global_offs[s] + r);
    std::uint64_t* d_row_global = nullptr;
    if (l_rows > 0)
    {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_row_global, l_rows * sizeof(std::uint64_t)));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_row_global, h_row_global.data(),
            l_rows * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream));
    }

    cuda_nccl::distributed_houseQR_panel_factor_block_cyclic_1d<T>(
        n, m_global, seg_global_offs, seg_local_offs, seg_lens, ldv, V, 0, n, nb_dist, d_tau,
        cublas_handle, d_real_scalar, d_T_scalar, d_one, d_zero, d_minus_one,
        d_panel_scalars, d_w, nccl_col_comm, l_rows, d_row_global, nullptr, tuning,
        panel_prof_ptr);

    chase::linalg::internal::cuda::t_lacpy('A', l_rows, n, V, ldv, d_VH, l_rows, &stream);
    CHECK_CUDA_ERROR(cudaMemsetAsync(V, 0, ldv * n * sizeof(T), stream));
    for (std::size_t c = 0; c < n; ++c)
    {
        if (row_seg[c] < nseg)
            CHECK_CUDA_ERROR(cudaMemcpyAsync(V + row_local[c] + c * ldv, d_one, sizeof(T),
                                             cudaMemcpyDeviceToDevice, stream));
    }

    cudaEvent_t ev_formq_start = nullptr, ev_formq_end = nullptr;
    float formq_backward_ms = 0.f;
    if (qr_debug_timing)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_formq_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_formq_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_formq_start, stream));
    }

    for (std::size_t jj = 0; jj < n; ++jj)
    {
        const std::size_t j = n - 1 - jj;
        const std::size_t n_cols = n - j;
        CHECK_CUDA_ERROR(cudaMemsetAsync(d_w, 0, n_cols * sizeof(T), stream));
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, cublas_op_c, CUBLAS_OP_N,
            n_cols, 1, l_rows, d_one,
            V + j * ldv, ldv,
            d_VH + j * l_rows, l_rows,
            d_zero, d_w, n_cols));
#if CHASE_PANEL_HIPREC
        nccl_split_sync_fix_allreduce<T>(
            stream, d_w, d_w, n_cols, nccl_col_comm, d_split_hi, d_split_lo,
            cublas_handle, qr_rank, j, "formQ_unblocked");
#else
        chase::nccl::ncclAllReduceWrapper<T>(d_w, d_w, n_cols, ncclSum,
                                             nccl_col_comm, &stream);
#endif
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_T_scalar, d_tau + j, sizeof(T), cudaMemcpyDeviceToDevice, stream));
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTscal(cublas_handle, 1, d_minus_one, d_T_scalar, 1));
        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, cublas_op_c,
            l_rows, n_cols, 1, d_T_scalar,
            d_VH + j * l_rows, l_rows, d_w, n_cols,
            d_one, V + j * ldv, ldv));
    }

    if (qr_debug_timing)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_formq_end, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_formq_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&formq_backward_ms, ev_formq_start, ev_formq_end));
        CHECK_CUDA_ERROR(cudaEventDestroy(ev_formq_start));
        CHECK_CUDA_ERROR(cudaEventDestroy(ev_formq_end));
        if (qr_rank == 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "[NCCL formQ block-cyclic 1d unblocked] panel_gpu_wall_ms="
                << panel_prof.panel_total_ms
                << "  formQ_backward_gpu_wall_ms=" << formq_backward_ms << '\n';
            chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), qr_rank);
        }
    }

    cudaFree(d_real_scalar); cudaFree(d_T_scalar); cudaFree(d_one); cudaFree(d_zero);
    cudaFree(d_minus_one); cudaFree(d_panel_scalars); cudaFree(d_w); cudaFree(d_tau); cudaFree(d_VH);
    cudaFree(d_split_hi); cudaFree(d_split_lo);
    if (d_row_global) cudaFree(d_row_global);
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
}

template <typename T>
void cuda_nccl::distributed_blocked_houseQR_formQ_block_cyclic_1d(
    std::size_t m_global, std::size_t n,
    const std::vector<std::size_t>& seg_global_offs,
    const std::vector<std::size_t>& seg_local_offs,
    const std::vector<std::size_t>& seg_lens,
    std::size_t ldv, T* V, MPI_Comm mpi_comm,
    std::size_t nb_dist,
    cublasHandle_t cublas_handle, T* d_workspace, std::size_t lwork_elems,
    ncclComm_t nccl_col_comm,
    const HouseQRTuning* tuning)
{
    using RealT = chase::Base<T>;
    const std::size_t nb = nccl_qr_block_nb(tuning);
    if (n == 0) return;
    if (nb == 0 || nb >= n)
    {
        cuda_nccl::distributed_houseQR_formQ_block_cyclic_1d<T>(
            m_global, n, seg_global_offs, seg_local_offs, seg_lens, ldv, V,
            nb_dist, mpi_comm, cublas_handle, d_workspace, lwork_elems, nccl_col_comm,
            tuning);
        return;
    }

    const std::size_t nseg = seg_global_offs.size();
    if (nseg != seg_local_offs.size() || nseg != seg_lens.size())
        throw std::runtime_error(
            "distributed_blocked_houseQR_formQ_block_cyclic_1d: segment vector sizes must match");
    const std::size_t l_rows = nseg == 0 ? 0 : (seg_local_offs.back() + seg_lens.back());
    if (l_rows != 0 && n > std::numeric_limits<std::size_t>::max() / l_rows)
        throw std::runtime_error(
            "distributed_blocked_houseQR_formQ_block_cyclic_1d: l_rows * n exceeds size_t");

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
    cudaStream_t stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream));
    int rank = 0;
    MPI_Comm_rank(mpi_comm, &rank);
    const bool enable_timing =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);
    const bool timing_blocking = nccl_qr_timing_blocking(tuning);
    float t_panel_ms = 0.f, t_tbuild_ms = 0.f, t_trail_ms = 0.f;
    float t_initq_ms = 0.f, t_formq_ms = 0.f, t_total_ms = 0.f;
    float t_vhq_gemm_ms = 0.f, t_vhq_nccl_ms = 0.f;
    float t_apply_t_ms = 0.f, t_rankk_update_ms = 0.f;
    cudaEvent_t ev_a = nullptr, ev_b = nullptr, ev_total_start = nullptr, ev_total_end = nullptr;
    if (enable_timing)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_a));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_b));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_total_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_total_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_total_start, stream));
    }
    auto time_scope = [&](float& acc_ms, auto&& fn) {
        if (!enable_timing)
        {
            fn();
#if STRICT_TIMING
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
#endif
            return;
        }
        if (!timing_blocking)
        {
            // Production mode: never block host on per-phase timing.
            fn();
#if STRICT_TIMING
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
#endif
            return;
        }
        CHECK_CUDA_ERROR(cudaEventRecord(ev_a, stream));
        fn();
#if STRICT_TIMING
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
#endif
        CHECK_CUDA_ERROR(cudaEventRecord(ev_b, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_b));
        float ms = 0.f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, ev_a, ev_b));
        acc_ms += ms;
    };

    const T one = T(1), zero = T(0), minus_one = T(-1);
    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value) ? CUBLAS_OP_C : CUBLAS_OP_T;

    std::vector<std::size_t> row_seg(n, nseg), row_local(n, 0);
    nccl_fill_pivot_row_seg_index(row_seg, n, m_global, seg_global_offs, seg_lens, nseg);
    nccl_fill_pivot_row_local_index(row_local, row_seg, n, seg_global_offs, seg_local_offs, nseg);

    const std::size_t num_blocks = (n + nb - 1) / nb;
    const int formq_chunks = nccl_qr_formq_chunks(tuning, 1);
    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming panel_blk;
    float sum_panel_cuda_wall_ms = 0.f;

    RealT* d_real_scalar[2] = {nullptr, nullptr};
    T *d_T_scalar[2] = {nullptr, nullptr}, *d_one = nullptr, *d_zero = nullptr, *d_minus_one = nullptr;
    T *d_panel_scalars[2] = {nullptr, nullptr}, *d_tau = nullptr, *d_VH = nullptr;
    T *d_w_panel[2] = {nullptr, nullptr};
    T *d_T_blocks = nullptr, *d_W = nullptr, *d_TW = nullptr, *d_r_diag = nullptr;
    T *d_W_buf[2] = {nullptr, nullptr}, *d_TW_buf[2] = {nullptr, nullptr};
    double* d_split_hi = nullptr;
    double* d_split_lo = nullptr;
    std::uint64_t* d_row_global = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar[0], sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar[1], sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar[0], sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar[1], sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_one, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_zero, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_minus_one, sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars[0], 4 * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars[1], 4 * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w_panel[0], nb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w_panel[1], nb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tau, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_VH, l_rows * n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_blocks, num_blocks * nb * nb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_W_buf[0], nb * n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_W_buf[1], nb * n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_TW_buf[0], nb * n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_TW_buf[1], nb * n * sizeof(T)));
    d_W = d_W_buf[0];
    d_TW = d_TW_buf[0];
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_r_diag, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_split_hi,
        nb * n * chase::nccl::DataCountMultiplier<T>::value * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_split_lo,
        nb * n * chase::nccl::DataCountMultiplier<T>::value * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_one, &one, sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_zero, &zero, sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_minus_one, &minus_one, sizeof(T), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_T_blocks, 0, num_blocks * nb * nb * sizeof(T), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_r_diag, 0, n * sizeof(T), stream));

    std::vector<std::uint64_t> h_row_global(l_rows);
    for (std::size_t s = 0; s < nseg; ++s)
        for (std::size_t r = 0; r < seg_lens[s]; ++r)
            h_row_global[seg_local_offs[s] + r] =
                static_cast<std::uint64_t>(seg_global_offs[s] + r);
    if (l_rows > 0)
    {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_row_global, l_rows * sizeof(std::uint64_t)));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_row_global, h_row_global.data(),
            l_rows * sizeof(std::uint64_t), cudaMemcpyHostToDevice, stream));
    }

    cudaStream_t stream_panel = nullptr;
    int prio_low = 0, prio_high = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(
        &stream_panel, cudaStreamNonBlocking, prio_high));
    cublasHandle_t cublas_panel = nullptr;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_panel));
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_panel, CUBLAS_POINTER_MODE_DEVICE));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_panel, stream_panel));

    std::vector<cudaEvent_t> event_cols_ready(num_blocks), event_panel_done(num_blocks);
    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event_cols_ready[b], cudaEventDisableTiming));
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event_panel_done[b], cudaEventDisableTiming));
    }

    // Prime pipeline with block-0 panel factor on stream_panel.
    if (num_blocks > 0)
    {
        const std::size_t jb0 = std::min(nb, n);
        CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[0], stream));
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[0], 0));
        cuda_nccl::distributed_houseQR_panel_factor_block_cyclic_1d<T>(
            n, m_global, seg_global_offs, seg_local_offs, seg_lens, ldv, V, 0,
            jb0, nb_dist, d_tau, cublas_panel, d_real_scalar[0], d_T_scalar[0],
            d_one, d_zero, d_minus_one, d_panel_scalars[0], d_w_panel[0],
            nccl_col_comm, l_rows, d_row_global, d_r_diag,
            tuning, enable_timing ? &panel_blk : nullptr);
        CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[0], stream_panel));
        if (enable_timing)
            sum_panel_cuda_wall_ms += panel_blk.panel_total_ms;
    }

    // Forward blocked factor/update with physically cleaned V blocks.
    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        const std::size_t k  = b * nb;
        const std::size_t jb = std::min(nb, n - k);
        T* Tb = d_T_blocks + b * nb * nb;
        const int next = static_cast<int>(1 - (b % 2));

        CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, event_panel_done[b], 0));
        // Keep aggregate panel timer semantics.
        if (enable_timing)
            t_panel_ms += panel_blk.panel_total_ms;

        // t_tbuild_ms: S = V^H V (GEMM K = l_rows), jb×jb AllReduce, optional
        // strict barrier, run_compute_T_block — not the T kernel in isolation.
        time_scope(t_tbuild_ms, [&]() {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, cublas_op_c, CUBLAS_OP_N,
                jb, jb, l_rows, d_one,
                V + k * ldv, ldv,
                V + k * ldv, ldv,
                d_zero, d_TW, jb));
#if CHASE_PANEL_HIPREC
            nccl_split_sync_fix_allreduce<T>(
                stream, d_TW, d_TW, jb * jb, nccl_col_comm, d_split_hi,
                d_split_lo, cublas_handle, rank, b, "tbuild");
#else
            chase::nccl::ncclAllReduceWrapper<T>(
                d_TW, d_TW, jb * jb, ncclSum, nccl_col_comm, &stream);
#endif
            chase::linalg::internal::cuda::run_compute_T_block<T>(
                stream, Tb, d_TW, d_tau + k, static_cast<int>(jb), static_cast<int>(nb));
#if CHASE_PANEL_HIPREC
            nccl_trace_tmatrix_condition<T>(Tb, static_cast<int>(jb), static_cast<int>(nb), rank, b);
#endif
        });

        const std::size_t n_trail = n - k - jb;
        if (n_trail > 0)
        {
            time_scope(t_trail_ms, [&]() {
                const std::size_t jb_next = std::min(nb, n_trail);
                const std::size_t n_rest = n_trail - jb_next;

                // Phase A: update next panel columns first.
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    jb, jb_next, l_rows, d_one,
                    V + k * ldv, ldv,
                    V + (k + jb) * ldv, ldv,
                    d_zero, d_W, jb));
#if CHASE_PANEL_HIPREC
                nccl_split_sync_fix_allreduce<T>(
                    stream, d_W, d_W, jb * jb_next, nccl_col_comm, d_split_hi,
                    d_split_lo, cublas_handle, rank, b, "trail_next");
#else
                chase::nccl::ncclAllReduceWrapper<T>(
                    d_W, d_W, jb * jb_next, ncclSum, nccl_col_comm, &stream);
#endif
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                    jb, jb_next, jb, d_one, Tb, nb, d_W, jb, d_zero, d_TW, jb));
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    l_rows, jb_next, jb, d_minus_one,
                    V + k * ldv, ldv,
                    d_TW, jb, d_one,
                    V + (k + jb) * ldv, ldv));

                // Launch next panel as soon as next block columns are ready.
                if (b + 1 < num_blocks)
                {
                    CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[b + 1], stream));
                    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[b + 1], 0));
                    const std::size_t k_next  = k + jb;
                    const std::size_t jb_fact = std::min(nb, n - k_next);
                    cuda_nccl::distributed_houseQR_panel_factor_block_cyclic_1d<T>(
                        n, m_global, seg_global_offs, seg_local_offs, seg_lens, ldv,
                        V, k_next, jb_fact, nb_dist, d_tau, cublas_panel,
                        d_real_scalar[next], d_T_scalar[next], d_one, d_zero,
                        d_minus_one, d_panel_scalars[next], d_w_panel[next],
                        nccl_col_comm, l_rows, d_row_global, d_r_diag,
                        tuning, enable_timing ? &panel_blk : nullptr);
                    CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[b + 1], stream_panel));
                    if (enable_timing)
                        sum_panel_cuda_wall_ms += panel_blk.panel_total_ms;
                }

                // Phase B: update remaining trailing columns while next panel runs.
                if (n_rest > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, cublas_op_c, CUBLAS_OP_N,
                        jb, n_rest, l_rows, d_one,
                        V + k * ldv, ldv,
                        V + (k + jb + jb_next) * ldv, ldv,
                        d_zero, d_W, jb));
#if CHASE_PANEL_HIPREC
                    nccl_split_sync_fix_allreduce<T>(
                        stream, d_W, d_W, jb * n_rest, nccl_col_comm, d_split_hi,
                        d_split_lo, cublas_handle, rank, b, "trail_rest");
#else
                    chase::nccl::ncclAllReduceWrapper<T>(
                        d_W, d_W, jb * n_rest, ncclSum, nccl_col_comm, &stream);
#endif
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                        jb, n_rest, jb, d_one, Tb, nb, d_W, jb, d_zero, d_TW, jb));
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        l_rows, n_rest, jb, d_minus_one,
                        V + k * ldv, ldv, d_TW, jb, d_one,
                        V + (k + jb + jb_next) * ldv, ldv));
                }
            });
        }
        else if (b + 1 < num_blocks)
        {
            // No trailing matrix left to hide behind; launch next panel directly.
            const std::size_t k_next  = k + jb;
            const std::size_t jb_fact = std::min(nb, n - k_next);
            CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[b + 1], stream));
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[b + 1], 0));
            cuda_nccl::distributed_houseQR_panel_factor_block_cyclic_1d<T>(
                n, m_global, seg_global_offs, seg_local_offs, seg_lens, ldv, V,
                k_next, jb_fact, nb_dist, d_tau, cublas_panel,
                d_real_scalar[next], d_T_scalar[next], d_one, d_zero, d_minus_one,
                d_panel_scalars[next], d_w_panel[next], nccl_col_comm, l_rows,
                d_row_global, d_r_diag, tuning, enable_timing ? &panel_blk : nullptr);
            CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[b + 1], stream_panel));
            if (enable_timing)
                sum_panel_cuda_wall_ms += panel_blk.panel_total_ms;
        }
    }
    if (num_blocks > 0)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, event_panel_done[num_blocks - 1], 0));

    time_scope(t_initq_ms, [&]() {
        chase::linalg::internal::cuda::t_lacpy('A', l_rows, n, V, ldv, d_VH, l_rows, &stream);
        CHECK_CUDA_ERROR(cudaMemsetAsync(V, 0, ldv * n * sizeof(T), stream));
        for (std::size_t c = 0; c < n; ++c)
        {
            if (row_seg[c] < nseg)
                CHECK_CUDA_ERROR(cudaMemcpyAsync(V + row_local[c] + c * ldv, d_one,
                                                 sizeof(T), cudaMemcpyDeviceToDevice, stream));
        }
    });

    // Backward blocked application to form Q.
    for (std::size_t bb = 0; bb < num_blocks; ++bb)
    {
        const std::size_t b = num_blocks - 1 - bb;
        const std::size_t k = b * nb;
        const std::size_t jb = std::min(nb, n - k);
        const std::size_t n_cols = n - k;
        const int bw = static_cast<int>(bb & 1u);
        T* d_W_curr = d_W_buf[bw];
        T* d_TW_curr = d_TW_buf[bw];
        T* Tb = d_T_blocks + b * nb * nb;

        const std::size_t chunks = std::max<std::size_t>(
            1, std::min<std::size_t>(static_cast<std::size_t>(formq_chunks), n_cols));
        for (std::size_t ch = 0; ch < chunks; ++ch)
        {
            const std::size_t c0 = (ch * n_cols) / chunks;
            const std::size_t c1 = ((ch + 1) * n_cols) / chunks;
            const std::size_t cw = c1 - c0;
            if (cw == 0) continue;

            time_scope(t_vhq_gemm_ms, [&]() {
                // NOTE [blocked-cyclic backward formQ]:
                // This GEMM may execute with l_rows==0; we intentionally rely on
                // cuBLAS zero-size behavior instead of explicit memset of d_W_curr.
                // The following NCCL allreduce is still called on all ranks with
                // globally identical sizes, so no collective-order divergence.
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    jb, cw, l_rows, d_one,
                    d_VH + k * l_rows, l_rows,
                    V + (k + c0) * ldv, ldv,
                    d_zero, d_W_curr, jb));
            });
            time_scope(t_vhq_nccl_ms, [&]() {
#if CHASE_PANEL_HIPREC
                nccl_split_sync_fix_allreduce<T>(
                    stream, d_W_curr, d_W_curr, jb * cw, nccl_col_comm, d_split_hi,
                    d_split_lo, cublas_handle, rank, b, "formQ_chunk");
#else
                chase::nccl::ncclAllReduceWrapper<T>(
                    d_W_curr, d_W_curr, jb * cw, ncclSum, nccl_col_comm, &stream);
#endif
            });
            time_scope(t_apply_t_ms, [&]() {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    jb, cw, jb, d_one, Tb, nb, d_W_curr, jb, d_zero, d_TW_curr, jb));
            });
            time_scope(t_rankk_update_ms, [&]() {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    l_rows, cw, jb, d_minus_one,
                    d_VH + k * l_rows, l_rows,
                    d_TW_curr, jb, d_one,
                    V + (k + c0) * ldv, ldv));
            });
        }
    }
    t_formq_ms = t_vhq_gemm_ms + t_vhq_nccl_ms + t_apply_t_ms + t_rankk_update_ms;

    if (enable_timing)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_total_end, stream));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_total_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_total_ms, ev_total_start, ev_total_end));
        if (rank == 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "[Householder QR blocked cyclic] Total: " << (t_total_ms / 1000.0) << " s  "
                << "(n=" << n << " nb=" << nb << " l_rows=" << l_rows << ")\n"
                << "  Breakdown: panel=" << (t_panel_ms / 1000.0) << " s  "
                << "T_build=" << (t_tbuild_ms / 1000.0) << " s  "
                << "trailing=" << (t_trail_ms / 1000.0) << " s  "
                << "init_formQ=" << (t_initq_ms / 1000.0) << " s  "
                << "formQ_backward=" << (t_formq_ms / 1000.0) << " s\n"
                << "  Backward breakdown: vhq_gemm="
                << (t_vhq_gemm_ms / 1000.0) << " s  vhq_nccl="
                << (t_vhq_nccl_ms / 1000.0) << " s  apply_t="
                << (t_apply_t_ms / 1000.0) << " s  rankk_update="
                << (t_rankk_update_ms / 1000.0) << " s  chunks="
                << formq_chunks << "\n"
                << "  Panel CUDA wall (sum over blocks): " << sum_panel_cuda_wall_ms << " ms\n";
            chase::GetLogger().Log(chase::LogLevel::Debug, "linalg", oss.str(), rank);
        }
        cudaEventDestroy(ev_a);
        cudaEventDestroy(ev_b);
        cudaEventDestroy(ev_total_start);
        cudaEventDestroy(ev_total_end);
    }

    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        cudaEventDestroy(event_cols_ready[b]);
        cudaEventDestroy(event_panel_done[b]);
    }
    cublasDestroy(cublas_panel);
    cudaStreamDestroy(stream_panel);

    cudaFree(d_real_scalar[0]); cudaFree(d_real_scalar[1]);
    cudaFree(d_T_scalar[0]); cudaFree(d_T_scalar[1]);
    cudaFree(d_w_panel[0]); cudaFree(d_w_panel[1]);
    cudaFree(d_one); cudaFree(d_zero);
    cudaFree(d_minus_one); cudaFree(d_panel_scalars[0]); cudaFree(d_panel_scalars[1]); cudaFree(d_tau); cudaFree(d_VH);
    cudaFree(d_T_blocks);
    cudaFree(d_W_buf[0]); cudaFree(d_W_buf[1]);
    cudaFree(d_TW_buf[0]); cudaFree(d_TW_buf[1]);
    cudaFree(d_r_diag);
    cudaFree(d_split_hi); cudaFree(d_split_lo);
    if (d_row_global) cudaFree(d_row_global);
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
}

//==============================================================================
// houseQR1_formQ
//
// GPU-facing entry point.  V stays on device; distributed_blocked_houseQR_formQ
// runs GPU-resident (no full matrix H2D/D2H), so timing improves vs host round-trip.
//==============================================================================
template <typename InputMultiVectorType>
void cuda_nccl::houseQR1_formQ(cublasHandle_t cublas_handle,
                               InputMultiVectorType& V1,
                               typename InputMultiVectorType::value_type* workspace,
                               int lwork,
                               const HouseQRTuning* tuning)
{
    using T = typename InputMultiVectorType::value_type;

    const std::size_t m_global = V1.g_rows();
    const std::size_t n        = V1.l_cols();
    const std::size_t l_rows   = V1.l_rows();
    const std::size_t g_off    = V1.g_off();
    const std::size_t ldv      = V1.l_ld();
    MPI_Comm mpi_comm          = V1.getMpiGrid()->get_col_comm();
    ncclComm_t nccl_col_comm   = V1.getMpiGrid()->get_nccl_col_comm();
    int qr_rank = 0;
    MPI_Comm_rank(mpi_comm, &qr_rank);

    // Create a dedicated non-blocking stream for this QR and temporarily
    // attach it to the cuBLAS / cuSOLVER handles, so callers don't have to
    // manage streams explicitly.
    cudaStream_t prev_cublas_stream = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &prev_cublas_stream));

    cudaStream_t qr_stream = nullptr;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&qr_stream, cudaStreamNonBlocking));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, qr_stream));
 
    cudaEvent_t evt;
    cudaEventCreate(&evt);
    
    // record completion in your stream
    cudaEventRecord(evt, 0);
    // make default stream wait
    cudaStreamWaitEvent(qr_stream, evt, 0);

    std::size_t lwork_elems = (workspace && lwork > 0) ? static_cast<std::size_t>(lwork) : 0;
    if constexpr (chase::distMultiVector::is_block_cyclic_1d_multivector<InputMultiVectorType>::value)
    {
        const auto seg_g = V1.m_contiguous_global_offs();
        const auto seg_l = V1.m_contiguous_local_offs();
        const auto seg_n = V1.m_contiguous_lens();
        const std::size_t nb_dist = V1.mb();
        cuda_nccl::distributed_blocked_houseQR_formQ_block_cyclic_1d<T>(
            m_global, n, seg_g, seg_l, seg_n, ldv, V1.l_data(), mpi_comm,
            nb_dist, cublas_handle, workspace, lwork_elems, nccl_col_comm,
            tuning);
    }
    else
    {
        cuda_nccl::distributed_blocked_houseQR_formQ<T>(
            m_global, n, l_rows, g_off,
            ldv,
            V1.l_data(),
            mpi_comm,
            cublas_handle,
            workspace,
            lwork_elems,
            nccl_col_comm,
            tuning);
    }

    if (nccl_should_validate_orthogonality())
    {
        nccl_validate_orthogonality<T>(cublas_handle, V1.l_data(), l_rows, n,
                                       ldv, nccl_col_comm, qr_rank);
    }

    // Restore original streams and destroy the temporary one.
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, prev_cublas_stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(qr_stream));
}

} // namespace internal
} // namespace linalg
} // namespace chase
