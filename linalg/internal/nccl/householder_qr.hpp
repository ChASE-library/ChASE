// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
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

namespace chase
{
namespace linalg
{
namespace internal
{

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
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

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
    }

    for (std::size_t jj = 0; jj < jb; ++jj)
    {
        const std::size_t col       = k + jj;
        const bool        pivot_here = (g_off <= col && col < g_off + l_rows);
        const std::size_t pivot_loc  = pivot_here ? (col - g_off) : 0;

        std::size_t rs, vr;
        if (col < g_off)             { rs = 0;           vr = l_rows; }
        else if (col < g_off+l_rows) { rs = col - g_off; vr = l_rows - rs; }
        else                         { rs = 0;           vr = 0; }

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
                CHECK_CUDA_ERROR(cudaMemcpyAsync(d_real_scalar, d_T_scalar, sizeof(RealT), cudaMemcpyDeviceToDevice, stream));
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

        // Phase 2: scalar allreduces
        if (do_timing && jj < jb_ev)
            CHECK_CUDA_ERROR(cudaEventRecord(ev_start[jj * 5u + 2], stream));
        chase::nccl::ncclAllReduceWrapper<T>(d_tau + col, d_T_scalar, 1, ncclSum, nccl_col_comm, &stream);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_tau + col, d_T_scalar, sizeof(T), cudaMemcpyDeviceToDevice, stream));
        chase::nccl::ncclAllReduceWrapper<T>(d_denom_bcast, d_denom_bcast, 1, ncclSum, nccl_col_comm, &stream);
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

    if (do_timing && jb_ev > 0)
    {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
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
                                             ncclComm_t nccl_col_comm)
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
// distributed_blocked_houseQR_formQ (NCCL — V on device, NCCL collectives only)
//
// Blocked distributed Householder QR + form Q (compact WY). nb = panel size.
//==============================================================================
template <typename T>
void cuda_nccl::distributed_blocked_houseQR_formQ(std::size_t m_global,
                                                     std::size_t n,
                                                     std::size_t l_rows,
                                                     std::size_t g_off,
                                                     std::size_t ldv,
                                                     T*          V,
                                                     MPI_Comm    mpi_comm,
                                                     std::size_t nb,
                                                     cublasHandle_t cublas_handle,
                                                     T* d_workspace,
                                                     std::size_t lwork_elems,
                                                     ncclComm_t nccl_col_comm)
{
    using RealT = chase::Base<T>;

    if (n == 0)
        return;

    if (nb == 0 || nb >= n)
    {
        cuda_nccl::distributed_houseQR_formQ<T>(
            m_global, n, l_rows, g_off, ldv, V, mpi_comm,
            cublas_handle, d_workspace, lwork_elems, nccl_col_comm);
        return;
    }

    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));

    // stream_compute = stream already attached to cublas_handle (set by houseQR1_formQ).
    cudaStream_t stream_compute = nullptr;
    CHECK_CUBLAS_ERROR(cublasGetStream(cublas_handle, &stream_compute));

    // stream_panel = high-priority non-blocking stream for look-ahead panel factor.
    int leastPriority = 0, greatestPriority = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    cudaStream_t stream_panel = nullptr;
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream_panel,
                                                   cudaStreamNonBlocking,
                                                   greatestPriority));

    // Dedicated cuBLAS handle for the panel stream so panel GEMM/dot ops run on stream_panel.
    cublasHandle_t cublas_panel = nullptr;
    CHECK_CUBLAS_ERROR(cublasCreate(&cublas_panel));
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_panel, stream_panel));
    CHECK_CUBLAS_ERROR(cublasSetPointerMode(cublas_panel, CUBLAS_POINTER_MODE_DEVICE));

    // Synchronisation events between the two streams.
    // event_cols_ready[b]: signals that the next block's columns have been updated
    //                      on stream_compute and are safe to factor on stream_panel.
    // event_panel_done[b]: signals that block b's panel factor is complete on stream_panel.
    const std::size_t num_blocks = (n + nb - 1) / nb;
    std::vector<cudaEvent_t> event_cols_ready(num_blocks, nullptr);
    std::vector<cudaEvent_t> event_panel_done(num_blocks, nullptr);
    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        // cudaEventDisableTiming avoids timing overhead on hot-path sync events.
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event_cols_ready[b], cudaEventDisableTiming));
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&event_panel_done[b], cudaEventDisableTiming));
    }

    int rank = 0;
    MPI_Comm_rank(mpi_comm, &rank);

    const bool enable_timing_blocked =
        static_cast<int>(chase::GetLogger().GetLevel()) >=
        static_cast<int>(chase::LogLevel::Debug);

    cudaEvent_t ev_start = nullptr, ev_panel_end = nullptr, ev_formQ_end = nullptr;
    if (enable_timing_blocked)
    {
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_start));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_panel_end));
        CHECK_CUDA_ERROR(cudaEventCreate(&ev_formQ_end));
        CHECK_CUDA_ERROR(cudaEventRecord(ev_start, stream_compute));
    }

    const T one  = T(1);
    const T zero = T(0);
    const T minus_one = T(-1);
    const cublasOperation_t cublas_op_c =
        (std::is_same<T, std::complex<float>>::value ||
         std::is_same<T, std::complex<double>>::value)
            ? CUBLAS_OP_C : CUBLAS_OP_T;

    const std::size_t need_VH    = l_rows * n;
    const std::size_t need_Tall  = num_blocks * nb * nb;
    const std::size_t need_W     = nb * n;
    // Double-buffer: 2 x (d_w[nb] + d_panel_scalars[4] + d_real_scalar[1] + d_T_scalar[1])
    const std::size_t need_blocked = 7 + need_VH + n + n + need_Tall
        + need_W * 2 + nb * 2 + nb * nb; // +4 for panel scalars
    bool use_ws = (d_workspace != nullptr && lwork_elems >= need_blocked);
    T* d_V          = V;
    T* d_VH         = nullptr;
    T* d_tau        = nullptr;
    T* d_w[2]       = {nullptr, nullptr};        // double-buffered panel scratch
    T* d_T_blocks   = nullptr;
    T* d_W          = nullptr;
    T* d_TW         = nullptr;
    T* d_t_col      = nullptr;
    T* d_tmp        = nullptr;
    T* d_saved      = nullptr;
    T* d_one        = nullptr;
    T* d_zero       = nullptr;
    T* d_minus_one  = nullptr;
    T* d_panel_scalars[2] = {nullptr, nullptr};  // double-buffered panel scalars
    RealT* d_real_scalar[2] = {nullptr, nullptr}; // double-buffered per-stream scalar
    T*     d_T_scalar[2]    = {nullptr, nullptr};  // double-buffered per-stream scalar

    bool alloc_blocked = !use_ws;
    if (use_ws)
    {
        d_one           = d_workspace;
        d_zero          = d_workspace + 1;
        d_minus_one     = d_workspace + 2;
        // panel_scalars[0] uses workspace slots 3-6, [1] is freshly allocated below
        d_panel_scalars[0] = d_workspace + 3;
        d_VH            = d_workspace + 7;
        d_tau           = d_workspace + 7 + need_VH;
        d_w[0]          = d_workspace + 7 + need_VH + n;
        d_T_blocks      = d_workspace + 7 + need_VH + 2 * n;
        d_W             = d_workspace + 7 + need_VH + 2 * n + need_Tall;
        d_TW            = d_W + need_W;
        d_t_col         = d_TW + need_W;
        d_tmp           = d_t_col + nb;
        d_saved         = d_tmp + nb;
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_one, &one, sizeof(T), cudaMemcpyHostToDevice, stream_compute));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_zero, &zero, sizeof(T), cudaMemcpyHostToDevice, stream_compute));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_minus_one, &minus_one, sizeof(T), cudaMemcpyHostToDevice, stream_compute));
    }
    else
    {
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_one, sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_zero, sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_minus_one, sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars[0], 4 * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_one, &one, sizeof(T), cudaMemcpyHostToDevice, stream_compute));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_zero, &zero, sizeof(T), cudaMemcpyHostToDevice, stream_compute));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_minus_one, &minus_one, sizeof(T), cudaMemcpyHostToDevice, stream_compute));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_VH, need_VH * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tau, n * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w[0], nb * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_blocks, need_Tall * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_W, need_W * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_TW, need_W * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_t_col, nb * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tmp, nb * sizeof(T)));
        CHECK_CUDA_ERROR(cudaMalloc((void**)&d_saved, nb * nb * sizeof(T)));
    }
    // Second buffer slot always freshly allocated (small: nb + 4 elements).
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_panel_scalars[1], 4 * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_w[1], nb * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar[0], sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_real_scalar[1], sizeof(RealT)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar[0], sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_T_scalar[1], sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemsetAsync(d_T_blocks, 0, need_Tall * sizeof(T), stream_compute));

    // -------------------------------------------------------------------------
    // Block 0 panel factor: no look-ahead yet; use stream_panel slot 0.
    // -------------------------------------------------------------------------
    // Make stream_panel wait until stream_compute has initialised all constants.
    CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[0], stream_compute));
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[0], 0));

    {
        const std::size_t jb0 = std::min(nb, n);
        cublasSetStream(cublas_panel, stream_panel);
        cuda_nccl::distributed_houseQR_panel_factor<T>(
            n, l_rows, g_off, ldv, d_V, 0, jb0, d_tau,
            cublas_panel,
            d_real_scalar[0], d_T_scalar[0], d_one, d_zero, d_minus_one,
            d_panel_scalars[0], d_w[0],
            nccl_col_comm,
            nullptr);
        CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[0], stream_panel));
    }

    chase::linalg::internal::cuda_nccl::HouseholderPanelTiming panel_timing;
    // -------------------------------------------------------------------------
    // Main look-ahead loop.
    // -------------------------------------------------------------------------
    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        const std::size_t k  = b * nb;
        const std::size_t jb = std::min(nb, n - k);
        T* Tb = d_T_blocks + b * nb * nb;
        const int next = static_cast<int>(1 - b % 2);

        // Wait for panel factor of block b (launched on stream_panel in previous iteration).
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_compute, event_panel_done[b], 0));

        // Build T-matrix for block b (compute stream).
        std::size_t rs_k, vr_k;
        if (k < g_off)             { rs_k = 0;         vr_k = l_rows; }
        else if (k < g_off+l_rows) { rs_k = k - g_off; vr_k = l_rows - rs_k; }
        else                       { rs_k = 0;         vr_k = 0; }

        chase::linalg::internal::cuda::run_batch_save_restore_upper_triangular<T>(
            stream_compute, d_V, d_saved, ldv, jb, k, g_off, l_rows,
            d_one, d_zero, true);

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_TW, 0, jb * jb * sizeof(T), stream_compute));
        if (vr_k > 0)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, cublas_op_c, CUBLAS_OP_N,
                jb, jb, vr_k, d_one,
                d_V + rs_k + k * ldv, ldv,
                d_V + rs_k + k * ldv, ldv,
                d_zero, d_TW, jb));
        }
        chase::nccl::ncclAllReduceWrapper<T>(d_TW, d_TW, jb * jb, ncclSum, nccl_col_comm, &stream_compute);
        chase::linalg::internal::cuda::run_compute_T_block<T>(
            stream_compute, Tb, d_TW, d_tau + k, static_cast<int>(jb), static_cast<int>(nb));

        chase::linalg::internal::cuda::run_batch_save_restore_upper_triangular<T>(
            stream_compute, d_V, d_saved, ldv, jb, k, g_off, l_rows,
            d_one, d_zero, false);

        // --- Trailing update (look-ahead split) ---
        const std::size_t n_trail = n - k - jb;
        if (n_trail > 0)
        {
            // Phase A: update only the next block's nb columns on stream_compute first,
            //          so stream_panel can start the next panel factor as early as possible.
            const std::size_t jb_next   = std::min(nb, n_trail);
            const std::size_t n_rest    = n_trail - jb_next;

            chase::linalg::internal::cuda::run_batch_save_restore_upper_triangular<T>(
                stream_compute, d_V, d_saved, ldv, jb, k, g_off, l_rows,
                d_one, d_zero, true);

            // W_next = V_curr^H * V_trail_next  (jb x jb_next)
            CHECK_CUDA_ERROR(cudaMemsetAsync(d_W, 0, jb * jb_next * sizeof(T), stream_compute));
            if (vr_k > 0)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, cublas_op_c, CUBLAS_OP_N,
                    jb, jb_next, vr_k, d_one,
                    d_V + rs_k + k * ldv, ldv,
                    d_V + rs_k + (k + jb) * ldv, ldv,
                    d_zero, d_W, jb));
            }
            chase::nccl::ncclAllReduceWrapper<T>(d_W, d_W, jb * jb_next, ncclSum, nccl_col_comm, &stream_compute);

            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                jb, jb_next, jb, d_one,
                Tb, nb, d_W, jb, d_zero, d_TW, jb));

            if (vr_k > 0)
            {
                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    vr_k, jb_next, jb, d_minus_one,
                    d_V + rs_k + k * ldv, ldv,
                    d_TW, jb, d_one,
                    d_V + rs_k + (k + jb) * ldv, ldv));
            }

            // Signal that the next block's columns are ready to factor.
            if (b + 1 < num_blocks)
            {
                CHECK_CUDA_ERROR(cudaEventRecord(event_cols_ready[b + 1], stream_compute));

                // Phase B: launch next panel factor on stream_panel (look-ahead).
                CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_panel, event_cols_ready[b + 1], 0));
                CHECK_CUBLAS_ERROR(cublasSetStream(cublas_panel, stream_panel));
                cuda_nccl::distributed_houseQR_panel_factor<T>(
                    n, l_rows, g_off, ldv, d_V, k + jb, jb_next, d_tau,
                    cublas_panel,
                    d_real_scalar[next], d_T_scalar[next], d_one, d_zero, d_minus_one,
                    d_panel_scalars[next], d_w[next],
                    nccl_col_comm,
                    &panel_timing);
                CHECK_CUDA_ERROR(cudaEventRecord(event_panel_done[b + 1], stream_panel));
            }

            // Phase C: update the remaining trailing columns on stream_compute
            //          (this GEMM runs concurrently with the panel factor above).
            if (n_rest > 0)
            {
                CHECK_CUDA_ERROR(cudaMemsetAsync(d_W, 0, jb * n_rest * sizeof(T), stream_compute));
                if (vr_k > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, cublas_op_c, CUBLAS_OP_N,
                        jb, n_rest, vr_k, d_one,
                        d_V + rs_k + k * ldv, ldv,
                        d_V + rs_k + (k + jb + jb_next) * ldv, ldv,
                        d_zero, d_W, jb));
                }
                chase::nccl::ncclAllReduceWrapper<T>(d_W, d_W, jb * n_rest, ncclSum, nccl_col_comm, &stream_compute);

                CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                    cublas_handle, CUBLAS_OP_C, CUBLAS_OP_N,
                    jb, n_rest, jb, d_one,
                    Tb, nb, d_W, jb, d_zero, d_TW, jb));

                if (vr_k > 0)
                {
                    CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        vr_k, n_rest, jb, d_minus_one,
                        d_V + rs_k + k * ldv, ldv,
                        d_TW, jb, d_one,
                        d_V + rs_k + (k + jb + jb_next) * ldv, ldv));
                }
            }

            chase::linalg::internal::cuda::run_batch_save_restore_upper_triangular<T>(
                stream_compute, d_V, d_saved, ldv, jb, k, g_off, l_rows,
                d_one, d_zero, false);
        }
    }

    // Ensure all panel and compute work is done before formQ.
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream_compute, event_panel_done[num_blocks - 1], 0));

    if (enable_timing_blocked)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_panel_end, stream_compute));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_panel_end));
    }

    chase::linalg::internal::cuda::t_lacpy('A', l_rows, n, d_V, ldv, d_VH, l_rows, &stream_compute);
    chase::linalg::internal::cuda::run_init_identity_distributed<T>(
        stream_compute, d_V, ldv, n, g_off, l_rows);

    // Form-Q backward loop uses T_blocks (compact WY) — single stream_compute.
    for (std::size_t bb = 0; bb < num_blocks; ++bb)
    {
        const std::size_t b      = num_blocks - 1 - bb;
        const std::size_t k      = b * nb;
        const std::size_t jb     = std::min(nb, n - k);
        const std::size_t n_cols = n - k;
        T* Tb = d_T_blocks + b * nb * nb;

        std::size_t rs_k, vr_k;
        if (k < g_off)             { rs_k = 0;         vr_k = l_rows; }
        else if (k < g_off+l_rows) { rs_k = k - g_off; vr_k = l_rows - rs_k; }
        else                       { rs_k = 0;         vr_k = 0; }

        chase::linalg::internal::cuda::run_batch_save_restore_upper_triangular<T>(
            stream_compute, d_VH, d_saved, l_rows, jb, k, g_off, l_rows,
            d_one, d_zero, true);

        CHECK_CUDA_ERROR(cudaMemsetAsync(d_W, 0, jb * n_cols * sizeof(T), stream_compute));
        if (vr_k > 0)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, cublas_op_c, CUBLAS_OP_N,
                jb, n_cols, vr_k, d_one,
                d_VH + rs_k + k * l_rows, l_rows,
                d_V + rs_k + k * ldv, ldv,
                d_zero, d_W, jb));
        }
        chase::nccl::ncclAllReduceWrapper<T>(d_W, d_W, jb * n_cols, ncclSum, nccl_col_comm, &stream_compute);

        CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            jb, n_cols, jb, d_one,
            Tb, nb, d_W, jb, d_zero, d_TW, jb));

        if (vr_k > 0)
        {
            CHECK_CUBLAS_ERROR(chase::linalg::cublaspp::cublasTgemm(
                cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                vr_k, n_cols, jb, d_minus_one,
                d_VH + rs_k + k * l_rows, l_rows,
                d_TW, jb, d_one,
                d_V + rs_k + k * ldv, ldv));
        }

        chase::linalg::internal::cuda::run_batch_save_restore_upper_triangular<T>(
            stream_compute, d_VH, d_saved, l_rows, jb, k, g_off, l_rows,
            d_one, d_zero, false);
    }

    if (enable_timing_blocked)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(ev_formQ_end, stream_compute));
        CHECK_CUDA_ERROR(cudaEventSynchronize(ev_formQ_end));
        float t_panel_ms = 0.f, t_formQ_ms = 0.f, t_total_ms = 0.f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_panel_ms, ev_start, ev_panel_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_formQ_ms, ev_panel_end, ev_formQ_end));
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&t_total_ms, ev_start, ev_formQ_end));
        if (rank == 0)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3)
                << "[Householder QR blocked look-ahead] Total: " << (t_total_ms / 1000.0) << " s  "
                << "Panel: " << (t_panel_ms / 1000.0) << " s  "
                << "FormQ: " << (t_formQ_ms / 1000.0) << " s  "
                << "(n=" << n << " nb=" << nb << " l_rows=" << l_rows << ")\n";
            oss << "  Panel breakdown: norm=" << (panel_timing.norm_ms / 1000.0)
                << " s  scalar_kernel=" << (panel_timing.scalar_kernel_ms / 1000.0)
                << " s  allreduce_tau=" << (panel_timing.allreduce_tau_ms / 1000.0)
                << " s  scal=" << (panel_timing.scal_ms / 1000.0)
                << " s  trail=" << (panel_timing.trail_ms / 1000.0) << " s\n";
            chase::GetLogger().Log(chase::LogLevel::Trace, "linalg", oss.str(), rank);
        }
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_panel_end);
        cudaEventDestroy(ev_formQ_end);
    }

    // Destroy look-ahead events and secondary resources.
    for (std::size_t b = 0; b < num_blocks; ++b)
    {
        cudaEventDestroy(event_cols_ready[b]);
        cudaEventDestroy(event_panel_done[b]);
    }
    cublasDestroy(cublas_panel);
    cudaStreamDestroy(stream_panel);

    cudaFree(d_real_scalar[0]);
    cudaFree(d_real_scalar[1]);
    cudaFree(d_T_scalar[0]);
    cudaFree(d_T_scalar[1]);
    cudaFree(d_panel_scalars[1]);
    cudaFree(d_w[1]);
    if (alloc_blocked)
    {
        cudaFree(d_one);
        cudaFree(d_zero);
        cudaFree(d_minus_one);
        cudaFree(d_panel_scalars[0]);
        cudaFree(d_VH);
        cudaFree(d_tau);
        cudaFree(d_w[0]);
        cudaFree(d_T_blocks);
        cudaFree(d_W);
        cudaFree(d_TW);
        cudaFree(d_t_col);
        cudaFree(d_tmp);
        cudaFree(d_saved);
    }
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
                               std::size_t nb)
{
    using T = typename InputMultiVectorType::value_type;

    const std::size_t m_global = V1.g_rows();
    const std::size_t n        = V1.l_cols();
    const std::size_t l_rows   = V1.l_rows();
    const std::size_t g_off    = V1.g_off();
    const std::size_t ldv      = V1.l_ld();
    MPI_Comm mpi_comm          = V1.getMpiGrid()->get_col_comm();
    ncclComm_t nccl_col_comm   = V1.getMpiGrid()->get_nccl_col_comm();

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
    cuda_nccl::distributed_blocked_houseQR_formQ<T>(
        m_global, n, l_rows, g_off,
        ldv,
        V1.l_data(),
        mpi_comm,
        nb,
        cublas_handle,
        workspace,
        lwork_elems,
        nccl_col_comm);

    // Restore original streams and destroy the temporary one.
    CHECK_CUBLAS_ERROR(cublasSetStream(cublas_handle, prev_cublas_stream));
    CHECK_CUDA_ERROR(cudaStreamDestroy(qr_stream));
}

} // namespace internal
} // namespace linalg
} // namespace chase
