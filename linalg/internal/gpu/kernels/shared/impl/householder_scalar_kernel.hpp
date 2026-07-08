// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <climits>
#include <cstdint>
#include <complex>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace CHASE_GPU_DEVICE_NAMESPACE
{

namespace
{
} // namespace

template <typename T, typename RealT>
void run_householder_scalar_kernel(CHASE_GPU_STREAM_T stream, int pivot_here,
                                  const T* d_x0, const RealT* d_nrm_sq, T* d_tau,
                                  T* d_inv_denom, T* d_neg_beta, T* d_denom_bcast,
                                  T* d_saved_rkk)
{
    if (stream == nullptr)
        stream = 0;
    if constexpr (std::is_same<T, float>::value)
    {
        householder_scalar_kernel_s<<<1, 1, 0, stream>>>(
            pivot_here, d_x0, d_nrm_sq, d_tau, d_inv_denom, d_neg_beta,
            d_denom_bcast, d_saved_rkk);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        householder_scalar_kernel_d<<<1, 1, 0, stream>>>(
            pivot_here, d_x0, d_nrm_sq, d_tau, d_inv_denom, d_neg_beta,
            d_denom_bcast, d_saved_rkk);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        householder_scalar_kernel_c<<<1, 1, 0, stream>>>(
            pivot_here, reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_x0),
            reinterpret_cast<const float*>(d_nrm_sq),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_tau),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_neg_beta),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_saved_rkk));
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value, "T must be float, double, std::complex<float>, or std::complex<double>");
        householder_scalar_kernel_z<<<1, 1, 0, stream>>>(
            pivot_here, reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_x0),
            reinterpret_cast<const double*>(d_nrm_sq),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_tau),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_neg_beta),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_saved_rkk));
    }
}

template <typename T>
void run_inv_denom_from_denom_bcast(CHASE_GPU_STREAM_T stream,
                                    const T* d_denom_bcast, T* d_inv_denom)
{
    if (stream == nullptr)
        stream = 0;
    if constexpr (std::is_same<T, float>::value)
    {
        inv_denom_from_denom_bcast_s<<<1, 1, 0, stream>>>(d_denom_bcast, d_inv_denom);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        inv_denom_from_denom_bcast_d<<<1, 1, 0, stream>>>(d_denom_bcast, d_inv_denom);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        inv_denom_from_denom_bcast_c<<<1, 1, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_inv_denom));
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value, "T must be float, double, or complex");
        inv_denom_from_denom_bcast_z<<<1, 1, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_inv_denom));
    }
}

template <typename T>
void run_nonpivot_scal_if_denom_nonzero(CHASE_GPU_STREAM_T stream,
                                       const T* d_denom_bcast, T* d_inv_denom,
                                       T* d_V_col, int n)
{
    if (stream == nullptr)
        stream = 0;
    if (n <= 0)
        return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    if constexpr (std::is_same<T, float>::value)
    {
        nonpivot_scal_if_denom_nonzero_s<<<grid, block, 0, stream>>>(
            d_denom_bcast, d_inv_denom, d_V_col, n);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        nonpivot_scal_if_denom_nonzero_d<<<grid, block, 0, stream>>>(
            d_denom_bcast, d_inv_denom, d_V_col, n);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        nonpivot_scal_if_denom_nonzero_c<<<grid, block, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_V_col), n);
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value, "T must be float, double, or complex");
        nonpivot_scal_if_denom_nonzero_z<<<grid, block, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_V_col), n);
    }
}

template <typename T>
void run_bc1d_post_comm_scal_pivot(CHASE_GPU_STREAM_T stream, int pivot_here,
                                   const T* d_denom_bcast, T* d_inv_denom,
                                   T* d_v_tail, int vr, int pivot_rel)
{
    if (stream == nullptr)
        stream = 0;
    if (vr <= 0 && !pivot_here)
        return;
    constexpr int kBlock = 256;
    const int ph = pivot_here ? 1 : 0;
    const int grid =
        vr > 0 ? (static_cast<int>(vr) + kBlock - 1) / kBlock : 1;
    if constexpr (std::is_same<T, float>::value)
    {
        bc1d_post_comm_scal_pivot_s<<<grid, kBlock, 0, stream>>>(
            ph, pivot_rel, d_denom_bcast, d_inv_denom, d_v_tail, vr);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        bc1d_post_comm_scal_pivot_d<<<grid, kBlock, 0, stream>>>(
            ph, pivot_rel, d_denom_bcast, d_inv_denom, d_v_tail, vr);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        bc1d_post_comm_scal_pivot_c<<<grid, kBlock, 0, stream>>>(
            ph,
            pivot_rel,
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_v_tail),
            vr);
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        bc1d_post_comm_scal_pivot_z<<<grid, kBlock, 0, stream>>>(
            ph,
            pivot_rel,
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_v_tail),
            vr);
    }
}

template <typename T>
void run_guarded_scaling(CHASE_GPU_STREAM_T stream, int n, const T* d_tau,
                         const T* d_inv_denom, T* d_V_col, bool pivot_here,
                         int pivot_loc, const T* d_neg_beta)
{
    if (stream == nullptr)
        stream = 0;
    if (n <= 0)
        return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    const int pivot = pivot_here ? 1 : 0;
    if constexpr (std::is_same<T, float>::value)
    {
        guarded_scaling_s<<<grid, block, 0, stream>>>(
            n, d_tau, d_inv_denom, d_V_col, pivot, pivot_loc, d_neg_beta);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        guarded_scaling_d<<<grid, block, 0, stream>>>(
            n, d_tau, d_inv_denom, d_V_col, pivot, pivot_loc, d_neg_beta);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        guarded_scaling_c<<<grid, block, 0, stream>>>(
            n,
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_tau),
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_V_col),
            pivot,
            pivot_loc,
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_neg_beta));
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value, "T must be float, double, or complex");
        guarded_scaling_z<<<grid, block, 0, stream>>>(
            n,
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_tau),
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_inv_denom),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_V_col),
            pivot,
            pivot_loc,
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_neg_beta));
    }
}

template <typename T>
void run_batch_save_restore_upper_triangular(CHASE_GPU_STREAM_T stream, T* d_V,
                                             T* d_saved, std::size_t ldv,
                                             std::size_t jb, std::size_t k,
                                             std::size_t g_off,
                                             std::size_t l_rows,
                                             const T* d_one, const T* d_zero,
                                             bool save_mode)
{
    if (stream == nullptr)
        stream = 0;
    if (jb == 0)
        return;
    (void)d_one;
    (void)d_zero;

    const int save_i = save_mode ? 1 : 0;
    constexpr int tile = 16;
    const dim3 block(tile, tile);
    const std::size_t grid_x = (jb + tile - 1) / tile;
    if (grid_x > static_cast<std::size_t>(UINT_MAX))
        throw std::runtime_error("run_batch_save_restore_upper_triangular: grid dimension overflow");
    const dim3 grid(static_cast<unsigned int>(grid_x),
                    static_cast<unsigned int>(grid_x),
                    1u);

    (void)CHASE_GPU_GET_LAST_ERROR();
    if constexpr (std::is_same<T, float>::value)
    {
        batch_save_restore_upper_triangular_s<<<grid, block, 0, stream>>>(
            d_V, d_saved, ldv, jb, k, g_off, l_rows, save_i);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        batch_save_restore_upper_triangular_d<<<grid, block, 0, stream>>>(
            d_V, d_saved, ldv, jb, k, g_off, l_rows, save_i);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        batch_save_restore_upper_triangular_c<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_V),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_saved),
            ldv, jb, k, g_off, l_rows, save_i);
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        batch_save_restore_upper_triangular_z<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_V),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_saved),
            ldv, jb, k, g_off, l_rows, save_i);
    }

    const CHASE_GPU_ERROR_T err = CHASE_GPU_PEEK_AT_LAST_ERROR();
    if (err != CHASE_GPU_SUCCESS)
    {
        std::ostringstream oss;
        oss << "batch_save_restore_upper_triangular kernel launch failed: "
            << CHASE_GPU_GET_ERROR_STRING(err)
            << " (jb=" << jb << ", grid=" << grid.x << "x" << grid.y
            << ", block=" << block.x << "x" << block.y
            << ", ldv=" << ldv << ", k=" << k
            << ", g_off=" << g_off << ", l_rows=" << l_rows << ")";
        throw std::runtime_error(oss.str());
    }
}

template <typename T>
void run_init_identity_distributed(CHASE_GPU_STREAM_T stream, T* d_V, std::size_t ldv,
                                   std::size_t n, std::size_t g_off,
                                   std::size_t l_rows)
{
    if (stream == nullptr)
        stream = 0;
    if (n == 0 || l_rows == 0)
        return;

    constexpr int block = 256;
    const std::size_t grid_x = (n + block - 1) / block;
    if (grid_x > static_cast<std::size_t>(UINT_MAX))
        throw std::runtime_error("run_init_identity_distributed: grid dimension overflow");
    const dim3 grid(static_cast<unsigned int>(grid_x), 1u, 1u);

    (void)CHASE_GPU_GET_LAST_ERROR();
    if constexpr (std::is_same<T, float>::value)
    {
        init_identity_distributed_s<<<grid, block, 0, stream>>>(
            d_V, ldv, n, g_off, l_rows);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        init_identity_distributed_d<<<grid, block, 0, stream>>>(
            d_V, ldv, n, g_off, l_rows);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        init_identity_distributed_c<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_V), ldv, n, g_off, l_rows);
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        init_identity_distributed_z<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_V), ldv, n, g_off, l_rows);
    }

    const CHASE_GPU_ERROR_T err = CHASE_GPU_PEEK_AT_LAST_ERROR();
    if (err != CHASE_GPU_SUCCESS)
    {
        std::ostringstream oss;
        oss << "init_identity_distributed kernel launch failed: "
            << CHASE_GPU_GET_ERROR_STRING(err)
            << " (n=" << n << ", l_rows=" << l_rows
            << ", ldv=" << ldv << ", g_off=" << g_off
            << ", grid=" << grid.x << ", block=" << block << ")";
        throw std::runtime_error(oss.str());
    }
}

template <typename T>
void run_split_and_pad_v_column(CHASE_GPU_STREAM_T stream, T* d_V_col0,
                                const std::uint64_t* d_row_global, int l_rows,
                                std::uint64_t pivot_global, int pivot_here,
                                int pivot_loc, const T* d_saved_rkk,
                                T* d_r_diag_out)
{
    if (stream == nullptr)
        stream = 0;
    if (l_rows <= 0 || d_V_col0 == nullptr || d_row_global == nullptr)
        return;
    constexpr int block = 256;
    const int grid = (l_rows + block - 1) / block;
    if (grid <= 0)
        return;
    (void)CHASE_GPU_GET_LAST_ERROR();
    if constexpr (std::is_same<T, float>::value)
    {
        split_and_pad_v_column_s<<<grid, block, 0, stream>>>(
            d_V_col0, reinterpret_cast<const unsigned long long*>(d_row_global),
            l_rows, static_cast<unsigned long long>(pivot_global), pivot_here,
            pivot_loc,
            reinterpret_cast<const float*>(d_saved_rkk),
            reinterpret_cast<float*>(d_r_diag_out));
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        split_and_pad_v_column_d<<<grid, block, 0, stream>>>(
            d_V_col0, reinterpret_cast<const unsigned long long*>(d_row_global),
            l_rows, static_cast<unsigned long long>(pivot_global), pivot_here,
            pivot_loc,
            reinterpret_cast<const double*>(d_saved_rkk),
            reinterpret_cast<double*>(d_r_diag_out));
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        split_and_pad_v_column_c<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_V_col0),
            reinterpret_cast<const unsigned long long*>(d_row_global),
            l_rows, static_cast<unsigned long long>(pivot_global), pivot_here,
            pivot_loc,
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_saved_rkk),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_r_diag_out));
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        split_and_pad_v_column_z<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_V_col0),
            reinterpret_cast<const unsigned long long*>(d_row_global),
            l_rows, static_cast<unsigned long long>(pivot_global), pivot_here,
            pivot_loc,
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_saved_rkk),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_r_diag_out));
    }
    CHASE_GPU_ERROR_T err = CHASE_GPU_PEEK_AT_LAST_ERROR();
    if (err != CHASE_GPU_SUCCESS)
    {
        std::ostringstream oss;
        oss << "split_and_pad_v_column kernel launch failed: "
            << CHASE_GPU_GET_ERROR_STRING(err) << " (l_rows=" << l_rows << ")";
        throw std::runtime_error(oss.str());
    }
}

template <typename T>
void run_fused_householder_finish_kernel(CHASE_GPU_STREAM_T stream, int pivot_here,
                                         T* d_V_col0, int ldv,
                                         const std::uint64_t* d_row_global,
                                         int l_rows,
                                         std::uint64_t pivot_global,
                                         int pivot_loc,
                                         int active_row_start,
                                         const T* d_denom_bcast,
                                         T* d_inv_denom,
                                         const T* d_saved_rkk,
                                         T* d_r_diag_out)
{
    (void)ldv;
    if (stream == nullptr)
        stream = 0;
    if (l_rows <= 0 || d_V_col0 == nullptr || d_row_global == nullptr)
        return;
    constexpr int block = 256;
    const int grid = (l_rows + block - 1) / block;
    if (grid <= 0)
        return;
    if constexpr (std::is_same<T, float>::value)
    {
        fused_householder_finish_s<<<grid, block, 0, stream>>>(
            d_V_col0, l_rows, active_row_start,
            reinterpret_cast<const unsigned long long*>(d_row_global),
            static_cast<unsigned long long>(pivot_global), pivot_here, pivot_loc,
            reinterpret_cast<const float*>(d_denom_bcast),
            reinterpret_cast<float*>(d_inv_denom),
            reinterpret_cast<const float*>(d_saved_rkk),
            reinterpret_cast<float*>(d_r_diag_out));
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        fused_householder_finish_d<<<grid, block, 0, stream>>>(
            d_V_col0, l_rows, active_row_start,
            reinterpret_cast<const unsigned long long*>(d_row_global),
            static_cast<unsigned long long>(pivot_global), pivot_here, pivot_loc,
            reinterpret_cast<const double*>(d_denom_bcast),
            reinterpret_cast<double*>(d_inv_denom),
            reinterpret_cast<const double*>(d_saved_rkk),
            reinterpret_cast<double*>(d_r_diag_out));
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        fused_householder_finish_c<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_V_col0), l_rows, active_row_start,
            reinterpret_cast<const unsigned long long*>(d_row_global),
            static_cast<unsigned long long>(pivot_global), pivot_here, pivot_loc,
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_inv_denom),
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_saved_rkk),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_r_diag_out));
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        fused_householder_finish_z<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_V_col0), l_rows,
            active_row_start,
            reinterpret_cast<const unsigned long long*>(d_row_global),
            static_cast<unsigned long long>(pivot_global), pivot_here, pivot_loc,
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_denom_bcast),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_inv_denom),
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_saved_rkk),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_r_diag_out));
    }
}

template <typename T, typename RealT>
void run_extract_real_part_from_scalar(CHASE_GPU_STREAM_T stream, const T* d_in,
                                       RealT* d_out)
{
    if (stream == nullptr)
        stream = 0;
    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value)
    {
        static_assert(std::is_same<T, RealT>::value, "real type mismatch");
        CHASE_GPU_ERROR_T err = CHASE_GPU_MEMCPY_ASYNC(
            d_out, d_in, sizeof(RealT), CHASE_GPU_MEMCPY_DEVICE_TO_DEVICE, stream);
        if (err != CHASE_GPU_SUCCESS)
            throw std::runtime_error("run_extract_real_part_from_scalar memcpy failed");
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        static_assert(std::is_same<RealT, float>::value, "real type mismatch");
        extract_real_part_from_c<<<1, 1, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_in), d_out);
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value &&
                          std::is_same<RealT, double>::value,
                      "T must be float, double, or complex");
        extract_real_part_from_z<<<1, 1, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_in), d_out);
    }
}

template <typename T>
void run_copy_scalar_kernel(CHASE_GPU_STREAM_T stream, const T* d_src, T* d_dst)
{
    if (stream == nullptr)
        stream = 0;
    if constexpr (std::is_same<T, float>::value)
    {
        copy_scalar_s<<<1, 1, 0, stream>>>(d_src, d_dst);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        copy_scalar_d<<<1, 1, 0, stream>>>(d_src, d_dst);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        copy_scalar_c<<<1, 1, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_src),
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_dst));
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        copy_scalar_z<<<1, 1, 0, stream>>>(
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_src),
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_dst));
    }
}

template <typename T>
void run_panel_pre_clean(CHASE_GPU_STREAM_T stream, T* d_V_panel, std::size_t ldv,
                         const std::uint64_t* d_row_global, int l_rows,
                         std::size_t k_col0, int jb_cols)
{
    if (stream == nullptr)
        stream = 0;
    if (jb_cols <= 0 || l_rows <= 0 || d_V_panel == nullptr ||
        d_row_global == nullptr || ldv > static_cast<std::size_t>(INT_MAX))
        return;
    const int ldv_i = static_cast<int>(ldv);
    constexpr int block = 256;
    const int grid = (l_rows + block - 1) / block;
    if (grid <= 0)
        return;
    (void)CHASE_GPU_GET_LAST_ERROR();
    const unsigned long long k0 = static_cast<unsigned long long>(k_col0);
    if constexpr (std::is_same<T, float>::value)
    {
        panel_pre_clean_s<<<grid, block, 0, stream>>>(
            d_V_panel, ldv_i,
            reinterpret_cast<const unsigned long long*>(d_row_global), l_rows,
            k0, jb_cols);
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        panel_pre_clean_d<<<grid, block, 0, stream>>>(
            d_V_panel, ldv_i,
            reinterpret_cast<const unsigned long long*>(d_row_global), l_rows,
            k0, jb_cols);
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        panel_pre_clean_c<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(d_V_panel), ldv_i,
            reinterpret_cast<const unsigned long long*>(d_row_global), l_rows,
            k0, jb_cols);
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        panel_pre_clean_z<<<grid, block, 0, stream>>>(
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(d_V_panel), ldv_i,
            reinterpret_cast<const unsigned long long*>(d_row_global), l_rows,
            k0, jb_cols);
    }
    CHASE_GPU_ERROR_T err = CHASE_GPU_PEEK_AT_LAST_ERROR();
    if (err != CHASE_GPU_SUCCESS)
    {
        std::ostringstream oss;
        oss << "panel_pre_clean kernel launch failed: "
            << CHASE_GPU_GET_ERROR_STRING(err) << " (l_rows=" << l_rows << ")";
        throw std::runtime_error(oss.str());
    }
}

template <typename T>
void run_compute_T_block(CHASE_GPU_STREAM_T stream, T* Tb, T* d_S, const T* d_tau,
                        int jb, int nb)
{
    if (stream == nullptr)
        stream = 0;
    if (jb <= 0)
        return;
    if (jb > nb)
        throw std::runtime_error("run_compute_T_block: jb > nb");
    (void)CHASE_GPU_GET_LAST_ERROR();
    if constexpr (std::is_same<T, float>::value)
    {
        const int threads = std::max(1, std::min(128, jb));
        const std::size_t shmem_bytes = static_cast<std::size_t>(jb) * sizeof(float);
#if CHASE_PANEL_HIPREC
        compute_T_block_s_dpacc<<<1, threads, shmem_bytes, stream>>>(Tb, d_S, d_tau, jb, nb);
#else
        compute_T_block_s<<<1, threads, shmem_bytes, stream>>>(Tb, d_S, d_tau, jb, nb);
#endif
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        const int threads = std::max(1, std::min(128, jb));
        const std::size_t shmem_bytes = static_cast<std::size_t>(jb) * sizeof(double);
#if CHASE_PANEL_HIPREC
        compute_T_block_d_qdacc<<<1, threads, shmem_bytes, stream>>>(Tb, d_S, d_tau, jb, nb);
#else
        compute_T_block_d<<<1, threads, shmem_bytes, stream>>>(Tb, d_S, d_tau, jb, nb);
#endif
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        const int threads = std::max(1, std::min(128, jb));
        const std::size_t shmem_bytes =
            static_cast<std::size_t>(jb) * sizeof(CHASE_GPU_FLOAT_COMPLEX);
#if CHASE_PANEL_HIPREC
        compute_T_block_c_dpacc<<<1, threads, shmem_bytes, stream>>>(
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(Tb),
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_S),
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_tau),
            jb, nb);
#else
        compute_T_block_c<<<1, threads, shmem_bytes, stream>>>(
            reinterpret_cast<CHASE_GPU_FLOAT_COMPLEX*>(Tb),
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_S),
            reinterpret_cast<const CHASE_GPU_FLOAT_COMPLEX*>(d_tau),
            jb, nb);
#endif
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "T must be float, double, or complex");
        const int threads = std::max(1, std::min(128, jb));
        const std::size_t shmem_bytes =
            static_cast<std::size_t>(jb) * sizeof(CHASE_GPU_DOUBLE_COMPLEX);
#if CHASE_PANEL_HIPREC
        compute_T_block_z_qdacc<<<1, threads, shmem_bytes, stream>>>(
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(Tb),
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_S),
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_tau),
            jb, nb);
#else
        compute_T_block_z<<<1, threads, shmem_bytes, stream>>>(
            reinterpret_cast<CHASE_GPU_DOUBLE_COMPLEX*>(Tb),
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_S),
            reinterpret_cast<const CHASE_GPU_DOUBLE_COMPLEX*>(d_tau),
            jb, nb);
#endif
    }
    CHASE_GPU_ERROR_T err = CHASE_GPU_PEEK_AT_LAST_ERROR();
    if (err != CHASE_GPU_SUCCESS)
    {
        std::ostringstream oss;
        oss << "compute_T_block kernel launch failed: " << CHASE_GPU_GET_ERROR_STRING(err);
        throw std::runtime_error(oss.str());
    }
}

template <typename T>
void run_split_to_hilo(CHASE_GPU_STREAM_T stream, const T* d_in, std::size_t count,
                       double* d_hi, double* d_lo)
{
    if (stream == nullptr)
        stream = 0;
    if (count == 0)
        return;

    const std::size_t scalar_count = split_sync_scalar_count<T>(count);

    constexpr int block = 256;
    const int grid = static_cast<int>((scalar_count + block - 1) / block);

    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value)
    {
        split_to_hilo_kernel<T><<<grid, block, 0, stream>>>(
            d_in, d_hi, d_lo, scalar_count);
        if (CHASE_GPU_PEEK_AT_LAST_ERROR() != CHASE_GPU_SUCCESS)
            throw std::runtime_error("split_to_hilo_kernel launch failed");

        (void)d_hi;
        (void)d_lo;
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        const float* in_flat = reinterpret_cast<const float*>(d_in);

        split_to_hilo_kernel<float><<<grid, block, 0, stream>>>(
            in_flat, d_hi, d_lo, scalar_count);
        if (CHASE_GPU_PEEK_AT_LAST_ERROR() != CHASE_GPU_SUCCESS)
            throw std::runtime_error("split_to_hilo_kernel(complex<float>) failed");
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "Unsupported type for split_to_hilo");
        const double* in_flat = reinterpret_cast<const double*>(d_in);

        split_to_hilo_kernel<double><<<grid, block, 0, stream>>>(
            in_flat, d_hi, d_lo, scalar_count);
        if (CHASE_GPU_PEEK_AT_LAST_ERROR() != CHASE_GPU_SUCCESS)
            throw std::runtime_error("split_to_hilo_kernel(complex<double>) failed");
    }
}

void run_renorm_hilo(CHASE_GPU_STREAM_T stream, double* d_hi, double* d_lo,
                     std::size_t scalar_count)
{
    if (stream == nullptr)
        stream = 0;
    if (scalar_count == 0)
        return;
    constexpr int block = 256;
    const int grid = static_cast<int>((scalar_count + block - 1) / block);
    renorm_hilo_kernel<<<grid, block, 0, stream>>>(d_hi, d_lo, scalar_count);
    if (CHASE_GPU_PEEK_AT_LAST_ERROR() != CHASE_GPU_SUCCESS)
        throw std::runtime_error("renorm_hilo_kernel launch failed");
}

template <typename T>
void run_merge_hilo_to_out(CHASE_GPU_STREAM_T stream, const double* d_hi,
                           const double* d_lo, T* d_out, std::size_t count)
{
    if (stream == nullptr)
        stream = 0;
    if (count == 0)
        return;
    const std::size_t scalar_count = split_sync_scalar_count<T>(count);
    constexpr int block = 256;
    const int grid = static_cast<int>((scalar_count + block - 1) / block);

    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value)
    {
        merge_hilo_to_out_kernel<T><<<grid, block, 0, stream>>>(
            d_hi, d_lo, d_out, scalar_count);
        if (CHASE_GPU_PEEK_AT_LAST_ERROR() != CHASE_GPU_SUCCESS)
            throw std::runtime_error("merge_hilo_to_out_kernel launch failed");
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        float* out_flat = reinterpret_cast<float*>(d_out);
        merge_hilo_to_out_kernel<float><<<grid, block, 0, stream>>>(
            d_hi, d_lo, out_flat, scalar_count);
        if (CHASE_GPU_PEEK_AT_LAST_ERROR() != CHASE_GPU_SUCCESS)
            throw std::runtime_error("merge_hilo_to_out_kernel(complex<float>) failed");
    }
    else
    {
        static_assert(std::is_same<T, std::complex<double>>::value,
                      "Unsupported type for merge_hilo_to_out");
        double* out_flat = reinterpret_cast<double*>(d_out);
        merge_hilo_to_out_kernel<double><<<grid, block, 0, stream>>>(
            d_hi, d_lo, out_flat, scalar_count);
        if (CHASE_GPU_PEEK_AT_LAST_ERROR() != CHASE_GPU_SUCCESS)
            throw std::runtime_error("merge_hilo_to_out_kernel(complex<double>) failed");
    }
}

// Explicit instantiations
template void run_householder_scalar_kernel<float, float>(CHASE_GPU_STREAM_T, int,
    const float*, const float*, float*, float*, float*, float*, float*);
template void run_householder_scalar_kernel<double, double>(CHASE_GPU_STREAM_T, int,
    const double*, const double*, double*, double*, double*, double*, double*);
template void run_householder_scalar_kernel<std::complex<float>, float>(CHASE_GPU_STREAM_T, int,
    const std::complex<float>*, const float*, std::complex<float>*, std::complex<float>*,
    std::complex<float>*, std::complex<float>*, std::complex<float>*);
template void run_householder_scalar_kernel<std::complex<double>, double>(CHASE_GPU_STREAM_T, int,
    const std::complex<double>*, const double*, std::complex<double>*, std::complex<double>*,
    std::complex<double>*, std::complex<double>*, std::complex<double>*);

template void run_inv_denom_from_denom_bcast<float>(CHASE_GPU_STREAM_T, const float*, float*);
template void run_inv_denom_from_denom_bcast<double>(CHASE_GPU_STREAM_T, const double*, double*);
template void run_inv_denom_from_denom_bcast<std::complex<float>>(CHASE_GPU_STREAM_T, const std::complex<float>*, std::complex<float>*);
template void run_inv_denom_from_denom_bcast<std::complex<double>>(CHASE_GPU_STREAM_T, const std::complex<double>*, std::complex<double>*);

template void run_nonpivot_scal_if_denom_nonzero<float>(CHASE_GPU_STREAM_T, const float*, float*, float*, int);
template void run_nonpivot_scal_if_denom_nonzero<double>(CHASE_GPU_STREAM_T, const double*, double*, double*, int);
template void run_nonpivot_scal_if_denom_nonzero<std::complex<float>>(CHASE_GPU_STREAM_T, const std::complex<float>*, std::complex<float>*, std::complex<float>*, int);
template void run_nonpivot_scal_if_denom_nonzero<std::complex<double>>(CHASE_GPU_STREAM_T, const std::complex<double>*, std::complex<double>*, std::complex<double>*, int);

template void run_bc1d_post_comm_scal_pivot<float>(CHASE_GPU_STREAM_T, int, const float*, float*, float*, int, int);
template void run_bc1d_post_comm_scal_pivot<double>(CHASE_GPU_STREAM_T, int, const double*, double*, double*, int, int);
template void run_bc1d_post_comm_scal_pivot<std::complex<float>>(CHASE_GPU_STREAM_T, int, const std::complex<float>*, std::complex<float>*, std::complex<float>*, int, int);
template void run_bc1d_post_comm_scal_pivot<std::complex<double>>(CHASE_GPU_STREAM_T, int, const std::complex<double>*, std::complex<double>*, std::complex<double>*, int, int);

template void run_guarded_scaling<float>(CHASE_GPU_STREAM_T, int, const float*, const float*, float*, bool, int, const float*);
template void run_guarded_scaling<double>(CHASE_GPU_STREAM_T, int, const double*, const double*, double*, bool, int, const double*);
template void run_guarded_scaling<std::complex<float>>(CHASE_GPU_STREAM_T, int, const std::complex<float>*, const std::complex<float>*, std::complex<float>*, bool, int, const std::complex<float>*);
template void run_guarded_scaling<std::complex<double>>(CHASE_GPU_STREAM_T, int, const std::complex<double>*, const std::complex<double>*, std::complex<double>*, bool, int, const std::complex<double>*);

template void run_batch_save_restore_upper_triangular<float>(CHASE_GPU_STREAM_T, float*, float*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, const float*, const float*, bool);
template void run_batch_save_restore_upper_triangular<double>(CHASE_GPU_STREAM_T, double*, double*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, const double*, const double*, bool);
template void run_batch_save_restore_upper_triangular<std::complex<float>>(CHASE_GPU_STREAM_T, std::complex<float>*, std::complex<float>*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, const std::complex<float>*, const std::complex<float>*, bool);
template void run_batch_save_restore_upper_triangular<std::complex<double>>(CHASE_GPU_STREAM_T, std::complex<double>*, std::complex<double>*, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, const std::complex<double>*, const std::complex<double>*, bool);

template void run_init_identity_distributed<float>(CHASE_GPU_STREAM_T, float*, std::size_t, std::size_t, std::size_t, std::size_t);
template void run_init_identity_distributed<double>(CHASE_GPU_STREAM_T, double*, std::size_t, std::size_t, std::size_t, std::size_t);
template void run_init_identity_distributed<std::complex<float>>(CHASE_GPU_STREAM_T, std::complex<float>*, std::size_t, std::size_t, std::size_t, std::size_t);
template void run_init_identity_distributed<std::complex<double>>(CHASE_GPU_STREAM_T, std::complex<double>*, std::size_t, std::size_t, std::size_t, std::size_t);

template void run_split_and_pad_v_column<float>(CHASE_GPU_STREAM_T, float*, const std::uint64_t*, int,
    std::uint64_t, int, int, const float*, float*);
template void run_split_and_pad_v_column<double>(CHASE_GPU_STREAM_T, double*, const std::uint64_t*, int,
    std::uint64_t, int, int, const double*, double*);
template void run_split_and_pad_v_column<std::complex<float>>(CHASE_GPU_STREAM_T, std::complex<float>*,
    const std::uint64_t*, int, std::uint64_t, int, int, const std::complex<float>*, std::complex<float>*);
template void run_split_and_pad_v_column<std::complex<double>>(CHASE_GPU_STREAM_T, std::complex<double>*,
    const std::uint64_t*, int, std::uint64_t, int, int, const std::complex<double>*, std::complex<double>*);
template void run_fused_householder_finish_kernel<float>(CHASE_GPU_STREAM_T, int, float*, int,
    const std::uint64_t*, int, std::uint64_t, int, int, const float*, float*, const float*, float*);
template void run_fused_householder_finish_kernel<double>(CHASE_GPU_STREAM_T, int, double*, int,
    const std::uint64_t*, int, std::uint64_t, int, int, const double*, double*, const double*, double*);
template void run_fused_householder_finish_kernel<std::complex<float>>(CHASE_GPU_STREAM_T, int, std::complex<float>*, int,
    const std::uint64_t*, int, std::uint64_t, int, int, const std::complex<float>*, std::complex<float>*,
    const std::complex<float>*, std::complex<float>*);
template void run_fused_householder_finish_kernel<std::complex<double>>(CHASE_GPU_STREAM_T, int, std::complex<double>*, int,
    const std::uint64_t*, int, std::uint64_t, int, int, const std::complex<double>*, std::complex<double>*,
    const std::complex<double>*, std::complex<double>*);
template void run_extract_real_part_from_scalar<float, float>(CHASE_GPU_STREAM_T, const float*, float*);
template void run_extract_real_part_from_scalar<double, double>(CHASE_GPU_STREAM_T, const double*, double*);
template void run_extract_real_part_from_scalar<std::complex<float>, float>(CHASE_GPU_STREAM_T, const std::complex<float>*, float*);
template void run_extract_real_part_from_scalar<std::complex<double>, double>(CHASE_GPU_STREAM_T, const std::complex<double>*, double*);
template void run_copy_scalar_kernel<float>(CHASE_GPU_STREAM_T, const float*, float*);
template void run_copy_scalar_kernel<double>(CHASE_GPU_STREAM_T, const double*, double*);
template void run_copy_scalar_kernel<std::complex<float>>(CHASE_GPU_STREAM_T, const std::complex<float>*, std::complex<float>*);
template void run_copy_scalar_kernel<std::complex<double>>(CHASE_GPU_STREAM_T, const std::complex<double>*, std::complex<double>*);

template void run_panel_pre_clean<float>(CHASE_GPU_STREAM_T, float*, std::size_t,
    const std::uint64_t*, int, std::size_t, int);
template void run_panel_pre_clean<double>(CHASE_GPU_STREAM_T, double*, std::size_t,
    const std::uint64_t*, int, std::size_t, int);
template void run_panel_pre_clean<std::complex<float>>(CHASE_GPU_STREAM_T, std::complex<float>*, std::size_t,
    const std::uint64_t*, int, std::size_t, int);
template void run_panel_pre_clean<std::complex<double>>(CHASE_GPU_STREAM_T, std::complex<double>*, std::size_t,
    const std::uint64_t*, int, std::size_t, int);

template void run_compute_T_block<float>(CHASE_GPU_STREAM_T, float*, float*, const float*, int, int);
template void run_compute_T_block<double>(CHASE_GPU_STREAM_T, double*, double*, const double*, int, int);
template void run_compute_T_block<std::complex<float>>(CHASE_GPU_STREAM_T, std::complex<float>*, std::complex<float>*, const std::complex<float>*, int, int);
template void run_compute_T_block<std::complex<double>>(CHASE_GPU_STREAM_T, std::complex<double>*, std::complex<double>*, const std::complex<double>*, int, int);

template void run_split_to_hilo<float>(CHASE_GPU_STREAM_T, const float*, std::size_t, double*, double*);
template void run_split_to_hilo<double>(CHASE_GPU_STREAM_T, const double*, std::size_t, double*, double*);
template void run_split_to_hilo<std::complex<float>>(CHASE_GPU_STREAM_T, const std::complex<float>*, std::size_t, double*, double*);
template void run_split_to_hilo<std::complex<double>>(CHASE_GPU_STREAM_T, const std::complex<double>*, std::size_t, double*, double*);

template void run_merge_hilo_to_out<float>(CHASE_GPU_STREAM_T, const double*, const double*, float*, std::size_t);
template void run_merge_hilo_to_out<double>(CHASE_GPU_STREAM_T, const double*, const double*, double*, std::size_t);
template void run_merge_hilo_to_out<std::complex<float>>(CHASE_GPU_STREAM_T, const double*, const double*, std::complex<float>*, std::size_t);
template void run_merge_hilo_to_out<std::complex<double>>(CHASE_GPU_STREAM_T, const double*, const double*, std::complex<double>*, std::size_t);

} // namespace CHASE_GPU_DEVICE_NAMESPACE
} // namespace internal
} // namespace linalg
} // namespace chase
