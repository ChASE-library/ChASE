// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#include "linalg/distMatrix/distMatrix.hpp"
#include "mpi.h"
#include "linalg/internal/cuda/pseudo_hermitian_lanczos_diag.cuh"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace chase
{
namespace linalg
{
namespace internal
{
namespace detail_nccl_ph_diag
{

/** Off by default. Set \c CHASE_PH_LANCZOS_DIAG=1/true/on/yes to enable (rank 0
 *  prints). */
inline bool ph_lanczos_diag_enabled()
{
    const char* e = std::getenv("CHASE_PH_LANCZOS_DIAG");
    if (e == nullptr || *e == '\0')
        return false;
    while (*e != '\0' && std::isspace(static_cast<unsigned char>(*e)))
        ++e;
    if (*e == '\0')
        return false;
    std::string v(e);
    while (!v.empty() &&
           std::isspace(static_cast<unsigned char>(v.back())))
        v.pop_back();
    for (char& c : v)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return (v == "1" || v == "true" || v == "on" || v == "yes");
}

template <typename T>
void report_pseudo_hermitian_coupling_diag_nccl(
    chase::distMatrix::PseudoHermitianBlockBlockMatrix<T,
                                                      chase::platform::GPU>& H,
    const chase::Base<T>* ritzv, std::size_t M)
{
    if (!ph_lanczos_diag_enabled())
        return;

    const std::size_t N = H.g_rows();
    if (N % 2 != 0 || M == 0)
        return;

    const std::size_t half_m = N / 2;
    std::size_t* goff = H.g_offs();
    double local_sq = 0.0;
    double local_min_abs_diag = 0.0;
    cudaStream_t stream = nullptr;

    if constexpr (std::is_same_v<T, float>)
    {
        cuda_nccl_ph_diag::chase_ph_diag_B_fro_sq_float(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1],
            half_m, N, stream, &local_sq);
        cuda_nccl_ph_diag::chase_ph_diag_min_abs_diag_float(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1], N,
            stream, &local_min_abs_diag);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        cuda_nccl_ph_diag::chase_ph_diag_B_fro_sq_double(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1],
            half_m, N, stream, &local_sq);
        cuda_nccl_ph_diag::chase_ph_diag_min_abs_diag_double(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1], N,
            stream, &local_min_abs_diag);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        cuda_nccl_ph_diag::chase_ph_diag_B_fro_sq_complex_float(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1],
            half_m, N, stream, &local_sq);
        cuda_nccl_ph_diag::chase_ph_diag_min_abs_diag_complex_float(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1], N,
            stream, &local_min_abs_diag);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        cuda_nccl_ph_diag::chase_ph_diag_B_fro_sq_complex_double(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1],
            half_m, N, stream, &local_sq);
        cuda_nccl_ph_diag::chase_ph_diag_min_abs_diag_complex_double(
            H.l_data(), H.l_ld(), H.l_rows(), H.l_cols(), goff[0], goff[1], N,
            stream, &local_min_abs_diag);
    }
    else
    {
        return;
    }

    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM,
                  H.getMpiGrid()->get_comm());

    double global_min_abs_diag = 0.0;
    MPI_Allreduce(&local_min_abs_diag, &global_min_abs_diag, 1, MPI_DOUBLE,
                  MPI_MIN, H.getMpiGrid()->get_comm());

    const double normF_B = std::sqrt(global_sq);

    using R = chase::Base<T>;
    const R* it = std::min_element(
        ritzv, ritzv + M,
        [](R a, R b) {
            return std::abs(static_cast<double>(a)) <
                   std::abs(static_cast<double>(b));
        });
    const R lambda1 = *it;

    int rank = 0;
    MPI_Comm_rank(H.getMpiGrid()->get_comm(), &rank);
    if (rank != 0)
        return;

    std::cerr << std::scientific << std::setprecision(8);
    const double lam_abs = std::abs(static_cast<double>(lambda1));
    const double denom_ritz =
        2.0 * lam_abs * std::sqrt(static_cast<double>(half_m));
    const double gamma_ub_approx =
        (denom_ritz > 0.0) ? (normF_B / denom_ritz)
                           : std::numeric_limits<double>::infinity();

    const double sqrt_m = std::sqrt(static_cast<double>(half_m));
    const double denom_diag = 2.0 * global_min_abs_diag * sqrt_m;
    const bool diag_ok =
        std::isfinite(global_min_abs_diag) && global_min_abs_diag > 0.0;
    const double gamma_diag_approx =
        diag_ok && denom_diag > 0.0 ? (normF_B / denom_diag)
                                    : std::numeric_limits<double>::infinity();

    std::cerr << "[CHASE_PH_LANCZOS_DIAG] m=" << half_m << " ||B||_F=" << normF_B
              << " lambda_1(argmin |Ritz|)=" << static_cast<double>(lambda1)
              << " gamma_ub_approx(||B||_F/(2*|lambda_1|*sqrt(m)))="
              << gamma_ub_approx << " min_abs_diag(A)=" << global_min_abs_diag
              << " gamma_diag(||B||_F/(2*min_abs_diag(A)*sqrt(m)))="
              << gamma_diag_approx << '\n';
    std::cerr << std::defaultfloat;
}

template <typename MatrixType>
void maybe_nccl_ph_lanczos_coupling_report(
    MatrixType& H,
    const chase::Base<typename MatrixType::value_type>* ritzv, std::size_t M)
{
    using T = typename MatrixType::value_type;
    if constexpr (std::is_same_v<
                      MatrixType,
                      chase::distMatrix::PseudoHermitianBlockBlockMatrix<
                          T, chase::platform::GPU>>)
    {
        report_pseudo_hermitian_coupling_diag_nccl(H, ritzv, M);
    }
    else
    {
        (void)H;
        (void)ritzv;
        (void)M;
    }
}

} // namespace detail_nccl_ph_diag

} // namespace internal
} // namespace linalg
} // namespace chase
