// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/pseudo_hermitian_lanczos_diag.hpp"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <limits>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class PseudoHermitianLanczosDiagHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHermitianLanczosDiagHIPTest, TestTypes);

} // namespace

TYPED_TEST(PseudoHermitianLanczosDiagHIPTest, FrobeniusNormOfBBlock)
{
    using T = TypeParam;

    constexpr std::size_t ld = 2;
    constexpr std::size_t lrows = 2;
    constexpr std::size_t lcols = 2;
    constexpr std::size_t g_row_off = 0;
    constexpr std::size_t g_col_off = 2;
    constexpr std::size_t half_m = 2;
    constexpr std::size_t N = 4;

    std::vector<T> host = {T(1), T(2), T(3), T(4)};
    T* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, host.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    double sum_sq = 0.0;
    if constexpr (std::is_same_v<T, float>)
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_B_fro_sq_float(
            reinterpret_cast<const float*>(device), ld, lrows, lcols, g_row_off,
            g_col_off, half_m, N, nullptr, &sum_sq);
        EXPECT_NEAR(sum_sq, 30.0, 1e-6);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_B_fro_sq_double(
            reinterpret_cast<const double*>(device), ld, lrows, lcols, g_row_off,
            g_col_off, half_m, N, nullptr, &sum_sq);
        EXPECT_NEAR(sum_sq, 30.0, 1e-6);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_B_fro_sq_complex_float(
            reinterpret_cast<const std::complex<float>*>(device), ld, lrows,
            lcols, g_row_off, g_col_off, half_m, N, nullptr, &sum_sq);
        EXPECT_NEAR(sum_sq, 30.0, 1e-6);
    }
    else
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_B_fro_sq_complex_double(
            reinterpret_cast<const std::complex<double>*>(device), ld, lrows,
            lcols, g_row_off, g_col_off, half_m, N, nullptr, &sum_sq);
        EXPECT_NEAR(sum_sq, 30.0, 1e-6);
    }

    ASSERT_HIP_SUCCESS(hipFree(device));
}

TYPED_TEST(PseudoHermitianLanczosDiagHIPTest, MinAbsDiagonal)
{
    using T = TypeParam;

    constexpr std::size_t ld = 4;
    constexpr std::size_t lrows = 4;
    constexpr std::size_t lcols = 4;
    constexpr std::size_t N = 4;

    std::vector<T> host(ld * lcols, T(0));
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        host[0] = T(2);
        host[5] = T(5);
        host[10] = T(-3);
        host[15] = T(7);
    }
    else
    {
        host[0] = T(2, 0);
        host[5] = T(5, 0);
        host[10] = T(-3, 0);
        host[15] = T(7, 0);
    }

    T* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, host.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    double min_abs = 0.0;
    if constexpr (std::is_same_v<T, float>)
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_min_abs_diag_float(
            reinterpret_cast<const float*>(device), ld, lrows, lcols, 0, 0, N,
            nullptr, &min_abs);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_min_abs_diag_double(
            reinterpret_cast<const double*>(device), ld, lrows, lcols, 0, 0, N,
            nullptr, &min_abs);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_min_abs_diag_complex_float(
            reinterpret_cast<const std::complex<float>*>(device), ld, lrows,
            lcols, 0, 0, N, nullptr, &min_abs);
    }
    else
    {
        chase::linalg::internal::hip_ph_diag::chase_ph_diag_min_abs_diag_complex_double(
            reinterpret_cast<const std::complex<double>*>(device), ld, lrows,
            lcols, 0, 0, N, nullptr, &min_abs);
    }

    EXPECT_NEAR(min_abs, 2.0, 1e-6);
    ASSERT_HIP_SUCCESS(hipFree(device));
}
