// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/householder_scalar_kernel.hpp"

#include "algorithm/types.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class HouseholderScalarKernelHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(HouseholderScalarKernelHIPTest, TestTypes);

} // namespace

TYPED_TEST(HouseholderScalarKernelHIPTest, HouseholderScalars)
{
    using T = TypeParam;
    using RealT = chase::Base<T>;

    T x0_host = T(3);
    RealT nrm_sq_host = RealT(25);
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>)
    {
        x0_host = T(3, 0);
    }

    T tau_host{};
    T inv_denom_host{};
    T neg_beta_host{};
    T denom_bcast_host{};
    T saved_rkk_host{};

    T* d_x0 = nullptr;
    RealT* d_nrm_sq = nullptr;
    T* d_tau = nullptr;
    T* d_inv_denom = nullptr;
    T* d_neg_beta = nullptr;
    T* d_denom_bcast = nullptr;
    T* d_saved_rkk = nullptr;

    ASSERT_HIP_SUCCESS(hipMalloc(&d_x0, sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_nrm_sq, sizeof(RealT)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_tau, sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_inv_denom, sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_neg_beta, sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_denom_bcast, sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_saved_rkk, sizeof(T)));

    ASSERT_HIP_SUCCESS(hipMemcpy(d_x0, &x0_host, sizeof(T), hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_nrm_sq, &nrm_sq_host, sizeof(RealT),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::run_householder_scalar_kernel<T, RealT>(
        nullptr, 1, d_x0, d_nrm_sq, d_tau, d_inv_denom, d_neg_beta,
        d_denom_bcast, d_saved_rkk);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(&tau_host, d_tau, sizeof(T), hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(
        hipMemcpy(&inv_denom_host, d_inv_denom, sizeof(T), hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(
        hipMemcpy(&neg_beta_host, d_neg_beta, sizeof(T), hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipMemcpy(&denom_bcast_host, d_denom_bcast, sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipMemcpy(&saved_rkk_host, d_saved_rkk, sizeof(T),
                                 hipMemcpyDeviceToHost));

    ASSERT_HIP_SUCCESS(hipFree(d_x0));
    ASSERT_HIP_SUCCESS(hipFree(d_nrm_sq));
    ASSERT_HIP_SUCCESS(hipFree(d_tau));
    ASSERT_HIP_SUCCESS(hipFree(d_inv_denom));
    ASSERT_HIP_SUCCESS(hipFree(d_neg_beta));
    ASSERT_HIP_SUCCESS(hipFree(d_denom_bcast));
    ASSERT_HIP_SUCCESS(hipFree(d_saved_rkk));

    if constexpr (std::is_same_v<T, float>)
    {
        EXPECT_FLOAT_EQ(tau_host, 1.6f);
        EXPECT_FLOAT_EQ(inv_denom_host, 0.125f);
        EXPECT_FLOAT_EQ(neg_beta_host, -5.0f);
        EXPECT_FLOAT_EQ(denom_bcast_host, 8.0f);
        EXPECT_FLOAT_EQ(saved_rkk_host, 3.0f);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        EXPECT_DOUBLE_EQ(tau_host, 1.6);
        EXPECT_DOUBLE_EQ(inv_denom_host, 0.125);
        EXPECT_DOUBLE_EQ(neg_beta_host, -5.0);
        EXPECT_DOUBLE_EQ(denom_bcast_host, 8.0);
        EXPECT_DOUBLE_EQ(saved_rkk_host, 3.0);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        EXPECT_FLOAT_EQ(tau_host.real(), 1.6f);
        EXPECT_FLOAT_EQ(inv_denom_host.real(), 0.125f);
        EXPECT_FLOAT_EQ(neg_beta_host.real(), -5.0f);
        EXPECT_FLOAT_EQ(denom_bcast_host.real(), 8.0f);
        EXPECT_FLOAT_EQ(saved_rkk_host.real(), 3.0f);
    }
    else
    {
        EXPECT_DOUBLE_EQ(tau_host.real(), 1.6);
        EXPECT_DOUBLE_EQ(inv_denom_host.real(), 0.125);
        EXPECT_DOUBLE_EQ(neg_beta_host.real(), -5.0);
        EXPECT_DOUBLE_EQ(denom_bcast_host.real(), 8.0);
        EXPECT_DOUBLE_EQ(saved_rkk_host.real(), 3.0);
    }
}

TYPED_TEST(HouseholderScalarKernelHIPTest, CopyScalarKernel)
{
    using T = TypeParam;

    T src_host = T(7);
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>)
    {
        src_host = T(7, -2);
    }

    T* d_src = nullptr;
    T* d_dst = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_src, sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_dst, sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_src, &src_host, sizeof(T), hipMemcpyHostToDevice));

    chase::linalg::internal::hip::run_copy_scalar_kernel<T>(nullptr, d_src, d_dst);

    T dst_host{};
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(&dst_host, d_dst, sizeof(T), hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_src));
    ASSERT_HIP_SUCCESS(hipFree(d_dst));

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        EXPECT_EQ(dst_host, src_host);
    }
    else
    {
        EXPECT_EQ(dst_host.real(), src_host.real());
        EXPECT_EQ(dst_host.imag(), src_host.imag());
    }
}
