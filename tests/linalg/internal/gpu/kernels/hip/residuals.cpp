// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/residuals.hpp"

#include "algorithm/types.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class ResidualsHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ResidualsHIPTest, TestTypes);

} // namespace

TYPED_TEST(ResidualsHIPTest, ExactEigenpairResidualIsZero)
{
    using T = TypeParam;
    using Scalar = chase::Base<T>;

    constexpr std::size_t m = 4;
    constexpr std::size_t n = 1;
    constexpr std::size_t ld = 4;
    constexpr Scalar lambda = Scalar(2);

    std::vector<T> a_host(m, T(0));
    std::vector<T> b_host(m, T(0));
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        b_host[0] = T(1);
        a_host[0] = T(lambda);
    }
    else
    {
        b_host[0] = T(1, 0);
        a_host[0] = T(lambda, 0);
    }

    T *d_a = nullptr, *d_b = nullptr;
    Scalar *d_ritzv = nullptr, *d_resids = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_a, m * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_b, m * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_ritzv, n * sizeof(Scalar)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_resids, n * sizeof(Scalar)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_a, a_host.data(), m * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_b, b_host.data(), m * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_ritzv, &lambda, sizeof(Scalar),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::residual_gpu(m, n, d_a, ld, d_b, ld, d_ritzv,
                                               d_resids, true);

    Scalar residual = Scalar(-1);
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(&residual, d_resids, sizeof(Scalar),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_a));
    ASSERT_HIP_SUCCESS(hipFree(d_b));
    ASSERT_HIP_SUCCESS(hipFree(d_ritzv));
    ASSERT_HIP_SUCCESS(hipFree(d_resids));

    EXPECT_NEAR(static_cast<double>(residual), 0.0, 1e-6);
}
