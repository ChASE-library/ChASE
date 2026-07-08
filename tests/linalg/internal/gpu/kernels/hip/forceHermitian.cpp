// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/force_hermitian.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class ForceHermitianHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ForceHermitianHIPTest, TestTypes);

} // namespace

TYPED_TEST(ForceHermitianHIPTest, SymmetrizesMatrix)
{
    using T = TypeParam;

    constexpr std::size_t n = 2;
    constexpr std::size_t ld = 2;

    std::vector<T> host;
    std::vector<T> expected;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        // Column-major [1, 5; 3, 2] -> off-diagonal average is 4.
        host = {T(1), T(5), T(3), T(2)};
        expected = {T(1), T(4), T(4), T(2)};
    }
    else
    {
        // Column-major: (0,1)=3 and (1,0)=2 -> Hermitian average is (2.5, 0).
        host = {T(1, 0), T(2, 0), T(3, 0), T(4, 0)};
        expected = {T(1, 0), T(2.5, 0), T(2.5, 0), T(4, 0)};
    }

    T* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, host.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::forceHermitian(device, n, ld);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(host.data(), device, host.size() * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(device));

    for (std::size_t i = 0; i < host.size(); ++i)
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            EXPECT_EQ(host[i], expected[i]);
        }
        else
        {
            EXPECT_NEAR(host[i].real(), expected[i].real(), 1e-6);
            EXPECT_NEAR(host[i].imag(), expected[i].imag(), 1e-6);
        }
    }
}
