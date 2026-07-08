// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/flip_sign.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class FlipLowerHalfSignHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(FlipLowerHalfSignHIPTest, TestTypes);

} // namespace

TYPED_TEST(FlipLowerHalfSignHIPTest, FlipsLowerHalfSignInColumnMajorMatrix)
{
    using T = TypeParam;

    constexpr std::size_t rows = 4;
    constexpr std::size_t cols = 3;
    constexpr std::size_t ld = 4;

    std::vector<T> host(ld * cols);
    for (std::size_t i = 0; i < host.size(); ++i)
    {
        host[i] = T(i + 1);
    }

    T* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, host.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::flipLowerHalfMatrixSign(device, rows, cols,
                                                          ld);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(host.data(), device, host.size() * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(device));

    const std::vector<T> expected = {T(1),  T(2),  T(-3), T(-4), T(5),  T(6),
                                     T(-7), T(-8), T(9),  T(10), T(-11), T(-12)};
    EXPECT_EQ(host, expected);
}
