// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/conjugate.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class ConjugateHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ConjugateHIPTest, TestTypes);

} // namespace

TYPED_TEST(ConjugateHIPTest, ConjugateInPlace)
{
    using T = TypeParam;

    constexpr std::size_t rows = 2;
    constexpr std::size_t cols = 2;
    constexpr std::size_t ld = 2;

    std::vector<T> host = {T(1), T(2), T(3), T(4)};
    if constexpr (std::is_same_v<T, std::complex<float>> ||
                  std::is_same_v<T, std::complex<double>>)
    {
        host = {T(1, 2), T(3, 4), T(5, -6), T(7, 8)};
    }

    const std::vector<T> input = host;

    T* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, host.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(device, host.data(), host.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::conjugateInPlace(device, rows, cols, ld);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(host.data(), device, host.size() * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(device));

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        EXPECT_EQ(host, input);
    }
    else
    {
        for (std::size_t i = 0; i < host.size(); ++i)
        {
            EXPECT_EQ(host[i].real(), input[i].real());
            EXPECT_EQ(host[i].imag(), -input[i].imag());
        }
    }
}
