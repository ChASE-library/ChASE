// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/backend/handle_utils.hpp"
#include "linalg/internal/gpu/backend/runtime.hpp"
#include "linalg/internal/gpu/kernels/shared/device_kernels.hpp"

#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <random>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

constexpr std::size_t k_state_count = 256 * 32;
constexpr unsigned long long k_seed = 24141;

template <typename T>
class RandomNormalHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(RandomNormalHIPTest, TestTypes);

template <typename T>
bool is_finite_value(const T& value)
{
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
    {
        return std::isfinite(value);
    }
    else
    {
        return std::isfinite(value.real()) && std::isfinite(value.imag());
    }
}

} // namespace

TYPED_TEST(RandomNormalHIPTest, ReproducibleAndFinite)
{
    using T = TypeParam;

    constexpr std::size_t n = 1024;

    hiprandStatePhilox4_32_10_t* states = nullptr;
    T* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&states, k_state_count *
                                             sizeof(hiprandStatePhilox4_32_10_t)));
    ASSERT_HIP_SUCCESS(hipMalloc(&device, n * sizeof(T)));

    chase::linalg::internal::hip::chase_rand_normal(k_seed, states, device, n);
    std::vector<T> first(n);
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(first.data(), device, n * sizeof(T),
                                 hipMemcpyDeviceToHost));

    chase::linalg::internal::hip::chase_rand_normal(k_seed, states, device, n);
    std::vector<T> second(n);
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(second.data(), device, n * sizeof(T),
                                 hipMemcpyDeviceToHost));

    ASSERT_HIP_SUCCESS(hipFree(device));
    ASSERT_HIP_SUCCESS(hipFree(states));

    for (std::size_t i = 0; i < n; ++i)
    {
        ASSERT_TRUE(is_finite_value(first[i]));
        ASSERT_TRUE(is_finite_value(second[i]));
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            EXPECT_EQ(first[i], second[i]);
        }
        else
        {
            EXPECT_EQ(first[i].real(), second[i].real());
            EXPECT_EQ(first[i].imag(), second[i].imag());
            EXPECT_EQ(first[i].real(), first[i].imag());
        }
    }
}

TYPED_TEST(RandomNormalHIPTest, SharedDispatchFillRandom)
{
    using T = TypeParam;
    using util = chase::linalg::internal::gpu_backend::handle_utils<chase::linalg::internal::gpu_backend::hip_backend>;
    using kernels = chase::linalg::internal::gpu_device::kernels<
        chase::linalg::internal::gpu_backend::hip_backend>;

    constexpr std::size_t n = 512;
    T* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, n * sizeof(T)));

    auto* states = kernels::allocate_states();
    kernels::fill_random(k_seed, states, device, n);
    chase::linalg::internal::gpu_backend::device_free<
        chase::linalg::internal::gpu_backend::hip_backend>(states);

    std::vector<T> values(n);
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(values.data(), device, n * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(device));

    for (const T& value : values)
    {
        ASSERT_TRUE(is_finite_value(value));
    }
}
