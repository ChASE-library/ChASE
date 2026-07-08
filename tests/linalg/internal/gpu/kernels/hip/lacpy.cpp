// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/lacpy.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class LacpyHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LacpyHIPTest, TestTypes);

} // namespace

TYPED_TEST(LacpyHIPTest, FullCopy)
{
    using T = TypeParam;

    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 3;
    constexpr std::size_t ld = 4;

    std::vector<T> src(ld * cols);
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        src[i] = T(i + 1);
    }

    std::vector<T> dst(ld * cols, T(0));

    T *d_src = nullptr, *d_dst = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_src, src.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_dst, dst.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_src, src.data(), src.size() * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_dst, dst.data(), dst.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::t_lacpy_gpu('A', rows, cols, d_src, ld,
                                              d_dst, ld);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(dst.data(), d_dst, dst.size() * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_src));
    ASSERT_HIP_SUCCESS(hipFree(d_dst));

    for (std::size_t j = 0; j < cols; ++j)
    {
        for (std::size_t i = 0; i < rows; ++i)
        {
            EXPECT_EQ(dst[j * ld + i], src[j * ld + i]);
        }
    }
}

TYPED_TEST(LacpyHIPTest, UpperCopy)
{
    using T = TypeParam;

    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 3;
    constexpr std::size_t ld = rows;

    std::vector<T> src(rows * cols);
    std::vector<T> dst(rows * cols);
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        src[i] = T(i + 1);
        dst[i] = T(-(i + 1));
    }

    T *d_src = nullptr, *d_dst = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_src, src.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_dst, dst.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_src, src.data(), src.size() * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_dst, dst.data(), dst.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::t_lacpy_gpu('U', rows, cols, d_src, ld,
                                              d_dst, ld);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(dst.data(), d_dst, dst.size() * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_src));
    ASSERT_HIP_SUCCESS(hipFree(d_dst));

    for (std::size_t j = 0; j < cols; ++j)
    {
        for (std::size_t i = 0; i < rows; ++i)
        {
            if (j >= i)
            {
                EXPECT_EQ(dst[j * ld + i], src[j * ld + i]);
            }
            else
            {
                EXPECT_EQ(dst[j * ld + i], T(0));
            }
        }
    }
}

TYPED_TEST(LacpyHIPTest, LowerCopy)
{
    using T = TypeParam;

    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 3;
    constexpr std::size_t ld = rows;

    std::vector<T> src(rows * cols);
    std::vector<T> dst(rows * cols);
    for (std::size_t i = 0; i < src.size(); ++i)
    {
        src[i] = T(i + 1);
        dst[i] = T(-(i + 1));
    }

    T *d_src = nullptr, *d_dst = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_src, src.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_dst, dst.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_src, src.data(), src.size() * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_dst, dst.data(), dst.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::t_lacpy_gpu('L', rows, cols, d_src, ld,
                                              d_dst, ld);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(dst.data(), d_dst, dst.size() * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_src));
    ASSERT_HIP_SUCCESS(hipFree(d_dst));

    for (std::size_t j = 0; j < cols; ++j)
    {
        for (std::size_t i = 0; i < rows; ++i)
        {
            if (j <= i)
            {
                EXPECT_EQ(dst[j * ld + i], src[j * ld + i]);
            }
            else
            {
                EXPECT_EQ(dst[j * ld + i], T(0));
            }
        }
    }
}
