// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/orthog_launchers.hpp"

#include <hip/hip_complex.h>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

} // namespace

TEST(LanczosKernelsHIPTest, BatchedSqrtLauncher)
{
    std::vector<double> data = {4.0, 9.0, 16.0};
    double* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, data.size() * sizeof(double)));
    ASSERT_HIP_SUCCESS(hipMemcpy(device, data.data(), data.size() * sizeof(double),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::batchedSqrt(device, static_cast<int>(data.size()));

    std::vector<double> out(data.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(out.data(), device, out.size() * sizeof(double),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(device));

    EXPECT_DOUBLE_EQ(out[0], 2.0);
    EXPECT_DOUBLE_EQ(out[1], 3.0);
    EXPECT_DOUBLE_EQ(out[2], 4.0);
}

TEST(LanczosKernelsHIPTest, BatchedSqrt)
{
    std::vector<double> data = {4.0, 9.0, 16.0};
    double* device = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&device, data.size() * sizeof(double)));
    ASSERT_HIP_SUCCESS(hipMemcpy(device, data.data(), data.size() * sizeof(double),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::batched_sqrt_gpu(device, static_cast<int>(data.size()));

    std::vector<double> out(data.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(out.data(), device, out.size() * sizeof(double),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(device));

    EXPECT_DOUBLE_EQ(out[0], 2.0);
    EXPECT_DOUBLE_EQ(out[1], 3.0);
    EXPECT_DOUBLE_EQ(out[2], 4.0);
}

template <typename T>
class LanczosBatchedScaleHIPTest : public ::testing::Test
{
};

using ScaleTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosBatchedScaleHIPTest, ScaleTypes);

TYPED_TEST(LanczosBatchedScaleHIPTest, ScalesColumns)
{
    using T = TypeParam;

    constexpr int rows = 2;
    constexpr int numvec = 2;
    constexpr int ld = 2;

    std::vector<T> v = {T(1), T(3), T(2), T(4)};
    std::vector<T> scale = {T(2), T(3)};
    std::vector<T> expected = {T(2), T(6), T(6), T(12)};

    T *d_v = nullptr, *d_scale = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_v, v.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_scale, scale.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_v, v.data(), v.size() * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_scale, scale.data(), scale.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        chase::linalg::internal::hip::batched_scale_gpu(
            reinterpret_cast<const hipFloatComplex*>(d_scale),
            reinterpret_cast<hipFloatComplex*>(d_v), rows, numvec, ld);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        chase::linalg::internal::hip::batched_scale_gpu(
            reinterpret_cast<const hipDoubleComplex*>(d_scale),
            reinterpret_cast<hipDoubleComplex*>(d_v), rows, numvec, ld);
    }
    else
    {
        chase::linalg::internal::hip::batched_scale_gpu(d_scale, d_v, rows, numvec,
                                                         ld);
    }

    std::vector<T> out(v.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(out.data(), d_v, out.size() * sizeof(T),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_v));
    ASSERT_HIP_SUCCESS(hipFree(d_scale));

    for (std::size_t i = 0; i < out.size(); ++i)
    {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>)
        {
            EXPECT_EQ(out[i], expected[i]);
        }
        else
        {
            EXPECT_EQ(out[i].real(), expected[i].real());
            EXPECT_EQ(out[i].imag(), expected[i].imag());
        }
    }
}

template <typename T>
class LanczosBatchedDotHIPTest : public ::testing::Test
{
};

using DotTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosBatchedDotHIPTest, DotTypes);

TYPED_TEST(LanczosBatchedDotHIPTest, DotProductPerColumn)
{
    using T = TypeParam;

    constexpr int rows = 2;
    constexpr int numvec = 1;
    constexpr int ld = 2;

    std::vector<T> v1 = {T(1), T(2)};
    std::vector<T> v2 = {T(3), T(4)};

    T *d_v1 = nullptr, *d_v2 = nullptr, *d_results = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_v1, v1.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_v2, v2.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_results, numvec * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_v1, v1.data(), v1.size() * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_v2, v2.data(), v2.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        chase::linalg::internal::hip::batched_dot_product_gpu(
            reinterpret_cast<const hipFloatComplex*>(d_v1),
            reinterpret_cast<const hipFloatComplex*>(d_v2),
            reinterpret_cast<hipFloatComplex*>(d_results), rows, numvec, ld, false);
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>)
    {
        chase::linalg::internal::hip::batched_dot_product_gpu(
            reinterpret_cast<const hipDoubleComplex*>(d_v1),
            reinterpret_cast<const hipDoubleComplex*>(d_v2),
            reinterpret_cast<hipDoubleComplex*>(d_results), rows, numvec, ld, false);
    }
    else
    {
        chase::linalg::internal::hip::batched_dot_product_gpu(
            d_v1, d_v2, d_results, rows, numvec, ld, false);
    }

    T result{};
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(&result, d_results, sizeof(T), hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_v1));
    ASSERT_HIP_SUCCESS(hipFree(d_v2));
    ASSERT_HIP_SUCCESS(hipFree(d_results));

    if constexpr (std::is_same_v<T, float>)
    {
        EXPECT_FLOAT_EQ(result, 11.0f);
    }
    else if constexpr (std::is_same_v<T, double>)
    {
        EXPECT_DOUBLE_EQ(result, 11.0);
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>)
    {
        EXPECT_FLOAT_EQ(result.real(), 11.0f);
        EXPECT_FLOAT_EQ(result.imag(), 0.0f);
    }
    else
    {
        EXPECT_DOUBLE_EQ(result.real(), 11.0);
        EXPECT_DOUBLE_EQ(result.imag(), 0.0);
    }
}

TEST(LanczosKernelsHIPTest, CopyRealNegate)
{
    std::vector<float> in = {1.0f, -2.0f, 3.5f};
    std::vector<float> expected = {-1.0f, 2.0f, -3.5f};

    float *d_in = nullptr, *d_out = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_in, in.size() * sizeof(float)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_out, expected.size() * sizeof(float)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_in, in.data(), in.size() * sizeof(float),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::copy_real_negate_to_T_gpu(d_in, d_out,
                                                             static_cast<int>(in.size()));

    std::vector<float> out(expected.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_in));
    ASSERT_HIP_SUCCESS(hipFree(d_out));

    for (std::size_t i = 0; i < out.size(); ++i)
    {
        EXPECT_FLOAT_EQ(out[i], expected[i]);
    }
}
