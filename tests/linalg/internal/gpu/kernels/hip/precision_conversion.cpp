// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/kernels/hip/launchers/precision_conversion.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

} // namespace

TEST(PrecisionConversionHIPTest, DoubleToFloat)
{
    const std::vector<double> input = {1.0, 2.5, -3.75};
    const std::vector<float> expected = {1.0f, 2.5f, -3.75f};

    double* d_input = nullptr;
    float* d_output = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_input, input.size() * sizeof(double)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_output, expected.size() * sizeof(float)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_input, input.data(), input.size() * sizeof(double),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::convert_DP_TO_SP_GPU(d_input, d_output,
                                                         input.size());

    std::vector<float> output(expected.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(output.data(), d_output,
                                 output.size() * sizeof(float),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_input));
    ASSERT_HIP_SUCCESS(hipFree(d_output));

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(output[i], expected[i]);
    }
}

TEST(PrecisionConversionHIPTest, FloatToDouble)
{
    const std::vector<float> input = {1.0f, 2.5f, -3.75f};
    const std::vector<double> expected = {1.0, 2.5, -3.75};

    float* d_input = nullptr;
    double* d_output = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_input, input.size() * sizeof(float)));
    ASSERT_HIP_SUCCESS(hipMalloc(&d_output, expected.size() * sizeof(double)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_input, input.data(), input.size() * sizeof(float),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::convert_SP_TO_DP_GPU(d_input, d_output,
                                                         input.size());

    std::vector<double> output(expected.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(output.data(), d_output,
                                 output.size() * sizeof(double),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_input));
    ASSERT_HIP_SUCCESS(hipFree(d_output));

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(output[i], expected[i]);
    }
}

TEST(PrecisionConversionHIPTest, ComplexDoubleToFloat)
{
    const std::vector<std::complex<double>> input = {{1.0, 2.0}, {3.0, -4.0}};
    const std::vector<std::complex<float>> expected = {{1.0f, 2.0f}, {3.0f, -4.0f}};

    std::complex<double>* d_input = nullptr;
    std::complex<float>* d_output = nullptr;
    ASSERT_HIP_SUCCESS(
        hipMalloc(&d_input, input.size() * sizeof(std::complex<double>)));
    ASSERT_HIP_SUCCESS(
        hipMalloc(&d_output, expected.size() * sizeof(std::complex<float>)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_input, input.data(),
                                 input.size() * sizeof(std::complex<double>),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::convert_DP_TO_SP_GPU(d_input, d_output,
                                                         input.size());

    std::vector<std::complex<float>> output(expected.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(output.data(), d_output,
                                 output.size() * sizeof(std::complex<float>),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_input));
    ASSERT_HIP_SUCCESS(hipFree(d_output));

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_FLOAT_EQ(output[i].real(), expected[i].real());
        EXPECT_FLOAT_EQ(output[i].imag(), expected[i].imag());
    }
}

TEST(PrecisionConversionHIPTest, ComplexFloatToDouble)
{
    const std::vector<std::complex<float>> input = {{1.0f, 2.0f}, {3.0f, -4.0f}};
    const std::vector<std::complex<double>> expected = {{1.0, 2.0}, {3.0, -4.0}};

    std::complex<float>* d_input = nullptr;
    std::complex<double>* d_output = nullptr;
    ASSERT_HIP_SUCCESS(
        hipMalloc(&d_input, input.size() * sizeof(std::complex<float>)));
    ASSERT_HIP_SUCCESS(
        hipMalloc(&d_output, expected.size() * sizeof(std::complex<double>)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_input, input.data(),
                                 input.size() * sizeof(std::complex<float>),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::convert_SP_TO_DP_GPU(d_input, d_output,
                                                         input.size());

    std::vector<std::complex<double>> output(expected.size());
    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());
    ASSERT_HIP_SUCCESS(hipMemcpy(output.data(), d_output,
                                 output.size() * sizeof(std::complex<double>),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_input));
    ASSERT_HIP_SUCCESS(hipFree(d_output));

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(output[i].real(), expected[i].real());
        EXPECT_DOUBLE_EQ(output[i].imag(), expected[i].imag());
    }
}
