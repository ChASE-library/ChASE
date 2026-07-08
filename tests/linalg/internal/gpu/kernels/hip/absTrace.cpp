// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "algorithm/types.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/abs_trace.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

template <typename T>
class AbsTraceHIPTest : public ::testing::Test
{
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(AbsTraceHIPTest, TestTypes);

} // namespace

TYPED_TEST(AbsTraceHIPTest, AbsTrace)
{
    using T = TypeParam;
    using TraceT = chase::Base<T>;

    constexpr std::size_t rows = 3;
    constexpr std::size_t cols = 3;
    constexpr std::size_t ld = 4;
    constexpr std::size_t n = std::min(rows, cols);

    std::vector<T> host(ld * cols);
    for (std::size_t i = 0; i < host.size(); ++i)
    {
        host[i] = T(i + 1);
    }
    host[5] = T(chase::Base<T>(-4));

    T* d_matrix = nullptr;
    TraceT* d_trace = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_matrix, host.size() * sizeof(T)));
    ASSERT_HIP_SUCCESS(hipMalloc(reinterpret_cast<void**>(&d_trace),
                                 sizeof(TraceT)));
    ASSERT_HIP_SUCCESS(hipMemcpy(d_matrix, host.data(), host.size() * sizeof(T),
                                 hipMemcpyHostToDevice));

    chase::linalg::internal::hip::absTrace_gpu(d_matrix, d_trace, n, ld);

    ASSERT_HIP_SUCCESS(hipDeviceSynchronize());

    TraceT trace = TraceT(0);
    ASSERT_HIP_SUCCESS(hipMemcpy(&trace, d_trace, sizeof(TraceT),
                                 hipMemcpyDeviceToHost));
    ASSERT_HIP_SUCCESS(hipFree(d_matrix));
    ASSERT_HIP_SUCCESS(hipFree(d_trace));

    EXPECT_EQ(trace, TraceT(16.0));
}
