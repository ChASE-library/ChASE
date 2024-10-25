#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda/absTrace.hpp"


template <typename T>
class absTraceGPUTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(absTraceGPUTest, TestTypes);

TYPED_TEST(absTraceGPUTest, absTrace) {
    using T = TypeParam;

    std::size_t rows = 3;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);
    chase::Base<T> *d_abstrace, abstrace = 0;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_abstrace, sizeof(chase::Base<T>)));

    for(auto i = 0; i < ld * cols; i++){
        buffer[i] = i + 1;
    }

    buffer[5] = chase::Base<T>(-4);

    chase::matrix::Matrix<T, chase::platform::GPU> matrix(rows, cols, ld, buffer.data());
    chase::Base<T> expect_value = chase::Base<T>(16.0);

    chase::linalg::internal::cuda::absTrace(matrix, d_abstrace);
    cudaMemcpy(&abstrace, d_abstrace, sizeof(chase::Base<T>), cudaMemcpyDeviceToHost);

    EXPECT_EQ(abstrace, expect_value);
}