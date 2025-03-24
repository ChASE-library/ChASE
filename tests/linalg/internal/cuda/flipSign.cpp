// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda/flipSign.hpp"


template <typename T>
class flipLowerHalfSignGPUTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(flipLowerHalfSignGPUTest, TestTypes);

TYPED_TEST(flipLowerHalfSignGPUTest, flipSign) {
    using T = TypeParam;

    std::size_t rows = 4;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);

    /*
    1,5,9                                  1,  5,9
    2,6,10   --> flip lower half sign ->   2,  6,10
    3,7,11                                -3, -7,-11
    4,8,12                                -4, -8,-12
     */
    for(auto i = 0; i < ld * cols; i++){
        buffer[i] = i + 1;
    }

    chase::matrix::Matrix<T, chase::platform::GPU> *matrix = new chase::matrix::Matrix<T, chase::platform::GPU>(rows, cols, ld, buffer.data());
    T expected[12] = {1, 2, -3, -4, 5, 6, -7, -8, 9, 10, -11, -12};

    chase::linalg::internal::cuda::flipLowerHalfMatrixSign(matrix);
    
    matrix->D2H();

    for(auto i = 0; i < ld * cols; i++)
    {
        EXPECT_EQ(matrix->cpu_data()[i], expected[i]);
    }
	
    delete matrix;
}
