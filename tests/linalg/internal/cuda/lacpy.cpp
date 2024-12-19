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
#include "linalg/internal/cuda/lacpy.hpp"
#include "linalg/matrix/matrix.hpp"

template <typename T>
class lacpyGPUTest : public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(lacpyGPUTest, TestTypes);

TYPED_TEST(lacpyGPUTest, fullCopy) {
    using T = TypeParam;

    std::size_t rows = 3;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);

    for(auto i = 0; i < ld * cols; i++){
        buffer[i] = i + 1;
    }

    chase::matrix::Matrix<T, chase::platform::GPU> src_matrix(rows, cols, ld, buffer.data());
    chase::matrix::Matrix<T, chase::platform::GPU> target_matrix(rows, cols);
    chase::linalg::internal::cuda::t_lacpy('A', 
                                           rows, 
                                           cols, 
                                           src_matrix.data(), 
                                           src_matrix.ld(), 
                                           target_matrix.data(), 
                                           target_matrix.ld());
    target_matrix.D2H();

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
            EXPECT_EQ(target_matrix.cpu_data()[j * rows + i], buffer.data()[j * ld + i]);
        }
    }
}

TYPED_TEST(lacpyGPUTest, upperCopy) {
    using T = TypeParam;

    std::size_t rows = 3;
    std::size_t cols = 3;
    std::vector<T> buffer(rows * cols);

    for(auto i = 0; i < rows * cols; i++){
        buffer[i] = i + 1;
    }

    std::vector<T> buffer2(rows * cols);
    for(auto i = 0; i < rows * cols; i++){
        buffer2[i] = -(i + 1);
    }

    chase::matrix::Matrix<T, chase::platform::GPU> src_matrix(rows, cols, rows, buffer.data());
    chase::matrix::Matrix<T, chase::platform::GPU> target_matrix(rows, cols, rows, buffer2.data());
    chase::linalg::internal::cuda::t_lacpy('U', 
                                           rows, 
                                           cols, 
                                           src_matrix.data(), 
                                           src_matrix.ld(), 
                                           target_matrix.data(), 
                                           target_matrix.ld());
    target_matrix.D2H();

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
            if(j >= i){
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i], buffer.data()[j * rows + i]);
            }
            else
            {
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i],  T(0.0));
            }
        }
    }
}

TYPED_TEST(lacpyGPUTest, lowerCopy) {
    using T = TypeParam;

    std::size_t rows = 3;
    std::size_t cols = 3;
    std::vector<T> buffer(rows * cols);

    for(auto i = 0; i < rows * cols; i++){
        buffer[i] = i + 1;
    }

    std::vector<T> buffer2(rows * cols);
    for(auto i = 0; i < rows * cols; i++){
        buffer2[i] = -(i + 1);
    }

    chase::matrix::Matrix<T, chase::platform::GPU> src_matrix(rows, cols, rows, buffer.data());
    chase::matrix::Matrix<T, chase::platform::GPU> target_matrix(rows, cols, rows, buffer2.data());
    chase::linalg::internal::cuda::t_lacpy('L', 
                                           rows, 
                                           cols, 
                                           src_matrix.data(), 
                                           src_matrix.ld(), 
                                           target_matrix.data(), 
                                           target_matrix.ld());
    target_matrix.D2H();

    for(auto j = 0; j < cols; j++)
    {
        for(auto i = 0; i < rows; i++)
        {
            if(j <= i){
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i], buffer.data()[j * rows + i]);
            }
            else
            {
                EXPECT_EQ(target_matrix.cpu_data()[j * rows + i],  T(0.0));
            }
        }
    }
}
