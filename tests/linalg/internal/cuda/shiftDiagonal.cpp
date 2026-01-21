// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cuda/shiftDiagonal.hpp"
#include "algorithm/types.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <random>

template <typename T>
class shiftMatrixGPUTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(shiftMatrixGPUTest, TestTypes);

TYPED_TEST(shiftMatrixGPUTest, ShiftMatrix)
{
    using T = TypeParam;

    std::size_t rows = 4;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);
    chase::Base<T> shift = chase::Base<T>(-2.0);

    /*
    1,5,9                     -1,5,9
    2,6,10   --> shift -2 ->  2, 4, 10
    3,7,11                    3, 7, 9
    4,8,12                    4, 8, 12
     */
    for (auto i = 0; i < ld * cols; i++)
    {
        buffer[i] = i + 1;
    }

    chase::matrix::Matrix<T, chase::platform::GPU>* matrix =
        new chase::matrix::Matrix<T, chase::platform::GPU>(rows, cols, ld,
                                                           buffer.data());
    T expected[12] = {-1, 2, 3, 4, 5, 4, 7, 8, 9, 10, 9, 12};

    chase::linalg::internal::cuda::shiftDiagonal(matrix, shift);

    matrix->D2H();

    for (auto i = 0; i < ld * cols; i++)
    {
        EXPECT_EQ(matrix->cpu_data()[i], expected[i]);
    }

    delete matrix;
}

TYPED_TEST(shiftMatrixGPUTest, SetDiagonal)
{
    using T = TypeParam;

    std::size_t rows = 4;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);
    T coef = T(-2.0);

    /*
    1,5,9                       -2, 5,  9
    2,6,10   --> set diag -2 ->  2,-2, 10
    3,7,11                       3, 7, -2
    4,8,12                       4, 8, 12
     */
    for (auto i = 0; i < ld * cols; i++)
    {
        buffer[i] = i + 1;
    }

    chase::matrix::Matrix<T, chase::platform::GPU>* matrix =
        new chase::matrix::Matrix<T, chase::platform::GPU>(rows, cols, ld,
                                                           buffer.data());
    T expected[12] = {-2, 2, 3, 4, 5, -2, 7, 8, 9, 10, -2, 12};

    chase::linalg::internal::cuda::setDiagonal(matrix, coef);

    matrix->D2H();

    for (auto i = 0; i < ld * cols; i++)
    {
        EXPECT_EQ(matrix->cpu_data()[i], expected[i]);
    }

    delete matrix;
}

TYPED_TEST(shiftMatrixGPUTest, ScaleRows)
{
    using T = TypeParam;

    std::size_t rows = 4;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);

    std::vector<chase::Base<T>> diag_vals(rows);

    for (auto i = 0; i < rows; i++)
    {
        diag_vals[i] = -1.0 / (i + 1);
    }

    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>* coefs =
        new chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(
            rows, 1, ld, diag_vals.data());

    /*
    1,1,1                               -1,-1,-1
    2,2,2   --> scale matrix rows  ->   -1,-1,-1
    3,3,3                               -1,-1,-1
    4,4,4                               -1,-1,-1
     */

    for (auto i = 0; i < ld * cols; i++)
    {
        buffer[i] = i % ld + 1;
    }

    chase::matrix::Matrix<T, chase::platform::GPU>* matrix =
        new chase::matrix::Matrix<T, chase::platform::GPU>(rows, cols, ld,
                                                           buffer.data());
    T expected[12] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    chase::linalg::internal::cuda::scaleMatrixRows(matrix, coefs->data());

    matrix->D2H();

    for (auto i = 0; i < ld * cols; i++)
    {
        EXPECT_EQ(matrix->cpu_data()[i], expected[i]);
    }

    delete matrix;
}

TYPED_TEST(shiftMatrixGPUTest, SubtractInverseDiagonal)
{
    using T = TypeParam;

    std::size_t rows = 4;
    std::size_t cols = 3;
    std::size_t ld = 4;
    std::vector<T> buffer(ld * cols);

    chase::Base<T> coef = 2.0;
    std::vector<chase::Base<T>> diag_vals(rows);

    for (auto i = 0; i < rows; i++)
    {
        diag_vals[i] = 0.0;
    }

    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>* new_diag =
        new chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(
            rows, 1, ld, diag_vals.data());

    /*
    3,1,1                                   -1
    2,3,2   --> new diag vector in GPU ->   -1
    3,3,3                                   -1
    4,4,4
     */

    for (auto i = 0; i < ld * cols; i++)
    {
        buffer[i] = i;
    }

    buffer[0] = 3;
    buffer[5] = 3;
    buffer[10] = 3;

    chase::matrix::Matrix<T, chase::platform::GPU>* matrix =
        new chase::matrix::Matrix<T, chase::platform::GPU>(rows, cols, ld,
                                                           buffer.data());
    T expected[3] = {-1, -1, -1};

    chase::linalg::internal::cuda::subtractInverseDiagonal(matrix, coef,
                                                           new_diag->data());

    new_diag->D2H();

    for (auto i = 0; i < cols; i++)
    {
        EXPECT_EQ(new_diag->cpu_data()[i], expected[i]);
    }

    delete matrix;
}
