// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cpu/utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>

template <typename T>
class flipSignCPUTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(flipSignCPUTest, TestTypes);

TYPED_TEST(flipSignCPUTest, flipLowerSignCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 10;

    auto H = chase::matrix::Matrix<T, chase::platform::CPU>(N, N);

    for (auto i = 0; i < H.rows() * H.cols(); i++)
    {
        H.data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu::flipLowerHalfMatrixSign(H.rows(), H.cols(),
                                                          H.data(), H.ld());

    for (auto j = 0; j < H.cols(); j++)
    {
        for (auto i = 0; i < H.rows() / 2; i++)
        {
            EXPECT_EQ(H.data()[i + j * H.ld()], T(1.0));
        }
        for (auto i = H.rows() / 2; i < H.rows(); i++)
        {
            EXPECT_EQ(H.data()[i + j * H.ld()], -T(1.0));
        }
    }
}

TYPED_TEST(flipSignCPUTest, flipRightSignCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 10;

    auto H = chase::matrix::Matrix<T, chase::platform::CPU>(N, N);

    for (auto i = 0; i < H.rows() * H.cols(); i++)
    {
        H.data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu::flipRightHalfMatrixSign(H.rows(), H.cols(),
                                                          H.data(), H.ld());

    for (auto j = 0; j < H.rows(); j++)
    {
        for (auto i = 0; i < H.cols() / 2; i++)
        {
            EXPECT_EQ(H.data()[j + i * H.ld()], T(1.0));
        }
        for (auto i = H.cols() / 2; i < H.cols(); i++)
        {
            EXPECT_EQ(H.data()[j + i * H.ld()], -T(1.0));
        }
    }
}
