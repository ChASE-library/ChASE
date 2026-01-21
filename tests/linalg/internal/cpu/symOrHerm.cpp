// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cpu/symOrHerm.hpp"
#include "algorithm/types.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>

template <typename T>
class SymOrHermCPUTest : public ::testing::Test
{
protected:
    void SetUp() override {}

    void TearDown() override {}
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SymOrHermCPUTest, TestTypes);

TYPED_TEST(SymOrHermCPUTest, UpperTriangularMatrix)
{
    using T = TypeParam;
    const std::size_t N = 3;

    // Create matrix on heap to avoid potential stack issues
    std::vector<T> H(N * N);

    // Initialize values one by one
    H[0] = static_cast<T>(1);
    H[1] = static_cast<T>(2);
    H[2] = static_cast<T>(3);
    H[3] = static_cast<T>(0);
    H[4] = static_cast<T>(4);
    H[5] = static_cast<T>(5);
    H[6] = static_cast<T>(0);
    H[7] = static_cast<T>(0);
    H[8] = static_cast<T>(6);

    // Call the function with pointer to data
    chase::linalg::internal::cpu::symOrHermMatrix('L', N, H.data(), N);

    // Test expected values
    T expected2 = static_cast<T>(2);
    T expected3 = static_cast<T>(3);
    T expected5 = static_cast<T>(5);

    EXPECT_EQ(H[1], conjugate(expected2));
    EXPECT_EQ(H[2], conjugate(expected3));
    EXPECT_EQ(H[5], conjugate(expected5));
}

TYPED_TEST(SymOrHermCPUTest, LowerTriangularMatrix)
{
    using T = TypeParam;
    const std::size_t N = 3;

    // Create matrix on heap using vector
    std::vector<T> H(N * N);

    // Initialize values explicitly
    H[0] = static_cast<T>(1);
    H[1] = static_cast<T>(0);
    H[2] = static_cast<T>(0);
    H[3] = static_cast<T>(2);
    H[4] = static_cast<T>(4);
    H[5] = static_cast<T>(0);
    H[6] = static_cast<T>(3);
    H[7] = static_cast<T>(5);
    H[8] = static_cast<T>(6);

    chase::linalg::internal::cpu::symOrHermMatrix('U', N, H.data(), N);

    // Test expected values
    EXPECT_EQ(H[3], conjugate(static_cast<T>(2)));
    EXPECT_EQ(H[6], conjugate(static_cast<T>(3)));
    EXPECT_EQ(H[7], conjugate(static_cast<T>(5)));
}

TYPED_TEST(SymOrHermCPUTest, SymmetricMatrixCheck)
{
    using T = TypeParam;
    const std::size_t N = 3;

    // Create matrix on heap using vector
    std::vector<T> H(N * N);

    // Initialize values explicitly
    H[0] = static_cast<T>(1);
    H[1] = static_cast<T>(2);
    H[2] = static_cast<T>(3);
    H[3] = static_cast<T>(2);
    H[4] = static_cast<T>(4);
    H[5] = static_cast<T>(5);
    H[6] = static_cast<T>(3);
    H[7] = static_cast<T>(5);
    H[8] = static_cast<T>(6);

    EXPECT_TRUE(
        chase::linalg::internal::cpu::checkSymmetryEasy(N, H.data(), N));
}

TYPED_TEST(SymOrHermCPUTest, NonSymmetricMatrixCheck)
{
    using T = TypeParam;
    const std::size_t N = 3;

    // Create matrix on heap using vector
    std::vector<T> H(N * N);

    // Initialize values explicitly
    H[0] = static_cast<T>(1);
    H[1] = static_cast<T>(2);
    H[2] = static_cast<T>(4);
    H[3] = static_cast<T>(2);
    H[4] = static_cast<T>(4);
    H[5] = static_cast<T>(5);
    H[6] = static_cast<T>(3);
    H[7] = static_cast<T>(5);
    H[8] = static_cast<T>(6);

    EXPECT_FALSE(
        chase::linalg::internal::cpu::checkSymmetryEasy(N, H.data(), N));
}

TYPED_TEST(SymOrHermCPUTest, PseudoHermitianMatrixCheck)
{
    const std::size_t N = 4;

    // Create matrix on heap using vector
    std::vector<std::complex<double>> H(N * N);

    // Initialize values explicitly
    H[0] = std::complex<double>(1.010, 0);
    H[1] = std::complex<double>(0, -0.20);
    H[2] = std::complex<double>(0.010, 0);
    H[3] = std::complex<double>(0, 0.010);
    H[4] = std::complex<double>(0, 0.200);
    H[5] = std::complex<double>(1.010, 0);
    H[6] = std::complex<double>(0, 0.010);
    H[7] = std::complex<double>(0.010, 0);
    H[8] = std::complex<double>(-0.01, 0);
    H[9] = std::complex<double>(0, 0.010);
    H[10] = std::complex<double>(-1.01, 0);
    H[11] = std::complex<double>(0, -0.20);
    H[12] = std::complex<double>(0, 0.010);
    H[13] = std::complex<double>(-0.01, 0);
    H[14] = std::complex<double>(0, 0.200);
    H[15] = std::complex<double>(-1.01, 0);

    chase::linalg::internal::cpu::flipLowerHalfMatrixSign(N, N, &(H[0]), N);

    EXPECT_TRUE(
        chase::linalg::internal::cpu::checkSymmetryEasy(N, H.data(), N));
}
