// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include "linalg/internal/cpu/symOrHerm.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "algorithm/types.hpp"

template <typename T>
class SymOrHermCPUTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {}

};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SymOrHermCPUTest, TestTypes);

TYPED_TEST(SymOrHermCPUTest, UpperTriangularMatrix) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 3;
    T H[N * N] = { 1, 2, 3,
                   0, 4, 5,
                   0, 0, 6 };
    chase::linalg::internal::cpu::symOrHermMatrix('L', N, H, N);
    EXPECT_EQ(H[1], conjugate(T(2.0))); // Check conjugation
    EXPECT_EQ(H[2],conjugate(T(3.0))); // Check conjugation
    EXPECT_EQ(H[5],conjugate(T(5.0))); // Check conjugation
}

TYPED_TEST(SymOrHermCPUTest, LowerTriangularMatrix) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 3;
    T H[N * N] = { 1, 0, 0,
                   2, 4, 0,
                   3, 5, 6 };
    chase::linalg::internal::cpu::symOrHermMatrix('U', N, H, N);
    EXPECT_EQ(H[3], conjugate(T(2.0))); // Check conjugation
    EXPECT_EQ(H[6],conjugate(T(3.0))); // Check conjugation
    EXPECT_EQ(H[7],conjugate(T(5.0))); // Check conjugation
}

TYPED_TEST(SymOrHermCPUTest, SymmetricMatrixCheck) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 3;
    T H[N * N] = { 1, 2, 3,
                   2, 4, 5,
                   3, 5, 6 };
    EXPECT_TRUE(chase::linalg::internal::cpu::checkSymmetryEasy(N, H, N));
}

TYPED_TEST(SymOrHermCPUTest, NonSymmetricMatrixCheck) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 3;
    T H[N * N] = { 1, 2, 4,
                   2, 4, 5,
                   3, 5, 6 };
    EXPECT_FALSE(chase::linalg::internal::cpu::checkSymmetryEasy(N, H, N));
}