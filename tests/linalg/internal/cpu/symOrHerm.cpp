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

TYPED_TEST(SymOrHermCPUTest, LowerTriangularMatrix) {
    using T = TypeParam;
    const std::size_t N = 3;
    
    // Create matrix on heap using vector
    std::vector<T> H(N * N);
    
    // Initialize values explicitly
    H[0] = static_cast<T>(1); H[1] = static_cast<T>(0); H[2] = static_cast<T>(0);
    H[3] = static_cast<T>(2); H[4] = static_cast<T>(4); H[5] = static_cast<T>(0);
    H[6] = static_cast<T>(3); H[7] = static_cast<T>(5); H[8] = static_cast<T>(6);

    chase::linalg::internal::cpu::symOrHermMatrix('U', N, H.data(), N);

    // Test expected values
    EXPECT_EQ(H[3], conjugate(static_cast<T>(2)));
    EXPECT_EQ(H[6], conjugate(static_cast<T>(3)));
    EXPECT_EQ(H[7], conjugate(static_cast<T>(5)));
}

TYPED_TEST(SymOrHermCPUTest, SymmetricMatrixCheck) {
    using T = TypeParam;
    const std::size_t N = 3;
    
    // Create matrix on heap using vector
    std::vector<T> H(N * N);
    
    // Initialize values explicitly
    H[0] = static_cast<T>(1); H[1] = static_cast<T>(2); H[2] = static_cast<T>(3);
    H[3] = static_cast<T>(2); H[4] = static_cast<T>(4); H[5] = static_cast<T>(5);
    H[6] = static_cast<T>(3); H[7] = static_cast<T>(5); H[8] = static_cast<T>(6);

    EXPECT_TRUE(chase::linalg::internal::cpu::checkSymmetryEasy(N, H.data(), N));
}

TYPED_TEST(SymOrHermCPUTest, NonSymmetricMatrixCheck) {
    using T = TypeParam;
    const std::size_t N = 3;
    
    // Create matrix on heap using vector
    std::vector<T> H(N * N);
    
    // Initialize values explicitly
    H[0] = static_cast<T>(1); H[1] = static_cast<T>(2); H[2] = static_cast<T>(4);
    H[3] = static_cast<T>(2); H[4] = static_cast<T>(4); H[5] = static_cast<T>(5);
    H[6] = static_cast<T>(3); H[7] = static_cast<T>(5); H[8] = static_cast<T>(6);

    EXPECT_FALSE(chase::linalg::internal::cpu::checkSymmetryEasy(N, H.data(), N));
}