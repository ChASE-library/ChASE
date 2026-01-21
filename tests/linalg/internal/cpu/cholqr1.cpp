// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cpu/cholqr1.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>

template <typename T>
class CholQRCPUTest : public ::testing::Test
{
protected:
    void SetUp() override { V.resize(m * n); }

    void TearDown() override {}

    std::size_t m = 100;
    std::size_t n = 50;
    std::vector<T> V;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(CholQRCPUTest, TestTypes);

TYPED_TEST(CholQRCPUTest, cholQR1)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_10.bin", 0, this->m,
                 this->m, this->n, 0);
    int info = chase::linalg::internal::cpu::cholQR1<T>(
        this->m, this->n, this->V.data(), this->m);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CholQRCPUTest, cholQR1BadlyCond)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_1e4.bin", 0,
                 this->m, this->m, this->n, 0);
    int info = chase::linalg::internal::cpu::cholQR1<T>(
        this->m, this->n, this->V.data(), this->m);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    EXPECT_GT(orth, machineEpsilon);
    EXPECT_LT(orth, 1.0);
}

TYPED_TEST(CholQRCPUTest, cholQR1IllCond)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    int info = chase::linalg::internal::cpu::cholQR1<T>(
        this->m, this->n, this->V.data(), this->m);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CholQRCPUTest, cholQR2)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_1e4.bin", 0,
                 this->m, this->m, this->n, 0);
    int info = chase::linalg::internal::cpu::cholQR2<T>(
        this->m, this->n, this->V.data(), this->m);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CholQRCPUTest, cholQR2IllCond)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    int info = chase::linalg::internal::cpu::cholQR2<T>(
        this->m, this->n, this->V.data(), this->m);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CholQRCPUTest, scholQR)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    int info = chase::linalg::internal::cpu::shiftedcholQR2<T>(
        this->m, this->n, this->V.data(), this->m);
    ASSERT_EQ(info, 0);
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);
}
