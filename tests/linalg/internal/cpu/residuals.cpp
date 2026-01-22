// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cpu/residuals.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <random>

template <typename T>
class ResidsCPUTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        H.resize(N * N);
        evals.resize(N);
        evecs.resize(N * N);
        resids.resize(N);
    }

    void TearDown() override {}

    std::size_t N = 64;
    std::vector<T> H;
    std::vector<chase::Base<T>> evals;
    std::vector<T> evecs;
    std::vector<chase::Base<T>> resids;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ResidsCPUTest, TestTypes);

TYPED_TEST(ResidsCPUTest, DiagonalMatrix)
{
    using T = TypeParam; // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();

    for (auto i = 0; i < this->N; i++)
    {
        this->H[i * this->N + i] = T(i + 1);
    }

    for (auto i = 0; i < this->N; i++)
    {
        this->evals[i] = chase::Base<T>(i + 1);
        this->evecs[i + this->N * i] = T(1.0);
    }

    chase::linalg::internal::cpu::residuals(
        this->N, this->H.data(), this->N, this->N, this->evals.data(),
        this->evecs.data(), this->N, this->resids.data());

    for (auto i = 0; i < this->N; i++)
    {
        EXPECT_NEAR(this->resids[i], machineEpsilon, machineEpsilon * 10);
    }
}

TYPED_TEST(ResidsCPUTest, DenseMatrix)
{
    using T = TypeParam; // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    T One = T(1.0);
    T Zero = T(0.0);
    for (int i = 0; i < this->N; ++i)
    {
        this->H[i * this->N + i] = T(0.1 * i + 0.1);
    }

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;
    std::vector<T> V(this->N * this->N);

    for (auto i = 0; i < this->N * this->N; i++)
    {
        V[i] = getRandomT<T>([&]() { return d(gen); });
    }

    std::unique_ptr<T[]> tau(new T[this->N]);

    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, this->N, this->N,
                                     V.data(), this->N, tau.get());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, this->N, this->N, this->N,
                                   V.data(), this->N, tau.get());

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                  this->N, this->N, this->N, &One,
                                  this->H.data(), this->N, V.data(), this->N,
                                  &Zero, this->H.data(), this->N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                  this->N, this->N, this->N, &One, V.data(),
                                  this->N, this->H.data(), this->N, &Zero,
                                  this->evecs.data(), this->N);

    chase::linalg::lapackpp::t_lacpy('A', this->N, this->N, this->evecs.data(),
                                     this->N, this->H.data(), this->N);

    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'V', 'U', this->N,
                                     this->evecs.data(), this->N,
                                     this->evals.data());

    chase::linalg::internal::cpu::residuals(
        this->N, this->H.data(), this->N, this->N, this->evals.data(),
        this->evecs.data(), this->N, this->resids.data());

    for (auto i = 0; i < this->N; i++)
    {
        EXPECT_NEAR(this->resids[i], machineEpsilon, machineEpsilon * 1e2);
    }
}