// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cpu/lanczos.hpp"
#include "external/cpu/blaspp/blaspp.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "linalg/matrix/matrix.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/lanczos_reference.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <random>

template <typename T>
class LaczosCPUTest : public ::testing::Test
{
protected:
    void SetUp() override
    {

        H = chase::matrix::Matrix<T>(N, N);
        V.resize(N * M);
        ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        for (auto i = 0; i < N * M; i++)
        {
            V[i] = getRandomT<T>([&]() { return d(gen); });
        }

        chase::tests::lanczos::initializeReferenceHermitianMatrix(H.data(), N);
        chase::tests::lanczos::computeReferenceEigenBounds(
            H.data(), N, &lambda_min, &lambda_max);
    }

    void TearDown() override {}

    std::size_t M = 10;
    std::size_t numvec = 4;
    std::size_t N = 500;
    chase::matrix::Matrix<T> H;
    std::vector<T> V;
    std::size_t ldv = N;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;
    chase::Base<T> lambda_min;
    chase::Base<T> lambda_max;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LaczosCPUTest, TestTypes);

TYPED_TEST(LaczosCPUTest, mlanczos)
{
    using T = TypeParam; // Get the current type
    chase::Base<T> upperb;
    chase::linalg::internal::cpu::lanczos(
        this->M, this->numvec, &(this->H), this->V.data(), this->N, &upperb,
        this->ritzv.data(), this->Tau.data(), this->ritzV.data());

    chase::tests::lanczos::expectRitzValuesInsideReferenceSpectrum(
        this->ritzv, this->M, this->numvec, this->lambda_min,
        this->lambda_max);
    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}

TYPED_TEST(LaczosCPUTest, lanczos)
{
    using T = TypeParam; // Get the current type
    chase::Base<T> upperb;
    chase::linalg::internal::cpu::lanczos(this->M, &(this->H), this->V.data(),
                                          this->N, &upperb);

    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}
