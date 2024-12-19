// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "linalg/internal/cpu/lanczos.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"

template <typename T>
class LaczosCPUTest : public ::testing::Test {
protected:
    void SetUp() override {

        H.resize(N * N);
        V.resize(N * M);
        ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        for(auto i = 0; i < N * M; i++)
        {
            V[i] =  getRandomT<T>([&]() { return d(gen); });
        }

        //set H to be Clement matrix
        //Clement matrix has eigenvalues -(N-1),-(N-2)...(N-2), (N-1)
        for (auto i = 0; i < N; ++i)
        {
            H[i + N * i] = 0;
            if (i != N - 1)
                H[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
            if (i != N - 1)
                H[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
        }

    }

    void TearDown() override {}

    std::size_t M = 10;
    std::size_t numvec = 4;
    std::size_t N = 500;
    std::vector<T> H;
    std::size_t ldh = N;
    std::vector<T> V;
    std::size_t ldv = N;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LaczosCPUTest, TestTypes);

TYPED_TEST(LaczosCPUTest, mlanczos) {
    using T = TypeParam;  // Get the current type
    chase::Base<T> upperb;
    chase::linalg::internal::cpu::lanczos(this->M, this->numvec, this->N, this->H.data(), this->N, this->V.data(), this->N, 
                &upperb, this->ritzv.data(), this->Tau.data(), this->ritzV.data());

    for(auto i = 0; i < this->numvec; i++)
    {
        EXPECT_GT(this->ritzv[i * this->M], 1.0 - chase::Base<T>(this->N));
        EXPECT_LT(this->ritzv[(i + 1) * this->M-1], chase::Base<T>(this->N - 1));
    }
    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );
}

TYPED_TEST(LaczosCPUTest, lanczos) {
    using T = TypeParam;  // Get the current type
    chase::Base<T> upperb;
    chase::linalg::internal::cpu::lanczos(this->M, this->N, this->H.data(), this->N, this->V.data(), this->N, 
                &upperb);

    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );
}