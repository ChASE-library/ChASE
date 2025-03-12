// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "external/blaspp/blaspp.hpp"
#include "linalg/internal/cpu/lanczos.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "linalg/matrix/matrix.hpp"

template <typename T>
class QuasiHermitianRRCPUTest : public ::testing::Test {
protected:
    void SetUp() override {

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        //Standard variables init
	H = new chase::matrix::QuasiHermitianMatrix<T,chase::platform::CPU>(N,N);
	exact_eigsl_H = new chase::matrix::Matrix<T,chase::platform::CPU>(N,1);
        
        for(auto i = 0; i < N * nev; i++)
        {
            V[i] =  getRandomT<T>([&]() { return d(gen); });
        }

        //Tiny variables init	
	H_tiny = new chase::matrix::QuasiHermitianMatrix<T,chase::platform::CPU>(N_tiny,N_tiny);
	exact_eigsl_H_tiny = new chase::matrix::Matrix<T,chase::platform::CPU>(N_tiny,1);
        
        for(auto i = 0; i < N_tiny * nev_tiny; i++)
        {
            V_tiny[i] =  getRandomT<T>([&]() { return d(gen); });
        }
    }

    void TearDown() override {}

    //Standard variable sets
    std::size_t k = 100;
    std::size_t N = 2*k;
    std::size_t nev = N;

    std::vector<T> V (N * nev, T(0.0));
    std::vector<T> Q (N * nev, T(0.0));
    std::vector<T> W((nev * nev, T(0.0));
    std::vector<T> G((nev * nev, T(0.0));
    std::vector<T> halfQ(k * nev, T(0.0));
    std::vector<Base<T>> ritzv(nev, chase::Base<T>(0.0));

    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;
    chase::matrix::Matrix<T> * exact_eigsl_H; 
    chase::matrix::QuasiHermitianMatrix<T> * H;
    
    //Tiny variable sets
    std::size_t k_tiny = 5;
    std::size_t N_tiny = 2*k_tiny;
    std::size_t nev_tiny = N_tiny;

    std::vector<T> V_tiny (N_tiny * nev_tiny, T(0.0));
    std::vector<T> Q_tiny (N_tiny * nev_tiny, T(0.0));
    std::vector<T> W_tiny (nev_tiny * nev_tiny, T(0.0));
    std::vector<T> G_tiny (nev_tiny * nev_tiny, T(0.0));
    std::vector<T> halfQ_tiny (k_tiny * nev_tiny, T(0.0));
    std::vector<Base<T>> ritzv_tiny(nev_tiny, chase::Base<T>(0.0));
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(QuasiHermitianRRCPUTest, TestTypes);

TYPED_TEST(QuasiHermitianRRCPUTest, QuasiHermitianRayleighRitz) {
    using T = TypeParam;

    this->H->readFromBinaryFile(GetBSE_Matrix<T>());
    this->exact_eigsl_H->readFromBinaryFile(GetBSE_Eigs<T>());

    chase::linalg::internal::cpu::rayleighRitz(this->H,this->nev,this->Q.data(),this->N,
			this->V.data(),this->nev,this->ritzv.data(),this->G.data(),this->halfQ.data());

    for(auto i = 0; i < this->nev; i++)
    {
    	EXPECT_LT(T(this->ritzv.data()[i]),this->exact_eigsl_H->data()[i] + 1e3*MachineEpsilon<chase::Base<T>>::value());
    	EXPECT_GT(T(this->ritzv.data()[i]),this->exact_eigsl_H->data()[i] - 1e3*MachineEpsilon<chase::Base<T>>::value());
    }
}

TYPED_TEST(QuasiHermitianRRCPUTest, tinyQuasiHermitianRayleighRitz) {
    using T = TypeParam;

    this->H_tiny->readFromBinaryFile(GetBSE_TinyMatrix<T>());
    this->exact_eigsl_H_tiny->readFromBinaryFile(GetBSE_TinyEigs<T>());

    chase::linalg::internal::cpu::rayleighRitz(this->H_tiny,this->nev_tiny,this->Q_tiny.data(),this->N_tiny,
			this->V_tiny.data(),this->nev_tiny,this->ritzv_tiny.data(),this->G_tiny.data(),this->halfQ_tiny.data());

    for(auto i = 0; i < this->nev_tiny; i++)
    {
    	EXPECT_LT(T(this->ritzv_tiny.data()[i]),this->exact_eigsl_H_tiny->data()[i] + 1e3*MachineEpsilon<chase::Base<T>>::value());
    	EXPECT_GT(T(this->ritzv_tiny.data()[i]),this->exact_eigsl_H_tiny->data()[i] - 1e3*MachineEpsilon<chase::Base<T>>::value());
    }
}
