// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "external/blaspp/blaspp.hpp"
#include "linalg/internal/cpu/rayleighRitz.hpp"
#include "linalg/internal/cpu/utils.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "linalg/matrix/matrix.hpp"

template <typename T>
class PseudoHermitianRayleighRitzCPUTest : public ::testing::Test {
protected:
    void SetUp() override {

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        //Standard variables init
	H = new chase::matrix::PseudoHermitianMatrix<T,chase::platform::CPU>(N,N);
	exact_eigsl_H = new chase::matrix::Matrix<T,chase::platform::CPU>(N,1);
        
    	ritzv.resize(nev);
	V.resize(N * nev);
	Q.resize(N * nev);
	W.resize(nev * nev);
	Workspace.resize(3 * nev * nev);

        for(auto i = 0; i < nev; i++)
        {
	    Q.data()[i*(N+1)] = 1.0;
        }

        //Tiny variables init	
	H_tiny = new chase::matrix::PseudoHermitianMatrix<T,chase::platform::CPU>(N_tiny,N_tiny);
	exact_eigsl_H_tiny = new chase::matrix::Matrix<T,chase::platform::CPU>(N_tiny,1);

    	ritzv_tiny.resize(nev_tiny);
	V_tiny.resize(N_tiny   * nev_tiny);
	Q_tiny.resize(N_tiny   * nev_tiny);
	W_tiny.resize(nev_tiny * nev_tiny);
	Workspace_tiny.resize(3 * nev_tiny * nev_tiny);
        
        for(auto i = 0; i < nev_tiny; i++)
        {
	    Q_tiny.data()[i*(N_tiny+1)] = 1.0;
        }
    }

    void TearDown() override {}

    //Standard variable sets
    std::size_t k = 100;
    std::size_t N = 2*k;
    std::size_t nev = N;

    std::vector<T> V;
    std::vector<T> Q;
    std::vector<T> W;
    std::vector<T> Workspace;
    std::vector<chase::Base<T>> ritzv;

    chase::matrix::Matrix<T> * exact_eigsl_H; 
    chase::matrix::PseudoHermitianMatrix<T> * H;
    
    //Tiny variable sets
    std::size_t k_tiny = 5;
    std::size_t N_tiny = 2*k_tiny;
    std::size_t nev_tiny = N_tiny;

    std::vector<T> V_tiny;
    std::vector<T> Q_tiny;
    std::vector<T> W_tiny;
    std::vector<T> Workspace_tiny;
    std::vector<chase::Base<T>> ritzv_tiny;
    
    chase::matrix::Matrix<T> * exact_eigsl_H_tiny; 
    chase::matrix::PseudoHermitianMatrix<T> * H_tiny;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHermitianRayleighRitzCPUTest, TestTypes);

TYPED_TEST(PseudoHermitianRayleighRitzCPUTest, PseudoHermitianRayleighRitz) {
    using T = TypeParam;

    this->H->readFromBinaryFile(GetBSE_Matrix<T>());
    this->exact_eigsl_H->readFromBinaryFile(GetBSE_Eigs<T>());

    chase::linalg::internal::cpu::rayleighRitz(this->H,this->nev,this->Q.data(),this->N,
			this->V.data(),this->N,this->ritzv.data(),this->Workspace.data());

    std::sort(this->ritzv.begin(), this->ritzv.end());

    for(auto i = 0; i < this->nev; i++)
    {
    	EXPECT_LT(this->ritzv.data()[i],std::real(this->exact_eigsl_H->data()[i]) + GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    	EXPECT_GT(this->ritzv.data()[i],std::real(this->exact_eigsl_H->data()[i]) - GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    }
}

TYPED_TEST(PseudoHermitianRayleighRitzCPUTest, tinyPseudoHermitianRayleighRitz) {
    using T = TypeParam;

    this->H_tiny->readFromBinaryFile(GetBSE_TinyMatrix<T>());
    this->exact_eigsl_H_tiny->readFromBinaryFile(GetBSE_TinyEigs<T>());

    chase::linalg::internal::cpu::rayleighRitz(this->H_tiny,this->nev_tiny,this->Q_tiny.data(),this->N_tiny,
			this->V_tiny.data(),this->N_tiny,this->ritzv_tiny.data(),this->Workspace_tiny.data());

    std::sort(this->ritzv_tiny.begin(), this->ritzv_tiny.end());

    for(auto i = 0; i < this->nev_tiny; i++)
    {
    	EXPECT_LT(this->ritzv_tiny.data()[i],std::real(this->exact_eigsl_H_tiny->data()[i]) + GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    	EXPECT_GT(this->ritzv_tiny.data()[i],std::real(this->exact_eigsl_H_tiny->data()[i]) - GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    }
}
