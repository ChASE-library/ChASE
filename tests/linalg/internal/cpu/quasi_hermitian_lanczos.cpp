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
class QuasiHermitianLanczosCPUTest : public ::testing::Test {
protected:
    void SetUp() override {

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

	//Standard variables init
        V.resize(N * M);
        
	H = new chase::matrix::QuasiHermitianMatrix<T,chase::platform::CPU>(N,N);
	exact_eigsl_H = new chase::matrix::Matrix<T,chase::platform::CPU>(N,1);
        
	ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);

        for(auto i = 0; i < N * M; i++)
        {
            V[i] =  getRandomT<T>([&]() { return d(gen); });
        }

        //Tiny variables init
        V_tiny.resize(N_tiny * M_tiny);
        
	H_tiny = new chase::matrix::QuasiHermitianMatrix<T,chase::platform::CPU>(N_tiny,N_tiny);
	exact_eigsl_H_tiny = new chase::matrix::Matrix<T,chase::platform::CPU>(N_tiny,1);
        
	ritzv_tiny.resize(M_tiny * numvec_tiny);
        ritzV_tiny.resize(M_tiny * M_tiny);
        Tau_tiny.resize(M_tiny * M_tiny * numvec_tiny);

        for(auto i = 0; i < N_tiny * M_tiny; i++)
        {
            V_tiny[i] =  getRandomT<T>([&]() { return d(gen); });
        }
    }

    void TearDown() override {
    	delete H;
	delete exact_eigsl_H;

	delete H_tiny;
	delete exact_eigsl_H_tiny;
    }

    //Standard variable sets
    std::size_t N = 200, M = 200, numvec = 1; 
    std::vector<T> V; 
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;
    chase::matrix::Matrix<T> * exact_eigsl_H; 
    chase::matrix::QuasiHermitianMatrix<T> * H;
    
    //Tiny variable sets
    std::size_t N_tiny = 10, M_tiny = 10, numvec_tiny = 1;
    std::vector<T> V_tiny;
    std::vector<chase::Base<T>> ritzv_tiny;
    std::vector<chase::Base<T>> ritzV_tiny;
    std::vector<chase::Base<T>> Tau_tiny;
    chase::matrix::Matrix<T> * exact_eigsl_H_tiny; 
    chase::matrix::QuasiHermitianMatrix<T> * H_tiny;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(QuasiHermitianLanczosCPUTest, TestTypes);

TYPED_TEST(QuasiHermitianLanczosCPUTest, tinyQuasiHermitianSimplefiedLanczos) {
    using T = TypeParam;

    this->H_tiny->readFromBinaryFile(GetBSE_TinyMatrix<T>());
    this->exact_eigsl_H_tiny->readFromBinaryFile(GetBSE_TinyEigs<T>());

    chase::Base<T>  upperb;

    chase::linalg::internal::cpu::lanczos<T>(this->M_tiny, this->H_tiny, this->V_tiny.data(), this->N_tiny, &upperb);
    
    chase::Base<T> actual_upperb = (chase::Base<T>)std::real(this->exact_eigsl_H_tiny->data()[this->N_tiny-1]);
    EXPECT_TRUE(upperb >=  actual_upperb ||
                std::abs(upperb - actual_upperb) / std::abs(actual_upperb) <= 1e-2)
        << "Value " << upperb << " is below the acceptable threshold of "
        << actual_upperb << "\n" ;

    EXPECT_LT(upperb,5*(chase::Base<T>)std::real(this->exact_eigsl_H_tiny->data()[this->N_tiny-1]));

}

TYPED_TEST(QuasiHermitianLanczosCPUTest, tinyQuasiHermitianLanczos) {
    using T = TypeParam;

    this->H_tiny->readFromBinaryFile(GetBSE_TinyMatrix<T>());
    this->exact_eigsl_H_tiny->readFromBinaryFile(GetBSE_TinyEigs<T>());

    chase::Base<T>  upperb;

    chase::linalg::internal::cpu::lanczos<T>(this->M_tiny, this->numvec_tiny, this->H_tiny, this->V_tiny.data(), this->N_tiny,
                			  &upperb,this->ritzv_tiny.data(),this->Tau_tiny.data(),this->ritzV_tiny.data());

    chase::Base<T> diff_min = std::norm(this->ritzv_tiny[0]        - std::real(this->exact_eigsl_H_tiny->data()[0])); 
    chase::Base<T> diff_max = std::norm(this->ritzv_tiny[this->M_tiny-1] - std::real(this->exact_eigsl_H_tiny->data()[this->N_tiny-1])); 

    for(auto i = 0; i < this->M_tiny; i++){
	std::cout << this->ritzv_tiny[i] << " " << std::endl;
    }

    EXPECT_LT(diff_min,1e3*MachineEpsilon<chase::Base<T>>::value());
    EXPECT_LT(diff_max,1e3*MachineEpsilon<chase::Base<T>>::value());
}

TYPED_TEST(QuasiHermitianLanczosCPUTest, QuasiHermitianSimplefiedLanczos) {
    using T = TypeParam;

    this->H->readFromBinaryFile(GetBSE_Matrix<T>());
    this->exact_eigsl_H->readFromBinaryFile(GetBSE_Eigs<T>());

    chase::Base<T>  upperb;

    chase::linalg::internal::cpu::lanczos<T>(this->M, this->H, this->V.data(), this->N, &upperb);
    
    chase::Base<T> actual_upperb = (chase::Base<T>)std::real(this->exact_eigsl_H->data()[this->N-1]);
    EXPECT_TRUE(upperb >=  actual_upperb || 
                std::abs(upperb - actual_upperb) / std::abs(actual_upperb) <= 1e-2)
        << "Value " << upperb << " is below the acceptable threshold of " 
        << actual_upperb << "\n" ;

    EXPECT_LT(upperb,5*(chase::Base<T>)std::real(this->exact_eigsl_H->data()[this->N-1]));
}

TYPED_TEST(QuasiHermitianLanczosCPUTest, QuasiHermitianLanczos) {
    using T = TypeParam;
    
    this->H->readFromBinaryFile(GetBSE_Matrix<T>());
    this->exact_eigsl_H->readFromBinaryFile(GetBSE_Eigs<T>());

    chase::Base<T>  upperb;

    chase::linalg::internal::cpu::lanczos<T>(this->M, this->numvec, this->H, this->V.data(), this->N,
                			  &upperb,this->ritzv.data(),this->Tau.data(),this->ritzV.data());

    chase::Base<T> diff_min = std::norm(this->ritzv[0]   - std::real(this->exact_eigsl_H->data()[0])); 
    chase::Base<T> diff_max = std::norm(this->ritzv[this->M-1] - std::real(this->exact_eigsl_H->data()[this->N-1])); 

    EXPECT_LT(diff_min,1e3*MachineEpsilon<chase::Base<T>>::value());
    EXPECT_LT(diff_max,1e3*MachineEpsilon<chase::Base<T>>::value());
}
