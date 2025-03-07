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

TYPED_TEST(LaczosCPUTest, tinyQuasiHermitianLanczos) {
    using T = std::complex<double>;

    std::size_t nHmat = 10; 
    
    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    auto Hmat = new chase::matrix::QuasiHermitianMatrix<T>(nHmat,nHmat);
    Hmat->readFromBinaryFile(GetBSEPath<T>()+"/tiny_random_pseudohermitian.bin");
    
    auto eigsl_H = new chase::matrix::Matrix<T>(nHmat,1); 
    eigsl_H->readFromBinaryFile(GetBSEPath<T>()+"/eigenvalues_tiny_random_pseudohermitian.bin");

    auto V_lanczos = new chase::matrix::Matrix<T>(nHmat,nHmat);
    for(auto i = 0; i < nHmat* nHmat; i++) V_lanczos->data()[i] = getRandomT<T>([&]() { return d(gen); }); 

    chase::Base<T>  upperb, error_norm, alpha = -chase::Base<T>(1.0);
    chase::Base<T>* Theta = new chase::Base<T>[nHmat]();
    chase::Base<T>* Tau   = new chase::Base<T>[nHmat]();
    chase::Base<T>* ritzV = new chase::Base<T>[nHmat*nHmat]();
    chase::Base<T>* real_eigsl_H = new chase::Base<T>[nHmat]();

    for(auto i = 0; i < nHmat; i++) real_eigsl_H[i] = std::real(eigsl_H->data()[i]);

    chase::linalg::internal::cpu::quasi_hermitian_lanczos(nHmat, 1, Hmat, V_lanczos->data(), nHmat,
                			  		  &upperb,Theta,Tau,ritzV);
    
    blaspp::t_axpy(nHmat, &alpha, real_eigsl_H, 1, Theta, 1);

    error_norm = blaspp::t_nrm2(nHmat, Theta, 1);

    error_norm /= nHmat;
    
    delete Hmat;
    delete eigsl_H;
    delete V_lanczos;
    delete [] Theta;
    delete [] Tau;
    delete [] ritzV;
    delete [] real_eigsl_H;
    
    EXPECT_LT(error_norm,1e3*MachineEpsilon<chase::Base<T>>::value());
}

TYPED_TEST(LaczosCPUTest, tinyFlipedQuasiHermitianLanczos) {
    using T = std::complex<double>;

    std::size_t nHmat = 10; 
    
    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    auto Hmat = new chase::matrix::Matrix<T>(nHmat,nHmat);
    Hmat->readFromBinaryFile(GetBSEPath<T>()+"/tiny_random_pseudohermitian.bin");
    
    chase::linalg::internal::cpu::flipLowerHalfMatrixSign(nHmat,nHmat,Hmat->data(),nHmat);

    auto eigsl_SH = new chase::matrix::Matrix<T>(nHmat,1); 
    eigsl_SH->readFromBinaryFile(GetBSEPath<T>()+"/eigenvalues_SH_tiny_random_pseudohermitian.bin");

    auto V_lanczos = new chase::matrix::Matrix<T>(nHmat,nHmat);
    for(auto i = 0; i < nHmat* nHmat; i++) V_lanczos->data()[i] = getRandomT<T>([&]() { return d(gen); }); 

    chase::Base<T>  upperb, error_norm, alpha = -chase::Base<T>(1.0);
    chase::Base<T>* Theta = new chase::Base<T>[nHmat]();
    chase::Base<T>* Tau   = new chase::Base<T>[nHmat]();
    chase::Base<T>* ritzV = new chase::Base<T>[nHmat*nHmat]();
    chase::Base<T>* real_eigsl_SH = new chase::Base<T>[nHmat]();

    for(auto i = 0; i < nHmat; i++) real_eigsl_SH[i] = std::real(eigsl_SH->data()[i]);
    
    chase::linalg::internal::cpu::lanczos(nHmat,1,Hmat->rows(),Hmat->data(),Hmat->ld(),V_lanczos->data(),V_lanczos->ld(),&upperb,Theta,Tau,ritzV);

//    for(auto i = 0; i < 10; i++){
//	    std::cout << eigsl_SH->data()[i] << " : " << real_eigsl_SH[i] << " vs. " << Theta[i] << std::endl;
//    }

    blaspp::t_axpy(nHmat, &alpha, real_eigsl_SH, 1, Theta, 1);

    error_norm = blaspp::t_nrm2(nHmat, Theta, 1);

    error_norm /= nHmat;
    
    delete Hmat;
    delete eigsl_SH;
    delete V_lanczos;
    delete [] Theta;
    delete [] Tau;
    delete [] ritzV;
    delete [] real_eigsl_SH;
    
    EXPECT_LT(error_norm,1e3*MachineEpsilon<chase::Base<T>>::value());
}

TYPED_TEST(LaczosCPUTest, quasiHermitianLanczos) {
    using T = std::complex<double>;

    std::size_t nHmat = 200; 
    
    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    auto Hmat = new chase::matrix::QuasiHermitianMatrix<T>(nHmat,nHmat);
    Hmat->readFromBinaryFile(GetBSEPath<T>()+"/random_pseudohermitian.bin");
    
    auto eigsl_H = new chase::matrix::Matrix<T>(nHmat,1); 
    eigsl_H->readFromBinaryFile(GetBSEPath<T>()+"/eigenvalues_random_pseudohermitian.bin");

    auto V_lanczos = new chase::matrix::Matrix<T>(nHmat,nHmat);
    for(auto i = 0; i < nHmat* nHmat; i++) V_lanczos->data()[i] = getRandomT<T>([&]() { return d(gen); }); 

    chase::Base<T>  upperb, error_norm, alpha = -chase::Base<T>(1.0);
    chase::Base<T>* Theta = new chase::Base<T>[nHmat]();
    chase::Base<T>* Tau   = new chase::Base<T>[nHmat]();
    chase::Base<T>* ritzV = new chase::Base<T>[nHmat*nHmat]();
    chase::Base<T>* real_eigsl_H = new chase::Base<T>[nHmat]();

    for(auto i = 0; i < nHmat; i++) real_eigsl_H[i] = std::real(eigsl_H->data()[i]);

    chase::linalg::internal::cpu::quasi_hermitian_lanczos(nHmat, 1, Hmat, V_lanczos->data(), nHmat,
                			  		  &upperb,Theta,Tau,ritzV);
    
    blaspp::t_axpy(nHmat, &alpha, real_eigsl_H, 1, Theta, 1);

    error_norm = blaspp::t_nrm2(nHmat, Theta, 1);

    error_norm /= nHmat;
    
    delete Hmat;
    delete eigsl_H;
    delete V_lanczos;
    delete [] Theta;
    delete [] Tau;
    delete [] ritzV;
    delete [] real_eigsl_H;
    
    EXPECT_LT(error_norm,1e3*MachineEpsilon<chase::Base<T>>::value());
}

TYPED_TEST(LaczosCPUTest, flipedQuasiHermitianLanczos) {
    using T = std::complex<double>;

    std::size_t nHmat = 200; 
    
    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    auto Hmat = new chase::matrix::Matrix<T>(nHmat,nHmat);
    Hmat->readFromBinaryFile(GetBSEPath<T>()+"/random_pseudohermitian.bin");
    
    chase::linalg::internal::cpu::flipLowerHalfMatrixSign(nHmat,nHmat,Hmat->data(),nHmat);

    auto eigsl_SH = new chase::matrix::Matrix<chase::Base<T>>(nHmat,1); 
    eigsl_SH->readFromBinaryFile(GetBSEPath<T>()+"/eigenvalues_SH_random_pseudohermitian.bin");

    auto V_lanczos = new chase::matrix::Matrix<T>(nHmat,nHmat);
    for(auto i = 0; i < nHmat* nHmat; i++) V_lanczos->data()[i] = getRandomT<T>([&]() { return d(gen); }); 

    chase::Base<T>  upperb, error_norm, alpha = -chase::Base<T>(1.0);
    chase::Base<T>* Theta = new chase::Base<T>[nHmat]();
    chase::Base<T>* Tau   = new chase::Base<T>[nHmat]();
    chase::Base<T>* ritzV = new chase::Base<T>[nHmat*nHmat]();
    chase::Base<T>* real_eigsl_SH = new chase::Base<T>[nHmat]();

    for(auto i = 0; i < nHmat; i++) real_eigsl_SH[i] = std::real(eigsl_SH->data()[i]);
    
    chase::linalg::internal::cpu::lanczos(nHmat,1,Hmat->rows(),Hmat->data(),Hmat->ld(),V_lanczos->data(),V_lanczos->ld(),&upperb,Theta,Tau,ritzV);

//    for(auto i = 0; i < 10; i++){
//	    std::cout << eigsl_SH->data()[i] << " : " << real_eigsl_SH[i] << " vs. " << Theta[i] << std::endl;
//    }
    
//    for(auto i = nHmat-10; i < nHmat; i++){
//	    std::cout << eigsl_SH->data()[i] << " : " << real_eigsl_SH[i] << " vs. " << Theta[i] << std::endl;
//    }

    blaspp::t_axpy(nHmat, &alpha, real_eigsl_SH, 1, Theta, 1);

    error_norm = blaspp::t_nrm2(nHmat, Theta, 1);

    error_norm /= nHmat;
    
    delete Hmat;
    delete eigsl_SH;
    delete V_lanczos;
    delete [] Theta;
    delete [] Tau;
    delete [] ritzV;
    delete [] real_eigsl_SH;
    
    EXPECT_LT(error_norm,1e3*MachineEpsilon<chase::Base<T>>::value());
}
