// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "external/blaspp/blaspp.hpp"
#include "linalg/internal/cuda/rayleighRitz.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "linalg/matrix/matrix.hpp"

template <typename T>
class PseudoHermitianRayleighRitzGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
	CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));        
	CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));
	CHECK_CUSOLVER_ERROR(cusolverDnCreateParams(&params_));

        std::mt19937 gen(1337.0);
        std::normal_distribution<> d;

        //Standard variables init
	Q_buffer.resize(N * nev);
        for(auto i = 0; i < nev; i++)
        {
	    Q_buffer.data()[i*(N+1)] = 1.0;
        }

	H = new chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU>(N,N);
	H->allocate_cpu_data();
	Q = chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU>(N,nev,N,Q_buffer.data());
	V = chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU>(N,nev);
    	ritzv = chase::matrix::Matrix<chase::Base<T>,chase::platform::GPU>(nev,1); 
	exact_eigsl_H = chase::matrix::Matrix<T,chase::platform::CPU>(N,1);

    ritzv.allocate_cpu_data();

        //Tiny variables init	
	Q_buffer_tiny.resize(N_tiny * nev_tiny);
        for(auto i = 0; i < nev_tiny; i++)
        {
	    Q_buffer_tiny.data()[i*(N_tiny+1)] = 1.0;
        }

	H_tiny = new chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU>(N_tiny,N_tiny);
	H_tiny->allocate_cpu_data();
	Q_tiny = chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU>(N_tiny,nev_tiny,N_tiny,Q_buffer_tiny.data());
	V_tiny = chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU>(N_tiny,nev_tiny);
    	ritzv_tiny = chase::matrix::Matrix<chase::Base<T>,chase::platform::GPU>(nev,1); 
	exact_eigsl_H_tiny = chase::matrix::Matrix<T,chase::platform::CPU>(N_tiny,1);
    ritzv_tiny.allocate_cpu_data();
    }

    void TearDown() override {}

    //Standard variable sets
    std::size_t k = 100;
    std::size_t N = 2*k;
    std::size_t nev = N;

    std::vector<T> Q_buffer;

    chase::matrix::Matrix<T,chase::platform::GPU> Q; 
    chase::matrix::Matrix<T,chase::platform::GPU> V; 
    chase::matrix::Matrix<chase::Base<T>,chase::platform::GPU> ritzv; 
    chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU> * H;
    chase::matrix::Matrix<T,chase::platform::CPU> exact_eigsl_H; 
    
    //Tiny variable sets
    std::size_t k_tiny = 5;
    std::size_t N_tiny = 2*k_tiny;
    std::size_t nev_tiny = N_tiny;

    std::vector<T> Q_buffer_tiny;
 
    chase::matrix::Matrix<T,chase::platform::GPU> Q_tiny; 
    chase::matrix::Matrix<T,chase::platform::GPU> V_tiny; 
    chase::matrix::Matrix<chase::Base<T>,chase::platform::GPU> ritzv_tiny; 
    chase::matrix::PseudoHermitianMatrix<T,chase::platform::GPU> * H_tiny;
    chase::matrix::Matrix<T,chase::platform::CPU> exact_eigsl_H_tiny; 

    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;
    cudaStream_t stream_;
    cusolverDnParams_t params_;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHermitianRayleighRitzGPUTest, TestTypes);

TYPED_TEST(PseudoHermitianRayleighRitzGPUTest, PseudoHermitianRayleighRitz) {
    using T = TypeParam;

    this->H->readFromBinaryFile(GetBSE_Matrix<T>());
    this->H->H2D();

    int info = 0;

    chase::linalg::internal::cuda::rayleighRitz(this->cublasH_,
		    				this->cusolverH_,
						this->params_,
						this->H,
						this->Q,
						this->V,
						this->ritzv,
						(std::size_t)0,
						this->nev,
						&info);
    
    this->ritzv.D2H();

    this->exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    for(auto i = 0; i < this->nev; i++)
    {
	std::cout << this->ritzv.cpu_data()[i] << "vs." << this->exact_eigsl_H.data()[i] << std::endl;
    	EXPECT_LT(this->ritzv.cpu_data()[i],std::real(this->exact_eigsl_H.data()[i]) + GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    	EXPECT_GT(this->ritzv.cpu_data()[i],std::real(this->exact_eigsl_H.data()[i]) - GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    }
}

TYPED_TEST(PseudoHermitianRayleighRitzGPUTest, tinyPseudoHermitianRayleighRitz) {
    using T = TypeParam;

    this->H_tiny->readFromBinaryFile(GetBSE_TinyMatrix<T>());
    this->H_tiny->H2D();

    int info = 0;

    chase::linalg::internal::cuda::rayleighRitz(this->cublasH_,
		    				this->cusolverH_,
						this->params_,
						this->H_tiny,
						this->Q_tiny,
						this->V_tiny,
						this->ritzv_tiny,
						(std::size_t)0,
						this->nev_tiny,
						&info);
    
    this->ritzv_tiny.D2H();

    this->exact_eigsl_H_tiny.readFromBinaryFile(GetBSE_TinyEigs<T>());

    for(auto i = 0; i < this->nev_tiny; i++)
    {
	std::cout << this->ritzv_tiny.cpu_data()[i] << "vs." << this->exact_eigsl_H_tiny.data()[i] << std::endl;
    	EXPECT_LT(this->ritzv_tiny.cpu_data()[i],std::real(this->exact_eigsl_H_tiny.data()[i]) + GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    	EXPECT_GT(this->ritzv_tiny.cpu_data()[i],std::real(this->exact_eigsl_H_tiny.data()[i]) - GetErrorTolerance<T>());//MachineEpsilon<chase::Base<T>>::value());
    }
}
