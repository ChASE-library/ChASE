// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <random>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda/rayleighRitz.hpp"
#include "tests/linalg/internal/utils.hpp"

template <typename T>
class rayleighRitzGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        H.resize(N * N);
        Q.resize(N * n);
        W.resize(N * n);
        evals.resize(N);
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));   
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
        CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH_, stream_));

    }

    void TearDown() override {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
        if (cusolverH_)
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH_));        
    }

    std::size_t N = 50;
    std::size_t n = 10;
    std::vector<T> H;
    std::vector<T> Q;
    std::vector<T> W;
    std::vector<chase::Base<T>> evals;
    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;
    cudaStream_t stream_;

};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(rayleighRitzGPUTest, TestTypes);

TYPED_TEST(rayleighRitzGPUTest, eigenpairs) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    T One = T(1.0);
    T Zero = T(0.0);
    std::size_t offset = 2;
    std::size_t subSize = 5;
    
    for (int i = 0; i < this->N; ++i) {
        this->H[i * this->N + i] = T(0.1 * i + 0.1);
    }

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;
    std::vector<T> V(this->N * this->N);
    std::vector<T> H2(this->N * this->N);

    for(auto i = 0; i < this->N * this->N; i++)
    {
        V[i] =  getRandomT<T>([&]() { return d(gen); });
    }

    std::unique_ptr<T[]> tau(new T[this->N]);

    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, this->N, this->N, V.data(), this->N, tau.get());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, this->N, this->N, this->N, V.data(), this->N, tau.get());

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, this->N, this->N, this->N,
               &One, this->H.data(), this->N, V.data(), this->N, &Zero,
               H2.data(), this->N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasConjTrans, this->N, this->N, this->N,
               &One, V.data(), this->N, H2.data(), this->N, &Zero,
               this->H.data(), this->N);

    std::memcpy(H2.data(), this->H.data(), sizeof(T) * this->N * this->N);

    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'V', 'U', this->N,
                    this->H.data(), this->N, this->evals.data());
    
    chase::matrix::Matrix<T, chase::platform::GPU> * Hmat = new chase::matrix::Matrix<T, chase::platform::GPU>(this->N, this->N, this->N, H2.data());
    chase::matrix::Matrix<T, chase::platform::GPU> V1(this->N, this->n);
    chase::matrix::Matrix<T, chase::platform::GPU> V2(this->N, this->n);
    std::vector<chase::Base<T>> ritzv_data(this->n);
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> ritzv(this->n, 1, this->n, ritzv_data.data());

    CHECK_CUDA_ERROR(cudaMemcpy(V1.data(), this->H.data(), this->N * this->n * sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(ritzv.data(), this->evals.data(), this->n * sizeof(chase::Base<T>), cudaMemcpyHostToDevice));
    
    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));

    chase::linalg::internal::cuda::rayleighRitz(this->cublasH_,
                                                 this->cusolverH_,
                                                 Hmat,
                                                 V1,
                                                 V2,
                                                 ritzv,
                                                 offset,
                                                 subSize,
                                                 devInfo);

    ritzv.D2H();
    //check the eigenvalues
    for(auto i = offset; i < offset + subSize; i++)
    {
        EXPECT_NEAR(ritzv.cpu_data()[i], this->evals[i], 100 * machineEpsilon);
    }
    cudaFree(devInfo);  
}
