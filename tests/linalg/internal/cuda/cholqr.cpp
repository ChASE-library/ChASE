// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/cuda/cholqr.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>

template <typename T>
class CholQRGPUTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        V.resize(m * n);
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
        CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH_, stream_));
    }

    void TearDown() override
    {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
        if (cusolverH_)
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH_));
    }

    std::size_t m = 100;
    std::size_t n = 50;
    std::vector<T> V;

    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;
    cudaStream_t stream_;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(CholQRGPUTest, TestTypes);

TYPED_TEST(CholQRGPUTest, cholQR1)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_10.bin", 0, this->m,
                 this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    int info = chase::linalg::internal::cuda::cholQR1<T>(this->cublasH_,
                                                         this->cusolverH_, Vec);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CholQRGPUTest, cholQR1BadlyCond)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_1e4.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    int info = chase::linalg::internal::cuda::cholQR1<T>(this->cublasH_,
                                                         this->cusolverH_, Vec);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    EXPECT_GT(orth, machineEpsilon);
    EXPECT_LT(orth, 1.0);
}

TYPED_TEST(CholQRGPUTest, cholQR1IllCond)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    int info = chase::linalg::internal::cuda::cholQR1<T>(this->cublasH_,
                                                         this->cusolverH_, Vec);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CholQRGPUTest, cholQR2)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_1e4.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    int info = chase::linalg::internal::cuda::cholQR2<T>(this->cublasH_,
                                                         this->cusolverH_, Vec);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CholQRGPUTest, cholQR2IllCond)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    int info = chase::linalg::internal::cuda::cholQR2<T>(this->cublasH_,
                                                         this->cusolverH_, Vec);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CholQRGPUTest, scholQR)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    int info = chase::linalg::internal::cuda::shiftedcholQR2<T>(
        this->cublasH_, this->cusolverH_, Vec);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);
}

TYPED_TEST(CholQRGPUTest, householderQR)
{
    using T = TypeParam; // Get the current type

    auto machineEpsilon = MachineEpsilon<T>::value();

    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);

    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());

    T* d_tau;
    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tau, this->n * sizeof(T)));

    chase::linalg::internal::cuda::houseHoulderQR(this->cusolverH_, Vec, d_tau,
                                                  devInfo);
    Vec.D2H();
    auto orth = orthogonality<T>(this->m, this->n, Vec.cpu_data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);

    cudaFree(devInfo);
    cudaFree(d_tau);
}