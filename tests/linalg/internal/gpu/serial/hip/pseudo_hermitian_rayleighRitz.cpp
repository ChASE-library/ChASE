// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/serial/hip/pseudo_hermitian_rayleighRitz.hpp"

#include "tests/linalg/internal/utils.hpp"
#include <algorithm>
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <random>
#include <rocblas/rocblas.h>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

#define ASSERT_ROCBLAS_SUCCESS(status)                                         \
    ASSERT_EQ((status), rocblas_status_success)

} // namespace

template <typename T>
class PseudoHermitianRayleighRitzHIPTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_ROCBLAS_SUCCESS(rocblas_create_handle(&handle_));
        ASSERT_HIP_SUCCESS(hipStreamCreate(&stream_));
        ASSERT_ROCBLAS_SUCCESS(rocblas_set_stream(handle_, stream_));

        std::mt19937 gen(1337);
        std::normal_distribution<> d;

        Q_buffer.resize(N * nev);
        for (auto i = 0; i < nev; i++)
        {
            Q_buffer.data()[i * (N + 1)] = 1.0;
        }

        H = new chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>(
            N, N);
        H->allocate_cpu_data();
        Q = chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>(
            N, nev, N, Q_buffer.data());
        V = chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>(N,
                                                                          nev);
        ritzv =
            chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(nev, 1);
        exact_eigsl_H = chase::matrix::Matrix<T, chase::platform::CPU>(N, 1);

        ritzv.allocate_cpu_data();

        Q_buffer_tiny.resize(N_tiny * nev_tiny);
        for (auto i = 0; i < nev_tiny; i++)
        {
            Q_buffer_tiny.data()[i * (N_tiny + 1)] = 1.0;
        }

        H_tiny =
            new chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>(
                N_tiny, N_tiny);
        H_tiny->allocate_cpu_data();
        Q_tiny = chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>(
            N_tiny, nev_tiny, N_tiny, Q_buffer_tiny.data());
        V_tiny = chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>(
            N_tiny, nev_tiny);
        ritzv_tiny = chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU>(
            nev_tiny, 1);
        exact_eigsl_H_tiny =
            chase::matrix::Matrix<T, chase::platform::CPU>(N_tiny, 1);
        ritzv_tiny.allocate_cpu_data();
    }

    void TearDown() override
    {
        delete H;
        delete H_tiny;

        if (stream_)
            ASSERT_HIP_SUCCESS(hipStreamDestroy(stream_));
        if (handle_)
            ASSERT_ROCBLAS_SUCCESS(rocblas_destroy_handle(handle_));
    }

    std::size_t k = 100;
    std::size_t N = 2 * k;
    std::size_t nev = N;

    std::vector<T> Q_buffer;

    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU> Q;
    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU> V;
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> ritzv;
    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H;
    chase::matrix::Matrix<T, chase::platform::CPU> exact_eigsl_H;

    std::size_t k_tiny = 5;
    std::size_t N_tiny = 2 * k_tiny;
    std::size_t nev_tiny = N_tiny;

    std::vector<T> Q_buffer_tiny;

    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU> Q_tiny;
    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU> V_tiny;
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> ritzv_tiny;
    chase::matrix::PseudoHermitianMatrix<T, chase::platform::GPU>* H_tiny;
    chase::matrix::Matrix<T, chase::platform::CPU> exact_eigsl_H_tiny;

    rocblas_handle handle_ = nullptr;
    hipStream_t stream_ = nullptr;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHermitianRayleighRitzHIPTest, TestTypes);

TYPED_TEST(PseudoHermitianRayleighRitzHIPTest, PseudoHermitianRayleighRitz)
{
    using T = TypeParam;

    this->H->readFromBinaryFile(GetBSE_Matrix<T>());
    this->H->H2D();

    rocblas_int* devInfo = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&devInfo, sizeof(rocblas_int)));

    chase::linalg::internal::hip::rayleighRitz_v2(
        this->handle_, this->H, this->Q, this->V, this->ritzv, (std::size_t)0,
        this->nev, devInfo);

    this->ritzv.D2H();
    std::sort(this->ritzv.cpu_data(), this->ritzv.cpu_data() + this->nev);

    this->exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    for (auto i = 0; i < this->nev; i++)
    {
        EXPECT_LT(this->ritzv.cpu_data()[i],
                  std::real(this->exact_eigsl_H.data()[i]) +
                      GetErrorTolerance<T>());
        EXPECT_GT(this->ritzv.cpu_data()[i],
                  std::real(this->exact_eigsl_H.data()[i]) -
                      GetErrorTolerance<T>());
    }

    ASSERT_HIP_SUCCESS(hipFree(devInfo));
}

TYPED_TEST(PseudoHermitianRayleighRitzHIPTest, tinyPseudoHermitianRayleighRitz)
{
    using T = TypeParam;

    this->H_tiny->readFromBinaryFile(GetBSE_TinyMatrix<T>());
    this->H_tiny->H2D();

    rocblas_int* devInfo = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&devInfo, sizeof(rocblas_int)));

    chase::linalg::internal::hip::rayleighRitz_v2(
        this->handle_, this->H_tiny, this->Q_tiny, this->V_tiny,
        this->ritzv_tiny, (std::size_t)0, this->nev_tiny, devInfo);

    this->ritzv_tiny.D2H();
    std::sort(this->ritzv_tiny.cpu_data(),
              this->ritzv_tiny.cpu_data() + this->nev_tiny);

    this->exact_eigsl_H_tiny.readFromBinaryFile(GetBSE_TinyEigs<T>());

    for (auto i = 0; i < this->nev_tiny; i++)
    {
        EXPECT_LT(this->ritzv_tiny.cpu_data()[i],
                  std::real(this->exact_eigsl_H_tiny.data()[i]) +
                      GetErrorTolerance<T>());
        EXPECT_GT(this->ritzv_tiny.cpu_data()[i],
                  std::real(this->exact_eigsl_H_tiny.data()[i]) -
                      GetErrorTolerance<T>());
    }

    ASSERT_HIP_SUCCESS(hipFree(devInfo));
}
