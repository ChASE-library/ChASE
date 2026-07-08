// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/serial/cuda/lanczos.hpp"
#include "tests/linalg/internal/lanczos_reference.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <random>

template <typename T>
class LaczosGPUTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));

        H.resize(N * N);
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

    void TearDown() override
    {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
    }

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
    chase::Base<T> lambda_min;
    chase::Base<T> lambda_max;
    cublasHandle_t cublasH_;
    cudaStream_t stream_;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LaczosGPUTest, TestTypes);

TYPED_TEST(LaczosGPUTest, mlanczos)
{
    using T = TypeParam; // Get the current type
    chase::Base<T> upperb;
    chase::matrix::Matrix<T, chase::platform::GPU>* Hmat =
        new chase::matrix::Matrix<T, chase::platform::GPU>(
            this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->N, this->M,
                                                       this->N, this->V.data());

    chase::linalg::internal::cuda::lanczos(
        this->cublasH_, this->M, this->numvec, Hmat, Vec, &upperb,
        this->ritzv.data(), this->Tau.data(), this->ritzV.data());

    chase::tests::lanczos::expectRitzValuesInsideReferenceSpectrum(
        this->ritzv, this->M, this->numvec, this->lambda_min,
        this->lambda_max);
    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}

TYPED_TEST(LaczosGPUTest, lanczos)
{
    using T = TypeParam; // Get the current type
    chase::Base<T> upperb;
    chase::matrix::Matrix<T, chase::platform::GPU>* Hmat =
        new chase::matrix::Matrix<T, chase::platform::GPU>(
            this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->N, 1, this->N,
                                                       this->V.data());

    chase::linalg::internal::cuda::lanczos(this->cublasH_, this->M, Hmat, Vec,
                                           &upperb);

    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}
