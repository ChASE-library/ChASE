// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/serial/hip/lanczos.hpp"

#include "tests/linalg/internal/lanczos_reference.hpp"
#include "tests/linalg/internal/utils.hpp"
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
class LanczosHIPTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_ROCBLAS_SUCCESS(rocblas_create_handle(&handle_));
        ASSERT_HIP_SUCCESS(hipStreamCreate(&stream_));
        ASSERT_ROCBLAS_SUCCESS(rocblas_set_stream(handle_, stream_));

        H.resize(N * N);
        V.resize(N * M);
        ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);

        std::mt19937 gen(1337);
        std::normal_distribution<> d;

        for (std::size_t i = 0; i < N * M; ++i)
        {
            V[i] = getRandomT<T>([&]() { return d(gen); });
        }

        chase::tests::lanczos::initializeReferenceHermitianMatrix(H.data(), N);
        chase::tests::lanczos::computeReferenceEigenBounds(
            H.data(), N, &lambda_min, &lambda_max);
    }

    void TearDown() override
    {
        if (stream_)
            ASSERT_HIP_SUCCESS(hipStreamDestroy(stream_));
        if (handle_)
            ASSERT_ROCBLAS_SUCCESS(rocblas_destroy_handle(handle_));
    }

    std::size_t M = 10;
    std::size_t numvec = 4;
    std::size_t N = 500;
    std::vector<T> H;
    std::vector<T> V;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;
    chase::Base<T> lambda_min;
    chase::Base<T> lambda_max;
    rocblas_handle handle_ = nullptr;
    hipStream_t stream_ = nullptr;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosHIPTest, TestTypes);

TYPED_TEST(LanczosHIPTest, mlanczos)
{
    using T = TypeParam;
    chase::Base<T> upperb = 0;
    auto* Hmat = new chase::matrix::Matrix<T, chase::platform::GPU>(
        this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->N, this->M,
                                                       this->N, this->V.data());

    chase::linalg::internal::hip::lanczos(
        this->handle_, this->M, this->numvec, Hmat, Vec, &upperb,
        this->ritzv.data(), this->Tau.data(), this->ritzV.data());

    chase::tests::lanczos::expectRitzValuesInsideReferenceSpectrum(
        this->ritzv, this->M, this->numvec, this->lambda_min,
        this->lambda_max);
    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
    delete Hmat;
}

TYPED_TEST(LanczosHIPTest, lanczos)
{
    using T = TypeParam;
    chase::Base<T> upperb = 0;
    auto* Hmat = new chase::matrix::Matrix<T, chase::platform::GPU>(
        this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->N, 1, this->N,
                                                       this->V.data());

    chase::linalg::internal::hip::lanczos(this->handle_, this->M, Hmat, Vec,
                                          &upperb);

    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
    delete Hmat;
}
