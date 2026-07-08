// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/serial/impl/residuals.hpp"

#include "external/cpu/blaspp/blaspp.hpp"
#include "external/cpu/lapackpp/lapackpp.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <cmath>
#include <complex>
#include <cstring>
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
class ResidsHIPTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        H.resize(N * N);
        evals.resize(N);
        evecs.resize(N * N);
        resids.resize(N);
        ASSERT_HIP_SUCCESS(
            hipMalloc(&d_resids, N * sizeof(chase::Base<T>)));
        ASSERT_HIP_SUCCESS(hipMalloc(&d_ritzv, N * sizeof(chase::Base<T>)));
        ASSERT_ROCBLAS_SUCCESS(rocblas_create_handle(&handle_));
    }

    void TearDown() override
    {
        ASSERT_HIP_SUCCESS(hipFree(d_resids));
        ASSERT_HIP_SUCCESS(hipFree(d_ritzv));
        if (handle_)
            ASSERT_ROCBLAS_SUCCESS(rocblas_destroy_handle(handle_));
    }

    std::size_t N = 64;
    std::vector<T> H;
    std::vector<chase::Base<T>> evals;
    std::vector<T> evecs;
    std::vector<chase::Base<T>> resids;
    chase::Base<T>* d_resids = nullptr;
    chase::Base<T>* d_ritzv = nullptr;
    rocblas_handle handle_ = nullptr;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(ResidsHIPTest, TestTypes);

TYPED_TEST(ResidsHIPTest, DiagonalMatrix)
{
    using T = TypeParam;
    const auto machineEpsilon = MachineEpsilon<T>::value();

    for (std::size_t i = 0; i < this->N; ++i)
    {
        this->H[i * this->N + i] = T(i + 1);
    }

    for (std::size_t i = 0; i < this->N; ++i)
    {
        this->evals[i] = chase::Base<T>(i + 1);
        this->evecs[i + this->N * i] = T(1.0);
    }

    ASSERT_HIP_SUCCESS(hipMemcpy(this->d_ritzv, this->evals.data(),
                                 this->N * sizeof(chase::Base<T>),
                                 hipMemcpyHostToDevice));
    auto* Hmat = new chase::matrix::Matrix<T, chase::platform::GPU>(
        this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> V1(this->N, this->N, this->N,
                                                      this->evecs.data());
    chase::linalg::internal::gpu_serial::impl::residuals<
        chase::linalg::internal::gpu_backend::hip_backend>(
        this->handle_, Hmat, V1, this->d_ritzv, this->d_resids, 0, this->N);
    ASSERT_HIP_SUCCESS(hipMemcpy(this->resids.data(), this->d_resids,
                                 this->N * sizeof(chase::Base<T>),
                                 hipMemcpyDeviceToHost));

    for (std::size_t i = 0; i < this->N; ++i)
    {
        EXPECT_NEAR(this->resids[i], machineEpsilon, machineEpsilon * 10);
    }
    delete Hmat;
}

TYPED_TEST(ResidsHIPTest, DenseMatrix)
{
    using T = TypeParam;
    const auto machineEpsilon = MachineEpsilon<T>::value();
    const T one = T(1.0);
    const T zero = T(0.0);
    const std::size_t offset = 2;
    const std::size_t subSize = 10;

    for (std::size_t i = 0; i < this->N; ++i)
    {
        this->H[i * this->N + i] = T(0.1 * static_cast<double>(i) + 0.1);
    }

    std::mt19937 gen(1337);
    std::normal_distribution<> d;
    std::vector<T> V(this->N * this->N);

    for (std::size_t i = 0; i < this->N * this->N; ++i)
    {
        V[i] = getRandomT<T>([&]() { return d(gen); });
    }

    std::vector<T> tau(this->N);
    std::vector<T> H_tmp(this->N * this->N);

    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, this->N, this->N,
                                     V.data(), this->N, tau.data());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, this->N, this->N, this->N,
                                   V.data(), this->N, tau.data());

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                  this->N, this->N, this->N, &one,
                                  this->H.data(), this->N, V.data(), this->N,
                                  &zero, H_tmp.data(), this->N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                  this->N, this->N, this->N, &one, V.data(),
                                  this->N, H_tmp.data(), this->N, &zero,
                                  this->evecs.data(), this->N);

    chase::linalg::lapackpp::t_lacpy('A', this->N, this->N, this->evecs.data(),
                                     this->N, this->H.data(), this->N);

    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'V', 'U', this->N,
                                     this->evecs.data(), this->N,
                                     this->evals.data());

    ASSERT_HIP_SUCCESS(hipMemcpy(this->d_ritzv, this->evals.data(),
                                 this->N * sizeof(chase::Base<T>),
                                 hipMemcpyHostToDevice));
    auto* Hmat = new chase::matrix::Matrix<T, chase::platform::GPU>(
        this->N, this->N, this->N, this->H.data());
    chase::matrix::Matrix<T, chase::platform::GPU> V1(this->N, this->N, this->N,
                                                      this->evecs.data());
    chase::linalg::internal::gpu_serial::impl::residuals<
        chase::linalg::internal::gpu_backend::hip_backend>(
        this->handle_, Hmat, V1, this->d_ritzv, this->d_resids, offset,
        subSize);
    ASSERT_HIP_SUCCESS(
        hipMemcpy(this->resids.data() + offset, this->d_resids + offset,
                  subSize * sizeof(chase::Base<T>), hipMemcpyDeviceToHost));

    for (std::size_t i = offset; i < offset + subSize; ++i)
    {
        EXPECT_NEAR(this->resids[i], machineEpsilon, machineEpsilon * 1e2);
    }
    delete Hmat;
}
