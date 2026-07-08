// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/serial/hip/rayleighRitz.hpp"

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
class RayleighRitzHIPTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        H.resize(N * N);
        Q.resize(N * n);
        W.resize(N * n);
        evals.resize(N);
        ASSERT_ROCBLAS_SUCCESS(rocblas_create_handle(&handle_));
    }

    void TearDown() override
    {
        if (handle_)
            ASSERT_ROCBLAS_SUCCESS(rocblas_destroy_handle(handle_));
    }

    std::size_t N = 50;
    std::size_t n = 10;
    std::vector<T> H;
    std::vector<T> Q;
    std::vector<T> W;
    std::vector<chase::Base<T>> evals;
    rocblas_handle handle_ = nullptr;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(RayleighRitzHIPTest, TestTypes);

TYPED_TEST(RayleighRitzHIPTest, eigenpairs)
{
    using T = TypeParam;
    const auto machineEpsilon = MachineEpsilon<T>::value();
    const T one = T(1.0);
    const T zero = T(0.0);
    const std::size_t offset = 2;
    const std::size_t subSize = 5;

    for (std::size_t i = 0; i < this->N; ++i)
    {
        this->H[i * this->N + i] = T(0.1 * static_cast<double>(i) + 0.1);
    }

    std::mt19937 gen(1337);
    std::normal_distribution<> d;
    std::vector<T> V(this->N * this->N);
    std::vector<T> H2(this->N * this->N);

    for (std::size_t i = 0; i < this->N * this->N; ++i)
    {
        V[i] = getRandomT<T>([&]() { return d(gen); });
    }

    std::vector<T> tau(this->N);
    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, this->N, this->N,
                                     V.data(), this->N, tau.data());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, this->N, this->N, this->N,
                                   V.data(), this->N, tau.data());

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                  this->N, this->N, this->N, &one,
                                  this->H.data(), this->N, V.data(), this->N,
                                  &zero, H2.data(), this->N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                                  this->N, this->N, this->N, &one, V.data(),
                                  this->N, H2.data(), this->N, &zero,
                                  this->H.data(), this->N);

    std::memcpy(H2.data(), this->H.data(), sizeof(T) * this->N * this->N);

    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'V', 'U', this->N,
                                     this->H.data(), this->N,
                                     this->evals.data());

    auto* Hmat = new chase::matrix::Matrix<T, chase::platform::GPU>(
        this->N, this->N, this->N, H2.data());
    chase::matrix::Matrix<T, chase::platform::GPU> V1(this->N, this->n);
    chase::matrix::Matrix<T, chase::platform::GPU> V2(this->N, this->n);
    std::vector<chase::Base<T>> ritzv_data(this->n);
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> ritzv(
        this->n, 1, this->n, ritzv_data.data());

    ASSERT_HIP_SUCCESS(hipMemcpy(V1.data(), this->H.data(),
                                 this->N * this->n * sizeof(T),
                                 hipMemcpyHostToDevice));
    ASSERT_HIP_SUCCESS(hipMemcpy(ritzv.data(), this->evals.data(),
                                 this->n * sizeof(chase::Base<T>),
                                 hipMemcpyHostToDevice));

    rocblas_int* devInfo = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&devInfo, sizeof(rocblas_int)));

    chase::linalg::internal::hip::rayleighRitz(
        this->handle_, Hmat, V1, V2, ritzv, offset, subSize, devInfo);

    ritzv.D2H();
    for (std::size_t i = offset; i < offset + subSize; ++i)
    {
        EXPECT_NEAR(static_cast<double>(ritzv.cpu_data()[i]),
                    static_cast<double>(this->evals[i]),
                    100 * machineEpsilon);
    }

    ASSERT_HIP_SUCCESS(hipFree(devInfo));
    delete Hmat;
}
