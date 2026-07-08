// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/solver.hpp"
#include "linalg/internal/gpu/backend/workspace_plan.hpp"
#include "linalg/internal/gpu/backend/types.hpp"
#include "linalg/internal/gpu/serial/hip/cholqr.hpp"
#include "tests/linalg/internal/cpu/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <algorithm>

namespace
{

#define ASSERT_HIP_SUCCESS(status)                                             \
    ASSERT_EQ((status), hipSuccess) << hipGetErrorString(status)

#define ASSERT_ROCBLAS_SUCCESS(status)                                         \
    ASSERT_EQ((status), rocblas_status_success)

} // namespace

template <typename T>
class CholQRHIPTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        V.resize(m * n);
        ASSERT_ROCBLAS_SUCCESS(rocblas_create_handle(&blas_handle_));
        ASSERT_ROCBLAS_SUCCESS(rocblas_create_handle(&solver_handle_));
        ASSERT_HIP_SUCCESS(hipStreamCreate(&stream_));
        ASSERT_ROCBLAS_SUCCESS(rocblas_set_stream(blas_handle_, stream_));
        ASSERT_ROCBLAS_SUCCESS(rocblas_set_stream(solver_handle_, stream_));
        ASSERT_HIP_SUCCESS(hipMalloc(&devInfo_, sizeof(int)));
    }

    void TearDown() override
    {
        if (devInfo_)
            ASSERT_HIP_SUCCESS(hipFree(devInfo_));
        if (stream_)
            ASSERT_HIP_SUCCESS(hipStreamDestroy(stream_));
        if (solver_handle_)
            ASSERT_ROCBLAS_SUCCESS(rocblas_destroy_handle(solver_handle_));
        if (blas_handle_)
            ASSERT_ROCBLAS_SUCCESS(rocblas_destroy_handle(blas_handle_));
    }

    std::size_t m = 100;
    std::size_t n = 50;
    std::vector<T> V;

    rocblas_handle blas_handle_ = nullptr;
    rocblas_handle solver_handle_ = nullptr;
    hipStream_t stream_ = nullptr;
    int* devInfo_ = nullptr;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(CholQRHIPTest, TestTypes);

TYPED_TEST(CholQRHIPTest, cholQR1)
{
    using T = TypeParam;

    const auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_10.bin", 0, this->m,
                 this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    const int info = chase::linalg::internal::hip::cholQR1<T>(
        this->blas_handle_, this->solver_handle_, Vec, nullptr, 0, nullptr,
        this->devInfo_);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    const auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

// ChASEGPU allocates A as (2*n) x n but cholQR uses packed lda = n on the
// leading n x n block (same as master CUDA). This must work on tall buffers.
TYPED_TEST(CholQRHIPTest, cholQR1PackedLdaOnTallWorkspace)
{
    using T = TypeParam;

    const auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_10.bin", 0, this->m,
                 this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    chase::matrix::Matrix<T, chase::platform::GPU> A(2 * this->n, this->n);
    const int info = chase::linalg::internal::hip::cholQR1<T>(
        this->blas_handle_, this->solver_handle_, Vec, nullptr, 0, &A,
        this->devInfo_);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    const auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

// ChASEGPU passes pre-allocated d_work_; POTRF binds it on the solver handle only.
TYPED_TEST(CholQRHIPTest, cholQR1PackedLdaWithExternalWorkspace)
{
    using T = TypeParam;
    using hip_solver = chase::linalg::internal::gpu_backend::solver<
        chase::linalg::internal::gpu_backend::hip_backend>;

    const auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_10.bin", 0, this->m,
                 this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    chase::matrix::Matrix<T, chase::platform::GPU> A(2 * this->n, this->n);

    int lwork_potrf = 0;
    int lwork_geqrf = 0;
    ASSERT_ROCBLAS_SUCCESS(hip_solver::potrf_buffer_size<T>(
        this->solver_handle_, rocblas_fill_upper, this->n, A.data(), this->n,
        &lwork_potrf));
    ASSERT_ROCBLAS_SUCCESS(hip_solver::geqrf_buffer_size<T>(
        this->solver_handle_, this->m, this->n, Vec.data(), Vec.ld(),
        &lwork_geqrf));
    const int lwork = std::max(lwork_potrf, lwork_geqrf);

    T* workspace = nullptr;
    ASSERT_HIP_SUCCESS(
        hipMalloc(&workspace, sizeof(T) * static_cast<std::size_t>(lwork)));

    const int info = chase::linalg::internal::hip::cholQR1<T>(
        this->blas_handle_, this->solver_handle_, Vec, workspace, lwork, &A,
        this->devInfo_);
    ASSERT_HIP_SUCCESS(hipFree(workspace));
    ASSERT_EQ(info, 0);
    Vec.D2H();
    const auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

// ChASEGPU::CUBLAS_INIT runs workspace-size queries on the solver handle, then
// cholQR1 with d_work_. Queries must not corrupt that handle (scratch handle).
TYPED_TEST(CholQRHIPTest, cholQR1AfterChASEGPUWorkspaceSizing)
{
    using T = TypeParam;
    using hip_solver = chase::linalg::internal::gpu_backend::solver<
        chase::linalg::internal::gpu_backend::hip_backend>;
    namespace gpu_backend = chase::linalg::internal::gpu_backend;

    const std::size_t m = this->m;
    const std::size_t n = this->n;
    const auto machineEpsilon = MachineEpsilon<T>::value();

    std::vector<T> host_v(m * n);
    read_vectors(host_v.data(), GetQRFileName<T>() + "cond_10.bin", 0, m, m, n,
                 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(m, n, m, host_v.data());
    chase::matrix::Matrix<T, chase::platform::GPU> A(2 * n, n);
    chase::matrix::Matrix<chase::Base<T>, chase::platform::GPU> ritzv(n, 1);

    T* d_tau = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_tau, n * sizeof(T)));

    gpu_backend::SolverWorkspacePlan workspace_plan;
    gpu_backend::plan_query_qr_workspace<hip_solver, T>(
        workspace_plan, this->solver_handle_, m, n, Vec.data(), Vec.ld(), d_tau);
    gpu_backend::plan_query_heevd_workspace<hip_solver, T>(
        workspace_plan, this->solver_handle_, n, A.data(), A.ld(), ritzv.data());
    gpu_backend::plan_query_potrf_workspace<hip_solver, T>(
        workspace_plan, this->solver_handle_, n, A.data(), A.ld());
    gpu_backend::plan_include_packed_triangular(workspace_plan, n);

    const int lwork = workspace_plan.device_element_count();

    T* workspace = nullptr;
    ASSERT_HIP_SUCCESS(
        hipMalloc(&workspace, sizeof(T) * static_cast<std::size_t>(lwork)));

    const int info = chase::linalg::internal::hip::cholQR1<T>(
        this->blas_handle_, this->solver_handle_, Vec, workspace, lwork, &A,
        this->devInfo_);

    ASSERT_HIP_SUCCESS(hipFree(d_tau));
    ASSERT_HIP_SUCCESS(hipFree(workspace));
    ASSERT_EQ(info, 0);
    Vec.D2H();
    const auto orth = orthogonality<T>(m, n, host_v.data(), m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CholQRHIPTest, cholQR1BadlyCond)
{
    using T = TypeParam;

    const auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_1e4.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    const int info = chase::linalg::internal::hip::cholQR1<T>(
        this->blas_handle_, this->solver_handle_, Vec, nullptr, 0, nullptr,
        this->devInfo_);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    const auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    EXPECT_GT(orth, machineEpsilon);
    EXPECT_LT(orth, 1.0);
}

TYPED_TEST(CholQRHIPTest, cholQR1IllCond)
{
    using T = TypeParam;

    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    const int info = chase::linalg::internal::hip::cholQR1<T>(
        this->blas_handle_, this->solver_handle_, Vec, nullptr, 0, nullptr,
        this->devInfo_);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, static_cast<int>(this->n));
}

TYPED_TEST(CholQRHIPTest, cholQR2)
{
    using T = TypeParam;

    const auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_1e4.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    const int info = chase::linalg::internal::hip::cholQR2<T>(
        this->blas_handle_, this->solver_handle_, Vec, nullptr, 0, nullptr,
        this->devInfo_);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    const auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CholQRHIPTest, cholQR2IllCond)
{
    using T = TypeParam;

    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    const int info = chase::linalg::internal::hip::cholQR2<T>(
        this->blas_handle_, this->solver_handle_, Vec, nullptr, 0, nullptr,
        this->devInfo_);
    EXPECT_GT(info, 0);
    EXPECT_LE(info, static_cast<int>(this->n));
}

TYPED_TEST(CholQRHIPTest, householderQR)
{
    using T = TypeParam;

    const auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());

    T* d_tau = nullptr;
    ASSERT_HIP_SUCCESS(hipMalloc(&d_tau, this->n * sizeof(T)));

    chase::linalg::internal::hip::houseHoulderQR(
        this->solver_handle_, Vec, d_tau, this->devInfo_);
    Vec.D2H();
    const auto orth =
        orthogonality<T>(this->m, this->n, Vec.cpu_data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);

    ASSERT_HIP_SUCCESS(hipFree(d_tau));
}

TYPED_TEST(CholQRHIPTest, scholQR)
{
    using T = TypeParam;

    const auto machineEpsilon = MachineEpsilon<T>::value();
    read_vectors(this->V.data(), GetQRFileName<T>() + "cond_ill.bin", 0,
                 this->m, this->m, this->n, 0);
    chase::matrix::Matrix<T, chase::platform::GPU> Vec(this->m, this->n,
                                                       this->m, this->V.data());
    const int info = chase::linalg::internal::hip::shiftedcholQR2<T>(
        this->blas_handle_, this->solver_handle_, Vec, nullptr, 0, nullptr,
        this->devInfo_);
    ASSERT_EQ(info, 0);
    Vec.D2H();
    const auto orth = orthogonality<T>(this->m, this->n, this->V.data(), this->m);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 10);
}
