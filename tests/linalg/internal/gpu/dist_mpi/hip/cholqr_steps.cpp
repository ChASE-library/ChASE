// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/error.hpp"
#include "external/amd_gpu/rocblaspp/rocblaspp.hpp"
#include "grid/mpiGrid2D.hpp"
#include "grid/mpiTypes.hpp"
#include "linalg/internal/gpu/kernels/hip/launchers/lacpy.hpp"
#include "external/amd_gpu/rocsolverpp/rocsolverpp.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_utils.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/cholqr.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <mpi.h>
#include <rocblas/rocblas.h>

#include <stdexcept>
#include <vector>

namespace
{

constexpr std::size_t kN = 100;
constexpr std::size_t kNcols = 50;

struct CholQRStepResources
{
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        grid;
    rocblas_handle blasH = nullptr;
    rocblas_handle solverH = nullptr;
    hipStream_t stream = nullptr;
    int rank = 0;
    int size = 0;
    std::size_t xlen = 0;

    void init()
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size != 4)
        {
            throw std::runtime_error("CholQRHIPDistSteps requires 4 MPI ranks");
        }
        xlen = kN / static_cast<std::size_t>(size);

        grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            4, 1, MPI_COMM_WORLD);

        int device = -1;
        CHECK_HIP_ERROR(hipGetDevice(&device));

        CHECK_ROCBLAS_ERROR(rocblas_create_handle(&blasH));
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_ROCBLAS_ERROR(rocblas_set_stream(blasH, stream));
        CHECK_ROCBLAS_ERROR(rocblas_create_handle(&solverH));
        CHECK_ROCBLAS_ERROR(rocblas_set_stream(solverH, stream));
    }

    void shutdown()
    {
        if (solverH)
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(solverH));
        if (blasH)
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(blasH));
        if (stream)
            CHECK_HIP_ERROR(hipStreamDestroy(stream));
        grid.reset();
    }
};

CholQRStepResources& resources()
{
    static CholQRStepResources r;
    return r;
}

class CholQRStepsEnvironment : public ::testing::Environment
{
public:
    void SetUp() override { resources().init(); }
    void TearDown() override { resources().shutdown(); }
};

void load_local_block(float* host, std::size_t xoff)
{
    read_vectors(host, GetQRFileName<float>() + "cond_10.bin", xoff,
                 resources().xlen, kN, kNcols, 0);
}

} // namespace

TEST(CholQRHIPDistSteps, Step00_MpiGridAndHandles)
{
    ASSERT_NE(resources().grid, nullptr);
    ASSERT_NE(resources().blasH, nullptr);
    ASSERT_NE(resources().solverH, nullptr);
}

TEST(CholQRHIPDistSteps, Step01_DeviceAllocAndH2D)
{
    const std::size_t xlen = resources().xlen;
    const std::size_t xoff = static_cast<std::size_t>(resources().rank) * xlen;
    std::vector<float> host(xlen * kNcols);
    load_local_block(host.data(), xoff);

    float* d_V = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_V, sizeof(float) * xlen * kNcols));
    CHECK_HIP_ERROR(hipMemcpy(d_V, host.data(), sizeof(float) * xlen * kNcols,
                              hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    CHECK_HIP_ERROR(hipFree(d_V));
}

TEST(CholQRHIPDistSteps, Step02_SyherkOnly)
{
    const std::size_t xlen = resources().xlen;
    const std::size_t xoff = static_cast<std::size_t>(resources().rank) * xlen;
    const std::size_t n = kNcols;
    const int ldv = static_cast<int>(xlen);

    std::vector<float> host(xlen * n);
    load_local_block(host.data(), xoff);

    float *d_V = nullptr, *d_A = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_V, sizeof(float) * xlen * n));
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * n * n));
    CHECK_HIP_ERROR(hipMemcpy(d_V, host.data(), sizeof(float) * xlen * n,
                              hipMemcpyHostToDevice));

    float one = 1.f, zero = 0.f;
    CHECK_ROCBLAS_ERROR(chase::linalg::rocblaspp::rocblasTsyherk(
        resources().blasH, rocblas_fill_upper, rocblas_operation_transpose, n,
        xlen, &one, d_V, ldv, &zero, d_A, static_cast<int>(n)));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    CHECK_HIP_ERROR(hipFree(d_V));
    CHECK_HIP_ERROR(hipFree(d_A));
}

TEST(CholQRHIPDistSteps, Step03_ExtractUnpackLocal)
{
    const std::size_t xlen = resources().xlen;
    const std::size_t xoff = static_cast<std::size_t>(resources().rank) * xlen;
    const std::size_t n = kNcols;
    const int ldv = static_cast<int>(xlen);
    const std::size_t upperTriangularSize = n * (n + 1) / 2;

    std::vector<float> host(xlen * n);
    load_local_block(host.data(), xoff);

    float *d_V = nullptr, *d_A = nullptr, *d_workspace = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_V, sizeof(float) * xlen * n));
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * n * n));
    CHECK_HIP_ERROR(
        hipMalloc(&d_workspace, sizeof(float) * upperTriangularSize));
    CHECK_HIP_ERROR(hipMemcpy(d_V, host.data(), sizeof(float) * xlen * n,
                              hipMemcpyHostToDevice));

    float one = 1.f, zero = 0.f;
    CHECK_ROCBLAS_ERROR(chase::linalg::rocblaspp::rocblasTsyherk(
        resources().blasH, rocblas_fill_upper, rocblas_operation_transpose, n,
        xlen, &one, d_V, ldv, &zero, d_A, static_cast<int>(n)));

    chase::linalg::internal::hip::extractUpperTriangular(
        d_A, n, d_workspace, n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    chase::linalg::internal::hip::unpackUpperTriangular(d_A, n, d_workspace,
                                                        n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    CHECK_HIP_ERROR(hipFree(d_V));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_workspace));
}

TEST(CholQRHIPDistSteps, Step04_GpuMpiAllreduce)
{
    const std::size_t xlen = resources().xlen;
    const std::size_t xoff = static_cast<std::size_t>(resources().rank) * xlen;
    const std::size_t n = kNcols;
    const int ldv = static_cast<int>(xlen);
    const std::size_t upperTriangularSize = n * (n + 1) / 2;

    std::vector<float> host(xlen * n);
    load_local_block(host.data(), xoff);

    float *d_V = nullptr, *d_A = nullptr, *d_workspace = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_V, sizeof(float) * xlen * n));
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * n * n));
    CHECK_HIP_ERROR(
        hipMalloc(&d_workspace, sizeof(float) * upperTriangularSize));
    CHECK_HIP_ERROR(hipMemcpy(d_V, host.data(), sizeof(float) * xlen * n,
                              hipMemcpyHostToDevice));

    float one = 1.f, zero = 0.f;
    CHECK_ROCBLAS_ERROR(chase::linalg::rocblaspp::rocblasTsyherk(
        resources().blasH, rocblas_fill_upper, rocblas_operation_transpose, n,
        xlen, &one, d_V, ldv, &zero, d_A, static_cast<int>(n)));

    chase::linalg::internal::hip::extractUpperTriangular(
        d_A, n, d_workspace, n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    chase::linalg::internal::hip_mpi_utils::mpiAllreduceSum(
        d_workspace, upperTriangularSize, resources().grid->get_comm());
    chase::linalg::internal::hip::unpackUpperTriangular(d_A, n, d_workspace,
                                                        n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    CHECK_HIP_ERROR(hipFree(d_V));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_workspace));
}

TEST(CholQRHIPDistSteps, Step05_Potrf)
{
    const std::size_t xlen = resources().xlen;
    const std::size_t xoff = static_cast<std::size_t>(resources().rank) * xlen;
    const std::size_t n = kNcols;
    const int ldv = static_cast<int>(xlen);
    const std::size_t upperTriangularSize = n * (n + 1) / 2;

    std::vector<float> host(xlen * n);
    load_local_block(host.data(), xoff);

    float *d_V = nullptr, *d_A = nullptr, *d_workspace = nullptr;
    rocblas_int* d_info = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_V, sizeof(float) * xlen * n));
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * n * n));
    CHECK_HIP_ERROR(
        hipMalloc(&d_workspace, sizeof(float) * upperTriangularSize));
    CHECK_HIP_ERROR(hipMalloc(&d_info, sizeof(rocblas_int)));
    CHECK_HIP_ERROR(hipMemcpy(d_V, host.data(), sizeof(float) * xlen * n,
                              hipMemcpyHostToDevice));

    float one = 1.f, zero = 0.f;
    CHECK_ROCBLAS_ERROR(chase::linalg::rocblaspp::rocblasTsyherk(
        resources().blasH, rocblas_fill_upper, rocblas_operation_transpose, n,
        xlen, &one, d_V, ldv, &zero, d_A, static_cast<int>(n)));

    chase::linalg::internal::hip::extractUpperTriangular(
        d_A, n, d_workspace, n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    chase::linalg::internal::hip_mpi_utils::mpiAllreduceSum(
        d_workspace, upperTriangularSize, resources().grid->get_comm());
    chase::linalg::internal::hip::unpackUpperTriangular(d_A, n, d_workspace,
                                                        n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    CHECK_ROCBLAS_ERROR(
        chase::linalg::rocsolverpp::rocsolverTpotrf<float>(
            resources().solverH, rocblas_fill_upper, static_cast<rocblas_int>(n),
            d_A, static_cast<rocblas_int>(n), d_info));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    int info = 0;
    CHECK_HIP_ERROR(hipMemcpy(&info, d_info, sizeof(int), hipMemcpyDeviceToHost));
    ASSERT_EQ(info, 0);

    CHECK_HIP_ERROR(hipFree(d_V));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIP_ERROR(hipFree(d_info));
}

TEST(CholQRHIPDistSteps, Step06_PotrfAndTrsm)
{
    const std::size_t xlen = resources().xlen;
    const std::size_t xoff = static_cast<std::size_t>(resources().rank) * xlen;
    const std::size_t n = kNcols;
    const int ldv = static_cast<int>(xlen);
    const std::size_t upperTriangularSize = n * (n + 1) / 2;

    std::vector<float> host(xlen * n);
    load_local_block(host.data(), xoff);

    float *d_V = nullptr, *d_A = nullptr, *d_workspace = nullptr;
    rocblas_int* d_info = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_V, sizeof(float) * xlen * n));
    CHECK_HIP_ERROR(hipMalloc(&d_A, sizeof(float) * n * n));
    CHECK_HIP_ERROR(
        hipMalloc(&d_workspace, sizeof(float) * upperTriangularSize));
    CHECK_HIP_ERROR(hipMalloc(&d_info, sizeof(rocblas_int)));
    CHECK_HIP_ERROR(hipMemcpy(d_V, host.data(), sizeof(float) * xlen * n,
                              hipMemcpyHostToDevice));

    float one = 1.f, zero = 0.f;
    CHECK_ROCBLAS_ERROR(chase::linalg::rocblaspp::rocblasTsyherk(
        resources().blasH, rocblas_fill_upper, rocblas_operation_transpose, n,
        xlen, &one, d_V, ldv, &zero, d_A, static_cast<int>(n)));

    chase::linalg::internal::hip::extractUpperTriangular(
        d_A, n, d_workspace, n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    chase::linalg::internal::hip_mpi_utils::mpiAllreduceSum(
        d_workspace, upperTriangularSize, resources().grid->get_comm());
    chase::linalg::internal::hip::unpackUpperTriangular(d_A, n, d_workspace,
                                                        n);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    CHECK_ROCBLAS_ERROR(
        chase::linalg::rocsolverpp::rocsolverTpotrf<float>(
            resources().solverH, rocblas_fill_upper, static_cast<rocblas_int>(n),
            d_A, static_cast<rocblas_int>(n), d_info));

    CHECK_ROCBLAS_ERROR(chase::linalg::rocblaspp::rocblasTtrsm(
        resources().blasH, rocblas_side_right, rocblas_fill_upper,
        rocblas_operation_none, rocblas_diagonal_non_unit, xlen, n, &one, d_A,
        static_cast<int>(n), d_V, ldv));
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    CHECK_HIP_ERROR(hipFree(d_V));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_workspace));
    CHECK_HIP_ERROR(hipFree(d_info));
}

TEST(CholQRHIPDistSteps, Step07_FullCholQR1)
{
    const std::size_t xlen = resources().xlen;
    const std::size_t xoff = static_cast<std::size_t>(resources().rank) * xlen;
    std::vector<float> host(xlen * kNcols);
    load_local_block(host.data(), xoff);

    float* d_V = nullptr;
    CHECK_HIP_ERROR(hipMalloc(&d_V, sizeof(float) * xlen * kNcols));
    CHECK_HIP_ERROR(hipMemcpy(d_V, host.data(), sizeof(float) * xlen * kNcols,
                              hipMemcpyHostToDevice));

    const int info = chase::linalg::internal::hip_mpi::cholQR1<float>(
        resources().blasH, resources().solverH, xlen, kNcols, d_V,
        static_cast<int>(xlen), resources().grid->get_comm());
    ASSERT_EQ(info, 0);

    CHECK_HIP_ERROR(hipFree(d_V));
}

::testing::Environment* const cholqr_steps_env =
    ::testing::AddGlobalTestEnvironment(new CholQRStepsEnvironment);
