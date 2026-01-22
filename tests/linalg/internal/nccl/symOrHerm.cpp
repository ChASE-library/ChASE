// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/nccl/symOrHerm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/nccl/shiftDiagonal.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>

// Global static resources that persist across all test suites
namespace
{
bool resources_initialized = false;
cublasHandle_t cublasH;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
} // namespace

template <typename T>
class SymOrHermGPUDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes

        if (!resources_initialized)
        {
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                2, 2, MPI_COMM_WORLD);
            CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));
            resources_initialized = true;
        }
    }

    void TearDown() override
    {
        // Don't free resources here - they will be reused
    }

    int world_rank;
    int world_size;
};

// Add a global test environment to handle resource cleanup at program exit
class ResourceCleanupEnvironment : public ::testing::Environment
{
public:
    ~ResourceCleanupEnvironment() override
    {
        if (resources_initialized)
        {
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH));
            mpi_grid.reset();
            resources_initialized = false;
        }
    }
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SymOrHermGPUDistTest, TestTypes);

TYPED_TEST(SymOrHermGPUDistTest, UpperTriangularMatrix)
{
    using T = TypeParam;
    const std::size_t N = 5;

    std::vector<T> U(N * N);
    // Initialize values explicitly
    U[0] = static_cast<T>(1);
    U[1] = static_cast<T>(2);
    U[2] = static_cast<T>(3);
    U[3] = static_cast<T>(4);
    U[4] = static_cast<T>(5);
    U[5] = static_cast<T>(0);
    U[6] = static_cast<T>(6);
    U[7] = static_cast<T>(7);
    U[8] = static_cast<T>(8);
    U[9] = static_cast<T>(9);
    U[10] = static_cast<T>(0);
    U[11] = static_cast<T>(0);
    U[12] = static_cast<T>(10);
    U[13] = static_cast<T>(11);
    U[14] = static_cast<T>(12);
    U[15] = static_cast<T>(0);
    U[16] = static_cast<T>(0);
    U[17] = static_cast<T>(0);
    U[18] = static_cast<T>(13);
    U[19] = static_cast<T>(14);
    U[20] = static_cast<T>(0);
    U[21] = static_cast<T>(0);
    U[22] = static_cast<T>(0);
    U[23] = static_cast<T>(0);
    U[24] = static_cast<T>(15);

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        N, N, mpi_grid);
    H.allocate_cpu_data();
    std::size_t* goffs = H.g_offs();

    for (auto j = 0; j < H.l_cols(); j++)
    {
        for (auto i = 0; i < H.l_rows(); i++)
        {
            H.cpu_data()[i + j * H.cpu_ld()] =
                U[i + goffs[0] + (j + goffs[1]) * N];
        }
    }
    H.H2D();
    bool is_sym =
        chase::linalg::internal::cuda_nccl::checkSymmetryEasy(cublasH, H);
    EXPECT_FALSE(is_sym);
#ifdef HAS_SCALAPACK
    chase::linalg::internal::cuda_nccl::symOrHermMatrix('U', H);
    H.H2D();
    is_sym = chase::linalg::internal::cuda_nccl::checkSymmetryEasy(cublasH, H);
    EXPECT_TRUE(is_sym);
#endif
}

TYPED_TEST(SymOrHermGPUDistTest, LowerTriangularMatrix)
{
    using T = TypeParam;
    const std::size_t N = 5;

    std::vector<T> U(N * N);
    // Initialize values explicitly
    U[0] = static_cast<T>(1);
    U[1] = static_cast<T>(0);
    U[2] = static_cast<T>(0);
    U[3] = static_cast<T>(0);
    U[4] = static_cast<T>(0);
    U[5] = static_cast<T>(2);
    U[6] = static_cast<T>(6);
    U[7] = static_cast<T>(0);
    U[8] = static_cast<T>(0);
    U[9] = static_cast<T>(0);
    U[10] = static_cast<T>(3);
    U[11] = static_cast<T>(7);
    U[12] = static_cast<T>(10);
    U[13] = static_cast<T>(0);
    U[14] = static_cast<T>(0);
    U[15] = static_cast<T>(4);
    U[16] = static_cast<T>(8);
    U[17] = static_cast<T>(11);
    U[18] = static_cast<T>(13);
    U[19] = static_cast<T>(0);
    U[20] = static_cast<T>(5);
    U[21] = static_cast<T>(9);
    U[22] = static_cast<T>(12);
    U[23] = static_cast<T>(14);
    U[24] = static_cast<T>(15);

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        N, N, mpi_grid);
    H.allocate_cpu_data();
    std::size_t* goffs = H.g_offs();

    for (auto j = 0; j < H.l_cols(); j++)
    {
        for (auto i = 0; i < H.l_rows(); i++)
        {
            H.cpu_data()[i + j * H.cpu_ld()] =
                U[i + goffs[0] + (j + goffs[1]) * N];
        }
    }
    H.H2D();

    bool is_sym =
        chase::linalg::internal::cuda_nccl::checkSymmetryEasy(cublasH, H);
    EXPECT_FALSE(is_sym);
#ifdef HAS_SCALAPACK
    chase::linalg::internal::cuda_nccl::symOrHermMatrix('L', H);
    H.H2D();
    is_sym = chase::linalg::internal::cuda_nccl::checkSymmetryEasy(cublasH, H);
    EXPECT_TRUE(is_sym);
#endif
}

// Add this at the end of the file, before main()
::testing::Environment* const resource_env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);
