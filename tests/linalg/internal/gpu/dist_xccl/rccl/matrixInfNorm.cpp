// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/error.hpp"
#include "linalg/internal/gpu/dist_xccl/rccl/matrix_inf_norm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <type_traits>

namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
} // namespace

template <typename T>
class matrixInfNormGPURCCLDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ASSERT_EQ(world_size, 4);

        if (!resources_initialized)
        {
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                2, 2, MPI_COMM_WORLD);
            resources_initialized = true;
        }
    }

    void TearDown() override {}

    int world_rank;
    int world_size;
};

class ResourceCleanupEnvironment : public ::testing::Environment
{
public:
    ~ResourceCleanupEnvironment() override
    {
        if (resources_initialized)
        {
            mpi_grid.reset();
            resources_initialized = false;
        }
    }
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(matrixInfNormGPURCCLDistTest, TestTypes);

// 2x2 ColMajor, balanced N, fill each block with (rank+1).
// Real:    row sums 4n, 6n → ||H||_∞ = 3N.
// Complex: fill (r+1)+i(r+1), |z|=2(r+1) → ||H||_∞ = 6N.
TYPED_TEST(matrixInfNormGPURCCLDistTest, RankPlusOneFill)
{
    using T = TypeParam;
    using RealT = chase::Base<T>;

    constexpr std::size_t N = 8;
    constexpr bool is_complex = std::is_same_v<T, std::complex<RealT>>;

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        N, N, mpi_grid);
    H.allocate_cpu_data();

    ASSERT_EQ(H.l_rows(), N / 2);
    ASSERT_EQ(H.l_cols(), N / 2);

    const RealT v = RealT(this->world_rank + 1);
    T fill{};
    if constexpr (is_complex)
        fill = T(v, v);
    else
        fill = T(v);
    for (std::size_t j = 0; j < H.l_cols(); ++j)
    {
        for (std::size_t i = 0; i < H.l_rows(); ++i)
        {
            H.cpu_data()[i + j * H.cpu_ld()] = fill;
        }
    }
    H.H2D();

    RealT* d_row_sums = nullptr;
    RealT* d_result = nullptr;
    CHECK_HIP_ERROR(hipMalloc(reinterpret_cast<void**>(&d_row_sums),
                              H.l_rows() * sizeof(RealT)));
    CHECK_HIP_ERROR(
        hipMalloc(reinterpret_cast<void**>(&d_result), sizeof(RealT)));

    chase::linalg::internal::hip_rccl::matrixInfNorm(H, d_row_sums, d_result,
                                                     nullptr);

    RealT norm = RealT(-1);
    CHECK_HIP_ERROR(
        hipMemcpy(&norm, d_result, sizeof(RealT), hipMemcpyDeviceToHost));

    EXPECT_EQ(norm, RealT(is_complex ? 6 * N : 3 * N));

    CHECK_HIP_ERROR(hipFree(d_row_sums));
    CHECK_HIP_ERROR(hipFree(d_result));
}

::testing::Environment* const resource_env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);
