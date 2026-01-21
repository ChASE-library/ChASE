// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/mpi/symOrHerm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/shiftDiagonal.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>

template <typename T>
class SymOrHermCPUDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    void TearDown() override {}

    int world_rank;
    int world_size;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SymOrHermCPUDistTest, TestTypes);

TYPED_TEST(SymOrHermCPUDistTest, UpperTriangularMatrix)
{
    using T = TypeParam;
    const std::size_t N = 5;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes

    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

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

    auto R = chase::distMatrix::RedundantMatrix<T>(N, N, N, U.data(), mpi_grid);
    auto H = chase::distMatrix::BlockBlockMatrix<T>(N, N, mpi_grid);
    R.redistributeImpl(&H);
    bool is_sym = chase::linalg::internal::cpu_mpi::checkSymmetryEasy(H);
    EXPECT_FALSE(is_sym);
#ifdef HAS_SCALAPACK
    chase::linalg::internal::cpu_mpi::symOrHermMatrix('U', H);
    is_sym = chase::linalg::internal::cpu_mpi::checkSymmetryEasy(H);
    EXPECT_TRUE(is_sym);
#endif
}

TYPED_TEST(SymOrHermCPUDistTest, LowerTriangularMatrix)
{
    using T = TypeParam;
    const std::size_t N = 5;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes

    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

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

    auto R = chase::distMatrix::RedundantMatrix<T>(N, N, N, U.data(), mpi_grid);
    auto H = chase::distMatrix::BlockBlockMatrix<T>(N, N, mpi_grid);
    R.redistributeImpl(&H);

    bool is_sym = chase::linalg::internal::cpu_mpi::checkSymmetryEasy(H);
    EXPECT_FALSE(is_sym);
#ifdef HAS_SCALAPACK
    chase::linalg::internal::cpu_mpi::symOrHermMatrix('L', H);
    is_sym = chase::linalg::internal::cpu_mpi::checkSymmetryEasy(H);
    EXPECT_TRUE(is_sym);
#endif
}