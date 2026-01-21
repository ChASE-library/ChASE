// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/mpi/shiftDiagonal.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>

template <typename T>
class shiftDiagonalCPUDistTest : public ::testing::Test
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
TYPED_TEST_SUITE(shiftDiagonalCPUDistTest, TestTypes);

TYPED_TEST(shiftDiagonalCPUDistTest, ShiftDistCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 10;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    auto R = chase::distMatrix::RedundantMatrix<T>(N, N, mpi_grid);

    for (auto i = 0; i < std::min(R.g_cols(), R.g_rows()); i++)
    {
        R.l_data()[i + i * R.l_ld()] = T(1.0);
    }

    auto H = chase::distMatrix::BlockBlockMatrix<T>(N, N, mpi_grid);
    R.redistributeImpl(&H);
    chase::linalg::internal::cpu_mpi::shiftDiagonal(H, T(-5.0));

    if (this->world_rank == 0 || this->world_rank == 3)
    {
        for (auto i = 0; i < H.l_rows(); i++)
        {
            for (auto j = 0; j < H.l_cols(); j++)
            {
                // assume is squared grid
                if (i == j)
                {
                    EXPECT_EQ(H.l_data()[i + i * H.l_ld()], T(-4.0));
                }
                else
                {
                    EXPECT_EQ(H.l_data()[i + j * H.l_ld()], T(0.0));
                }
            }
        }
    }
    else
    {
        for (auto i = 0; i < H.l_ld() * H.l_cols(); i++)
        {
            EXPECT_EQ(H.l_data()[i], T(0.0));
        }
    }
}