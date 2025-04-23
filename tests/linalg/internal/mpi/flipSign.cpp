// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/mpi/flipSign.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>

template <typename T>
class flipSignCPUDistTest : public ::testing::Test
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
TYPED_TEST_SUITE(flipSignCPUDistTest, TestTypes);

TYPED_TEST(flipSignCPUDistTest, flipSignCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 10;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>(
        N, N, mpi_grid);

    for (auto i = 0; i < H.l_rows() * H.l_cols(); i++)
    {
        H.l_data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(H);

    for (auto i = 0; i < H.l_rows(); i++)
    {
        for (auto j = 0; j < H.l_cols(); j++)
        {
            // assume is squared grid
            if (this->world_rank == 0 || this->world_rank == 2)
            {
                EXPECT_EQ(H.l_data()[i + j * H.l_ld()], T(1.0));
            }
            else
            {
                EXPECT_EQ(H.l_data()[i + j * H.l_ld()], -T(1.0));
            }
        }
    }
}

TYPED_TEST(flipSignCPUDistTest, flipSignCommColMultiVectorsCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 10;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    auto V = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::CPU>(N, N, mpi_grid);

    for (auto i = 0; i < V.l_rows() * V.l_cols(); i++)
    {
        V.l_data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V);

    for (auto i = 0; i < V.l_rows(); i++)
    {
        for (auto j = 0; j < V.l_cols(); j++)
        {
            // assume is squared grid
            if (this->world_rank == 0 || this->world_rank == 2)
            {
                EXPECT_EQ(V.l_data()[i + j * V.l_ld()], T(1.0));
            }
            else
            {
                EXPECT_EQ(V.l_data()[i + j * V.l_ld()], -T(1.0));
            }
        }
    }
}

TYPED_TEST(flipSignCPUDistTest, flipSignCommRowMultiVectorsCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 10;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    auto V = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::CPU>(
        N, N, mpi_grid);

    for (auto i = 0; i < V.l_rows() * V.l_cols(); i++)
    {
        V.l_data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V);

    for (auto i = 0; i < V.l_rows(); i++)
    {
        for (auto j = 0; j < V.l_cols(); j++)
        {
            // assume is squared grid
            if (this->world_rank == 0 || this->world_rank == 1)
            {
                EXPECT_EQ(V.l_data()[i + j * V.l_ld()], T(1.0));
            }
            else
            {
                EXPECT_EQ(V.l_data()[i + j * V.l_ld()], -T(1.0));
            }
        }
    }
}
