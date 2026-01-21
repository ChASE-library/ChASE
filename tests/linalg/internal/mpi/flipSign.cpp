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

    std::size_t N = 200;
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

TYPED_TEST(flipSignCPUDistTest, flipSignCorrectnessBlockCyclic)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 200, mb = 20;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    auto H = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(
        N, N, mb, mb, mpi_grid);

    for (auto i = 0; i < H.l_rows() * H.l_cols(); i++)
    {
        H.l_data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(H);

    for (auto idx_i = 0; idx_i < H.mblocks(); idx_i++)
    {
        for (auto idx_j = 0; idx_j < H.nblocks(); idx_j++)
        {
            for (auto i = 0; i < H.m_contiguous_lens()[idx_i]; i++)
            {
                for (auto j = 0; j < H.n_contiguous_lens()[idx_j]; j++)
                {
                    if (i + H.m_contiguous_global_offs()[idx_i] < (N / 2))
                    {
                        EXPECT_EQ(
                            H.l_data()[(j +
                                        H.n_contiguous_local_offs()[idx_j]) *
                                           H.l_ld() +
                                       i + H.m_contiguous_local_offs()[idx_i]],
                            T(1.0));
                    }
                    else
                    {

                        EXPECT_EQ(
                            H.l_data()[(j +
                                        H.n_contiguous_local_offs()[idx_j]) *
                                           H.l_ld() +
                                       i + H.m_contiguous_local_offs()[idx_i]],
                            T(-1.0));
                    }
                }
            }
        }
    }
}

TYPED_TEST(flipSignCPUDistTest, flipSignCommColMultiVectorsCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 200;
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

    std::size_t offset = 10, subSize = 190;

    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V, offset,
                                                              subSize);

    for (auto i = 0; i < V.l_rows(); i++)
    {
        for (auto j = 0; j < V.l_cols(); j++)
        {
            // assume is squared grid
            if (this->world_rank == 0 || this->world_rank == 2 || j < offset)
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

    std::size_t N = 200;
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

TYPED_TEST(flipSignCPUDistTest,
           flipSignCommColMultiVectorsBlockCyclicCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 200, mb = 20;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    auto V = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::CPU>(N, N, mb, mpi_grid);

    for (auto i = 0; i < V.l_rows() * V.l_cols(); i++)
    {
        V.l_data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V);

    for (auto idx_i = 0; idx_i < V.mblocks(); idx_i++)
    {
        for (auto i = 0; i < V.m_contiguous_lens()[idx_i]; i++)
        {
            for (auto j = 0; j < V.l_cols(); j++)
            {
                if (i + V.m_contiguous_global_offs()[idx_i] < (N / 2))
                {
                    EXPECT_EQ(V.l_data()[j * V.l_ld() + i +
                                         V.m_contiguous_local_offs()[idx_i]],
                              T(1.0));
                }
                else
                {
                    EXPECT_EQ(V.l_data()[j * V.l_ld() + i +
                                         V.m_contiguous_local_offs()[idx_i]],
                              T(-1.0));
                }
            }
        }
    }
}

TYPED_TEST(flipSignCPUDistTest,
           flipSignCommRowMultiVectorsBlockCyclicCorrectness)
{
    using T = TypeParam; // Get the current type

    std::size_t N = 200, mb = 20;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    auto V = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::CPU>(
        N, N, mb, mpi_grid);

    for (auto i = 0; i < V.l_rows() * V.l_cols(); i++)
    {
        V.l_data()[i] = T(1.0);
    }

    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(V);

    for (auto idx_i = 0; idx_i < V.mblocks(); idx_i++)
    {
        for (auto i = 0; i < V.m_contiguous_lens()[idx_i]; i++)
        {
            for (auto j = 0; j < V.l_cols(); j++)
            {
                if (i + V.m_contiguous_global_offs()[idx_i] < (N / 2))
                {
                    EXPECT_EQ(V.l_data()[j * V.l_ld() + i +
                                         V.m_contiguous_local_offs()[idx_i]],
                              T(1.0));
                }
                else
                {
                    EXPECT_EQ(V.l_data()[j * V.l_ld() + i +
                                         V.m_contiguous_local_offs()[idx_i]],
                              T(-1.0));
                }
            }
        }
    }
}
