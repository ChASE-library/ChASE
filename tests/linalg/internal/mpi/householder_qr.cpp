// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
// Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/mpi/mpi_kernels.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "tests/linalg/internal/mpi/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <complex>
#include <gtest/gtest.h>

template <typename T>
class HouseholderMPIDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ASSERT_EQ(world_size, 4);
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            4, 1, MPI_COMM_WORLD);
    }

    void TearDown() override {}

    int world_rank = 0;
    int world_size = 0;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid;

    static constexpr std::size_t N = 100;
    static constexpr std::size_t n = 50;
};

using HouseholderMPITypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(HouseholderMPIDistTest, HouseholderMPITypes);

// Same binary matrices as CholQR CPU tests (cond_10 / cond_1e4); unblocked
// distributed Householder QR + form Q.
TYPED_TEST(HouseholderMPIDistTest, distributedHouseQRFormQCond10)
{
    using T = TypeParam;
    auto machineEpsilon = MachineEpsilon<T>::value();

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(
        this->N, this->n, this->mpi_grid);

    const std::size_t xlen = this->N / static_cast<std::size_t>(this->world_size);
    const std::size_t xoff =
        static_cast<std::size_t>(this->world_rank) * (this->N / 4u);

    read_vectors(V_.l_data(), GetQRFileName<T>() + "cond_10.bin", xoff, xlen,
                 this->N, this->n, 0);

    chase::linalg::internal::cpu_mpi::cpu_distributed_houseQR_formQ(V_);

    const auto orth = orthogonality<T>(xlen, this->n, V_.l_data(), xlen,
                                       this->mpi_grid->get_col_comm());
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 25);
}

// Blocked compact-WY path (panel width nb); same data as cholQR2 cond_1e4.
TYPED_TEST(HouseholderMPIDistTest, distributedBlockedHouseQRFormQCond1e4)
{
    using T = TypeParam;
    auto machineEpsilon = MachineEpsilon<T>::value();

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(
        this->N, this->n, this->mpi_grid);

    const std::size_t xlen = this->N / static_cast<std::size_t>(this->world_size);
    const std::size_t xoff =
        static_cast<std::size_t>(this->world_rank) * (this->N / 4u);

    read_vectors(V_.l_data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen,
                 this->N, this->n, 0);

    constexpr std::size_t nb = 8;
    chase::linalg::internal::cpu_mpi::cpu_distributed_blocked_houseQR_formQ(
        V_, nb);

    const auto orth = orthogonality<T>(xlen, this->n, V_.l_data(), xlen,
                                       this->mpi_grid->get_col_comm());
    EXPECT_GT(orth, machineEpsilon);
    EXPECT_LT(orth, chase::Base<T>(1));
}
