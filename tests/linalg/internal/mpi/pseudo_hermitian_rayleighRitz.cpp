// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "external/scalapackpp/scalapackpp.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/hemm.hpp"
#include "linalg/internal/mpi/rayleighRitz.hpp"
#include "linalg/internal/mpi/residuals.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <random>

template <typename T>
class PseudoHermitianRRCPUDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        ritzv.resize(N);
        ritzv_tiny.resize(N_tiny);
    }

    void TearDown() override {}

    int world_rank;
    int world_size;

    std::size_t N_tiny = 10;
    std::vector<chase::Base<T>> ritzv_tiny;

    std::size_t N = 200;
    std::vector<chase::Base<T>> ritzv;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHermitianRRCPUDistTest, TestTypes);

TYPED_TEST(PseudoHermitianRRCPUDistTest, PseudoHermitianRRTinyCorrectness)
{
    using T = TypeParam;            // Get the current type
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    int* coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<T>(
        this->N_tiny, this->N_tiny, mpi_grid);
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());

    chase::matrix::Matrix<T> exact_eigsl_H =
        chase::matrix::Matrix<T>(this->N_tiny, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_TinyEigs<T>());

    auto V1_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(
        this->N_tiny, this->N_tiny, mpi_grid);
    auto V2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(
        this->N_tiny, this->N_tiny, mpi_grid);

    auto W1_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row>(
        this->N_tiny, this->N_tiny, mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row>(
        this->N_tiny, this->N_tiny, mpi_grid);

    std::size_t g_off = V1_.g_off();

    for (auto i = 0; i < V1_.l_rows(); i++)
    {
        V1_.l_data()[g_off * V1_.l_ld() + i * (V1_.l_ld() + 1)] = T(1.0);
        V2_.l_data()[g_off * V2_.l_ld() + i * (V2_.l_ld() + 1)] = T(1.0);
    }

    std::size_t offset = 0, subSize = this->N_tiny;

    chase::linalg::internal::cpu_mpi::pseudo_hermitian_rayleighRitz(
        H_, V1_, V2_, W1_, W2_, this->ritzv_tiny.data(), offset, subSize);

    for (auto i = offset; i < offset + subSize; i++)
    {
        EXPECT_NEAR(this->ritzv_tiny.data()[i],
                    chase::Base<T>(std::real(exact_eigsl_H.data()[i])),
                    100 * MachineEpsilon<T>::value());
    }
}

TYPED_TEST(PseudoHermitianRRCPUDistTest, PseudoHermitianRRCorrectness)
{
    using T = TypeParam;            // Get the current type
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    chase::Base<T> tolerance;
    if constexpr (std::is_same<T, float>::value)
    {
        tolerance = 1.0e-3;
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        tolerance = 1.0e-3;
    }
    else
    {
        tolerance = 1.0e-9;
    }

    int* coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<T>(
        this->N, this->N, mpi_grid);
    H_.readFromBinaryFile(GetBSE_Matrix<T>());

    chase::matrix::Matrix<T> exact_eigsl_H =
        chase::matrix::Matrix<T>(this->N, 1);
    exact_eigsl_H.readFromBinaryFile(GetBSE_Eigs<T>());

    auto V1_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(this->N, this->N,
                                                             mpi_grid);
    auto V2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(this->N, this->N,
                                                             mpi_grid);

    auto W1_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row>(this->N, this->N,
                                                          mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row>(this->N, this->N,
                                                          mpi_grid);

    std::size_t g_off = V1_.g_off();

    for (auto i = 0; i < V1_.l_rows(); i++)
    {
        V1_.l_data()[g_off * V1_.l_ld() + i * (V1_.l_ld() + 1)] = T(1.0);
        V2_.l_data()[g_off * V2_.l_ld() + i * (V2_.l_ld() + 1)] = T(1.0);
    }

    std::size_t offset = 0, subSize = this->N;

    chase::linalg::internal::cpu_mpi::pseudo_hermitian_rayleighRitz(
        H_, V1_, V2_, W1_, W2_, this->ritzv.data(), offset, subSize);

    for (auto i = offset; i < offset + subSize; i++)
    {
        EXPECT_NEAR(this->ritzv.data()[i],
                    chase::Base<T>(std::real(exact_eigsl_H.data()[i])),
                    tolerance);
    }
}
