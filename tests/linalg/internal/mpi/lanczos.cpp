// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/mpi/lanczos.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "tests/linalg/internal/lanczos_reference.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <random>

template <typename T>
class LanczosCPUDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        Reference.resize(N * N);

        chase::tests::lanczos::initializeReferenceHermitianMatrix(
            Reference.data(), N);
        chase::tests::lanczos::computeReferenceEigenBounds(
            Reference.data(), N, &lambda_min, &lambda_max);
        ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);
    }

    void TearDown() override {}

    int world_rank;
    int world_size;

    std::size_t M = 10;
    std::size_t numvec = 4;
    std::size_t N = 500;
    std::vector<T> Reference;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;
    chase::Base<T> lambda_min;
    chase::Base<T> lambda_max;
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosCPUDistTest, TestTypes);

TYPED_TEST(LanczosCPUDistTest, mlanczos)
{
    using T = TypeParam; // Get the current type

    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    int* coords = mpi_grid.get()->get_coords();

    auto H_ =
        chase::distMatrix::BlockBlockMatrix<T>(this->N, this->N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(this->N, this->M,
                                                             mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    auto Reference_ = chase::distMatrix::RedundantMatrix<T>(
        this->N, this->N, this->N, this->Reference.data(), mpi_grid);
    Reference_.redistributeImpl(&H_);

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos(
        this->M, this->numvec, H_, V_, &upperb, this->ritzv.data(),
        this->Tau.data(), this->ritzV.data());

    chase::tests::lanczos::expectRitzValuesInsideReferenceSpectrum(
        this->ritzv, this->M, this->numvec, this->lambda_min,
        this->lambda_max);
    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}

TYPED_TEST(LanczosCPUDistTest, lanczos)
{
    using T = TypeParam; // Get the current type

    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    int* coords = mpi_grid.get()->get_coords();

    auto H_ =
        chase::distMatrix::BlockBlockMatrix<T>(this->N, this->N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column>(this->N, 1,
                                                             mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    auto Reference_ = chase::distMatrix::RedundantMatrix<T>(
        this->N, this->N, this->N, this->Reference.data(), mpi_grid);
    Reference_.redistributeImpl(&H_);

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos(this->M, H_, V_, &upperb);

    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}

TYPED_TEST(LanczosCPUDistTest, mlanczosBlockCyclic)
{
    using T = TypeParam; // Get the current type

    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    int* coords = mpi_grid.get()->get_coords();
    std::size_t blocksize = 16;
    auto H_ = chase::distMatrix::BlockCyclicMatrix<T>(
        this->N, this->N, blocksize, blocksize, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column>(
        this->N, this->M, blocksize, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    auto Reference_ = chase::distMatrix::RedundantMatrix<T>(
        this->N, this->N, this->N, this->Reference.data(), mpi_grid);
    Reference_.redistributeImpl(&H_);

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos(
        this->M, this->numvec, H_, V_, &upperb, this->ritzv.data(),
        this->Tau.data(), this->ritzV.data());

    chase::tests::lanczos::expectRitzValuesInsideReferenceSpectrum(
        this->ritzv, this->M, this->numvec, this->lambda_min,
        this->lambda_max);
    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}

TYPED_TEST(LanczosCPUDistTest, lanczosBlockCyclic)
{
    using T = TypeParam; // Get the current type

    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        mpi_grid = std::make_shared<
            chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
            2, 2, MPI_COMM_WORLD);

    int* coords = mpi_grid.get()->get_coords();
    std::size_t blocksize = 16;
    auto H_ = chase::distMatrix::BlockCyclicMatrix<T>(
        this->N, this->N, blocksize, blocksize, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column>(
        this->N, 1, blocksize, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    auto Reference_ = chase::distMatrix::RedundantMatrix<T>(
        this->N, this->N, this->N, this->Reference.data(), mpi_grid);
    Reference_.redistributeImpl(&H_);

    chase::Base<T> upperb;

    chase::linalg::internal::cpu_mpi::lanczos(this->M, H_, V_, &upperb);

    chase::tests::lanczos::expectLanczosUpperBoundNearReference(
        upperb, this->lambda_min, this->lambda_max);
}
