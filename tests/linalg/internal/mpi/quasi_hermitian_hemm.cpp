// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "tests/linalg/internal/utils.hpp"
#include "linalg/internal/mpi/hemm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

template <typename T>
class QuasiHEMMCPUDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);          
    }

    void TearDown() override {              
    }

    int world_rank;
    int world_size;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(QuasiHEMMCPUDistTest, TestTypes);


TYPED_TEST(QuasiHEMMCPUDistTest, QuasiHEMMDistCorrectness) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 10;
    std::size_t n = 4;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    auto SH_ = chase::distMatrix::BlockBlockMatrix<T>(N, N, mpi_grid);
    SH_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(SH_); //We assume the flipping function works

    auto H_  = chase::distMatrix::QuasiHermitianBlockBlockMatrix<T>(N, N, mpi_grid);
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());

    auto V_   = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(N, n, mpi_grid);
    auto W1_  = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(N, n, mpi_grid);
    auto W2_  = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(N, n, mpi_grid);

    T alpha = T(1.0);
    T beta = T(0.0);

    std::size_t offset = 1;
    std::size_t subSize = 2;

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            V_.l_data()[i * V_.l_ld() + j] = getRandomT<T>([&]() { return d(gen); });
        }
    }

    //SH x V = W1 => H x V = SW1 - we assume the standard HEMM works.
    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, SH_, V_, &beta, W1_, offset, subSize);
    
    //H x V = W2 => SH x V = SW2
    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H_, V_, &beta, W2_, offset, subSize);

    //We check that SW2 = W1
    chase::linalg::internal::cpu_mpi::flipLowerHalfMatrixSign(W2_); //We assume the flipping function works

    for(auto i = 0; i < W1_.l_cols(); i++)
    {
	if(i >= offset && i < offset+subSize)
	{
        	for(auto j = 0; j < W1_.l_rows(); j++)
        	{
	    		EXPECT_EQ(W1_.l_data()[i * W1_.l_ld() + j], W2_.l_data()[i * W2_.l_ld() + j]);
        	}
	}
    }
}
