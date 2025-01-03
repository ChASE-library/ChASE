// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/mpi/shiftDiagonal.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/mpi/symOrHerm.hpp"

template <typename T>
class SymOrHermCPUDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);        
    }

    void TearDown() override {}

    int world_rank;
    int world_size;    
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SymOrHermCPUDistTest, TestTypes);

TYPED_TEST(SymOrHermCPUDistTest, UpperTriangularMatrix) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 5;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);
    
    T U[N * N] = {1, 2, 3,  4,  5,
                  0, 6, 7,  8,  9,
                  0, 0, 10, 11, 12,
                  0, 0, 0,  13, 14,
                  0, 0, 0,  0,  15};

    auto R = chase::distMatrix::RedundantMatrix<T>(N, N, N, U, mpi_grid);
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

TYPED_TEST(SymOrHermCPUDistTest, LowerTriangularMatrix) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 5;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);
    
    T U[N * N] = {1, 0, 0,  0,  0,
                  2, 6, 0,  0,  0,
                  3, 7, 10, 0,  0,
                  4, 8, 11, 13, 0,
                  5, 9, 12, 14, 15};

    auto R = chase::distMatrix::RedundantMatrix<T>(N, N, N, U, mpi_grid);
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