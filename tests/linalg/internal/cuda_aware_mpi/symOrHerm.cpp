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
#include "linalg/internal/cuda_aware_mpi/symOrHerm.hpp"

template <typename T>
class SymOrHermGPUDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);  
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));                 
    }

    void TearDown() override {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));        
    }

    int world_rank;
    int world_size;    
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid;
    cublasHandle_t cublasH_;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(SymOrHermGPUDistTest, TestTypes);

TYPED_TEST(SymOrHermGPUDistTest, UpperTriangularMatrix) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 5;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    
    T U[N * N] = {1, 2, 3,  4,  5,
                  0, 6, 7,  8,  9,
                  0, 0, 10, 11, 12,
                  0, 0, 0,  13, 14,
                  0, 0, 0,  0,  15};

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(N, N, this->mpi_grid);
    H.allocate_cpu_data();
    std::size_t *goffs = H.g_offs(); 

    for(auto j = 0; j < H.l_cols(); j++)
    {
        for(auto i = 0; i < H.l_rows(); i++)
        {
            H.cpu_data()[i + j * H.cpu_ld()] = U[i + goffs[0] + (j + goffs[1]) * N];
        }
    }
    H.H2D();
    bool is_sym = chase::linalg::internal::cuda_mpi::checkSymmetryEasy(this->cublasH_, H); 
    EXPECT_FALSE(is_sym);
#ifdef HAS_SCALAPACK
    chase::linalg::internal::cuda_mpi::symOrHermMatrix('U', H);
    H.H2D();
    is_sym = chase::linalg::internal::cuda_mpi::checkSymmetryEasy(this->cublasH_, H);    
    EXPECT_TRUE(is_sym);
#endif    
}

TYPED_TEST(SymOrHermGPUDistTest, LowerTriangularMatrix) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 5;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    
    T U[N * N] = {1, 0, 0,  0,  0,
                  2, 6, 0,  0,  0,
                  3, 7, 10, 0,  0,
                  4, 8, 11, 13, 0,
                  5, 9, 12, 14, 15};

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(N, N, this->mpi_grid);
    H.allocate_cpu_data();
    std::size_t *goffs = H.g_offs(); 

    for(auto j = 0; j < H.l_cols(); j++)
    {
        for(auto i = 0; i < H.l_rows(); i++)
        {
            H.cpu_data()[i + j * H.cpu_ld()] = U[i + goffs[0] + (j + goffs[1]) * N];
        }
    }
    H.H2D();

    bool is_sym = chase::linalg::internal::cuda_mpi::checkSymmetryEasy(this->cublasH_, H); 
    EXPECT_FALSE(is_sym);
#ifdef HAS_SCALAPACK
    chase::linalg::internal::cuda_mpi::symOrHermMatrix('L', H);
    H.H2D();
    is_sym = chase::linalg::internal::cuda_mpi::checkSymmetryEasy(this->cublasH_, H);    
    EXPECT_TRUE(is_sym);
#endif    
}
