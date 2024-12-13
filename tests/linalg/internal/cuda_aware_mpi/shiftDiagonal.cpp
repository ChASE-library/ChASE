// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda_aware_mpi/shiftDiagonal.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

template <typename T>
class shiftDiagonalGPUDistTest : public ::testing::Test {
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
TYPED_TEST_SUITE(shiftDiagonalGPUDistTest, TestTypes);

TYPED_TEST(shiftDiagonalGPUDistTest, ShiftDistCorrectnessGPU) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 10;
    std::size_t n = 4;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(N, N, mpi_grid);
    H.allocate_cpu_data();

    if(this->world_rank == 0 || this->world_rank == 3){
        for(auto i = 0; i < H.l_rows(); i++)
        {
            for(auto j = 0; j < H.l_cols(); j++)
            {
                //assume is squared grid
                if(i == j)
                {
                    H.cpu_data()[i + i * H.cpu_ld()] = T(1.0);
                }
            }
        }
    }

    H.H2D();

    std::vector<std::size_t> diag_xoffs, diag_yoffs;

    std::size_t *g_offs = H.g_offs();

    for(auto j = 0; j < H.l_cols(); j++)
    {
        for(auto i = 0; i < H.l_rows(); i++)
        {
            if(g_offs[0] + i == g_offs[1] + j)
            {
                diag_xoffs.push_back(i);
                diag_yoffs.push_back(j);
            }
        }
    }

    std::size_t off_cnt = diag_xoffs.size();

    std::size_t *d_diag_xoffs, *d_diag_yoffs;
    //std::size_t *d_diag_offs;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_diag_xoffs, sizeof(std::size_t) * off_cnt));    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_diag_yoffs, sizeof(std::size_t) * off_cnt));    

    CHECK_CUDA_ERROR(cudaMemcpy(d_diag_xoffs, diag_xoffs.data(), sizeof(std::size_t) * off_cnt , cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_diag_yoffs, diag_yoffs.data(), sizeof(std::size_t) * off_cnt , cudaMemcpyHostToDevice));

    chase::linalg::internal::cuda_mpi::shiftDiagonal(H, d_diag_xoffs, d_diag_yoffs, off_cnt, chase::Base<T>(-5.0));

    H.D2H();

    if(this->world_rank == 0 || this->world_rank == 3){
        for(auto i = 0; i < H.l_rows(); i++)
        {
            for(auto j = 0; j < H.l_cols(); j++)
            {
                //assume is squared grid
                if(i == j)
                {
                    EXPECT_EQ(H.cpu_data()[i + i * H.cpu_ld()], T(-4.0));
                }else
                {
                    EXPECT_EQ(H.cpu_data()[i + j * H.cpu_ld()], T(0.0));
                }
            }
        }
    }
    else
    {
        for(auto i = 0; i < H.cpu_ld() * H.l_cols(); i++)
        {
            EXPECT_EQ(H.cpu_data()[i], T(0.0));
        }
    }  
   
   CHECK_CUDA_ERROR(cudaFree(d_diag_xoffs));     
   CHECK_CUDA_ERROR(cudaFree(d_diag_yoffs));     
}