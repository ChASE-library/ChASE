// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "grid/mpiGrid2D.hpp"
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <mpi.h>
#ifdef HAS_NCCL
#include "Impl/chase_gpu/cuda_utils.hpp"
#include "grid/nccl_utils.hpp"
#endif

class MpiGrid2DTest : public ::testing::Test
{
protected:
    // Setup and teardown
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    void TearDown() override {}

    int world_rank;
    int world_size;
};

// Test for RowMajor Grid Creation
TEST_F(MpiGrid2DTest, RowMajorGridCreation)
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes
    chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor> grid(
        2, 2, MPI_COMM_WORLD);

    // Test that row communicator and column communicator are created
    MPI_Comm row_comm = grid.get_row_comm();
    MPI_Comm col_comm = grid.get_col_comm();

    ASSERT_NE(row_comm, MPI_COMM_NULL);
    ASSERT_NE(col_comm, MPI_COMM_NULL);

    // Test grid dimensions
    int* dims = grid.get_dims();
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 2);

    // Test grid major
    EXPECT_EQ(grid.getGridMajor(), chase::grid::GridMajor::RowMajor);

    // Test grid coordinates
    int* coords = grid.get_coords();
    ASSERT_NE(coords, nullptr);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    // Check row and column ranks are consistent with coordinates
    EXPECT_EQ(coords[0], col_rank);
    EXPECT_EQ(coords[1], row_rank);

    if (world_rank == 0)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 0);
    }

    if (world_rank == 1)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 1);
    }

    if (world_rank == 2)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 0);
    }

    if (world_rank == 3)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 1);
    }
}

// Test for ColMajor Grid Creation
TEST_F(MpiGrid2DTest, ColMajorGridCreation)
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes

    chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor> grid(
        2, 2, MPI_COMM_WORLD);

    // Test that row communicator and column communicator are created
    MPI_Comm row_comm = grid.get_row_comm();
    MPI_Comm col_comm = grid.get_col_comm();

    ASSERT_NE(row_comm, MPI_COMM_NULL);
    ASSERT_NE(col_comm, MPI_COMM_NULL);

    // Test grid dimensions
    int* dims = grid.get_dims();
    EXPECT_EQ(dims[0], 2);
    EXPECT_EQ(dims[1], 2);

    // Test grid major
    EXPECT_EQ(grid.getGridMajor(), chase::grid::GridMajor::ColMajor);

    // Test grid coordinates
    int* coords = grid.get_coords();
    ASSERT_NE(coords, nullptr);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    // Check row and column ranks are consistent with coordinates
    EXPECT_EQ(coords[1], row_rank);
    EXPECT_EQ(coords[0], col_rank);

    if (world_rank == 0)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 0);
    }

    if (world_rank == 1)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 0);
    }

    if (world_rank == 2)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 1);
    }

    if (world_rank == 3)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 1);
    }
}

// Test for auto dimensions creation using MPI_Dims_create
TEST_F(MpiGrid2DTest, AutoGridDimensionCreation)
{
    // Automatically create dimensions based on available processes
    chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor> grid(
        MPI_COMM_WORLD);

    // Test grid dimensions were automatically created
    int* dims = grid.get_dims();
    ASSERT_EQ(dims[0] * dims[1],
              world_size); // Check that total processes match

    // Test row and column communicators
    MPI_Comm row_comm = grid.get_row_comm();
    MPI_Comm col_comm = grid.get_col_comm();
    ASSERT_NE(row_comm, MPI_COMM_NULL);
    ASSERT_NE(col_comm, MPI_COMM_NULL);
}

#ifdef HAS_NCCL
// Test for RowMajor Grid Creation
TEST_F(MpiGrid2DTest, RowMajorGridCreationNCCLSquaredGrid)
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes
    chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor> grid(
        2, 2, MPI_COMM_WORLD);

    // Test that row communicator and column communicator are created
    ncclComm_t nccl_row_comm = grid.get_nccl_row_comm();
    ncclComm_t nccl_col_comm = grid.get_nccl_col_comm();
    ncclComm_t nccl_comm = grid.get_nccl_comm();

    int nccl_world_size = 0;
    int nccl_row_size = 0;
    int nccl_col_size = 0;

    CHECK_NCCL_ERROR(ncclCommCount(nccl_comm, &nccl_world_size));
    EXPECT_EQ(nccl_world_size, 4);
    CHECK_NCCL_ERROR(ncclCommCount(nccl_row_comm, &nccl_row_size));
    EXPECT_EQ(nccl_row_size, 2);
    CHECK_NCCL_ERROR(ncclCommCount(nccl_col_comm, &nccl_col_size));
    EXPECT_EQ(nccl_col_size, 2);

    int my_nccl_rank = 0;
    int my_nccl_row_rank = 0;
    int my_nccl_col_rank = 0;

    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_comm, &my_nccl_rank));
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_row_comm, &my_nccl_row_rank));
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &my_nccl_col_rank));

    EXPECT_EQ(my_nccl_rank, this->world_rank);

    if (this->world_rank == 0)
    {
        EXPECT_EQ(my_nccl_row_rank, 0);
        EXPECT_EQ(my_nccl_col_rank, 0);
    }
    else if (this->world_rank == 1)
    {
        EXPECT_EQ(my_nccl_row_rank, 1);
        EXPECT_EQ(my_nccl_col_rank, 0);
    }
    else if (this->world_rank == 2)
    {
        EXPECT_EQ(my_nccl_row_rank, 0);
        EXPECT_EQ(my_nccl_col_rank, 1);
    }
    else
    {
        EXPECT_EQ(my_nccl_row_rank, 1);
        EXPECT_EQ(my_nccl_col_rank, 1);
    }
}

TEST_F(MpiGrid2DTest, ColMajorGridCreationNCCLSquaredGrid)
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes
    chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor> grid(
        2, 2, MPI_COMM_WORLD);

    // Test that row communicator and column communicator are created
    ncclComm_t nccl_row_comm = grid.get_nccl_row_comm();
    ncclComm_t nccl_col_comm = grid.get_nccl_col_comm();
    ncclComm_t nccl_comm = grid.get_nccl_comm();

    int nccl_world_size = 0;
    int nccl_row_size = 0;
    int nccl_col_size = 0;

    CHECK_NCCL_ERROR(ncclCommCount(nccl_comm, &nccl_world_size));
    EXPECT_EQ(nccl_world_size, 4);
    CHECK_NCCL_ERROR(ncclCommCount(nccl_row_comm, &nccl_row_size));
    EXPECT_EQ(nccl_row_size, 2);
    CHECK_NCCL_ERROR(ncclCommCount(nccl_col_comm, &nccl_col_size));
    EXPECT_EQ(nccl_col_size, 2);

    int my_nccl_rank = 0;
    int my_nccl_row_rank = 0;
    int my_nccl_col_rank = 0;

    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_comm, &my_nccl_rank));
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_row_comm, &my_nccl_row_rank));
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &my_nccl_col_rank));

    EXPECT_EQ(my_nccl_rank, this->world_rank);

    if (this->world_rank == 0)
    {
        EXPECT_EQ(my_nccl_row_rank, 0);
        EXPECT_EQ(my_nccl_col_rank, 0);
    }
    else if (this->world_rank == 1)
    {
        EXPECT_EQ(my_nccl_row_rank, 0);
        EXPECT_EQ(my_nccl_col_rank, 1);
    }
    else if (this->world_rank == 2)
    {
        EXPECT_EQ(my_nccl_row_rank, 1);
        EXPECT_EQ(my_nccl_col_rank, 0);
    }
    else
    {
        EXPECT_EQ(my_nccl_row_rank, 1);
        EXPECT_EQ(my_nccl_col_rank, 1);
    }
}

TEST_F(MpiGrid2DTest, ColMajorGridCreationNCCLNonSquaredGrid)
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes
    chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor> grid(
        4, 1, MPI_COMM_WORLD);

    // Test that row communicator and column communicator are created
    ncclComm_t nccl_row_comm = grid.get_nccl_row_comm();
    ncclComm_t nccl_col_comm = grid.get_nccl_col_comm();
    ncclComm_t nccl_comm = grid.get_nccl_comm();

    int nccl_world_size = 0;
    int nccl_row_size = 0;
    int nccl_col_size = 0;

    CHECK_NCCL_ERROR(ncclCommCount(nccl_comm, &nccl_world_size));
    EXPECT_EQ(nccl_world_size, 4);
    CHECK_NCCL_ERROR(ncclCommCount(nccl_row_comm, &nccl_row_size));
    EXPECT_EQ(nccl_row_size, 1);
    CHECK_NCCL_ERROR(ncclCommCount(nccl_col_comm, &nccl_col_size));
    EXPECT_EQ(nccl_col_size, 4);

    int my_nccl_rank = 0;
    int my_nccl_row_rank = 0;
    int my_nccl_col_rank = 0;

    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_comm, &my_nccl_rank));
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_row_comm, &my_nccl_row_rank));
    CHECK_NCCL_ERROR(ncclCommUserRank(nccl_col_comm, &my_nccl_col_rank));

    EXPECT_EQ(my_nccl_rank, this->world_rank);
    EXPECT_EQ(my_nccl_row_rank, 0);
    EXPECT_EQ(my_nccl_col_rank, this->world_rank);
}

TEST_F(MpiGrid2DTest, NCCLAllreduceWrapper)
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes
    chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor> grid(
        2, 2, MPI_COMM_WORLD);

    // Test that row communicator and column communicator are created
    ncclComm_t nccl_row_comm = grid.get_nccl_row_comm();
    ncclComm_t nccl_col_comm = grid.get_nccl_col_comm();
    ncclComm_t nccl_comm = grid.get_nccl_comm();
    MPI_Comm comm = grid.get_comm();
    MPI_Comm row_comm = grid.get_row_comm();
    MPI_Comm col_comm = grid.get_col_comm();

    std::size_t data_count = 4;

    // float
    std::vector<float> data_sp(data_count);
    std::vector<float> data_sp_2(data_count);
    for (auto i = 0; i < data_count; i++)
    {
        data_sp[i] = i * 0.5;
    }

    float* data_sp_device;
    CHECK_CUDA_ERROR(cudaMalloc(&data_sp_device, data_count * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(data_sp_device, data_sp.data(),
                                data_count * sizeof(float),
                                cudaMemcpyHostToDevice));
    MPI_Allreduce(MPI_IN_PLACE, data_sp.data(), data_count, MPI_FLOAT, MPI_SUM,
                  comm);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        data_sp_device, data_sp_device, data_count, ncclSum, nccl_comm));
    CHECK_CUDA_ERROR(cudaMemcpy(data_sp_2.data(), data_sp_device,
                                data_count * sizeof(float),
                                cudaMemcpyDeviceToHost));
    for (auto i = 0; i < data_count; i++)
    {
        EXPECT_EQ(data_sp[i], data_sp_2[i]);
    }
    CHECK_CUDA_ERROR(cudaFree(data_sp_device));

    std::vector<double> data_dp(data_count);
    std::vector<double> data_dp_2(data_count);
    for (auto i = 0; i < data_count; i++)
    {
        data_sp[i] = i * 0.5;
    }

    double* data_dp_device;
    CHECK_CUDA_ERROR(cudaMalloc(&data_dp_device, data_count * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMemcpy(data_dp_device, data_dp.data(),
                                data_count * sizeof(double),
                                cudaMemcpyHostToDevice));
    MPI_Allreduce(MPI_IN_PLACE, data_dp.data(), data_count, MPI_DOUBLE, MPI_SUM,
                  comm);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        data_dp_device, data_dp_device, data_count, ncclSum, nccl_comm));
    CHECK_CUDA_ERROR(cudaMemcpy(data_dp_2.data(), data_dp_device,
                                data_count * sizeof(double),
                                cudaMemcpyDeviceToHost));
    for (auto i = 0; i < data_count; i++)
    {
        EXPECT_EQ(data_dp[i], data_dp_2[i]);
    }
    CHECK_CUDA_ERROR(cudaFree(data_dp_device));
    // complex float
    std::vector<std::complex<float>> data_cp(data_count);
    std::vector<std::complex<float>> data_cp_2(data_count);
    for (auto i = 0; i < data_count; i++)
    {
        data_cp[i] = std::complex<float>(i * 0.5, i * 0.2);
    }

    std::complex<float>* data_cp_device;
    CHECK_CUDA_ERROR(
        cudaMalloc(&data_cp_device, data_count * sizeof(std::complex<float>)));
    CHECK_CUDA_ERROR(cudaMemcpy(data_cp_device, data_cp.data(),
                                data_count * sizeof(std::complex<float>),
                                cudaMemcpyHostToDevice));
    MPI_Allreduce(MPI_IN_PLACE, data_cp.data(), data_count, MPI_COMPLEX,
                  MPI_SUM, comm);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        data_cp_device, data_cp_device, data_count, ncclSum, nccl_comm));
    CHECK_CUDA_ERROR(cudaMemcpy(data_cp_2.data(), data_cp_device,
                                data_count * sizeof(std::complex<float>),
                                cudaMemcpyDeviceToHost));
    for (auto i = 0; i < data_count; i++)
    {
        EXPECT_EQ(data_cp[i], data_cp_2[i]);
    }
    CHECK_CUDA_ERROR(cudaFree(data_cp_device));

    // complex double
    std::vector<std::complex<double>> data_zp(data_count);
    std::vector<std::complex<double>> data_zp_2(data_count);
    for (auto i = 0; i < data_count; i++)
    {
        data_zp[i] = std::complex<double>(i * 0.5, i * 0.2);
    }

    std::complex<double>* data_zp_device;
    CHECK_CUDA_ERROR(
        cudaMalloc(&data_zp_device, data_count * sizeof(std::complex<double>)));
    CHECK_CUDA_ERROR(cudaMemcpy(data_zp_device, data_zp.data(),
                                data_count * sizeof(std::complex<double>),
                                cudaMemcpyHostToDevice));
    MPI_Allreduce(MPI_IN_PLACE, data_zp.data(), data_count, MPI_DOUBLE_COMPLEX,
                  MPI_SUM, comm);
    CHECK_NCCL_ERROR(chase::nccl::ncclAllReduceWrapper(
        data_zp_device, data_zp_device, data_count, ncclSum, nccl_comm));
    CHECK_CUDA_ERROR(cudaMemcpy(data_zp_2.data(), data_zp_device,
                                data_count * sizeof(std::complex<double>),
                                cudaMemcpyDeviceToHost));
    for (auto i = 0; i < data_count; i++)
    {
        EXPECT_EQ(data_zp[i], data_zp_2[i]);
    }
    CHECK_CUDA_ERROR(cudaFree(data_zp_device));
}

TEST_F(MpiGrid2DTest, NCCLBcasteWrapper)
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes
    chase::grid::MpiGrid2D<chase::grid::GridMajor::RowMajor> grid(
        2, 2, MPI_COMM_WORLD);

    // Test that row communicator and column communicator are created
    ncclComm_t nccl_row_comm = grid.get_nccl_row_comm();
    ncclComm_t nccl_col_comm = grid.get_nccl_col_comm();
    ncclComm_t nccl_comm = grid.get_nccl_comm();
    MPI_Comm comm = grid.get_comm();
    MPI_Comm row_comm = grid.get_row_comm();
    MPI_Comm col_comm = grid.get_col_comm();

    std::size_t data_count = 4;

    // float
    std::vector<float> data_sp(data_count);
    if (this->world_rank == 0)
    {
        for (auto i = 0; i < data_count; i++)
        {
            data_sp[i] = i * 0.5;
        }
    }

    float* data_sp_device;
    CHECK_CUDA_ERROR(cudaMalloc(&data_sp_device, data_count * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMemcpy(data_sp_device, data_sp.data(),
                                data_count * sizeof(float),
                                cudaMemcpyHostToDevice));

    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(data_sp_device, data_count,
                                                   0, nccl_comm));
    CHECK_CUDA_ERROR(cudaMemcpy(data_sp.data(), data_sp_device,
                                data_count * sizeof(float),
                                cudaMemcpyDeviceToHost));

    for (auto i = 0; i < data_count; i++)
    {
        EXPECT_EQ(data_sp[i], i * 0.5);
    }

    // float
    std::vector<std::complex<double>> data_zp(data_count);
    if (this->world_rank == 0)
    {
        for (auto i = 0; i < data_count; i++)
        {
            data_zp[i] = std::complex<double>(i * 0.5, i * 0.1);
        }
    }

    std::complex<double>* data_zp_device;
    CHECK_CUDA_ERROR(
        cudaMalloc(&data_zp_device, data_count * sizeof(std::complex<double>)));
    CHECK_CUDA_ERROR(cudaMemcpy(data_zp_device, data_zp.data(),
                                data_count * sizeof(std::complex<double>),
                                cudaMemcpyHostToDevice));

    CHECK_NCCL_ERROR(chase::nccl::ncclBcastWrapper(data_zp_device, data_count,
                                                   0, nccl_comm));
    CHECK_CUDA_ERROR(cudaMemcpy(data_zp.data(), data_zp_device,
                                data_count * sizeof(std::complex<double>),
                                cudaMemcpyDeviceToHost));

    for (auto i = 0; i < data_count; i++)
    {
        EXPECT_EQ(data_zp[i], std::complex<double>(i * 0.5, i * 0.1));
    }
}
#endif