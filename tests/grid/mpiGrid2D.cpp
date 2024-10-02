#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <mpi.h>
#include "Impl/grid/mpiGrid2D.hpp"

class MpiGrid2DTest : public ::testing::Test {
protected:
    // Setup and teardown
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }

    void TearDown() override {
    }

    int world_rank;
    int world_size;
};

// Test for RowMajor Grid Creation
TEST_F(MpiGrid2DTest, RowMajorGridCreation) 
{
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4);  // Ensure we're running with 4 processes
    chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::RowMajor> grid(2, 2, MPI_COMM_WORLD);

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
    EXPECT_EQ(grid.getGridMajor(), chase::Impl::mpi::GridMajor::RowMajor);

    // Test grid coordinates
    int* coords = grid.get_coords();
    ASSERT_NE(coords, nullptr);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
    
    // Check row and column ranks are consistent with coordinates
    EXPECT_EQ(coords[0], col_rank);
    EXPECT_EQ(coords[1], row_rank);    

    if(world_rank == 0)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 0);
    }

    if(world_rank == 1)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 1);
    }

    if(world_rank == 2)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 0);
    }

    if(world_rank == 3)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 1);
    }

}

// Test for ColMajor Grid Creation
TEST_F(MpiGrid2DTest, ColMajorGridCreation) {
    // Assuming we want a 2x2 grid for testing (when world_size is 4)
    ASSERT_EQ(world_size, 4);  // Ensure we're running with 4 processes

    chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor> grid(2, 2, MPI_COMM_WORLD);

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
    EXPECT_EQ(grid.getGridMajor(), chase::Impl::mpi::GridMajor::ColMajor);

    // Test grid coordinates
    int* coords = grid.get_coords();
    ASSERT_NE(coords, nullptr);

    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
    
    // Check row and column ranks are consistent with coordinates
    EXPECT_EQ(coords[1], row_rank);  
    EXPECT_EQ(coords[0], col_rank);  

    if(world_rank == 0)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 0);
    }

    if(world_rank == 1)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 0);
    }

    if(world_rank == 2)
    {
        EXPECT_EQ(coords[0], 0);
        EXPECT_EQ(coords[1], 1);
    }

    if(world_rank == 3)
    {
        EXPECT_EQ(coords[0], 1);
        EXPECT_EQ(coords[1], 1);
    }

}

// Test for auto dimensions creation using MPI_Dims_create
TEST_F(MpiGrid2DTest, AutoGridDimensionCreation) {
    // Automatically create dimensions based on available processes
    chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::RowMajor> grid(MPI_COMM_WORLD);

    // Test grid dimensions were automatically created
    int* dims = grid.get_dims();
    ASSERT_EQ(dims[0] * dims[1], world_size);  // Check that total processes match

    // Test row and column communicators
    MPI_Comm row_comm = grid.get_row_comm();
    MPI_Comm col_comm = grid.get_col_comm();
    ASSERT_NE(row_comm, MPI_COMM_NULL);
    ASSERT_NE(col_comm, MPI_COMM_NULL);
}