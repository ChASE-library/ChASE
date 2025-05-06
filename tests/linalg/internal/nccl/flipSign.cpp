// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/nccl/flipSign.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

// Global static resources that persist across all test suites
namespace {
    bool resources_initialized = false;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid;
}

template <typename T>
class flipSignGPUNCCLDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);        
        ASSERT_EQ(world_size, 4);  // Ensure we're running with 4 processes
        
        if (!resources_initialized) {
            mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);
            resources_initialized = true;
        }
    }

    void TearDown() override {
        // Don't free resources here - they will be reused
    }

    int world_rank;
    int world_size;    
};

// Add a global test environment to handle resource cleanup at program exit
class ResourceCleanupEnvironment : public ::testing::Environment {
public:
    ~ResourceCleanupEnvironment() override {
        if (resources_initialized) {
            mpi_grid.reset();
            resources_initialized = false;
        }
    }
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(flipSignGPUNCCLDistTest, TestTypes);

TYPED_TEST(flipSignGPUNCCLDistTest, FlipSignCorrectnessGPU) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 10;

    auto H = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(N, N, mpi_grid);
    H.allocate_cpu_data();

    for(auto i = 0; i < H.l_rows(); i++)
    {
    	for(auto j = 0; j < H.l_cols(); j++)
        {
        	H.cpu_data()[i + j * H.cpu_ld()] = T(1.0);
        }
    }

    H.H2D();

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(H);

    H.D2H();

    for(auto i = 0; i < H.l_rows(); i++)
    {
    	for(auto j = 0; j < H.l_cols(); j++)
        {
		if(H.g_offs()[0] + i >= H.g_rows() / 2)
           	{ 
        		EXPECT_EQ(H.cpu_data()[i + j * H.cpu_ld()], T(-1.0));
		}
		else
		{
        		EXPECT_EQ(H.cpu_data()[i + j * H.cpu_ld()], T(1.0));
		}
        }
    }
}

// Add this at the end of the file, before main()
::testing::Environment* const resource_env = ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);
