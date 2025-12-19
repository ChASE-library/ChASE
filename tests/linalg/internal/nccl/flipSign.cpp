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

    std::size_t N = 100, offset = 10, subSize = 90;

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

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(H, offset, subSize);

    H.D2H();

    for(auto i = 0; i < H.l_rows(); i++)
    {
    	for(auto j = 0; j < H.l_cols(); j++)
        {
		if(H.g_offs()[0] + i < H.g_rows() / 2 || j < offset)
           	{ 
        		EXPECT_EQ(H.cpu_data()[i + j * H.cpu_ld()], T(1.0));
		}
		else
		{
        		EXPECT_EQ(H.cpu_data()[i + j * H.cpu_ld()], -T(1.0));
		}
        }
    }
}

TYPED_TEST(flipSignGPUNCCLDistTest, FlipSignBlockCyclicCorrectnessGPU) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 100, mb = 20;

    auto H = chase::distMatrix::PseudoHermitianBlockCyclicMatrix<T, chase::platform::GPU>(N, N, mb, mb, mpi_grid);
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

    for(auto idx_i = 0; idx_i < H.mblocks(); idx_i++)
    {
    	for(auto idx_j = 0; idx_j < H.nblocks(); idx_j++)
    	{
		for(auto i = 0; i < H.m_contiguous_lens()[idx_i]; i++)
		{
			for(auto j = 0; j < H.n_contiguous_lens()[idx_j]; j++)
			{
				if(i + H.m_contiguous_global_offs()[idx_i] < (N / 2)){
                			EXPECT_EQ(H.cpu_data()[(j + H.n_contiguous_local_offs()[idx_j]) * H.cpu_ld() + i + H.m_contiguous_local_offs()[idx_i]], T(1.0));
				}else{
                			EXPECT_EQ(H.cpu_data()[(j + H.n_contiguous_local_offs()[idx_j]) * H.cpu_ld() + i + H.m_contiguous_local_offs()[idx_i]], T(-1.0));
				}
			}
		}
    	}
    }
}

TYPED_TEST(flipSignGPUNCCLDistTest, FlipSignColMultiVectorsBlockBlockCorrectnessGPU) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 100, n = 60, offset = 20, subSize = 40;

    auto V = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(N, n, mpi_grid);
    V.allocate_cpu_data();

    for(auto i = 0; i < V.l_rows(); i++)
    {
    	for(auto j = 0; j < V.l_cols(); j++)
        {
        	V.cpu_data()[i + j * V.cpu_ld()] = T(1.0);
        }
    }

    V.H2D();

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(V, offset, subSize);

    V.D2H();

    for(auto i = 0; i < V.l_rows(); i++)
    {
    	for(auto j = 0; j < V.l_cols(); j++)
        {
		if(V.g_off() + i < V.g_rows() / 2 || j < offset)
           	{ 
        		EXPECT_EQ(V.cpu_data()[i + j * V.cpu_ld()], T(1.0));
		}
		else
		{
        		EXPECT_EQ(V.cpu_data()[i + j * V.cpu_ld()], -T(1.0));
		}
        }
    }

}

TYPED_TEST(flipSignGPUNCCLDistTest, FlipSignRowMultiVectorsBlockBlockCorrectnessGPU) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 100, n = 60, offset = 20, subSize = 40;

    auto V = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(N, n, mpi_grid);
    V.allocate_cpu_data();

    for(auto i = 0; i < V.l_rows(); i++)
    {
    	for(auto j = 0; j < V.l_cols(); j++)
        {
        	V.cpu_data()[i + j * V.cpu_ld()] = T(1.0);
        }
    }

    V.H2D();

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(V, offset, subSize);

    V.D2H();

    for(auto i = 0; i < V.l_rows(); i++)
    {
    	for(auto j = 0; j < V.l_cols(); j++)
        {
		if(V.g_off() + i < V.g_rows() / 2 || j < offset)
           	{ 
        		EXPECT_EQ(V.cpu_data()[i + j * V.cpu_ld()], T(1.0));
		}
		else
		{
        		EXPECT_EQ(V.cpu_data()[i + j * V.cpu_ld()], -T(1.0));
		}
        }
    }
}

TYPED_TEST(flipSignGPUNCCLDistTest, FlipSignColMultiVectorsBlockCyclicCorrectnessGPU) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 100, n = 60, mb = 10, offset = 20, subSize = 40;

    auto V = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(N, n, mb, mpi_grid);
    V.allocate_cpu_data();

    for(auto i = 0; i < V.l_rows(); i++)
    {
    	for(auto j = 0; j < V.l_cols(); j++)
        {
        	V.cpu_data()[i + j * V.cpu_ld()] = T(1.0);
        }
    }

    V.H2D();

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(V, offset, subSize);

    V.D2H();

    for(auto idx_i = 0; idx_i < V.mblocks(); idx_i++)
    {
	    for(auto i = 0; i < V.m_contiguous_lens()[idx_i]; i++)
	    {
		for(auto j = 0; j < V.l_cols(); j++)
		{
			if(i + V.m_contiguous_global_offs()[idx_i] < (N / 2) || j < offset){
                		EXPECT_EQ(V.cpu_data()[j * V.cpu_ld() + i + V.m_contiguous_local_offs()[idx_i]], T(1.0));
			}else{
                		EXPECT_EQ(V.cpu_data()[j * V.cpu_ld() + i + V.m_contiguous_local_offs()[idx_i]], T(-1.0));
			}
		}
    	}
    }

}

TYPED_TEST(flipSignGPUNCCLDistTest, FlipSignRowMultiVectorsBlockCyclicCorrectnessGPU) {
    using T = TypeParam;  // Get the current type

    std::size_t N = 100, n = 60, mb = 10, offset = 20, subSize = 40;

    auto V = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(N, n, mb, mpi_grid);
    V.allocate_cpu_data();

    for(auto i = 0; i < V.l_rows(); i++)
    {
    	for(auto j = 0; j < V.l_cols(); j++)
        {
        	V.cpu_data()[i + j * V.cpu_ld()] = T(1.0);
        }
    }

    V.H2D();

    chase::linalg::internal::cuda_nccl::flipLowerHalfMatrixSign(V, offset, subSize);

    V.D2H();

    for(auto idx_i = 0; idx_i < V.mblocks(); idx_i++)
    {
	    for(auto i = 0; i < V.m_contiguous_lens()[idx_i]; i++)
	    {
		for(auto j = 0; j < V.l_cols(); j++)
		{
			if(i + V.m_contiguous_global_offs()[idx_i] < (N / 2) || j < offset){
                		EXPECT_EQ(V.cpu_data()[j * V.cpu_ld() + i + V.m_contiguous_local_offs()[idx_i]], T(1.0));
			}else{
                		EXPECT_EQ(V.cpu_data()[j * V.cpu_ld() + i + V.m_contiguous_local_offs()[idx_i]], T(-1.0));
			}
		}
    	}
    }

}
// Add this at the end of the file, before main()
::testing::Environment* const resource_env = ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);
