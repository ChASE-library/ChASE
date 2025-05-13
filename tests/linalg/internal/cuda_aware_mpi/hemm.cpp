// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda_aware_mpi/hemm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"


// Global static buffers that persist across all test suites
namespace {
    bool resources_initialized = false;
    std::complex<double> *H_buff = nullptr;
    std::complex<double> *V_buff = nullptr;
    std::complex<double> *W_buff = nullptr;
    std::complex<double> *V2_buff = nullptr;
    std::complex<double> *W2_buff = nullptr;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid;
    cublasHandle_t cublasH;
}

template <typename T>
class HEMMGPUDistTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize resources only once for all test suites
        if (!resources_initialized) {
            // Initialize MPI grid
            int world_rank, world_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            ASSERT_EQ(world_size, 4);  // Ensure we're running with 4 processes
            mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

            // Initialize cuBLAS handle
            CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));

            // Allocate GPU buffers
            CHECK_CUDA_ERROR(cudaMalloc(&H_buff, m * m * sizeof(std::complex<double>)));
            CHECK_CUDA_ERROR(cudaMalloc(&V_buff, m * n * sizeof(std::complex<double>)));
            CHECK_CUDA_ERROR(cudaMalloc(&W_buff, m * n * sizeof(std::complex<double>)));
            CHECK_CUDA_ERROR(cudaMalloc(&V2_buff, m * n * sizeof(std::complex<double>)));
            CHECK_CUDA_ERROR(cudaMalloc(&W2_buff, m * n * sizeof(std::complex<double>)));

            resources_initialized = true;
        }
    }

    static void TearDownTestSuite() {
        // Don't free resources here - they should persist across test suites
    }

    void SetUp() override {
        // No need to initialize resources here as they are already initialized
    }

    void TearDown() override {
        // No need to clean up resources here as they persist
    }

    static constexpr std::size_t N = 10;
    static constexpr std::size_t n = 4;
    static constexpr std::size_t m = 5;

    // Accessors for the global resources
    static std::complex<double>* get_H_buff() { return H_buff; }
    static std::complex<double>* get_V_buff() { return V_buff; }
    static std::complex<double>* get_W_buff() { return W_buff; }
    static std::complex<double>* get_V2_buff() { return V2_buff; }
    static std::complex<double>* get_W2_buff() { return W2_buff; }
    static std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> get_mpi_grid() { return mpi_grid; }
    static cublasHandle_t get_cublas_handle() { return cublasH; }

    int world_rank;
    int world_size;
};

// Add a global test environment to handle resource cleanup at program exit
class ResourceCleanupEnvironment : public ::testing::Environment {
public:
    ~ResourceCleanupEnvironment() override {
        if (resources_initialized) {
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH));
            CHECK_CUDA_ERROR(cudaFree(H_buff));
            CHECK_CUDA_ERROR(cudaFree(V_buff));
            CHECK_CUDA_ERROR(cudaFree(W_buff));
            CHECK_CUDA_ERROR(cudaFree(V2_buff));
            CHECK_CUDA_ERROR(cudaFree(W2_buff));

            // Reset the mpi_grid shared_ptr before program exit
            // This ensures the MpiGrid2D destructor is called only once
            mpi_grid.reset();
                        
            resources_initialized = false;
            std::cout << "Resources freed at program exit" << std::endl;
        }
    }
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(HEMMGPUDistTest, TestTypes);

TYPED_TEST(HEMMGPUDistTest, HEMMDistCorrectnessCUDAAwareGPU) {
    using T = TypeParam;  // Get the current type

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_buff()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->m, this->n, this->m, reinterpret_cast<T*>(this->get_V_buff()), this->get_mpi_grid());
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->m, this->n, this->m, reinterpret_cast<T*>(this->get_W_buff()), this->get_mpi_grid());    
    
    H_.allocate_cpu_data();
    V_.allocate_cpu_data();
    W_.allocate_cpu_data();

    T alpha = T(2.0);
    T beta = T(3.0);
    std::size_t offset = 1;
    std::size_t subSize = 2;


    for(auto i = 0; i < H_.l_cols(); i++)
    {
        for(auto j = 0; j < H_.l_rows(); j++)
        {
            H_.cpu_data()[i * H_.cpu_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.cpu_ld() + j] = T(2.0);
        }
    }

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            W_.cpu_data()[i * W_.cpu_ld() + j] = T(3.0);
        }
    }

    H_.H2D();
    V_.H2D();
    W_.H2D();

    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(this->get_cublas_handle(), &alpha, H_, V_, &beta, W_, offset, subSize);
    W_.D2H();

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(49));
            }
            else
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(3));
            }            
        }
    }

    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(this->get_cublas_handle(), &alpha, H_, W_, &beta, V_, offset, subSize);

    V_.D2H();

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(986));
            }
            else
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(2));
            }           
        }
    }


}


TYPED_TEST(HEMMGPUDistTest, HEMMRedistribeAsyncDistCorrectness) {
    using T = TypeParam;  // Get the current type

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_buff()), this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->m, this->n, this->m, reinterpret_cast<T*>(this->get_V_buff()), this->get_mpi_grid());
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->m, this->n, this->m, reinterpret_cast<T*>(this->get_W_buff()), this->get_mpi_grid());    
        
    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->m, this->n, this->m, reinterpret_cast<T*>(this->get_V2_buff()), this->get_mpi_grid());
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->m, this->n, this->m, reinterpret_cast<T*>(this->get_W2_buff()), this->get_mpi_grid());    
    
    H_.allocate_cpu_data();
    V_.allocate_cpu_data();
    W_.allocate_cpu_data();
    V2_.allocate_cpu_data();
    W2_.allocate_cpu_data();

    std::size_t offset = 1;
    std::size_t subSize = 2;

    for(auto i = 0; i < H_.l_cols(); i++)
    {
        for(auto j = 0; j < H_.l_rows(); j++)
        {
            H_.cpu_data()[i * H_.cpu_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.cpu_ld() + j] = T(2.0);
            V2_.cpu_data()[i * V2_.cpu_ld() + j] = V_.cpu_data()[i * V_.cpu_ld() + j];
        }
    }

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            W_.cpu_data()[i * W_.cpu_ld() + j] = T(3.0);
            W2_.cpu_data()[i * W_.cpu_ld() + j] = T(-1.0);
        }
    }

    H_.H2D();
    V_.H2D();
    W_.H2D();
    V2_.H2D();
    W2_.H2D();    
    
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(this->get_cublas_handle(), H_, V_, W_, V2_, W2_, offset, subSize);

    W_.D2H();
    W2_.D2H();  

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(20));
            }
            else
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(3));
            }
            
        }
    }

    for(auto i = 0; i < W2_.l_cols(); i++)
    {
        for(auto j = 0; j < W2_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(W2_.cpu_data()[i * W2_.cpu_ld() + j], T(2.0));
            }
            else
            {
                EXPECT_EQ(W2_.cpu_data()[i * W2_.cpu_ld() + j], T(-1.0));
            }
            W2_.cpu_data()[i * W2_.cpu_ld() + j] = W_.cpu_data()[i * W2_.cpu_ld() + j];
        }
    }
    W2_.H2D();
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(this->get_cublas_handle(), H_, W_, V_, W2_, V2_, offset, subSize);
    
    V_.D2H();
    V2_.D2H();  


    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(200));
            }
            else
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(2));
            }
            
        }
    }

    for(auto i = 0; i < V2_.l_cols(); i++)
    {
        for(auto j = 0; j < V2_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(V2_.cpu_data()[i * V2_.cpu_ld() + j], T(20));
            }
            else
            {
                EXPECT_EQ(V2_.cpu_data()[i * V2_.cpu_ld() + j], T(2));
            }
            
        }
    }
}

// Add this at the end of the file, before main()
::testing::Environment* const resource_env = ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);


