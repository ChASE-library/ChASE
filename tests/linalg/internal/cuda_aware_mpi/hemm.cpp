#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda_aware_mpi/hemm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

template <typename T>
class HEMMGPUDistTest : public ::testing::Test {
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
TYPED_TEST_SUITE(HEMMGPUDistTest, TestTypes);

TYPED_TEST(HEMMGPUDistTest, HEMMDistCorrectnessCUDAAwareGPU) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 10;
    std::size_t n = 4;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(N, N, this->mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(N, n, this->mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(N, n, this->mpi_grid);    
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

    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(this->cublasH_, &alpha, H_, V_, &beta, W_, offset, subSize);
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

    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(this->cublasH_, &alpha, H_, W_, &beta, V_, offset, subSize);

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


TYPED_TEST(HEMMGPUDistTest, HEMMRedistribeDistCorrectness) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 10;
    std::size_t n = 4;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(N, N, this->mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(N, n, this->mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(N, n, this->mpi_grid);
    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(N, n, this->mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(N, n, this->mpi_grid);

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
    
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(this->cublasH_, H_, V_, W_, V2_, W2_, offset, subSize);

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
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectorsAndRedistribute(this->cublasH_, H_, W_, V_, W2_, V2_, offset, subSize);
    
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
