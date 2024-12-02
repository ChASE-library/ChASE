#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/mpi/hemm.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

template <typename T>
class HEMMCPUDistTest : public ::testing::Test {
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
TYPED_TEST_SUITE(HEMMCPUDistTest, TestTypes);


TYPED_TEST(HEMMCPUDistTest, HEMMDistCorrectness) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 10;
    std::size_t n = 4;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    auto H_ = chase::distMatrix::BlockBlockMatrix<T>(N, N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(N, n, mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(N, n, mpi_grid);

    T alpha = T(2.0);
    T beta = T(3.0);
    std::size_t offset = 1;
    std::size_t subSize = 2;

    for(auto i = 0; i < H_.l_cols(); i++)
    {
        for(auto j = 0; j < H_.l_rows(); j++)
        {
            H_.l_data()[i * H_.l_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            V_.l_data()[i * V_.l_ld() + j] = T(2.0);
        }
    }

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            W_.l_data()[i * W_.l_ld() + j] = T(3.0);
        }
    }

    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H_, V_, &beta, W_, offset, subSize);

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(W_.l_data()[i * W_.l_ld() + j], T(49));
            }
            else
            {
                EXPECT_EQ(W_.l_data()[i * W_.l_ld() + j], T(3));
            }
            
        }
    }

    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H_, W_, &beta, V_, offset, subSize);

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(V_.l_data()[i * V_.l_ld() + j], T(986));
            }
            else
            {
                EXPECT_EQ(V_.l_data()[i * V_.l_ld() + j], T(2));
            }           
        }
    }
}

TYPED_TEST(HEMMCPUDistTest, HEMMDistThrow) 
{
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    auto H_ = chase::distMatrix::BlockBlockMatrix<T>(10, 8, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(16, 4, mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(12, 4, mpi_grid);

    T alpha = T(1.0);
    T beta = T(2.0);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H_, W_, &beta, V_);}, std::runtime_error);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H_, V_, &beta, W_);}, std::runtime_error);

    auto H2_ = chase::distMatrix::BlockBlockMatrix<T>(10, 10, mpi_grid);
    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(10, 4, mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(10, 4, mpi_grid);

    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H2_, W2_, &beta, V2_, 100, 2);}, std::invalid_argument);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H2_, W2_, &beta, V2_, 0, 0);}, std::invalid_argument);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H2_, W2_, &beta, V2_, 0, 100);}, std::invalid_argument);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H2_, W2_, &beta, V2_, 0, 5);}, std::invalid_argument);
    EXPECT_NO_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H2_, W2_, &beta, V2_, 0, 4);});
}


TYPED_TEST(HEMMCPUDistTest, HEMMRedistribeAsyncDistCorrectness) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 10;
    std::size_t n = 4;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    auto H_ = chase::distMatrix::BlockBlockMatrix<T>(N, N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(N, n, mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(N, n, mpi_grid);
    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(N, n, mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(N, n, mpi_grid);

    std::size_t offset = 1;
    std::size_t subSize = 2;

    for(auto i = 0; i < H_.l_cols(); i++)
    {
        for(auto j = 0; j < H_.l_rows(); j++)
        {
            H_.l_data()[i * H_.l_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            V_.l_data()[i * V_.l_ld() + j] = T(2.0);
            V2_.l_data()[i * V2_.l_ld() + j] = V_.l_data()[i * V_.l_ld() + j];
        }
    }

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            W_.l_data()[i * W_.l_ld() + j] = T(3.0);
            W2_.l_data()[i * W_.l_ld() + j] = T(-1.0);
        }
    }

    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H_, V_, W_, V2_, W2_, offset, subSize);

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(W_.l_data()[i * W_.l_ld() + j], T(20));
            }
            else
            {
                EXPECT_EQ(W_.l_data()[i * W_.l_ld() + j], T(3));
            }
            
        }
    }

    for(auto i = 0; i < W2_.l_cols(); i++)
    {
        for(auto j = 0; j < W2_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(W2_.l_data()[i * W2_.l_ld() + j], T(2.0));
            }
            else
            {
                EXPECT_EQ(W2_.l_data()[i * W2_.l_ld() + j], T(-1.0));
            }
            W2_.l_data()[i * W2_.l_ld() + j] = W_.l_data()[i * W2_.l_ld() + j];
        }
    }

    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H_, W_, V_, W2_, V2_, offset, subSize);

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(V_.l_data()[i * V_.l_ld() + j], T(200));
            }
            else
            {
                EXPECT_EQ(V_.l_data()[i * V_.l_ld() + j], T(2));
            }
            
        }
    }

    for(auto i = 0; i < V2_.l_cols(); i++)
    {
        for(auto j = 0; j < V2_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(V2_.l_data()[i * V2_.l_ld() + j], T(20));
            }
            else
            {
                EXPECT_EQ(V2_.l_data()[i * V2_.l_ld() + j], T(2));
            }
            
        }
    }

}

TYPED_TEST(HEMMCPUDistTest, DistThrow) 
{
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    auto H_ = chase::distMatrix::BlockBlockMatrix<T>(10, 8, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(16, 4, mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(12, 4, mpi_grid);
    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(16, 4, mpi_grid);
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(12, 4, mpi_grid);

    T alpha = T(1.0);
    T beta = T(2.0);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H_, W_, V_, W2_, V2_);}, std::runtime_error);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H_, V_, W_, V2_, W2_);}, std::runtime_error);

    auto H3_ = chase::distMatrix::BlockBlockMatrix<T>(10, 10, mpi_grid);
    auto V3_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(10, 4, mpi_grid);
    auto W3_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(10, 4, mpi_grid);
    auto V4_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(10, 4, mpi_grid);
    auto W4_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(10, 4, mpi_grid);

    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H3_, W3_, V3_, W4_, V4_, 100, 2);}, std::invalid_argument);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H3_, W3_, V3_, W4_, V4_, 0, 0);}, std::invalid_argument);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H3_, W3_, V3_, W4_, V4_, 0, 100);}, std::invalid_argument);
    EXPECT_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H3_, W3_, V3_, W4_, V4_, 0, 5);}, std::invalid_argument);
    EXPECT_NO_THROW({chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectorsAndRedistributeAsync(H3_, W3_, V3_, W4_, V4_, 0, 4);});
}

//
TYPED_TEST(HEMMCPUDistTest, HEMMDistCorrectnessBlockCyclic) {
    using T = TypeParam;  // Get the current type
    std::size_t N = 10;
    std::size_t n = 4;
    std::size_t block_size = 2;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    auto H_ = chase::distMatrix::BlockCyclicMatrix<T>(N, N, block_size, block_size, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column>(N, n, block_size, mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row>(N, n, block_size, mpi_grid);

    T alpha = T(2.0);
    T beta = T(3.0);
    std::size_t offset = 1;
    std::size_t subSize = 2;

    for(auto i = 0; i < H_.l_cols(); i++)
    {
        for(auto j = 0; j < H_.l_rows(); j++)
        {
            H_.l_data()[i * H_.l_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            V_.l_data()[i * V_.l_ld() + j] = T(2.0);
        }
    }

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            W_.l_data()[i * W_.l_ld() + j] = T(3.0);
        }
    }

    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H_, V_, &beta, W_, offset, subSize);

    for(auto i = 0; i < W_.l_cols(); i++)
    {
        for(auto j = 0; j < W_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(W_.l_data()[i * W_.l_ld() + j], T(49));
            }
            else
            {
                EXPECT_EQ(W_.l_data()[i * W_.l_ld() + j], T(3));
            }
            
        }
    }

    chase::linalg::internal::cpu_mpi::MatrixMultiplyMultiVectors(&alpha, H_, W_, &beta, V_, offset, subSize);

    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(V_.l_data()[i * V_.l_ld() + j], T(986));
            }
            else
            {
                EXPECT_EQ(V_.l_data()[i * V_.l_ld() + j], T(2));
            }           
        }
    }
}

