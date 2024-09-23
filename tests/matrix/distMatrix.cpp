#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/matrix/distMatrix.hpp"
#include "Impl/mpi/mpiGrid2D.hpp"

template <typename T>
class MatrixCPUDistTest : public ::testing::Test {
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
TYPED_TEST_SUITE(MatrixCPUDistTest, TestTypes);

TYPED_TEST(MatrixCPUDistTest, RedundantInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 5;
    std::size_t N = 4;
    auto redundant_matrix_ = chase::distMatrix::RedundantMatrix<T>(M, N, mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(redundant_matrix_.g_rows(), M);
    EXPECT_EQ(redundant_matrix_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(redundant_matrix_.l_rows(), M);
    EXPECT_EQ(redundant_matrix_.l_cols(), N);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(redundant_matrix_.l_ld(), M);  

    EXPECT_NE(redundant_matrix_.l_data(), nullptr);    
}

TYPED_TEST(MatrixCPUDistTest, BlockBlockInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 6;
    auto blockblockmatrix_ = chase::distMatrix::BlockBlockMatrix<T>(M, N, mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(blockblockmatrix_.g_rows(), M);
    EXPECT_EQ(blockblockmatrix_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(blockblockmatrix_.l_rows(), M / 2);
    EXPECT_EQ(blockblockmatrix_.l_cols(), N / 2);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(blockblockmatrix_.l_ld(), M/2);  

    EXPECT_NE(blockblockmatrix_.l_data(), nullptr);    
}

TYPED_TEST(MatrixCPUDistTest, RedundantExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 5;
    std::size_t N = 4;
    std::size_t ld = 6;
    std::vector<T> buffer(ld * N);
    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < M; j++)
        {
            buffer[i * ld + j] = T(i + j);
        }
    }
    auto redundant_matrix_ = chase::distMatrix::RedundantMatrix<T>(M, N, ld, buffer.data(), mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(redundant_matrix_.g_rows(), M);
    EXPECT_EQ(redundant_matrix_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(redundant_matrix_.l_rows(), M);
    EXPECT_EQ(redundant_matrix_.l_cols(), N);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(redundant_matrix_.l_ld(), ld);  

    EXPECT_NE(redundant_matrix_.l_data(), nullptr);  

    for(auto i = 0; i < redundant_matrix_.l_cols(); i++)
    {
        for(auto j = 0; j < redundant_matrix_.l_rows(); j++)
        {
            EXPECT_EQ(redundant_matrix_.l_data()[i * ld + j], T(i + j));  
        }
    }      
}

TYPED_TEST(MatrixCPUDistTest, BlockBlockExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t l_rows = 3;
    std::size_t l_cols = 4;
    std::size_t l_ld = 4;
    std::vector<T> buffer(l_cols * l_ld);

    for(auto i = 0; i < l_cols; i++)
    {
        for(auto j = 0; j < l_rows; j++)
        {
            buffer[i * l_ld + j] = T(i + j);
        }
    }
    std::size_t g_rows = 6;
    std::size_t g_cols = 8;

    auto blockblockmatrix_ = chase::distMatrix::BlockBlockMatrix<T>(l_rows, l_cols, l_ld, buffer.data(), mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(blockblockmatrix_.g_rows(), g_rows);
    EXPECT_EQ(blockblockmatrix_.g_cols(), g_cols);

    // Check that the local dimensions are also correct
    EXPECT_EQ(blockblockmatrix_.l_rows(), l_rows);
    EXPECT_EQ(blockblockmatrix_.l_cols(), l_cols);

    EXPECT_EQ(blockblockmatrix_.l_ld(), l_ld); 

    EXPECT_NE(blockblockmatrix_.l_data(), nullptr);  

    for(auto i = 0; i < blockblockmatrix_.l_cols(); i++)
    {
        for(auto j = 0; j < blockblockmatrix_.l_rows(); j++)
        {
            EXPECT_EQ(blockblockmatrix_.l_data()[i * blockblockmatrix_.l_ld() + j], T(i + j));  
        }
    }          
}

#ifdef ENABLE_MIXED_PRECISION
TYPED_TEST(MatrixCPUDistTest, RedundantMixedPrecison) {
    using T = TypeParam;  // Get the current type
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 5;
    std::size_t N = 4;
    std::size_t ld = 6;
    std::vector<T> buffer(ld * N);
    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < M; j++)
        {
            buffer[i * ld + j] = T(i + j);
        }
    }
    auto redundant_matrix_ = chase::distMatrix::RedundantMatrix<T>(M, N, ld, buffer.data(), mpi_grid);
    
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value){
        redundant_matrix_.enableSinglePrecision();
        EXPECT_TRUE(redundant_matrix_.isSinglePrecisionEnabled());
        auto* single_precision_matrix = redundant_matrix_.getSinglePrecisionMatrix();
        ASSERT_NE(single_precision_matrix, nullptr);
        for(auto i = 0; i < single_precision_matrix->l_cols(); i++)
        {
            for(auto j = 0; j < single_precision_matrix->l_rows(); j++)
            {
                EXPECT_EQ(single_precision_matrix->l_data()[i * single_precision_matrix->l_ld() + j], SinglePrecisionType(i + j) );
                single_precision_matrix->l_data()[i * single_precision_matrix->l_ld() + j] += SinglePrecisionType(0.5);
            }
        }

        redundant_matrix_.disableSinglePrecision(true);
        for(auto i = 0; i < redundant_matrix_.l_cols(); i++)
        {
            for(auto j = 0; j < redundant_matrix_.l_rows(); j++)
            {
                EXPECT_EQ(redundant_matrix_.l_data()[i * redundant_matrix_.l_ld() + j], T(i + j + 0.5) );
            }
        }

        EXPECT_FALSE(redundant_matrix_.isSinglePrecisionEnabled());
        EXPECT_NE(single_precision_matrix, nullptr);
    }else
    {
        EXPECT_THROW({redundant_matrix_.enableSinglePrecision();}, std::runtime_error);
    }
}

TYPED_TEST(MatrixCPUDistTest, BlockBlockMixedPrecison) {
    using T = TypeParam;  // Get the current type
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t grows = 6;
    std::size_t gcols = 6;
    std::size_t lrows = 3;
    std::size_t lcols = 3;
    std::vector<T> buffer(lcols * lrows);
    auto blockblockmatrix_ = chase::distMatrix::BlockBlockMatrix<T>(lrows, lcols, lrows, buffer.data(), mpi_grid);

    for(auto i = 0; i < blockblockmatrix_.l_cols(); i++)
    {
        for(auto j = 0; j < blockblockmatrix_.l_rows(); j++)
        {
            blockblockmatrix_.l_data()[i * blockblockmatrix_.l_ld() + j] = T(i + j);
        }
    }
    
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value){
        blockblockmatrix_.enableSinglePrecision();
        EXPECT_TRUE(blockblockmatrix_.isSinglePrecisionEnabled());
        auto* single_precision_matrix = blockblockmatrix_.getSinglePrecisionMatrix();
        ASSERT_NE(single_precision_matrix, nullptr);
        for(auto i = 0; i < single_precision_matrix->l_cols(); i++)
        {
            for(auto j = 0; j < single_precision_matrix->l_rows(); j++)
            {
                EXPECT_EQ(single_precision_matrix->l_data()[i * single_precision_matrix->l_ld() + j], SinglePrecisionType(i + j) );
                single_precision_matrix->l_data()[i * single_precision_matrix->l_ld() + j] += SinglePrecisionType(0.5);
            }
        }

        blockblockmatrix_.disableSinglePrecision(true);
        for(auto i = 0; i < blockblockmatrix_.l_cols(); i++)
        {
            for(auto j = 0; j < blockblockmatrix_.l_rows(); j++)
            {
                EXPECT_EQ(blockblockmatrix_.l_data()[i * blockblockmatrix_.l_ld() + j], T(i + j + 0.5) );
            }
        }

        EXPECT_FALSE(blockblockmatrix_.isSinglePrecisionEnabled());
        EXPECT_NE(single_precision_matrix, nullptr);
    }else
    {
        EXPECT_THROW({blockblockmatrix_.enableSinglePrecision();}, std::runtime_error);
    }
}

#endif