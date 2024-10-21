#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/distMatrix/distMatrix.hpp"
#include "grid/mpiGrid2D.hpp"

template <typename T>
class MatrixDistTest : public ::testing::Test {
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
TYPED_TEST_SUITE(MatrixDistTest, TestTypes);

TYPED_TEST(MatrixDistTest, RedundantInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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

TYPED_TEST(MatrixDistTest, RedundantExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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

TYPED_TEST(MatrixDistTest, BlockBlockInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

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

TYPED_TEST(MatrixDistTest, BlockBlockExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

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

TYPED_TEST(MatrixDistTest, BlockBlockRedistributionToRedundantDivisibleSize) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t g_rows = 4;
    std::size_t g_cols = 4;

    auto blockblockmatrix_ = chase::distMatrix::BlockBlockMatrix<T>(g_rows, g_cols, mpi_grid);
    auto redundantmatrix_ = chase::distMatrix::RedundantMatrix<T>(g_rows, g_cols, mpi_grid);

    if(this->world_rank == 0)
    {
        blockblockmatrix_.l_data()[0] = 0;
        blockblockmatrix_.l_data()[1] = 1;
        blockblockmatrix_.l_data()[2] = 4;
        blockblockmatrix_.l_data()[3] = 5;
    }
    else if(this->world_rank == 1)
    {
        blockblockmatrix_.l_data()[0] = 2;
        blockblockmatrix_.l_data()[1] = 3;
        blockblockmatrix_.l_data()[2] = 6;
        blockblockmatrix_.l_data()[3] = 7;                
    }
    else if(this->world_rank == 2)
    {
        blockblockmatrix_.l_data()[0] = 8;
        blockblockmatrix_.l_data()[1] = 9;
        blockblockmatrix_.l_data()[2] = 12;
        blockblockmatrix_.l_data()[3] = 13;                
    }
    else
    {
        blockblockmatrix_.l_data()[0] = 10;
        blockblockmatrix_.l_data()[1] = 11;
        blockblockmatrix_.l_data()[2] = 14;
        blockblockmatrix_.l_data()[3] = 15;                
    }
    
    blockblockmatrix_.redistributeImpl(&redundantmatrix_);

    for(auto i = 0; i < redundantmatrix_.l_rows(); i++)
    {
        for(auto j = 0; j < redundantmatrix_.l_cols(); j++)
        {
            EXPECT_EQ(redundantmatrix_.l_data()[i + j * redundantmatrix_.l_ld()], T(i + j * redundantmatrix_.l_ld()));
        }
    }
    //test also from redundant to blockblock
    auto blockblockmatrix_2 = chase::distMatrix::BlockBlockMatrix<T>(g_rows, g_cols, mpi_grid);
    redundantmatrix_.redistributeImpl(&blockblockmatrix_2);

    for(auto i = 0; i < blockblockmatrix_2.l_rows(); i++)
    {
        for(auto j = 0; j < blockblockmatrix_2.l_cols(); j++)
        {
            EXPECT_EQ(blockblockmatrix_2.l_data()[i + j * blockblockmatrix_2.l_ld()], blockblockmatrix_.l_data()[i + j * blockblockmatrix_.l_ld()]);
        }
    }
}

TYPED_TEST(MatrixDistTest, BlockBlockRedistributionToRedundantIndivisibleSize) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t g_rows = 5;
    std::size_t g_cols = 5;

    auto blockblockmatrix_ = chase::distMatrix::BlockBlockMatrix<T>(g_rows, g_cols, mpi_grid);
    auto redundantmatrix_ = chase::distMatrix::RedundantMatrix<T>(g_rows, g_cols, mpi_grid);

    if(this->world_rank == 0) //3x3
    {
        blockblockmatrix_.l_data()[0] = 0;
        blockblockmatrix_.l_data()[1] = 1;
        blockblockmatrix_.l_data()[2] = 2;
        blockblockmatrix_.l_data()[3] = 5;
        blockblockmatrix_.l_data()[4] = 6;
        blockblockmatrix_.l_data()[5] = 7;
        blockblockmatrix_.l_data()[6] = 10;
        blockblockmatrix_.l_data()[7] = 11;
        blockblockmatrix_.l_data()[8] = 12;        
    }
    else if(this->world_rank == 1)
    {
        blockblockmatrix_.l_data()[0] = 3;
        blockblockmatrix_.l_data()[1] = 4;
        blockblockmatrix_.l_data()[2] = 8;
        blockblockmatrix_.l_data()[3] = 9;
        blockblockmatrix_.l_data()[4] = 13;
        blockblockmatrix_.l_data()[5] = 14;              
    }
    else if(this->world_rank == 2)
    {
        blockblockmatrix_.l_data()[0] = 15;
        blockblockmatrix_.l_data()[1] = 16;
        blockblockmatrix_.l_data()[2] = 17;
        blockblockmatrix_.l_data()[3] = 20;
        blockblockmatrix_.l_data()[4] = 21;
        blockblockmatrix_.l_data()[5] = 22;                 
    }
    else
    {
        blockblockmatrix_.l_data()[0] = 18;
        blockblockmatrix_.l_data()[1] = 19;
        blockblockmatrix_.l_data()[2] = 23;
        blockblockmatrix_.l_data()[3] = 24;                
    }
    
    blockblockmatrix_.redistributeImpl(&redundantmatrix_);

    for(auto i = 0; i < redundantmatrix_.l_rows(); i++)
    {
        for(auto j = 0; j < redundantmatrix_.l_cols(); j++)
        {
            EXPECT_EQ(redundantmatrix_.l_data()[i + j * redundantmatrix_.l_ld()], T(i + j * redundantmatrix_.l_ld()));
        }
    }
    
    //test also from redundant to blockblock
    auto blockblockmatrix_2 = chase::distMatrix::BlockBlockMatrix<T>(g_rows, g_cols, mpi_grid);
    std::size_t startRow, subRows, startCol, subCols;
    startRow = 2; subRows = 2; startCol = 3; subCols = 1;
    redundantmatrix_.redistributeImpl(&blockblockmatrix_2, startRow, subRows, startCol, subCols);

    std::size_t *g_offs = blockblockmatrix_2.g_offs();
    for(auto i = 0; i < blockblockmatrix_2.l_rows(); i++)
    {
        for(auto j = 0; j < blockblockmatrix_2.l_cols(); j++)
        {
            std::size_t x_g_off = g_offs[0] + i;
            std::size_t y_g_off = g_offs[1] + j;
            if(x_g_off >= startRow && x_g_off < startRow + subRows && y_g_off >= startCol && y_g_off < startCol + subCols )
            {
                EXPECT_EQ(blockblockmatrix_2.l_data()[i + j * blockblockmatrix_2.l_ld()], blockblockmatrix_.l_data()[i + j * blockblockmatrix_.l_ld()]);
            }
            else
            {
                EXPECT_EQ(blockblockmatrix_2.l_data()[i + j * blockblockmatrix_2.l_ld()], T(0));
            }            
        }
    }  
}

TYPED_TEST(MatrixDistTest, BlockCyclicInternalMemoryAllocationCPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t M = 11;
    std::size_t N = 11;
    std::size_t mb = 2;
    std::size_t nb = 2;

    auto blockcyclicmatrix_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(M, N, mb, nb, mpi_grid);
    // Check that the global matrix dimensions are correct
    EXPECT_EQ(blockcyclicmatrix_.g_rows(), M);
    EXPECT_EQ(blockcyclicmatrix_.g_cols(), N);

    if(this->world_rank == 0)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6); 
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else if(this->world_rank == 1)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6);   
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 5);         
    }else if(this->world_rank == 2)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);    
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);  
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 5);         
    }

    EXPECT_NE(blockcyclicmatrix_.l_data(), nullptr);    
}

TYPED_TEST(MatrixDistTest, BlockCyclicExternalMemoryAllocationCPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t M = 11;
    std::size_t N = 11;
    std::size_t mb = 2;
    std::size_t nb = 2;
    std::size_t m, n;
    if(this->world_rank == 0)
    {
        m = 6;
        n = 6;
    }else if(this->world_rank == 1)
    {
        m = 5;
        n = 6;
    }   
    else if(this->world_rank == 2)
    {
        m = 6;
        n = 5;
    }else
    {
        m = 5;
        n = 5;
    } 

    std::size_t ld = 6;

    std::vector<T> data(ld * n);

    auto blockcyclicmatrix_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(M, N, m, n, mb, nb, ld, data.data(), mpi_grid);

    EXPECT_EQ(blockcyclicmatrix_.g_rows(), M);
    EXPECT_EQ(blockcyclicmatrix_.g_cols(), N);

    if(this->world_rank == 0)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6); 
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else if(this->world_rank == 1)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6);   
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else if(this->world_rank == 2)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);    
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);  
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }

    EXPECT_EQ(blockcyclicmatrix_.l_data(), data.data());    
}

TYPED_TEST(MatrixDistTest, RedundantRedistributionToBlockCyclic) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t M = 5;
    std::size_t N = 5;
    std::size_t blocksize = 1;

    auto redundantmatrix_ = chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>(M, N, mpi_grid);
    auto blockcyclicmatrix_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(M, N, blocksize, blocksize, mpi_grid);

    for(auto i = 0; i < redundantmatrix_.l_rows() * redundantmatrix_.l_cols(); i++)
    {
        redundantmatrix_.l_data()[i] = T(i);
    }

    redundantmatrix_.redistributeImpl(&blockcyclicmatrix_);

    if(this->world_rank == 0)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_data()[0], T(0));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[1], T(2));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[2], T(4));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[3], T(10));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[4], T(12));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[5], T(14));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[6], T(20));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[7], T(22));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[8], T(24));

    }else if(this->world_rank == 1)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_data()[0], T(1));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[1], T(3));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[2], T(11));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[3], T(13));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[4], T(21));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[5], T(23));        
    }else if(this->world_rank == 2)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_data()[0], T(5));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[1], T(7));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[2], T(9));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[3], T(15));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[4], T(17));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[5], T(19));          
    }else
    {
        EXPECT_EQ(blockcyclicmatrix_.l_data()[0], T(6));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[1], T(8));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[2], T(16));
        EXPECT_EQ(blockcyclicmatrix_.l_data()[3], T(18));        
    }
}


#ifdef HAS_CUDA
TYPED_TEST(MatrixDistTest, RedundantInternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 5;
    std::size_t N = 4;
    auto redundant_matrix_ = chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>(M, N, mpi_grid);
    
    // Check that the global matrix dimensions are correct
    EXPECT_EQ(redundant_matrix_.g_rows(), M);
    EXPECT_EQ(redundant_matrix_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(redundant_matrix_.l_rows(), M);
    EXPECT_EQ(redundant_matrix_.l_cols(), N);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(redundant_matrix_.l_ld(), M);  

    EXPECT_NE(redundant_matrix_.l_data(), nullptr);   
    EXPECT_EQ(redundant_matrix_.cpu_data(), nullptr);    
    EXPECT_EQ(redundant_matrix_.cpu_ld(), 0);  
}

TYPED_TEST(MatrixDistTest, RedundantExternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 5;
    std::size_t N = 4;
    std::size_t ld = 6;
    std::vector<T> buffer(ld * N);

    auto redundant_matrix_ = chase::distMatrix::RedundantMatrix<T, chase::platform::GPU>(M, N, ld, buffer.data(), mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(redundant_matrix_.g_rows(), M);
    EXPECT_EQ(redundant_matrix_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(redundant_matrix_.l_rows(), M);
    EXPECT_EQ(redundant_matrix_.l_cols(), N);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(redundant_matrix_.l_ld(), M);  
    EXPECT_EQ(redundant_matrix_.cpu_ld(), ld);  

    EXPECT_NE(redundant_matrix_.l_data(), nullptr);  
    EXPECT_EQ(redundant_matrix_.cpu_data(), buffer.data());  
}

TYPED_TEST(MatrixDistTest, BlockBlockInternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 6;
    auto blockblockmatrix_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(M, N, mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(blockblockmatrix_.g_rows(), M);
    EXPECT_EQ(blockblockmatrix_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(blockblockmatrix_.l_rows(), M / 2);
    EXPECT_EQ(blockblockmatrix_.l_cols(), N / 2);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(blockblockmatrix_.l_ld(), M/2);  
    EXPECT_EQ(blockblockmatrix_.cpu_ld(), 0);  
    EXPECT_EQ(blockblockmatrix_.cpu_data(), nullptr);    
    EXPECT_NE(blockblockmatrix_.l_data(), nullptr);    
}

TYPED_TEST(MatrixDistTest, BlockBlockExternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t l_rows = 3;
    std::size_t l_cols = 4;
    std::size_t l_ld = 4;
    std::vector<T> buffer(l_cols * l_ld);

    std::size_t g_rows = 6;
    std::size_t g_cols = 8;

    auto blockblockmatrix_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(l_rows, l_cols, l_ld, buffer.data(), mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(blockblockmatrix_.g_rows(), g_rows);
    EXPECT_EQ(blockblockmatrix_.g_cols(), g_cols);

    // Check that the local dimensions are also correct
    EXPECT_EQ(blockblockmatrix_.l_rows(), l_rows);
    EXPECT_EQ(blockblockmatrix_.l_cols(), l_cols);

    EXPECT_EQ(blockblockmatrix_.l_ld(), l_rows); 
    EXPECT_EQ(blockblockmatrix_.cpu_ld(), l_ld); 
    EXPECT_EQ(blockblockmatrix_.cpu_data(), buffer.data());  
    EXPECT_NE(blockblockmatrix_.l_data(), nullptr);          
}

TYPED_TEST(MatrixDistTest, BlockCyclicInternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t M = 11;
    std::size_t N = 11;
    std::size_t mb = 2;
    std::size_t nb = 2;

    auto blockcyclicmatrix_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(M, N, mb, nb, mpi_grid);
    // Check that the global matrix dimensions are correct
    EXPECT_EQ(blockcyclicmatrix_.g_rows(), M);
    EXPECT_EQ(blockcyclicmatrix_.g_cols(), N);

    if(this->world_rank == 0)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6); 
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else if(this->world_rank == 1)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6);   
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 5);         
    }else if(this->world_rank == 2)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);    
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);  
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 5);         
    }

    EXPECT_NE(blockcyclicmatrix_.l_data(), nullptr);    
}

TYPED_TEST(MatrixDistTest, BlockCyclicExternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    std::size_t M = 11;
    std::size_t N = 11;
    std::size_t mb = 2;
    std::size_t nb = 2;
    std::size_t m, n;
    if(this->world_rank == 0)
    {
        m = 6;
        n = 6;
    }else if(this->world_rank == 1)
    {
        m = 5;
        n = 6;
    }   
    else if(this->world_rank == 2)
    {
        m = 6;
        n = 5;
    }else
    {
        m = 5;
        n = 5;
    } 

    std::size_t ld = 6;

    std::vector<T> data(ld * n);

    auto blockcyclicmatrix_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(M, N, m, n, mb, nb, ld, data.data(), mpi_grid);

    EXPECT_EQ(blockcyclicmatrix_.g_rows(), M);
    EXPECT_EQ(blockcyclicmatrix_.g_cols(), N);

    if(this->world_rank == 0)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6); 
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else if(this->world_rank == 1)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 6);   
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 5);         
    }else if(this->world_rank == 2)
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 6);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);    
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 6);         
    }else
    {
        EXPECT_EQ(blockcyclicmatrix_.l_rows(), 5);
        EXPECT_EQ(blockcyclicmatrix_.l_cols(), 5);  
        EXPECT_EQ(blockcyclicmatrix_.l_ld(), 5);         
    }

    EXPECT_NE(blockcyclicmatrix_.l_data(), nullptr);    
    EXPECT_EQ(blockcyclicmatrix_.cpu_data(), data.data());    
}

#endif
