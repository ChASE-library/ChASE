// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/distMatrix/distMultiVector.hpp"
#include "grid/mpiGrid2D.hpp"

template <typename T>
class MultiVectorDistTest : public ::testing::Test {
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
TYPED_TEST_SUITE(MultiVectorDistTest, TestTypes);

TYPED_TEST(MultiVectorDistTest, RowCommInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(multivector_.l_rows(), M / 2);
    EXPECT_EQ(multivector_.l_cols(), N);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(multivector_.l_ld(), M / 2);  

    EXPECT_NE(multivector_.l_data(), nullptr);   
}


TYPED_TEST(MultiVectorDistTest, ColumnCommInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);

    // Check that the local dimensions are also correct
    EXPECT_EQ(multivector_.l_rows(), M / 2);
    EXPECT_EQ(multivector_.l_cols(), N);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(multivector_.l_ld(), M / 2);  

    EXPECT_NE(multivector_.l_data(), nullptr);   
}

TYPED_TEST(MultiVectorDistTest, RowCommExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t lrows = 3;
    std::size_t lcols = 2;
    std::size_t lld = 4;
    std::vector<T> buffer(lld * lcols);

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(lrows, lcols, lld, buffer.data(), mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(multivector_.g_rows(), lrows * 2);
    EXPECT_EQ(multivector_.g_cols(), lcols);

    // Check that the local dimensions are also correct
    EXPECT_EQ(multivector_.l_rows(), lrows);
    EXPECT_EQ(multivector_.l_cols(), lcols);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(multivector_.l_ld(), lld);  

    EXPECT_NE(multivector_.l_data(), nullptr);   
    EXPECT_EQ(multivector_.l_data(), buffer.data());   

}

TYPED_TEST(MultiVectorDistTest, ColumnCommExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t lrows = 3;
    std::size_t lcols = 2;
    std::size_t lld = 4;
    std::vector<T> buffer(lld * lcols);

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(lrows, lcols, lld, buffer.data(), mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(multivector_.g_rows(), lrows * 2);
    EXPECT_EQ(multivector_.g_cols(), lcols);

    // Check that the local dimensions are also correct
    EXPECT_EQ(multivector_.l_rows(), lrows);
    EXPECT_EQ(multivector_.l_cols(), lcols);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(multivector_.l_ld(), lld);  

    EXPECT_NE(multivector_.l_data(), nullptr);   
    EXPECT_EQ(multivector_.l_data(), buffer.data());   
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromColumnToRowCommunicator) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }
    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(coords[1] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromColumnToRowCommunicatorNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }

    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                //EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(1.0)); 
                if(j >= 0 && j < 2)
                {
                    EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(0 + i + j));    
                }else if(j >= 2 && j < 4)
                {
                    EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(1 + i + j - 2));    
                }
                else if(j >= 4 && j < 6)
                {
                    EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(2 + i + j - 4));    
                }else
                {
                    EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(3 + i + j - 6));    
                }                
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromRowToColumnCommunicator) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(coords[1] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }
    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(coords[0] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
        }
    }
}


TYPED_TEST(MultiVectorDistTest, RedistributionFromRowToColumnCommunicatorNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }

    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(2 * coords[0] + i + j));             
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromSameCommunicatorExpectFail) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mpi_grid);

    EXPECT_THROW({multivector_.redistributeImpl(&target_);}, std::runtime_error);

    auto multivector_2 = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mpi_grid);
    auto target_2 = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mpi_grid);

    EXPECT_THROW({multivector_2.redistributeImpl(&target_2);}, std::runtime_error);

    EXPECT_THROW({multivector_.redistributeImpl(&target_2, 10, 10);}, std::invalid_argument);
    EXPECT_THROW({multivector_2.redistributeImpl(&target_, 10, 10);}, std::invalid_argument);
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRowCommInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mb, mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);
    if(this->world_rank == 0 || this->world_rank == 1)
    {
        EXPECT_EQ(multivector_.l_rows(), 4);
        EXPECT_EQ(multivector_.l_ld(), 4);
    }else
    {
        EXPECT_EQ(multivector_.l_rows(), 2);
        EXPECT_EQ(multivector_.l_ld(), 2);
    }

    EXPECT_EQ(multivector_.l_cols(), N);
    EXPECT_NE(multivector_.l_data(), nullptr);   
}


TYPED_TEST(MultiVectorDistTest, BlockCyclicColumnCommInternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mb, mpi_grid);

    if(this->world_rank == 0 || this->world_rank == 2)
    {
        EXPECT_EQ(multivector_.l_rows(), 4);
        EXPECT_EQ(multivector_.l_ld(), 4);
    }else
    {
        EXPECT_EQ(multivector_.l_rows(), 2);
        EXPECT_EQ(multivector_.l_ld(), 2);
    }

    EXPECT_EQ(multivector_.l_cols(), N);
    EXPECT_NE(multivector_.l_data(), nullptr);   
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRowCommExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t lld = 4;
    std::vector<T> buffer(lld * N);
    std::size_t m;
    if(this->world_rank == 0 || this->world_rank == 1)
    {
        m = 4;
    }else
    {
        m = 2;
    }
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row>(M, m, N, mb, lld, buffer.data(), mpi_grid);

    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);
    EXPECT_EQ(multivector_.l_rows(), m);
    EXPECT_EQ(multivector_.l_cols(), N);    
    EXPECT_EQ(multivector_.l_ld(), lld);  
    EXPECT_EQ(multivector_.l_data(), buffer.data());   
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicColumnCommExternalMemoryAllocation) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t lld = 4;
    std::vector<T> buffer(lld * N);
    std::size_t m;
    if(this->world_rank == 0 || this->world_rank == 2)
    {
        m = 4;
    }else
    {
        m = 2;
    }

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column>(M, m, N, mb, lld, buffer.data(), mpi_grid);

    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);
    EXPECT_EQ(multivector_.l_rows(), m);
    EXPECT_EQ(multivector_.l_cols(), N);    
    EXPECT_EQ(multivector_.l_ld(), lld);  
    EXPECT_EQ(multivector_.l_data(), buffer.data()); 
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromColumnToRowCommunicator) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }
    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(coords[1] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromRowToColumnCommunicator) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(coords[1] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }
    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(coords[0] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
        }
    }
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromColumnToRowCommunicatorNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }
    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromRowToColumnCommunicatorNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4 ,1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.l_data()[i *  multivector_.l_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.l_data()[i *  target_.l_ld() + j] = T(-1);
        }
    }
    multivector_.redistributeImpl(&target_, offset, subSize);

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.l_data()[i *  target_.l_ld() + j], T(-1));
            }
        }
    }
}

#ifdef ENABLE_MIXED_PRECISION
TYPED_TEST(MultiVectorDistTest, Block1DMixedPrecison) {
    using T = TypeParam;  // Get the current type
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t grows = 6;
    std::size_t gcols = 4;
    auto block1Dmultivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(grows, gcols, mpi_grid);

    for(auto i = 0; i < block1Dmultivector_.l_cols(); i++)
    {
        for(auto j = 0; j < block1Dmultivector_.l_rows(); j++)
        {
            block1Dmultivector_.l_data()[i * block1Dmultivector_.l_ld() + j] = T(i + j);
        }
    }
    
    if constexpr (std::is_same<T, double>::value || std::is_same<T, std::complex<double>>::value){
        block1Dmultivector_.enableSinglePrecision();
        EXPECT_TRUE(block1Dmultivector_.isSinglePrecisionEnabled());
        auto* single_precision_multivec = block1Dmultivector_.getSinglePrecisionMatrix();
        ASSERT_NE(single_precision_multivec, nullptr);
        for(auto i = 0; i < single_precision_multivec->l_cols(); i++)
        {
            for(auto j = 0; j < single_precision_multivec->l_rows(); j++)
            {
                EXPECT_EQ(single_precision_multivec->l_data()[i * single_precision_multivec->l_ld() + j], SinglePrecisionType(i + j) );
                single_precision_multivec->l_data()[i * single_precision_multivec->l_ld() + j] += SinglePrecisionType(0.5);
            }
        }

        block1Dmultivector_.disableSinglePrecision(true);
        for(auto i = 0; i < block1Dmultivector_.l_cols(); i++)
        {
            for(auto j = 0; j < block1Dmultivector_.l_rows(); j++)
            {
                EXPECT_EQ(block1Dmultivector_.l_data()[i * block1Dmultivector_.l_ld() + j], T(i + j + 0.5) );
            }
        }

        EXPECT_FALSE(block1Dmultivector_.isSinglePrecisionEnabled());
        EXPECT_NE(single_precision_multivec, nullptr);
    }else
    {
        EXPECT_THROW({block1Dmultivector_.enableSinglePrecision();}, std::runtime_error);
    }
}

#endif

#ifdef HAS_CUDA

TYPED_TEST(MultiVectorDistTest, RowCommExternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t lrows = 3;
    std::size_t lcols = 2;
    std::size_t lld = 4;
    std::vector<T> buffer(lld * lcols);

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(lrows, lcols, lld, buffer.data(), mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(multivector_.g_rows(), lrows * 2);
    EXPECT_EQ(multivector_.g_cols(), lcols);

    // Check that the local dimensions are also correct
    EXPECT_EQ(multivector_.l_rows(), lrows);
    EXPECT_EQ(multivector_.l_cols(), lcols);

    // Check that the leading dimension matches local row size
    EXPECT_EQ(multivector_.l_ld(), lrows);  
    EXPECT_EQ(multivector_.cpu_ld(), lld);  
    EXPECT_EQ(multivector_.cpu_data(),  buffer.data());  

    EXPECT_NE(multivector_.l_data(), nullptr);   
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromColumnToRowCommunicatorGPUCUDAAwareMPI) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::size_t lrows = 4;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImpl(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[1] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromRowToColumnCommunicatorGPUCUDAAwareMPI) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);
      
    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t lrows = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(coords[1] + i + j);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImpl(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[0] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}


#ifdef HAS_CUDA
TYPED_TEST(MultiVectorDistTest, RedistributionFromColumnToRowCommunicatorGPU) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::size_t lrows = 4;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImpl(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[1] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromColumnToRowCommunicatorGPUNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::size_t lrows = 2;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(M * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(1.0);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < M; j++)
        {
            dest_data[i *  M + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, M, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImpl(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromRowToColumnCommunicatorGPU) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);
      
    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t lrows = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(coords[1] + i + j);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImpl(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[0] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromRowToColumnCommunicatorGPUNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);
      
    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t lrows = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::vector<T> src_data(M * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < M; j++)
        {
            src_data[i *  M + j] = T(1.0);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, M, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImpl(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}


TYPED_TEST(MultiVectorDistTest, BlockCyclicRowCommInternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);

    // Check that the global matrix dimensions are correct
    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);
    if(this->world_rank == 0 || this->world_rank == 1)
    {
        EXPECT_EQ(multivector_.l_rows(), 4);
        EXPECT_EQ(multivector_.l_ld(), 4);
    }else
    {
        EXPECT_EQ(multivector_.l_rows(), 2);
        EXPECT_EQ(multivector_.l_ld(), 2);
    }

    EXPECT_EQ(multivector_.l_cols(), N);
    EXPECT_NE(multivector_.l_data(), nullptr);   
}


TYPED_TEST(MultiVectorDistTest, BlockCyclicColumnCommInternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);

    if(this->world_rank == 0 || this->world_rank == 2)
    {
        EXPECT_EQ(multivector_.l_rows(), 4);
        EXPECT_EQ(multivector_.l_ld(), 4);
    }else
    {
        EXPECT_EQ(multivector_.l_rows(), 2);
        EXPECT_EQ(multivector_.l_ld(), 2);
    }

    EXPECT_EQ(multivector_.l_cols(), N);
    EXPECT_NE(multivector_.l_data(), nullptr);   
}


TYPED_TEST(MultiVectorDistTest, BlockCyclicRowCommExternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t lld = 4;
    std::vector<T> buffer(lld * N);
    std::size_t m;
    if(this->world_rank == 0 || this->world_rank == 1)
    {
        m = 4;
    }else
    {
        m = 2;
    }
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, m, N, mb, lld, buffer.data(), mpi_grid);

    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);
    EXPECT_EQ(multivector_.l_rows(), m);
    EXPECT_EQ(multivector_.l_cols(), N);    
    EXPECT_EQ(multivector_.l_ld(), m);  
    EXPECT_NE(multivector_.l_data(), nullptr);   
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicColumnCommExternalMemoryAllocationGPU) {
    using T = TypeParam;  // Get the current type
    //using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t lld = 4;
    std::vector<T> buffer(lld * N);
    std::size_t m;
    if(this->world_rank == 0 || this->world_rank == 2)
    {
        m = 4;
    }else
    {
        m = 2;
    }

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, m, N, mb, lld, buffer.data(), mpi_grid);

    EXPECT_EQ(multivector_.g_rows(), M);
    EXPECT_EQ(multivector_.g_cols(), N);
    EXPECT_EQ(multivector_.l_rows(), m);
    EXPECT_EQ(multivector_.l_cols(), N);    
    EXPECT_EQ(multivector_.l_ld(), m);  
    EXPECT_NE(multivector_.l_data(), nullptr); 
}



TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromColumnToRowCommunicatorGPU) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();
    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    multivector_.redistributeImpl(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[1] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromRowToColumnCommunicatorGPU) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(coords[1] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    
    multivector_.redistributeImpl(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[0] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}


TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromColumnToRowCommunicatorGPUNonsquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();
    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    multivector_.redistributeImpl(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromRowToColumnCommunicatorGPUNonsquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    
    multivector_.redistributeImpl(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}

#endif

#ifdef HAS_NCCL
TYPED_TEST(MultiVectorDistTest, RedistributionFromColumnToRowCommunicatorGPUNCCL) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::size_t lrows = 4;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImplAsync(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[1] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionFromColumnToRowCommunicatorGPUNCCLNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::size_t lrows = 2;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(M * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(1.0);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < M; j++)
        {
            dest_data[i *  M + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, M, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImplAsync(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionAsyncFromRowToColumnCommunicatorGPUNCCL) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);
      
    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t lrows = 4;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::vector<T> src_data(lrows * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            src_data[i *  lrows + j] = T(coords[1] + i + j);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(lrows, N, lrows, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImplAsync(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[0] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}

TYPED_TEST(MultiVectorDistTest, RedistributionAsyncFromRowToColumnCommunicatorGPUNCCLNonSquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);
      
    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t lrows = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    std::vector<T> src_data(M * N);
    std::vector<T> dest_data(lrows * N);

    int *coords = mpi_grid.get()->get_coords();

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < M; j++)
        {
            src_data[i *  M + j] = T(1.0);
        }
    }

    for(auto i = 0; i < N; i++)
    {
        for(auto j = 0; j < lrows; j++)
        {
            dest_data[i *  lrows + j] = T(-1);
        }
    }

    auto multivector_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, M, src_data.data(), mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(lrows, N, lrows, dest_data.data(), mpi_grid);
    
    multivector_.redistributeImplAsync(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}



TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromColumnToRowCommunicatorGPUNCCL) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();
    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(coords[0] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    multivector_.redistributeImplAsync(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[1] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromRowToColumnCommunicatorGPUNCCL) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 6;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(coords[1] + i + j);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    
    multivector_.redistributeImplAsync(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(coords[0] + i + j));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}


TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromColumnToRowCommunicatorGPUNCCLNonsquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;
    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();
    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    multivector_.redistributeImplAsync(&target_, offset, subSize);
    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
            
        }
    }
}

TYPED_TEST(MultiVectorDistTest, BlockCyclicRedistributionFromRowToColumnCommunicatorGPUNCCLNonsquaredGrid) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(MPI_COMM_WORLD);

    std::size_t M = 8;
    std::size_t N = 4;
    std::size_t mb = 2;
    std::size_t offset = 1;
    std::size_t subSize = 2;

    auto multivector_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(M, N, mb, mpi_grid);
    auto target_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(M, N, mb, mpi_grid);
    int *coords = mpi_grid.get()->get_coords();
    multivector_.allocate_cpu_data();
    target_.allocate_cpu_data();

    for(auto i = 0; i < multivector_.l_cols(); i++)
    {
        for(auto j = 0; j < multivector_.l_rows(); j++)
        {
            multivector_.cpu_data()[i *  multivector_.cpu_ld() + j] = T(1.0);
        }
    }

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            target_.cpu_data()[i *  target_.cpu_ld() + j] = T(-1);
        }
    }

    multivector_.H2D();
    target_.H2D();
    
    multivector_.redistributeImplAsync(&target_, offset, subSize);

    target_.D2H();

    for(auto i = 0; i < target_.l_cols(); i++)
    {
        for(auto j = 0; j < target_.l_rows(); j++)
        {
            if(i >= offset && i < offset + subSize )
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(1.0));
            }
            else
            {
                EXPECT_EQ(target_.cpu_data()[i *  target_.cpu_ld() + j], T(-1));
            }
        }
    }
}

#endif
#endif