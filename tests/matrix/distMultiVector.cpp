#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/distMatrix/distMultiVector.hpp"
#include "Impl/grid/mpiGrid2D.hpp"

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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


TYPED_TEST(MultiVectorDistTest, RedistributionFromRowToColumnCommunicator) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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

TYPED_TEST(MultiVectorDistTest, RedistributionFromSameCommunicatorExpectFail) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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

#ifdef ENABLE_MIXED_PRECISION
TYPED_TEST(MultiVectorDistTest, Block1DMixedPrecison) {
    using T = TypeParam;  // Get the current type
    using SinglePrecisionType = typename chase::ToSinglePrecisionTrait<T>::Type;

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(MPI_COMM_WORLD);

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

#endif