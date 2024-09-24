#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include <random>
#include "linalg/internal/mpi/lanczos.hpp"
#include "Impl/mpi/mpiGrid2D.hpp"
#include "linalg/matrix/distMatrix.hpp"
#include "linalg/matrix/distMultiVector.hpp"

template <typename T>
class LanczosCPUDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);  
        Clement.resize(N * N);

        for (auto i = 0; i < N; ++i)
        {
            Clement[i + N * i] = 0;
            if (i != N - 1)
                Clement[i + 1 + N * i] = std::sqrt(i * (N + 1 - i));
            if (i != N - 1)
                Clement[i + N * (i + 1)] = std::sqrt(i * (N + 1 - i));
        }
        ritzv.resize(M * numvec);
        ritzV.resize(M * M);
        Tau.resize(M * M * numvec);
    }

    void TearDown() override {}

    int world_rank;
    int world_size;

    std::size_t M = 10;
    std::size_t numvec = 4;
    std::size_t N = 500;
    std::vector<T> Clement;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;        
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosCPUDistTest, TestTypes);

TYPED_TEST(LanczosCPUDistTest, mlanczos){
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T>(this->N, this->N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(this->N, this->M, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    std::size_t xoff, xlen, yoff, ylen;
    xlen = this->N / 2;
    ylen = this->N / 2;
    xoff = coords[0] * xlen;
    yoff = coords[1] * ylen;

    auto Clement_ = chase::distMatrix::RedundantMatrix<T>(this->N, this->N, this->N, this->Clement.data(), mpi_grid);
    Clement_.template redistributeImpl<chase::distMatrix::MatrixTypeTrait<decltype(H_)>::value>(&H_);

    chase::Base<T> upperb;

    chase::linalg::internal::mpi::lanczos(this->M,
                                          this->numvec,
                                          H_,
                                          V_,
                                          &upperb,
                                          this->ritzv.data(),
                                          this->Tau.data(),
                                          this->ritzV.data());


    for(auto i = 0; i < this->numvec; i++)
    {
        EXPECT_GT(this->ritzv[i * this->M], 1.0 - chase::Base<T>(this->N));
        EXPECT_LT(this->ritzv[(i + 1) * this->M-1], chase::Base<T>(this->N - 1));
    }

    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );   
}

TYPED_TEST(LanczosCPUDistTest, lanczos) {
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T>(this->N, this->N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column>(this->N, 1, mpi_grid);

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.l_data()[j] = rnd;
    }

    std::size_t xoff, xlen, yoff, ylen;
    xlen = this->N / 2;
    ylen = this->N / 2;
    xoff = coords[0] * xlen;
    yoff = coords[1] * ylen;
    
    auto Clement_ = chase::distMatrix::RedundantMatrix<T>(this->N, this->N, this->N, this->Clement.data(), mpi_grid);
    Clement_.template redistributeImpl<chase::distMatrix::MatrixTypeTrait<decltype(H_)>::value>(&H_);

    chase::Base<T> upperb;
    
    chase::linalg::internal::mpi::lanczos(this->M,
                                          H_,
                                          V_,
                                          &upperb);
    
    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );    
}