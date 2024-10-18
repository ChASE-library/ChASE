#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include <random>
#include "linalg/internal/nccl/lanczos.hpp"
#include "Impl/grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

template <typename T>
class LanczosNCCLDistTest : public ::testing::Test {
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
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));   
    }

    void TearDown() override {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));    
    }

    int world_rank;
    int world_size;

    std::size_t M = 10;
    std::size_t numvec = 4;
    std::size_t N = 500;
    std::vector<T> Clement;
    std::vector<chase::Base<T>> ritzv;
    std::vector<chase::Base<T>> ritzV;
    std::vector<chase::Base<T>> Tau;   
    cublasHandle_t cublasH_;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(LanczosNCCLDistTest, TestTypes);


TYPED_TEST(LanczosNCCLDistTest, mlanczosGPU){
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(this->N, this->N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->M, mpi_grid);
    auto H2_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>(this->N, this->N, mpi_grid);

    V_.allocate_cpu_data();
    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();

    auto Clement_ = chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>(this->N, this->N, this->N, this->Clement.data(), mpi_grid);
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::nccl::lanczos(this->cublasH_,
                                          this->M,
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

TYPED_TEST(LanczosNCCLDistTest, lanczosGPU) {
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(this->N, this->N, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, 1, mpi_grid);
    auto H2_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::CPU>(this->N, this->N, mpi_grid);

    V_.allocate_cpu_data();

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();
    
    auto Clement_ = chase::distMatrix::RedundantMatrix<T>(this->N, this->N, this->N, this->Clement.data(), mpi_grid);
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;
    
    chase::linalg::internal::nccl::lanczos(this->cublasH_,
                                          this->M,
                                          H_,
                                          V_,
                                          &upperb);
    
    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );    
}

TYPED_TEST(LanczosNCCLDistTest, mlanczosGPUBlockCyclic){
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();
    std::size_t blocksize = 16;

    auto H_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(this->N, this->N, blocksize, blocksize, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->M, blocksize, mpi_grid);
    auto H2_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(this->N, this->N, blocksize, blocksize, mpi_grid);

    V_.allocate_cpu_data();
    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();

    auto Clement_ = chase::distMatrix::RedundantMatrix<T, chase::platform::CPU>(this->N, this->N, this->N, this->Clement.data(), mpi_grid);
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;

    chase::linalg::internal::nccl::lanczos(this->cublasH_,
                                          this->M,
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

TYPED_TEST(LanczosNCCLDistTest, lanczosGPUBlockCyclic) {
    using T = TypeParam;  // Get the current type

    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);

    int *coords = mpi_grid.get()->get_coords();
    std::size_t blocksize = 16;

    auto H_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(this->N, this->N, blocksize, blocksize, mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, 1, blocksize, mpi_grid);
    auto H2_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::CPU>(this->N, this->N, blocksize, blocksize, mpi_grid);

    V_.allocate_cpu_data();

    std::mt19937 gen(1337.0 + coords[0]);
    std::normal_distribution<> d;

    for (auto j = 0; j < V_.l_rows() * V_.l_cols(); j++)
    {
        auto rnd = getRandomT<T>([&]() { return d(gen); });
        V_.cpu_data()[j] = rnd;
    }

    V_.H2D();
    
    auto Clement_ = chase::distMatrix::RedundantMatrix<T>(this->N, this->N, this->N, this->Clement.data(), mpi_grid);
    Clement_.redistributeImpl(&H2_);
    CHECK_CUDA_ERROR(cudaMemcpy(H_.l_data(), H2_.l_data(), sizeof(T) * H2_.l_rows() * H2_.l_cols(), cudaMemcpyHostToDevice));

    chase::Base<T> upperb;
    
    chase::linalg::internal::nccl::lanczos(this->cublasH_,
                                          this->M,
                                          H_,
                                          V_,
                                          &upperb);
    
    EXPECT_GT(upperb, chase::Base<T>(this->N - 1) ); //the computed upper bound should larger than the max eigenvalues
    EXPECT_LT(upperb, chase::Base<T>(5 * (this->N - 1) ) );    
}