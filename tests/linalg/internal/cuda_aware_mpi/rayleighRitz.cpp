// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <random>
#include <cstring>
#include "linalg/internal/cuda_aware_mpi/rayleighRitz.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"

template <typename T>
class RRGPUDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
        mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(2, 2, MPI_COMM_WORLD);
        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));   
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
    }

    void TearDown() override {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
        if (cusolverH_)
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH_)); 
    }

    int world_rank;
    int world_size;  

    std::size_t N = 50;
    std::size_t n = 8;   
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid;
    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;
    cudaStream_t stream_;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(RRGPUDistTest, TestTypes);

TYPED_TEST(RRGPUDistTest, RRCorrectnessGPU) {
    using T = TypeParam;  // Get the current type
    ASSERT_EQ(this->world_size, 4);  // Ensure we're running with 4 processes
    auto machineEpsilon = MachineEpsilon<T>::value();
    
    std::size_t offset = 1;
    std::size_t subSize = 2;

    T One = T(1.0);
    T Zero = T(0.0);
    std::vector<T> H(this->N * this->N);
    std::vector<T> H2(this->N * this->N);
    std::vector<chase::Base<T>> evals(this->N);
    std::vector<chase::Base<T>> ritzv(this->n);

    for (int i = 0; i < this->N; ++i) {
        H[i * this->N + i] = T(0.1 * i + 0.1);
    }

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;
    std::vector<T> V(this->N * this->N);

    for(auto i = 0; i < this->N * this->N; i++)
    {
        V[i] =  getRandomT<T>([&]() { return d(gen); });
    }

    std::unique_ptr<T[]> tau(new T[this->N]);

    chase::linalg::lapackpp::t_geqrf(LAPACK_COL_MAJOR, this->N, this->N, V.data(), this->N, tau.get());
    chase::linalg::lapackpp::t_gqr(LAPACK_COL_MAJOR, this->N, this->N, this->N, V.data(), this->N, tau.get());

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, this->N, this->N, this->N,
               &One, H.data(), this->N, V.data(), this->N, &Zero,
               H2.data(), this->N);

    chase::linalg::blaspp::t_gemm(CblasColMajor, CblasNoTrans, CblasConjTrans, this->N, this->N, this->N,
               &One, V.data(), this->N, H2.data(), this->N, &Zero,
               H.data(), this->N);

    int *coords = this->mpi_grid.get()->get_coords();

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(this->N, this->N, this->mpi_grid);
    auto V_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->n, this->mpi_grid);
    auto V2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::column, chase::platform::GPU>(this->N, this->n, this->mpi_grid);
    auto W_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->N, this->n, this->mpi_grid);  
    auto W2_ = chase::distMultiVector::DistMultiVector1D<T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(this->N, this->n, this->mpi_grid);
    auto ritzv_ = chase::distMatrix::RedundantMatrix<chase::Base<T>, chase::platform::GPU>(this->n, 1, this->mpi_grid);
    H_.allocate_cpu_data();
    V_.allocate_cpu_data();
    W_.allocate_cpu_data();
    V2_.allocate_cpu_data();
    W2_.allocate_cpu_data();
    ritzv_.allocate_cpu_data();

    //distribute H_ to BlockBlockMatrix
    for(auto i = 0; i < H_.l_cols(); i++)
    {
        for(auto j = 0; j < H_.l_rows(); j++)
        {
            H_.cpu_data()[i * H_.cpu_ld() + j] = H[j + coords[0] * 25 + (i + coords[1] * 25) * this->N];
        }
    }

    H_.H2D();
    chase::linalg::lapackpp::t_heevd(CblasColMajor, 'V', 'U', this->N,
                    H.data(), this->N, evals.data()); //H contains evecs

    //distribute evecs to V & V2
    for(auto i = 0; i < V_.l_cols(); i++)
    {
        for(auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.cpu_ld() + j] = H[j + coords[0] * 25 + i * this->N];
            V2_.cpu_data()[i * V2_.cpu_ld() + j] = H[j + coords[0] * 25 + i * this->N];
        }
    }
    V_.H2D();
    V2_.H2D();

    int* devInfo;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&devInfo, sizeof(int)));    

    chase::linalg::internal::cuda_mpi::rayleighRitz(this->cublasH_, 
                                               this->cusolverH_, 
                                               H_, 
                                               V_, 
                                               V2_, 
                                               W_, 
                                               W2_, 
                                               ritzv_, 
                                               offset, 
                                               subSize,
                                               devInfo);

    for(auto i = offset; i < offset + subSize; i++)
    {
        EXPECT_NEAR(ritzv_.cpu_data()[i], evals[i], 100 * machineEpsilon);
    }
    
}