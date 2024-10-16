#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/nccl/cholqr.hpp"
#include "tests/linalg/internal/mpi/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "Impl/grid/mpiGrid2D.hpp"

template <typename T>
class CHOLQRNCCLDistTest : public ::testing::Test {
protected:
    void SetUp() override {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);  

        CHECK_CUBLAS_ERROR(cublasCreate(&cublasH_));   
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        CHECK_CUBLAS_ERROR(cublasSetStream(cublasH_, stream_));
        CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH_));
        CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH_, stream_));
    }

    void TearDown() override {
        if (cublasH_)
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH_));
        if (cusolverH_)
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH_)); 
    }

    int world_rank;
    int world_size;    
    std::size_t N = 100;
    std::size_t n = 50;

    cublasHandle_t cublasH_;
    cusolverDnHandle_t cusolverH_;
    cudaStream_t stream_;
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(CHOLQRNCCLDistTest, TestTypes);

TYPED_TEST(CHOLQRNCCLDistTest, cholQR1GPU) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25 ;
    T *d_V;
    std::vector<T> V(xlen * this->n);
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(T) * xlen * this->n));
    read_vectors(V.data(), GetQRFileName<T>() + "cond_10.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::nccl::cholQR1<T>(this->cublasH_, this->cusolverH_, xlen, this->n, d_V, xlen, mpi_grid.get()->get_nccl_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), d_V, sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
    cudaFree(d_V);
}

TYPED_TEST(CHOLQRNCCLDistTest, cholQR1BadlyCondGPU) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25 ;
    T *d_V;
    std::vector<T> V(xlen * this->n);
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(T) * xlen * this->n));
    read_vectors(V.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::nccl::cholQR1<T>(this->cublasH_, this->cusolverH_, xlen, this->n, d_V, xlen, mpi_grid.get()->get_nccl_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), d_V, sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    EXPECT_GT(orth, machineEpsilon );
    EXPECT_LT(orth, 1.0);
    cudaFree(d_V);
}

TYPED_TEST(CHOLQRNCCLDistTest, cholQR1illCondGPU) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25 ;
    T *d_V;
    std::vector<T> V(xlen * this->n);
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(T) * xlen * this->n));
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::nccl::cholQR1<T>(this->cublasH_, this->cusolverH_, xlen, this->n, d_V, xlen, mpi_grid.get()->get_nccl_comm());
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CHOLQRNCCLDistTest, cholQR2GPU) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25 ;
    T *d_V;
    std::vector<T> V(xlen * this->n);
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(T) * xlen * this->n));
    read_vectors(V.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::nccl::cholQR2<T>(this->cublasH_, this->cusolverH_, xlen, this->n, d_V, xlen, mpi_grid.get()->get_nccl_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), d_V, sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}


TYPED_TEST(CHOLQRNCCLDistTest, cholQR2IllCondGPU) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25 ;
    T *d_V;
    std::vector<T> V(xlen * this->n);
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(T) * xlen * this->n));
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::nccl::cholQR2<T>(this->cublasH_, this->cusolverH_, xlen, this->n, d_V, xlen, mpi_grid.get()->get_nccl_comm());
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CHOLQRNCCLDistTest, scholQRGPU) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25 ;
    T *d_V;
    std::vector<T> V(xlen * this->n);
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(T) * xlen * this->n));
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::nccl::shiftedcholQR2<T>(this->cublasH_, this->cusolverH_, this->N, xlen, this->n, d_V, xlen, mpi_grid.get()->get_nccl_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), d_V, sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
    cudaFree(d_V);
}

/*
TYPED_TEST(CHOLQRNCCLDistTest, modifiedGramSchmidtCholQRGPU) {
    using T = TypeParam;  // Get the current type
    assert(this->world_size == 4);
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->world_size;
    std::size_t xoff = this->world_rank * 25 ;
    T *d_V;
    std::vector<T> V(xlen * this->n);
    std::shared_ptr<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>> mpi_grid 
            = std::make_shared<chase::Impl::mpi::MpiGrid2D<chase::Impl::mpi::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(T) * xlen * this->n));
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(d_V, V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::nccl::modifiedGramSchmidtCholQR<T>(this->cublasH_, this->cusolverH_, xlen, this->n, 0, d_V, xlen, mpi_grid.get()->get_nccl_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), d_V, sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
    cudaFree(d_V);
}
*/