// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <cstring>
#include "linalg/internal/cuda_aware_mpi/cholqr.hpp"
#include "tests/linalg/internal/mpi/TestConditions.hpp"
#include "tests/linalg/internal/utils.hpp"
#include "grid/mpiGrid2D.hpp"


// Global static resources that persist across all test suites
namespace {
    bool resources_initialized = false;
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> mpi_grid;
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    cudaStream_t stream;
    int world_rank, world_size;
    std::complex<double>* d_V = nullptr;  // Device memory for test vectors
    std::size_t d_V_size = 0;  // Current size of allocated device memory
}

template <typename T>
class CHOLQRCUDAAwareMPIDistTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize resources only once for all test suites
        if (!resources_initialized) {
            // Initialize MPI grid
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            assert(world_size == 4);
            mpi_grid = std::make_shared<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(4, 1, MPI_COMM_WORLD);

            // Initialize CUDA resources
            CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));
            CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
            CHECK_CUBLAS_ERROR(cublasSetStream(cublasH, stream));
            CHECK_CUSOLVER_ERROR(cusolverDnCreate(&cusolverH));
            CHECK_CUSOLVER_ERROR(cusolverDnSetStream(cusolverH, stream));

            resources_initialized = true;
        }
    }

    static void TearDownTestSuite() {
        // Don't free resources here - they should persist across test suites
    }

    void SetUp() override {
        // Ensure device memory is allocated if needed
        std::size_t required_size = (N / world_size) * n;
        if (d_V == nullptr || d_V_size < required_size) {
            if (d_V != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(d_V));
            }
            CHECK_CUDA_ERROR(cudaMalloc((void**)&d_V, sizeof(std::complex<double>) * required_size));
            d_V_size = required_size;
        }
    }

    void TearDown() override {
        // Don't free device memory here - it will be reused
    }

    static constexpr std::size_t N = 100;
    static constexpr std::size_t n = 50;

    // Accessors for the global resources
    static std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>> get_mpi_grid() { return mpi_grid; }
    static cublasHandle_t get_cublas_handle() { return cublasH; }
    static cusolverDnHandle_t get_cusolver_handle() { return cusolverH; }
    static cudaStream_t get_stream() { return stream; }
    static int get_world_rank() { return world_rank; }
    static int get_world_size() { return world_size; }
    static std::complex<double>* get_device_vector() { return d_V; }
};

// Add a global test environment to handle resource cleanup at program exit
class ResourceCleanupEnvironment : public ::testing::Environment {
public:
    ~ResourceCleanupEnvironment() override {
        if (resources_initialized) {
            if (d_V != nullptr) {
                CHECK_CUDA_ERROR(cudaFree(d_V));
                d_V = nullptr;
                d_V_size = 0;
            }
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH));
            CHECK_CUSOLVER_ERROR(cusolverDnDestroy(cusolverH));
            CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
            
            // Reset the mpi_grid shared_ptr before program exit
            // This ensures the MpiGrid2D destructor is called only once
            mpi_grid.reset();
            
            resources_initialized = false;
            std::cout << "Resources freed at program exit" << std::endl;
        }
    }
};

using TestTypes = ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(CHOLQRCUDAAwareMPIDistTest, TestTypes);

TYPED_TEST(CHOLQRCUDAAwareMPIDistTest, cholQR1GPU) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->get_world_size();
    std::size_t xoff = this->get_world_rank() * 25;
    std::vector<T> V(xlen * this->n);
    read_vectors(V.data(), GetQRFileName<T>() + "cond_10.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy((reinterpret_cast<T*>(this->get_device_vector())), V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::cuda_mpi::cholQR1<T>(this->get_cublas_handle(), this->get_cusolver_handle(), xlen, this->n, reinterpret_cast<T*>(this->get_device_vector()), xlen, this->get_mpi_grid()->get_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), reinterpret_cast<T*>(this->get_device_vector()), sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CHOLQRCUDAAwareMPIDistTest, cholQR1BadlyCondGPU) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->get_world_size();
    std::size_t xoff = this->get_world_rank() * 25;
    std::vector<T> V(xlen * this->n);
    read_vectors(V.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy((reinterpret_cast<T*>(this->get_device_vector())), V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::cuda_mpi::cholQR1<T>(this->get_cublas_handle(), this->get_cusolver_handle(), xlen, this->n, reinterpret_cast<T*>(this->get_device_vector()), xlen, this->get_mpi_grid()->get_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), reinterpret_cast<T*>(this->get_device_vector()), sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    EXPECT_GT(orth, machineEpsilon);
    EXPECT_LT(orth, 1.0);
}

TYPED_TEST(CHOLQRCUDAAwareMPIDistTest, cholQR1illCondGPU) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->get_world_size();
    std::size_t xoff = this->get_world_rank() * 25;
    std::vector<T> V(xlen * this->n);
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy((reinterpret_cast<T*>(this->get_device_vector())), V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::cuda_mpi::cholQR1<T>(this->get_cublas_handle(), this->get_cusolver_handle(), xlen, this->n, reinterpret_cast<T*>(this->get_device_vector()), xlen, this->get_mpi_grid()->get_comm());
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CHOLQRCUDAAwareMPIDistTest, cholQR2GPU) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->get_world_size();
    std::size_t xoff = this->get_world_rank() * 25;
    std::vector<T> V(xlen * this->n);
    read_vectors(V.data(), GetQRFileName<T>() + "cond_1e4.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy((reinterpret_cast<T*>(this->get_device_vector())), V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::cuda_mpi::cholQR2<T>(this->get_cublas_handle(), this->get_cusolver_handle(), xlen, this->n, reinterpret_cast<T*>(this->get_device_vector()), xlen, this->get_mpi_grid()->get_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), reinterpret_cast<T*>(this->get_device_vector()), sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

TYPED_TEST(CHOLQRCUDAAwareMPIDistTest, cholQR2IllCondGPU) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->get_world_size();
    std::size_t xoff = this->get_world_rank() * 25;
    std::vector<T> V(xlen * this->n);
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy((reinterpret_cast<T*>(this->get_device_vector())), V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::cuda_mpi::cholQR2<T>(this->get_cublas_handle(), this->get_cusolver_handle(), xlen, this->n, reinterpret_cast<T*>(this->get_device_vector()), xlen, this->get_mpi_grid()->get_comm());
    EXPECT_GT(info, 0);
    EXPECT_LE(info, this->n);
}

TYPED_TEST(CHOLQRCUDAAwareMPIDistTest, scholQRGPU) {
    using T = TypeParam;  // Get the current type
    auto machineEpsilon = MachineEpsilon<T>::value();
    std::size_t xlen = this->N / this->get_world_size();
    std::size_t xoff = this->get_world_rank() * 25;
    std::vector<T> V(xlen * this->n);
    read_vectors(V.data(), GetQRFileName<T>() + "cond_ill.bin", xoff, xlen, this->N, this->n, 0);
    CHECK_CUDA_ERROR(cudaMemcpy((reinterpret_cast<T*>(this->get_device_vector())), V.data(), sizeof(T) * xlen * this->n, cudaMemcpyHostToDevice));
    int info = chase::linalg::internal::cuda_mpi::shiftedcholQR2<T>(this->get_cublas_handle(), this->get_cusolver_handle(), this->N, xlen, this->n, reinterpret_cast<T*>(this->get_device_vector()), xlen, this->get_mpi_grid()->get_comm());
    ASSERT_EQ(info, 0);
    CHECK_CUDA_ERROR(cudaMemcpy(V.data(), reinterpret_cast<T*>(this->get_device_vector()), sizeof(T) * xlen * this->n, cudaMemcpyDeviceToHost));
    auto orth = orthogonality<T>(xlen, this->n, V.data(), xlen, MPI_COMM_WORLD);
    ASSERT_NEAR(orth, machineEpsilon, machineEpsilon * 15);
}

// Add this at the end of the file, before main()
::testing::Environment* const resource_env = ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);