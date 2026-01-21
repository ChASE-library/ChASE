// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/cuda_aware_mpi/cuda_mpi_kernels.hpp"
#include "linalg/internal/cuda_aware_mpi/flipSign.hpp"
#include "tests/linalg/internal/utils.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>

// Global static buffers that persist across all test suites
namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
cublasHandle_t cublasH;
} // namespace

template <typename T>
class PseudoHEMMGPUNCCLDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ASSERT_EQ(world_size, 4); // Ensure we're running with 4 processes

        if (!resources_initialized)
        {
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                2, 2, MPI_COMM_WORLD);
            CHECK_CUBLAS_ERROR(cublasCreate(&cublasH));
            resources_initialized = true;
        }
    }

    void TearDown() override
    {
        // Don't free resources here - they will be reused
    }

    int world_rank;
    int world_size;

    static cublasHandle_t get_cublas_handle() { return cublasH; }
    static std::shared_ptr<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    get_mpi_grid()
    {
        return mpi_grid;
    }
};

// Add a global test environment to handle resource cleanup at program exit
class ResourceCleanupEnvironment : public ::testing::Environment
{
public:
    ~ResourceCleanupEnvironment() override
    {
        if (resources_initialized)
        {
            CHECK_CUBLAS_ERROR(cublasDestroy(cublasH));
            mpi_grid.reset();
            resources_initialized = false;
            std::cout << "Resources freed at program exit" << std::endl;
        }
    }
};

// Register the global test environment
::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment());

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHEMMGPUNCCLDistTest, TestTypes);

TYPED_TEST(PseudoHEMMGPUNCCLDistTest, TinyPseudoHEMMDistCorrectness)
{
    using T = TypeParam; // Get the current type
    std::size_t N = 10;
    std::size_t n = 5;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes

    auto SH_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        N, N, this->get_mpi_grid());
    SH_.allocate_cpu_data();
    SH_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    SH_.H2D();
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        SH_); // We assume the flipping function works

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(N, N, this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    H_.H2D();

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, n, this->get_mpi_grid());
    V_.allocate_cpu_data();
    auto W1_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, this->get_mpi_grid());
    W1_.allocate_cpu_data();
    auto W2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, this->get_mpi_grid());
    W2_.allocate_cpu_data();

    T alpha = T(1.0);
    T beta = T(0.0);

    std::size_t offset = 0;
    std::size_t subSize = n;

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    for (auto i = 0; i < V_.l_cols(); i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.cpu_ld() + j] =
                getRandomT<T>([&]() { return d(gen); });
        }
    }

    V_.H2D();

    // SH x V = W1 => H x V = SW1 - we assume the standard HEMM works.
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, SH_, V_, &beta, W1_, offset,
        subSize);

    W1_.D2H();

    // H x V = W2 => SH x V = SW2
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, H_, V_, &beta, W2_, offset, subSize);

    // We check that SW2 = W1
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        W2_); // We assume the flipping function works

    W2_.D2H();

    for (auto i = 0; i < W1_.l_cols(); i++)
    {
        if (i >= offset && i < offset + subSize)
        {
            for (auto j = 0; j < W1_.l_rows(); j++)
            {
                EXPECT_EQ(W1_.cpu_data()[i * W1_.cpu_ld() + j],
                          W2_.cpu_data()[i * W2_.cpu_ld() + j]);
            }
        }
    }
}

TYPED_TEST(PseudoHEMMGPUNCCLDistTest, TinyPseudoHEMMBlockCyclicDistCorrectness)
{
    using T = TypeParam; // Get the current type
    std::size_t N = 10;
    std::size_t n = 5;
    std::size_t mb = 2;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes

    auto SH_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(
        N, N, mb, mb, this->get_mpi_grid());
    SH_.allocate_cpu_data();
    SH_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    SH_.H2D();
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        SH_); // We assume the flipping function works

    auto H_ = chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
        T, chase::platform::GPU>(N, N, mb, mb, this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_TinyMatrix<T>());
    H_.H2D();

    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, n, mb, this->get_mpi_grid());
    V_.allocate_cpu_data();
    auto W1_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, mb, this->get_mpi_grid());
    W1_.allocate_cpu_data();
    auto W2_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, mb, this->get_mpi_grid());
    W2_.allocate_cpu_data();

    T alpha = T(1.0);
    T beta = T(0.0);

    std::size_t offset = 0;
    std::size_t subSize = n;

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    for (auto i = 0; i < V_.l_cols(); i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.cpu_ld() + j] =
                getRandomT<T>([&]() { return d(gen); });
        }
    }

    V_.H2D();

    // SH x V = W1 => H x V = SW1 - we assume the standard HEMM works.
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, SH_, V_, &beta, W1_, offset,
        subSize);

    W1_.D2H();

    // H x V = W2 => SH x V = SW2
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, H_, V_, &beta, W2_, offset, subSize);

    // We check that SW2 = W1
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        W2_); // We assume the flipping function works

    W2_.D2H();

    for (auto i = 0; i < W1_.l_cols(); i++)
    {
        if (i >= offset && i < offset + subSize)
        {
            for (auto j = 0; j < W1_.l_rows(); j++)
            {
                EXPECT_EQ(W1_.cpu_data()[i * W1_.cpu_ld() + j],
                          W2_.cpu_data()[i * W2_.cpu_ld() + j]);
            }
        }
    }
}

TYPED_TEST(PseudoHEMMGPUNCCLDistTest, PseudoHEMMDistCorrectness)
{
    using T = TypeParam; // Get the current type
    std::size_t N = 200;
    std::size_t n = 20;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes

    auto SH_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        N, N, this->get_mpi_grid());
    SH_.allocate_cpu_data();
    SH_.readFromBinaryFile(GetBSE_Matrix<T>());
    SH_.H2D();
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        SH_); // We assume the flipping function works

    auto H_ = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(N, N, this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_Matrix<T>());
    H_.H2D();

    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, n, this->get_mpi_grid());
    V_.allocate_cpu_data();
    auto W1_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, this->get_mpi_grid());
    W1_.allocate_cpu_data();
    auto W2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, this->get_mpi_grid());
    W2_.allocate_cpu_data();

    T alpha = T(1.0);
    T beta = T(0.0);

    std::size_t offset = 10;
    std::size_t subSize = 10;

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    for (auto i = offset; i < offset + subSize; i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.l_ld() + j] =
                getRandomT<T>([&]() { return d(gen); });
        }
    }

    V_.H2D();

    // SH x V = W1 => H x V = SW1 - we assume the standard HEMM works.
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, SH_, V_, &beta, W1_, offset,
        subSize);

    W1_.D2H();

    // H x V = W2 => SH x V = SW2
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, H_, V_, &beta, W2_, offset, subSize);

    // We check that SW2 = W1
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        W2_); // We assume the flipping function works

    W2_.D2H();

    for (auto i = 0; i < W1_.l_cols(); i++)
    {
        if (i >= offset && i < offset + subSize)
        {
            for (auto j = 0; j < W1_.l_rows(); j++)
            {
                EXPECT_EQ(W1_.cpu_data()[i * W1_.cpu_ld() + j],
                          W2_.cpu_data()[i * W2_.cpu_ld() + j]);
            }
        }
    }
}

TYPED_TEST(PseudoHEMMGPUNCCLDistTest, PseudoHEMMBlockCyclicDistCorrectness)
{
    using T = TypeParam; // Get the current type
    std::size_t N = 200;
    std::size_t n = 20;
    std::size_t mb = 10;
    ASSERT_EQ(this->world_size, 4); // Ensure we're running with 4 processes

    auto SH_ = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(
        N, N, mb, mb, this->get_mpi_grid());
    SH_.allocate_cpu_data();
    SH_.readFromBinaryFile(GetBSE_Matrix<T>());
    SH_.H2D();
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        SH_); // We assume the flipping function works

    auto H_ = chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
        T, chase::platform::GPU>(N, N, mb, mb, this->get_mpi_grid());
    H_.allocate_cpu_data();
    H_.readFromBinaryFile(GetBSE_Matrix<T>());
    H_.H2D();

    auto V_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, n, mb, this->get_mpi_grid());
    V_.allocate_cpu_data();
    auto W1_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, mb, this->get_mpi_grid());
    W1_.allocate_cpu_data();
    auto W2_ = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, mb, this->get_mpi_grid());
    W2_.allocate_cpu_data();

    T alpha = T(1.0);
    T beta = T(0.0);

    std::size_t offset = 10;
    std::size_t subSize = 10;

    std::mt19937 gen(1337.0);
    std::normal_distribution<> d;

    for (auto i = 0; i < V_.l_cols(); i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.l_ld() + j] =
                getRandomT<T>([&]() { return d(gen); });
        }
    }

    V_.H2D();

    // SH x V = W1 => H x V = SW1 - we assume the standard HEMM works.
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, SH_, V_, &beta, W1_, offset,
        subSize);

    W1_.D2H();

    // H x V = W2 => SH x V = SW2
    chase::linalg::internal::cuda_mpi::MatrixMultiplyMultiVectors(
        this->get_cublas_handle(), &alpha, H_, V_, &beta, W2_, offset, subSize);

    // We check that SW2 = W1
    chase::linalg::internal::cuda_mpi::flipLowerHalfMatrixSign(
        W2_); // We assume the flipping function works

    W2_.D2H();

    for (auto i = 0; i < W1_.l_cols(); i++)
    {
        if (i >= offset && i < offset + subSize)
        {
            for (auto j = 0; j < W1_.l_rows(); j++)
            {
                EXPECT_EQ(W1_.cpu_data()[i * W1_.cpu_ld() + j],
                          W2_.cpu_data()[i * W2_.cpu_ld() + j]);
            }
        }
    }
}
