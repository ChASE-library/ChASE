// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/error.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/flipSign.hpp"
#include "linalg/internal/gpu/dist_mpi/hip/hip_mpi_kernels.hpp"
#include "tests/linalg/internal/utils.hpp"

#include <complex>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <rocblas/rocblas.h>

namespace
{
bool resources_initialized = false;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
rocblas_handle rocblasH = nullptr;
} // namespace

template <typename T>
class PseudoHEMMHIPAwareMPIDistTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        ASSERT_EQ(world_size, 4);

        if (!resources_initialized)
        {
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                2, 2, MPI_COMM_WORLD);
            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&rocblasH));
            resources_initialized = true;
        }
    }

    int world_rank = 0;
    int world_size = 0;

    static rocblas_handle get_rocblas_handle() { return rocblasH; }
    static std::shared_ptr<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    get_mpi_grid()
    {
        return mpi_grid;
    }
};

class ResourceCleanupEnvironment : public ::testing::Environment
{
public:
    ~ResourceCleanupEnvironment() override
    {
        if (resources_initialized)
        {
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(rocblasH));
            mpi_grid.reset();
            resources_initialized = false;
        }
    }
};

::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment());

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(PseudoHEMMHIPAwareMPIDistTest, TestTypes);

template <typename T, typename MatrixType, typename PseudoHermitianMatrixType,
          typename InputMultiVectorType, typename ResultMultiVectorType>
void runPseudoHEMMCorrectness(std::size_t N, std::size_t n,
                              std::size_t offset, std::size_t subSize,
                              MatrixType& SH,
                              PseudoHermitianMatrixType& H,
                              InputMultiVectorType& V,
                              ResultMultiVectorType& W1,
                              ResultMultiVectorType& W2,
                              rocblas_handle rocblasH)
{
    EXPECT_EQ(n, V.l_cols());

    SH.allocate_cpu_data();
    SH.readFromBinaryFile(N == 10 ? GetBSE_TinyMatrix<T>() : GetBSE_Matrix<T>());
    SH.H2D();
    chase::linalg::internal::hip_mpi::flipLowerHalfMatrixSign(SH);
    CHECK_HIP_ERROR(hipDeviceSynchronize());

    H.allocate_cpu_data();
    H.readFromBinaryFile(N == 10 ? GetBSE_TinyMatrix<T>() : GetBSE_Matrix<T>());
    H.H2D();

    V.allocate_cpu_data();
    W1.allocate_cpu_data();
    W2.allocate_cpu_data();

    std::mt19937 gen(1337);
    std::normal_distribution<> dist;
    for (std::size_t i = 0; i < V.l_cols(); ++i)
    {
        for (std::size_t j = 0; j < V.l_rows(); ++j)
        {
            V.cpu_data()[i * V.cpu_ld() + j] =
                getRandomT<T>([&]() { return dist(gen); });
        }
    }

    V.H2D();

    T alpha = T(1.0);
    T beta = T(0.0);

    // Reference relation from the NVIDIA CUDA-aware MPI test:
    // SH * V == S * (H * V), where S flips the lower pseudo-Hermitian half.
    chase::linalg::internal::hip_mpi::MatrixMultiplyMultiVectors(
        rocblasH, &alpha, SH, V, &beta, W1, offset, subSize);
    W1.D2H();

    chase::linalg::internal::hip_mpi::MatrixMultiplyMultiVectors(
        rocblasH, &alpha, H, V, &beta, W2, offset, subSize);
    chase::linalg::internal::hip_mpi::flipLowerHalfMatrixSign(W2);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    W2.D2H();

    for (std::size_t i = offset; i < offset + subSize; ++i)
    {
        for (std::size_t j = 0; j < W1.l_rows(); ++j)
        {
            EXPECT_EQ(W1.cpu_data()[i * W1.cpu_ld() + j],
                      W2.cpu_data()[i * W2.cpu_ld() + j]);
        }
    }
}

template <typename T>
void runPseudoHEMMBlockBlock(
    std::size_t N, std::size_t n, std::size_t offset, std::size_t subSize,
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        grid,
    rocblas_handle rocblasH)
{
    auto SH =
        chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(N, N, grid);
    auto H = chase::distMatrix::PseudoHermitianBlockBlockMatrix<
        T, chase::platform::GPU>(N, N, grid);
    auto V = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, n, grid);
    auto W1 = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, grid);
    auto W2 = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, grid);

    runPseudoHEMMCorrectness<T>(N, n, offset, subSize, SH, H, V, W1, W2,
                                rocblasH);
}

template <typename T>
void runPseudoHEMMBlockCyclic(
    std::size_t N, std::size_t n, std::size_t mb, std::size_t offset,
    std::size_t subSize,
    std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
        grid,
    rocblas_handle rocblasH)
{
    auto SH = chase::distMatrix::BlockCyclicMatrix<T, chase::platform::GPU>(
        N, N, mb, mb, grid);
    auto H = chase::distMatrix::PseudoHermitianBlockCyclicMatrix<
        T, chase::platform::GPU>(N, N, mb, mb, grid);
    auto V = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(N, n, mb, grid);
    auto W1 = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, mb, grid);
    auto W2 = chase::distMultiVector::DistMultiVectorBlockCyclic1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        N, n, mb, grid);

    runPseudoHEMMCorrectness<T>(N, n, offset, subSize, SH, H, V, W1, W2,
                                rocblasH);
}

TYPED_TEST(PseudoHEMMHIPAwareMPIDistTest, TinyPseudoHEMMDistCorrectness)
{
    using T = TypeParam;
    runPseudoHEMMBlockBlock<T>(10, 5, 0, 5, this->get_mpi_grid(),
                               this->get_rocblas_handle());
}

TYPED_TEST(PseudoHEMMHIPAwareMPIDistTest,
           TinyPseudoHEMMBlockCyclicDistCorrectness)
{
    using T = TypeParam;
    runPseudoHEMMBlockCyclic<T>(10, 5, 2, 0, 5, this->get_mpi_grid(),
                                this->get_rocblas_handle());
}

TYPED_TEST(PseudoHEMMHIPAwareMPIDistTest, PseudoHEMMDistCorrectness)
{
    using T = TypeParam;
    runPseudoHEMMBlockBlock<T>(200, 20, 10, 10, this->get_mpi_grid(),
                               this->get_rocblas_handle());
}

TYPED_TEST(PseudoHEMMHIPAwareMPIDistTest,
           PseudoHEMMBlockCyclicDistCorrectness)
{
    using T = TypeParam;
    runPseudoHEMMBlockCyclic<T>(200, 20, 10, 10, 10, this->get_mpi_grid(),
                                this->get_rocblas_handle());
}
