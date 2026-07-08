// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#include "linalg/internal/gpu/backend/error.hpp"
#include "grid/mpiGrid2D.hpp"
#include "linalg/distMatrix/distMatrix.hpp"
#include "linalg/distMatrix/distMultiVector.hpp"
#include "linalg/internal/gpu/dist_xccl/rccl/hemm.hpp"
#include <cmath>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

namespace
{
bool resources_initialized = false;
std::complex<double>* H_buff = nullptr;
std::complex<double>* V_buff = nullptr;
std::complex<double>* W_buff = nullptr;
std::complex<double>* V2_buff = nullptr;
std::complex<double>* W2_buff = nullptr;
std::shared_ptr<chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    mpi_grid;
rocblas_handle rocblasH = nullptr;
} // namespace

template <typename T>
class HEMMGPURCCLDistTest : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        if (!resources_initialized)
        {
            int world_rank, world_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            ASSERT_EQ(world_size, 4);
            mpi_grid = std::make_shared<
                chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>(
                2, 2, MPI_COMM_WORLD);

            CHECK_ROCBLAS_ERROR(rocblas_create_handle(&rocblasH));

            CHECK_HIP_ERROR(hipMalloc(&H_buff, m * m * sizeof(std::complex<double>)));
            CHECK_HIP_ERROR(hipMalloc(&V_buff, m * n * sizeof(std::complex<double>)));
            CHECK_HIP_ERROR(hipMalloc(&W_buff, m * n * sizeof(std::complex<double>)));
            CHECK_HIP_ERROR(hipMalloc(&V2_buff, m * n * sizeof(std::complex<double>)));
            CHECK_HIP_ERROR(hipMalloc(&W2_buff, m * n * sizeof(std::complex<double>)));

            resources_initialized = true;
        }
    }

    static void TearDownTestSuite() {}

    void SetUp() override {}
    void TearDown() override {}

    static constexpr std::size_t N = 10;
    static constexpr std::size_t n = 4;
    static constexpr std::size_t m = 5;

    static std::complex<double>* get_H_buff() { return H_buff; }
    static std::complex<double>* get_V_buff() { return V_buff; }
    static std::complex<double>* get_W_buff() { return W_buff; }
    static std::complex<double>* get_V2_buff() { return V2_buff; }
    static std::complex<double>* get_W2_buff() { return W2_buff; }
    static std::shared_ptr<
        chase::grid::MpiGrid2D<chase::grid::GridMajor::ColMajor>>
    get_mpi_grid()
    {
        return mpi_grid;
    }
    static rocblas_handle get_rocblas_handle() { return rocblasH; }

    int world_rank;
    int world_size;
};

class ResourceCleanupEnvironment : public ::testing::Environment
{
public:
    ~ResourceCleanupEnvironment() override
    {
        if (resources_initialized)
        {
            CHECK_ROCBLAS_ERROR(rocblas_destroy_handle(rocblasH));
            CHECK_HIP_ERROR(hipFree(H_buff));
            CHECK_HIP_ERROR(hipFree(V_buff));
            CHECK_HIP_ERROR(hipFree(W_buff));
            CHECK_HIP_ERROR(hipFree(V2_buff));
            CHECK_HIP_ERROR(hipFree(W2_buff));

            mpi_grid.reset();

            resources_initialized = false;
            std::cout << "Resources freed at program exit" << std::endl;
        }
    }
};

using TestTypes =
    ::testing::Types<float, double, std::complex<float>, std::complex<double>>;
TYPED_TEST_SUITE(HEMMGPURCCLDistTest, TestTypes);

TYPED_TEST(HEMMGPURCCLDistTest, HEMMDistCorrectnessGPURCCL)
{
    using T = TypeParam;

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_buff()),
        this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->m, this->n, this->m,
                              reinterpret_cast<T*>(this->get_V_buff()),
                              this->get_mpi_grid());
    auto W_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        this->m, this->n, this->m, reinterpret_cast<T*>(this->get_W_buff()),
        this->get_mpi_grid());

    H_.allocate_cpu_data();
    V_.allocate_cpu_data();
    W_.allocate_cpu_data();

    T alpha = T(2.0);
    T beta = T(3.0);
    std::size_t offset = 1;
    std::size_t subSize = 2;

    for (auto i = 0; i < H_.l_cols(); i++)
    {
        for (auto j = 0; j < H_.l_rows(); j++)
        {
            H_.cpu_data()[i * H_.cpu_ld() + j] = T(1.0);
        }
    }

    for (auto i = 0; i < V_.l_cols(); i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.cpu_ld() + j] = T(2.0);
        }
    }

    for (auto i = 0; i < W_.l_cols(); i++)
    {
        for (auto j = 0; j < W_.l_rows(); j++)
        {
            W_.cpu_data()[i * W_.cpu_ld() + j] = T(3.0);
        }
    }

    H_.H2D();
    V_.H2D();
    W_.H2D();

    chase::linalg::internal::hip_rccl::MatrixMultiplyMultiVectors(
        this->get_rocblas_handle(), &alpha, H_, V_, &beta, W_, offset,
        subSize);
    W_.D2H();

    for (auto i = 0; i < W_.l_cols(); i++)
    {
        for (auto j = 0; j < W_.l_rows(); j++)
        {
            if (i >= offset && i < offset + subSize)
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(49));
            }
            else
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(3));
            }
        }
    }

    chase::linalg::internal::hip_rccl::MatrixMultiplyMultiVectors(
        this->get_rocblas_handle(), &alpha, H_, W_, &beta, V_, offset,
        subSize);

    V_.D2H();

    for (auto i = 0; i < V_.l_cols(); i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            if (i >= offset && i < offset + subSize)
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(986));
            }
            else
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(2));
            }
        }
    }
}

TYPED_TEST(HEMMGPURCCLDistTest, HEMMRedistribeAsyncDistCorrectness)
{
    using T = TypeParam;

    auto H_ = chase::distMatrix::BlockBlockMatrix<T, chase::platform::GPU>(
        this->m, this->m, this->m, reinterpret_cast<T*>(this->get_H_buff()),
        this->get_mpi_grid());
    auto V_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->m, this->n, this->m,
                              reinterpret_cast<T*>(this->get_V_buff()),
                              this->get_mpi_grid());
    auto W_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        this->m, this->n, this->m, reinterpret_cast<T*>(this->get_W_buff()),
        this->get_mpi_grid());

    auto V2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::column,
        chase::platform::GPU>(this->m, this->n, this->m,
                              reinterpret_cast<T*>(this->get_V2_buff()),
                              this->get_mpi_grid());
    auto W2_ = chase::distMultiVector::DistMultiVector1D<
        T, chase::distMultiVector::CommunicatorType::row, chase::platform::GPU>(
        this->m, this->n, this->m, reinterpret_cast<T*>(this->get_W2_buff()),
        this->get_mpi_grid());

    H_.allocate_cpu_data();
    V_.allocate_cpu_data();
    W_.allocate_cpu_data();
    V2_.allocate_cpu_data();
    W2_.allocate_cpu_data();

    std::size_t offset = 1;
    std::size_t subSize = 2;

    for (auto i = 0; i < H_.l_cols(); i++)
    {
        for (auto j = 0; j < H_.l_rows(); j++)
        {
            H_.cpu_data()[i * H_.cpu_ld() + j] = T(1.0);
        }
    }

    for (auto i = 0; i < V_.l_cols(); i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            V_.cpu_data()[i * V_.cpu_ld() + j] = T(2.0);
            V2_.cpu_data()[i * V2_.cpu_ld() + j] =
                V_.cpu_data()[i * V_.cpu_ld() + j];
        }
    }

    for (auto i = 0; i < W_.l_cols(); i++)
    {
        for (auto j = 0; j < W_.l_rows(); j++)
        {
            W_.cpu_data()[i * W_.cpu_ld() + j] = T(3.0);
            W2_.cpu_data()[i * W_.cpu_ld() + j] = T(-1.0);
        }
    }

    H_.H2D();
    V_.H2D();
    W_.H2D();
    V2_.H2D();
    W2_.H2D();

    chase::linalg::internal::hip_rccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
        this->get_rocblas_handle(), H_, V_, W_, V2_, W2_, offset, subSize);

    W_.D2H();
    W2_.D2H();

    for (auto i = 0; i < W_.l_cols(); i++)
    {
        for (auto j = 0; j < W_.l_rows(); j++)
        {
            if (i >= offset && i < offset + subSize)
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(20));
            }
            else
            {
                EXPECT_EQ(W_.cpu_data()[i * W_.cpu_ld() + j], T(3));
            }
        }
    }

    for (auto i = 0; i < W2_.l_cols(); i++)
    {
        for (auto j = 0; j < W2_.l_rows(); j++)
        {
            if (i >= offset && i < offset + subSize)
            {
                EXPECT_EQ(W2_.cpu_data()[i * W2_.cpu_ld() + j], T(2.0));
            }
            else
            {
                EXPECT_EQ(W2_.cpu_data()[i * W2_.cpu_ld() + j], T(-1.0));
            }
            W2_.cpu_data()[i * W2_.cpu_ld() + j] =
                W_.cpu_data()[i * W2_.cpu_ld() + j];
        }
    }
    W2_.H2D();
    chase::linalg::internal::hip_rccl::MatrixMultiplyMultiVectorsAndRedistributeAsync(
        this->get_rocblas_handle(), H_, W_, V_, W2_, V2_, offset, subSize);

    V_.D2H();
    V2_.D2H();

    for (auto i = 0; i < V_.l_cols(); i++)
    {
        for (auto j = 0; j < V_.l_rows(); j++)
        {
            if (i >= offset && i < offset + subSize)
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(200));
            }
            else
            {
                EXPECT_EQ(V_.cpu_data()[i * V_.cpu_ld() + j], T(2));
            }
        }
    }

    for (auto i = 0; i < V2_.l_cols(); i++)
    {
        for (auto j = 0; j < V2_.l_rows(); j++)
        {
            if (i >= offset && i < offset + subSize)
            {
                EXPECT_EQ(V2_.cpu_data()[i * V2_.cpu_ld() + j], T(20));
            }
            else
            {
                EXPECT_EQ(V2_.cpu_data()[i * V2_.cpu_ld() + j], T(2));
            }
        }
    }
}

::testing::Environment* const resource_env =
    ::testing::AddGlobalTestEnvironment(new ResourceCleanupEnvironment);
